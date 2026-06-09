# Factored-critic SAC redesign (fixes 1+2+3)

Branch `feat/sac-test`. Replaces the joint-scalar HL-Gauss critic + REINFORCE
discrete gradient with a **factored (dueling) critic + true closed-form discrete
SAC**, plus per-state entropy targets. Reward semantics are UNTOUCHED.

## Action (unchanged)
Per owned source planet `i`: Bernoulli launch `l_i ~ σ(launch_logit_i)`, masked
Categorical target `t_i ~ π_t(·|i)` over legal slots, tanh-squashed Normal
fraction `f_i ∈ (0,1)` (reparam). Angle solved analytically downstream.

Per-planet discrete option set: `{noop} ∪ {(launch,t): t∈legal_i}`, with
`π(noop_i)=1-p_i`, `π(launch,t|i)=p_i·π_t(t|i)`.

## Critic: factored dueling, scalar (NO HL-Gauss)
State-only set-transformer encoder (no action conditioning at input) →
`planet_h [B,P,d]`, `summary_h [B,d]`. Heads output **normalized** values (O(1)):
- `nV(s)`            = value_head(summary_h)                  → [B]
- `nA0_i(s)`         = noop_head(planet_h)                    → [B,P]   (noop advantage)
- `nAL_i[t](s,f_i)`  = K-basis fraction-conditioned target attention → [B,P,P]
  - K attention score maps `S[B,P,P,K]` (query from [planet_h‖summary], key from planet_h)
  - fraction basis `φ(f_i) = [1, f, f², f³]` (K=4); `nAL_i[t] = Σ_k S[i,t,k]·φ_k(f_i)`
  - cubic-in-f ⇒ interior optima exist (a monotone/bilinear AL would push f→0/1, reproducing the fraction collapse).

Normalized total for a **taken** action (buffer): assemble in trainer
`n_taken = nV + Σ_{i owned}[(1-l_i)nA0_i + l_i·nAL_i[t_i](f_i)]`.
**Expected** (closed form): `n_exp = nV + Σ_{i owned}[(1-p_i)nA0_i + p_i·Σ_t π_t·nAL_i[t](f_i)]`.

## Value normalization (replaces HL-Gauss; reward untouched)
PopArt-lite running standardization of the **TD target** `y` (trainer-owned μ,σ,
shared across twins, EMA; first learning batch initializes μ,σ; σ floored):
- raw `Q = μ + σ·n_total`  (denormalize for targets/actor/soft-V).
- critic loss `smooth_l1(n_taken, ((y-μ)/σ).detach())` per twin (O(1) grads).
- No PopArt weight surgery (slow EMA ⇒ acceptable). Network outputs O(1).

## Bellman / q_step (no_grad target)
`a'~π(·|s')` resampled. Per twin target net compute `Q_raw_exp_j(s')`; `minQ' =
min_j Q_raw_exp_j`. `soft_V' = minQ' + α_d·H_disc' + α_c·H_cont'`.
`y = r + (1-done)·γ·soft_V'`. Update μ,σ from `y`. CE→ smooth_l1 on normalized
twins against `(y-μ)/σ`.

## Actor (closed-form, NO REINFORCE)
Resample `a~π(·|s)`. Per twin assemble `Q_raw_exp_j` with **detach routing**:
- `nV, nA0` → `.detach()` (state-only, no actor-grad anyway; detach saves compute)
- `nAL_i[t](f_i)` → **live** (carries pathwise grad to fraction through `f_i`)
- `p_i, π_t` → **live** (discrete policy gradient)
- one **single live expression** gives both gradients exactly once:
  - discrete PG `∂/∂logits Σ_a π(a)A(a) = Σ_a A(a)∂π(a)` — exact, baseline-free
    (Σ∂π=0 cancels constants ⇒ no variance, no `(Q-b)` term)
  - continuous pathwise `∂/∂f Σ_t p·π_t·nAL[t](f)` — weighted by actual prob (correct)
`min` over twins; pathwise grad flows through min-selected twin's AL.
`actor_loss = -(minQ_raw_exp + α_d·H_disc + α_c·H_cont).mean()`.
Critic params get (unused) grads during actor backward → cleared by
`q_optimizer.zero_grad(set_to_none=True)` at next q_step start (cleanrl idiom;
actor_optimizer holds only actor params).

## Entropies
- `H_disc = Σ_i g_i[H_bern(p_i) + p_i·H_cat(π_t(·|i))]` (summed = joint disc entropy; entropy BONUS scales with planet count — correct).
- `H_cont = Σ_i g_i·p_i·(−logp_frac_i)` (p-weighted: fraction only emitted when launching).

## Fix 3: per-state (bounded) entropy targets
- DISCRETE (the one that ran away to α=1.6): tune on **per-planet average**.
  `achieved = H_disc/max(n_owned,1)`; `target = ratio·(h_disc_max/max(n_owned,1))`
  where `h_disc_max = Σ_i g_i·(log(n_legal_i+1)+log2)`.
  `α_disc_loss = (log_α_disc·(achieved−target).detach()).mean()`  (↑α when achieved<target).
  Target no longer scales with planet count ⇒ no runaway. Keep `_LOG_ALPHA_DISC_MAX` clamp.
- CONTINUOUS (was stable at ~0.14): keep existing structure (target ∝ n_owned),
  but use the p-weighted `cont_logp_sum = Σ g_i·p_i·logp_frac_i (= −H_cont)`:
  `α_cont_loss = -(log_α_cont·(cont_logp_sum + target_entropy_per_dim·n_owned).detach()).mean()`.

## Config (sac_base.yaml)
- HL-Gauss knobs (`value_min/max/num_bins/symlog`) now unused by SAC critic
  (kept only for shared RunConfig parity). Add `value_norm_beta` (EMA rate),
  `value_norm_eps`/σ-floor. `disc_target_entropy_ratio` reinterpreted per-planet.

## Files
- `policies/sac_model.py`: drop HLGaussLoss import; remove ACTION_COND_DIM;
  rewrite SACAction (new fields), SACActor.get_action (closed-form quantities,
  no logp_disc), SACSoftQ (state-only encoder + dueling factored heads + API:
  encode/value/noop_adv/launch_adv).
- `training/sac.py`: rewrite _q_step (factored soft-V', value-norm), _actor_alpha_step
  (closed-form, per-state α targets), SACState (μ,σ running stats; save/load),
  drop REINFORCE/baseline.
- `training/config.py`: SACCfg value-norm knobs.
- `configs/sac_base.yaml`: comments + knobs.
