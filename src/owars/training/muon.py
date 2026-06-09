"""Muon optimizer with optional row normalization ("normuon").

Lifted from parameter-golf's `sota_train_gpt.py:Muon`, expanded for
readability. Designed for 2D matrix weights — the orthogonalization step
operates on the gradient as a matrix and produces an update with unit
spectral norm (modulo a `sqrt(rows / cols)` shape correction). Combine
with AdamW on parameters outside transformer block matrices (input
projections, task readouts, biases, scale parameters, summary tokens) — see
`_split_params` in `train.py`.

Why Muon for cold-start PPO: AdamW's first step takes a full-lr step in
the raw gradient direction (no running second moment to scale against),
which on a freshly-initialized policy can move log_probs by O(1) and
produce update-0 `approx_kl` spikes. Muon orthogonalizes the gradient via
Newton-Schulz before the lr multiply, so the first step has bounded
spectral norm regardless of gradient magnitude — first-step parameter
movement is *predictable*, not gradient-magnitude-dependent.

Row normalization (`row_normalize=True`) divides each row of the gradient
by its norm before orthogonalization. Per-row bounded updates further
stabilize matrix-weight training. parameter-golf uses this on all matrix
params by default (`muon_row_normalize=1`).
"""

from __future__ import annotations

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(
    g: torch.Tensor, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """5th-order Newton-Schulz iteration on `g / ‖g‖_F` toward an orthogonal
    matrix. Coefficients (a, b, c) are tuned for `steps=5` to map singular
    values into [0.5, 1.5] then collapse them toward 1, which is "good
    enough" for an SGD-like update direction without doing a full SVD.

    Operates in bf16 for speed; transposes tall matrices so the iteration
    forms the smaller square product. Supports either a single matrix
    `[rows, cols]` or a batch of same-shaped matrices `[..., rows, cols]`.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    transposed = g.size(-2) > g.size(-1)
    if transposed:
        x = x.transpose(-2, -1)
    for _ in range(steps):
        a_mat = x @ x.transpose(-2, -1)
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    return x.transpose(-2, -1) if transposed else x


def _shape_correction(g: torch.Tensor) -> float:
    return max(1.0, g.size(-2) / g.size(-1)) ** 0.5


class Muon(torch.optim.Optimizer):
    """Matrix-only optimizer with Newton-Schulz orthogonalization.

    Per parameter, on each step:
      1. Update Nesterov momentum buffer.
      2. (Optional) Row-normalize the gradient.
      3. Run 5-step Newton-Schulz to orthogonalize.
      4. Scale by `sqrt(max(1, rows / cols))` (corrects for non-square
         matrices: tall-thin matrices need a larger norm to match the
         spectral magnitude of square matrices).
      5. Apply decoupled weight decay and the orthogonalized update.
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.95,
        backend_steps: int = 5,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        row_normalize: bool = False,
        fused: bool = True,
        momentum_warmup_steps: int = 0,
        momentum_warmup_start: float = 0.85,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            backend_steps=backend_steps,
            nesterov=nesterov,
            weight_decay=weight_decay,
            row_normalize=row_normalize,
            fused=fused,
            momentum_warmup_steps=momentum_warmup_steps,
            momentum_warmup_start=momentum_warmup_start,
        )
        super().__init__(params, defaults)
        # Per-instance step counter for the momentum-warmup schedule.
        # Mirrors parameter-golf `step_fn` (sota_train_gpt.py:402): linear
        # ramp from `momentum_warmup_start` → `momentum` over the first
        # `momentum_warmup_steps` calls to `step()`. The point is to avoid
        # building a long momentum tail over the *first few* gradient
        # samples, which on a fresh policy are atypically noisy and would
        # otherwise persist in the EMA for ~1/(1-momentum) steps.
        self._step_count: int = 0

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D401
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            warmup_steps = group["momentum_warmup_steps"]
            if warmup_steps > 0:
                frac = min(self._step_count / warmup_steps, 1.0)
                warmup_start = group["momentum_warmup_start"]
                momentum = (1.0 - frac) * warmup_start + frac * momentum
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            row_normalize = group["row_normalize"]
            fused = group["fused"]
            wd = group["weight_decay"]
            if fused:
                self._step_fused_group(
                    params=params,
                    lr=lr,
                    momentum=momentum,
                    backend_steps=backend_steps,
                    nesterov=nesterov,
                    row_normalize=row_normalize,
                    weight_decay=wd,
                )
                continue
            for p in params:
                if p.grad is None:
                    continue
                self._step_one(
                    p,
                    lr=lr,
                    momentum=momentum,
                    backend_steps=backend_steps,
                    nesterov=nesterov,
                    row_normalize=row_normalize,
                    weight_decay=wd,
                )
        self._step_count += 1
        return loss

    def _step_one(
        self,
        p: torch.nn.Parameter,
        *,
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool,
        row_normalize: bool,
        weight_decay: float,
    ) -> None:
        g = p.grad
        if g is None:
            return
        state = self.state[p]
        buf = state.setdefault("momentum_buffer", torch.zeros_like(g))
        buf.mul_(momentum).add_(g)
        if nesterov:
            g = g.add(buf, alpha=momentum)
        if row_normalize:
            row_norms = g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7)
            g = g / row_norms.to(g.dtype)
        g = zeropower_via_newtonschulz5(g, steps=backend_steps)
        g = g * _shape_correction(g)
        if weight_decay > 0.0:
            p.data.mul_(1.0 - lr * weight_decay)
        p.data.add_(g.to(p.dtype), alpha=-lr)

    def _step_fused_group(
        self,
        *,
        params: list[torch.nn.Parameter],
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool,
        row_normalize: bool,
        weight_decay: float,
    ) -> None:
        buckets: dict[tuple[torch.device, torch.dtype, torch.Size], list[torch.nn.Parameter]] = {}
        for p in params:
            if p.grad is None:
                continue
            key = (p.device, p.dtype, p.shape)
            buckets.setdefault(key, []).append(p)

        for bucket in buckets.values():
            if len(bucket) == 1:
                self._step_one(
                    bucket[0],
                    lr=lr,
                    momentum=momentum,
                    backend_steps=backend_steps,
                    nesterov=nesterov,
                    row_normalize=row_normalize,
                    weight_decay=weight_decay,
                )
                continue

            grads = [p.grad for p in bucket]
            if any(g is None for g in grads):
                raise RuntimeError("internal Muon bucket included a missing gradient")
            grad_tensors = [g for g in grads if g is not None]
            buffers = [
                self.state[p].setdefault("momentum_buffer", torch.zeros_like(p.grad))
                for p in bucket
            ]

            torch._foreach_mul_(buffers, momentum)
            torch._foreach_add_(buffers, grad_tensors)
            if nesterov:
                update_inputs = torch._foreach_add(
                    grad_tensors,
                    buffers,
                    alpha=momentum,
                )
            else:
                update_inputs = grad_tensors

            g = torch.stack(update_inputs)
            if row_normalize:
                row_norms = g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7)
                g = g / row_norms.to(g.dtype)
            g = zeropower_via_newtonschulz5(g, steps=backend_steps)
            g = g * _shape_correction(g)

            if weight_decay > 0.0:
                torch._foreach_mul_(bucket, 1.0 - lr * weight_decay)
            updates = list(g.to(bucket[0].dtype).unbind(0))
            torch._foreach_add_(bucket, updates, alpha=-lr)

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["_step_count"] = self._step_count
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        self._step_count = int(state_dict.get("_step_count", 0))
        base_state = {k: v for k, v in state_dict.items() if k != "_step_count"}
        super().load_state_dict(base_state)


class MultiOptimizer:
    """Tiny shim that forwards `zero_grad` / `step` / `state_dict` to a list
    of underlying optimizers. PPO's update loop calls `optimizer.step()`
    once per minibatch — we wrap the AdamW + Muon pair behind this so the
    loop doesn't need to know there are two.
    """

    def __init__(
        self,
        optimizers: list[torch.optim.Optimizer],
        *,
        lr_warmup_steps: int = 0,
    ):
        self.optimizers = optimizers
        # Linear LR warmup over the first `lr_warmup_steps` calls to `step()`,
        # ramping every param group's lr from ~0 → its configured value. This
        # is the cold-start guard the nGPT port needs: on a fresh policy AdamW's
        # second-moment estimate is uncalibrated, so the first ~tens of steps
        # would otherwise take near-full `lr`·sign() steps. At `control_lr≈0.02`
        # that lets the trunk-gating scalars (`sqk`/`suv`/`alpha`) swing by ~1
        # across the first PPO update, spiking the policy KL far outside the
        # trust region (`old_log_prob` is frozen per update). Mirrors the Muon
        # momentum warmup; counted in optimizer-step calls, not PPO updates.
        self.lr_warmup_steps = int(lr_warmup_steps)
        self._base_lrs = [group["lr"] for group in self.param_groups]
        self._step_count = 0

    @property
    def param_groups(self) -> list[dict]:
        groups: list[dict] = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        if self.lr_warmup_steps > 0:
            self._step_count += 1
            frac = min(self._step_count / self.lr_warmup_steps, 1.0)
            for group, base_lr in zip(self.param_groups, self._base_lrs):
                group["lr"] = base_lr * frac
        for opt in self.optimizers:
            opt.step()

    def state_dict(self) -> dict:
        state = {f"opt_{i}": opt.state_dict() for i, opt in enumerate(self.optimizers)}
        state["_lr_warmup_step"] = self._step_count
        return state

    def load_state_dict(self, state: dict) -> None:
        self._step_count = int(state.get("_lr_warmup_step", 0))
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(state[f"opt_{i}"])
