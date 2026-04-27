"""Muon optimizer with optional row normalization ("normuon").

Lifted from parameter-golf's `sota_train_gpt.py:Muon`, expanded for
readability. Designed for 2D matrix weights — the orthogonalization step
operates on the gradient as a matrix and produces an update with unit
spectral norm (modulo a `sqrt(rows / cols)` shape correction). Combine
with AdamW on the *non-matrix* parameters (LayerNorm gains, biases, scale
parameters, summary tokens) — see `_split_params` in `train.py`.

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

    Operates in bf16 for speed; transposes if rows < cols so the iteration
    runs on the smaller dimension. Identical math to parameter-golf's
    `zeropower_via_newtonschulz5`.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    x = x / (x.norm() + eps)
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    return x.T if transposed else x


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
            wd = group["weight_decay"]
            for p in params:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                buf = state.setdefault("momentum_buffer", torch.zeros_like(g))
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                if row_normalize:
                    row_norms = g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7)
                    g = g / row_norms.to(g.dtype)
                g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                # Spectral-norm shape correction.
                g = g * (max(1.0, g.size(0) / g.size(1)) ** 0.5)
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(g.to(p.dtype), alpha=-lr)
        self._step_count += 1
        return loss


class MultiOptimizer:
    """Tiny shim that forwards `zero_grad` / `step` / `state_dict` to a list
    of underlying optimizers. PPO's update loop calls `optimizer.step()`
    once per minibatch — we wrap the AdamW + Muon pair behind this so the
    loop doesn't need to know there are two.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers

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
        for opt in self.optimizers:
            opt.step()

    def state_dict(self) -> dict:
        return {f"opt_{i}": opt.state_dict() for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, state: dict) -> None:
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(state[f"opt_{i}"])
