from __future__ import annotations

import torch

from owars.training.muon import Muon, zeropower_via_newtonschulz5


def _clone_params(params: list[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    return [torch.nn.Parameter(p.detach().clone()) for p in params]


def _set_grads(params: list[torch.nn.Parameter], grads: list[torch.Tensor]) -> None:
    for p, g in zip(params, grads, strict=True):
        p.grad = g.detach().clone()


def test_batched_newton_schulz_matches_per_matrix():
    torch.manual_seed(0)
    g = torch.randn(4, 8, 6)

    got = zeropower_via_newtonschulz5(g, steps=3)
    expected = torch.stack(
        [zeropower_via_newtonschulz5(row, steps=3) for row in g]
    )

    assert torch.allclose(got.float(), expected.float(), atol=1e-5, rtol=1e-5)


def test_fused_muon_matches_scalar_muon_step():
    for nesterov in (False, True):
        torch.manual_seed(0)
        base = [
            torch.nn.Parameter(torch.randn(8, 8)),
            torch.nn.Parameter(torch.randn(8, 8)),
            torch.nn.Parameter(torch.randn(8, 4)),
            torch.nn.Parameter(torch.randn(4, 8)),
            torch.nn.Parameter(torch.randn(4, 8)),
        ]
        scalar_params = _clone_params(base)
        fused_params = _clone_params(base)

        scalar = Muon(
            [
                {"params": scalar_params[:3], "lr": 0.02},
                {"params": scalar_params[3:], "lr": 0.008},
            ],
            lr=0.02,
            momentum=0.9,
            backend_steps=3,
            nesterov=nesterov,
            normuon=True,
            fused=False,
            momentum_warmup_steps=2,
            momentum_warmup_start=0.8,
        )
        fused = Muon(
            [
                {"params": fused_params[:3], "lr": 0.02},
                {"params": fused_params[3:], "lr": 0.008},
            ],
            lr=0.02,
            momentum=0.9,
            backend_steps=3,
            nesterov=nesterov,
            normuon=True,
            fused=True,
            momentum_warmup_steps=2,
            momentum_warmup_start=0.8,
        )

        for step in range(4):
            grads = [torch.randn_like(p) * (step + 1) for p in base]
            _set_grads(scalar_params, grads)
            _set_grads(fused_params, grads)
            scalar.step()
            fused.step()

            for p_scalar, p_fused in zip(scalar_params, fused_params, strict=True):
                assert torch.allclose(p_scalar, p_fused, atol=2e-5, rtol=2e-5)

        for p_scalar, p_fused in zip(scalar_params, fused_params, strict=True):
            b_scalar = scalar.state[p_scalar]["momentum_buffer"]
            b_fused = fused.state[p_fused]["momentum_buffer"]
            assert torch.allclose(b_scalar, b_fused, atol=1e-6, rtol=1e-6)
            # NorMuon's per-neuron second-moment EMA must also stay in lockstep
            # between the scalar and fused (bucketed) paths.
            v_scalar = scalar.state[p_scalar]["second_momentum_buffer"]
            v_fused = fused.state[p_fused]["second_momentum_buffer"]
            assert torch.allclose(v_scalar, v_fused, atol=1e-6, rtol=1e-6)


def test_muon_state_dict_preserves_warmup_step_count():
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(4, 4)) for _ in range(2)]
    opt = Muon(
        params,
        lr=0.02,
        momentum=0.9,
        backend_steps=3,
        fused=True,
        momentum_warmup_steps=10,
    )
    for _ in range(3):
        _set_grads(params, [torch.randn_like(p) for p in params])
        opt.step()

    restored_params = _clone_params(params)
    restored = Muon(
        restored_params,
        lr=0.02,
        momentum=0.9,
        backend_steps=3,
        fused=True,
        momentum_warmup_steps=10,
    )
    restored.load_state_dict(opt.state_dict())

    assert restored._step_count == opt._step_count
