from __future__ import annotations

import pytest

from owars.training.config import OptimCfg
from owars.training.train import kl_lr_ema_alpha, update_kl_lr_controller


def test_kl_lr_ema_alpha_matches_half_life():
    alpha = kl_lr_ema_alpha(20.0)
    value = 1.0
    for _ in range(20):
        value = (1.0 - alpha) * value

    assert value == pytest.approx(0.5)


def test_kl_lr_controller_tracks_target_and_clamps():
    cfg = OptimCfg(
        kl_lr_target=0.025,
        kl_lr_ema_half_life=20.0,
        kl_lr_min_scale=0.1,
        kl_lr_max_scale=10.0,
    )

    kl_ema, lr_scale = update_kl_lr_controller(
        kl_ema=cfg.kl_lr_target,
        lr_scale=1.0,
        observed_kl=cfg.kl_lr_target,
        cfg=cfg,
    )
    assert kl_ema == pytest.approx(cfg.kl_lr_target)
    assert lr_scale == pytest.approx(1.0)

    _, lr_scale = update_kl_lr_controller(
        kl_ema=1e-9,
        lr_scale=10.0,
        observed_kl=0.0,
        cfg=cfg,
    )
    assert lr_scale == pytest.approx(10.0)

    _, lr_scale = update_kl_lr_controller(
        kl_ema=100.0,
        lr_scale=0.1,
        observed_kl=100.0,
        cfg=cfg,
    )
    assert lr_scale == pytest.approx(0.1)
