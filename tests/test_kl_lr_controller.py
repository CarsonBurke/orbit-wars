from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from owars.training.config import OptimCfg
from owars.training.train import (
    kl_lr_ema_alpha,
    kl_lr_signal_from_log,
    update_kl_lr_controller,
)


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


def test_kl_lr_controller_uses_sqrt_damped_scale_updates():
    cfg = OptimCfg(
        kl_lr_target=0.02,
        kl_lr_ema_half_life=0.01,
        kl_lr_min_scale=0.1,
        kl_lr_max_scale=10.0,
    )

    _, lr_scale = update_kl_lr_controller(
        kl_ema=cfg.kl_lr_target,
        lr_scale=1.0,
        observed_kl=0.08,
        cfg=cfg,
    )
    assert lr_scale == pytest.approx(0.5)

    _, lr_scale = update_kl_lr_controller(
        kl_ema=cfg.kl_lr_target,
        lr_scale=1.0,
        observed_kl=0.005,
        cfg=cfg,
    )
    assert lr_scale == pytest.approx(2.0)


@pytest.mark.parametrize("observed_kl", [math.nan, math.inf, -math.inf])
def test_kl_lr_controller_ignores_non_finite_kl(observed_kl: float):
    cfg = OptimCfg(
        kl_lr_target=0.025,
        kl_lr_ema_half_life=20.0,
        kl_lr_min_scale=0.1,
        kl_lr_max_scale=10.0,
    )

    kl_ema, lr_scale = update_kl_lr_controller(
        kl_ema=0.04,
        lr_scale=2.5,
        observed_kl=observed_kl,
        cfg=cfg,
    )

    assert kl_ema == pytest.approx(0.04)
    assert lr_scale == pytest.approx(2.5)


def test_kl_lr_signal_uses_latest_per_planet_kl_not_joint_kl():
    log = SimpleNamespace(approx_kl=123.0, per_planet_approx_kl=0.0125)

    assert kl_lr_signal_from_log(log) == pytest.approx(0.0125)
