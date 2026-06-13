#!/usr/bin/env python
"""SAC entry point: `python scripts/train_sac.py --config configs/sac_base.yaml`.

Mirrors `scripts/train.py` (PPO) but dispatches to the SAC trainer. Use
`--override section.key=value` to tweak a single field without editing the
YAML. Multiple overrides allowed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from owars.training.config import RunConfig, deep_override
from owars.training.sac import train


def _parse_override(arg: str) -> tuple[str, str, str]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"--override must be `section.key=value`, got {arg!r}"
        )
    lhs, value = arg.split("=", 1)
    if "." not in lhs:
        raise argparse.ArgumentTypeError(
            f"--override key must be `section.key`, got {lhs!r}"
        )
    section, key = lhs.split(".", 1)
    return section, key, value


def _coerce(value: str) -> bool | int | float | str:
    lo = value.lower()
    if lo in {"true", "false"}:
        return lo == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def main() -> None:
    p = argparse.ArgumentParser(description="Train SAC for Orbit Wars")
    p.add_argument(
        "--config",
        required=True,
        type=Path,
        help="path to YAML config (e.g. configs/sac_base.yaml)",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="single-field override, e.g. --override sac.batch_size=128",
    )
    args = p.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f) or {}
    for raw in args.override:
        section, key, value = _parse_override(raw)
        cfg_dict = deep_override(cfg_dict, {section: {key: _coerce(value)}})

    cfg = RunConfig.from_dict(cfg_dict)
    train(cfg)


if __name__ == "__main__":
    main()
