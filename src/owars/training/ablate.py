"""Ablation runner — same shape as hull-tactical's `ablate.py`.

A matrix YAML looks like:

    base: configs/ppo_base.yaml
    name: arch_v0
    cells:
      - { run.name: depth_2, model.depth: 2 }
      - { run.name: depth_4, model.depth: 4 }

Each cell deep-merges into the base config and runs as its own training
run, with its own tensorboard subdir under `runs/<matrix>/<cell>`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .config import RunConfig, deep_override
from .train import train_one_run


def _flatten_to_nested(flat: dict) -> dict:
    out: dict = {}
    for key, value in flat.items():
        parts = key.split(".")
        cursor = out
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})
        cursor[parts[-1]] = value
    return out


def run_matrix(matrix_path: str | Path) -> list[dict]:
    matrix_path = Path(matrix_path)
    matrix = yaml.safe_load(matrix_path.read_text())
    base_path = Path(matrix["base"])
    base = yaml.safe_load(base_path.read_text()) or {}
    matrix_name = matrix.get("name") or matrix_path.stem

    results = []
    for cell in matrix["cells"]:
        nested = _flatten_to_nested(cell)
        merged = deep_override(base, nested)
        merged.setdefault("run", {})
        cell_name = merged["run"].get("name", "cell")
        merged["run"]["name"] = f"{matrix_name}/{cell_name}"
        cfg = RunConfig.from_dict(merged)
        summary = train_one_run(cfg)
        results.append({
            "cell": cell,
            "summary": {k: v for k, v in summary.items() if k != "updates"},
        })

    out_path = matrix_path.with_suffix(".results.json")
    out_path.write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    args = parser.parse_args()
    for r in run_matrix(args.matrix):
        print(r)


if __name__ == "__main__":
    main()
