from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _latest_replay_module():
    path = Path("scripts/latest_replay.py")
    spec = importlib.util.spec_from_file_location("latest_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_latest_replay_prefers_final_checkpoint(tmp_path):
    mod = _latest_replay_module()
    run_dir = tmp_path / "ppo_base"
    run_dir.mkdir()
    snapshot = run_dir / "snapshot_9999.pt"
    final = run_dir / "final.pt"
    snapshot.write_bytes(b"snapshot")
    final.write_bytes(b"final")

    assert mod._latest_checkpoint(tmp_path, "ppo_base") == final


def test_latest_replay_defaults_to_learned_self_play():
    mod = _latest_replay_module()
    args = mod._parser().parse_args([])

    assert args.opponent == "learned"
