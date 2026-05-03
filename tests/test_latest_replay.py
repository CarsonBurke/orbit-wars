from __future__ import annotations

import importlib.util
import os
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


def test_latest_replay_uses_newest_checkpoint(tmp_path):
    mod = _latest_replay_module()
    run_dir = tmp_path / "ppo_base"
    run_dir.mkdir()
    snapshot = run_dir / "snapshot_9999.pt"
    final = run_dir / "final.pt"
    snapshot.write_bytes(b"snapshot")
    final.write_bytes(b"final")

    os.utime(final, (1.0, 1.0))
    os.utime(snapshot, (2.0, 2.0))

    assert mod._latest_checkpoint(tmp_path, "ppo_base") == snapshot


def test_latest_replay_uses_final_when_it_is_newest(tmp_path):
    mod = _latest_replay_module()
    run_dir = tmp_path / "ppo_base"
    run_dir.mkdir()
    snapshot = run_dir / "snapshot_9999.pt"
    final = run_dir / "final.pt"
    snapshot.write_bytes(b"snapshot")
    final.write_bytes(b"final")

    os.utime(snapshot, (1.0, 1.0))
    os.utime(final, (2.0, 2.0))

    assert mod._latest_checkpoint(tmp_path, "ppo_base") == final


def test_latest_replay_defaults_to_learned_self_play():
    mod = _latest_replay_module()
    args = mod._parser().parse_args([])

    assert args.opponent == "learned"
    assert args.device == "cuda"


def test_latest_run_fallback_uses_configured_checkpoint_root(tmp_path):
    mod = _latest_replay_module()
    runs_root = tmp_path / "runs"
    ckpt_root = tmp_path / "custom_checkpoints"
    run_dir = ckpt_root / "ppo_custom"
    runs_root.mkdir()
    run_dir.mkdir(parents=True)
    (run_dir / "final.pt").write_bytes(b"final")

    choice = mod._latest_run(runs_root, ckpt_root)

    assert choice.name == "ppo_custom"
    assert choice.path is None
