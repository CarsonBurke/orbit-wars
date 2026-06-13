from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


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
    assert not args.deterministic
    assert not args.stochastic


def test_latest_replay_deterministic_is_explicit_opt_in():
    mod = _latest_replay_module()
    args = mod._parser().parse_args(["--deterministic"])

    assert args.deterministic


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


def test_latest_run_errors_on_newest_event_run_without_checkpoint(tmp_path):
    mod = _latest_replay_module()
    runs_root = tmp_path / "runs"
    ckpt_root = tmp_path / "checkpoints"
    stale_event_dir = runs_root / "ppo_base" / "older"
    missing_ckpt_event_dir = runs_root / "ppo_vs_sniper" / "newer"
    ckpt_run_dir = ckpt_root / "ppo_base"
    stale_event_dir.mkdir(parents=True)
    missing_ckpt_event_dir.mkdir(parents=True)
    ckpt_run_dir.mkdir(parents=True)
    stale_event = stale_event_dir / "events.out.tfevents.1"
    missing_ckpt_event = missing_ckpt_event_dir / "events.out.tfevents.2"
    stale_event.write_bytes(b"event")
    missing_ckpt_event.write_bytes(b"event")
    (ckpt_run_dir / "final.pt").write_bytes(b"final")

    os.utime(stale_event, (1.0, 1.0))
    os.utime(missing_ckpt_event, (2.0, 2.0))

    with pytest.raises(FileNotFoundError, match="newest run 'ppo_vs_sniper'"):
        mod._latest_run(runs_root, ckpt_root)

    choice = mod._latest_run(runs_root, ckpt_root, allow_stale_fallback=True)

    assert choice.name == "ppo_base"
    assert choice.path == stale_event_dir


def test_latest_run_requires_explicit_stale_fallback_when_events_have_no_checkpoints(tmp_path):
    mod = _latest_replay_module()
    runs_root = tmp_path / "runs"
    ckpt_root = tmp_path / "checkpoints"
    event_dir = runs_root / "ppo_vs_sniper" / "latest"
    ckpt_run_dir = ckpt_root / "ppo_base"
    event_dir.mkdir(parents=True)
    ckpt_run_dir.mkdir(parents=True)
    (event_dir / "events.out.tfevents.1").write_bytes(b"event")
    (ckpt_run_dir / "final.pt").write_bytes(b"final")

    with pytest.raises(FileNotFoundError, match="newest run 'ppo_vs_sniper'"):
        mod._latest_run(runs_root, ckpt_root)

    choice = mod._latest_run(runs_root, ckpt_root, allow_stale_fallback=True)

    assert choice.name == "ppo_base"
    assert choice.path is None
