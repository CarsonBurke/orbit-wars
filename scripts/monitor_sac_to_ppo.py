#!/usr/bin/env python
"""Launch PPO-vs-sniper after a SAC TensorBoard run reaches N games."""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def _game_count(run_dir: Path) -> tuple[int, int | None]:
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if "episode/win_rate" not in tags:
        return 0, None
    events = acc.Scalars("episode/win_rate")
    return len(events), (events[-1].step if events else None)


def _tail(path: Path, max_bytes: int = 8192) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as file:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(max(0, size - max_bytes), os.SEEK_SET)
        return file.read().decode(errors="replace")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_cmdline(pid: int) -> str:
    with contextlib.suppress(OSError):
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    return ""


def _pid_cwd(pid: int) -> Path | None:
    with contextlib.suppress(OSError):
        return Path(f"/proc/{pid}/cwd").resolve()
    return None


def _marker_pid(marker: Path) -> int | None:
    with contextlib.suppress(OSError, ValueError):
        for line in marker.read_text().splitlines():
            if line.startswith("pid="):
                return int(line.split("=", 1)[1])
    return None


def _marker_blocks_launch(marker: Path, ppo_config: str) -> bool:
    pid = _marker_pid(marker)
    if pid is None or not _pid_alive(pid):
        stale = marker.with_suffix(marker.suffix + f".stale-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        marker.replace(stale)
        _log(f"stale marker moved to {stale}")
        return False
    cmdline = _pid_cmdline(pid)
    if "scripts/train.py" in cmdline and ppo_config in cmdline:
        _log(f"live PPO marker exists at {marker}; pid={pid}; not starting duplicate PPO")
        return True
    stale = marker.with_suffix(marker.suffix + f".stale-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    marker.replace(stale)
    _log(f"marker pid={pid} does not match PPO command; moved to {stale}")
    return False


def _reserve_marker(marker: Path) -> int | None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        return os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None


def _release_reserved_marker(marker_fd: int | None, marker: Path) -> None:
    if marker_fd is not None:
        with contextlib.suppress(OSError):
            os.close(marker_fd)
    with contextlib.suppress(OSError):
        marker.unlink()


def _find_sac_pids(root: Path, match: str) -> list[int]:
    pids: list[int] = []
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        cmdline = _pid_cmdline(pid)
        if match not in cmdline:
            continue
        cwd = _pid_cwd(pid)
        if cwd == root:
            pids.append(pid)
    return sorted(set(pids))


def _stop_process(pid: int, timeout_seconds: float) -> None:
    if not _pid_alive(pid):
        _log(f"SAC pid={pid} is already stopped")
        return
    _log(f"stopping SAC pid={pid} before PPO launch")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            _log(f"SAC pid={pid} stopped")
            return
        time.sleep(1.0)
    if _pid_alive(pid):
        _log(f"SAC pid={pid} still alive after {timeout_seconds:.0f}s; sending SIGKILL")
        os.kill(pid, signal.SIGKILL)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if not _pid_alive(pid):
                _log(f"SAC pid={pid} killed")
                return
            time.sleep(0.5)
        raise RuntimeError(f"SAC pid={pid} survived SIGKILL")


def _stop_sac_processes(root: Path, args: argparse.Namespace) -> None:
    pids = [args.sac_pid] if args.sac_pid is not None else _find_sac_pids(root, args.sac_match)
    if not pids:
        message = f"no SAC process matched {args.sac_match!r}"
        if args.allow_missing_sac:
            _log(message)
            return
        raise RuntimeError(message)
    for pid in pids:
        _stop_process(pid, args.stop_sac_timeout_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sac-run", required=True, type=Path)
    parser.add_argument("--threshold", type=int, default=600)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--ppo-config", default="configs/ppo_vs_sniper.yaml")
    parser.add_argument("--marker", type=Path, default=Path("logs/sac_to_ppo_started.marker"))
    parser.add_argument("--ppo-log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--sac-pid", type=int, default=None)
    parser.add_argument("--sac-match", default="scripts/train_sac.py --config configs/sac_vs_sniper.yaml")
    parser.add_argument("--allow-missing-sac", action="store_true")
    parser.add_argument("--stop-sac-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--startup-grace-seconds", type=float, default=30.0)
    parser.add_argument("--confirm-seconds", type=float, default=300.0)
    parser.add_argument("--retry-seconds", type=float, default=60.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    python = root / ".venv/bin/python"
    ppo_cmd = [str(python), "scripts/train.py", "--config", args.ppo_config]
    marker = args.marker if args.marker.is_absolute() else root / args.marker
    sac_run = args.sac_run if args.sac_run.is_absolute() else root / args.sac_run
    ppo_log_dir = args.ppo_log_dir if args.ppo_log_dir.is_absolute() else root / args.ppo_log_dir
    ppo_log_dir.mkdir(parents=True, exist_ok=True)

    _log(f"monitoring {sac_run} until {args.threshold} SAC games")
    while True:
        try:
            count, step = _game_count(sac_run)
            _log(f"SAC games={count} last_step={step}")
            if count >= args.threshold:
                if marker.exists() and _marker_blocks_launch(marker, args.ppo_config):
                    return
                marker_fd = _reserve_marker(marker)
                if marker_fd is None:
                    _log(f"marker appeared at {marker}; checking it before retry")
                    continue
                try:
                    _stop_sac_processes(root, args)
                    ppo_log = ppo_log_dir / f"ppo_vs_sniper_after_sac600_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
                    _log(f"threshold reached; launching PPO: {' '.join(ppo_cmd)}")
                    with ppo_log.open("ab", buffering=0) as log_file:
                        proc = subprocess.Popen(
                            ppo_cmd,
                            cwd=root,
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    _log(f"PPO pid={proc.pid} log={ppo_log}")
                    os.write(
                        marker_fd,
                        (
                            f"reserved_at={datetime.now().isoformat(timespec='seconds')}\n"
                            f"sac_run={sac_run}\n"
                            f"sac_games={count}\n"
                            f"sac_last_step={step}\n"
                            f"cmd={' '.join(ppo_cmd)}\n"
                            f"pid={proc.pid}\n"
                            f"log={ppo_log}\n"
                        ).encode(),
                    )
                    os.close(marker_fd)
                    marker_fd = None
                except Exception:
                    _release_reserved_marker(marker_fd, marker)
                    raise
                deadline = time.monotonic() + max(args.startup_grace_seconds, args.confirm_seconds)
                while True:
                    exit_code = proc.poll()
                    if exit_code is not None or time.monotonic() >= deadline:
                        break
                    time.sleep(min(args.poll_seconds, 15.0))
                if exit_code is None:
                    marker.write_text(
                        f"started_at={datetime.now().isoformat(timespec='seconds')}\n"
                        f"sac_run={sac_run}\n"
                        f"sac_games={count}\n"
                        f"sac_last_step={step}\n"
                        f"cmd={' '.join(ppo_cmd)}\n"
                        f"pid={proc.pid}\n"
                        f"log={ppo_log}\n"
                    )
                    _log(f"PPO still running after {args.confirm_seconds:.0f}s; marker written")
                    return
                _log(f"PPO exited before confirmation with code={exit_code}; will retry")
                tail = _tail(ppo_log)
                if tail:
                    _log("PPO log tail follows:\n" + tail)
                with contextlib.suppress(OSError):
                    marker.unlink()
                time.sleep(args.retry_seconds)
        except Exception as exc:
            _log(f"monitor error: {exc!r}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
