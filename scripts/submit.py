#!/usr/bin/env python
"""Build, validate, and submit an Orbit Wars Kaggle bundle."""

from __future__ import annotations

import argparse
import inspect
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from owars.submission.bundle import build_submission  # noqa: E402


def _load_dotenv(env: dict[str, str], path: Path = Path(".env")) -> dict[str, str]:
    if not path.exists():
        return env
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key.startswith("KAGGLE_") and key not in env:
            env[key] = value
    return env


def _validate_bundle(bundle: Path, timeout_s: int) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="orbit-submission."))
    try:
        with tarfile.open(bundle, "r:gz") as tar:
            kwargs = {"filter": "data"} if "filter" in inspect.signature(tar.extractall).parameters else {}
            tar.extractall(tmp, **kwargs)
        validation = (
            "from kaggle_environments import make\n"
            "import main\n"
            "env = make('orbit_wars', debug=True)\n"
            "env.run([main.agent, main.agent])\n"
            "final = env.steps[-1]\n"
            "print({i: float(s.reward or 0.0) for i, s in enumerate(final)})\n"
        )
        subprocess.run(
            [sys.executable, "-c", validation],
            cwd=tmp,
            check=True,
            timeout=timeout_s,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _submit(bundle: Path, competition: str, message: str, env: dict[str, str]) -> None:
    try:
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "submit",
                "-c",
                competition,
                "-f",
                str(bundle),
                "-m",
                message,
            ],
            check=True,
            env=env,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "Kaggle CLI not found. Install with `pip install -e '.[kaggle]'` "
            "or run via `uv run --extra kaggle python scripts/submit.py ...`."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--out", type=Path, default=Path("submission.tar.gz"))
    parser.add_argument("--main", type=Path, default=Path("submission/main.py"))
    parser.add_argument("--message", "-m", required=True)
    parser.add_argument("--competition", default="orbit-wars")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validation-timeout", type=int, default=180)
    args = parser.parse_args()

    if args.ckpt is None and args.main == Path("submission/main.py"):
        raise SystemExit("default learned-agent main requires --ckpt")
    if args.ckpt is not None and not args.ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt}")
    if not args.main.exists():
        raise SystemExit(f"main file not found: {args.main}")

    bundle = build_submission(args.ckpt, args.out, main_py=args.main)
    print(f"wrote {bundle}", flush=True)

    if not args.skip_validation:
        _validate_bundle(bundle, args.validation_timeout)
        print("local validation passed", flush=True)

    if args.dry_run:
        print("dry run: not submitting", flush=True)
        return

    env = _load_dotenv(os.environ.copy())
    _submit(bundle, args.competition, args.message, env)


if __name__ == "__main__":
    main()
