#!/usr/bin/env python
"""Pull the competition starter files into `data/raw/` via the Kaggle API.

Reads `KAGGLE_API_TOKEN` from the environment (or a `.env` file in the
repo root). The Kaggle CLI must be installed and an account joined to the
competition.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("KAGGLE_API_TOKEN")
    env_path = Path(".env")
    if not token and env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("KAGGLE_API_TOKEN="):
                token = line.split("=", 1)[1].strip()
                break
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token

    try:
        subprocess.check_call(
            ["kaggle", "competitions", "download", "-c", "orbit-wars", "-p", str(out)]
        )
    except FileNotFoundError:
        print("Install the Kaggle CLI: `pip install kaggle`", file=sys.stderr)
        sys.exit(1)

    zip_path = out / "orbit-wars.zip"
    if zip_path.exists():
        subprocess.check_call(["unzip", "-o", str(zip_path), "-d", str(out)])
    print(f"competition files in {out}")


if __name__ == "__main__":
    main()
