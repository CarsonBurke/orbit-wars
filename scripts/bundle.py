#!/usr/bin/env python
"""Entry point: `python scripts/bundle.py --ckpt <path> --out submission.tar.gz`."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from owars.submission.bundle import main  # noqa: E402

if __name__ == "__main__":
    main()
