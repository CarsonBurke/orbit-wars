"""Build a Kaggle submission tarball.

Layout produced (matches the rules — `main.py` at the root):

    submission_<name>.tar.gz/
      main.py                       # imports owars + loads weights/policy.pt
      weights/policy.pt
      owars/                        # vendored package (no network ingress)
        ...

Why we vendor the package: the runner has no internet access, so we can't
`pip install owars`. Putting the package alongside `main.py` lets `import
owars.*` resolve from the bundle.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import tarfile
from pathlib import Path


def build_submission(
    ckpt_path: str | Path,
    out_path: str | Path,
    *,
    package_root: str | Path = "src/owars",
    main_py: str | Path = "submission/main.py",
    runtime_packages: tuple[str, ...] = (
        "hl_gauss_pytorch",
        "einops",
        "einx",
        "frozendict",
        "mpmath",
        "sympy",
        "torch_einops_utils",
    ),
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    staging = out_path.with_suffix("").with_name(out_path.stem + "_build")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    shutil.copy(main_py, staging / "main.py")
    (staging / "weights").mkdir()
    shutil.copy(ckpt_path, staging / "weights" / "policy.pt")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    shutil.copytree(package_root, staging / "owars", ignore=ignore)
    for package in runtime_packages:
        _copy_runtime_package(package, staging)

    with tarfile.open(out_path, "w:gz") as tar:
        for p in staging.rglob("*"):
            arcname = p.relative_to(staging)
            tar.add(p, arcname=arcname)
    shutil.rmtree(staging)
    return out_path


def _copy_runtime_package(package: str, staging: Path) -> None:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"runtime package {package!r} is not importable")
    source = Path(spec.origin).parent
    destination = staging / package
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    shutil.copytree(source, destination, ignore=ignore)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="submission.tar.gz")
    p.add_argument("--package-root", default="src/owars")
    p.add_argument("--main", default="submission/main.py")
    args = p.parse_args()
    out = build_submission(args.ckpt, args.out, package_root=args.package_root, main_py=args.main)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
