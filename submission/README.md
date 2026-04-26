# Submission

This directory holds the runtime entry point for Kaggle (`main.py`). The bundled `submission.tar.gz` is **built**, not committed — see `scripts/bundle.py`.

## Build

```bash
python scripts/bundle.py --ckpt checkpoints/ppo_base/final.pt --out submission.tar.gz
```

Produces a tarball with `main.py` at the root, `weights/policy.pt`, and a vendored copy of `src/owars/`. The tarball is what you upload to Kaggle:

```bash
kaggle competitions submit orbit-wars -f submission.tar.gz -m "ppo_base v1"
```

## Sanity check

```bash
python submission/main.py
```

Runs one local match (the bundled `main.py` against the `"random"` built-in) and prints the final scores. If this raises, the submission won't pass Kaggle's validation episode either.

## Notes

- **No internet.** The runner has `KAGGLE_IS_COMPETITION_RERUN=1` set and no network egress; the bundle has to be self-contained.
- **One second per turn.** `actTimeout=1` per the official config; budget for ~50 ms inference and leave headroom.
- **Validation episode.** Kaggle plays your submission against a copy of itself first. If that fails, the submission is marked Error — pull logs with `kaggle competitions logs <EPISODE_ID> 0`.
