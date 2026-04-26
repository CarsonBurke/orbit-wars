# Notebooks

Use these for one-off explorations and replay analysis. Anything that produces a reproducible artifact belongs in `src/owars/` or a config — notebooks are scratch.

Suggested:
- `01_env_smoke.ipynb` — instantiate `kaggle_environments.make("orbit_wars")`, run a match between two heuristic agents, render the result.
- `02_replay_inspection.ipynb` — pull a replay JSON via `kaggle competitions replay <EPISODE_ID>`, parse it, plot trajectories per agent, find turning points.
- `03_feature_audit.ipynb` — instantiate a random match, dump the per-step `EncodedObs` to inspect feature ranges (all in `[-3, 3]`-ish?), masking, and owner one-hots.
- `04_policy_attention.ipynb` — load a trained policy, run it on a held-out episode, plot per-planet target attention to confirm the model isn't fixated on a single target.
