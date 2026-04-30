# owars_env

Standalone Rust transition core for Orbit Wars rollout experiments.

Current scope:

- Loaded-state stepping for planets, fleets, launches, production, sun/planet collisions, combat, orbiting planets, active comet movement, comet spawning, and reset/map generation.
- A generated-map benchmark binary: `cargo run --release --bin bench_env -- --scenario generated --num-envs 128 --workload simple`.
- Rust unit and integration tests.

Current single-thread release benchmark on generated 128-env full episodes:

- noop: ~32k env-steps/s
- simple launches: ~21k env-steps/s

Not yet a Python training backend:

- No Kaggle-style observation materialization or policy feature buffer API.
- No PyO3/DLPack integration.

Integrate with `owars.training` only after adding the Python extension boundary and policy batch feature parity tests.
