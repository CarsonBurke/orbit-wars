use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use owars_env::{Action, Game, GameConfig, Planet, PlayerAction};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

const BOARD_SIZE: f64 = 100.0;
const CENTER: f64 = 50.0;
const SUN_RADIUS: f64 = 10.0;
const ROTATION_RADIUS_LIMIT: f64 = 50.0;
const MAX_SHIP_SPEED: f64 = 6.0;
const MAX_OMEGA: f64 = 0.05;
const MAX_PLANETS: usize = 64;
const MAX_FLEETS: usize = 384;
const PLANET_FEAT_DIM: usize = 19;
const FLEET_FEAT_DIM: usize = 20;
const LOG_1000: f64 = 6.907_755_278_982_137;
const LEAD_T_HORIZON_STEPS: f64 = 600.0;
const LEAD_MAX_TURNS: i32 = LEAD_T_HORIZON_STEPS as i32;
const LEAD_MAX_SCAN_DISTANCE: f64 = std::f64::consts::SQRT_2 * BOARD_SIZE + 8.0;

#[derive(Clone, Copy)]
struct LeadSolution {
    angle: f64,
    time: f64,
    x: f64,
    y: f64,
}

#[derive(Clone)]
struct TargetMotion {
    id: i32,
    x: f64,
    y: f64,
    radius: f64,
    radius_sq: f64,
    is_orbiting: bool,
    theta0: f64,
    orbit_radius: f64,
    angular_velocity: f64,
    max_turns: usize,
    positions: Option<Vec<(f64, f64)>>,
}

#[derive(Clone)]
struct LegalMaskState {
    planet_limit: usize,
    target_motions: Vec<Option<TargetMotion>>,
    target_cols: Vec<usize>,
    source_cols_by_player: Vec<Vec<usize>>,
    static_cols: Vec<usize>,
    moving_cols: Vec<usize>,
}

#[derive(Clone)]
struct LegalMaskCacheEntry {
    step: i32,
    planets_len: usize,
    state: Arc<LegalMaskState>,
}

#[derive(Clone)]
struct RowActionResult {
    actions: Vec<Action>,
    materialized: Vec<bool>,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct NativeActionList {
    actions: PlayerAction,
}

#[pyclass]
struct RustCoreVecEnv {
    games: Vec<Game>,
    legal_mask_cache: Vec<Option<LegalMaskCacheEntry>>,
    num_envs: usize,
    num_players: usize,
    episode_steps: i32,
    ship_speed: f64,
    random_seed: u32,
    reset_counts: Vec<u32>,
}

#[pymethods]
impl RustCoreVecEnv {
    #[new]
    fn new(
        num_envs: usize,
        num_players: usize,
        episode_steps: i32,
        ship_speed: f64,
        random_seed: u32,
    ) -> Self {
        let mut env = Self {
            games: Vec::new(),
            legal_mask_cache: Vec::new(),
            num_envs,
            num_players,
            episode_steps,
            ship_speed,
            random_seed,
            reset_counts: vec![0; num_envs],
        };
        env.reset_core(false);
        env
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.num_envs
    }

    fn reset(&mut self) {
        self.reset_core(true);
    }

    fn reset_subset(&mut self, indices: Vec<usize>) -> PyResult<()> {
        for env_idx in indices {
            if env_idx >= self.games.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "env index out of range",
                ));
            }
            self.games[env_idx] = Game::new(
                GameConfig::new(self.num_players, self.episode_steps, self.ship_speed),
                self.random_seed
                    + env_idx as u32
                    + self.reset_counts[env_idx] * self.num_envs as u32,
            );
            self.reset_counts[env_idx] += 1;
            self.legal_mask_cache[env_idx] = None;
        }
        Ok(())
    }

    fn step_subset_fast<'py>(
        &mut self,
        py: Python<'py>,
        indices: Vec<usize>,
        actions: &Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if actions.len() != indices.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "actions length must match indices length",
            ));
        }
        let mut actions_by_env: Vec<Option<(usize, Vec<PlayerAction>)>> =
            vec![None; self.games.len()];
        for (pos, &env_idx) in indices.iter().enumerate() {
            if env_idx >= self.games.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "env index out of range",
                ));
            }
            if actions_by_env[env_idx].is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "indices must be unique",
                ));
            }
            let action_obj = actions.get_item(pos)?;
            actions_by_env[env_idx] =
                Some((pos, parse_env_actions(&action_obj, self.num_players)?));
        }
        for &env_idx in &indices {
            if env_idx < self.legal_mask_cache.len() {
                self.legal_mask_cache[env_idx] = None;
            }
        }
        let mut results = py.detach(|| {
            self.games
                .par_iter_mut()
                .enumerate()
                .filter_map(|(env_idx, game)| {
                    let (pos, actions) = actions_by_env[env_idx].as_ref()?;
                    let result = game.step(actions);
                    let final_result = if result.done {
                        Some((result.rewards, game.scores()))
                    } else {
                        None
                    };
                    Some((*pos, env_idx, result.done, final_result))
                })
                .collect::<Vec<_>>()
        });
        results.sort_unstable_by_key(|(pos, _, _, _)| *pos);
        let out = PyDict::new(py);
        for (_, env_idx, done, final_result) in results {
            if let Some((rewards, scores)) = final_result {
                out.set_item(env_idx, (py.None(), done, Some((rewards, scores))))?;
            } else {
                out.set_item(env_idx, (py.None(), done, py.None()))?;
            }
        }
        Ok(out)
    }

    fn observation<'py>(
        &self,
        py: Python<'py>,
        idx: usize,
        player: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        observation_dict(py, &self.games[idx], player)
    }

    fn observations<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<(usize, usize)>,
    ) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for (idx, player) in rows {
            out.append(observation_dict(py, &self.games[idx], player)?)?;
        }
        Ok(out)
    }

    fn reward_potentials<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<(usize, usize)>,
        production_weight: f64,
    ) -> Bound<'py, numpy::PyArray1<f32>> {
        let values = rows
            .into_iter()
            .map(|(idx, player)| {
                self.games[idx].projected_margin_potential(player, production_weight)
            })
            .collect::<Vec<_>>();
        Array1::from_vec(values).into_pyarray(py)
    }

    fn production_margins<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<(usize, usize)>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
        for &(idx, player) in &rows {
            if idx >= self.games.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "env index out of range",
                ));
            }
            if player >= self.num_players {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "player index out of range",
                ));
            }
        }
        let values = rows
            .into_iter()
            .map(|(idx, player)| production_margin(&self.games[idx], player))
            .collect::<Vec<_>>();
        Ok(Array1::from_vec(values).into_pyarray(py))
    }

    fn legal_target_mask<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<(usize, usize)>,
        frac: PyReadonlyArray2<'_, f32>,
        owned: PyReadonlyArray2<'_, bool>,
        pmask: PyReadonlyArray2<'_, bool>,
        ids: PyReadonlyArray2<'_, i64>,
    ) -> PyResult<Bound<'py, numpy::PyArray3<bool>>> {
        let shape = frac.shape();
        let (batch, planets) = (shape[0], shape[1]);
        if rows.len() != batch {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rows length must match batch",
            ));
        }
        let frac_v = frac.as_array().to_owned();
        let owned_v = owned.as_array().to_owned();
        let pmask_v = pmask.as_array().to_owned();
        let ids_v = ids.as_array().to_owned();
        let games = &self.games;
        let flat = py.detach(|| {
            (0..batch)
                .into_par_iter()
                .map(|row| {
                    let game = &games[rows[row].0];
                    let mut out = vec![false; planets * planets];
                    fill_legal_mask_row(
                        game,
                        planets,
                        |col| frac_v[[row, col]] as f64,
                        |col| owned_v[[row, col]],
                        |col| pmask_v[[row, col]],
                        |col| ids_v[[row, col]] as i32,
                        &mut out,
                    );
                    out
                })
                .flatten()
                .collect::<Vec<_>>()
        });
        let out = Array3::from_shape_vec((batch, planets, planets), flat).map_err(|err| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to build legal mask: {err}"))
        })?;
        Ok(out.into_pyarray(py))
    }

    fn legal_target_mask_from_state<'py>(
        &mut self,
        py: Python<'py>,
        rows: Vec<(usize, usize)>,
        frac: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, numpy::PyArray3<bool>>> {
        let shape = frac.shape();
        let (batch, planets) = (shape[0], shape[1]);
        if rows.len() != batch {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rows length must match batch",
            ));
        }
        let frac_v = frac.as_array().to_owned();
        self.legal_target_mask_from_state_arrays(py, rows, frac_v, None, batch, planets)
    }

    fn legal_target_mask_from_state_active<'py>(
        &mut self,
        py: Python<'py>,
        rows: Vec<(usize, usize)>,
        frac: PyReadonlyArray2<'_, f32>,
        active_source: PyReadonlyArray2<'_, bool>,
    ) -> PyResult<Bound<'py, numpy::PyArray3<bool>>> {
        let shape = frac.shape();
        let active_shape = active_source.shape();
        if shape != active_shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "active_source shape must match frac shape",
            ));
        }
        let (batch, planets) = (shape[0], shape[1]);
        if rows.len() != batch {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rows length must match batch",
            ));
        }
        let frac_v = frac.as_array().to_owned();
        let active_v = active_source.as_array().to_owned();
        self.legal_target_mask_from_state_arrays(py, rows, frac_v, Some(active_v), batch, planets)
    }

    fn legal_target_mask_from_state_active_fields<'py>(
        &mut self,
        py: Python<'py>,
        rows: Vec<(usize, usize)>,
        fields: PyReadonlyArray3<'_, f32>,
    ) -> PyResult<Bound<'py, numpy::PyArray3<bool>>> {
        let shape = fields.shape();
        let (batch, planets, field_width) = (shape[0], shape[1], shape[2]);
        if field_width < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fields must have shape [batch, planets, >=2]",
            ));
        }
        if rows.len() != batch {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rows length must match batch",
            ));
        }
        let fields_v = fields.as_array().to_owned();
        self.legal_target_mask_from_state_active_field_array(py, rows, fields_v, batch, planets)
    }

    fn materialize_actions<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<(usize, usize)>,
        launch: PyReadonlyArray2<'_, f32>,
        target_idx: PyReadonlyArray2<'_, i64>,
        frac: PyReadonlyArray2<'_, f32>,
        owned: PyReadonlyArray2<'_, bool>,
        pmask: PyReadonlyArray2<'_, bool>,
        ids: PyReadonlyArray2<'_, i64>,
        native: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let shape = launch.shape();
        let (batch, planets) = (shape[0], shape[1]);
        if rows.len() != batch {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rows length must match batch",
            ));
        }
        let launch_v = launch.as_array().to_owned();
        let target_v = target_idx.as_array().to_owned();
        let frac_v = frac.as_array().to_owned();
        let owned_v = owned.as_array().to_owned();
        let pmask_v = pmask.as_array().to_owned();
        let ids_v = ids.as_array().to_owned();
        let games = &self.games;
        let results = py.detach(|| {
            (0..batch)
                .into_par_iter()
                .map(|row| {
                    materialize_action_row(
                        &games[rows[row].0],
                        planets,
                        |col| launch_v[[row, col]] as f64,
                        |col| target_v[[row, col]] as usize,
                        |col| frac_v[[row, col]] as f64,
                        |col| owned_v[[row, col]],
                        |col| pmask_v[[row, col]],
                        |col| ids_v[[row, col]] as i32,
                    )
                })
                .collect::<Vec<_>>()
        });

        build_materialized_actions_dict(py, batch, planets, native, &results)
    }

    fn materialize_actions_from_state<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<(usize, usize)>,
        launch: PyReadonlyArray2<'_, f32>,
        target_idx: PyReadonlyArray2<'_, i64>,
        frac: PyReadonlyArray2<'_, f32>,
        native: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let shape = launch.shape();
        let (batch, planets) = (shape[0], shape[1]);
        if rows.len() != batch {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rows length must match batch",
            ));
        }
        let launch_v = launch.as_array().to_owned();
        let target_v = target_idx.as_array().to_owned();
        let frac_v = frac.as_array().to_owned();
        let games = &self.games;
        let results = py.detach(|| {
            (0..batch)
                .into_par_iter()
                .map(|row| {
                    let (env_idx, player) = rows[row];
                    materialize_action_row_from_state(
                        &games[env_idx],
                        player,
                        planets,
                        |col| launch_v[[row, col]] as f64,
                        |col| target_v[[row, col]] as usize,
                        |col| frac_v[[row, col]] as f64,
                    )
                })
                .collect::<Vec<_>>()
        });

        build_materialized_actions_dict(py, batch, planets, native, &results)
    }

    fn materialize_masked_actions_from_state<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<(usize, usize)>,
        launch: PyReadonlyArray2<'_, f32>,
        target_idx: PyReadonlyArray2<'_, i64>,
        frac: PyReadonlyArray2<'_, f32>,
        native: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let shape = launch.shape();
        let (batch, planets) = (shape[0], shape[1]);
        if rows.len() != batch {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rows length must match batch",
            ));
        }
        let launch_v = launch.as_array().to_owned();
        let target_v = target_idx.as_array().to_owned();
        let frac_v = frac.as_array().to_owned();
        let games = &self.games;
        let results = py.detach(|| {
            (0..batch)
                .into_par_iter()
                .map(|row| {
                    let (env_idx, player) = rows[row];
                    materialize_masked_action_row_from_state(
                        &games[env_idx],
                        player,
                        planets,
                        |col| launch_v[[row, col]] as f64,
                        |col| target_v[[row, col]] as usize,
                        |col| frac_v[[row, col]] as f64,
                    )
                })
                .collect::<Vec<_>>()
        });

        build_materialized_actions_dict(py, batch, planets, native, &results)
    }

    fn materialize_masked_action_fields_from_state<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<(usize, usize)>,
        fields: PyReadonlyArray3<'_, f32>,
        native: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let shape = fields.shape();
        let (batch, planets, field_width) = (shape[0], shape[1], shape[2]);
        if field_width < 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fields must have shape [batch, planets, >=3]",
            ));
        }
        if rows.len() != batch {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rows length must match batch",
            ));
        }
        let fields_v = fields.as_array().to_owned();
        let games = &self.games;
        let results = py.detach(|| {
            (0..batch)
                .into_par_iter()
                .map(|row| {
                    let (env_idx, player) = rows[row];
                    materialize_masked_action_row_from_state(
                        &games[env_idx],
                        player,
                        planets,
                        |col| fields_v[[row, col, 0]] as f64,
                        |col| fields_v[[row, col, 1]].round().max(0.0) as usize,
                        |col| fields_v[[row, col, 2]] as f64,
                    )
                })
                .collect::<Vec<_>>()
        });

        build_materialized_actions_dict(py, batch, planets, native, &results)
    }

    fn policy_batch<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<(usize, usize)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        policy_batch_dict(py, &self.games, rows, true)
    }

    fn policy_batch_no_context<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<(usize, usize)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        policy_batch_dict(py, &self.games, rows, false)
    }

    fn builtin_actions<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        rows: Vec<(usize, usize)>,
        native: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        if name != "sniper" {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported native builtin opponent: {name}"
            )));
        }
        for &(env_idx, player) in &rows {
            if env_idx >= self.games.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "env index out of range",
                ));
            }
            if player >= self.num_players {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "player index out of range",
                ));
            }
        }
        let games = &self.games;
        let actions = py.detach(|| {
            rows.into_par_iter()
                .map(|(env_idx, player)| sniper_actions(&games[env_idx], player))
                .collect::<Vec<_>>()
        });
        let out = PyList::empty(py);
        for row_actions in actions {
            if native {
                out.append(Py::new(
                    py,
                    NativeActionList {
                        actions: row_actions,
                    },
                )?)?;
            } else {
                let py_actions = PyList::empty(py);
                for action in row_actions {
                    let item = PyList::empty(py);
                    item.append(action.from_planet_id)?;
                    item.append(action.angle)?;
                    item.append(action.ships)?;
                    py_actions.append(item)?;
                }
                out.append(py_actions)?;
            }
        }
        Ok(out)
    }
}

impl RustCoreVecEnv {
    fn legal_target_mask_from_state_arrays<'py>(
        &mut self,
        py: Python<'py>,
        rows: Vec<(usize, usize)>,
        frac_v: Array2<f32>,
        active_v: Option<Array2<bool>>,
        batch: usize,
        planets: usize,
    ) -> PyResult<Bound<'py, numpy::PyArray3<bool>>> {
        let row_states = rows
            .iter()
            .map(|(env_idx, _)| self.legal_mask_state_cached(*env_idx, planets))
            .collect::<Vec<_>>();
        let stride = planets * planets;
        let games = &self.games;
        let flat = py.detach(|| {
            let mut flat = vec![false; batch * stride];
            flat.par_chunks_mut(stride)
                .enumerate()
                .for_each(|(row, row_out)| {
                    let (env_idx, player) = rows[row];
                    fill_legal_mask_row_from_state(
                        &games[env_idx],
                        &row_states[row],
                        player,
                        planets,
                        |col| frac_v[[row, col]] as f64,
                        |col| active_v.as_ref().is_none_or(|active| active[[row, col]]),
                        row_out,
                    );
                });
            flat
        });
        let out = Array3::from_shape_vec((batch, planets, planets), flat).map_err(|err| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to build legal mask: {err}"))
        })?;
        Ok(out.into_pyarray(py))
    }

    fn legal_target_mask_from_state_active_field_array<'py>(
        &mut self,
        py: Python<'py>,
        rows: Vec<(usize, usize)>,
        fields_v: Array3<f32>,
        batch: usize,
        planets: usize,
    ) -> PyResult<Bound<'py, numpy::PyArray3<bool>>> {
        let row_states = rows
            .iter()
            .map(|(env_idx, _)| self.legal_mask_state_cached(*env_idx, planets))
            .collect::<Vec<_>>();
        let stride = planets * planets;
        let games = &self.games;
        let flat = py.detach(|| {
            let mut flat = vec![false; batch * stride];
            flat.par_chunks_mut(stride)
                .enumerate()
                .for_each(|(row, row_out)| {
                    let (env_idx, player) = rows[row];
                    fill_legal_mask_row_from_state(
                        &games[env_idx],
                        &row_states[row],
                        player,
                        planets,
                        |col| fields_v[[row, col, 0]] as f64,
                        |col| fields_v[[row, col, 1]] >= 0.5,
                        row_out,
                    );
                });
            flat
        });
        let out = Array3::from_shape_vec((batch, planets, planets), flat).map_err(|err| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to build legal mask: {err}"))
        })?;
        Ok(out.into_pyarray(py))
    }

    fn reset_core(&mut self, advance_counts: bool) {
        let mut games = Vec::with_capacity(self.num_envs);
        for idx in 0..self.num_envs {
            let seed =
                self.random_seed + idx as u32 + self.reset_counts[idx] * self.num_envs as u32;
            if advance_counts {
                self.reset_counts[idx] += 1;
            }
            games.push(Game::new(
                GameConfig::new(self.num_players, self.episode_steps, self.ship_speed),
                seed,
            ));
        }
        self.games = games;
        self.legal_mask_cache = vec![None; self.games.len()];
    }

    fn legal_mask_state_cached(
        &mut self,
        env_idx: usize,
        planets_len: usize,
    ) -> Arc<LegalMaskState> {
        let game = &self.games[env_idx];
        let step = game.step;
        if let Some(entry) = self
            .legal_mask_cache
            .get(env_idx)
            .and_then(|entry| entry.as_ref())
        {
            if entry.step == step && entry.planets_len == planets_len {
                return Arc::clone(&entry.state);
            }
        }
        let state = Arc::new(legal_mask_state(game, planets_len));
        if env_idx >= self.legal_mask_cache.len() {
            self.legal_mask_cache.resize(env_idx + 1, None);
        }
        self.legal_mask_cache[env_idx] = Some(LegalMaskCacheEntry {
            step,
            planets_len,
            state: Arc::clone(&state),
        });
        state
    }
}

fn policy_batch_dict<'py>(
    py: Python<'py>,
    games: &[Game],
    rows: Vec<(usize, usize)>,
    include_contexts: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let batch = rows.len();
    let mut planet_feats = Array3::<f32>::zeros((batch, MAX_PLANETS, PLANET_FEAT_DIM));
    let mut planet_mask = Array2::<bool>::from_elem((batch, MAX_PLANETS), false);
    let mut planet_owned = Array2::<bool>::from_elem((batch, MAX_PLANETS), false);
    let mut planet_ids = Array2::<i64>::from_elem((batch, MAX_PLANETS), -1);
    let mut planet_garrison = Array2::<f32>::zeros((batch, MAX_PLANETS));
    let mut fleet_feats = Array3::<f32>::zeros((batch, MAX_FLEETS, FLEET_FEAT_DIM));
    let mut fleet_mask = Array2::<bool>::from_elem((batch, MAX_FLEETS), false);
    let contexts = PyList::empty(py);

    for (row, (env_idx, player)) in rows.into_iter().enumerate() {
        let game = &games[env_idx];
        fill_planet_features(
            game,
            player,
            row,
            &mut PlanetBatchMut {
                feats: &mut planet_feats,
                mask: &mut planet_mask,
                owned_mask: &mut planet_owned,
                ids: &mut planet_ids,
                garrison: &mut planet_garrison,
            },
        );
        fill_fleet_features(game, player, row, &mut fleet_feats, &mut fleet_mask);
        if include_contexts {
            let planet_rows = planet_rows_array(game);
            let comet_ids = game
                .comets
                .iter()
                .flat_map(|group| group.planet_ids.iter().copied())
                .map(i64::from)
                .collect::<Vec<_>>();
            contexts.append((
                planet_rows.into_pyarray(py),
                game.angular_velocity,
                Array1::from_vec(comet_ids).into_pyarray(py),
            ))?;
        }
    }

    let out = PyDict::new(py);
    out.set_item("planet_feats", planet_feats.into_pyarray(py))?;
    out.set_item("planet_mask", planet_mask.into_pyarray(py))?;
    out.set_item("planet_owned_mask", planet_owned.into_pyarray(py))?;
    out.set_item("planet_ids", planet_ids.into_pyarray(py))?;
    out.set_item("planet_garrison", planet_garrison.into_pyarray(py))?;
    out.set_item("fleet_feats", fleet_feats.into_pyarray(py))?;
    out.set_item("fleet_mask", fleet_mask.into_pyarray(py))?;
    out.set_item("contexts", contexts)?;
    Ok(out)
}

fn parse_env_actions(obj: &Bound<'_, PyAny>, num_players: usize) -> PyResult<Vec<PlayerAction>> {
    let mut out = vec![Vec::new(); num_players];
    let Ok(players) = obj.cast::<PyList>() else {
        return Ok(out);
    };
    for (player, player_out) in out.iter_mut().enumerate().take(num_players) {
        let Ok(player_obj) = players.get_item(player) else {
            continue;
        };
        if let Ok(native) = player_obj.extract::<PyRef<'_, NativeActionList>>() {
            *player_out = native.actions.clone();
            continue;
        }
        let Ok(moves) = player_obj.cast::<PyList>() else {
            continue;
        };
        for move_obj in moves.iter() {
            let Ok(row) = move_obj.cast::<PyList>() else {
                continue;
            };
            if row.len() < 3 {
                continue;
            }
            let from_planet_id = row.get_item(0)?.extract::<i32>()?;
            let angle = row.get_item(1)?.extract::<f64>()?;
            let ships = row.get_item(2)?.extract::<i32>()?;
            let target_id = get_or(row, 3, -1)?;
            let eta = get_or(row, 4, 0.0)?;
            let target_x = get_or(row, 5, 0.0)?;
            let target_y = get_or(row, 6, 0.0)?;
            player_out.push(Action {
                from_planet_id,
                angle,
                ships,
                target_id,
                eta,
                target_x,
                target_y,
            });
        }
    }
    Ok(out)
}

fn sniper_actions(game: &Game, player: usize) -> PlayerAction {
    let player = player as i32;
    let mut targets = game
        .planets
        .iter()
        .enumerate()
        .filter_map(|(idx, planet)| (planet.owner != player).then_some(idx))
        .collect::<Vec<_>>();
    if targets.is_empty() {
        return Vec::new();
    }
    let comet_ids = comet_id_set(game);
    let mut blockers = Vec::with_capacity(game.planets.len());
    let mut static_cols = Vec::new();
    let mut moving_cols = Vec::new();
    for (idx, planet) in game.planets.iter().enumerate() {
        let motion = target_motion(
            planet,
            game.angular_velocity,
            comet_ids
                .as_ref()
                .is_some_and(|ids| ids.contains(&planet.id)),
        );
        if motion.is_orbiting {
            moving_cols.push(idx);
        } else {
            static_cols.push(idx);
        }
        blockers.push(Some(motion));
    }

    let mut moves = Vec::new();
    for mine in game.planets.iter().filter(|planet| planet.owner == player) {
        targets.sort_by(|a, b| {
            distance_sq(mine, &game.planets[*a])
                .partial_cmp(&distance_sq(mine, &game.planets[*b]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &target_idx in &targets {
            let target = &game.planets[target_idx];
            let ships_needed = target.ships + 1;
            if mine.ships < ships_needed {
                continue;
            }
            let speed = fleet_speed_local(ships_needed, game.ship_speed);
            let Some(target_motion) = blockers.get(target_idx).and_then(|motion| motion.as_ref())
            else {
                continue;
            };
            let Some(solution) = lead_solution_cached_with_speed(mine, target_motion, speed) else {
                continue;
            };
            if !route_clear_to_solution_with_cols(
                mine.id,
                target.id,
                mine.x,
                mine.y,
                mine.radius,
                &solution,
                speed,
                &blockers,
                &static_cols,
                &moving_cols,
            ) {
                continue;
            }
            moves.push(Action::launch(mine.id, solution.angle, ships_needed));
            break;
        }
    }
    moves
}

fn production_margin(game: &Game, player: usize) -> f32 {
    let mut production = vec![0.0_f64; game.num_players];
    for planet in &game.planets {
        if planet.owner != -1 {
            production[planet.owner as usize] += planet.production as f64;
        }
    }
    let own = production[player];
    let enemy = production
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| (idx != player).then_some(*value))
        .fold(0.0_f64, f64::max);
    (own - enemy) as f32
}

fn distance_sq(a: &Planet, b: &Planet) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy
}

fn get_or<T>(row: &Bound<'_, PyList>, idx: usize, default: T) -> PyResult<T>
where
    T: for<'py> FromPyObject<'py, 'py, Error = PyErr> + Clone,
{
    if idx >= row.len() {
        return Ok(default);
    }
    row.get_item(idx)?.extract::<T>()
}

fn observation_dict<'py>(
    py: Python<'py>,
    game: &Game,
    player: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let obs = PyDict::new(py);
    obs.set_item("remainingOverageTime", 60)?;
    obs.set_item("step", game.step)?;
    obs.set_item("player", player)?;
    obs.set_item("planets", planet_rows_list(py, game)?)?;
    obs.set_item("fleets", fleet_rows_list(py, game)?)?;
    obs.set_item("fleet_targets", fleet_targets_dict(py, game, player)?)?;
    obs.set_item("angular_velocity", game.angular_velocity)?;
    obs.set_item("initial_planets", initial_planet_rows_list(py, game)?)?;
    obs.set_item("next_fleet_id", game.next_fleet_id)?;
    obs.set_item("comets", comets_list(py, game)?)?;
    obs.set_item("comet_planet_ids", comet_planet_ids(py, game)?)?;
    Ok(obs)
}

fn planet_rows_list<'py>(py: Python<'py>, game: &Game) -> PyResult<Bound<'py, PyList>> {
    let rows = PyList::empty(py);
    for p in &game.planets {
        rows.append((p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production))?;
    }
    Ok(rows)
}

fn initial_planet_rows_list<'py>(py: Python<'py>, game: &Game) -> PyResult<Bound<'py, PyList>> {
    let rows = PyList::empty(py);
    for p in &game.initial_planets {
        rows.append((p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production))?;
    }
    Ok(rows)
}

fn fleet_rows_list<'py>(py: Python<'py>, game: &Game) -> PyResult<Bound<'py, PyList>> {
    let rows = PyList::empty(py);
    for f in &game.fleets {
        rows.append((f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships))?;
    }
    Ok(rows)
}

fn fleet_targets_dict<'py>(
    py: Python<'py>,
    game: &Game,
    player: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for f in &game.fleets {
        if f.owner == player as i32 && f.target_id >= 0 && f.eta > 0.0 {
            out.set_item(
                f.id.to_string(),
                (f.target_id, f.eta.max(0.0), f.target_x, f.target_y),
            )?;
        }
    }
    Ok(out)
}

fn comets_list<'py>(py: Python<'py>, game: &Game) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for group in &game.comets {
        let dict = PyDict::new(py);
        dict.set_item("planet_ids", group.planet_ids.clone())?;
        let paths = PyList::empty(py);
        for path in &group.paths {
            let rows = PyList::empty(py);
            for p in path {
                rows.append((p.x, p.y))?;
            }
            paths.append(rows)?;
        }
        dict.set_item("paths", paths)?;
        dict.set_item("path_index", group.path_index)?;
        out.append(dict)?;
    }
    Ok(out)
}

fn comet_planet_ids<'py>(py: Python<'py>, game: &Game) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for group in &game.comets {
        for pid in &group.planet_ids {
            out.append(*pid)?;
        }
    }
    Ok(out)
}

fn planet_rows_array(game: &Game) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((game.planets.len(), 7));
    for (i, p) in game.planets.iter().enumerate() {
        out[[i, 0]] = p.id as f64;
        out[[i, 1]] = p.owner as f64;
        out[[i, 2]] = p.x;
        out[[i, 3]] = p.y;
        out[[i, 4]] = p.radius;
        out[[i, 5]] = p.ships as f64;
        out[[i, 6]] = p.production as f64;
    }
    out
}

struct PlanetBatchMut<'a> {
    feats: &'a mut Array3<f32>,
    mask: &'a mut Array2<bool>,
    owned_mask: &'a mut Array2<bool>,
    ids: &'a mut Array2<i64>,
    garrison: &'a mut Array2<f32>,
}

fn fill_planet_features(game: &Game, player: usize, row: usize, batch: &mut PlanetBatchMut<'_>) {
    let comet_motion = comet_motion_by_id(game);
    for (col, p) in game.planets.iter().take(MAX_PLANETS).enumerate() {
        let owner = p.owner;
        let x = p.x;
        let y = p.y;
        let rx = x - CENTER;
        let ry = y - CENTER;
        let orbital_radius = (rx * rx + ry * ry).sqrt();
        batch.feats[[row, col, 0]] = (rx / BOARD_SIZE) as f32;
        batch.feats[[row, col, 1]] = (ry / BOARD_SIZE) as f32;
        batch.feats[[row, col, 2]] = (orbital_radius / BOARD_SIZE) as f32;
        batch.feats[[row, col, 3]] = (p.radius / 5.0) as f32;
        batch.feats[[row, col, 4]] = ((p.ships as f64).ln_1p() / 8.0) as f32;
        batch.feats[[row, col, 5]] = (p.production as f64 / 5.0) as f32;
        if let Some(step) = comet_motion.get(&p.id) {
            batch.feats[[row, col, 12]] = 1.0;
            if let Some((dx, dy)) = step {
                let norm = (dx * dx + dy * dy).sqrt();
                if norm > 0.0 {
                    batch.feats[[row, col, 6]] = (dx / norm) as f32;
                    batch.feats[[row, col, 7]] = (dy / norm) as f32;
                    batch.feats[[row, col, 8]] = (norm / MAX_SHIP_SPEED).min(1.0) as f32;
                }
            }
        } else {
            let is_orbiting = orbital_radius + p.radius < ROTATION_RADIUS_LIMIT;
            if is_orbiting && orbital_radius > 1e-9 {
                let vx = -ry * game.angular_velocity;
                let vy = rx * game.angular_velocity;
                let speed = (vx * vx + vy * vy).sqrt();
                batch.feats[[row, col, 6]] = (vx / speed.max(1e-9)) as f32;
                batch.feats[[row, col, 7]] = (vy / speed.max(1e-9)) as f32;
                batch.feats[[row, col, 8]] = (speed / MAX_SHIP_SPEED).min(1.0) as f32;
                batch.feats[[row, col, 10]] = (game.angular_velocity.abs() / MAX_OMEGA) as f32;
                batch.feats[[row, col, 11]] = 1.0;
            }
            batch.feats[[row, col, 9]] = (orbital_radius / 50.0) as f32;
        }
        fill_owner_features(batch.feats, row, col, 13, owner, player, game.num_players);
        batch.feats[[row, col, 18]] = 1.0;
        batch.mask[[row, col]] = true;
        batch.owned_mask[[row, col]] = owner == player as i32;
        batch.ids[[row, col]] = p.id as i64;
        batch.garrison[[row, col]] = p.ships as f32;
    }
}

fn fill_fleet_features(
    game: &Game,
    player: usize,
    row: usize,
    feats: &mut Array3<f32>,
    mask: &mut Array2<bool>,
) {
    let pos_by_id = game
        .planets
        .iter()
        .map(|p| (p.id, (p.x, p.y)))
        .collect::<HashMap<_, _>>();
    for (col, f) in game.fleets.iter().take(MAX_FLEETS).enumerate() {
        feats[[row, col, 0]] = ((f.x - CENTER) / BOARD_SIZE) as f32;
        feats[[row, col, 1]] = ((f.y - CENTER) / BOARD_SIZE) as f32;
        feats[[row, col, 2]] = f.angle.cos() as f32;
        feats[[row, col, 3]] = f.angle.sin() as f32;
        feats[[row, col, 4]] = ((f.ships as f64).ln_1p() / 8.0) as f32;
        if let Some((sx, sy)) = pos_by_id.get(&f.from_planet_id) {
            feats[[row, col, 5]] = ((*sx - CENTER) / BOARD_SIZE) as f32;
            feats[[row, col, 6]] = ((*sy - CENTER) / BOARD_SIZE) as f32;
            feats[[row, col, 7]] = 1.0;
        }
        let sp = 1.0 + (MAX_SHIP_SPEED - 1.0) * ((f.ships.max(1) as f64).ln() / LOG_1000).powf(1.5);
        feats[[row, col, 8]] = (sp.min(MAX_SHIP_SPEED) / MAX_SHIP_SPEED).min(1.0) as f32;
        if f.owner == player as i32 && f.target_id >= 0 {
            feats[[row, col, 9]] = (f.target_id as f32) / 128.0;
            feats[[row, col, 10]] = (f.eta.max(0.0) / 500.0).min(1.0) as f32;
            feats[[row, col, 11]] = ((f.target_x - CENTER) / BOARD_SIZE) as f32;
            feats[[row, col, 12]] = ((f.target_y - CENTER) / BOARD_SIZE) as f32;
            feats[[row, col, 13]] = 1.0;
        }
        fill_owner_features(feats, row, col, 14, f.owner, player, game.num_players);
        mask[[row, col]] = true;
    }
}

fn fill_owner_features(
    feats: &mut Array3<f32>,
    row: usize,
    col: usize,
    start: usize,
    owner: i32,
    player: usize,
    num_players: usize,
) {
    if owner == player as i32 {
        feats[[row, col, start]] = 1.0;
    } else if owner == -1 {
        feats[[row, col, start + 1]] = 1.0;
    } else {
        let diff = (owner - player as i32).rem_euclid(num_players.max(2) as i32);
        let slot = diff - 1;
        if (0..=2).contains(&slot) {
            feats[[row, col, start + 2 + slot as usize]] = 1.0;
        }
    }
}

fn comet_motion_by_id(game: &Game) -> HashMap<i32, Option<(f64, f64)>> {
    let mut out = HashMap::new();
    for group in &game.comets {
        for pid in &group.planet_ids {
            out.insert(*pid, None);
        }
        let idx = group.path_index.max(0) as usize;
        for (i, pid) in group.planet_ids.iter().copied().enumerate() {
            let Some(path) = group.paths.get(i) else {
                continue;
            };
            if idx + 1 >= path.len() {
                continue;
            }
            out.insert(
                pid,
                Some((path[idx + 1].x - path[idx].x, path[idx + 1].y - path[idx].y)),
            );
        }
    }
    out
}

fn fill_legal_mask_row<FFrac, FOwned, FMask, FIds>(
    game: &Game,
    planets_len: usize,
    frac_at: FFrac,
    owned_at: FOwned,
    mask_at: FMask,
    id_at: FIds,
    out: &mut [bool],
) where
    FFrac: Fn(usize) -> f64,
    FOwned: Fn(usize) -> bool,
    FMask: Fn(usize) -> bool,
    FIds: Fn(usize) -> i32,
{
    let planets_by_col = planets_by_col(game, planets_len, &id_at, &mask_at);
    let comet_ids = comet_id_set(game);
    let is_comet_col = (0..planets_len)
        .map(|col| {
            let target_id = id_at(col);
            target_id >= 0
                && comet_ids
                    .as_ref()
                    .is_some_and(|ids| ids.contains(&target_id))
        })
        .collect::<Vec<_>>();
    let target_motions = planets_by_col
        .iter()
        .enumerate()
        .map(|(col, planet)| {
            planet.map(|target| {
                cached_target_motion(&target, game.angular_velocity, is_comet_col[col])
            })
        })
        .collect::<Vec<_>>();
    for i in 0..planets_len {
        if !(owned_at(i) && mask_at(i)) {
            continue;
        }
        let Some(source) = planets_by_col[i] else {
            continue;
        };
        if source.ships < 2 {
            continue;
        }
        let send = ships_to_send(source.ships, frac_at(i));
        if send <= 0 {
            continue;
        }
        let speed = fleet_speed_local(send, game.ship_speed);
        if speed <= 0.0 {
            continue;
        }
        for j in 0..planets_len {
            if i == j || !mask_at(j) {
                continue;
            }
            let Some(target) = target_motions[j].as_ref() else {
                continue;
            };
            let Some(solution) = lead_solution_cached_with_speed(&source, target, speed) else {
                continue;
            };
            if !route_clear_to_solution(
                source.id,
                target.id,
                source.x,
                source.y,
                source.radius,
                &solution,
                speed,
                &target_motions,
            ) {
                continue;
            }
            out[i * planets_len + j] = true;
        }
    }
}

fn fill_legal_mask_row_from_state<FFrac, FActive>(
    game: &Game,
    state: &LegalMaskState,
    player: usize,
    planets_len: usize,
    frac_at: FFrac,
    active_at: FActive,
    out: &mut [bool],
) where
    FFrac: Fn(usize) -> f64,
    FActive: Fn(usize) -> bool,
{
    let planet_limit = state.planet_limit.min(planets_len);

    let Some(source_cols) = state.source_cols_by_player.get(player) else {
        return;
    };
    for &i in source_cols {
        if i >= planet_limit || !active_at(i) {
            continue;
        }
        let source = game.planets[i];
        if source.owner != player as i32 || source.ships < 2 {
            continue;
        }
        let send = ships_to_send(source.ships, frac_at(i));
        if send <= 0 {
            continue;
        }
        let speed = fleet_speed_local(send, game.ship_speed);
        if speed <= 0.0 {
            continue;
        }
        let src_offset = i * planets_len;
        for &j in &state.target_cols {
            if i == j {
                continue;
            }
            let Some(solution) = lead_solution_cached_with_speed(
                &source,
                state.target_motions[j]
                    .as_ref()
                    .expect("target column is present"),
                speed,
            ) else {
                continue;
            };
            if !route_clear_to_solution_with_cols(
                source.id,
                game.planets[j].id,
                source.x,
                source.y,
                source.radius,
                &solution,
                speed,
                &state.target_motions,
                &state.static_cols,
                &state.moving_cols,
            ) {
                continue;
            }
            out[src_offset + j] = true;
        }
    }
}

fn legal_mask_state(game: &Game, planets_len: usize) -> LegalMaskState {
    let planet_limit = planets_len.min(game.planets.len());
    let comet_ids = comet_id_set(game);
    let is_comet_col = game
        .planets
        .iter()
        .take(planet_limit)
        .map(|target| {
            comet_ids
                .as_ref()
                .is_some_and(|ids| ids.contains(&target.id))
        })
        .collect::<Vec<_>>();
    let target_motions = game
        .planets
        .iter()
        .take(planet_limit)
        .enumerate()
        .map(|(idx, target)| {
            Some(cached_target_motion(
                target,
                game.angular_velocity,
                is_comet_col[idx],
            ))
        })
        .collect::<Vec<_>>();
    let target_cols = (0..planet_limit).collect::<Vec<_>>();
    let mut source_cols_by_player = vec![Vec::new(); game.num_players];
    for (idx, planet) in game.planets.iter().take(planet_limit).enumerate() {
        if planet.owner < 0 || planet.ships < 2 {
            continue;
        }
        let owner = planet.owner as usize;
        if let Some(cols) = source_cols_by_player.get_mut(owner) {
            cols.push(idx);
        }
    }
    let static_cols = target_motions
        .iter()
        .enumerate()
        .filter_map(|(idx, motion)| {
            motion
                .as_ref()
                .is_some_and(|motion| !motion.is_orbiting)
                .then_some(idx)
        })
        .collect::<Vec<_>>();
    let moving_cols = target_motions
        .iter()
        .enumerate()
        .filter_map(|(idx, motion)| {
            motion
                .as_ref()
                .is_some_and(|motion| motion.is_orbiting)
                .then_some(idx)
        })
        .collect::<Vec<_>>();
    LegalMaskState {
        planet_limit,
        target_motions,
        target_cols,
        source_cols_by_player,
        static_cols,
        moving_cols,
    }
}

fn build_materialized_actions_dict<'py>(
    py: Python<'py>,
    batch: usize,
    planets: usize,
    native: bool,
    results: &[RowActionResult],
) -> PyResult<Bound<'py, PyDict>> {
    let actions = PyList::empty(py);
    let mut materialized = Array2::<bool>::from_elem((batch, planets), false);
    for (row, result) in results.iter().enumerate() {
        if native {
            actions.append(Py::new(
                py,
                NativeActionList {
                    actions: result.actions.clone(),
                },
            )?)?;
        } else {
            let row_actions = PyList::empty(py);
            for action in &result.actions {
                let item = PyList::empty(py);
                item.append(action.from_planet_id)?;
                item.append(action.angle)?;
                item.append(action.ships)?;
                item.append(action.target_id)?;
                item.append(action.eta)?;
                item.append(action.target_x)?;
                item.append(action.target_y)?;
                row_actions.append(item)?;
            }
            actions.append(row_actions)?;
        }
        for col in 0..planets {
            materialized[[row, col]] = result.materialized[col];
        }
    }
    let out = PyDict::new(py);
    out.set_item("actions", actions)?;
    out.set_item("materialized", materialized.into_pyarray(py))?;
    Ok(out)
}

fn planets_by_col<FIds, FMask>(
    game: &Game,
    planets_len: usize,
    id_at: &FIds,
    mask_at: &FMask,
) -> Vec<Option<Planet>>
where
    FIds: Fn(usize) -> i32,
    FMask: Fn(usize) -> bool,
{
    (0..planets_len)
        .map(|col| {
            if !mask_at(col) {
                return None;
            }
            let id = id_at(col);
            if id < 0 {
                return None;
            }
            if let Some(planet) = game.planets.get(col).filter(|planet| planet.id == id) {
                return Some(*planet);
            }
            game.planets.iter().find(|planet| planet.id == id).copied()
        })
        .collect()
}

fn comet_id_set(game: &Game) -> Option<HashSet<i32>> {
    (!game.comets.is_empty()).then(|| {
        game.comets
            .iter()
            .flat_map(|group| group.planet_ids.iter().copied())
            .collect()
    })
}

fn materialize_action_row<FLaunch, FTarget, FFrac, FOwned, FMask, FIds>(
    game: &Game,
    planets_len: usize,
    launch_at: FLaunch,
    target_at: FTarget,
    frac_at: FFrac,
    owned_at: FOwned,
    mask_at: FMask,
    id_at: FIds,
) -> RowActionResult
where
    FLaunch: Fn(usize) -> f64,
    FTarget: Fn(usize) -> usize,
    FFrac: Fn(usize) -> f64,
    FOwned: Fn(usize) -> bool,
    FMask: Fn(usize) -> bool,
    FIds: Fn(usize) -> i32,
{
    let by_id = game
        .planets
        .iter()
        .map(|p| (p.id, *p))
        .collect::<HashMap<_, _>>();
    let comet_ids = comet_id_set(game);
    let mut remaining = game
        .planets
        .iter()
        .map(|p| (p.id, p.ships))
        .collect::<HashMap<_, _>>();
    let blockers = game
        .planets
        .iter()
        .map(|planet| {
            Some(target_motion(
                planet,
                game.angular_velocity,
                comet_ids
                    .as_ref()
                    .is_some_and(|ids| ids.contains(&planet.id)),
            ))
        })
        .collect::<Vec<_>>();
    let mut result = RowActionResult {
        actions: Vec::new(),
        materialized: vec![false; planets_len],
    };
    for i in 0..planets_len {
        if !(launch_at(i) >= 0.5 && owned_at(i) && mask_at(i)) {
            continue;
        }
        let ti = target_at(i);
        if ti == i || ti >= planets_len {
            continue;
        }
        let target_id = id_at(ti);
        if target_id < 0 {
            continue;
        }
        let source_id = id_at(i);
        let (Some(source), Some(_target)) = (
            by_id.get(&source_id).copied(),
            by_id.get(&target_id).copied(),
        ) else {
            continue;
        };
        let source_ships = *remaining.get(&source_id).unwrap_or(&source.ships);
        if source_ships < 2 {
            continue;
        }
        let send = ships_to_send(source_ships, frac_at(i));
        if send <= 0 {
            continue;
        }
        let speed = fleet_speed_local(send, game.ship_speed);
        if speed <= 0.0 {
            continue;
        }
        let Some(target_motion) = blockers.get(ti).and_then(|motion| motion.as_ref()) else {
            continue;
        };
        let Some(solution) = lead_solution_cached_with_speed(&source, target_motion, speed) else {
            continue;
        };
        if !route_clear_to_solution(
            source.id,
            target_id,
            source.x,
            source.y,
            source.radius,
            &solution,
            speed,
            &blockers,
        ) {
            continue;
        }
        result.actions.push(Action {
            from_planet_id: source.id,
            angle: solution.angle,
            ships: send,
            target_id,
            eta: solution.time,
            target_x: solution.x,
            target_y: solution.y,
        });
        result.materialized[i] = true;
        remaining.insert(source_id, source_ships - send);
    }
    result
}

fn materialize_action_row_from_state<FLaunch, FTarget, FFrac>(
    game: &Game,
    player: usize,
    planets_len: usize,
    launch_at: FLaunch,
    target_at: FTarget,
    frac_at: FFrac,
) -> RowActionResult
where
    FLaunch: Fn(usize) -> f64,
    FTarget: Fn(usize) -> usize,
    FFrac: Fn(usize) -> f64,
{
    let planet_limit = planets_len.min(game.planets.len());
    let mut target_motions: Option<Vec<Option<TargetMotion>>> = None;
    let mut result = RowActionResult {
        actions: Vec::new(),
        materialized: vec![false; planets_len],
    };
    let mut remaining = game
        .planets
        .iter()
        .take(planet_limit)
        .map(|planet| planet.ships)
        .collect::<Vec<_>>();
    for i in 0..planet_limit {
        if launch_at(i) < 0.5 {
            continue;
        }
        let source = game.planets[i];
        let source_ships = remaining[i];
        if source.owner != player as i32 || source_ships < 2 {
            continue;
        }
        let ti = target_at(i);
        if ti == i || ti >= planet_limit {
            continue;
        }
        let target = game.planets[ti];
        let target_id = target.id;
        let send = ships_to_send(source_ships, frac_at(i));
        if send <= 0 {
            continue;
        }
        let speed = fleet_speed_local(send, game.ship_speed);
        if speed <= 0.0 {
            continue;
        }
        let target_motions = target_motions.get_or_insert_with(|| {
            let comet_ids = comet_id_set(game);
            game.planets
                .iter()
                .take(planet_limit)
                .map(|planet| {
                    Some(target_motion(
                        planet,
                        game.angular_velocity,
                        comet_ids
                            .as_ref()
                            .is_some_and(|ids| ids.contains(&planet.id)),
                    ))
                })
                .collect::<Vec<_>>()
        });
        let Some(target_motion) = target_motions.get(ti).and_then(|motion| motion.as_ref()) else {
            continue;
        };
        let Some(solution) = lead_solution_cached_with_speed(&source, target_motion, speed) else {
            continue;
        };
        if !route_clear_to_solution(
            source.id,
            target_id,
            source.x,
            source.y,
            source.radius,
            &solution,
            speed,
            target_motions,
        ) {
            continue;
        }
        result.actions.push(Action {
            from_planet_id: source.id,
            angle: solution.angle,
            ships: send,
            target_id,
            eta: solution.time,
            target_x: solution.x,
            target_y: solution.y,
        });
        result.materialized[i] = true;
        remaining[i] = source_ships - send;
    }
    result
}

fn materialize_masked_action_row_from_state<FLaunch, FTarget, FFrac>(
    game: &Game,
    player: usize,
    planets_len: usize,
    launch_at: FLaunch,
    target_at: FTarget,
    frac_at: FFrac,
) -> RowActionResult
where
    FLaunch: Fn(usize) -> f64,
    FTarget: Fn(usize) -> usize,
    FFrac: Fn(usize) -> f64,
{
    let planet_limit = planets_len.min(game.planets.len());
    let mut target_motions: Option<Vec<Option<TargetMotion>>> = None;
    let mut result = RowActionResult {
        actions: Vec::new(),
        materialized: vec![false; planets_len],
    };
    let mut remaining = game
        .planets
        .iter()
        .take(planet_limit)
        .map(|planet| planet.ships)
        .collect::<Vec<_>>();
    for i in 0..planet_limit {
        if launch_at(i) < 0.5 {
            continue;
        }
        let source = game.planets[i];
        let source_ships = remaining[i];
        if source.owner != player as i32 || source_ships < 2 {
            continue;
        }
        let ti = target_at(i);
        if ti == i || ti >= planet_limit {
            continue;
        }
        let target = game.planets[ti];
        let target_id = target.id;
        let send = ships_to_send(source_ships, frac_at(i));
        if send <= 0 {
            continue;
        }
        let speed = fleet_speed_local(send, game.ship_speed);
        if speed <= 0.0 {
            continue;
        }
        let target_motions = target_motions.get_or_insert_with(|| {
            let comet_ids = comet_id_set(game);
            game.planets
                .iter()
                .take(planet_limit)
                .map(|planet| {
                    Some(target_motion(
                        planet,
                        game.angular_velocity,
                        comet_ids
                            .as_ref()
                            .is_some_and(|ids| ids.contains(&planet.id)),
                    ))
                })
                .collect::<Vec<_>>()
        });
        let Some(target_motion) = target_motions.get(ti).and_then(|motion| motion.as_ref()) else {
            continue;
        };
        let Some(solution) = lead_solution_cached_with_speed(&source, target_motion, speed) else {
            continue;
        };
        if !route_clear_to_solution(
            source.id,
            target_id,
            source.x,
            source.y,
            source.radius,
            &solution,
            speed,
            target_motions,
        ) {
            continue;
        }
        result.actions.push(Action {
            from_planet_id: source.id,
            angle: solution.angle,
            ships: send,
            target_id,
            eta: solution.time,
            target_x: solution.x,
            target_y: solution.y,
        });
        result.materialized[i] = true;
        remaining[i] = source_ships - send;
    }
    result
}

fn ships_to_send(remaining: i32, frac: f64) -> i32 {
    if remaining < 2 {
        return 0;
    }
    let raw = (remaining as f64 * frac.clamp(0.0, 1.0)).round() as i32;
    raw.clamp(1, remaining - 1)
}

fn fleet_speed_local(ships: i32, max_speed: f64) -> f64 {
    if ships <= 1 {
        return 1.0;
    }
    let frac = (ships as f64).ln() / LOG_1000;
    (1.0 + (max_speed - 1.0) * frac.powf(1.5)).min(max_speed)
}

fn target_motion(target: &Planet, angular_velocity: f64, is_comet: bool) -> TargetMotion {
    target_motion_with_cache(target, angular_velocity, is_comet, false)
}

fn cached_target_motion(target: &Planet, angular_velocity: f64, is_comet: bool) -> TargetMotion {
    target_motion_with_cache(target, angular_velocity, is_comet, true)
}

fn target_motion_with_cache(
    target: &Planet,
    angular_velocity: f64,
    is_comet: bool,
    cache_positions: bool,
) -> TargetMotion {
    let orbit_radius = ((target.x - CENTER).powi(2) + (target.y - CENTER).powi(2)).sqrt();
    let is_orbiting = !is_comet
        && orbit_radius + target.radius < ROTATION_RADIUS_LIMIT
        && angular_velocity.abs() > 1e-12
        && orbit_radius > 1e-9;
    if !is_orbiting {
        return TargetMotion {
            id: target.id,
            x: target.x,
            y: target.y,
            radius: target.radius,
            radius_sq: target.radius * target.radius,
            is_orbiting: false,
            theta0: 0.0,
            orbit_radius: 0.0,
            angular_velocity: 0.0,
            max_turns: 0,
            positions: None,
        };
    }
    let theta0 = (target.y - CENTER).atan2(target.x - CENTER);
    let max_turns = bounded_lead_scan_turns(1.0, target.radius) as usize;
    let positions = cache_positions.then(|| {
        (0..max_turns)
            .map(|step| {
                let theta = theta0 + angular_velocity * step as f64;
                (
                    CENTER + orbit_radius * theta.cos(),
                    CENTER + orbit_radius * theta.sin(),
                )
            })
            .collect()
    });
    TargetMotion {
        id: target.id,
        x: target.x,
        y: target.y,
        radius: target.radius,
        radius_sq: target.radius * target.radius,
        is_orbiting: true,
        theta0,
        orbit_radius,
        angular_velocity,
        max_turns,
        positions,
    }
}

fn lead_solution_cached_with_speed(
    source: &Planet,
    target: &TargetMotion,
    speed: f64,
) -> Option<LeadSolution> {
    let mut solution =
        lead_solution_from_point_cached_with_speed(source.x, source.y, target, speed)?;
    let mut angle = solution.angle;
    let offset = (source.radius + 0.1).max(0.0);
    if offset <= 0.0 {
        return Some(solution);
    }
    for _ in 0..4 {
        let start_x = source.x + angle.cos() * offset;
        let start_y = source.y + angle.sin() * offset;
        let refined = lead_solution_from_point_cached_with_speed(start_x, start_y, target, speed)?;
        if angle_delta(refined.angle, angle).abs() < 1e-6 {
            return Some(refined);
        }
        angle = refined.angle;
        solution = refined;
    }
    Some(solution)
}

fn lead_solution_from_point_cached_with_speed(
    source_x: f64,
    source_y: f64,
    target: &TargetMotion,
    speed: f64,
) -> Option<LeadSolution> {
    if speed <= 0.0 {
        return None;
    }
    if !target.is_orbiting {
        let distance = ((target.x - source_x).powi(2) + (target.y - source_y).powi(2)).sqrt();
        if distance / speed > LEAD_T_HORIZON_STEPS {
            return None;
        }
        return Some(LeadSolution {
            angle: (target.y - source_y).atan2(target.x - source_x),
            time: ((distance - target.radius).max(0.0) / speed)
                .ceil()
                .max(1.0),
            x: target.x,
            y: target.y,
        });
    }
    let max_turns = bounded_lead_scan_turns(speed, target.radius) as usize;
    let mut previous_error: Option<f64> = None;
    for idx in 0..max_turns.min(target.max_turns) {
        let (tx, ty) = motion_position_at(target, idx);
        let k = (idx + 1) as i32;
        let distance = ((tx - source_x).powi(2) + (ty - source_y).powi(2)).sqrt();
        let error = distance - k as f64 * speed;
        if error <= target.radius {
            let prev_dist = ((k - 1) as f64 * speed).max(0.0);
            if distance >= prev_dist - target.radius {
                return Some(LeadSolution {
                    angle: (ty - source_y).atan2(tx - source_x),
                    time: k as f64,
                    x: tx,
                    y: ty,
                });
            }
        }
        if let Some(prev) = previous_error {
            if prev < -target.radius && error > target.radius {
                break;
            }
        }
        previous_error = Some(error);
    }
    None
}

fn bounded_lead_scan_turns(speed: f64, target_radius: f64) -> i32 {
    LEAD_MAX_TURNS
        .min((((LEAD_MAX_SCAN_DISTANCE + target_radius) / speed).ceil() as i32 + 1).max(1))
}

#[cfg(test)]
fn lead_solution(
    source: &Planet,
    target: &Planet,
    angular_velocity: f64,
    send: i32,
    max_speed: f64,
) -> Option<LeadSolution> {
    let mut solution = lead_solution_from_point(
        source.x,
        source.y,
        target.x,
        target.y,
        target.radius,
        angular_velocity,
        send,
        max_speed,
    )?;
    let mut angle = solution.angle;
    let offset = (source.radius + 0.1).max(0.0);
    if offset <= 0.0 {
        return Some(solution);
    }
    for _ in 0..4 {
        let start_x = source.x + angle.cos() * offset;
        let start_y = source.y + angle.sin() * offset;
        let refined = lead_solution_from_point(
            start_x,
            start_y,
            target.x,
            target.y,
            target.radius,
            angular_velocity,
            send,
            max_speed,
        )?;
        if angle_delta(refined.angle, angle).abs() < 1e-6 {
            return Some(refined);
        }
        angle = refined.angle;
        solution = refined;
    }
    Some(solution)
}

#[cfg(test)]
fn lead_solution_from_point(
    source_x: f64,
    source_y: f64,
    target_x: f64,
    target_y: f64,
    target_radius: f64,
    angular_velocity: f64,
    send: i32,
    max_speed: f64,
) -> Option<LeadSolution> {
    let speed = fleet_speed_local(send, max_speed);
    if speed <= 0.0 {
        return None;
    }
    let orbit_radius = ((target_x - CENTER).powi(2) + (target_y - CENTER).powi(2)).sqrt();
    let is_orbiting = orbit_radius + target_radius < ROTATION_RADIUS_LIMIT
        && angular_velocity.abs() > 1e-12
        && orbit_radius > 1e-9;
    if !is_orbiting {
        let distance = ((target_x - source_x).powi(2) + (target_y - source_y).powi(2)).sqrt();
        if distance / speed > LEAD_T_HORIZON_STEPS {
            return None;
        }
        return Some(LeadSolution {
            angle: (target_y - source_y).atan2(target_x - source_x),
            time: ((distance - target_radius).max(0.0) / speed)
                .ceil()
                .max(1.0),
            x: target_x,
            y: target_y,
        });
    }
    let theta0 = (target_y - CENTER).atan2(target_x - CENTER);
    let mut previous_error: Option<f64> = None;
    let max_turns = bounded_lead_scan_turns(speed, target_radius);
    for k in 1..=max_turns {
        let theta = theta0 + angular_velocity * (k - 1) as f64;
        let tx = CENTER + orbit_radius * theta.cos();
        let ty = CENTER + orbit_radius * theta.sin();
        let distance = ((tx - source_x).powi(2) + (ty - source_y).powi(2)).sqrt();
        let error = distance - k as f64 * speed;
        if error <= target_radius {
            let prev_dist = ((k - 1) as f64 * speed).max(0.0);
            if distance >= prev_dist - target_radius {
                return Some(LeadSolution {
                    angle: (ty - source_y).atan2(tx - source_x),
                    time: k as f64,
                    x: tx,
                    y: ty,
                });
            }
        }
        if let Some(prev) = previous_error {
            if prev < -target_radius && error > target_radius {
                break;
            }
        }
        previous_error = Some(error);
    }
    None
}

fn motion_position_at(motion: &TargetMotion, steps: usize) -> (f64, f64) {
    if !motion.is_orbiting || steps == 0 {
        return (motion.x, motion.y);
    }
    if let Some(positions) = motion.positions.as_ref() {
        return positions
            .get(steps)
            .copied()
            .unwrap_or_else(|| positions.last().copied().unwrap_or((motion.x, motion.y)));
    }
    let bounded_steps = steps.min(motion.max_turns.saturating_sub(1));
    let theta = motion.theta0 + motion.angular_velocity * bounded_steps as f64;
    (
        CENTER + motion.orbit_radius * theta.cos(),
        CENTER + motion.orbit_radius * theta.sin(),
    )
}

fn route_clear_to_solution(
    source_id: i32,
    target_id: i32,
    source_x: f64,
    source_y: f64,
    source_radius: f64,
    solution: &LeadSolution,
    speed: f64,
    blockers: &[Option<TargetMotion>],
) -> bool {
    if speed <= 0.0 {
        return false;
    }
    let offset = (source_radius + 0.1).max(0.0);
    let dir_x = solution.angle.cos();
    let dir_y = solution.angle.sin();
    let start = (source_x + dir_x * offset, source_y + dir_y * offset);
    let final_turn = (solution.time.ceil() as usize).max(1);
    let final_point = (
        start.0 + dir_x * speed * final_turn as f64,
        start.1 + dir_y * speed * final_turn as f64,
    );
    if !inside_board(start.0, start.1) || !inside_board(final_point.0, final_point.1) {
        return false;
    }
    if point_to_segment_distance_sq_local((CENTER, CENTER), start, final_point)
        < SUN_RADIUS * SUN_RADIUS
    {
        return false;
    }

    for motion in blockers.iter().flatten() {
        if motion.id == source_id || motion.id == target_id || motion.is_orbiting {
            continue;
        }
        if point_to_segment_distance_sq_local((motion.x, motion.y), start, final_point)
            < motion.radius_sq
        {
            return false;
        }
    }
    if !blockers
        .iter()
        .flatten()
        .any(|motion| motion.id != target_id && motion.is_orbiting)
    {
        return true;
    }

    for turn in 1..=final_turn {
        let old = (
            start.0 + dir_x * speed * (turn - 1) as f64,
            start.1 + dir_y * speed * (turn - 1) as f64,
        );
        let new = (
            start.0 + dir_x * speed * turn as f64,
            start.1 + dir_y * speed * turn as f64,
        );
        for motion in blockers.iter().flatten() {
            if motion.id == target_id || !motion.is_orbiting {
                continue;
            }
            let pos = motion_position_at(motion, turn - 1);
            if motion.id != source_id
                && point_to_segment_distance_sq_local(pos, old, new) < motion.radius_sq
            {
                return false;
            }
            if turn < final_turn {
                let next_pos = motion_position_at(motion, turn);
                if point_to_segment_distance_sq_local(new, pos, next_pos) < motion.radius_sq {
                    return false;
                }
            }
        }
    }
    true
}

fn route_clear_to_solution_with_cols(
    source_id: i32,
    target_id: i32,
    source_x: f64,
    source_y: f64,
    source_radius: f64,
    solution: &LeadSolution,
    speed: f64,
    blockers: &[Option<TargetMotion>],
    static_cols: &[usize],
    moving_cols: &[usize],
) -> bool {
    if speed <= 0.0 {
        return false;
    }
    let offset = (source_radius + 0.1).max(0.0);
    let dir_x = solution.angle.cos();
    let dir_y = solution.angle.sin();
    let start = (source_x + dir_x * offset, source_y + dir_y * offset);
    let final_turn = (solution.time.ceil() as usize).max(1);
    let final_point = (
        start.0 + dir_x * speed * final_turn as f64,
        start.1 + dir_y * speed * final_turn as f64,
    );
    if !inside_board(start.0, start.1) || !inside_board(final_point.0, final_point.1) {
        return false;
    }
    if point_to_segment_distance_sq_local((CENTER, CENTER), start, final_point)
        < SUN_RADIUS * SUN_RADIUS
    {
        return false;
    }

    for &idx in static_cols {
        let Some(motion) = blockers[idx].as_ref() else {
            continue;
        };
        if motion.id == source_id || motion.id == target_id {
            continue;
        }
        if point_to_segment_distance_sq_local((motion.x, motion.y), start, final_point)
            < motion.radius_sq
        {
            return false;
        }
    }
    if moving_cols.is_empty()
        || moving_cols.iter().all(|&idx| {
            blockers[idx]
                .as_ref()
                .is_none_or(|motion| motion.id == target_id)
        })
    {
        return true;
    }

    for turn in 1..=final_turn {
        let old = (
            start.0 + dir_x * speed * (turn - 1) as f64,
            start.1 + dir_y * speed * (turn - 1) as f64,
        );
        let new = (
            start.0 + dir_x * speed * turn as f64,
            start.1 + dir_y * speed * turn as f64,
        );
        for &idx in moving_cols {
            let Some(motion) = blockers[idx].as_ref() else {
                continue;
            };
            if motion.id == target_id {
                continue;
            }
            let pos = motion_position_at(motion, turn - 1);
            if motion.id != source_id
                && point_to_segment_distance_sq_local(pos, old, new) < motion.radius_sq
            {
                return false;
            }
            if turn < final_turn {
                let next_pos = motion_position_at(motion, turn);
                if point_to_segment_distance_sq_local(new, pos, next_pos) < motion.radius_sq {
                    return false;
                }
            }
        }
    }
    true
}

fn inside_board(x: f64, y: f64) -> bool {
    (0.0..=BOARD_SIZE).contains(&x) && (0.0..=BOARD_SIZE).contains(&y)
}

fn point_to_segment_distance_sq_local(
    point: (f64, f64),
    start: (f64, f64),
    end: (f64, f64),
) -> f64 {
    let seg_x = end.0 - start.0;
    let seg_y = end.1 - start.1;
    let l2 = seg_x * seg_x + seg_y * seg_y;
    if l2 == 0.0 {
        let dx = point.0 - start.0;
        let dy = point.1 - start.1;
        return dx * dx + dy * dy;
    }
    let raw_t = ((point.0 - start.0) * seg_x + (point.1 - start.1) * seg_y) / l2;
    let t = raw_t.clamp(0.0, 1.0);
    let proj = (start.0 + t * seg_x, start.1 + t * seg_y);
    let dx = point.0 - proj.0;
    let dy = point.1 - proj.1;
    dx * dx + dy * dy
}

fn angle_delta(a: f64, b: f64) -> f64 {
    (a - b).sin().atan2((a - b).cos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use owars_env::{CometGroup, GameState, Point};

    #[test]
    fn comet_legal_mask_and_materializer_use_same_motion_model() {
        let planets = vec![
            Planet {
                id: 13,
                owner: 0,
                x: 30.230528337169282,
                y: 37.113714576315715,
                radius: 1.0,
                ships: 2,
                production: 1,
            },
            Planet {
                id: 18,
                owner: 1,
                x: 70.0,
                y: 75.0,
                radius: 1.0,
                ships: 20,
                production: 1,
            },
            Planet {
                id: 29,
                owner: -1,
                x: 47.52883413524248,
                y: 97.05854540070332,
                radius: 1.0,
                ships: 7,
                production: 1,
            },
        ];
        let game = Game::from_state(
            GameConfig::new(2, 500, 6.0),
            GameState::new(
                50,
                0.03,
                planets.clone(),
                planets,
                vec![],
                vec![CometGroup {
                    planet_ids: vec![29],
                    paths: vec![vec![
                        Point::new(47.52883413524248, 97.05854540070332),
                        Point::new(47.0, 95.0),
                    ]],
                    path_index: 0,
                }],
                0,
            ),
        );
        let planets_len = 3;
        let frac = 0.16880422830581665;
        let legal_state = legal_mask_state(&game, planets_len);
        let mut legal = vec![false; planets_len * planets_len];
        fill_legal_mask_row_from_state(
            &game,
            &legal_state,
            0,
            planets_len,
            |col| if col == 0 { frac } else { 0.5 },
            |_| true,
            &mut legal,
        );
        assert!(legal[2], "source 0 should be allowed to target comet col 2");

        let result = materialize_masked_action_row_from_state(
            &game,
            0,
            planets_len,
            |col| if col == 0 { 1.0 } else { 0.0 },
            |col| if col == 0 { 2 } else { 0 },
            |col| if col == 0 { frac } else { 0.5 },
        );
        assert!(result.materialized[0]);
        assert_eq!(result.actions.len(), 1);
        assert_eq!(result.actions[0].from_planet_id, 13);
        assert_eq!(result.actions[0].target_id, 29);
    }

    #[test]
    fn native_sniper_treats_comet_target_as_non_orbiting() {
        let planets = vec![
            Planet {
                id: 13,
                owner: 0,
                x: 30.230528337169282,
                y: 37.113714576315715,
                radius: 1.0,
                ships: 20,
                production: 1,
            },
            Planet {
                id: 29,
                owner: -1,
                x: 47.52883413524248,
                y: 97.05854540070332,
                radius: 1.0,
                ships: 7,
                production: 1,
            },
        ];
        let game = Game::from_state(
            GameConfig::new(2, 500, 6.0),
            GameState::new(
                50,
                0.04,
                planets.clone(),
                planets,
                vec![],
                vec![CometGroup {
                    planet_ids: vec![29],
                    paths: vec![vec![
                        Point::new(47.52883413524248, 97.05854540070332),
                        Point::new(47.0, 95.0),
                    ]],
                    path_index: 0,
                }],
                0,
            ),
        );
        let source = game.planets[0];
        let target = game.planets[1];
        let ships_needed = target.ships + 1;
        let speed = fleet_speed_local(ships_needed, game.ship_speed);
        let comet_motion = target_motion(&target, game.angular_velocity, true);
        let expected = lead_solution_cached_with_speed(&source, &comet_motion, speed).unwrap();
        let old_orbit_inferred = lead_solution(
            &source,
            &target,
            game.angular_velocity,
            ships_needed,
            game.ship_speed,
        )
        .unwrap();

        let actions = sniper_actions(&game, 0);

        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].from_planet_id, 13);
        assert_eq!(actions[0].ships, ships_needed);
        assert!((actions[0].angle - expected.angle).abs() < 1e-12);
        assert!(angle_delta(actions[0].angle, old_orbit_inferred.angle).abs() > 0.1);
    }
}

#[pymodule]
fn _owars_env(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustCoreVecEnv>()?;
    m.add_class::<NativeActionList>()?;
    Ok(())
}
