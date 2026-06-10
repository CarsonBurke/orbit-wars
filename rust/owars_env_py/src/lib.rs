use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use owars_env::oracle::{self, FleetDestination};
use owars_env::{
    Action, CometGroup, Fleet, Game, GameConfig, GameState, Planet, PlayerAction, Point,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySequence};
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

#[derive(Clone, Copy)]
struct SniperProfile {
    reserve_base: i32,
    reserve_production: f64,
    send_buffer: i32,
    enemy_growth: bool,
    enemy_value: f64,
    neutral_value: f64,
    production_weight: f64,
    ship_cost_weight: f64,
    time_cost_weight: f64,
    duplicate_penalty: f64,
    allow_partial: bool,
    partial_min_fraction: f64,
    partial_score_scale: f64,
    net_defense_reserve: bool,
    defense_horizon: f64,
    contested_extra_buffer: i32,
    contested_window: f64,
    reinforce_owned: bool,
    defense_arrival_slack: f64,
    defense_score_weight: f64,
    chronological_forecast: bool,
    comet_max_eta: Option<f64>,
    counter_recapture: bool,
    recapture_min_gap: f64,
    recapture_max_gap: f64,
    recapture_score_weight: f64,
    recapture_gap_cost: f64,
    aggressive_sources: bool,
    speed_bid: bool,
    speed_bid_max_factor: f64,
    speed_bid_tempo_weight: f64,
    global_assignment: bool,
    strict_defense: bool,
    shadow_capture: bool,
}

fn default_sniper_profile() -> SniperProfile {
    SniperProfile {
        reserve_base: 0,
        reserve_production: 0.0,
        send_buffer: 1,
        enemy_growth: false,
        enemy_value: 1.0,
        neutral_value: 1.0,
        production_weight: 1.0,
        ship_cost_weight: 1.0,
        time_cost_weight: 0.0,
        duplicate_penalty: 0.0,
        allow_partial: false,
        partial_min_fraction: 0.5,
        partial_score_scale: 0.45,
        net_defense_reserve: false,
        defense_horizon: 35.0,
        contested_extra_buffer: 0,
        contested_window: 2.0,
        reinforce_owned: false,
        defense_arrival_slack: 1.0,
        defense_score_weight: 7.5,
        chronological_forecast: false,
        comet_max_eta: None,
        counter_recapture: false,
        recapture_min_gap: 0.5,
        recapture_max_gap: 8.0,
        recapture_score_weight: 6.0,
        recapture_gap_cost: 0.25,
        aggressive_sources: true,
        speed_bid: false,
        speed_bid_max_factor: 1.5,
        speed_bid_tempo_weight: 0.45,
        global_assignment: false,
        strict_defense: false,
        shadow_capture: false,
    }
}

fn sniper_profile_from_dict(profile: &Bound<'_, PyDict>) -> PyResult<SniperProfile> {
    let mut out = default_sniper_profile();
    out.reserve_base = dict_i32(profile, "reserve_base", out.reserve_base)?;
    out.reserve_production = dict_f64(profile, "reserve_production", out.reserve_production)?;
    out.send_buffer = dict_i32(profile, "send_buffer", out.send_buffer)?;
    out.enemy_growth = dict_bool(profile, "enemy_growth", out.enemy_growth)?;
    out.enemy_value = dict_f64(profile, "enemy_value", out.enemy_value)?;
    out.neutral_value = dict_f64(profile, "neutral_value", out.neutral_value)?;
    out.production_weight = dict_f64(profile, "production_weight", out.production_weight)?;
    out.ship_cost_weight = dict_f64(profile, "ship_cost_weight", out.ship_cost_weight)?;
    out.time_cost_weight = dict_f64(profile, "time_cost_weight", out.time_cost_weight)?;
    out.duplicate_penalty = dict_f64(profile, "duplicate_penalty", out.duplicate_penalty)?;
    out.allow_partial = dict_bool(profile, "allow_partial", out.allow_partial)?;
    out.partial_min_fraction = dict_f64(profile, "partial_min_fraction", out.partial_min_fraction)?;
    out.partial_score_scale = dict_f64(profile, "partial_score_scale", out.partial_score_scale)?;
    out.net_defense_reserve = dict_bool(profile, "net_defense_reserve", out.net_defense_reserve)?;
    out.defense_horizon = dict_f64(profile, "defense_horizon", out.defense_horizon)?;
    out.contested_extra_buffer = dict_i32(
        profile,
        "contested_extra_buffer",
        out.contested_extra_buffer,
    )?;
    out.contested_window = dict_f64(profile, "contested_window", out.contested_window)?;
    out.reinforce_owned = dict_bool(profile, "reinforce_owned", out.reinforce_owned)?;
    out.defense_arrival_slack =
        dict_f64(profile, "defense_arrival_slack", out.defense_arrival_slack)?;
    out.defense_score_weight = dict_f64(profile, "defense_score_weight", out.defense_score_weight)?;
    out.chronological_forecast = dict_bool(
        profile,
        "chronological_forecast",
        out.chronological_forecast,
    )?;
    out.comet_max_eta = dict_optional_f64(profile, "comet_max_eta", out.comet_max_eta)?;
    out.counter_recapture = dict_bool(profile, "counter_recapture", out.counter_recapture)?;
    out.recapture_min_gap = dict_f64(profile, "recapture_min_gap", out.recapture_min_gap)?;
    out.recapture_max_gap = dict_f64(profile, "recapture_max_gap", out.recapture_max_gap)?;
    out.recapture_score_weight = dict_f64(
        profile,
        "recapture_score_weight",
        out.recapture_score_weight,
    )?;
    out.recapture_gap_cost = dict_f64(profile, "recapture_gap_cost", out.recapture_gap_cost)?;
    out.aggressive_sources = dict_bool(profile, "aggressive_sources", out.aggressive_sources)?;
    out.speed_bid = dict_bool(profile, "speed_bid", out.speed_bid)?;
    out.speed_bid_max_factor = dict_f64(profile, "speed_bid_max_factor", out.speed_bid_max_factor)?;
    out.speed_bid_tempo_weight = dict_f64(
        profile,
        "speed_bid_tempo_weight",
        out.speed_bid_tempo_weight,
    )?;
    out.global_assignment = dict_bool(profile, "global_assignment", out.global_assignment)?;
    out.strict_defense = dict_bool(profile, "strict_defense", out.strict_defense)?;
    out.shadow_capture = dict_bool(profile, "shadow_capture", out.shadow_capture)?;
    Ok(out)
}

fn dict_f64(profile: &Bound<'_, PyDict>, key: &str, default: f64) -> PyResult<f64> {
    Ok(match profile.get_item(key)? {
        Some(value) => value.extract::<f64>()?,
        None => default,
    })
}

fn dict_i32(profile: &Bound<'_, PyDict>, key: &str, default: i32) -> PyResult<i32> {
    Ok(match profile.get_item(key)? {
        Some(value) => value.extract::<i32>()?,
        None => default,
    })
}

fn dict_bool(profile: &Bound<'_, PyDict>, key: &str, default: bool) -> PyResult<bool> {
    Ok(match profile.get_item(key)? {
        Some(value) => value.extract::<bool>()?,
        None => default,
    })
}

fn dict_optional_f64(
    profile: &Bound<'_, PyDict>,
    key: &str,
    default: Option<f64>,
) -> PyResult<Option<f64>> {
    Ok(match profile.get_item(key)? {
        Some(value) if value.is_none() => None,
        Some(value) => Some(value.extract::<f64>()?),
        None => default,
    })
}

#[derive(Clone, Copy)]
struct ScoredAction {
    target_idx: usize,
    eta: f64,
    score: f64,
    action: Action,
}

#[derive(Clone, Copy)]
struct PressureEntry {
    eta: f64,
    owner: i32,
    ships: i32,
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
    /// Per-env destination-oracle output, keyed implicitly by env state:
    /// every state mutation (reset, reset_subset, load_observation,
    /// step_subset_fast) must clear the slot explicitly.
    oracle_cache: Vec<Option<Arc<Vec<FleetDestination>>>>,
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
            oracle_cache: Vec::new(),
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
            self.oracle_cache[env_idx] = None;
        }
        Ok(())
    }

    fn load_observation(&mut self, idx: usize, obs: Bound<'_, PyDict>) -> PyResult<()> {
        if idx >= self.games.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "env index out of range",
            ));
        }
        self.games[idx] =
            game_from_observation(&obs, self.num_players, self.episode_steps, self.ship_speed)?;
        self.legal_mask_cache[idx] = None;
        self.oracle_cache[idx] = None;
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
            if env_idx < self.oracle_cache.len() {
                self.oracle_cache[env_idx] = None;
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

    /// Exact fleet-destination oracle for a batch of (env_idx, player) rows.
    ///
    /// Output is player-independent (all fleets on the board are resolved),
    /// so duplicate env rows share one computation via the per-env cache.
    /// The cache is invalidated by every state mutation (reset, reset_subset,
    /// load_observation, step_subset_fast).
    fn fleet_destination_oracle<'py>(
        &mut self,
        py: Python<'py>,
        rows: Vec<(usize, usize)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.fleet_destination_oracle_impl(py, rows, false)
    }

    /// Same contract as `fleet_destination_oracle`, but runs the literal
    /// simulator-mirror rollout. Slow; exists for parity testing only.
    fn fleet_destination_oracle_reference<'py>(
        &mut self,
        py: Python<'py>,
        rows: Vec<(usize, usize)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.fleet_destination_oracle_impl(py, rows, true)
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
        let action_fn: fn(&Game, usize) -> PlayerAction = match name {
            "sniper" => sniper_actions,
            "sniper_v2" => sniper_v2_actions,
            "sniper_v3" => sniper_v3_actions,
            "sniper_v4" => sniper_v4_actions,
            "sniper_v5" => sniper_v5_actions,
            "sniper_v6" => sniper_v6_actions,
            "sniper_v7" => sniper_v7_actions,
            "sniper_v8" => sniper_v8_actions,
            "sniper_v9" => sniper_v9_actions,
            "sniper_v10" => sniper_v10_actions,
            "sniper_v11" => sniper_v11_actions,
            "sniper_v12" => sniper_v12_actions,
            "sniper_v13" => sniper_v13_actions,
            "sniper_v14" => sniper_v14_actions,
            "sniper_v15" => sniper_v15_actions,
            "sniper_v16" => sniper_v16_actions,
            "sniper_v17" => sniper_v17_actions,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unsupported native builtin opponent: {name}"
                )));
            }
        };
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
                .map(|(env_idx, player)| action_fn(&games[env_idx], player))
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

    fn sniper_profile_actions<'py>(
        &self,
        py: Python<'py>,
        profile: Bound<'py, PyDict>,
        rows: Vec<(usize, usize)>,
        native: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let profile = sniper_profile_from_dict(&profile)?;
        for &(env_idx, player) in &rows {
            if env_idx >= self.games.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "env index out of range",
                ));
            }
            if player >= self.games[env_idx].num_players {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "player index out of range",
                ));
            }
        }
        let games = &self.games;
        let actions = py.detach(|| {
            rows.into_par_iter()
                .map(|(env_idx, player)| scored_sniper_actions(&games[env_idx], player, profile))
                .collect::<Vec<_>>()
        });
        let out = PyList::empty(py);
        for action_list in actions {
            if native {
                out.append(Py::new(
                    py,
                    NativeActionList {
                        actions: action_list,
                    },
                )?)?;
            } else {
                let row = PyList::empty(py);
                for action in action_list {
                    let item = PyList::empty(py);
                    item.append(action.from_planet_id)?;
                    item.append(action.angle)?;
                    item.append(action.ships)?;
                    row.append(item)?;
                }
                out.append(row)?;
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
        self.oracle_cache = vec![None; self.games.len()];
    }

    fn fleet_destination_oracle_impl<'py>(
        &mut self,
        py: Python<'py>,
        rows: Vec<(usize, usize)>,
        use_reference: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
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
        if self.oracle_cache.len() < self.games.len() {
            self.oracle_cache.resize(self.games.len(), None);
        }
        // Dedup by env: oracle output is per-board, so duplicate player rows
        // resolve to the same Arc. The reference path bypasses the cache to
        // keep parity tests honest.
        let mut pending: Vec<usize> = Vec::new();
        for &(env_idx, _) in &rows {
            if (use_reference || self.oracle_cache[env_idx].is_none())
                && !pending.contains(&env_idx)
            {
                pending.push(env_idx);
            }
        }
        let games = &self.games;
        let computed = py.detach(|| {
            pending
                .par_iter()
                .map(|&env_idx| {
                    let game = &games[env_idx];
                    let dests = if use_reference {
                        oracle::infer_fleet_destinations_reference(game, MAX_FLEETS)
                    } else {
                        oracle::infer_fleet_destinations(game, MAX_FLEETS)
                    };
                    (env_idx, Arc::new(dests))
                })
                .collect::<Vec<_>>()
        });
        let mut reference_results: Vec<Option<Arc<Vec<FleetDestination>>>> = if use_reference {
            vec![None; self.games.len()]
        } else {
            Vec::new()
        };
        for (env_idx, dests) in computed {
            if use_reference {
                reference_results[env_idx] = Some(dests);
            } else {
                self.oracle_cache[env_idx] = Some(dests);
            }
        }

        let batch = rows.len();
        let mut dest_idx = Array2::<i64>::from_elem((batch, MAX_FLEETS), -1);
        let mut eta = Array2::<f64>::zeros((batch, MAX_FLEETS));
        let mut status = Array2::<i64>::from_elem((batch, MAX_FLEETS), oracle::STATUS_NONE);
        for (row, &(env_idx, _)) in rows.iter().enumerate() {
            let entries = if use_reference {
                reference_results[env_idx]
                    .as_ref()
                    .expect("reference oracle output missing for requested env")
            } else {
                self.oracle_cache[env_idx]
                    .as_ref()
                    .expect("oracle cache entry missing for requested env")
            };
            for (col, entry) in entries.iter().enumerate().take(MAX_FLEETS) {
                dest_idx[[row, col]] = entry.dest_idx;
                eta[[row, col]] = entry.eta;
                status[[row, col]] = entry.status;
            }
        }
        let out = PyDict::new(py);
        out.set_item("dest_idx", dest_idx.into_pyarray(py))?;
        out.set_item("eta", eta.into_pyarray(py))?;
        out.set_item("status", status.into_pyarray(py))?;
        Ok(out)
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

fn game_from_observation(
    obs: &Bound<'_, PyDict>,
    num_players: usize,
    episode_steps: i32,
    ship_speed: f64,
) -> PyResult<Game> {
    let step = dict_get_i32(obs, "step", 0)?;
    let angular_velocity = dict_get_f64(obs, "angular_velocity", 0.0)?;
    let planets = parse_planets_from_dict(obs, "planets")?;
    let mut initial_planets = parse_planets_from_dict(obs, "initial_planets")?;
    if initial_planets.is_empty() {
        initial_planets = planets.clone();
    }
    let fleets = parse_fleets_from_dict(obs, "fleets")?;
    let comets = parse_comets_from_dict(obs)?;
    let next_fleet_id = dict_get_i32(obs, "next_fleet_id", 0)?;
    Ok(Game::from_state(
        GameConfig::new(num_players, episode_steps, ship_speed),
        GameState::new(
            step,
            angular_velocity,
            planets,
            initial_planets,
            fleets,
            comets,
            next_fleet_id,
        ),
    ))
}

fn dict_get_f64(obs: &Bound<'_, PyDict>, key: &str, default: f64) -> PyResult<f64> {
    Ok(match obs.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<f64>()?,
        _ => default,
    })
}

fn dict_get_i32(obs: &Bound<'_, PyDict>, key: &str, default: i32) -> PyResult<i32> {
    Ok(match obs.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<i32>()?,
        _ => default,
    })
}

fn parse_planets_from_dict(obs: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<Planet>> {
    let Some(obj) = obs.get_item(key)? else {
        return Ok(Vec::new());
    };
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let rows = obj.cast::<PyList>()?;
    let mut out = Vec::with_capacity(rows.len());
    for row_obj in rows.iter() {
        let row = row_obj.cast::<PySequence>()?;
        if row.len()? < 7 {
            continue;
        }
        out.push(Planet {
            id: row.get_item(0)?.extract::<i32>()?,
            owner: row.get_item(1)?.extract::<i32>()?,
            x: row.get_item(2)?.extract::<f64>()?,
            y: row.get_item(3)?.extract::<f64>()?,
            radius: row.get_item(4)?.extract::<f64>()?,
            ships: row.get_item(5)?.extract::<i32>()?,
            production: row.get_item(6)?.extract::<i32>()?,
        });
    }
    Ok(out)
}

fn parse_fleets_from_dict(obs: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<Fleet>> {
    let Some(obj) = obs.get_item(key)? else {
        return Ok(Vec::new());
    };
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let rows = obj.cast::<PyList>()?;
    let mut out = Vec::with_capacity(rows.len());
    for row_obj in rows.iter() {
        let row = row_obj.cast::<PySequence>()?;
        if row.len()? < 7 {
            continue;
        }
        out.push(Fleet {
            id: row.get_item(0)?.extract::<i32>()?,
            owner: row.get_item(1)?.extract::<i32>()?,
            x: row.get_item(2)?.extract::<f64>()?,
            y: row.get_item(3)?.extract::<f64>()?,
            angle: row.get_item(4)?.extract::<f64>()?,
            from_planet_id: row.get_item(5)?.extract::<i32>()?,
            ships: row.get_item(6)?.extract::<i32>()?,
            target_id: get_seq_or(&row, 7, -1)?,
            eta: get_seq_or(&row, 8, 0.0)?,
            target_x: get_seq_or(&row, 9, 0.0)?,
            target_y: get_seq_or(&row, 10, 0.0)?,
        });
    }
    Ok(out)
}

fn parse_comets_from_dict(obs: &Bound<'_, PyDict>) -> PyResult<Vec<CometGroup>> {
    let Some(obj) = obs.get_item("comets")? else {
        return Ok(Vec::new());
    };
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let rows = obj.cast::<PyList>()?;
    let mut out = Vec::with_capacity(rows.len());
    for group_obj in rows.iter() {
        let group = group_obj.cast::<PyDict>()?;
        let planet_ids = match group.get_item("planet_ids")? {
            Some(value) if !value.is_none() => value.extract::<Vec<i32>>()?,
            _ => Vec::new(),
        };
        let path_index = match group.get_item("path_index")? {
            Some(value) if !value.is_none() => value.extract::<i32>()?,
            _ => -1,
        };
        let mut paths = Vec::new();
        if let Some(paths_obj) = group.get_item("paths")? {
            let path_rows = paths_obj.cast::<PyList>()?;
            for path_obj in path_rows.iter() {
                let point_rows = path_obj.cast::<PyList>()?;
                let mut path = Vec::with_capacity(point_rows.len());
                for point_obj in point_rows.iter() {
                    let point = point_obj.cast::<PySequence>()?;
                    if point.len()? < 2 {
                        continue;
                    }
                    path.push(Point::new(
                        point.get_item(0)?.extract::<f64>()?,
                        point.get_item(1)?.extract::<f64>()?,
                    ));
                }
                paths.push(path);
            }
        }
        out.push(CometGroup {
            planet_ids,
            paths,
            path_index,
        });
    }
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
        let is_comet = comet_ids
            .as_ref()
            .is_some_and(|ids| ids.contains(&planet.id));
        let motion = target_motion(planet, game.angular_velocity, is_comet);
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

fn sniper_v2_actions(game: &Game, player: usize) -> PlayerAction {
    scored_sniper_actions(
        game,
        player,
        SniperProfile {
            reserve_base: 6,
            reserve_production: 1.2,
            send_buffer: 2,
            enemy_growth: true,
            enemy_value: 2.4,
            neutral_value: 1.35,
            production_weight: 5.0,
            ship_cost_weight: 0.82,
            time_cost_weight: 0.55,
            duplicate_penalty: 0.45,
            allow_partial: false,
            partial_min_fraction: 0.5,
            partial_score_scale: 0.45,
            net_defense_reserve: false,
            defense_horizon: 35.0,
            contested_extra_buffer: 0,
            contested_window: 2.0,
            reinforce_owned: false,
            defense_arrival_slack: 1.0,
            defense_score_weight: 7.5,
            chronological_forecast: false,
            comet_max_eta: None,
            counter_recapture: false,
            recapture_min_gap: 0.5,
            recapture_max_gap: 8.0,
            recapture_score_weight: 6.0,
            recapture_gap_cost: 0.25,
            aggressive_sources: true,
            ..default_sniper_profile()
        },
    )
}

fn sniper_v3_actions(game: &Game, player: usize) -> PlayerAction {
    scored_sniper_actions(
        game,
        player,
        SniperProfile {
            reserve_base: 3,
            reserve_production: 0.7,
            send_buffer: 1,
            enemy_growth: true,
            enemy_value: 3.1,
            neutral_value: 1.15,
            production_weight: 5.8,
            ship_cost_weight: 0.72,
            time_cost_weight: 0.42,
            duplicate_penalty: 0.30,
            allow_partial: false,
            partial_min_fraction: 0.5,
            partial_score_scale: 0.45,
            net_defense_reserve: false,
            defense_horizon: 35.0,
            contested_extra_buffer: 0,
            contested_window: 2.0,
            reinforce_owned: false,
            defense_arrival_slack: 1.0,
            defense_score_weight: 7.5,
            chronological_forecast: false,
            comet_max_eta: None,
            counter_recapture: false,
            recapture_min_gap: 0.5,
            recapture_max_gap: 8.0,
            recapture_score_weight: 6.0,
            recapture_gap_cost: 0.25,
            aggressive_sources: true,
            ..default_sniper_profile()
        },
    )
}

fn sniper_v4_actions(game: &Game, player: usize) -> PlayerAction {
    scored_sniper_actions(
        game,
        player,
        SniperProfile {
            reserve_base: 2,
            reserve_production: 0.5,
            send_buffer: 1,
            enemy_growth: true,
            enemy_value: 3.4,
            neutral_value: 1.05,
            production_weight: 6.2,
            ship_cost_weight: 0.68,
            time_cost_weight: 0.35,
            duplicate_penalty: 0.22,
            allow_partial: false,
            partial_min_fraction: 0.5,
            partial_score_scale: 0.45,
            net_defense_reserve: false,
            defense_horizon: 35.0,
            contested_extra_buffer: 0,
            contested_window: 2.0,
            reinforce_owned: false,
            defense_arrival_slack: 1.0,
            defense_score_weight: 7.5,
            chronological_forecast: false,
            comet_max_eta: None,
            counter_recapture: false,
            recapture_min_gap: 0.5,
            recapture_max_gap: 8.0,
            recapture_score_weight: 6.0,
            recapture_gap_cost: 0.25,
            aggressive_sources: true,
            ..default_sniper_profile()
        },
    )
}

fn sniper_v5_actions(game: &Game, player: usize) -> PlayerAction {
    scored_sniper_actions(
        game,
        player,
        SniperProfile {
            reserve_base: 2,
            reserve_production: 0.45,
            send_buffer: 1,
            enemy_growth: true,
            enemy_value: 3.7,
            neutral_value: 1.0,
            production_weight: 6.6,
            ship_cost_weight: 0.64,
            time_cost_weight: 0.34,
            duplicate_penalty: 0.16,
            allow_partial: true,
            partial_min_fraction: 0.38,
            partial_score_scale: 0.62,
            net_defense_reserve: false,
            defense_horizon: 35.0,
            contested_extra_buffer: 0,
            contested_window: 2.0,
            reinforce_owned: false,
            defense_arrival_slack: 1.0,
            defense_score_weight: 7.5,
            chronological_forecast: false,
            comet_max_eta: None,
            counter_recapture: false,
            recapture_min_gap: 0.5,
            recapture_max_gap: 8.0,
            recapture_score_weight: 6.0,
            recapture_gap_cost: 0.25,
            aggressive_sources: true,
            ..default_sniper_profile()
        },
    )
}

fn sniper_v6_actions(game: &Game, player: usize) -> PlayerAction {
    scored_sniper_actions(
        game,
        player,
        SniperProfile {
            reserve_base: 1,
            reserve_production: 0.35,
            send_buffer: 1,
            enemy_growth: true,
            enemy_value: 3.55,
            neutral_value: 1.0,
            production_weight: 6.5,
            ship_cost_weight: 0.66,
            time_cost_weight: 0.34,
            duplicate_penalty: 0.20,
            allow_partial: false,
            partial_min_fraction: 0.5,
            partial_score_scale: 0.45,
            net_defense_reserve: true,
            defense_horizon: 42.0,
            contested_extra_buffer: 0,
            contested_window: 2.0,
            reinforce_owned: false,
            defense_arrival_slack: 1.0,
            defense_score_weight: 7.5,
            chronological_forecast: false,
            comet_max_eta: None,
            counter_recapture: false,
            recapture_min_gap: 0.5,
            recapture_max_gap: 8.0,
            recapture_score_weight: 6.0,
            recapture_gap_cost: 0.25,
            aggressive_sources: true,
            ..default_sniper_profile()
        },
    )
}

fn sniper_v7_actions(game: &Game, player: usize) -> PlayerAction {
    scored_sniper_actions(
        game,
        player,
        SniperProfile {
            reserve_base: 2,
            reserve_production: 0.5,
            send_buffer: 1,
            enemy_growth: true,
            enemy_value: 3.5,
            neutral_value: 1.0,
            production_weight: 6.4,
            ship_cost_weight: 0.67,
            time_cost_weight: 0.34,
            duplicate_penalty: 0.20,
            allow_partial: false,
            partial_min_fraction: 0.5,
            partial_score_scale: 0.45,
            net_defense_reserve: false,
            defense_horizon: 35.0,
            contested_extra_buffer: 2,
            contested_window: 2.0,
            reinforce_owned: false,
            defense_arrival_slack: 1.0,
            defense_score_weight: 7.5,
            chronological_forecast: false,
            comet_max_eta: None,
            counter_recapture: false,
            recapture_min_gap: 0.5,
            recapture_max_gap: 8.0,
            recapture_score_weight: 6.0,
            recapture_gap_cost: 0.25,
            aggressive_sources: true,
            ..default_sniper_profile()
        },
    )
}

fn sniper_v8_actions(game: &Game, player: usize) -> PlayerAction {
    scored_sniper_actions(
        game,
        player,
        SniperProfile {
            reserve_base: 1,
            reserve_production: 0.35,
            send_buffer: 1,
            enemy_growth: true,
            enemy_value: 3.55,
            neutral_value: 1.0,
            production_weight: 6.5,
            ship_cost_weight: 0.66,
            time_cost_weight: 0.34,
            duplicate_penalty: 0.20,
            allow_partial: false,
            partial_min_fraction: 0.5,
            partial_score_scale: 0.45,
            net_defense_reserve: true,
            defense_horizon: 42.0,
            contested_extra_buffer: 0,
            contested_window: 2.0,
            reinforce_owned: true,
            defense_arrival_slack: 1.0,
            defense_score_weight: 9.0,
            chronological_forecast: false,
            comet_max_eta: None,
            counter_recapture: false,
            recapture_min_gap: 0.5,
            recapture_max_gap: 8.0,
            recapture_score_weight: 6.0,
            recapture_gap_cost: 0.25,
            aggressive_sources: true,
            ..default_sniper_profile()
        },
    )
}

fn sniper_v9_actions(game: &Game, player: usize) -> PlayerAction {
    scored_sniper_actions(
        game,
        player,
        SniperProfile {
            reserve_base: 1,
            reserve_production: 0.35,
            send_buffer: 1,
            enemy_growth: true,
            enemy_value: 3.55,
            neutral_value: 1.0,
            production_weight: 6.5,
            ship_cost_weight: 0.66,
            time_cost_weight: 0.34,
            duplicate_penalty: 0.20,
            allow_partial: false,
            partial_min_fraction: 0.5,
            partial_score_scale: 0.45,
            net_defense_reserve: true,
            defense_horizon: 42.0,
            contested_extra_buffer: 0,
            contested_window: 2.0,
            reinforce_owned: true,
            defense_arrival_slack: 1.0,
            defense_score_weight: 9.0,
            chronological_forecast: true,
            comet_max_eta: None,
            counter_recapture: false,
            recapture_min_gap: 0.5,
            recapture_max_gap: 8.0,
            recapture_score_weight: 6.0,
            recapture_gap_cost: 0.25,
            aggressive_sources: true,
            ..default_sniper_profile()
        },
    )
}

fn sniper_v10_actions(game: &Game, player: usize) -> PlayerAction {
    scored_sniper_actions(
        game,
        player,
        SniperProfile {
            reserve_base: 1,
            reserve_production: 0.35,
            send_buffer: 1,
            enemy_growth: true,
            enemy_value: 3.55,
            neutral_value: 1.0,
            production_weight: 6.5,
            ship_cost_weight: 0.66,
            time_cost_weight: 0.34,
            duplicate_penalty: 0.20,
            allow_partial: false,
            partial_min_fraction: 0.5,
            partial_score_scale: 0.45,
            net_defense_reserve: true,
            defense_horizon: 42.0,
            contested_extra_buffer: 0,
            contested_window: 2.0,
            reinforce_owned: true,
            defense_arrival_slack: 1.0,
            defense_score_weight: 9.0,
            chronological_forecast: false,
            comet_max_eta: Some(8.0),
            counter_recapture: false,
            recapture_min_gap: 0.5,
            recapture_max_gap: 8.0,
            recapture_score_weight: 6.0,
            recapture_gap_cost: 0.25,
            aggressive_sources: true,
            ..default_sniper_profile()
        },
    )
}

fn sniper_v11_actions(game: &Game, player: usize) -> PlayerAction {
    scored_sniper_actions(
        game,
        player,
        SniperProfile {
            reserve_base: 1,
            reserve_production: 0.35,
            send_buffer: 1,
            enemy_growth: true,
            enemy_value: 3.55,
            neutral_value: 1.0,
            production_weight: 6.5,
            ship_cost_weight: 0.66,
            time_cost_weight: 0.34,
            duplicate_penalty: 0.20,
            allow_partial: false,
            partial_min_fraction: 0.5,
            partial_score_scale: 0.45,
            net_defense_reserve: true,
            defense_horizon: 42.0,
            contested_extra_buffer: 0,
            contested_window: 2.0,
            reinforce_owned: true,
            defense_arrival_slack: 1.0,
            defense_score_weight: 9.0,
            chronological_forecast: false,
            comet_max_eta: Some(8.0),
            counter_recapture: true,
            recapture_min_gap: 0.5,
            recapture_max_gap: 8.0,
            recapture_score_weight: 6.0,
            recapture_gap_cost: 0.25,
            aggressive_sources: true,
            ..default_sniper_profile()
        },
    )
}

fn sniper_v12_actions(game: &Game, player: usize) -> PlayerAction {
    scored_sniper_actions(
        game,
        player,
        SniperProfile {
            reserve_base: 1,
            reserve_production: 0.35,
            send_buffer: 1,
            enemy_growth: true,
            enemy_value: 3.55,
            neutral_value: 1.0,
            production_weight: 6.5,
            ship_cost_weight: 0.66,
            time_cost_weight: 0.34,
            duplicate_penalty: 0.20,
            net_defense_reserve: true,
            defense_horizon: 42.0,
            reinforce_owned: true,
            defense_arrival_slack: 1.0,
            defense_score_weight: 9.0,
            comet_max_eta: Some(8.0),
            speed_bid: true,
            ..default_sniper_profile()
        },
    )
}

fn sniper_v13_actions(game: &Game, player: usize) -> PlayerAction {
    scored_sniper_actions(
        game,
        player,
        SniperProfile {
            reserve_base: 1,
            reserve_production: 0.35,
            send_buffer: 1,
            enemy_growth: true,
            enemy_value: 3.55,
            neutral_value: 1.0,
            production_weight: 6.5,
            ship_cost_weight: 0.66,
            time_cost_weight: 0.34,
            duplicate_penalty: 0.20,
            net_defense_reserve: true,
            defense_horizon: 42.0,
            reinforce_owned: true,
            defense_arrival_slack: 1.0,
            defense_score_weight: 9.0,
            comet_max_eta: Some(8.0),
            global_assignment: true,
            ..default_sniper_profile()
        },
    )
}

fn sniper_v14_actions(game: &Game, player: usize) -> PlayerAction {
    scored_sniper_actions(
        game,
        player,
        SniperProfile {
            reserve_base: 1,
            reserve_production: 0.35,
            send_buffer: 1,
            enemy_growth: true,
            enemy_value: 3.55,
            neutral_value: 1.0,
            production_weight: 6.5,
            ship_cost_weight: 0.66,
            time_cost_weight: 0.34,
            duplicate_penalty: 0.20,
            net_defense_reserve: true,
            defense_horizon: 42.0,
            reinforce_owned: true,
            defense_arrival_slack: 1.0,
            defense_score_weight: 9.0,
            chronological_forecast: true,
            comet_max_eta: Some(8.0),
            counter_recapture: true,
            recapture_min_gap: 0.5,
            recapture_max_gap: 8.0,
            recapture_score_weight: 6.0,
            recapture_gap_cost: 0.25,
            ..default_sniper_profile()
        },
    )
}

fn sniper_v15_actions(game: &Game, player: usize) -> PlayerAction {
    scored_sniper_actions(
        game,
        player,
        SniperProfile {
            reserve_base: 1,
            reserve_production: 0.35,
            send_buffer: 1,
            enemy_growth: true,
            enemy_value: 3.55,
            neutral_value: 1.0,
            production_weight: 6.5,
            ship_cost_weight: 0.66,
            time_cost_weight: 0.34,
            duplicate_penalty: 0.20,
            net_defense_reserve: true,
            defense_horizon: 42.0,
            reinforce_owned: true,
            defense_arrival_slack: 1.0,
            defense_score_weight: 9.0,
            comet_max_eta: Some(8.0),
            counter_recapture: true,
            recapture_min_gap: 0.5,
            recapture_max_gap: 8.0,
            recapture_score_weight: 6.0,
            recapture_gap_cost: 0.25,
            strict_defense: true,
            ..default_sniper_profile()
        },
    )
}

fn sniper_v16_actions(game: &Game, player: usize) -> PlayerAction {
    scored_sniper_actions(
        game,
        player,
        SniperProfile {
            reserve_base: 1,
            reserve_production: 0.35,
            send_buffer: 1,
            enemy_growth: true,
            enemy_value: 3.55,
            neutral_value: 1.0,
            production_weight: 6.5,
            ship_cost_weight: 0.66,
            time_cost_weight: 0.34,
            duplicate_penalty: 0.20,
            net_defense_reserve: true,
            defense_horizon: 42.0,
            reinforce_owned: true,
            defense_arrival_slack: 1.0,
            defense_score_weight: 9.0,
            comet_max_eta: Some(8.0),
            counter_recapture: true,
            recapture_min_gap: 0.5,
            recapture_max_gap: 8.0,
            recapture_score_weight: 6.0,
            recapture_gap_cost: 0.25,
            shadow_capture: true,
            ..default_sniper_profile()
        },
    )
}

fn sniper_v17_actions(game: &Game, player: usize) -> PlayerAction {
    scored_sniper_actions(
        game,
        player,
        SniperProfile {
            reserve_base: 0,
            reserve_production: 0.35,
            send_buffer: 1,
            enemy_growth: true,
            enemy_value: 3.55,
            neutral_value: 1.45,
            production_weight: 4.7243564847164325,
            ship_cost_weight: 0.66,
            time_cost_weight: 0.34358831285363617,
            duplicate_penalty: 0.20,
            net_defense_reserve: true,
            defense_horizon: 41.30069766848027,
            reinforce_owned: true,
            defense_arrival_slack: 1.0,
            defense_score_weight: 13.0,
            chronological_forecast: true,
            comet_max_eta: Some(5.067770406031305),
            counter_recapture: true,
            recapture_min_gap: 0.5,
            recapture_max_gap: 14.0,
            recapture_score_weight: 8.52469036246334,
            recapture_gap_cost: 0.24472159101961058,
            speed_bid: false,
            speed_bid_max_factor: 1.2770669321193258,
            speed_bid_tempo_weight: 0.47561154430408675,
            ..default_sniper_profile()
        },
    )
}

fn scored_sniper_actions(game: &Game, player: usize, profile: SniperProfile) -> PlayerAction {
    let player = player as i32;
    let mut targets = game
        .planets
        .iter()
        .enumerate()
        .filter_map(|(idx, planet)| (planet.owner >= 0 && planet.owner != player).then_some(idx))
        .collect::<Vec<_>>();
    targets.extend(
        game.planets
            .iter()
            .enumerate()
            .filter_map(|(idx, planet)| (planet.owner == -1).then_some(idx)),
    );
    let comet_ids = comet_id_set(game);
    let mut is_comet_by_idx = Vec::with_capacity(game.planets.len());
    let mut blockers = Vec::with_capacity(game.planets.len());
    let mut static_cols = Vec::new();
    let mut moving_cols = Vec::new();
    for (idx, planet) in game.planets.iter().enumerate() {
        let is_comet = comet_ids
            .as_ref()
            .is_some_and(|ids| ids.contains(&planet.id));
        let motion = target_motion(planet, game.angular_velocity, is_comet);
        is_comet_by_idx.push(is_comet);
        if motion.is_orbiting {
            moving_cols.push(idx);
        } else {
            static_cols.push(idx);
        }
        blockers.push(Some(motion));
    }

    let mut sources = game
        .planets
        .iter()
        .enumerate()
        .filter_map(|(idx, planet)| (planet.owner == player).then_some(idx))
        .collect::<Vec<_>>();
    if profile.aggressive_sources {
        sources.sort_by(|&a, &b| {
            let lhs = &game.planets[a];
            let rhs = &game.planets[b];
            rhs.ships
                .cmp(&lhs.ships)
                .then_with(|| rhs.production.cmp(&lhs.production))
        });
    }
    if targets.is_empty() && sources.is_empty() {
        return Vec::new();
    }

    let pressure = fleet_pressure(game, &blockers);
    let mut planned_by_target: HashMap<usize, Vec<(f64, i32)>> = HashMap::new();
    let mut moves = Vec::new();
    if profile.global_assignment {
        let mut used_sources = vec![false; game.planets.len()];
        for _ in 0..sources.len() {
            let mut best: Option<(usize, ScoredAction)> = None;
            for &source_idx in &sources {
                if used_sources.get(source_idx).copied().unwrap_or(true) {
                    continue;
                }
                let Some(candidate) = best_scored_source_action(
                    game,
                    player,
                    source_idx,
                    &sources,
                    &targets,
                    &planned_by_target,
                    &pressure,
                    profile,
                    &blockers,
                    &is_comet_by_idx,
                    &static_cols,
                    &moving_cols,
                ) else {
                    continue;
                };
                if best
                    .as_ref()
                    .is_none_or(|(_, current)| candidate.score > current.score)
                {
                    best = Some((source_idx, candidate));
                }
            }
            let Some((source_idx, choice)) = best else {
                break;
            };
            used_sources[source_idx] = true;
            planned_by_target
                .entry(choice.target_idx)
                .or_default()
                .push((choice.eta, choice.action.ships));
            moves.push(choice.action);
        }
        return moves;
    }

    for &source_idx in &sources {
        if let Some(choice) = best_scored_source_action(
            game,
            player,
            source_idx,
            &sources,
            &targets,
            &planned_by_target,
            &pressure,
            profile,
            &blockers,
            &is_comet_by_idx,
            &static_cols,
            &moving_cols,
        ) {
            planned_by_target
                .entry(choice.target_idx)
                .or_default()
                .push((choice.eta, choice.action.ships));
            moves.push(choice.action);
        }
    }
    moves
}

#[allow(clippy::too_many_arguments)]
fn best_scored_source_action(
    game: &Game,
    player: i32,
    source_idx: usize,
    sources: &[usize],
    targets: &[usize],
    planned_by_target: &HashMap<usize, Vec<(f64, i32)>>,
    pressure: &[Vec<PressureEntry>],
    profile: SniperProfile,
    blockers: &[Option<TargetMotion>],
    is_comet_by_idx: &[bool],
    static_cols: &[usize],
    moving_cols: &[usize],
) -> Option<ScoredAction> {
    let source = &game.planets[source_idx];
    let mut reserve = (profile.reserve_base as f64
        + profile.reserve_production * source.production as f64)
        .ceil() as i32;
    reserve += if profile.net_defense_reserve {
        defensive_reserve(
            pressure,
            source_idx,
            source.owner,
            source.production,
            profile.defense_horizon,
        )
    } else {
        enemy_pressure_by(pressure, source_idx, source.owner, profile.defense_horizon)
    };
    let budget = source.ships - reserve;
    if budget <= 1 {
        return None;
    }

    let mut best: Option<ScoredAction> = None;
    if profile.reinforce_owned {
        for &target_idx in sources {
            if target_idx == source_idx {
                continue;
            }
            let planned = planned_by_target
                .get(&target_idx)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if let Some(candidate) = defense_sniper_candidate(
                game,
                source,
                target_idx,
                budget,
                planned,
                pressure,
                profile,
                blockers,
                static_cols,
                moving_cols,
            ) {
                if best
                    .as_ref()
                    .is_none_or(|current| candidate.score > current.score)
                {
                    best = Some(candidate);
                }
            }
            if profile.counter_recapture {
                if let Some(candidate) = recapture_sniper_candidate(
                    game,
                    source,
                    target_idx,
                    budget,
                    planned,
                    pressure,
                    profile,
                    blockers,
                    static_cols,
                    moving_cols,
                ) {
                    if best
                        .as_ref()
                        .is_none_or(|current| candidate.score > current.score)
                    {
                        best = Some(candidate);
                    }
                }
            }
        }
    }
    for &target_idx in targets {
        let planned = planned_by_target
            .get(&target_idx)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        if profile.shadow_capture {
            if let Some(candidate) = shadow_sniper_candidate(
                game,
                player,
                source,
                target_idx,
                budget,
                planned,
                pressure,
                profile,
                blockers,
                static_cols,
                moving_cols,
            ) {
                if best
                    .as_ref()
                    .is_none_or(|current| candidate.score > current.score)
                {
                    best = Some(candidate);
                }
            }
        }
        let Some(candidate) = scored_sniper_candidate(
            game,
            player,
            source,
            target_idx,
            budget,
            planned,
            pressure,
            profile,
            blockers,
            is_comet_by_idx,
            static_cols,
            moving_cols,
        ) else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|current| candidate.score > current.score)
        {
            best = Some(candidate);
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
fn defense_sniper_candidate(
    game: &Game,
    source: &Planet,
    target_idx: usize,
    budget: i32,
    planned: &[(f64, i32)],
    pressure: &[Vec<PressureEntry>],
    profile: SniperProfile,
    blockers: &[Option<TargetMotion>],
    static_cols: &[usize],
    moving_cols: &[usize],
) -> Option<ScoredAction> {
    let target = &game.planets[target_idx];
    let (threat_eta, mut ships_needed) = defense_need(
        target,
        target_idx,
        planned,
        pressure,
        profile.defense_horizon,
    )?;
    if profile.strict_defense && ships_needed > budget {
        return None;
    }
    ships_needed = ships_needed.min(budget);
    if ships_needed <= 0 {
        return None;
    }
    let speed = fleet_speed_local(ships_needed, game.ship_speed);
    let target_motion = blockers
        .get(target_idx)
        .and_then(|motion| motion.as_ref())?;
    let solution = lead_solution_cached_with_speed(source, target_motion, speed)?;
    if solution.time > threat_eta + profile.defense_arrival_slack {
        return None;
    }
    if !route_clear_to_solution_with_cols(
        source.id,
        target.id,
        source.x,
        source.y,
        source.radius,
        &solution,
        speed,
        blockers,
        static_cols,
        moving_cols,
    ) {
        return None;
    }
    let urgency = 1.0 + (profile.defense_horizon - threat_eta).max(0.0) / profile.defense_horizon;
    let value = profile.defense_score_weight * urgency * (2.0 + target.production as f64);
    let cost = ships_needed as f64 + 0.35 * solution.time.max(1.0);
    Some(ScoredAction {
        target_idx,
        eta: solution.time,
        score: value / cost.max(1.0),
        action: Action {
            from_planet_id: source.id,
            angle: solution.angle,
            ships: ships_needed,
            target_id: target.id,
            eta: solution.time,
            target_x: solution.x,
            target_y: solution.y,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn recapture_sniper_candidate(
    game: &Game,
    source: &Planet,
    target_idx: usize,
    budget: i32,
    planned: &[(f64, i32)],
    pressure: &[Vec<PressureEntry>],
    profile: SniperProfile,
    blockers: &[Option<TargetMotion>],
    static_cols: &[usize],
    moving_cols: &[usize],
) -> Option<ScoredAction> {
    let target = &game.planets[target_idx];
    let (capture_eta, captor, surplus) = project_hostile_capture(
        target,
        target_idx,
        planned,
        pressure,
        profile.defense_horizon,
    )?;
    let mut ships_needed = surplus + profile.send_buffer;
    let mut solution: Option<LeadSolution> = None;
    let mut converged = false;
    for _ in 0..4 {
        if ships_needed > budget {
            return None;
        }
        let speed = fleet_speed_local(ships_needed, game.ship_speed);
        let target_motion = blockers
            .get(target_idx)
            .and_then(|motion| motion.as_ref())?;
        let candidate_solution = lead_solution_cached_with_speed(source, target_motion, speed)?;
        let gap = candidate_solution.time - capture_eta;
        if gap < profile.recapture_min_gap || gap > profile.recapture_max_gap {
            return None;
        }
        let later_enemy = pressure_between(
            pressure,
            target_idx,
            captor,
            capture_eta,
            candidate_solution.time,
        );
        let planned_recapture = planned_between(planned, capture_eta, candidate_solution.time);
        let revised = (surplus
            + (gap.max(0.0) * target.production as f64).floor() as i32
            + later_enemy
            + profile.send_buffer
            - planned_recapture)
            .max(1);
        solution = Some(candidate_solution);
        if revised == ships_needed {
            converged = true;
            break;
        }
        ships_needed = revised;
    }
    if !converged {
        return None;
    }
    let solution = solution?;
    if ships_needed > budget {
        return None;
    }
    let speed = fleet_speed_local(ships_needed, game.ship_speed);
    if !route_clear_to_solution_with_cols(
        source.id,
        target.id,
        source.x,
        source.y,
        source.radius,
        &solution,
        speed,
        blockers,
        static_cols,
        moving_cols,
    ) {
        return None;
    }
    let gap = solution.time - capture_eta;
    let urgency = 1.0 + (profile.defense_horizon - capture_eta).max(0.0) / profile.defense_horizon;
    let gap_penalty = 1.0 / (1.0 + profile.recapture_gap_cost * gap);
    let value =
        profile.recapture_score_weight * urgency * (2.0 + target.production as f64) * gap_penalty;
    let cost = ships_needed as f64 + 0.35 * solution.time.max(1.0);
    Some(ScoredAction {
        target_idx,
        eta: solution.time,
        score: value / cost.max(1.0),
        action: Action {
            from_planet_id: source.id,
            angle: solution.angle,
            ships: ships_needed,
            target_id: target.id,
            eta: solution.time,
            target_x: solution.x,
            target_y: solution.y,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn shadow_sniper_candidate(
    game: &Game,
    player: i32,
    source: &Planet,
    target_idx: usize,
    budget: i32,
    planned: &[(f64, i32)],
    pressure: &[Vec<PressureEntry>],
    profile: SniperProfile,
    blockers: &[Option<TargetMotion>],
    static_cols: &[usize],
    moving_cols: &[usize],
) -> Option<ScoredAction> {
    let target = &game.planets[target_idx];
    if target.owner == player {
        return None;
    }
    let (capture_eta, captor, surplus) = project_hostile_capture(
        target,
        target_idx,
        planned,
        pressure,
        profile.defense_horizon,
    )?;
    if captor == player {
        return None;
    }
    if pressure_by(pressure, target_idx, player, capture_eta) + planned_by(planned, capture_eta) > 0
    {
        return None;
    }
    let mut ships_needed = surplus + profile.send_buffer;
    let mut solution: Option<LeadSolution> = None;
    let mut converged = false;
    for _ in 0..4 {
        if ships_needed > budget {
            return None;
        }
        let speed = fleet_speed_local(ships_needed, game.ship_speed);
        let target_motion = blockers
            .get(target_idx)
            .and_then(|motion| motion.as_ref())?;
        let candidate_solution = lead_solution_cached_with_speed(source, target_motion, speed)?;
        let gap = candidate_solution.time - capture_eta;
        if gap < profile.recapture_min_gap || gap > profile.recapture_max_gap {
            return None;
        }
        let later_enemy = pressure_between(
            pressure,
            target_idx,
            captor,
            capture_eta,
            candidate_solution.time,
        );
        let planned_recapture = planned_between(planned, capture_eta, candidate_solution.time);
        let revised = (surplus
            + (gap.max(0.0) * target.production as f64).floor() as i32
            + later_enemy
            + profile.send_buffer
            - planned_recapture)
            .max(1);
        solution = Some(candidate_solution);
        if revised == ships_needed {
            converged = true;
            break;
        }
        ships_needed = revised;
    }
    if !converged {
        return None;
    }
    let solution = solution?;
    if ships_needed > budget {
        return None;
    }
    let speed = fleet_speed_local(ships_needed, game.ship_speed);
    if !route_clear_to_solution_with_cols(
        source.id,
        target.id,
        source.x,
        source.y,
        source.radius,
        &solution,
        speed,
        blockers,
        static_cols,
        moving_cols,
    ) {
        return None;
    }
    let gap = solution.time - capture_eta;
    let urgency = 1.0 + (profile.defense_horizon - capture_eta).max(0.0) / profile.defense_horizon;
    let gap_penalty = 1.0 / (1.0 + profile.recapture_gap_cost * gap);
    let owner_scale = if target.owner == -1 { 1.1 } else { 0.85 };
    let value = owner_scale
        * profile.recapture_score_weight
        * urgency
        * (2.0 + target.production as f64)
        * gap_penalty;
    let cost = ships_needed as f64 + 0.35 * solution.time.max(1.0);
    Some(ScoredAction {
        target_idx,
        eta: solution.time,
        score: value / cost.max(1.0),
        action: Action {
            from_planet_id: source.id,
            angle: solution.angle,
            ships: ships_needed,
            target_id: target.id,
            eta: solution.time,
            target_x: solution.x,
            target_y: solution.y,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn scored_sniper_candidate(
    game: &Game,
    player: i32,
    source: &Planet,
    target_idx: usize,
    budget: i32,
    planned: &[(f64, i32)],
    pressure: &[Vec<PressureEntry>],
    profile: SniperProfile,
    blockers: &[Option<TargetMotion>],
    is_comet_by_idx: &[bool],
    static_cols: &[usize],
    moving_cols: &[usize],
) -> Option<ScoredAction> {
    let target = &game.planets[target_idx];
    let mut ships_needed = target.ships + profile.send_buffer - planned_by(planned, 0.0);
    if ships_needed <= 0 {
        return None;
    }
    let mut partial_required = ships_needed;
    let mut partial = false;
    let target_motion = blockers
        .get(target_idx)
        .and_then(|motion| motion.as_ref())?;
    let mut last_solution: Option<(i32, f64, LeadSolution)> = None;
    for _ in 0..4 {
        if ships_needed > budget {
            if !partial_allowed(ships_needed, budget, profile) {
                return None;
            }
            partial_required = ships_needed;
            ships_needed = budget;
            partial = true;
        }
        let speed = fleet_speed_local(ships_needed, game.ship_speed);
        let solution = lead_solution_cached_with_speed(source, target_motion, speed)?;
        last_solution = Some((ships_needed, speed, solution));
        if profile.comet_max_eta.is_some_and(|max_eta| {
            is_comet_by_idx.get(target_idx).copied().unwrap_or(false) && solution.time > max_eta
        }) {
            return None;
        }
        let revised = ships_needed_at_eta(
            player,
            target_idx,
            target,
            solution.time,
            planned,
            pressure,
            profile,
        );
        if revised <= 0 {
            return None;
        }
        if partial && revised >= ships_needed {
            partial_required = revised;
            break;
        }
        if revised == ships_needed {
            break;
        }
        ships_needed = revised;
    }
    if ships_needed > budget {
        return None;
    }

    let (speed, solution) = if let Some((last_ships, last_speed, last_solution)) = last_solution
        && last_ships == ships_needed
    {
        (last_speed, last_solution)
    } else {
        let speed = fleet_speed_local(ships_needed, game.ship_speed);
        let solution = lead_solution_cached_with_speed(source, target_motion, speed)?;
        (speed, solution)
    };
    if !route_clear_to_solution_with_cols(
        source.id,
        target.id,
        source.x,
        source.y,
        source.radius,
        &solution,
        speed,
        blockers,
        static_cols,
        moving_cols,
    ) {
        return None;
    }

    let owner_value = if target.owner == -1 {
        profile.neutral_value
    } else {
        profile.enemy_value
    };
    let production_value =
        owner_value * (1.0 + profile.production_weight * target.production as f64);
    let dist = ((source.x - target.x).powi(2) + (source.y - target.y).powi(2)).sqrt();
    let distance_bonus = 1.0 / (1.0 + 0.02 * dist);
    let already_planned = planned_by(planned, solution.time);
    let duplicate_scale = 1.0 / (1.0 + profile.duplicate_penalty * already_planned.max(0) as f64);
    let cost = profile.ship_cost_weight * ships_needed.max(1) as f64
        + profile.time_cost_weight * solution.time.max(1.0);
    let mut score = production_value * distance_bonus * duplicate_scale / cost.max(1.0);
    if partial {
        score *=
            profile.partial_score_scale * (ships_needed as f64 / partial_required.max(1) as f64);
    }
    let mut best_solution = solution;
    let mut best_ships = ships_needed;
    let mut best_score = score;
    if profile.speed_bid && !partial {
        for factor in [1.25, profile.speed_bid_max_factor] {
            let bid_ships =
                budget.min((ships_needed + 1).max((ships_needed as f64 * factor).ceil() as i32));
            if bid_ships <= ships_needed {
                continue;
            }
            let bid_speed = fleet_speed_local(bid_ships, game.ship_speed);
            let Some(bid_solution) =
                lead_solution_cached_with_speed(source, target_motion, bid_speed)
            else {
                continue;
            };
            if profile.comet_max_eta.is_some_and(|max_eta| {
                is_comet_by_idx.get(target_idx).copied().unwrap_or(false)
                    && bid_solution.time > max_eta
            }) {
                continue;
            }
            let revised = ships_needed_at_eta(
                player,
                target_idx,
                target,
                bid_solution.time,
                planned,
                pressure,
                profile,
            );
            if revised <= 0 || revised > bid_ships {
                continue;
            }
            if !route_clear_to_solution_with_cols(
                source.id,
                target.id,
                source.x,
                source.y,
                source.radius,
                &bid_solution,
                bid_speed,
                blockers,
                static_cols,
                moving_cols,
            ) {
                continue;
            }
            let bid_already_planned = planned_by(planned, bid_solution.time);
            let bid_duplicate_scale =
                1.0 / (1.0 + profile.duplicate_penalty * bid_already_planned.max(0) as f64);
            let bid_cost = profile.ship_cost_weight * bid_ships.max(1) as f64
                + profile.time_cost_weight * bid_solution.time.max(1.0);
            let saved = (solution.time - bid_solution.time).max(0.0) / solution.time.max(1.0);
            let bid_score = production_value * distance_bonus * bid_duplicate_scale
                / bid_cost.max(1.0)
                * (1.0 + profile.speed_bid_tempo_weight * saved);
            if bid_score > best_score {
                best_score = bid_score;
                best_solution = bid_solution;
                best_ships = bid_ships;
            }
        }
    }
    Some(ScoredAction {
        target_idx,
        eta: best_solution.time,
        score: best_score,
        action: Action {
            from_planet_id: source.id,
            angle: best_solution.angle,
            ships: best_ships,
            target_id: target.id,
            eta: best_solution.time,
            target_x: best_solution.x,
            target_y: best_solution.y,
        },
    })
}

fn partial_allowed(ships_needed: i32, budget: i32, profile: SniperProfile) -> bool {
    profile.allow_partial
        && budget > 1
        && budget
            >= (ships_needed as f64 * profile.partial_min_fraction)
                .ceil()
                .max(2.0) as i32
}

fn ships_needed_at_eta(
    player: i32,
    target_idx: usize,
    target: &Planet,
    eta: f64,
    planned: &[(f64, i32)],
    pressure: &[Vec<PressureEntry>],
    profile: SniperProfile,
) -> i32 {
    if profile.chronological_forecast {
        let (owner, ships) =
            forecast_target_state(player, target, target_idx, eta, planned, pressure);
        if owner == player {
            return 0;
        }
        return (ships + profile.send_buffer).max(0);
    }

    let friendly = planned_by(planned, eta) + pressure_by(pressure, target_idx, player, eta);
    let contested_eta = eta + profile.contested_window;
    let hostile = if target.owner != -1 && target.owner != player {
        pressure_by(pressure, target_idx, target.owner, eta)
    } else {
        non_player_pressure_by(pressure, target_idx, player, eta)
    };
    let contested_hostile = if target.owner != -1 && target.owner != player {
        pressure_by(pressure, target_idx, target.owner, contested_eta)
    } else {
        non_player_pressure_by(pressure, target_idx, player, contested_eta)
    };
    let growth = if profile.enemy_growth && target.owner != player && target.owner != -1 {
        (target.production as f64 * eta.max(1.0)).ceil() as i32
    } else {
        0
    };
    let contested_buffer = if contested_hostile > friendly {
        profile.contested_extra_buffer
    } else {
        0
    };
    (target.ships + growth + hostile + profile.send_buffer + contested_buffer - friendly).max(0)
}

fn forecast_target_state(
    player: i32,
    target: &Planet,
    target_idx: usize,
    eta: f64,
    planned: &[(f64, i32)],
    pressure: &[Vec<PressureEntry>],
) -> (i32, i32) {
    let horizon = eta.ceil().max(0.0) as i32;
    let mut events: std::collections::BTreeMap<i32, HashMap<i32, i32>> =
        std::collections::BTreeMap::new();
    if let Some(entries) = pressure.get(target_idx) {
        for entry in entries {
            let turn = entry.eta.ceil() as i32;
            if (0..=horizon).contains(&turn) {
                *events
                    .entry(turn)
                    .or_default()
                    .entry(entry.owner)
                    .or_default() += entry.ships;
            }
        }
    }
    for (arrival, ships) in planned {
        let turn = arrival.ceil() as i32;
        if (0..=horizon).contains(&turn) {
            *events.entry(turn).or_default().entry(player).or_default() += *ships;
        }
    }

    let mut owner = target.owner;
    let mut garrison = target.ships;
    let mut prev_turn = 0;
    for (turn, arrivals) in events {
        if owner != -1 {
            garrison += (turn - prev_turn).max(0) * target.production;
        }
        let (survivor_owner, survivor_ships) = resolve_arrivals(&arrivals);
        if survivor_ships > 0 {
            if survivor_owner == owner {
                garrison += survivor_ships;
            } else {
                garrison -= survivor_ships;
                if garrison < 0 {
                    owner = survivor_owner;
                    garrison = -garrison;
                }
            }
        }
        prev_turn = turn;
    }
    if owner != -1 {
        garrison += (horizon - prev_turn).max(0) * target.production;
    }
    (owner, garrison.max(0))
}

fn resolve_arrivals(arrivals: &HashMap<i32, i32>) -> (i32, i32) {
    let mut rows = arrivals
        .iter()
        .map(|(owner, ships)| (*owner, *ships))
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if rows.is_empty() {
        return (-1, 0);
    }
    if rows.len() == 1 {
        return rows[0];
    }
    let diff = rows[0].1 - rows[1].1;
    if diff <= 0 {
        (-1, 0)
    } else {
        (rows[0].0, diff)
    }
}

fn planned_by(planned: &[(f64, i32)], eta: f64) -> i32 {
    planned
        .iter()
        .filter_map(|(arrival, ships)| (*arrival <= eta + 1.0).then_some(*ships))
        .sum()
}

fn pressure_by(pressure: &[Vec<PressureEntry>], target_idx: usize, owner: i32, eta: f64) -> i32 {
    pressure
        .get(target_idx)
        .into_iter()
        .flatten()
        .filter_map(|entry| (entry.owner == owner && entry.eta <= eta + 1.0).then_some(entry.ships))
        .sum()
}

fn non_player_pressure_by(
    pressure: &[Vec<PressureEntry>],
    target_idx: usize,
    player: i32,
    eta: f64,
) -> i32 {
    pressure
        .get(target_idx)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            (entry.owner != player && entry.eta <= eta + 1.0).then_some(entry.ships)
        })
        .sum()
}

fn enemy_pressure_by(
    pressure: &[Vec<PressureEntry>],
    target_idx: usize,
    owner: i32,
    eta: f64,
) -> i32 {
    pressure
        .get(target_idx)
        .into_iter()
        .flatten()
        .filter_map(|entry| (entry.owner != owner && entry.eta <= eta + 1.0).then_some(entry.ships))
        .sum()
}

fn defensive_reserve(
    pressure: &[Vec<PressureEntry>],
    target_idx: usize,
    owner: i32,
    production: i32,
    eta: f64,
) -> i32 {
    let mut needed = 0;
    if let Some(entries) = pressure.get(target_idx) {
        for entry in entries {
            if entry.owner == owner || entry.eta > eta {
                continue;
            }
            let hostile = enemy_pressure_by(pressure, target_idx, owner, entry.eta);
            let friendly = pressure_by(pressure, target_idx, owner, entry.eta);
            let produced = (entry.eta.max(0.0) * production as f64).floor() as i32;
            needed = needed.max(hostile - friendly - produced + 1);
        }
    }
    needed.max(0)
}

fn defense_need(
    target: &Planet,
    target_idx: usize,
    planned: &[(f64, i32)],
    pressure: &[Vec<PressureEntry>],
    horizon: f64,
) -> Option<(f64, i32)> {
    let mut best: Option<(f64, i32)> = None;
    let entries = pressure.get(target_idx)?;
    for entry in entries {
        if entry.owner == target.owner || entry.eta > horizon {
            continue;
        }
        let hostile = enemy_pressure_by(pressure, target_idx, target.owner, entry.eta);
        let friendly = pressure_by(pressure, target_idx, target.owner, entry.eta);
        let planned_friendly = planned_by(planned, entry.eta);
        let produced = (entry.eta.max(0.0) * target.production as f64).floor() as i32;
        let deficit = hostile + 1 - target.ships - produced - friendly - planned_friendly;
        if deficit > 0 && best.is_none_or(|(arrival, _)| entry.eta < arrival) {
            best = Some((entry.eta, deficit));
        }
    }
    best
}

fn project_hostile_capture(
    target: &Planet,
    target_idx: usize,
    planned: &[(f64, i32)],
    pressure: &[Vec<PressureEntry>],
    horizon: f64,
) -> Option<(f64, i32, i32)> {
    let entries = pressure.get(target_idx)?;
    for entry in entries {
        if entry.owner == target.owner || entry.eta > horizon {
            continue;
        }
        let hostile = enemy_pressure_by(pressure, target_idx, target.owner, entry.eta);
        let friendly = pressure_by(pressure, target_idx, target.owner, entry.eta);
        let planned_friendly = planned_by(planned, entry.eta);
        let produced = (entry.eta.max(0.0) * target.production as f64).floor() as i32;
        let surplus = hostile - target.ships - produced - friendly - planned_friendly;
        if surplus > 0 {
            return Some((entry.eta, entry.owner, surplus));
        }
    }
    None
}

fn pressure_between(
    pressure: &[Vec<PressureEntry>],
    target_idx: usize,
    owner: i32,
    start: f64,
    end: f64,
) -> i32 {
    pressure
        .get(target_idx)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            (entry.owner == owner && start < entry.eta && entry.eta <= end + 1.0)
                .then_some(entry.ships)
        })
        .sum()
}

fn planned_between(planned: &[(f64, i32)], start: f64, end: f64) -> i32 {
    planned
        .iter()
        .filter_map(|(arrival, ships)| {
            (start < *arrival && *arrival <= end + 1.0).then_some(*ships)
        })
        .sum()
}

fn fleet_pressure(game: &Game, blockers: &[Option<TargetMotion>]) -> Vec<Vec<PressureEntry>> {
    let mut pressure = vec![Vec::new(); game.planets.len()];
    for fleet in &game.fleets {
        let Some((target_idx, eta)) = inferred_fleet_target(game, fleet, blockers) else {
            continue;
        };
        pressure[target_idx].push(PressureEntry {
            eta,
            owner: fleet.owner,
            ships: fleet.ships,
        });
    }
    for entries in &mut pressure {
        entries.sort_by(|a, b| {
            a.eta
                .partial_cmp(&b.eta)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    pressure
}

fn inferred_fleet_target(
    game: &Game,
    fleet: &owars_env::Fleet,
    blockers: &[Option<TargetMotion>],
) -> Option<(usize, f64)> {
    let speed = fleet_speed_local(fleet.ships, game.ship_speed);
    let dir_x = fleet.angle.cos();
    let dir_y = fleet.angle.sin();
    let mut best: Option<(f64, usize)> = None;
    for (idx, planet) in game.planets.iter().enumerate() {
        let Some(motion) = blockers.get(idx).and_then(|motion| motion.as_ref()) else {
            continue;
        };
        let max_turns = bounded_lead_scan_turns(speed, planet.radius).max(1) as usize;
        for turn in 1..=max_turns {
            let old = (
                fleet.x + dir_x * speed * (turn - 1) as f64,
                fleet.y + dir_y * speed * (turn - 1) as f64,
            );
            let new = (
                fleet.x + dir_x * speed * turn as f64,
                fleet.y + dir_y * speed * turn as f64,
            );
            let pos = motion_position_at(&motion, turn - 1);
            if point_to_segment_distance_sq_local(pos, old, new) >= (planet.radius + 0.05).powi(2) {
                continue;
            }
            let eta = turn as f64;
            if best.is_none_or(|(current_eta, _)| eta < current_eta) {
                best = Some((eta, idx));
            }
            break;
        }
    }
    best.map(|(eta, idx)| (idx, eta))
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

fn get_seq_or<T>(row: &Bound<'_, PySequence>, idx: usize, default: T) -> PyResult<T>
where
    T: for<'py> FromPyObject<'py, 'py, Error = PyErr> + Clone,
{
    if idx >= row.len()? {
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
