use std::collections::HashMap;

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyReadonlyArray2, PyUntypedArrayMethods};
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

struct TargetMotion {
    x: f64,
    y: f64,
    radius: f64,
    is_orbiting: bool,
    positions: Vec<(f64, f64)>,
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
    num_envs: usize,
    num_players: usize,
    episode_steps: i32,
    ship_speed: f64,
    random_seed: u32,
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
            num_envs,
            num_players,
            episode_steps,
            ship_speed,
            random_seed,
        };
        env.reset_core();
        env
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.num_envs
    }

    fn reset(&mut self) {
        self.reset_core();
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

    fn policy_batch<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<(usize, usize)>,
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
            let game = &self.games[env_idx];
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
}

impl RustCoreVecEnv {
    fn reset_core(&mut self) {
        self.games = (0..self.num_envs)
            .map(|idx| {
                Game::new(
                    GameConfig::new(self.num_players, self.episode_steps, self.ship_speed),
                    self.random_seed + idx as u32,
                )
            })
            .collect();
    }
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
    let target_motions = planets_by_col
        .iter()
        .map(|planet| {
            planet.map(|target| target_motion(&target, game.angular_velocity, game.ship_speed))
        })
        .collect::<Vec<_>>();
    let is_comet_col = (0..planets_len)
        .map(|col| {
            let target_id = id_at(col);
            target_id >= 0 && is_comet_planet(game, target_id)
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
        for j in 0..planets_len {
            if i == j || !mask_at(j) {
                continue;
            }
            if is_comet_col[j] {
                continue;
            }
            let Some(target) = target_motions[j].as_ref() else {
                continue;
            };
            let Some(solution) = lead_solution_cached(&source, target, send, game.ship_speed)
            else {
                continue;
            };
            if !safe_flight_segment(
                source.x,
                source.y,
                source.radius,
                solution.angle,
                solution.x,
                solution.y,
            ) {
                continue;
            }
            out[i * planets_len + j] = true;
        }
    }
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

fn is_comet_planet(game: &Game, planet_id: i32) -> bool {
    game.comets
        .iter()
        .any(|group| group.planet_ids.iter().any(|id| *id == planet_id))
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
    let comet_ids = game
        .comets
        .iter()
        .flat_map(|group| group.planet_ids.iter().copied())
        .collect::<std::collections::HashSet<_>>();
    let mut remaining = game
        .planets
        .iter()
        .map(|p| (p.id, p.ships))
        .collect::<HashMap<_, _>>();
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
        if target_id < 0 || comet_ids.contains(&target_id) {
            continue;
        }
        let source_id = id_at(i);
        let (Some(source), Some(target)) = (
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
        let Some(solution) = lead_solution(
            &source,
            &target,
            game.angular_velocity,
            send,
            game.ship_speed,
        ) else {
            continue;
        };
        if !safe_flight_segment(
            source.x,
            source.y,
            source.radius,
            solution.angle,
            solution.x,
            solution.y,
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

fn target_motion(target: &Planet, angular_velocity: f64, max_speed: f64) -> TargetMotion {
    let orbit_radius = ((target.x - CENTER).powi(2) + (target.y - CENTER).powi(2)).sqrt();
    let is_orbiting = orbit_radius + target.radius < ROTATION_RADIUS_LIMIT
        && angular_velocity.abs() > 1e-12
        && orbit_radius > 1e-9;
    if !is_orbiting {
        return TargetMotion {
            x: target.x,
            y: target.y,
            radius: target.radius,
            is_orbiting: false,
            positions: Vec::new(),
        };
    }
    let theta0 = (target.y - CENTER).atan2(target.x - CENTER);
    let speed_floor = fleet_speed_local(1, max_speed).max(1e-9);
    let max_turns = bounded_lead_scan_turns(speed_floor, target.radius);
    let positions = (1..=max_turns)
        .map(|k| {
            let theta = theta0 + angular_velocity * (k - 1) as f64;
            (
                CENTER + orbit_radius * theta.cos(),
                CENTER + orbit_radius * theta.sin(),
            )
        })
        .collect();
    TargetMotion {
        x: target.x,
        y: target.y,
        radius: target.radius,
        is_orbiting: true,
        positions,
    }
}

fn lead_solution_cached(
    source: &Planet,
    target: &TargetMotion,
    send: i32,
    max_speed: f64,
) -> Option<LeadSolution> {
    let mut solution =
        lead_solution_from_point_cached(source.x, source.y, target, send, max_speed)?;
    let mut angle = solution.angle;
    let offset = (source.radius + 0.1).max(0.0);
    if offset <= 0.0 {
        return Some(solution);
    }
    for _ in 0..4 {
        let start_x = source.x + angle.cos() * offset;
        let start_y = source.y + angle.sin() * offset;
        let refined = lead_solution_from_point_cached(start_x, start_y, target, send, max_speed)?;
        if angle_delta(refined.angle, angle).abs() < 1e-6 {
            return Some(refined);
        }
        angle = refined.angle;
        solution = refined;
    }
    Some(solution)
}

fn lead_solution_from_point_cached(
    source_x: f64,
    source_y: f64,
    target: &TargetMotion,
    send: i32,
    max_speed: f64,
) -> Option<LeadSolution> {
    let speed = fleet_speed_local(send, max_speed);
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
    for (idx, &(tx, ty)) in target.positions.iter().take(max_turns).enumerate() {
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

fn safe_flight_segment(
    source_x: f64,
    source_y: f64,
    source_radius: f64,
    angle: f64,
    end_x: f64,
    end_y: f64,
) -> bool {
    let offset = (source_radius + 0.1).max(0.0);
    let start_x = source_x + angle.cos() * offset;
    let start_y = source_y + angle.sin() * offset;
    if !inside_board(start_x, start_y) || !inside_board(end_x, end_y) {
        return false;
    }
    point_to_segment_distance_local((CENTER, CENTER), (start_x, start_y), (end_x, end_y))
        >= SUN_RADIUS
}

fn inside_board(x: f64, y: f64) -> bool {
    (0.0..=BOARD_SIZE).contains(&x) && (0.0..=BOARD_SIZE).contains(&y)
}

fn point_to_segment_distance_local(point: (f64, f64), start: (f64, f64), end: (f64, f64)) -> f64 {
    let seg_x = end.0 - start.0;
    let seg_y = end.1 - start.1;
    let l2 = seg_x * seg_x + seg_y * seg_y;
    if l2 == 0.0 {
        return ((point.0 - start.0).powi(2) + (point.1 - start.1).powi(2)).sqrt();
    }
    let raw_t = ((point.0 - start.0) * seg_x + (point.1 - start.1) * seg_y) / l2;
    let t = raw_t.clamp(0.0, 1.0);
    let proj = (start.0 + t * seg_x, start.1 + t * seg_y);
    ((point.0 - proj.0).powi(2) + (point.1 - proj.1).powi(2)).sqrt()
}

fn angle_delta(a: f64, b: f64) -> f64 {
    (a - b).sin().atan2((a - b).cos())
}

#[pymodule]
fn _owars_env(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustCoreVecEnv>()?;
    m.add_class::<NativeActionList>()?;
    Ok(())
}
