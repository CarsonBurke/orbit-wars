use std::collections::HashMap;

use numpy::IntoPyArray;
use numpy::ndarray::{Array1, Array2, Array3};
use owars_env::{Action, Game, GameConfig, PlayerAction};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

const BOARD_SIZE: f64 = 100.0;
const CENTER: f64 = 50.0;
const ROTATION_RADIUS_LIMIT: f64 = 50.0;
const MAX_SHIP_SPEED: f64 = 6.0;
const MAX_OMEGA: f64 = 0.05;
const MAX_PLANETS: usize = 64;
const MAX_FLEETS: usize = 384;
const PLANET_FEAT_DIM: usize = 19;
const FLEET_FEAT_DIM: usize = 20;
const LOG_1000: f64 = 6.907_755_278_982_137;

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
        let out = PyDict::new(py);
        for (pos, env_idx) in indices.into_iter().enumerate() {
            let action_obj = actions.get_item(pos)?;
            let parsed = parse_env_actions(&action_obj, self.num_players)?;
            let game = &mut self.games[env_idx];
            let result = game.step(&parsed);
            let final_rewards = if result.done {
                Some(result.rewards)
            } else {
                None
            };
            out.set_item(env_idx, (py.None(), result.done, final_rewards))?;
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

#[pymodule]
fn _owars_env(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustCoreVecEnv>()?;
    Ok(())
}
