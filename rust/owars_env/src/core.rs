use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::f64::consts::PI;

const BOARD_SIZE: f64 = 100.0;
const CENTER: f64 = 50.0;
const SUN_RADIUS: f64 = 10.0;
const ROTATION_RADIUS_LIMIT: f64 = 50.0;
const COMET_RADIUS: f64 = 1.0;
const COMET_PRODUCTION: i32 = 1;
const PLANET_CLEARANCE: f64 = 7.0;
const MIN_PLANET_GROUPS: i32 = 5;
const MAX_PLANET_GROUPS: i32 = 10;
const MIN_STATIC_GROUPS: i32 = 3;
const COMET_SPAWN_STEPS: [i32; 5] = [50, 150, 250, 350, 450];
const LOG_1000: f64 = 6.907_755_278_982_137;

type MovingPlanet = (i32, f64, (f64, f64), (f64, f64));

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Planet {
    pub id: i32,
    pub owner: i32,
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    pub ships: i32,
    pub production: i32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fleet {
    pub id: i32,
    pub owner: i32,
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub from_planet_id: i32,
    pub ships: i32,
    pub target_id: i32,
    pub eta: f64,
    pub target_x: f64,
    pub target_y: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CometGroup {
    pub planet_ids: Vec<i32>,
    pub paths: Vec<Vec<Point>>,
    pub path_index: i32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Action {
    pub from_planet_id: i32,
    pub angle: f64,
    pub ships: i32,
    pub target_id: i32,
    pub eta: f64,
    pub target_x: f64,
    pub target_y: f64,
}

impl Action {
    pub fn launch(from_planet_id: i32, angle: f64, ships: i32) -> Self {
        Self {
            from_planet_id,
            angle,
            ships,
            target_id: -1,
            eta: 0.0,
            target_x: 0.0,
            target_y: 0.0,
        }
    }
}

pub type PlayerAction = Vec<Action>;

#[derive(Clone, Debug, PartialEq)]
pub struct StepResult {
    pub done: bool,
    pub rewards: Vec<i32>,
}

#[derive(Clone, Debug)]
pub struct GameConfig {
    pub num_players: usize,
    pub episode_steps: i32,
    pub ship_speed: f64,
}

impl GameConfig {
    pub fn new(num_players: usize, episode_steps: i32, ship_speed: f64) -> Self {
        Self {
            num_players,
            episode_steps,
            ship_speed,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub step: i32,
    pub angular_velocity: f64,
    pub planets: Vec<Planet>,
    pub initial_planets: Vec<Planet>,
    pub fleets: Vec<Fleet>,
    pub comets: Vec<CometGroup>,
    pub next_fleet_id: i32,
}

impl GameState {
    pub fn new(
        step: i32,
        angular_velocity: f64,
        planets: Vec<Planet>,
        initial_planets: Vec<Planet>,
        fleets: Vec<Fleet>,
        comets: Vec<CometGroup>,
        next_fleet_id: i32,
    ) -> Self {
        Self {
            step,
            angular_velocity,
            planets,
            initial_planets,
            fleets,
            comets,
            next_fleet_id,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Game {
    pub num_players: usize,
    pub episode_steps: i32,
    pub ship_speed: f64,
    pub step: i32,
    pub done: bool,
    pub angular_velocity: f64,
    pub planets: Vec<Planet>,
    pub initial_planets: Vec<Planet>,
    pub fleets: Vec<Fleet>,
    pub comets: Vec<CometGroup>,
    pub next_fleet_id: i32,
    initialized: bool,
    rng: PyRandom,
}

impl Game {
    pub fn new(config: GameConfig, seed: u32) -> Self {
        Self {
            num_players: config.num_players,
            episode_steps: config.episode_steps,
            ship_speed: config.ship_speed,
            step: 0,
            done: false,
            angular_velocity: 0.0,
            planets: Vec::new(),
            initial_planets: Vec::new(),
            fleets: Vec::new(),
            comets: Vec::new(),
            next_fleet_id: 0,
            initialized: false,
            rng: PyRandom::seed(seed),
        }
    }

    pub fn from_state(config: GameConfig, state: GameState) -> Self {
        Self {
            num_players: config.num_players,
            episode_steps: config.episode_steps,
            ship_speed: config.ship_speed,
            step: state.step,
            done: false,
            angular_velocity: state.angular_velocity,
            planets: state.planets,
            initial_planets: state.initial_planets,
            fleets: state.fleets,
            comets: state.comets,
            next_fleet_id: state.next_fleet_id,
            initialized: true,
            rng: PyRandom::seed(0),
        }
    }

    pub fn step(&mut self, actions: &[PlayerAction]) -> StepResult {
        if self.done {
            return StepResult {
                done: true,
                rewards: self.rewards(),
            };
        }
        self.step += 1;
        if !self.initialized {
            self.initialize();
            self.check_done();
            return StepResult {
                done: self.done,
                rewards: self.rewards(),
            };
        }
        self.remove_expired_comets_before_launch();
        self.spawn_comets();
        let planet_idx_by_id = self.planet_idx_by_id();
        self.process_moves(actions, &planet_idx_by_id);
        self.produce();
        let mut combat = self.move_fleets();
        self.move_planets_and_sweep(&mut combat);
        self.resolve_combat(&combat);
        self.check_done();
        StepResult {
            done: self.done,
            rewards: self.rewards(),
        }
    }

    fn planet_idx_by_id(&self) -> HashMap<i32, usize> {
        self.planets
            .iter()
            .enumerate()
            .map(|(idx, p)| (p.id, idx))
            .collect()
    }

    fn initial_by_id(&self) -> HashMap<i32, Planet> {
        self.initial_planets.iter().map(|p| (p.id, *p)).collect()
    }

    fn process_moves(&mut self, actions: &[PlayerAction], planet_idx_by_id: &HashMap<i32, usize>) {
        let mut launches = Vec::new();
        for player in 0..self.num_players {
            let Some(player_actions) = actions.get(player) else {
                continue;
            };
            for action in player_actions {
                if action.ships <= 0 {
                    continue;
                }
                let Some(&planet_idx) = planet_idx_by_id.get(&action.from_planet_id) else {
                    continue;
                };
                let planet = &mut self.planets[planet_idx];
                if planet.owner != player as i32 || planet.ships < action.ships {
                    continue;
                }
                planet.ships -= action.ships;
                let start_x = planet.x + action.angle.cos() * (planet.radius + 0.1);
                let start_y = planet.y + action.angle.sin() * (planet.radius + 0.1);
                launches.push(Fleet {
                    id: self.next_fleet_id,
                    owner: player as i32,
                    x: start_x,
                    y: start_y,
                    angle: action.angle,
                    from_planet_id: action.from_planet_id,
                    ships: action.ships,
                    target_id: action.target_id,
                    eta: action.eta,
                    target_x: action.target_x,
                    target_y: action.target_y,
                });
                self.next_fleet_id += 1;
            }
        }
        self.fleets.extend(launches);
    }

    fn produce(&mut self) {
        for planet in &mut self.planets {
            if planet.owner != -1 {
                planet.ships += planet.production;
            }
        }
    }

    fn move_fleets(&mut self) -> BTreeMap<i32, Vec<Fleet>> {
        let mut combat: BTreeMap<i32, Vec<Fleet>> = BTreeMap::new();
        if self.fleets.is_empty() {
            return combat;
        }

        let planets = self.planets.clone();
        let ship_speed = self.ship_speed;
        let mut keep = Vec::with_capacity(self.fleets.len());
        for mut fleet in self.fleets.drain(..) {
            let old = (fleet.x, fleet.y);
            let ships = fleet.ships.max(1) as f64;
            let normalized = (ships.ln() / LOG_1000).max(0.0);
            let speed = (1.0 + (ship_speed - 1.0) * normalized.powf(1.5)).min(ship_speed);
            fleet.x += fleet.angle.cos() * speed;
            fleet.y += fleet.angle.sin() * speed;
            if fleet.target_id >= 0 {
                fleet.eta = (fleet.eta - 1.0).max(0.0);
                if fleet.eta <= 0.0 {
                    fleet.target_id = -1;
                }
            }
            let new = (fleet.x, fleet.y);
            if fleet.x < 0.0 || fleet.x > BOARD_SIZE || fleet.y < 0.0 || fleet.y > BOARD_SIZE {
                continue;
            }
            if point_to_segment_distance((CENTER, CENTER), old, new) < SUN_RADIUS {
                continue;
            }

            let mut hit: Option<i32> = None;
            for planet in &planets {
                if point_to_segment_distance((planet.x, planet.y), old, new) < planet.radius {
                    hit = Some(planet.id);
                    break;
                }
            }
            if let Some(pid) = hit {
                combat.entry(pid).or_default().push(fleet);
            } else {
                keep.push(fleet);
            }
        }
        self.fleets = keep;
        combat
    }

    fn move_planets_and_sweep(&mut self, combat: &mut BTreeMap<i32, Vec<Fleet>>) {
        if self.planets.is_empty() {
            return;
        }
        let initial_by_id = self.initial_by_id();
        let mut moving = Vec::new();
        let comet_ids = self.comet_planet_ids();
        for planet in &mut self.planets {
            if comet_ids.contains(&planet.id) {
                continue;
            }
            let Some(initial) = initial_by_id.get(&planet.id) else {
                continue;
            };
            let dx = initial.x - CENTER;
            let dy = initial.y - CENTER;
            let orbital_radius = (dx * dx + dy * dy).sqrt();
            let old = (planet.x, planet.y);
            if orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT {
                let angle = dy.atan2(dx) + self.angular_velocity * (self.step - 1) as f64;
                planet.x = CENTER + orbital_radius * angle.cos();
                planet.y = CENTER + orbital_radius * angle.sin();
            }
            if old != (planet.x, planet.y) {
                moving.push((planet.id, planet.radius, old, (planet.x, planet.y)));
            }
        }
        self.move_comets(&mut moving);
        if moving.is_empty() || self.fleets.is_empty() {
            return;
        }
        let mut remove = vec![false; self.fleets.len()];
        for (fleet_idx, fleet) in self.fleets.iter().enumerate() {
            for (pid, radius, old, new) in &moving {
                if point_to_segment_distance((fleet.x, fleet.y), *old, *new) < *radius {
                    combat.entry(*pid).or_default().push(*fleet);
                    remove[fleet_idx] = true;
                    break;
                }
            }
        }
        let mut idx = 0;
        self.fleets.retain(|_| {
            let keep = !remove[idx];
            idx += 1;
            keep
        });
    }

    fn comet_planet_ids(&self) -> HashSet<i32> {
        self.comets
            .iter()
            .flat_map(|group| group.planet_ids.iter().copied())
            .collect()
    }

    fn remove_expired_comets_before_launch(&mut self) {
        let mut expired = Vec::new();
        for group in &self.comets {
            let path_idx = group.path_index.max(0) as usize;
            for (i, pid) in group.planet_ids.iter().enumerate() {
                if group
                    .paths
                    .get(i)
                    .is_some_and(|path| path_idx >= path.len())
                {
                    expired.push(*pid);
                }
            }
        }
        self.remove_comet_planets(&expired);
    }

    fn move_comets(&mut self, moving: &mut Vec<MovingPlanet>) {
        if self.comets.is_empty() {
            return;
        }
        let planet_idx_by_id = self.planet_idx_by_id();
        let mut expired = Vec::new();
        for group in &mut self.comets {
            group.path_index += 1;
            let path_idx = group.path_index.max(0) as usize;
            for (i, pid) in group.planet_ids.iter().copied().enumerate() {
                let Some(path) = group.paths.get(i) else {
                    expired.push(pid);
                    continue;
                };
                let Some(&planet_idx) = planet_idx_by_id.get(&pid) else {
                    continue;
                };
                if path_idx >= path.len() {
                    expired.push(pid);
                    continue;
                }
                let planet = &mut self.planets[planet_idx];
                let old = (planet.x, planet.y);
                planet.x = path[path_idx].x;
                planet.y = path[path_idx].y;
                if old.0 >= 0.0 && old != (planet.x, planet.y) {
                    moving.push((pid, planet.radius, old, (planet.x, planet.y)));
                }
            }
        }
        self.remove_comet_planets(&expired);
    }

    fn remove_comet_planets(&mut self, pids: &[i32]) {
        if pids.is_empty() {
            return;
        }
        let expired: HashSet<i32> = pids.iter().copied().collect();
        self.planets.retain(|planet| !expired.contains(&planet.id));
        self.initial_planets
            .retain(|planet| !expired.contains(&planet.id));
        for group in &mut self.comets {
            let mut new_ids = Vec::with_capacity(group.planet_ids.len());
            let mut new_paths = Vec::with_capacity(group.paths.len());
            for (idx, pid) in group.planet_ids.iter().copied().enumerate() {
                if expired.contains(&pid) {
                    continue;
                }
                new_ids.push(pid);
                if let Some(path) = group.paths.get(idx) {
                    new_paths.push(path.clone());
                }
            }
            group.planet_ids = new_ids;
            group.paths = new_paths;
        }
        self.comets.retain(|group| !group.planet_ids.is_empty());
    }

    fn resolve_combat(&mut self, combat: &BTreeMap<i32, Vec<Fleet>>) {
        if combat.is_empty() {
            return;
        }
        let planet_idx_by_id = self.planet_idx_by_id();
        for (pid, fleets) in combat {
            let Some(&planet_idx) = planet_idx_by_id.get(pid) else {
                continue;
            };
            let mut by_owner: HashMap<i32, i32> = HashMap::new();
            for fleet in fleets {
                *by_owner.entry(fleet.owner).or_default() += fleet.ships;
            }
            let mut rows: Vec<(i32, i32)> = by_owner.into_iter().collect();
            rows.sort_by(|a, b| {
                b.1.cmp(&a.1)
                    .then_with(|| a.0.cmp(&b.0))
                    .then(Ordering::Equal)
            });
            let (top_owner, top_ships) = rows[0];
            let (survivor_owner, survivor_ships) = if rows.len() > 1 {
                let diff = top_ships - rows[1].1;
                if diff > 0 { (top_owner, diff) } else { (-1, 0) }
            } else {
                (top_owner, top_ships)
            };
            if survivor_ships <= 0 {
                continue;
            }
            let planet = &mut self.planets[planet_idx];
            if planet.owner == survivor_owner {
                planet.ships += survivor_ships;
            } else {
                planet.ships -= survivor_ships;
                if planet.ships < 0 {
                    planet.owner = survivor_owner;
                    planet.ships = -planet.ships;
                }
            }
        }
    }

    fn check_done(&mut self) {
        let mut terminated = self.step > self.episode_steps - 2;
        let mut alive = HashSet::new();
        for planet in &self.planets {
            if planet.owner != -1 {
                alive.insert(planet.owner);
            }
        }
        for fleet in &self.fleets {
            alive.insert(fleet.owner);
        }
        if alive.len() <= 1 {
            terminated = true;
        }
        self.done = terminated;
    }

    fn initialize(&mut self) {
        self.angular_velocity = self.rng.uniform(0.025, 0.05);
        let mut planets = self.generate_planets();
        let initial_planets = planets.clone();
        let num_groups = planets.len() / 4;
        if num_groups > 0 {
            let mut base = self.rng.randint(0, num_groups as i32 - 1) as usize * 4;
            if self.num_players == 4 {
                let q1 = planets[base];
                let orb_r = distance((q1.x, q1.y), (CENTER, CENTER));
                if orb_r + q1.radius < ROTATION_RADIUS_LIMIT {
                    for group_id in 0..num_groups {
                        let gb = group_id * 4;
                        let gp = planets[gb];
                        let g_orb = distance((gp.x, gp.y), (CENTER, CENTER));
                        if g_orb + gp.radius < ROTATION_RADIUS_LIMIT
                            && ((gp.x - CENTER) - (gp.y - CENTER)).abs() < 0.01
                        {
                            base = gb;
                            break;
                        }
                    }
                }
            }
            if self.num_players == 2 {
                planets[base].owner = 0;
                planets[base].ships = 10;
                planets[base + 3].owner = 1;
                planets[base + 3].ships = 10;
            } else if self.num_players == 4 {
                for j in 0..4 {
                    planets[base + j].owner = j as i32;
                    planets[base + j].ships = 10;
                }
            }
        }
        self.planets = planets;
        self.initial_planets = initial_planets;
        self.fleets.clear();
        self.next_fleet_id = 0;
        self.comets.clear();
        self.initialized = true;
    }

    fn generate_planets(&mut self) -> Vec<Planet> {
        let mut planets = Vec::new();
        let num_q1 = self.rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS);
        let mut id_counter = 0;
        let mut static_groups = 0;
        for _ in 0..5000 {
            if static_groups >= MIN_STATIC_GROUPS {
                break;
            }
            let prod = self.rng.randint(1, 5);
            let radius = 1.0 + (prod as f64).ln();
            let angle = self.rng.uniform(0.0, PI / 2.0);
            let min_orbital = ROTATION_RADIUS_LIMIT - radius;
            let max_orbital = (BOARD_SIZE - CENTER - radius) / angle.cos().max(angle.sin());
            if min_orbital > max_orbital {
                continue;
            }
            let orbital_r = self.rng.uniform(min_orbital, max_orbital);
            let x = CENTER + orbital_r * angle.cos();
            let y = CENTER + orbital_r * angle.sin();
            if x + radius > BOARD_SIZE
                || x - radius < 0.0
                || y + radius > BOARD_SIZE
                || y - radius < 0.0
            {
                continue;
            }
            if (BOARD_SIZE - x) - radius < 0.0 || (BOARD_SIZE - y) - radius < 0.0 {
                continue;
            }
            if x - CENTER < radius + 5.0 || y - CENTER < radius + 5.0 {
                continue;
            }
            let ships = self.rng.randint(5, 99).min(self.rng.randint(5, 99));
            let temp = symmetric_group(id_counter, x, y, radius, ships, prod);
            if valid_against_existing(&temp, &planets) {
                planets.extend(temp);
                id_counter += 4;
                static_groups += 1;
            }
        }

        for _ in 0..1000 {
            let prod = self.rng.randint(1, 5);
            let radius = 1.0 + (prod as f64).ln();
            let min_orbital = SUN_RADIUS + radius + 10.0;
            let max_orbital = ROTATION_RADIUS_LIMIT - radius;
            if min_orbital >= max_orbital {
                continue;
            }
            let orbital_r = self.rng.uniform(min_orbital, max_orbital);
            let x = CENTER + orbital_r * (PI / 4.0).cos();
            let y = CENTER + orbital_r * (PI / 4.0).sin();
            let ships = self.rng.randint(5, 99).min(self.rng.randint(5, 99));
            let temp = symmetric_group(id_counter, x, y, radius, ships, prod);
            if valid_orbit_group(&temp, &planets, true) {
                planets.extend(temp);
                id_counter += 4;
                break;
            }
        }

        let mut attempts = 0;
        let mut has_orbiting = false;
        while planets.len() < (num_q1 * 4) as usize || (!has_orbiting && attempts < 5000) {
            attempts += 1;
            if attempts >= 5000 {
                break;
            }
            let prod = self.rng.randint(1, 5);
            let radius = 1.0 + (prod as f64).ln();
            let x = self.rng.uniform(CENTER + 15.0, BOARD_SIZE - radius - 5.0);
            let y = self.rng.uniform(CENTER + 15.0, BOARD_SIZE - radius - 5.0);
            let orbital_radius = distance((x, y), (CENTER, CENTER));
            if orbital_radius < SUN_RADIUS + radius + 10.0 {
                continue;
            }
            if orbital_radius + radius >= ROTATION_RADIUS_LIMIT
                && (x + radius > BOARD_SIZE
                    || x - radius < 0.0
                    || y + radius > BOARD_SIZE
                    || y - radius < 0.0)
            {
                continue;
            }
            let ships = self.rng.randint(5, 30);
            let temp = symmetric_group(id_counter, x, y, radius, ships, prod);
            if valid_orbit_group(&temp, &planets, false) {
                if orbital_radius + radius < ROTATION_RADIUS_LIMIT {
                    has_orbiting = true;
                }
                planets.extend(temp);
                id_counter += 4;
            }
        }
        planets
    }

    fn spawn_comets(&mut self) {
        if !COMET_SPAWN_STEPS.contains(&self.step) {
            return;
        }
        let Some(paths) = self.generate_comet_paths(self.step) else {
            return;
        };
        let next_id = self.planets.iter().map(|p| p.id).max().unwrap_or(-1) + 1;
        let comet_ships = self
            .rng
            .randint(1, 99)
            .min(self.rng.randint(1, 99))
            .min(self.rng.randint(1, 99))
            .min(self.rng.randint(1, 99));
        let mut planet_ids = Vec::new();
        for (i, _path) in paths.iter().enumerate() {
            let pid = next_id + i as i32;
            planet_ids.push(pid);
            let planet = Planet {
                id: pid,
                owner: -1,
                x: -99.0,
                y: -99.0,
                radius: COMET_RADIUS,
                ships: comet_ships,
                production: COMET_PRODUCTION,
            };
            self.planets.push(planet);
            self.initial_planets.push(planet);
        }
        self.comets.push(CometGroup {
            planet_ids,
            paths,
            path_index: -1,
        });
    }

    fn generate_comet_paths(&mut self, spawn_step: i32) -> Option<Vec<Vec<Point>>> {
        let comet_ids = self.comet_planet_ids();
        for _ in 0..300 {
            let e = self.rng.uniform(0.75, 0.93);
            let a = self.rng.uniform(60.0, 150.0);
            if a * (1.0 - e) < SUN_RADIUS + COMET_RADIUS {
                continue;
            }
            let b = a * (1.0 - e * e).sqrt();
            let c_val = a * e;
            let phi = self.rng.uniform(PI / 6.0, PI / 3.0);
            let cos_phi = phi.cos();
            let sin_phi = phi.sin();
            let mut dense = Vec::with_capacity(5000);
            for i in 0..5000 {
                let t = 0.3 * PI + (1.7 * PI - 0.3 * PI) * (i as f64) / 4999.0;
                let ex = c_val + a * t.cos();
                let ey = b * t.sin();
                dense.push(Point::new(
                    CENTER + ex * cos_phi - ey * sin_phi,
                    CENTER + ex * sin_phi + ey * cos_phi,
                ));
            }
            let mut cumulative = Vec::with_capacity(dense.len() - 1);
            let mut total = 0.0;
            for pair in dense.windows(2) {
                total += distance((pair[1].x, pair[1].y), (pair[0].x, pair[0].y));
                cumulative.push(total);
            }
            let mut path = vec![dense[0]];
            let mut target = 4.0;
            while target < total + 4.0 {
                let idx = cumulative.partition_point(|v| *v < target) + 1;
                if idx < dense.len() {
                    path.push(dense[idx]);
                }
                target += 4.0;
            }
            let on_board: Vec<usize> = path
                .iter()
                .enumerate()
                .filter_map(|(idx, p)| {
                    (p.x >= 0.0 && p.x <= BOARD_SIZE && p.y >= 0.0 && p.y <= BOARD_SIZE)
                        .then_some(idx)
                })
                .collect();
            if on_board.is_empty() {
                continue;
            }
            let visible = path[on_board[0]..=*on_board.last().unwrap()].to_vec();
            if !(5..=40).contains(&visible.len()) {
                continue;
            }
            let paths = vec![
                visible.clone(),
                visible
                    .iter()
                    .map(|p| Point::new(BOARD_SIZE - p.x, p.y))
                    .collect(),
                visible
                    .iter()
                    .map(|p| Point::new(p.x, BOARD_SIZE - p.y))
                    .collect(),
                visible
                    .iter()
                    .map(|p| Point::new(BOARD_SIZE - p.x, BOARD_SIZE - p.y))
                    .collect(),
            ];
            if self.comet_paths_valid(&visible, &paths, &comet_ids, spawn_step) {
                return Some(paths);
            }
        }
        None
    }

    fn comet_paths_valid(
        &self,
        visible: &[Point],
        paths: &[Vec<Point>],
        comet_ids: &HashSet<i32>,
        spawn_step: i32,
    ) -> bool {
        if visible
            .iter()
            .any(|p| distance((p.x, p.y), (CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS)
        {
            return false;
        }
        let planets: Vec<Planet> = self
            .initial_planets
            .iter()
            .copied()
            .filter(|p| !comet_ids.contains(&p.id))
            .collect();
        if planets.is_empty() {
            return true;
        }
        let mut static_planets = Vec::new();
        let mut orbiting_planets = Vec::new();
        for p in planets {
            let orbital_r = distance((p.x, p.y), (CENTER, CENTER));
            if orbital_r + p.radius < ROTATION_RADIUS_LIMIT {
                orbiting_planets.push((p, orbital_r, (p.y - CENTER).atan2(p.x - CENTER)));
            } else {
                static_planets.push(p);
            }
        }
        for path in paths {
            for point in path {
                for p in &static_planets {
                    if distance((point.x, point.y), (p.x, p.y)) < p.radius + COMET_RADIUS + 0.5 {
                        return false;
                    }
                }
            }
        }
        for (step_idx, sym_points) in (0..visible.len()).map(|i| {
            (
                i,
                paths
                    .iter()
                    .map(move |path| path[i])
                    .collect::<Vec<Point>>(),
            )
        }) {
            let game_step = spawn_step - 1 + step_idx as i32;
            for point in sym_points {
                for (p, orbit_r, init_angle) in &orbiting_planets {
                    let angle = init_angle + self.angular_velocity * game_step as f64;
                    let ox = CENTER + orbit_r * angle.cos();
                    let oy = CENTER + orbit_r * angle.sin();
                    if distance((point.x, point.y), (ox, oy)) < p.radius + COMET_RADIUS {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub fn rewards(&self) -> Vec<i32> {
        if !self.done {
            return vec![0; self.num_players];
        }
        let scores = self.scores();
        let max_score = scores.iter().copied().max().unwrap_or(0);
        scores
            .into_iter()
            .map(|score| {
                if score == max_score && max_score > 0 {
                    1
                } else {
                    -1
                }
            })
            .collect()
    }

    pub fn scores(&self) -> Vec<i32> {
        let mut scores = vec![0; self.num_players];
        for planet in &self.planets {
            if planet.owner >= 0 {
                scores[planet.owner as usize] += planet.ships;
            }
        }
        for fleet in &self.fleets {
            if fleet.owner >= 0 {
                scores[fleet.owner as usize] += fleet.ships;
            }
        }
        scores
    }

    pub fn projected_margin_potential(&self, player: usize, production_weight: f64) -> f32 {
        let mut ships = vec![0.0; self.num_players];
        let mut production = vec![0.0; self.num_players];
        for planet in &self.planets {
            if planet.owner >= 0 {
                let owner = planet.owner as usize;
                ships[owner] += f64::from(planet.ships);
                production[owner] += f64::from(planet.production);
            }
        }
        for fleet in &self.fleets {
            if fleet.owner >= 0 {
                ships[fleet.owner as usize] += f64::from(fleet.ships);
            }
        }
        let turns_left = f64::from((self.episode_steps - self.step).max(0));
        let projected = |idx: usize| ships[idx] + production_weight * turns_left * production[idx];
        let own = projected(player);
        let enemy = (0..self.num_players)
            .filter(|&idx| idx != player)
            .map(projected)
            .fold(0.0, f64::max);
        (own - enemy) as f32
    }
}

pub fn point_to_segment_distance(point: (f64, f64), start: (f64, f64), end: (f64, f64)) -> f64 {
    let seg_x = end.0 - start.0;
    let seg_y = end.1 - start.1;
    let l2 = seg_x * seg_x + seg_y * seg_y;
    if l2 == 0.0 {
        return hypot(point.0 - start.0, point.1 - start.1);
    }
    let raw_t = ((point.0 - start.0) * seg_x + (point.1 - start.1) * seg_y) / l2;
    let t = raw_t.clamp(0.0, 1.0);
    let proj = (start.0 + t * seg_x, start.1 + t * seg_y);
    hypot(point.0 - proj.0, point.1 - proj.1)
}

fn distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    hypot(a.0 - b.0, a.1 - b.1)
}

fn hypot(x: f64, y: f64) -> f64 {
    (x * x + y * y).sqrt()
}

fn symmetric_group(
    id_counter: i32,
    x: f64,
    y: f64,
    radius: f64,
    ships: i32,
    prod: i32,
) -> Vec<Planet> {
    vec![
        Planet {
            id: id_counter,
            owner: -1,
            x,
            y,
            radius,
            ships,
            production: prod,
        },
        Planet {
            id: id_counter + 1,
            owner: -1,
            x: BOARD_SIZE - x,
            y,
            radius,
            ships,
            production: prod,
        },
        Planet {
            id: id_counter + 2,
            owner: -1,
            x,
            y: BOARD_SIZE - y,
            radius,
            ships,
            production: prod,
        },
        Planet {
            id: id_counter + 3,
            owner: -1,
            x: BOARD_SIZE - x,
            y: BOARD_SIZE - y,
            radius,
            ships,
            production: prod,
        },
    ]
}

fn valid_against_existing(temp: &[Planet], planets: &[Planet]) -> bool {
    temp.iter().all(|tp| {
        planets
            .iter()
            .all(|p| distance((p.x, p.y), (tp.x, tp.y)) >= p.radius + tp.radius + PLANET_CLEARANCE)
    })
}

fn valid_orbit_group(temp: &[Planet], planets: &[Planet], allow_same_mode: bool) -> bool {
    for tp in temp {
        let tp_orb = distance((tp.x, tp.y), (CENTER, CENTER));
        let tp_rot = tp_orb + tp.radius < ROTATION_RADIUS_LIMIT;
        for p in planets {
            let p_orb = distance((p.x, p.y), (CENTER, CENTER));
            let p_rot = p_orb + p.radius < ROTATION_RADIUS_LIMIT;
            if distance((p.x, p.y), (tp.x, tp.y)) < p.radius + tp.radius + PLANET_CLEARANCE {
                return false;
            }
            if (allow_same_mode || tp_rot != p_rot) && p_orb + p.radius >= ROTATION_RADIUS_LIMIT {
                if (tp_orb - p_orb).abs() < tp.radius + p.radius + PLANET_CLEARANCE {
                    return false;
                }
            } else if !allow_same_mode
                && tp_rot != p_rot
                && (tp_orb - p_orb).abs() < tp.radius + p.radius + PLANET_CLEARANCE
            {
                return false;
            }
        }
    }
    true
}

#[derive(Clone, Debug)]
struct PyRandom {
    state: [u32; 624],
    index: usize,
}

impl PyRandom {
    fn seed(seed: u32) -> Self {
        let mut rng = Self {
            state: [0; 624],
            index: 624,
        };
        rng.init_by_array(&[seed]);
        rng
    }

    fn random(&mut self) -> f64 {
        let a = (self.gen_u32() >> 5) as u64;
        let b = (self.gen_u32() >> 6) as u64;
        ((a * 67_108_864 + b) as f64) / 9_007_199_254_740_992.0
    }

    fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.random()
    }

    fn randint(&mut self, a: i32, b: i32) -> i32 {
        a + self.randbelow((b - a + 1) as u32) as i32
    }

    fn randbelow(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        let k = 32 - (n - 1).leading_zeros();
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    fn getrandbits(&mut self, k: u32) -> u32 {
        if k == 0 {
            return 0;
        }
        self.gen_u32() >> (32 - k)
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for i in 1..624 {
            self.state[i] = 1_812_433_253_u32
                .wrapping_mul(self.state[i - 1] ^ (self.state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        self.index = 624;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = 624.max(key.len());
        while k > 0 {
            self.state[i] = (self.state[i]
                ^ ((self.state[i - 1] ^ (self.state[i - 1] >> 30)).wrapping_mul(1_664_525)))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                self.state[0] = self.state[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = 623;
        while k > 0 {
            self.state[i] = (self.state[i]
                ^ ((self.state[i - 1] ^ (self.state[i - 1] >> 30)).wrapping_mul(1_566_083_941)))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                self.state[0] = self.state[623];
                i = 1;
            }
            k -= 1;
        }
        self.state[0] = 0x8000_0000;
        self.index = 624;
    }

    fn gen_u32(&mut self) -> u32 {
        const N: usize = 624;
        const M: usize = 397;
        const MATRIX_A: u32 = 0x9908_b0df;
        const UPPER_MASK: u32 = 0x8000_0000;
        const LOWER_MASK: u32 = 0x7fff_ffff;
        if self.index >= N {
            for kk in 0..(N - M) {
                let y = (self.state[kk] & UPPER_MASK) | (self.state[kk + 1] & LOWER_MASK);
                self.state[kk] =
                    self.state[kk + M] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
            }
            for kk in (N - M)..(N - 1) {
                let y = (self.state[kk] & UPPER_MASK) | (self.state[kk + 1] & LOWER_MASK);
                self.state[kk] =
                    self.state[kk + M - N] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
            }
            let y = (self.state[N - 1] & UPPER_MASK) | (self.state[0] & LOWER_MASK);
            self.state[N - 1] =
                self.state[M - 1] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }
}

pub fn fixture_game(num_players: usize, episode_steps: i32) -> Game {
    let planets = vec![
        Planet {
            id: 0,
            owner: 0,
            x: 20.0,
            y: 20.0,
            radius: 1.0,
            ships: 50,
            production: 5,
        },
        Planet {
            id: 1,
            owner: 1,
            x: 80.0,
            y: 80.0,
            radius: 1.0,
            ships: 50,
            production: 5,
        },
        Planet {
            id: 2,
            owner: -1,
            x: 20.0,
            y: 80.0,
            radius: 1.0,
            ships: 20,
            production: 3,
        },
        Planet {
            id: 3,
            owner: -1,
            x: 80.0,
            y: 20.0,
            radius: 1.0,
            ships: 20,
            production: 3,
        },
    ];
    Game::from_state(
        GameConfig::new(num_players, episode_steps, 6.0),
        GameState::new(1, 0.03, planets.clone(), planets, Vec::new(), Vec::new(), 0),
    )
}

pub fn simple_actions(game: &Game) -> Vec<PlayerAction> {
    let mut actions = vec![Vec::new(); game.num_players];
    for (player, player_actions) in actions.iter_mut().enumerate() {
        let player = player as i32;
        let mut launched = 0;
        for src in game
            .planets
            .iter()
            .filter(|p| p.owner == player && p.ships >= 8)
        {
            if launched >= 2 {
                break;
            }
            let Some(target) = game
                .planets
                .iter()
                .filter(|p| p.owner != player)
                .min_by(|a, b| {
                    let da = (a.x - src.x).powi(2) + (a.y - src.y).powi(2);
                    let db = (b.x - src.x).powi(2) + (b.y - src.y).powi(2);
                    da.total_cmp(&db)
                })
            else {
                continue;
            };
            let angle = (target.y - src.y).atan2(target.x - src.x);
            player_actions.push(Action::launch(src.id, angle, (src.ships / 3).max(1)));
            launched += 1;
        }
    }
    actions
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn sun_distance_hits_segment() {
        assert!(point_to_segment_distance((50.0, 50.0), (20.0, 20.0), (80.0, 80.0)) < 1e-9);
        assert!(point_to_segment_distance((50.0, 50.0), (20.0, 20.0), (20.0, 80.0)) > 10.0);
    }

    #[test]
    fn launches_debit_garrison_and_create_fleet() {
        let mut game = fixture_game(2, 80);
        let actions = vec![vec![Action::launch(0, 0.0, 10)], vec![]];
        let result = game.step(&actions);
        assert!(!result.done);
        assert_eq!(game.planets[0].ships, 45);
        assert_eq!(game.fleets.len(), 1);
        assert_eq!(game.fleets[0].ships, 10);
        assert_eq!(game.fleets[0].owner, 0);
    }

    #[test]
    fn projected_margin_potential_keeps_remaining_horizon_after_early_done() {
        let planets = vec![
            Planet {
                id: 0,
                owner: 0,
                x: 20.0,
                y: 20.0,
                radius: 1.0,
                ships: 10,
                production: 2,
            },
            Planet {
                id: 1,
                owner: 1,
                x: 80.0,
                y: 80.0,
                radius: 1.0,
                ships: 20,
                production: 1,
            },
        ];
        let mut game = Game::from_state(
            GameConfig::new(2, 100, 6.0),
            GameState::new(
                10,
                0.03,
                planets.clone(),
                planets,
                Vec::new(),
                Vec::new(),
                0,
            ),
        );
        game.done = true;

        let own = 10.0 + 90.0 * 2.0;
        let enemy = 20.0 + 90.0;
        let expected = own - enemy;
        assert!((f64::from(game.projected_margin_potential(0, 1.0)) - expected).abs() < 1e-6);
    }

    #[test]
    fn simple_fixture_runs_to_terminal() {
        let mut game = fixture_game(2, 80);
        while !game.done {
            let actions = simple_actions(&game);
            game.step(&actions);
        }
        assert_eq!(game.rewards().len(), 2);
    }

    #[test]
    fn combat_can_capture_planet() {
        let planets = vec![
            Planet {
                id: 0,
                owner: 0,
                x: 10.0,
                y: 10.0,
                radius: 1.0,
                ships: 100,
                production: 1,
            },
            Planet {
                id: 1,
                owner: 1,
                x: 20.0,
                y: 10.0,
                radius: 1.0,
                ships: 5,
                production: 1,
            },
        ];
        let fleets = vec![Fleet {
            id: 0,
            owner: 0,
            x: 18.0,
            y: 10.0,
            angle: 0.0,
            from_planet_id: 0,
            ships: 20,
            target_id: -1,
            eta: 0.0,
            target_x: 0.0,
            target_y: 0.0,
        }];
        let mut game = Game::from_state(
            GameConfig::new(2, 30, 6.0),
            GameState::new(1, 0.0, planets.clone(), planets, fleets, Vec::new(), 1),
        );
        game.step(&[vec![], vec![]]);
        assert_eq!(game.planets[1].owner, 0);
    }

    #[test]
    fn orbiting_planets_rotate() {
        let planets = vec![Planet {
            id: 0,
            owner: 0,
            x: CENTER + 20.0,
            y: CENTER,
            radius: 1.0,
            ships: 10,
            production: 1,
        }];
        let mut game = Game::from_state(
            GameConfig::new(2, 10, 6.0),
            GameState::new(1, PI / 4.0, planets.clone(), planets, vec![], Vec::new(), 0),
        );
        game.step(&[vec![], vec![]]);
        assert!((game.planets[0].x - (CENTER + 20.0 * (PI / 4.0).cos())).abs() < 1e-9);
        assert!((game.planets[0].y - (CENTER + 20.0 * (PI / 4.0).sin())).abs() < 1e-9);
    }

    #[test]
    fn active_comets_move_along_paths_and_expire() {
        let planets = vec![
            Planet {
                id: 0,
                owner: 0,
                x: 10.0,
                y: 80.0,
                radius: 1.0,
                ships: 20,
                production: 1,
            },
            Planet {
                id: 1,
                owner: 1,
                x: 80.0,
                y: 10.0,
                radius: 1.0,
                ships: 20,
                production: 1,
            },
            Planet {
                id: 10,
                owner: -1,
                x: -99.0,
                y: -99.0,
                radius: 1.0,
                ships: 5,
                production: 1,
            },
        ];
        let comets = vec![CometGroup {
            planet_ids: vec![10],
            paths: vec![vec![Point::new(10.0, 10.0), Point::new(14.0, 10.0)]],
            path_index: -1,
        }];
        let mut game = Game::from_state(
            GameConfig::new(2, 10, 6.0),
            GameState::new(1, 0.0, planets.clone(), planets, vec![], comets, 0),
        );
        game.step(&[vec![], vec![]]);
        assert_eq!(game.planets[2].x, 10.0);
        game.step(&[vec![], vec![]]);
        assert_eq!(game.planets[2].x, 14.0);
        game.step(&[vec![], vec![]]);
        assert_eq!(game.planets.len(), 2);
        assert!(game.comets.is_empty());
    }

    #[test]
    fn python_random_seed_zero_matches_cpython() {
        let mut rng = PyRandom::seed(0);
        let expected = [
            0.8444218515250481,
            0.7579544029403025,
            0.420571580830845,
            0.25891675029296335,
            0.5112747213686085,
        ];
        for value in expected {
            assert!((rng.random() - value).abs() < 1e-16);
        }
        let mut rng = PyRandom::seed(0);
        let ints = [4, 4, 1, 3, 5, 4, 4, 3, 4, 3];
        for value in ints {
            assert_eq!(rng.randint(1, 5), value);
        }
    }
}
