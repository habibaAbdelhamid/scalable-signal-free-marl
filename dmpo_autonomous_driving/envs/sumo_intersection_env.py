"""
SUMO Intersection Environment Wrapper for DMPO

Main environment wrapper that adapts 4-lane intersection SUMO simulation
for Networked-MB-MARL DMPO algorithm. Follows the CACCWrapper pattern.
"""

import gym
import numpy as np

# SUMO interface selection - check environment variable first
# Set DMPO_USE_TRACI=1 to force TraCI (required for GUI and full metric access)
import os
FORCE_TRACI = os.environ.get('DMPO_USE_TRACI', '0') == '1'

if FORCE_TRACI:
    import traci
    USING_LIBSUMO = False
    print("📊 Using TraCI (DMPO_USE_TRACI=1)")
else:
    # Use libsumo for faster simulation (5x speedup over TraCI)
    # libsumo is a singleton - we start it once and use load() for resets
    try:
        import libsumo as traci
        USING_LIBSUMO = True
        print("🚀 Using libsumo (fast mode - ~5x speedup over TraCI)")
    except ImportError:
        import traci
        USING_LIBSUMO = False
        print("⚠️  libsumo not available, falling back to TraCI (slower)")

import sumolib
from gym.spaces import Box, Dict
from typing import Dict as TypeDict, List, Tuple, Any, Optional
import os
import sys
import threading
import socket
import json
import time

# Add Networked-MB-MARL to path for imports (DMPO_FRAMEWORK_PATH, default <repo>/third_party/Networked-MB-MARL)
sys.path.append(os.environ.get('DMPO_FRAMEWORK_PATH', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'third_party', 'Networked-MB-MARL')))

from .observation_processor import ObservationProcessor
from .action_translator import ActionTranslator
from .reward_computer import RewardComputer
from ..utils.safety_monitor import SafetyMonitor
from ..utils.route_manager import RouteManager


def _reward_kappa():
    """DMPO_REWARD_KAPPA (reward-radius ablation): None when unset (legacy), else int ≥ 0."""
    _v = os.environ.get('DMPO_REWARD_KAPPA', '').strip()
    return int(_v) if _v != '' else None


def _reach(adj, r):
    """≤r-hop reachability (self included) of a reflexive adjacency; r=0 → identity."""
    n = adj.shape[0]
    if r <= 0:
        return np.eye(n, dtype=np.float32)
    A = (np.asarray(adj) > 0).astype(np.float32)
    np.fill_diagonal(A, 1.0)
    R = A.copy(); cur = A.copy()
    for _ in range(1, r):
        cur = (cur @ A > 0).astype(np.float32)
        R = np.maximum(R, cur)
    return R


class SUMOIntersectionEnv(gym.Env):
    """
    SUMO intersection environment wrapper for DMPO.
    
    Adapts 4-lane intersection SUMO simulation to provide:
    - Multi-agent interface compatible with Networked-MB-MARL
    - κ-hop neighbor observations for decentralized learning
    - Continuous action space for vehicle control
    - Multi-objective reward function for autonomous driving
    """
    
    def __init__(self, sumo_config_path: str, use_gui: bool = False,
                 max_steps: int = 3600, step_length: float = 0.2,
                 enable_external_eval: bool = False, training_mode: bool = True,
                 communication_radius: float = 100.0, max_ai_vehicles: int = 0,
                 kappa: int = 2):
        """
        Initialize SUMO intersection environment.

        Args:
            sumo_config_path: Path to SUMO configuration file
            use_gui: Whether to use SUMO GUI
            max_steps: Maximum simulation steps
            step_length: SUMO simulation step length
            enable_external_eval: Enable external TraCI connections for evaluation
            training_mode: If True, use relaxed safety constraints for learning
        """
        self.training_mode = training_mode
        self.communication_radius = communication_radius
        super().__init__()

        # SUMO configuration
        self.sumo_config_path = sumo_config_path
        self.use_gui = use_gui
        # Optional GUI frame capture for eval videos (DMPO_VIDEO_DIR). GUI-only; each step
        # requests a sumo-gui screenshot → stitch to mp4 offline. No-op headless/libsumo.
        self._video_dir = os.environ.get('DMPO_VIDEO_DIR', '').strip()
        self._video_view_set = False
        self._video_halfspan = float(os.environ.get('DMPO_VIDEO_HALFSPAN', '70'))
        if self._video_dir and use_gui:
            os.makedirs(self._video_dir, exist_ok=True)
        self.max_steps = max_steps
        # Allow DMPO_STEP_LENGTH env var to override, for ablation studies.
        # Default 1.0s matches human-realistic decision frequency at intersections
        # (AASHTO Green Book: 1.0-2.5s for yield/go decisions).
        _env_step = os.environ.get('DMPO_STEP_LENGTH', '')
        self.step_length = float(_env_step) if _env_step else step_length
        self.enable_external_eval = enable_external_eval
        self.sumo_port = None
        self.sumo_process = None

        print(f"🔧 SUMOIntersectionEnv: enable_external_eval = {enable_external_eval}")
        print(f"🔧 SUMO step_length = {self.step_length}s (decision frequency = {1.0/self.step_length:.1f} Hz)")

        self.sumo_cmd = self._build_sumo_command()
        print(f"🔧 SUMO command: {' '.join(self.sumo_cmd)}")
        
        # Environment state
        self.current_step = 0
        self.ai_vehicle_ids = set()
        self.human_vehicle_ids = set()
        self.all_vehicle_ids = set()

        # Evaluation proxy
        self.eval_proxy_server = None
        self.eval_proxy_thread = None
        self.eval_proxy_port = None

        # libsumo tracking
        self._libsumo_started = False
        
        # P3.6: store kappa so CLI --kappa propagates to obs proc + reward team_bonus.
        # Previously hardcoded to 2, silently invalidating any κ-ablation.
        self.kappa = kappa

        # Component initialization
        self.observation_processor = ObservationProcessor(kappa=kappa, communication_radius=self.communication_radius)
        self.action_translator = ActionTranslator()
        self.reward_computer = RewardComputer()
        self.safety_monitor = SafetyMonitor(training_mode=self.training_mode)
        self.route_manager = RouteManager()
        
        # Spaces (will be updated dynamically)
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
        
        # Communication adjacency matrix (dynamic)
        self.adjacency_matrix = None

        self._max_ai_vehicles = max_ai_vehicles
        self.n_agents = 0

        # Previous state for reward computation
        self.prev_observations = {}
        self.prev_actions = {}

        # Vehicle cap: 0 = no cap, >0 = max AI vehicles kept per step
        self.max_ai_vehicles = max_ai_vehicles

        # V2 inflow-curriculum: SUMO traffic scale applied on every reset.
        # The curriculum ramps this (0.2→1.0) to vary concurrent density while
        # max_ai_vehicles stays fixed above the peak, so the density cap never
        # releases a vehicle to Krauss — every vehicle stays policy-controlled.
        self._traffic_scale = float(os.environ.get('DMPO_TRAFFIC_SCALE', '1.0'))

        # Per-step vehicle state cache — avoids redundant SUMO API calls
        self._vehicle_cache = {}
        self._vehicle_cache_step = -1
        # M12-0: monotone cache-epoch token. Bumped exactly once per fresh cache build
        # (and once per episode reset), published to the obs processor next to the cache
        # so its per-step memos have ONE explicit invalidation point (never id()-based).
        self._cache_epoch = 0
        # OBS-V2 (Stage-3): env-side gate mirror + the per-step scaffold-hold set the
        # obs temporal/conflict blocks consume (one-step lag documented in the spec).
        self._obs_v2 = os.environ.get('DMPO_OBS_V2', '0') == '1'
        # ARM-B (user design 2026-07-26): per-vehicle lifetime driving profile
        # [n_steps, sum_speed, sum_|accel|, n_brake_steps], accumulated at cache
        # build, published to the obs processor. Endogenous distinguishability —
        # no rule, no priority semantics. Gated; OFF = zero overhead.
        self._profile_obs_on = self._obs_v2 and os.environ.get('DMPO_PROFILE_OBS', '0') == '1'
        self._veh_profiles = {}
        self._shield_engaged_ids = set()

        # Collision trace diagnostics — when DMPO_COLLISION_TRACE=1, every
        # collision record gets enriched with full state snapshot + 10-step
        # history of both vehicles. Off in training to avoid overhead.
        self._collision_trace_enabled = os.environ.get('DMPO_COLLISION_TRACE', '0') == '1'
        self._vehicle_state_history = {}  # vid → deque of last 10 state dicts

        # Track previously seen AI vehicles for setSpeedMode optimization
        self._prev_ai_veh_set = set()

        # SLOT IDENTITY (B1 proper): persistent vehicle → slot mapping.
        # Each vehicle keeps its slot for its entire lifetime; slots are reused
        # after the previous occupant leaves. This makes slot identity stable
        # within a vehicle's lifetime, enabling correct GAE bootstrapping.
        self._vehicle_to_slot = {}  # vehicle_id → slot index (0..max_ai_vehicles-1)
        self._free_slots = None      # set of free slot indices, lazily initialized

        # Gridlock detector: tracks no-progress windows so the env can
        # terminate an episode that has gone into deadlock instead of
        # relying on SUMO's silent teleport rescue.
        # DMPO_GRIDLOCK_DETECT=1 enables it. Tunable thresholds:
        #   DMPO_GRIDLOCK_NO_COMPLETION_STEPS  (default 150 = 30 sim-sec at dt=0.2)
        #   DMPO_GRIDLOCK_MIN_AI_SPEED_M_S     (default 0.5 m/s)
        self._gridlock_enabled = os.environ.get('DMPO_GRIDLOCK_DETECT', '0') == '1'
        self._gridlock_no_comp_thresh = int(os.environ.get('DMPO_GRIDLOCK_NO_COMPLETION_STEPS', '150'))
        self._gridlock_speed_thresh = float(os.environ.get('DMPO_GRIDLOCK_MIN_AI_SPEED_M_S', '0.5'))
        self._steps_since_completion = 0
        self._gridlock_triggered = False
        self._overload_triggered = False
        self._stopped_steps = {}   # veh_id -> consecutive frozen steps (clear-stopped)

        # AI "stuck after collision" state — when an AI collides, we hold it
        # at speed 0 until its leader gap clears, then release control back
        # to the policy. Humans are never touched. See step() for logic.
        # Tracking is per-vehicle so we know how long each one has been
        # frozen (used for the time-based release fallback).
        # DMPO_COLLISION_STUCK_RELEASE_GAP_M (default 5m) — gap to declare path clear
        # DMPO_COLLISION_STUCK_MAX_STEPS (default 50 = 10s) — max time stuck
        #   before we force-release regardless of gap, returning control to
        #   the policy. Prevents indefinite stuck when blocker is also stuck.
        self._stuck_ai = {}  # veh_id -> step_when_stuck
        self._stuck_release_gap = float(os.environ.get('DMPO_COLLISION_STUCK_RELEASE_GAP_M', '5.0'))
        self._stuck_max_steps = int(os.environ.get('DMPO_COLLISION_STUCK_MAX_STEPS', '50'))
        # Collision pair deduplication: SUMO's getCollisions() reports the same
        # pair every step the vehicles are physically overlapping. Without
        # deduplication, collision_penalty fires hundreds of times per episode
        # for what is morally a single collision event. Track which (a, b) pairs
        # have already been penalized this episode and skip repeats. Reset per
        # reset() in env.reset().
        self._collision_pairs_seen = set()  # set of tuple(sorted([col_id, vic_id]))

        # P0.1: REAL collision metric (replaces the stale safety_monitor.collision_count
        # that was never incremented in training). Accumulates DISTINCT new
        # AI-involving collision pairs over the current episode. Mechanism + pair_kind
        # breakdowns also accumulate when DMPO_COLLISION_TRACE=1.
        self._ep_ai_collision_pairs = 0
        self._ep_ai_collision_breakdown = {}   # {'AI-AI': int, 'AI-human(...)': int, ...}
        self._ep_mechanism_breakdown = {}      # {'drove_over_stationary': int, ...}
        
    def _build_sumo_command(self) -> List[str]:
        """Build SUMO command based on configuration."""
        if self.use_gui:
            cmd = ["sumo-gui", "-c", self.sumo_config_path]
            # Add GUI-specific options for auto-start
            cmd.extend([
                "--start"  # Auto-start simulation
                # NOTE: Removed --quit-on-end to allow screenshot capture
            ])
            # DMPO_GUI_DELAY_MS: per-step delay in ms for GUI rendering.
            # Default 0 = run as fast as possible. ALWAYS pass to avoid SUMO
            # falling back to a non-zero default.
            _gui_delay = os.environ.get('DMPO_GUI_DELAY_MS', '0')
            cmd.extend(["--delay", _gui_delay or '0'])
        else:
            cmd = ["sumo", "-c", self.sumo_config_path]
            
        # SPEED OPTIMIZATION: Optimized SUMO settings for faster simulation
        # Combined optimizations provide ~2-3x speedup
        #
        # DETERMINISTIC SEED: Pass --seed so SUMO's RNG is reproducible.
        # Without this, human vehicle departLane="random" and departSpeed="random"
        # produce different traffic patterns each launch, making training
        # non-reproducible across runs with the same PyTorch seed.
        _sumo_seed = os.environ.get('DMPO_SUMO_SEED', '23423')
        # Collision action configurable — "none" (default), "warn", "stop",
        # "teleport", "remove". For honest accounting use "remove" + a real
        # collision penalty in the reward path; teleport hides failures.
        _collision_action = os.environ.get('DMPO_COLLISION_ACTION', 'none')
        # Junction collisions: default false (legacy). Set to "true" to detect
        # collisions inside the intersection (recommended with collision.action=remove).
        _check_junctions = os.environ.get('DMPO_COLLISION_CHECK_JUNCTIONS', 'false')
        # Time the colliding vehicles stay halted before being removed/resuming.
        # Only meaningful for collision.action in {"stop", "remove"}.
        _stoptime = os.environ.get('DMPO_COLLISION_STOPTIME', '10')
        # If teleport is used for collision.action, also set a finite
        # time-to-teleport so stuck vehicles can be freed; otherwise default to
        # -1 (disabled) so deadlocks are visible and a gridlock detector handles
        # them at the env level instead.
        _ttt = os.environ.get('DMPO_TIME_TO_TELEPORT',
                              '30' if _collision_action == 'teleport' else '-1')
        cmd.extend([
            "--seed", _sumo_seed,
            "--step-length", str(self.step_length),
            "--collision.action", _collision_action,
            "--collision.check-junctions", _check_junctions,
            "--collision.stoptime", _stoptime,
            "--time-to-teleport", _ttt,
            # SPEED OPT 3: Reduce output overhead
            # (forensic override: DMPO_SUMO_WARNINGS=1 re-enables warnings so
            #  SUMO-internal interventions like emergency stops become visible)
            "--no-warnings", "false" if os.environ.get('DMPO_SUMO_WARNINGS', '0') == '1' else "true",
            "--no-step-log", "true",
            "--log", "/dev/null",
            # SPEED OPT 4: Single thread (multi-threading adds overhead with no speedup)
            "--threads", "1",
            # SPEED OPT 5: Simpler internal models
            "--lanechange.duration", "0",  # Instant lane changes
            "--default.emergencydecel", "9.0",  # Faster emergency response
            # Sublane model: enables continuous lateral positioning
            # Without this, lateral actions map to discrete lane changes only
            "--lateral-resolution", "0.8"
        ])

        # Add evaluation proxy port when external evaluation is enabled
        if self.enable_external_eval:
            if not self.sumo_port:
                self.sumo_port = self._get_free_port()
            self.eval_proxy_port = self._get_free_port()
            print(f"🔗 Evaluation proxy will be available on port {self.eval_proxy_port}")
            print(f"🔗 Training will connect to SUMO directly (single client mode)")
            print(f"🔧 Auto-detection hint: Proxy port = {self.eval_proxy_port}")

        return cmd

    def _get_free_port(self):
        """Get a free port for SUMO TraCI server."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def _create_observation_space(self) -> Dict:
        """Create observation space for intersection driving."""
        if os.environ.get('DMPO_OBS_V2', '0') == '1':
            # OBS-V2 (spec Step 7): declare ALL blocks with the real v2 shapes and
            # honest bounds (every slot is constructed inside [-2, 2]). Informational
            # (only observation_dim is consumed downstream) but it is a paper artifact
            # and must not lie — the legacy Dict omitted context/identity entirely.
            _max_nb = int(os.environ.get('DMPO_MAX_NEIGHBORS', '10'))
            # ARM-B: size ego/conflict_gap from the processor's (instance-shadowed)
            # constants so the declared space never lies under DMPO_PROFILE_OBS=1.
            _ego_d = int(getattr(self.observation_processor, 'V2_EGO_DIM', 12))
            _conf_d = int(getattr(self.observation_processor, 'V2_CONFLICT_BLOCK', 31))
            return Dict({
                'ego': Box(-2.0, 2.0, shape=(_ego_d,), dtype=np.float32),
                'infrastructure': Box(-2.0, 2.0, shape=(5,), dtype=np.float32),
                'neighbors': Box(-2.0, 2.0, shape=(_max_nb, 12), dtype=np.float32),
                'neighbor_mask': Box(0, 1, shape=(_max_nb,), dtype=np.float32),
                'context': Box(-2.0, 2.0, shape=(8,), dtype=np.float32),
                'identity': Box(0.0, 1.0, shape=(4,), dtype=np.float32),
                'temporal': Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
                'conflict_gap': Box(-2.0, 2.0, shape=(_conf_d,), dtype=np.float32),
            })
        # Ego vehicle state: [pos_x, pos_y, vel, accel, heading, lane_id,
        #                     distance_to_intersection, route_direction, time_to_dest]
        ego_space = Box(-1000.0, 1000.0, shape=(10,), dtype=np.float32)
        
        # Infrastructure state: [traffic_light_phase, traffic_light_timer, 
        #                       speed_limit, lane_count, visibility]
        infrastructure_space = Box(-1.0, 100.0, shape=(5,), dtype=np.float32)
        
        # Neighbor states: variable length list of neighbor info
        # Each neighbor: [rel_x, rel_y, rel_vel, rel_heading, lane_id, is_ai,
        #                 nb_dist_to_junction, nb_abs_vx, nb_abs_vy, nb_ttc]  (10 features)
        # max_neighbors follows DMPO_MAX_NEIGHBORS (same source as the processor + config)
        # so the declared space, the processor array, and observation_dim never drift.
        _max_nb = int(os.environ.get('DMPO_MAX_NEIGHBORS', '10'))
        neighbor_space = Box(-1000.0, 1000.0, shape=(_max_nb, 10), dtype=np.float32)  # max_nb × 10 features

        return Dict({
            'ego': ego_space,
            'infrastructure': infrastructure_space,
            'neighbors': neighbor_space,
            'neighbor_mask': Box(0, 1, shape=(_max_nb,), dtype=np.float32)
        })
    
    def _create_action_space(self) -> Dict:
        """Create action space for intersection driving."""
        return Dict({
            'longitudinal': Box(-1.0, 1.0, shape=(1,), dtype=np.float32),  # Normalized acceleration
            'lateral': Box(-1.0, 1.0, shape=(1,), dtype=np.float32),       # Normalized steering
            'lane_intention': Box(-1.0, 1.0, shape=(1,), dtype=np.float32) # Lane change intent
        })
    
    def _needs_hard_start(self) -> bool:
        """Check if SUMO needs a full (re)start vs a soft reset.

        Threshold is `DMPO_SUMO_HARDRESET_AT` (default 300000s for the legacy
        long-horizon scenarios). Set lower (e.g., 25000) for scenarios with
        short route-file horizons so the env cycles through fresh traffic
        rather than running out of vehicles mid-training.
        """
        if USING_LIBSUMO:
            if not self._libsumo_started:
                return True
            try:
                sim_time = traci.simulation.getTime()
                threshold = float(os.environ.get('DMPO_SUMO_HARDRESET_AT', '300000'))
                if sim_time >= threshold:
                    print(f"  SUMO hard reset: sim_time={sim_time:.0f}s ≥ threshold={threshold:.0f}s")
                    return True
            except:
                pass
            return False
        else:
            if not traci.isLoaded():
                return True
            try:
                sim_time = traci.simulation.getTime()
                end_time = traci.simulation.getEndTime()
                if end_time > 0 and sim_time >= end_time - 60000:
                    return True
            except Exception:
                return True
            return False

    def _hard_start_sumo(self):
        """Perform a full SUMO (re)start — used on first call or when simulation ends."""
        # F27 fix (soundness review): the per-step cache MUST NOT survive a hard
        # restart. Flow-derived vehicle ids restart from `.0` after traci.load, so the
        # first cache build of the new run would otherwise read the PREVIOUS run's
        # final speed as `prev_speed` for every reused id — emitting a saturated
        # phantom brake/accel event on nb_dv in the first frame of the new episode.
        # SOFT resets deliberately keep the cache (continuity is correct there).
        self._vehicle_cache = {}
        self._vehicle_cache_step = -1
        # ARM-B: profiles are per-vehicle lifetimes — a hard restart despawns every
        # vehicle AND reuses flow-derived ids from .0, so stale stats would leak
        # a previous run's history into a new vehicle (same hazard class as F27).
        self._veh_profiles = {}
        # ARM-A DIVERSITY (DMPO_SEED_CYCLE=1, default OFF): rotate the SUMO --seed on
        # every hard (re)start so each cold-start episode draws a fresh traffic
        # realization — with exp-period (Poisson) flows the seed drives every insertion
        # time. Without this, sumo_cmd (built once at init) is replayed VERBATIM by
        # every restart, so training replays ONE arrival schedule for the whole run
        # (the metronome-specialization root cause). Deterministic sequence
        # (base + restart index, first start = base) keeps runs reproducible.
        # Training-side data-diversity only — never set in eval (eval pins one seed).
        # Single counter for BOTH rotation levers below (verified bug 2026-07-31:
        # the increment used to live inside the SEED_CYCLE branch, so scenario
        # rotation silently froze at _paths[0] whenever SEED_CYCLE was off).
        self._hard_start_count = getattr(self, '_hard_start_count', -1) + 1
        if os.environ.get('DMPO_SEED_CYCLE', '0') == '1':
            try:
                _i = self.sumo_cmd.index('--seed')
                _base = int(os.environ.get('DMPO_SUMO_SEED', '23423'))
                self.sumo_cmd[_i + 1] = str(_base + self._hard_start_count)
                print(f"🎲 seed-cycle: hard start #{self._hard_start_count} --seed {self.sumo_cmd[_i + 1]}")
            except ValueError:
                pass
        # ARM-C DRILL CURRICULUM (DMPO_SCENARIO_CYCLE, default OFF): rotate the
        # SUMO config through a comma-separated list per hard restart — used to
        # interleave drill episodes (concentrated sudden-brake practice) with the
        # main scenario. ALL listed scenarios MUST share the same .net.xml (the
        # obs processor's conflict maps are built once per processor lifetime).
        # Training-side data-curriculum only — never set at eval.
        _cyc = os.environ.get('DMPO_SCENARIO_CYCLE', '')
        if _cyc:
            _paths = [p for p in _cyc.split(',') if p]
            if _paths:
                _pick = _paths[self._hard_start_count % len(_paths)]
                try:
                    _ci = self.sumo_cmd.index('-c')
                    self.sumo_cmd[_ci + 1] = _pick
                    print(f"🎯 scenario-cycle: hard start #{self._hard_start_count} -> {_pick.split('/')[-2]}")
                except ValueError:
                    pass
        if USING_LIBSUMO:
            if self._libsumo_started:
                traci.load(self.sumo_cmd[1:])
            else:
                traci.start(self.sumo_cmd)
                self._libsumo_started = True
            self._apply_tls_off_list()
            self._init_tls_controller()
        else:
            if traci.isLoaded():
                traci.close()

            if self.sumo_process:
                try:
                    self.sumo_process.terminate()
                    self.sumo_process.wait(timeout=5)
                except Exception:
                    self.sumo_process.kill()
                self.sumo_process = None

            if self.enable_external_eval:
                print(f"EXTERNAL EVAL MODE: Starting SUMO with evaluation proxy")

                if self.eval_proxy_port is None:
                    self.eval_proxy_port = self._get_free_port()
                    print(f"Allocated evaluation proxy port: {self.eval_proxy_port}")

                normal_cmd = [arg for arg in self.sumo_cmd if arg not in ["--remote-port", str(self.sumo_port)]]
                traci.start(normal_cmd)

                self._start_evaluation_proxy()

                print(f"Connected to SUMO TraCI server")
                print(f"Evaluation proxy running on port {self.eval_proxy_port}")

                self._external_port = self.eval_proxy_port
                self._write_port_file()
            else:
                normal_cmd = [arg for arg in self.sumo_cmd if arg not in ["--num-clients", "2"]]
                traci.start(normal_cmd)
            self._apply_tls_off_list()
            self._init_tls_controller()

        print(f"SUMO hard start complete (sim time: {traci.simulation.getTime():.1f}s)")

    def _init_tls_controller(self):
        """Network signal baseline (gated DMPO_TLS_CONTROLLER=maxpressure, default unset =
        no-op): (re)build the controller after every SUMO (re)start, AFTER the off-list so
        unsignalized cells stay off. See utils/max_pressure.py."""
        from dmpo_autonomous_driving.utils import max_pressure as _mp
        self._tls_ctrl = _mp.make(traci)

    def _apply_tls_off_list(self):
        """GRID-MATRIX suite (gated DMPO_TLS_OFF_LIST, default unset = no-op):
        switch the listed traffic lights to their built-in 'off' program right
        after SUMO (re)starts, turning those junctions unsignalized while the
        rest keep their signals — the per-junction mixing lever for the
        signalized/unsignalized comparison cells (sumocfg tls.all-off is
        all-or-nothing). Must run after BOTH the libsumo and TraCI start paths
        and after every traci.load (programs reset on reload)."""
        _off = os.environ.get('DMPO_TLS_OFF_LIST', '')
        if not _off:
            return
        _done = 0
        for _t in _off.split(','):
            _t = _t.strip()
            if not _t:
                continue
            try:
                traci.trafficlight.setProgram(_t, 'off')
                _done += 1
            except Exception as _e:
                print(f"⚠️ TLS_OFF_LIST: could not switch '{_t}' off: {_e}")
        print(f"🚦 TLS_OFF_LIST: {_done} junction(s) switched to unsignalized")

    def set_traffic_scale(self, scale: float):
        """Set the SUMO inflow scale for the V2 inflow-curriculum.

        Stored and re-applied on every reset() (a hard reset would otherwise
        reset the connection's scale to the command-line default). Also applied
        immediately if a connection is live, so a mid-run change takes effect.
        """
        self._traffic_scale = float(scale)
        try:
            traci.simulation.setScale(self._traffic_scale)
        except Exception:
            pass

    def reset(self) -> TypeDict[int, Any]:
        """Reset RL episode state. SUMO continues running (soft reset).

        Soft reset: only RL bookkeeping is cleared (step counter, rewards,
        component state). SUMO keeps running with vehicles already in the
        network, so vehicles don't need to re-traverse long approach roads.

        Hard start is only done on:
        - First call (no SUMO running yet)
        - SUMO simulation reached its configured end time
        """
        if self._needs_hard_start() or getattr(self, '_force_hard_next', False):
            if getattr(self, '_force_hard_next', False):
                # SKIP-LOOP RECOVERY (Stage-4 overnight finding): a jammed network
                # survives soft resets forever — scaffsel_s7 burned 5000 iterations
                # of T<2 skip-episodes because every soft reset inherited the jam.
                # The trainer sets this flag after K consecutive skipped updates.
                print("Environment reset: FORCED HARD START (skip-loop recovery — "
                      "clearing the inherited jammed network)")
            else:
                print(f"Environment reset: HARD START (first call or simulation ended)")
            self._force_hard_next = False
            self._hard_start_sumo()
        elif os.environ.get('DMPO_QUIET_RESET', '0') != '1':
            # LOG-SPAM FIX: this fires once per episode; at high n_episodes it
            # floods the log (GBs/day) and fills disk. Gate it behind
            # DMPO_QUIET_RESET=1 to silence soft-reset lines.
            print(f"Environment reset: soft reset (SUMO continues at t={traci.simulation.getTime():.1f}s)")

        # Apply the current traffic-inflow scale on the active connection.
        # Used by the V2 inflow-curriculum (ramp density 0.2→1.0) so every
        # concurrent vehicle stays policy-controlled (no Krauss release from a
        # density cap). setScale persists per-connection but a hard reset
        # (traci.load/start) resets it, so re-apply here every reset.
        try:
            traci.simulation.setScale(float(self._traffic_scale))
        except Exception:
            pass

        # Reset RL episode state
        self.current_step = 0
        self.ai_vehicle_ids = set()
        self.human_vehicle_ids = set()
        self.all_vehicle_ids = set()
        self.prev_observations = {}
        self.prev_actions = {}
        self._prev_ai_vehicle_ids = set()
        self._prev_ai_veh_set = set()
        # SLOT IDENTITY FIX: clear slot mapping at episode boundary so each
        # episode starts with fresh slot assignments.
        self._vehicle_to_slot = {}
        self._free_slots = None  # lazily reinitialized in _update_vehicle_tracking

        # Reset gridlock detector state at the start of each episode.
        self._steps_since_completion = 0
        self._gridlock_triggered = False
        self._overload_triggered = False
        self._stopped_steps = {}   # veh_id -> consecutive frozen steps (clear-stopped)
        # Clear stuck-AI tracking at episode boundary.
        self._stuck_ai = {}
        # Reset collision pair dedup tracker — a "new" episode means each pair
        # gets one penalty fresh.
        self._collision_pairs_seen = set()
        # Reset per-vehicle state history (used by collision trace diagnostics).
        self._vehicle_state_history = {}
        # Reset real collision counters (P0.1).
        # Print the PREVIOUS episode's collision mechanism breakdown (forensic) so
        # an eval can see junction-crossing vs rear-end without a custom harness.
        if getattr(self, '_collision_trace_enabled', False) and getattr(self, '_ep_mechanism_breakdown', None):
            print(f"🔬 MECH-BREAKDOWN: {dict(self._ep_mechanism_breakdown)} | pairs: {dict(self._ep_ai_collision_breakdown)}")
        self._ep_ai_collision_pairs = 0
        self._ep_ai_collision_breakdown = {}
        self._ep_mechanism_breakdown = {}

        # Reset ALL component state
        self.reward_computer.reset()
        self.safety_monitor.reset_metrics()
        self.action_translator.prev_actions.clear()
        # F-env fix (soundness review): clear the scaffold-hold set at the episode
        # boundary so the first obs of an episode never carries the PREVIOUS
        # episode's holds (one-step lag is within-episode only).
        self._shield_engaged_ids = set()
        # M12-0: invalidate the obs processor's per-step memos at the episode boundary
        # (parity-preserving — memos are recomputed from the same cache). M12-5B: the
        # per-episode obs-state clear (context[9] counters) is checkpoint-AFFECTING and
        # therefore gated DMPO_OBS_STATE_RESET (default 0 = legacy). Both live HERE in
        # the reset path — never in step() (the spec's env:617 anchor was a comment
        # inside step(); calling it there would wipe the memos every timestep).
        self._cache_epoch += 1
        if os.environ.get('DMPO_OBS_STATE_RESET', '0') == '1':
            self.observation_processor.reset_episode_state()

        # Step once to get current vehicles
        traci.simulationStep()
        self.current_step += 1

        # Update vehicle tracking (categorize + cap)
        self._update_vehicle_tracking()

        # Build cache before observations (consistent with step())
        self._build_vehicle_cache()

        # KAPPA-ADV package 1a: fresh graph at reset too (see step()).
        if os.environ.get('DMPO_GRAPH_FRESH', '0') == '1':
            self._update_communication_graph()

        # Get initial observations
        observations = self._get_observations()

        return observations
    
    def step(self, actions: TypeDict[str, TypeDict[str, np.ndarray]]) -> Tuple[
        TypeDict[str, Any], TypeDict[str, float], TypeDict[str, bool], TypeDict[str, Any]
    ]:
        """Execute one environment step.

        SLOT IDENTITY FIX: All dicts are now keyed by vehicle_id (str), not int.
        Actions, observations, rewards, dones use vehicle ID for stable identity
        across timesteps.
        """
        # Reuse last step's observations as prev instead of re-fetching
        self.prev_observations = getattr(self, '_last_observations', self._get_observations())
        self.prev_actions = actions.copy()

        # Snapshot AI vehicles BEFORE the SUMO step — we need this to detect
        # which vehicles complete their trip during this step.
        ai_vehicles_before = set(self.ai_vehicle_ids)
        ai_vehicles = sorted(self.ai_vehicle_ids)

        # Translate and apply actions
        # SLOT IDENTITY FIX: actions dict is now keyed by vehicle_id (str).
        sumo_actions = {}
        ai_set = set(ai_vehicles)
        for key, action in actions.items():
            # Resolve key → vehicle_id (support legacy int keys for backwards compat)
            if isinstance(key, str):
                vehicle_id = key
                if vehicle_id not in ai_set:
                    continue  # Stale key — vehicle no longer present
            elif isinstance(key, int) and key < len(ai_vehicles):
                vehicle_id = ai_vehicles[key]
            else:
                continue

            safe_action = self.safety_monitor.filter_action(
                vehicle_id, action, self.prev_observations.get(vehicle_id, {})
            )
            sumo_actions[vehicle_id] = self.action_translator.translate(
                vehicle_id, safe_action
            )

        # Ensure EVERY tracked vehicle gets an action. If the policy didn't
        # produce an action for a tracked vehicle (e.g., observation failed,
        # slot overflow), give it a neutral acceleration (0.0) so it still
        # receives a setAcceleration call this step. Without this, vehicles
        # without actions would fall through to SUMO default behavior, which
        # could be inconsistent with vehicles that DID get policy actions.
        for veh_id in ai_vehicles:
            if veh_id not in sumo_actions:
                sumo_actions[veh_id] = self.action_translator.translate(
                    veh_id, {
                        'longitudinal': np.array([0.0]),
                        'lateral': np.array([0.0]),
                        'lane_intention': np.array([0.0]),
                    }
                )

        # Apply actions to SUMO
        self._apply_sumo_actions(sumo_actions)

        # Snapshot slot map BEFORE the SUMO step so completed vehicles
        # (which are removed by _update_vehicle_tracking below) can still
        # be located in the trajectory buffer for completion-bonus injection.
        pre_step_slots = dict(self._vehicle_to_slot)
        # Also snapshot ai_vehicle_ids order + adjacency matrix so the
        # κ-hop team-bonus logic (CARL+ branch) can find who was a neighbor
        # of each completer at the moment the completion happened.
        pre_step_ai_list = sorted(self.ai_vehicle_ids)
        pre_step_adj = (
            self.adjacency_matrix.copy()
            if (self.adjacency_matrix is not None and self.adjacency_matrix.size > 0)
            else None
        )

        # Optional eval-video frame capture: request a GUI screenshot for THIS step
        # (sumo-gui writes it during the upcoming simulationStep). GUI-only, best-effort.
        if self._video_dir and self.use_gui:
            try:
                if not self._video_view_set:
                    _vc = os.environ.get('DMPO_VIDEO_CENTER', '').strip()
                    if _vc:
                        cx, cy = (float(v) for v in _vc.split(','))   # e.g. grid center "475,475"
                    else:
                        cx, cy = traci.junction.getPosition('center')
                    hs = self._video_halfspan
                    traci.gui.setBoundary("View #0", cx-hs, cy-hs, cx+hs, cy+hs)
                    try: traci.gui.setSchema("View #0", "real world")
                    except Exception: pass
                    self._video_view_set = True
                traci.gui.screenshot("View #0", os.path.join(
                    self._video_dir, f"frame_{self.current_step:05d}.png"))
            except Exception:
                pass

        # Network signal baseline (gated; None = SUMO's own programs)
        if getattr(self, '_tls_ctrl', None) is not None:
            self._tls_ctrl.step()

        # Step SUMO simulation
        traci.simulationStep()
        self.current_step += 1

        # Update vehicle tracking
        self._update_vehicle_tracking()

        # Detect AI vehicles that completed their trip this step.
        # Use SUMO's getArrivedIDList() which reliably reports vehicles that
        # reached their destination. Count ALL AI arrivals (not just tracked)
        # because the tracking cap untracking vehicles on exit roads right
        # before they complete — making tracked-only detection miss completions.
        #
        # BUG FIX (B10): Match on 'ai' prefix in vehicle ID, not
        # 'ai_vehicle'. Vehicle IDs come from flow IDs in the route file
        # which vary by scenario:
        #   CN+:    flow id="ai_vehicle_north" → vehicle "ai_vehicle_north.3"
        #   2-lane: flow id="ai_north"         → vehicle "ai_north.3"
        # The old filter `startswith('ai_vehicle')` missed ALL completions
        # in scenarios with short flow IDs. Cannot use getTypeID() because
        # SUMO removes arrived vehicles before reporting them.
        arrived = set(traci.simulation.getArrivedIDList())
        completed_vehicles = set(v for v in arrived
                                 if v.startswith('ai') and not v.startswith('ai_human_'))

        # Build per-step cache (one batch of SUMO calls for all vehicles)
        self._build_vehicle_cache()

        # KAPPA-ADV package 1a (gated DMPO_GRAPH_FRESH): the graph built inside
        # _update_vehicle_tracking above read the PREVIOUS step's cache, so newly
        # spawned vehicles were isolated nodes and every edge lagged one step.
        # Rebuild from the fresh cache so adjacency/kappa_reach describe THIS state.
        if os.environ.get('DMPO_GRAPH_FRESH', '0') == '1':
            self._update_communication_graph()

        # Get new observations (uses cache)
        observations = self._get_observations()
        self._last_observations = observations  # Save for next step's prev_observations

        # Compute rewards
        rewards = self._compute_rewards(observations, actions)

        # Check if done
        dones = self._compute_dones()

        # ------------------------------------------------------------------
        # SPARSE REWARD: inject completion bonus for vehicles that finished
        # this step. The reward computer cannot do this because completed
        # vehicles are no longer in observations. We attribute the bonus to
        # the trajectory step whose action triggered the crossing.
        #
        # IMPORTANT: do NOT set dones[veh_id] = True here. The training loop
        # treats `any(dones.values())` as a global episode-end signal, so a
        # per-vehicle done would terminate the entire episode on every
        # completion (causing length=1-30 episodes that crash model
        # ensemble stacking). Per-vehicle done for GAE bootstrapping is
        # marked the next step automatically: the completed vehicle leaves
        # observations, mask_t[slot]=0, and the buffer sets d_t[slot]=True
        # via the padded-slot logic in _build_trajectories.
        # ------------------------------------------------------------------
        sparse_mode = os.environ.get('DMPO_REWARD_SPARSE', '0') == '1'
        carl_plus_mode = os.environ.get('DMPO_REWARD_CARL_PLUS', '0') == '1'
        slots_for_info = dict(self._vehicle_to_slot)
        if sparse_mode:
            # SUMO-detected collisions (this step). Uses true physical
            # collision events, NOT the proximity heuristic in
            # reward_computer._is_collision (which fires thousands of times
            # per episode in dense traffic).
            collision_penalty = float(os.environ.get('DMPO_SPARSE_COLLISION_PENALTY', '0'))
            if collision_penalty != 0:
                try:
                    colliding = set(traci.simulation.getCollidingVehiclesIDList())
                    for veh_id in colliding:
                        if veh_id in rewards:
                            rewards[veh_id] = rewards.get(veh_id, 0.0) - collision_penalty
                except Exception:
                    pass

            if completed_vehicles:
                completion_bonus = float(os.environ.get('DMPO_COMPLETION_BONUS', '50'))
                for veh_id in completed_vehicles:
                    # P0.10: sparse mode now ADDS like CARL+ (was: OVERWRITES).
                    # Consistent semantics between reward paths.
                    rewards[veh_id] = rewards.get(veh_id, 0.0) + completion_bonus
                    # Re-expose the vehicle's slot so the trajectory buffer
                    # can find it during slot_pairs construction. Only added
                    # to the info snapshot — NOT to self._vehicle_to_slot,
                    # which must stay accurate for the next step's free-slot
                    # allocation.
                    if veh_id in pre_step_slots:
                        slots_for_info[veh_id] = pre_step_slots[veh_id]
        elif carl_plus_mode and completed_vehicles:
            # CARL+ (Path G): keep CaRL's dense per-step progress reward
            # (already in `rewards` from reward_computer) but ADD a heavy
            # completion bonus on top, so completing the trip is the
            # dominant return target. Combines the gradient signal of dense
            # rewards with the goal-prioritization of sparse bonuses.
            completion_bonus = float(os.environ.get('DMPO_COMPLETION_BONUS', '200'))
            team_bonus = float(os.environ.get('DMPO_TEAM_BONUS', '5.0'))
            kappa = getattr(self.observation_processor, 'kappa', 2)

            # Precompute κ-hop reachability from pre-step adjacency.
            # kappa_reach[i,j] > 0  ⇔  j is reachable from i in ≤κ hops.
            kappa_reach = None
            if pre_step_adj is not None:
                kappa_reach = pre_step_adj.copy()
                cur = pre_step_adj.copy()
                for _ in range(1, kappa):
                    cur = np.dot(cur, pre_step_adj)
                    kappa_reach = kappa_reach + cur
                _rk = _reward_kappa()            # reward-radius ablation (gated; see _reach)
                if _rk is not None:
                    kappa_reach = _reach(pre_step_adj, _rk)
            pre_step_idx = {vid: i for i, vid in enumerate(pre_step_ai_list)}

            for veh_id in completed_vehicles:
                # Individual completion bonus (unchanged)
                rewards[veh_id] = rewards.get(veh_id, 0.0) + completion_bonus
                if veh_id in pre_step_slots:
                    slots_for_info[veh_id] = pre_step_slots[veh_id]

                # Team bonus to the completer's κ-hop neighbors only —
                # agents that were part of this coordination event. Agents
                # far from the completer (outside the graph) receive nothing.
                if kappa_reach is None or veh_id not in pre_step_idx:
                    continue
                ci = pre_step_idx[veh_id]
                if ci >= kappa_reach.shape[0]:
                    continue
                for j in range(kappa_reach.shape[0]):
                    if j == ci or kappa_reach[ci, j] <= 0:
                        continue
                    if j >= len(pre_step_ai_list):
                        continue
                    nb_vid = pre_step_ai_list[j]
                    if nb_vid in rewards:
                        rewards[nb_vid] = rewards.get(nb_vid, 0.0) + team_bonus

        # Teleport penalty — charged whenever SUMO moves a pinned vehicle to
        # free space. Creates a real cost for getting into pileups so the
        # policy can't coast on "always throttle → SUMO rescues me". Only
        # applied when teleport is active as a safety net; no-ops when
        # DMPO_COLLISION_ACTION!=teleport. Configurable via env var.
        teleport_penalty = float(os.environ.get('DMPO_TELEPORT_PENALTY', '0.0'))
        teleported_ai = set()
        if teleport_penalty != 0.0:
            try:
                starting = set(traci.simulation.getStartingTeleportIDList())
                for veh_id in starting:
                    if veh_id.startswith('ai') and not veh_id.startswith('ai_human_'):
                        teleported_ai.add(veh_id)
                        if veh_id in rewards:
                            rewards[veh_id] = rewards.get(veh_id, 0.0) - teleport_penalty
            except Exception:
                pass

        # ------------------------------------------------------------------
        # AI-only collision handler — SUMO's --collision.action is global, so
        # we set it to "warn" (no SUMO intervention) and apply consequences
        # only to AI vehicles here. Humans are NEVER touched.
        # Behavior modes (set via env vars):
        #   STUCK (default, recommended): setSpeed(0) until leader gap clears,
        #     then release control back to the policy.
        #   REMOVE: traci.vehicle.remove and mark trajectory done.
        #   HALT:   setSpeed(0) for one step (no auto-release).
        # Env vars:
        #   DMPO_COLLISION_PENALTY    (per-AI reward penalty; default 200)
        #   DMPO_COLLISION_REMOVE_AI  (1 = remove on collision; default 0)
        #   DMPO_COLLISION_HALT_AI    (1 = single-step halt; default 0)
        #   DMPO_COLLISION_STUCK_RELEASE_GAP_M (gap to declare path clear; default 5m)
        # If neither REMOVE nor HALT is set, the default STUCK behavior applies.
        # ------------------------------------------------------------------
        ai_collisions_this_step = set()       # all AI in any collision this step (for STUCK/REMOVE/HALT logic)
        ai_collisions_to_penalize = set()      # only AI in NEW collisions (for reward penalty)
        ai_collision_new_pairs_this_step = 0   # count of distinct NEW collision pairs (for eval metrics)
        ai_new_collision_pair_records = []     # list of dicts: details of NEW AI-involving pairs (for diagnostics)
        all_new_collision_pair_records = []    # ALL new pairs (incl. human-human), for verbose audit
        try:
            collision_records = traci.simulation.getCollisions()
            for c in collision_records:
                col_id = getattr(c, 'collider', '') or ''
                vic_id = getattr(c, 'victim', '') or ''
                # Build pair key (sorted to be order-independent)
                pair = tuple(sorted([col_id, vic_id]))
                is_new = pair not in self._collision_pairs_seen
                if is_new:
                    self._collision_pairs_seen.add(pair)
                    col_is_ai = col_id.startswith('ai') and not col_id.startswith('ai_human_')
                    vic_is_ai = vic_id.startswith('ai') and not vic_id.startswith('ai_human_')
                    # Record EVERY new pair (incl. human-human) for verbose audits
                    all_new_collision_pair_records.append({
                        'collider': col_id, 'victim': vic_id,
                        'col_is_ai': col_is_ai, 'vic_is_ai': vic_is_ai,
                        'type': str(getattr(c, 'type', '')),
                        'lane': str(getattr(c, 'lane', '')),
                        'pos': float(getattr(c, 'pos', 0.0)) if hasattr(c, 'pos') else None,
                        'step': self.current_step,
                        'sim_time_s': float(traci.simulation.getTime()),
                        'collider_state': self._get_vehicle_state_for_trace(col_id),
                        'victim_state': self._get_vehicle_state_for_trace(vic_id),
                    })
                    # Subset: only AI-involving — used by eval scripts as the headline metric
                    if col_is_ai or vic_is_ai:
                        ai_collision_new_pairs_this_step += 1
                        ai_new_collision_pair_records.append({
                            'collider': col_id,
                            'victim': vic_id,
                            'col_is_ai': col_is_ai,
                            'vic_is_ai': vic_is_ai,
                            'type': str(getattr(c, 'type', '')),
                            'lane': str(getattr(c, 'lane', '')),
                            'pos': float(getattr(c, 'pos', 0.0)) if hasattr(c, 'pos') else None,
                            'step': self.current_step,
                            'sim_time_s': float(traci.simulation.getTime()),
                            'collider_state': self._get_vehicle_state_for_trace(col_id),
                            'victim_state': self._get_vehicle_state_for_trace(vic_id),
                        })
                for vid in (col_id, vic_id):
                    if vid.startswith('ai') and not vid.startswith('ai_human_'):
                        ai_collisions_this_step.add(vid)
                        if is_new:
                            ai_collisions_to_penalize.add(vid)
        except Exception:
            pass

        # Constrained-RL: when DMPO_CONSTRAINED_RL=1, collisions become a separate
        # cost signal handled by the Lagrangian dual variable in the trainer.
        # We zero out the per-step collision penalty in the reward so cost and
        # reward are cleanly decoupled (no double counting).
        constrained_rl = os.environ.get('DMPO_CONSTRAINED_RL', '0') == '1'

        if ai_collisions_this_step:
            ai_collision_penalty = float(os.environ.get('DMPO_COLLISION_PENALTY', '200.0'))
            if constrained_rl:
                ai_collision_penalty = 0.0  # cost signal handles it instead
            # P1.1 (Issue 4): collision cost shared with κ-hop neighbors,
            # mirroring DMPO_TEAM_BONUS for completions. Without this the
            # collider takes the full hit but its κ-hop teammates feel
            # nothing — asymmetric with the cooperative completion signal.
            # Default 5.0 matches DMPO_TEAM_BONUS. Set to 0 to disable.
            team_cost = float(os.environ.get('DMPO_TEAM_COST', '5.0'))
            if constrained_rl:
                team_cost = 0.0
            ai_remove = os.environ.get('DMPO_COLLISION_REMOVE_AI', '0') == '1'
            ai_halt = os.environ.get('DMPO_COLLISION_HALT_AI', '0') == '1'
            # Distribute team_cost across the κ-hop neighborhood of each new collider.
            # Use the same kappa_reach matrix the per-step averaging consumes.
            kappa_reach = getattr(self, 'kappa_reach', None)
            ai_vehicles_sorted = sorted(self.ai_vehicle_ids)
            ai_idx = {vid: i for i, vid in enumerate(ai_vehicles_sorted)}
            for vid in ai_collisions_this_step:
                # Apply reward penalty ONLY for vehicles in newly detected collision pairs.
                # Repeated overlap of the same pair (vehicles physically stuck inside each
                # other) would otherwise spam the penalty every step.
                if vid in ai_collisions_to_penalize:
                    # P0.1 (Issue 5): apply unconditionally via setdefault.
                    rewards[vid] = rewards.get(vid, 0.0) - ai_collision_penalty
                    # P1.1: spread team_cost to κ-hop neighbors of the collider.
                    if team_cost != 0.0 and kappa_reach is not None and vid in ai_idx:
                        ci = ai_idx[vid]
                        if ci < kappa_reach.shape[0]:
                            for j in range(kappa_reach.shape[0]):
                                if j == ci or kappa_reach[ci, j] <= 0:
                                    continue
                                if j >= len(ai_vehicles_sorted):
                                    continue
                                nb_vid = ai_vehicles_sorted[j]
                                if nb_vid in rewards:
                                    rewards[nb_vid] = rewards.get(nb_vid, 0.0) - team_cost
                if ai_remove:
                    # Remove the AI permanently (trip considered failed)
                    try:
                        traci.vehicle.remove(vid)
                    except Exception:
                        pass
                    # M12-1 exactness: scrub the removed vehicle from per-epoch memos so
                    # the shield's next action-phase query matches legacy live-TraCI.
                    self.observation_processor.on_vehicle_removed(vid)
                    dones[vid] = True
                    self.ai_vehicle_ids.discard(vid)
                    if vid in self._vehicle_to_slot:
                        slot = self._vehicle_to_slot.pop(vid)
                        if self._free_slots is not None:
                            self._free_slots.add(slot)
                else:
                    # Default: STUCK mode — halt the AI and add to tracking set.
                    # Each subsequent step, the release logic below will check
                    # whether the leader has cleared OR the time limit hit, and
                    # release control back to the policy.
                    try:
                        traci.vehicle.setSpeed(vid, 0.0)
                    except Exception:
                        pass
                    if not ai_halt:
                        # Record the step when this AI got stuck
                        self._stuck_ai[vid] = self.current_step

        # Check stuck-AI vehicles each step: release control back to the
        # policy (setSpeed(-1) = no override) when EITHER:
        #   (a) the leader gap is now ≥ stuck_release_gap, OR
        #   (b) the AI has been stuck for ≥ stuck_max_steps (time fallback).
        # This guarantees control always returns to the policy eventually
        # so the policy can keep learning.
        if self._stuck_ai:
            # DMPO_COLLISION_STUCK_REMOVE_ON_TIMEOUT=1: when a stuck AI hits the
            # time fallback (its crashed leader never cleared), REMOVE it instead
            # of releasing it back into the same jam. Gives every collided vehicle
            # a chance to recover (released the moment its path clears), but deletes
            # persistent blockers so dense traffic can't gridlock. Default 0 =
            # legacy behaviour (release on timeout).
            stuck_remove_on_timeout = os.environ.get('DMPO_COLLISION_STUCK_REMOVE_ON_TIMEOUT', '0') == '1'
            to_release = []   # path cleared (or vehicle gone) → return control to policy
            to_remove = []    # timed out under remove-on-timeout → delete the blocker
            for vid, stuck_since in list(self._stuck_ai.items()):
                steps_stuck = self.current_step - stuck_since
                # Time-based fallback after max steps
                if steps_stuck >= self._stuck_max_steps:
                    (to_remove if stuck_remove_on_timeout else to_release).append(vid)
                    continue
                try:
                    leader = traci.vehicle.getLeader(vid, dist=20.0)
                    # leader is None if no vehicle within dist on the same lane,
                    # or (leader_id, gap) tuple. Path clear = no leader OR gap >= threshold.
                    if leader is None or leader[1] >= self._stuck_release_gap:
                        to_release.append(vid)   # recovered → keep driving
                    else:
                        traci.vehicle.setSpeed(vid, 0.0)
                except Exception:
                    # Vehicle no longer exists — clean up
                    to_release.append(vid)
            for vid in to_release:
                self._stuck_ai.pop(vid, None)
                try:
                    # Release SUMO speed override; policy regains control.
                    traci.vehicle.setSpeed(vid, -1)
                except Exception:
                    pass
            for vid in to_remove:
                # Persistent blocker — penalty was already applied on collision;
                # now delete it (trip failed) and free its slot so the lane clears.
                self._stuck_ai.pop(vid, None)
                try:
                    traci.vehicle.remove(vid)
                except Exception:
                    pass
                self.observation_processor.on_vehicle_removed(vid)  # M12-1 memo scrub
                dones[vid] = True
                self.ai_vehicle_ids.discard(vid)
                if vid in self._vehicle_to_slot:
                    slot = self._vehicle_to_slot.pop(vid)
                    if self._free_slots is not None:
                        self._free_slots.add(slot)

        # ------------------------------------------------------------------
        # Clear-stopped (DMPO_CLEAR_STOPPED_STEPS): remove AI that have been
        # frozen (speed < thresh) for too long even WITHOUT colliding. In dense
        # all-AI the policy can learn to freeze (collision-safe, throughput-zero);
        # frozen non-collided vehicles escape the collision-clearing and ACCUMULATE
        # → episodes collapse to 1 step → learning stops. Removing them (with a
        # penalty) prevents the collapse AND makes freezing locally costly.
        # ------------------------------------------------------------------
        _clear_stopped_n = int(os.environ.get('DMPO_CLEAR_STOPPED_STEPS', '0'))
        if _clear_stopped_n > 0 and self.ai_vehicle_ids:
            _stop_thresh = float(os.environ.get('DMPO_CLEAR_STOPPED_SPEED', '0.3'))
            _stop_pen = float(os.environ.get('DMPO_CLEAR_STOPPED_PENALTY', '50.0'))
            _to_clear = []
            for vid in sorted(self.ai_vehicle_ids):
                cached = self._vehicle_cache.get(vid)
                spd = cached.get('speed', 0.0) if cached is not None else 0.0
                self._stopped_steps[vid] = (self._stopped_steps.get(vid, 0) + 1) if spd < _stop_thresh else 0
                if self._stopped_steps[vid] >= _clear_stopped_n:
                    _to_clear.append(vid)
            for vid in _to_clear:
                rewards[vid] = rewards.get(vid, 0.0) - _stop_pen
                dones[vid] = True
                self._stopped_steps.pop(vid, None)
                try:
                    traci.vehicle.remove(vid)
                except Exception:
                    pass
                self.observation_processor.on_vehicle_removed(vid)  # M12-1 memo scrub
                self.ai_vehicle_ids.discard(vid)
                if vid in self._vehicle_to_slot:
                    slot = self._vehicle_to_slot.pop(vid)
                    if self._free_slots is not None:
                        self._free_slots.add(slot)

        # ------------------------------------------------------------------
        # Gridlock detector — terminate the episode when no AI has completed
        # in a long window and the AI fleet is essentially motionless. This
        # prevents wasted compute on stuck simulations and gives the policy a
        # clean negative-progress signal in place of SUMO's silent teleport.
        # ------------------------------------------------------------------
        if self._gridlock_enabled:
            if completed_vehicles:
                self._steps_since_completion = 0
            else:
                self._steps_since_completion += 1
            mean_ai_speed = 0.0
            if self.ai_vehicle_ids:
                speeds = []
                for vid in sorted(self.ai_vehicle_ids):
                    cached = self._vehicle_cache.get(vid)
                    if cached is not None:
                        speeds.append(cached.get('speed', 0.0))
                if speeds:
                    mean_ai_speed = float(np.mean(speeds))
            if (self._steps_since_completion >= self._gridlock_no_comp_thresh and
                    mean_ai_speed < self._gridlock_speed_thresh):
                self._gridlock_triggered = True
                # Mark every active AI as done so the trajectory loop terminates
                for vid in sorted(self.ai_vehicle_ids):
                    dones[vid] = True
                # Apply a one-time gridlock penalty to all active AI rewards
                gridlock_pen = float(os.environ.get('DMPO_GRIDLOCK_PENALTY', '100.0'))
                for vid in sorted(self.ai_vehicle_ids):
                    rewards[vid] = rewards.get(vid, 0.0) - gridlock_pen
        else:
            mean_ai_speed = 0.0

        # Overload termination: the intersection accumulated more AI than the
        # network cap (max_ai_vehicles), so the excess would be RELEASED to Krauss
        # (uncontrolled, outside the training batch). Instead terminate the episode
        # — overload is a failure — so EVERY vehicle stays policy-controlled and the
        # batch can never blow past the cap. Gated by the same DMPO_GRIDLOCK_DETECT.
        if self._gridlock_enabled and getattr(self, '_overload_triggered', False):
            self._gridlock_triggered = True
            gridlock_pen = float(os.environ.get('DMPO_GRIDLOCK_PENALTY', '100.0'))
            for vid in sorted(self.ai_vehicle_ids):
                dones[vid] = True
                rewards[vid] = rewards.get(vid, 0.0) - gridlock_pen

        # Constrained-RL cost signal: per-AI binary collision indicator this step.
        # Trainer treats `costs[vid] = 1.0` as the safety violation count, separate
        # from the task reward. Only vehicles in NEW collision pairs are flagged
        # (matches the de-duplication used for reward penalty).
        ai_costs_this_step = {
            vid: (1.0 if vid in ai_collisions_to_penalize else 0.0)
            for vid in sorted(self.ai_vehicle_ids)
        }

        # P0.1: accumulate the real collision counters PER EPISODE so the
        # training loop can read them from info instead of relying on the stale
        # safety_monitor.collision_count which was never incremented in training.
        self._ep_ai_collision_pairs += ai_collision_new_pairs_this_step
        # Optional per-mechanism breakdown when collision-trace is enabled.
        if self._collision_trace_enabled and ai_new_collision_pair_records:
            for rec in ai_new_collision_pair_records:
                col_is_ai = rec.get('col_is_ai', False)
                vic_is_ai = rec.get('vic_is_ai', False)
                if col_is_ai and vic_is_ai:
                    pair_kind = 'AI-AI'
                elif col_is_ai != vic_is_ai:
                    pair_kind = ('AI-human(AI_is_collider)' if col_is_ai
                                 else 'AI-human(AI_is_victim)')
                else:
                    pair_kind = 'human-human'
                self._ep_ai_collision_breakdown[pair_kind] = (
                    self._ep_ai_collision_breakdown.get(pair_kind, 0) + 1)
                # Mechanism: rear-end vs junction (cheap heuristic without full classifier)
                lane = rec.get('lane', '') or ''
                mech = 'cross_traffic_junction' if lane.startswith(':') else 'rear_end_same_lane'
                self._ep_mechanism_breakdown[mech] = (
                    self._ep_mechanism_breakdown.get(mech, 0) + 1)

        # Create info dict
        info = {
            'step': self.current_step,
            'n_vehicles': len(self.all_vehicle_ids),
            'n_ai_vehicles': len(self.ai_vehicle_ids),
            'collisions': self.safety_monitor.get_collision_count(),  # legacy stale (kept for backcompat)
            'intersection_throughput': self._get_intersection_throughput(),
            'gridlock_triggered': self._gridlock_triggered,
            'mean_ai_speed': mean_ai_speed,
            'steps_since_completion': self._steps_since_completion,
            # Defensive copy: consumers mutating these would corrupt env internals
            'ai_collisions_this_step': set(ai_collisions_this_step),
            'ai_collision_new_pairs_this_step': ai_collision_new_pairs_this_step,  # for eval metrics
            'ai_new_collision_pair_records': list(ai_new_collision_pair_records),  # for diagnostics
            'all_new_collision_pair_records': list(all_new_collision_pair_records),  # for verbose audit (incl. human-human)
            # P0.1: REAL per-episode AI collision counter (replaces stale Safety dict).
            'ai_collisions_episode': self._ep_ai_collision_pairs,
            'ai_collision_breakdown_episode': dict(self._ep_ai_collision_breakdown),
            'ai_mechanism_breakdown_episode': dict(self._ep_mechanism_breakdown),
            'ai_costs_this_step': dict(ai_costs_this_step),
            # SLOT IDENTITY FIX: Set of vehicle IDs that completed their trip
            # this step. The training loop uses this to mark per-vehicle done=True
            # in the trajectory buffer for proper GAE bootstrapping.
            'completed_vehicles': completed_vehicles,
            # SLOT IDENTITY FIX: vehicle_id → slot index mapping (stable across
            # vehicle's lifetime). The trajectory buffer uses this to place each
            # vehicle's data in its assigned slot. In sparse mode, completed
            # vehicles are re-included with their pre-step slots so their
            # completion bonus is attributed correctly.
            'vehicle_slots': slots_for_info,
            'teleported_ai': teleported_ai if teleport_penalty != 0.0 else set(),
        }

        # DIAGNOSTIC: per-step wall-clock + speed trace (no-op unless DMPO_STEP_TRACE=1).
        # Distinguishes a frozen GUI caused by training-update PAUSES (large wall-clock
        # gap before the NEXT step) from a genuine vehicle FREEZE / gridlock (steps
        # keep firing but mean_ai_speed ≈ 0). Prints to stderr so it never pollutes
        # the metrics parsed from stdout.
        if os.environ.get('DMPO_STEP_TRACE', '0') == '1':
            import sys as _sys, time as _t
            _spd = 0.0
            if self.ai_vehicle_ids:
                _ss = [self._vehicle_cache.get(v, {}).get('speed', 0.0)
                       for v in self.ai_vehicle_ids if self._vehicle_cache.get(v)]
                if _ss:
                    _spd = float(np.mean(_ss))
            print(f"STEPTRACE wall={_t.time():.3f} step={self.current_step} "
                  f"n_ai={len(self.ai_vehicle_ids)} mean_speed={_spd:.3f} "
                  f"completed={len(completed_vehicles)}", file=_sys.stderr, flush=True)

        return observations, rewards, dones, info

    def _get_vehicle_state_for_trace(self, vid):
        """Build a serializable snapshot + history for collision diagnostics.

        Returns None when trace is disabled or the vehicle has no cache entry.
        Used by collision record enrichment so the eval script can classify
        the collision mechanism (rear-end, lane-change, cross-traffic, etc.).
        """
        if not self._collision_trace_enabled:
            return None
        st = self._vehicle_cache.get(vid, {})
        if not st:
            return {'id': vid, 'is_ai': vid.startswith('ai') and not vid.startswith('ai_human_'),
                    'note': 'no cache entry'}
        pos = st.get('position', (0.0, 0.0))
        result = {
            'id': vid,
            'is_ai': vid.startswith('ai') and not vid.startswith('ai_human_'),
            'speed_mps': float(st.get('speed', 0.0)),
            'acceleration_mps2': float(st.get('acceleration', 0.0)),
            'edge': st.get('edge_id', ''),
            'lane_id': st.get('lane_id', ''),
            'lane_position_m': float(st.get('lane_position', 0.0)),
            'pos_x': float(pos[0]),
            'pos_y': float(pos[1]),
            'angle_deg': float(st.get('angle', 0.0)),
            'type_id': st.get('type_id', ''),
        }
        hist = self._vehicle_state_history.get(vid)
        if hist:
            speeds = [float(h['speed']) for h in hist]
            lanes = [h['lane_id'] for h in hist]
            edges = [h['edge_id'] for h in hist]
            result['history_len'] = len(hist)
            result['speed_history'] = speeds
            result['lane_id_history'] = lanes
            result['edge_id_history'] = edges
            result['recent_lane_change'] = (len(set(lanes)) > 1)
            result['recent_edge_change'] = (len(set(edges)) > 1)
            result['stationary_steps_in_last_10'] = sum(1 for s in speeds if s < 0.5)
            result['max_speed_in_last_10'] = max(speeds) if speeds else 0.0
            result['mean_speed_in_last_10'] = sum(speeds)/len(speeds) if speeds else 0.0
        return result

    def _sensed_cache(self):
        """Perceived vehicle states for the policy (sensing robustness).
        Per step and per vehicle, independently: position += N(0, σp²) on x and y
        (DMPO_SENSE_POS_SIGMA, metres), speed = max(0, speed + N(0, σv²))
        (DMPO_SENSE_SPD_SIGMA, m/s); with probability DMPO_SENSE_DROP the report is lost
        and the vehicle's LAST received (noisy) state is held. Seeded by
        DMPO_SENSE_SEED (default: the SUMO seed) so a sweep cell is reproducible.
        All zero → returns the true cache object itself (no copy, no RNG draw)."""
        sp = float(os.environ.get('DMPO_SENSE_POS_SIGMA', '0') or 0)
        sv = float(os.environ.get('DMPO_SENSE_SPD_SIGMA', '0') or 0)
        pd = float(os.environ.get('DMPO_SENSE_DROP', '0') or 0)
        if sp == 0 and sv == 0 and pd == 0:
            return self._vehicle_cache
        if getattr(self, '_sense_rng', None) is None:
            seed = int(os.environ.get('DMPO_SENSE_SEED', os.environ.get('DMPO_SUMO_SEED', '0')) or 0)
            self._sense_rng = np.random.default_rng(seed)
            self._sense_last = {}
            print(f"📡 SENSING NOISE: pos σ={sp} m, speed σ={sv} m/s, drop p={pd} (seed {seed})")
        rng, last = self._sense_rng, self._sense_last
        out = {}
        for vid, st in (self._vehicle_cache or {}).items():
            if pd > 0 and vid in last and rng.random() < pd:
                held = dict(st)
                held['position'], held['speed'] = last[vid]
                out[vid] = held
                continue
            d = dict(st)
            pos = st.get('position')
            if pos is not None and sp > 0:
                d['position'] = (pos[0] + rng.normal(0, sp), pos[1] + rng.normal(0, sp))
            spd = st.get('speed')
            if spd is not None and sv > 0:
                d['speed'] = max(0.0, spd + rng.normal(0, sv))
            last[vid] = (d.get('position'), d.get('speed'))
            out[vid] = d
        for vid in list(last):
            if vid not in out:
                del last[vid]
        return out

    def _build_vehicle_cache(self):
        """Batch-fetch vehicle states from SUMO into a per-step cache.

        One SUMO API call per attribute per vehicle, done once per step.
        All downstream code (observations, rewards, communication graph,
        safety monitor) reads from cache instead of making individual calls.
        """
        if self._vehicle_cache_step == self.current_step:
            return  # Already cached this step

        cache = {}
        for veh_id in self.all_vehicle_ids:
            try:
                lane_id = traci.vehicle.getLaneID(veh_id)
                cache[veh_id] = {
                    'position': traci.vehicle.getPosition(veh_id),
                    'speed': traci.vehicle.getSpeed(veh_id),
                    'acceleration': traci.vehicle.getAcceleration(veh_id),
                    'angle': traci.vehicle.getAngle(veh_id),
                    'lane_id': lane_id,
                    'type_id': traci.vehicle.getTypeID(veh_id),
                    'lane_position': traci.vehicle.getLanePosition(veh_id),
                    'edge_id': traci.lane.getEdgeID(lane_id) if lane_id else '',
                }
                if self._obs_v2:
                    # OBS-V2 (spec Step 5, additive; gated so legacy wall-time is
                    # untouched): per-vehicle length (rear-bumper + occupancy windows),
                    # lane_index (lane_frac), prev_speed (nb_dv temporal dim).
                    cache[veh_id]['length'] = traci.vehicle.getLength(veh_id)
                    cache[veh_id]['lane_index'] = traci.vehicle.getLaneIndex(veh_id)
                    _old = (self._vehicle_cache or {}).get(veh_id)
                    cache[veh_id]['prev_speed'] = (
                        _old['speed'] if _old is not None else cache[veh_id]['speed'])
                    if self._profile_obs_on:
                        # ARM-B lifetime stats. accel from the cache's own
                        # prev_speed (first frame: accel 0 by construction).
                        _st = self._veh_profiles.get(veh_id)
                        if _st is None:
                            _st = self._veh_profiles[veh_id] = [0, 0.0, 0.0, 0]
                        _v = cache[veh_id]['speed']
                        _a = (_v - cache[veh_id]['prev_speed']) / max(self.step_length, 1e-6)
                        _st[0] += 1
                        _st[1] += _v
                        _st[2] += abs(_a)
                        if _a < -0.5:
                            _st[3] += 1
            except traci.exceptions.TraCIException:
                continue
        if self._profile_obs_on and len(self._veh_profiles) > 2 * max(len(cache), 8):
            # prune long-departed vehicles (bounded memory on long streams).
            # Conservative superset (verify-region-2): keep anything still ALIVE
            # per getIDList even if this build's per-vehicle try skipped it — a
            # transient TraCIException must not zero a living vehicle's history.
            _alive = set(getattr(self, 'all_vehicle_ids', ()) or ())
            self._veh_profiles = {k: v for k, v in self._veh_profiles.items()
                                  if k in cache or k in _alive}
        self._vehicle_cache = cache
        self._vehicle_cache_step = self.current_step
        self._cache_epoch += 1   # M12-0: monotone; never reset (survives soft/hard resets)

        # Append to per-vehicle history for collision-trace diagnostics.
        # Cheap when disabled (single bool check); deque keeps last 10 steps.
        if self._collision_trace_enabled:
            from collections import deque
            for vid, state in cache.items():
                if vid not in self._vehicle_state_history:
                    self._vehicle_state_history[vid] = deque(maxlen=10)
                self._vehicle_state_history[vid].append({
                    'step': self.current_step,
                    'speed': state.get('speed', 0.0),
                    'lane_id': state.get('lane_id', ''),
                    'edge_id': state.get('edge_id', ''),
                    'lane_position': state.get('lane_position', 0.0),
                    'angle': state.get('angle', 0.0),
                })

        # FOREFRONT MARKING: identify the lead vehicle on each approach lane.
        # The forefront vehicle is the one closest to the intersection (highest
        # lane_position) on each *_to_center lane. Only this vehicle can block
        # the entire lane — it's the gatekeeper.
        self._forefront_vehicles = {}   # vehicle_id → n_followers
        self._forefront_tls_state = {}  # vehicle_id → traffic light state ('r'/'g'/'G'/...)
        approach_lanes = {}  # lane_id → list of (lane_position, veh_id)
        for veh_id, state in cache.items():
            lane_id = state.get('lane_id', '')
            if 'to_center' in lane_id and not lane_id.startswith(':'):
                lane_pos = state.get('lane_position', 0.0)
                approach_lanes.setdefault(lane_id, []).append((lane_pos, veh_id))

        for lane_id, vehicles in approach_lanes.items():
            if not vehicles:
                continue
            # Vehicle with highest lane_position is at the front
            vehicles.sort(key=lambda x: x[0], reverse=True)
            front_pos, front_veh = vehicles[0]
            n_followers = len(vehicles) - 1
            self._forefront_vehicles[front_veh] = n_followers

            # Get traffic light state for this forefront vehicle
            try:
                next_tls = traci.vehicle.getNextTLS(front_veh)
                if next_tls:
                    self._forefront_tls_state[front_veh] = next_tls[0][3]
            except Exception:
                pass

    def get_vehicle_state(self, veh_id):
        """Get cached vehicle state. Returns None if vehicle not in cache."""
        return self._vehicle_cache.get(veh_id)

    def _update_vehicle_tracking(self):
        """Update tracking of AI and human vehicles with smart untracking.

        FIX A1+A2 (RF16, RF17): Two changes from the previous implementation:
          1. Vehicles are NEVER deleted via traci.vehicle.remove. Instead,
             excess AI vehicles are 'untracked' — removed from self.ai_vehicle_ids
             so the policy doesn't see them, but they keep driving in SUMO as
             background traffic. This preserves their interaction with other
             vehicles and their natural exit flow.
          2. Untracking priority: PREFER to untrack vehicles that have already
             passed the intersection (on center_to_* exit lanes) and are far
             from any other vehicle. NEVER untrack vehicles in the junction
             or approaching it from a *_to_center lane.
          3. Intersection center is now sourced correctly from SUMO directly
             rather than using a hardcoded (600, 600) fallback.
        """
        current_vehicles = set(traci.vehicle.getIDList())

        # Categorize vehicles by type
        ai_candidates = set()
        new_human_vehicles = set()

        for veh_id in current_vehicles:
            veh_type = traci.vehicle.getTypeID(veh_id)
            if veh_type.startswith('ai_vehicle'):
                ai_candidates.add(veh_id)
            else:
                new_human_vehicles.add(veh_id)

        # Smart untracking: only when AI count exceeds the cap
        new_ai_vehicles = ai_candidates
        if self.max_ai_vehicles > 0 and len(ai_candidates) > self.max_ai_vehicles:
            # Overload: more AI than the cap. Flag it so step() terminates the
            # episode (overload = failure) instead of leaving the excess on Krauss —
            # this keeps every vehicle policy-controlled. The release below still
            # runs for this single step (the network can't take >cap agents); the
            # episode then ends, so exposure is ≤1 step.
            self._overload_triggered = True
            if os.environ.get('DMPO_ACTION_DETAIL', '0') == '1':
                print(f"⚠️ OVERLOAD→TERMINATE: {len(ai_candidates) - self.max_ai_vehicles} of {len(ai_candidates)} AI exceed cap {self.max_ai_vehicles} (step {self.current_step})")
            # Get correct intersection center (A1 fix for RF16)
            cx_cy = self.observation_processor.intersection_center
            if cx_cy is None:
                try:
                    cx_cy = traci.junction.getPosition('center')
                except Exception:
                    cx_cy = (0.0, 0.0)
            cx, cy = cx_cy

            # Classify each AI vehicle by tracking priority:
            #   priority 0 (highest): in junction or approaching intersection
            #   priority 1: on exit lane but close to intersection (still relevant)
            #   priority 2 (lowest): on exit lane and far from intersection — safe to untrack
            classified = []  # list of (priority, distance_to_center, veh_id)
            for veh_id in ai_candidates:
                try:
                    lane_id = traci.vehicle.getLaneID(veh_id)
                    pos = traci.vehicle.getPosition(veh_id)
                    dx = pos[0] - cx
                    dy = pos[1] - cy
                    dist_sq = dx * dx + dy * dy
                except Exception:
                    classified.append((2, float('inf'), veh_id))
                    continue

                # Junction lanes start with ':'
                if lane_id.startswith(':'):
                    priority = 0  # in junction — must keep
                elif 'to_center' in lane_id:
                    priority = 0  # approaching — must keep
                elif 'center_to_' in lane_id:
                    # Past intersection. Prefer to untrack if far from center.
                    if dist_sq > 50.0 * 50.0:
                        priority = 2  # far past — safe to untrack
                    else:
                        priority = 1  # just exited — keep tracking briefly
                else:
                    priority = 1  # unknown lane state — middle priority

                classified.append((priority, dist_sq, veh_id))

            # Sort: priority 0 first (keep), then by distance ascending within
            # same priority. The first max_ai_vehicles are kept.
            # Vehicles past the intersection and far away are at the end → untracked first.
            classified.sort(key=lambda x: (x[0], x[1]))
            keep_list = classified[:self.max_ai_vehicles]
            new_ai_vehicles = set(v for _, _, v in keep_list)
            # Untracked AI vehicles stay in SUMO — they become background traffic.
            # They are NOT in self.ai_vehicle_ids, so the policy doesn't see them
            # and they don't get padded slots. They DO appear in self.all_vehicle_ids
            # so other vehicles still need to avoid them.

        # BUG FIX (B11): Release SUMO control for newly UNTRACKED vehicles.
        # With setAcceleration (auto-expiring), we only need to restore
        # setSpeedMode(31) and release any lingering setSpeed command.
        newly_untracked = self._prev_ai_veh_set - new_ai_vehicles
        for veh_id in newly_untracked:
            try:
                traci.vehicle.setSpeed(veh_id, -1)       # release to SUMO
                traci.vehicle.setSpeedMode(veh_id, 31)    # restore defaults
            except Exception:
                pass  # vehicle may have left simulation

        self.ai_vehicle_ids = new_ai_vehicles
        self.human_vehicle_ids = new_human_vehicles
        self.all_vehicle_ids = current_vehicles
        self.n_agents = len(self.ai_vehicle_ids)

        # setSpeedMode for newly tracked AI vehicles
        # Mode 6 = bits 1+2: no Krauss, no traffic lights (pure policy control)
        # Mode 7 = bits 0+1+2: Krauss ON (prevents overlaps), no traffic lights
        # Configurable via DMPO_SPEED_MODE env var for A/B testing.
        _speed_mode = int(os.environ.get('DMPO_SPEED_MODE', '6'))
        newly_tracked = new_ai_vehicles - self._prev_ai_veh_set
        _lc_mode = os.environ.get('DMPO_SET_LC_MODE', '')   # e.g. 0 = SUMO LC model fully off
        # DMPO_IGNORE_JUNCTION_ROW=1: AI vehicles ignore SUMO's junction right-of-way
        # (jmIgnoreFoeProb=1, jmCrossingGap=0) — SUMO no longer holds them at the
        # stop line for right_before_left. Full junction control passes to the policy
        # (no SUMO safety net → policy must coordinate). Removes the inherited
        # right-before-left deadlock + the stopped-at-junction freeze.
        _ignore_row = os.environ.get('DMPO_IGNORE_JUNCTION_ROW', '0') == '1'
        for veh_id in newly_tracked:
            try:
                traci.vehicle.setSpeedMode(veh_id, _speed_mode)
                if _lc_mode != '':
                    traci.vehicle.setLaneChangeMode(veh_id, int(_lc_mode))
                if _ignore_row:
                    traci.vehicle.setParameter(veh_id, "junctionModel.jmIgnoreFoeProb", "1")
                    traci.vehicle.setParameter(veh_id, "junctionModel.jmIgnoreFoeSpeed", "100")
                    traci.vehicle.setParameter(veh_id, "junctionModel.jmCrossingGap", "0")
                # Lateral physics constraints — read DMPO_MAX_LAT_VELOCITY so
                # SUMO's cap matches the action_translator's max commanded
                # lateral velocity. Default 1.0 m/s (no behavior change for
                # existing variants).
                _max_lat_v = float(os.environ.get('DMPO_MAX_LAT_VELOCITY', '1.0'))
                traci.vehicle.setMaxSpeedLat(veh_id, _max_lat_v)
                traci.vehicle.setMinGapLat(veh_id, 0.5)
            except Exception:
                pass
        self._prev_ai_veh_set = new_ai_vehicles.copy()

        # SLOT IDENTITY (B1 proper): maintain persistent vehicle → slot mapping.
        # 1. Free slots for vehicles that have left
        # 2. Assign new slots (lowest free) to newly-tracked vehicles
        # H6 FIX: cap=0 used to silently mean 8 SLOTS (a lie: docs said "no cap") and the
        # 9th+ AI became permanent ghosts (speedMode 6, never released). In DYNAMIC mode
        # slots are only identities (the net is per-vehicle), so cap=0 now grows slots
        # on demand = truly uncapped. Static mode keeps 8 but SCREAMS instead of lying.
        _uncapped_dynamic = (self.max_ai_vehicles <= 0
                             and os.environ.get('DMPO_DYNAMIC', '0') == '1')
        n_slots = self.max_ai_vehicles if self.max_ai_vehicles > 0 else 8
        if self._free_slots is None:
            self._free_slots = set(range(n_slots))
            self._slot_high = n_slots

        # Free slots whose vehicle is no longer tracked
        gone = [v for v in self._vehicle_to_slot if v not in new_ai_vehicles]
        for v in gone:
            slot = self._vehicle_to_slot.pop(v)
            self._free_slots.add(slot)

        # Assign slots to new vehicles (lowest free slot first)
        new_vehicles = sorted(new_ai_vehicles - set(self._vehicle_to_slot.keys()))
        for v in new_vehicles:
            if not self._free_slots:
                if _uncapped_dynamic:
                    # H6 FIX: grow the slot pool — cap=0 means NO cap in dynamic mode.
                    slot = self._slot_high
                    self._slot_high += 1
                    self._vehicle_to_slot[v] = slot
                    continue
                # Static-mode overflow: untrack — but RELEASE the vehicle back to full
                # SUMO safety first (it already got speedMode 6 at spawn; without this
                # it became a permanent junction-blind ghost), and say so loudly ONCE.
                try:
                    traci.vehicle.setSpeed(v, -1)
                    traci.vehicle.setSpeedMode(v, 31)
                except Exception:
                    pass
                if not getattr(self, '_slot_overflow_warned', False):
                    print(f"⚠️  SLOT OVERFLOW: >{n_slots} concurrent AI with cap="
                          f"{self.max_ai_vehicles} (static mode) — overflow vehicles are "
                          f"released to SUMO (mode 31) and NOT policy-controlled. "
                          f"Raise --max-ai-vehicles.", flush=True)
                    self._slot_overflow_warned = True
                self.ai_vehicle_ids.discard(v)
                continue
            slot = min(self._free_slots)
            self._free_slots.remove(slot)
            self._vehicle_to_slot[v] = slot

        # Update communication graph
        self._update_communication_graph()
    
    def _update_communication_graph(self):
        """Update adjacency matrix using sparse road-topology neighbors.

        Each agent connects to ~4-6 relevant neighbors (constant, O(n)):
          - Same lane: nearest 1 ahead + nearest 1 behind (leader-follower)
          - Adjacent lanes: nearest 1 per adjacent lane on same edge
          - Conflicting approach: nearest 1 per conflicting direction near intersection
        """
        if self.n_agents == 0:
            self.adjacency_matrix = np.array([])
            return

        ai_vehicles = sorted(self.ai_vehicle_ids)
        n = len(ai_vehicles)
        adj_matrix = np.zeros((n, n), dtype=np.float32)

        # Gather vehicle info from cache
        info = []  # (lane_id, edge_id, lane_position, position_xy)
        for veh_id in ai_vehicles:
            state = self._vehicle_cache.get(veh_id)
            if state:
                info.append((
                    state.get('lane_id', ''),
                    state.get('edge_id', ''),
                    state.get('lane_position', 0.0),
                    state.get('position', (0, 0)),
                ))
            else:
                # P0.26: track silent cache misses so the env doesn't appear to
                # work when a tracked AI is missing from _vehicle_cache. Without
                # this, the AI gets a synthetic ('', '', 0.0, (0, 0)) and becomes
                # functionally isolated in the adjacency matrix.
                self.cache_miss_warnings = getattr(self, 'cache_miss_warnings', 0) + 1
                info.append(('', '', 0.0, (0, 0)))

        # Determine intersection center.
        # P0.25: previously defaulted to (600, 600) hardcoded, which is wrong
        # for 1lane scenarios centered at (0,0). Now defer to the observation
        # processor's auto-detection (which falls back to (0,0) with a warning
        # via P0.22) so the conflict-zone gate matches the actual junction.
        # C1 FIX: that auto-detection ran pre-connect → stale (0,0); re-detect
        # once now that traci is live (gated DMPO_CENTER_REDETECT, default on).
        if (not getattr(self, '_center_redetect_done', False)
                and os.environ.get('DMPO_CENTER_REDETECT', '1') == '1'):
            self.observation_processor.redetect_intersection_center()
            self._center_redetect_done = True
        center = self.observation_processor.intersection_center
        cx, cy = (center if center else (0.0, 0.0))

        # C1 FIX: net-generic junction-topology conflict edges (replaces the
        # 'to_center' string + dist-to-center gate that NEVER fired). Two
        # vehicles are conflict-linked iff their next junction is the SAME
        # (approach edges via the obs NJ map; internal edges ':jxn_x' parse to
        # their junction, closing the in-junction blind spot F4b) and they are
        # within communication_radius of EACH OTHER. Gated (consistency-critical).
        topo_edges = os.environ.get('DMPO_TOPO_CONFLICT_EDGES', '0') == '1'
        _edge_jid_cache = {}

        def _edge_jid(edge):
            if edge in _edge_jid_cache:
                return _edge_jid_cache[edge]
            jid = None
            if edge.startswith(':'):
                jid = edge[1:].rsplit('_', 1)[0]
            else:
                try:
                    self.observation_processor._ensure_nj_maps()
                    rec = self.observation_processor._edge_nextjxn.get(edge)
                    jid = rec[0] if rec else None
                except Exception:
                    jid = None
            _edge_jid_cache[edge] = jid
            return jid

        self._cat3_edge_count = 0

        # ── M12-4: bucketed early-exit pair scan (gated DMPO_GRAPH_FAST, default on;
        # topo mode only — the legacy 'to_center' Cat-3 gate ranks by distance-to-center
        # and keeps the original full scan). The legacy if/elif/elif chain means each
        # ordered pair (i, j) contributes to AT MOST ONE category, priority
        # Cat1 > Cat2 > Cat3. Truth table reproduced by the buckets:
        #   Cat 1  lane_i == lane_j != ''                → by_lane[lane_i]
        #   Cat 2  edge_i == edge_j, != '', not internal → by_edge[edge_i], skipping
        #          lane_j == lane_i (exactly Cat 1's claim: lane=='' ⟹ edge=='' at the
        #          cache build, so by_edge members always have lane != '')
        #   Cat 3  edge_i != edge_j != '', same next junction → by_jid[jx_i], skipping
        #          edge_j == edge_i (subsumes Cat 1 — same lane ⟹ same edge — Cat 2,
        #          and the `elif edge_i != edge_j` guard). edge_i == '' ⟹ jx_i is None
        #          ⟹ no Cat-3 edges (legacy guard); internal ':jxn_k' edges ARE
        #          bucketed (legacy _edge_jid parses them too).
        # Buckets hold ascending j-indices, so every first-visited-wins tie-break
        # (strict < / not-in-or-<) resolves exactly as the legacy `for j in range(n)`.
        # DMPO_GRAPH_ASSERT=<K>: for steps <= K, also run the legacy scan per row and
        # hard-fail on ANY mismatch (mandatory in smokes for this item).
        graph_fast = (topo_edges
                      and os.environ.get('DMPO_GRAPH_FAST', '1') == '1'
                      and os.environ.get('DMPO_PERF_FIXES', '1') == '1')
        if graph_fast:
            by_lane, by_edge, by_jid = {}, {}, {}
            for _idx, (lane_k, edge_k, _lpk, _pk) in enumerate(info):
                if lane_k:
                    by_lane.setdefault(lane_k, []).append(_idx)
                if edge_k and not edge_k.startswith(':'):
                    by_edge.setdefault(edge_k, []).append(_idx)
                if edge_k:
                    _jk = _edge_jid(edge_k)
                    if _jk is not None:
                        by_jid.setdefault(_jk, []).append(_idx)

        def _scan_legacy(i):
            """Verbatim legacy O(n) inner scan for row i (also the assert reference)."""
            lane_i, edge_i, lpos_i, pos_i = info[i]

            best_ahead = (-1, float('inf'))   # (index, distance)
            best_behind = (-1, float('inf'))
            best_adjacent = {}  # lane_id -> (index, distance)
            best_conflict = {}  # edge_id -> (index, distance_to_intersection)

            for j in range(n):
                if i == j:
                    continue
                lane_j, edge_j, lpos_j, pos_j = info[j]

                # Category 1: Same lane — nearest ahead and behind.
                # P0.27: gap == 0 (identical lane_position, i.e., physically
                # overlapping AIs) is treated as "ahead" so the edge still
                # forms. Previously the strict >0 / <0 check left identical-
                # position AIs disconnected, AND the elif chain blocked Cat 2
                # from rescuing them.
                if lane_i == lane_j and lane_i != '':
                    gap = lpos_j - lpos_i
                    if gap >= 0 and gap < best_ahead[1]:  # j is ahead (or coincident)
                        best_ahead = (j, gap)
                    elif gap < 0 and abs(gap) < best_behind[1]:  # j is behind
                        best_behind = (j, abs(gap))

                # Category 2: Same edge, different lane — nearest by lane position
                elif edge_i == edge_j and edge_i != '' and not edge_i.startswith(':'):
                    gap = abs(lpos_j - lpos_i)
                    if lane_j not in best_adjacent or gap < best_adjacent[lane_j][1]:
                        best_adjacent[lane_j] = (j, gap)

                # Category 3: Conflicting approach — nearest to intersection
                elif edge_i != edge_j and edge_j != '':
                    if topo_edges:
                        # C1 FIX: same next junction + different edges + within
                        # communication_radius of each other → conflict edge.
                        jx_i = _edge_jid(edge_i) if edge_i != '' else None
                        jx_j = _edge_jid(edge_j)
                        if jx_i is not None and jx_i == jx_j:
                            dxp = pos_j[0] - pos_i[0]
                            dyp = pos_j[1] - pos_i[1]
                            d2 = dxp * dxp + dyp * dyp
                            if d2 <= self.communication_radius ** 2:
                                if edge_j not in best_conflict or d2 < best_conflict[edge_j][1]:
                                    best_conflict[edge_j] = (j, d2)
                    # Legacy gate (BROKEN by design on real nets — kept only for
                    # byte-identical legacy reproduction with the flag off):
                    elif 'to_center' in edge_i and 'to_center' in edge_j:
                        # Both approaching — potential conflict
                        dx = pos_j[0] - cx
                        dy = pos_j[1] - cy
                        dist_to_int = dx * dx + dy * dy
                        if dist_to_int <= self.communication_radius ** 2:
                            if edge_j not in best_conflict or dist_to_int < best_conflict[edge_j][1]:
                                best_conflict[edge_j] = (j, dist_to_int)
            return best_ahead, best_behind, best_adjacent, best_conflict

        def _scan_fast(i):
            """M12-4 bucketed scan for row i — identical outputs to _scan_legacy in
            topo mode (see truth-table comment above)."""
            lane_i, edge_i, lpos_i, pos_i = info[i]

            best_ahead = (-1, float('inf'))
            best_behind = (-1, float('inf'))
            best_adjacent = {}
            best_conflict = {}

            # Category 1 (lane_i == '' is never a by_lane key → empty scan)
            for j in by_lane.get(lane_i, ()):
                if j == i:
                    continue
                lpos_j = info[j][2]
                gap = lpos_j - lpos_i
                if gap >= 0 and gap < best_ahead[1]:  # j is ahead (or coincident)
                    best_ahead = (j, gap)
                elif gap < 0 and abs(gap) < best_behind[1]:  # j is behind
                    best_behind = (j, abs(gap))

            # Category 2 (''/internal edge_i is never a by_edge key)
            for j in by_edge.get(edge_i, ()):
                if j == i:
                    continue
                lane_j, _ej, lpos_j, _pj = info[j]
                if lane_j == lane_i:
                    continue                    # Cat-1 claimed this pair
                gap = abs(lpos_j - lpos_i)
                if lane_j not in best_adjacent or gap < best_adjacent[lane_j][1]:
                    best_adjacent[lane_j] = (j, gap)

            # Category 3 (topo mode — graph_fast implies topo_edges)
            jx_i = _edge_jid(edge_i) if edge_i != '' else None
            if jx_i is not None:
                for j in by_jid.get(jx_i, ()):
                    if j == i:
                        continue
                    _lj, edge_j, _lpj, pos_j = info[j]
                    if edge_j == edge_i:
                        continue                # Cat-1/Cat-2 claimed, or elif guard
                    dxp = pos_j[0] - pos_i[0]
                    dyp = pos_j[1] - pos_i[1]
                    d2 = dxp * dxp + dyp * dyp
                    if d2 <= self.communication_radius ** 2:
                        if edge_j not in best_conflict or d2 < best_conflict[edge_j][1]:
                            best_conflict[edge_j] = (j, d2)
            return best_ahead, best_behind, best_adjacent, best_conflict

        _assert_steps = int(os.environ.get('DMPO_GRAPH_ASSERT', '0') or 0)

        for i in range(n):
            if graph_fast:
                best_ahead, best_behind, best_adjacent, best_conflict = _scan_fast(i)
                if _assert_steps and self.current_step <= _assert_steps:
                    _ref = _scan_legacy(i)
                    if (best_ahead != _ref[0] or best_behind != _ref[1]
                            or best_adjacent != _ref[2] or best_conflict != _ref[3]):
                        raise RuntimeError(
                            f"DMPO_GRAPH_ASSERT: fast/legacy scan mismatch at step "
                            f"{self.current_step} i={i}: "
                            f"fast={(best_ahead, best_behind, best_adjacent, best_conflict)} "
                            f"legacy={_ref}")
            else:
                best_ahead, best_behind, best_adjacent, best_conflict = _scan_legacy(i)

            # Connect to best neighbors
            if best_ahead[0] >= 0:
                adj_matrix[i, best_ahead[0]] = 1.0
            if best_behind[0] >= 0:
                adj_matrix[i, best_behind[0]] = 1.0
            for lane_j, (j, _) in best_adjacent.items():
                adj_matrix[i, j] = 1.0
            for edge_j, (j, _) in best_conflict.items():
                adj_matrix[i, j] = 1.0
                self._cat3_edge_count += 1   # C1 smoke-assert counter (H15-class health signal)

        # Symmetrize and add self-connections
        adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
        np.fill_diagonal(adj_matrix, 1.0)

        self.adjacency_matrix = adj_matrix

        # P1.2 / P3.5: compute the κ-hop transitive closure once per step so
        # the per-step reward averaging (and CARL+ team_bonus) can consume a
        # consistent κ-hop neighborhood. Without this, the averaging at
        # _compute_rewards would silently use 1-hop adj despite the docstring
        # and the framework's cooperative-MARL claim.
        if adj_matrix.size > 0:
            if graph_fast and self.kappa >= 1:
                # M12-4 κ-closure: BFS to depth κ over the bounded-degree graph instead
                # of dense O(κ·N³) matrix powers. EXACT because fill_diagonal made A
                # reflexive: (Σ_{k=1..κ} A^k > 0)[i,j] ⟺ j reachable from i in ≤ κ
                # hops — precisely what the BFS computes. κ=0 keeps the legacy path
                # (the old code degenerates to plain 1-hop adj there, BFS would not).
                nbrs = [np.flatnonzero(adj_matrix[_r]) for _r in range(n)]
                reach = np.zeros_like(adj_matrix)
                for _r in range(n):
                    frontier = {_r}
                    seen = {_r}
                    for _hop in range(self.kappa):
                        nxt = set()
                        for _u in frontier:
                            for _v in nbrs[_u]:
                                _v = int(_v)
                                if _v not in seen:
                                    seen.add(_v)
                                    nxt.add(_v)
                        frontier = nxt
                        if not frontier:
                            break
                    reach[_r, list(seen)] = 1.0
                if _assert_steps and self.current_step <= _assert_steps:
                    kappa_reach = adj_matrix.copy()
                    cur = adj_matrix.copy()
                    for _ in range(1, self.kappa):
                        cur = np.dot(cur, adj_matrix)
                        kappa_reach = kappa_reach + cur
                    if not np.array_equal((kappa_reach > 0).astype(np.float32), reach):
                        raise RuntimeError(
                            f"DMPO_GRAPH_ASSERT: BFS κ-closure mismatch at step "
                            f"{self.current_step} (n={n}, kappa={self.kappa})")
                self.kappa_reach = reach
            else:
                kappa_reach = adj_matrix.copy()
                cur = adj_matrix.copy()
                for _ in range(1, self.kappa):
                    cur = np.dot(cur, adj_matrix)
                    kappa_reach = kappa_reach + cur
                self.kappa_reach = (kappa_reach > 0).astype(np.float32)
        else:
            self.kappa_reach = adj_matrix
        # REWARD-RADIUS ABLATION (gated DMPO_REWARD_KAPPA, default unset = legacy, where the
        # reward neighbourhood follows --kappa and κ=0 falls back to 1 hop). Set to r:
        # r=0 → identity (own reward only: no dense averaging, no team bonus/cost);
        # r≥1 → r-hop reach. Decouples reward sharing from advantage routing (--kappa).
        _rk = _reward_kappa()
        if _rk is not None and adj_matrix.size > 0:
            self.kappa_reach = _reach(adj_matrix, _rk)

    def _get_observations(self) -> TypeDict[str, Any]:
        """Get observations for all AI vehicles, keyed by vehicle ID.

        SLOT IDENTITY FIX (B1 proper): Returns dict keyed by vehicle ID (str)
        instead of integer index. This makes slot identity stable across
        timesteps — the trajectory buffer can correctly track per-vehicle
        transitions when agents enter/leave the network.
        """
        # Share vehicle cache with observation processor
        # SENSING-NOISE SWEEP (eval-only; gated DMPO_SENSE_POS_SIGMA / _SPD_SIGMA / _DROP,
        # all default 0 = the true cache, byte-identical legacy path): the POLICY perceives
        # noisy V2X/GNSS reports; SUMO, rewards and action execution keep true states.
        self.observation_processor._vehicle_cache = self._sensed_cache()
        self.observation_processor._cache_epoch = self._cache_epoch  # M12-0: lockstep publish
        if self._profile_obs_on:
            # ARM-B: publish lifetime profiles in lockstep with the cache
            self.observation_processor._veh_profiles = self._veh_profiles
        if self._obs_v2:
            # OBS-V2 wiring (spec Step 6): prev_actions is assigned at the START of
            # step() and _get_observations runs AFTER simulationStep, so this holds
            # the action that produced the new state (empty at reset → prev_long=0).
            # _shield_engaged carries the PREVIOUS action phase's scaffold holds
            # (one-step lag, documented; constant-empty in bare/eval runs).
            self.observation_processor._prev_actions = self.prev_actions
            self.observation_processor._shield_engaged = self._shield_engaged_ids

        observations = {}
        ai_vehicles = sorted(self.ai_vehicle_ids)
        # Pass authoritative AI ordering to obs_processor so its neighbor
        # index lookups stay consistent with the adjacency matrix.
        self.observation_processor._ai_ordered = ai_vehicles
        for i, veh_id in enumerate(ai_vehicles):
            try:
                obs = self.observation_processor.process_observation(
                    veh_id, self.all_vehicle_ids, self.adjacency_matrix, i
                )
                observations[veh_id] = obs  # ← keyed by vehicle ID, not int
            except traci.exceptions.TraCIException:
                continue

        return observations
    
    def _apply_sumo_actions(self, sumo_actions: TypeDict[str, TypeDict[str, float]]):
        """Apply translated actions to SUMO vehicles.

        Uses setAcceleration(accel, duration=step_length) instead of
        setSpeed(). Key difference: setAcceleration automatically expires
        after `duration` seconds, returning the vehicle to SUMO's normal
        car-following. This means:
        - No persistence problem: a vehicle that misses one policy step
          resumes normal driving instead of being stuck at the last speed
        - No stuck-at-zero: if the policy commanded decel during red,
          the vehicle automatically accelerates on green (via car-following)
        - Untracked vehicles self-heal without needing setSpeed(-1)

        SLOT IDENTITY FIX: sumo_actions is now keyed by vehicle_id (str).
        """
        # OBS-V2: fresh scaffold-hold set per action phase. Rebound (not mutated)
        # so the obs processor's reference from the previous step stays intact
        # until _get_observations republishes.
        self._shield_engaged_ids = set()
        # APPROACH A — "borrowed coordination" ceiling (DMPO_SUMO_BASELINE=1).
        # Hand every AI vehicle back to SUMO's native driver model (Krauss
        # car-following + right-before-left junction priority, enforced when
        # DMPO_SPEED_MODE=7) and apply NO policy action. This measures the
        # classical-coordination ceiling — collision-free flow — that the learned
        # policy (Approach B) should aim for. setSpeed(-1) = release to SUMO.
        if os.environ.get('DMPO_SUMO_BASELINE', '0') == '1':
            for veh_id in list(self.ai_vehicle_ids):
                try:
                    # FULL handback: setSpeed(-1) alone leaves the configured
                    # DMPO_SPEED_MODE (6 = right-of-way check OFF) active — the
                    # "SUMO baseline" then drives junction-blind and produces a
                    # phantom ~2-collision floor. Mode 31 = all native checks.
                    traci.vehicle.setSpeedMode(veh_id, 31)
                    traci.vehicle.setSpeed(veh_id, -1)
                except Exception:
                    pass
            return

        # Physics-level safe-following enforcement (Krauss-style cap).
        # Instead of penalizing overlap in the reward, we PREVENT overlap by
        # capping the policy's commanded acceleration so the resulting speed
        # never exceeds the Krauss safe-speed for the current leader gap.
        # This is what a real car-following model would do; the policy still
        # commands aggressive accel, but physics caps it when unsafe.
        # DMPO_KRAUSS_CAP=1 enables (default 1).
        # DMPO_KRAUSS_DESIRED_GAP_M (default 2m) — gap target at zero leader speed.
        # DMPO_KRAUSS_REACTION_S (default 1.0s) — driver reaction time.
        # H7 FIX: default flipped '1'->'0' — the documented recommended value (the
        # policy drives; the cap remains available via yield_intent or explicit =1).
        krauss_cap_enabled = os.environ.get('DMPO_KRAUSS_CAP', '0') == '1'
        krauss_desired_gap_m = float(os.environ.get('DMPO_KRAUSS_DESIRED_GAP_M', '2.0'))
        krauss_reaction_s = float(os.environ.get('DMPO_KRAUSS_REACTION_S', '1.0'))
        # Legacy proximity-brake (kept for backward compatibility, default off
        # when Krauss-cap is enabled).
        proximity_brake_m = float(os.environ.get('DMPO_PROXIMITY_BRAKE_M', '0'))
        proximity_lookahead_m = float(os.environ.get('DMPO_PROXIMITY_LOOKAHEAD_M', '20.0'))
        proximity_radius_m = float(os.environ.get('DMPO_PROXIMITY_RADIUS_M', '0'))

        # ── Gap-acceptance SHIELD for CROSSING conflicts (DMPO_GAP_SHIELD=1) ──
        # Hard reservation constraint (Dresner & Stone AIM, reduced to pairwise
        # gap-acceptance): a lower-priority vehicle may not ENTER the ray-
        # intersection conflict point unless it can clear it before the crossing
        # vehicle arrives. Enforced as a virtual stop line `margin` m before the
        # conflict point, capped kinematically. Complements the Krauss cap above,
        # which only guards the same-lane leader (rear-end) — this guards the
        # crossing (junction) conflict the Krauss cap is blind to. Monotone-safe:
        # only ever REDUCES commanded accel, so it can never create a collision.
        gap_shield_enabled = os.environ.get('DMPO_GAP_SHIELD', '0') == '1'
        # MARGIN_M = stop-bar buffer held BEFORE the junction entrance (lane end), not before the
        # mid-junction conflict point. Keep small so the yielder waits right at the box mouth.
        gap_shield_margin_m = float(os.environ.get('DMPO_GAP_SHIELD_MARGIN_M', '2.0'))
        gap_shield_pet_s = float(os.environ.get('DMPO_GAP_SHIELD_PET_S', '3.0'))

        # Snapshot positions of all vehicles for the Euclidean check.
        # E9/M12: all_pos is consumed ONLY by the proximity-radius brake below
        # (DMPO_PROXIMITY_RADIUS_M > 0, default off) — skip the N+1 TraCI calls when it
        # can never be read. When the brake IS on, the legacy snapshot runs verbatim.
        if proximity_radius_m > 0 or os.environ.get('DMPO_PERF_FIXES', '1') != '1':
            try:
                all_pos = {vid: traci.vehicle.getPosition(vid)
                           for vid in traci.vehicle.getIDList()}
            except Exception:
                all_pos = {}
        else:
            all_pos = {}

        def _force_instant_stop(vid):
            """Set speed to 0 instantly by bypassing max_decel via speed mode."""
            try:
                traci.vehicle.setSpeedMode(vid, 0)  # all constraints off
                traci.vehicle.setSpeed(vid, 0.0)
                traci.vehicle.setSpeedMode(vid, int(os.environ.get('DMPO_SPEED_MODE', '6')))  # restore configured mode
            except Exception:
                pass

        acted_vehicles = set()
        _detail_on = os.environ.get('DMPO_ACTION_DETAIL', '0') == '1'
        if _detail_on:
            self._action_detail = {}
        for veh_id, action_dict in sumo_actions.items():
            if veh_id in self.ai_vehicle_ids:
                # FORENSIC: capture the policy's commanded accel BEFORE any cap.
                _orig_accel = float(action_dict.get('acceleration', 0.0)) if _detail_on else 0.0
                # If this AI is stuck after a collision, freeze it.
                if veh_id in self._stuck_ai:
                    if _detail_on:
                        self._action_detail[veh_id] = {'commanded_accel': _orig_accel,
                            'yield_active': None, 'krauss_capped': False, 'final_accel': 0.0,
                            'frozen_stuck': True}
                    _force_instant_stop(veh_id)
                    continue

                # Krauss-style physics cap on policy-commanded acceleration.
                # Rather than penalize overlap in the reward, we cap the
                # commanded accel so resulting speed ≤ Krauss safe-speed.
                # safe_v = v_leader + (gap - desired_gap) / reaction_s
                # If commanded next-step speed exceeds safe_v, we clip the accel
                # so the next-step speed equals safe_v exactly.
                # Variant C: per-vehicle yield_intent forces Krauss-cap regardless
                # of the global flag, so the policy can opt into safe-following
                # for this step.
                yield_active = bool(action_dict.get('yield_intent_active', False))
                if os.environ.get('DMPO_FORCE_YIELD_OFF', '0') == '1':
                    yield_active = False   # RCA: disable the yield_intent Krauss cap
                if krauss_cap_enabled or yield_active:
                    try:
                        leader = traci.vehicle.getLeader(veh_id, dist=proximity_lookahead_m)
                    except Exception:
                        leader = None
                    if leader is not None:
                        leader_id, gap = leader
                        try:
                            v_leader = traci.vehicle.getSpeed(leader_id)
                        except Exception:
                            v_leader = 0.0
                        try:
                            v_self = traci.vehicle.getSpeed(veh_id)
                        except Exception:
                            v_self = 0.0
                        # V8 THIN EMERGENCY FLOOR: when DMPO_KRAUSS_THIN_FLOOR_M>0,
                        # the Krauss cap only engages once the gap is below this
                        # imminent-collision threshold. Above it, the policy's own
                        # longitudinal command is applied (so the policy must learn
                        # routine braking instead of leaning on the shield every step).
                        thin_floor_m = float(os.environ.get('DMPO_KRAUSS_THIN_FLOOR_M', '0'))
                        if thin_floor_m > 0 and gap > thin_floor_m:
                            # Outside the emergency zone — do NOT cap; policy drives.
                            self._krauss_floor_skipped = getattr(self, '_krauss_floor_skipped', 0) + 1
                            v_self = None  # sentinel to skip the cap block below
                        if v_self is not None:
                            if os.environ.get('DMPO_KRAUSS_CAP_V2', '0') == '1':
                                # H7 FIX (gated, default off — legacy byte-identical): braking-aware
                                # safe speed. The legacy formula has NO stopping-distance term, so a
                                # cruise-speed follower cannot stop (verified ~11-12 m/s impact at
                                # decel 4.5). v2: allow the speed from which we can stop within
                                # (gap - mingap) + the leader's own stopping distance, discrete-time.
                                _b = abs(self.action_translator.max_decel)
                                _bt = _b * self.step_length
                                _g = max(gap - krauss_desired_gap_m, 0.0) + (v_leader * v_leader) / (2.0 * _b)
                                safe_v = float(max(0.0, -_bt + np.sqrt(_bt * _bt + 2.0 * _b * _g)))
                            else:
                                safe_v = max(0.0, v_leader + (gap - krauss_desired_gap_m) / krauss_reaction_s)
                            # Commanded accel from policy
                            commanded_accel = action_dict.get('acceleration', 0.0)
                            # Resulting next-step speed if we apply commanded accel
                            commanded_next_v = v_self + commanded_accel * self.step_length
                            if commanded_next_v > safe_v:
                                # Cap accel to reach exactly safe_v at next step
                                capped_accel = (safe_v - v_self) / self.step_length
                                # Mutate the action dict so the standard apply path
                                # below uses the capped value
                                action_dict = dict(action_dict)
                                action_dict['acceleration'] = capped_accel
                                self._krauss_floor_fired = getattr(self, '_krauss_floor_fired', 0) + 1

                # ── Gap-acceptance SHIELD (crossing conflicts) ──────────────
                # Priority by arrival order at the NEAREST conflict point:
                #   ta<0 => ego arrives first  => GO   (not capped; it clears)
                #   ta>0 => ego arrives second => YIELD (hold before conflict pt)
                # Only engage when arrivals actually overlap (pet<PET_S). The
                # yielder is released automatically once the higher-priority
                # vehicle passes (its conflict point drops behind it -> no
                # crossing conflict -> cap lifts), so a 2-way conflict never
                # freezes. N-way priority cycles can still stall — measured
                # empirically via completions, not assumed away.
                if gap_shield_enabled and self.observation_processor is not None:
                    cc = self.observation_processor._crossing_conflict(veh_id)
                    cp = cc.get('cp')
                    if (cp is not None and cc.get('ta', 0.0) > 0.0
                            and cc.get('pet', 10.0) < gap_shield_pet_s):
                        # Hold at the JUNCTION ENTRANCE (lane end), NOT margin-before the mid-
                        # junction conflict point. The yielder must stop BEFORE entering the box so
                        # the box stays clear for the crossing vehicle; stopping partway in (near the
                        # centre conflict point) parks the yielder ON the crosser's path -> causes
                        # collisions, not prevents them. SUMO gives the entrance distance directly
                        # from lane geometry; internal-junction lanes are prefixed ':' (already
                        # inside the box -> committed, let it clear).
                        entry_dist = None
                        try:
                            _lane = traci.vehicle.getLaneID(veh_id)
                            if _lane and not _lane.startswith(':'):
                                entry_dist = (traci.lane.getLength(_lane)
                                              - traci.vehicle.getLanePosition(veh_id))
                        except Exception:
                            entry_dist = None
                        if entry_dist is not None:
                            # stop_dist = distance to the virtual stop line (margin before the box), clamped
                            # to 0 so v_allow -> 0 AT the line -> the yielder comes to a FULL STOP and HOLDS
                            # there until the conflict clears (cc stops flagging it), instead of coasting in.
                            stop_dist = max(entry_dist - gap_shield_margin_m, 0.0)
                            if os.environ.get('DMPO_GAP_SHIELD_TRACE', '0') == '1':
                                print(f"[SHIELD] t={self.current_step} veh={veh_id} nid={cc.get('nid')} "
                                      f"ta={cc.get('ta'):.2f} pet={cc.get('pet'):.2f} entry_dist={entry_dist:.1f} "
                                      f"stop_dist={stop_dist:.1f}", flush=True)
                            # Hold while still OUTSIDE the junction box. Only give up (commit) once the
                            # vehicle is essentially at/inside the entrance, where it physically cannot stop.
                            if entry_dist > 0.3:
                                try:
                                    v_self = traci.vehicle.getSpeed(veh_id)
                                except Exception:
                                    v_self = 0.0
                                decel_mag = abs(self.action_translator.max_decel)
                                # DISCRETE-TIME safe speed (not the kinematic ragged edge): the vehicle
                                # travels v*dt for one reaction step before max-decel braking bites, so
                                # solve v*dt + v^2/(2a) <= stop_dist -> v = -a*dt + sqrt((a*dt)^2 + 2a*d).
                                # This makes the yielder slow EARLIER and actually stop before the line
                                # (the naive sqrt(2a*d) let a fast approacher overshoot into the junction).
                                adt = decel_mag * self.step_length
                                v_allow = float(max(0.0, -adt + np.sqrt(adt * adt + 2.0 * decel_mag * stop_dist)))
                                # CAUTIOUS APPROACH: the confirmed root cause is that the policy barrels
                                # in at ~15 m/s and the reactive stopping-envelope cap only bites in the
                                # last ~15 m — too late to halt a full-speed car. Within an approach zone,
                                # force yielders down to a low crawl speed so they can ALWAYS stop at the
                                # line. This is the shield doing what a policy trained to approach slowly
                                # would do on its own (DMPO_GAP_SHIELD_APPROACH_ZONE_M=0 disables it).
                                _appr_zone = float(os.environ.get('DMPO_GAP_SHIELD_APPROACH_ZONE_M', '0'))
                                if _appr_zone > 0.0 and entry_dist < _appr_zone:
                                    _appr_mps = float(os.environ.get('DMPO_GAP_SHIELD_APPROACH_MPS', '6.0'))
                                    # SELECTIVE full-stop: come to a COMPLETE stop only for a GENUINE conflict
                                    # (arrivals within GENUINE_PET_S seconds — truly simultaneous); otherwise
                                    # just crawl. Blunt "always stop" desyncs flow and spawns NEW conflicts;
                                    # gating the full stop on tight PET stops only where it's actually needed.
                                    _gpet = float(os.environ.get('DMPO_GAP_SHIELD_GENUINE_PET_S', '0'))
                                    if _gpet > 0.0 and cc.get('pet', 10.0) < _gpet:
                                        _appr_mps = 0.0
                                    v_allow = min(v_allow, _appr_mps)
                                commanded_accel = action_dict.get('acceleration', 0.0)
                                commanded_next_v = v_self + commanded_accel * self.step_length
                                # F2 fix (soundness review, HIGH): record the HOLD whenever the
                                # shield's speed ceiling is BINDING — not only when the commanded
                                # action exceeded it. Recording only the clamped case fined the
                                # vehicle that VOLUNTARILY complied (commands stop while held)
                                # while exempting the one that fights the shield: an anti-yield
                                # gradient inside the very arm (M11) that needs the exemption.
                                if v_allow < v_self or commanded_next_v > v_allow:
                                    self._shield_engaged_ids.add(veh_id)
                                if commanded_next_v > v_allow:
                                    capped_accel = (v_allow - v_self) / self.step_length
                                    action_dict = dict(action_dict)
                                    action_dict['acceleration'] = min(commanded_accel, capped_accel)
                                    self._gap_shield_fired = getattr(self, '_gap_shield_fired', 0) + 1

                # ── FIX #4: car-following safe-speed cap (prevents rear-ending a shield-slowed leader) ──
                # speedMode=6 + KRAUSS_CAP=0 leave same-lane following UNGUARDED, so when the crossing
                # shield slows a leader, an unshielded follower (no crossing conflict of its own) can
                # rear-end it (grid #628). Cap every AI vehicle's speed to a safe following speed vs its
                # in-lane leader, using the SAME discrete-time stopping envelope as the crossing shield.
                # Kept in OUR shield (not delegated to SUMO speedMode). Gated DMPO_GAP_SHIELD_FOLLOW.
                if os.environ.get('DMPO_GAP_SHIELD_FOLLOW', '0') == '1':
                    try:
                        _ld = traci.vehicle.getLeader(veh_id, 100.0)   # (leader_id, gap) gap=bumper-to-bumper
                    except Exception:
                        _ld = None
                    if _ld is not None and _ld[0] and _ld[1] is not None and _ld[1] >= 0.0:
                        try:
                            _vf = traci.vehicle.getSpeed(veh_id)
                        except Exception:
                            _vf = 0.0
                        _b = abs(self.action_translator.max_decel)
                        _mingap = float(os.environ.get('DMPO_GAP_SHIELD_FOLLOW_MINGAP', '2.0'))
                        _g = max(_ld[1] - _mingap, 0.0)
                        _bt = _b * self.step_length
                        _vsafe = float(max(0.0, -_bt + np.sqrt(_bt * _bt + 2.0 * _b * _g)))
                        _ca = action_dict.get('acceleration', 0.0)
                        if _vf + _ca * self.step_length > _vsafe:
                            action_dict = dict(action_dict)
                            action_dict['acceleration'] = min(_ca, (_vsafe - _vf) / self.step_length)
                            self._follow_cap_fired = getattr(self, '_follow_cap_fired', 0) + 1
                            # F2 fix: a vehicle CHAIN-HELD behind a shield-slowed leader
                            # is just as held as the yielder itself — exempt it too,
                            # otherwise the scaffold fines the whole queue behind a hold.
                            self._shield_engaged_ids.add(veh_id)

                # Legacy proximity-brake (only if explicitly enabled via env vars)
                brake_now = False
                if proximity_brake_m > 0:
                    try:
                        leader2 = traci.vehicle.getLeader(veh_id, dist=proximity_lookahead_m)
                        if leader2 is not None and leader2[1] < proximity_brake_m:
                            brake_now = True
                    except Exception:
                        pass
                if not brake_now and proximity_radius_m > 0 and veh_id in all_pos:
                    ax, ay = all_pos[veh_id]
                    r2 = proximity_radius_m * proximity_radius_m
                    for other_id, (ox, oy) in all_pos.items():
                        if other_id == veh_id:
                            continue
                        dx = ox - ax
                        dy = oy - ay
                        if dx * dx + dy * dy < r2:
                            brake_now = True
                            break
                if brake_now:
                    _force_instant_stop(veh_id)
                    acted_vehicles.add(veh_id)
                    continue

                try:
                    if 'acceleration' in action_dict:
                        accel = action_dict['acceleration']
                        # Clamp to vType physical limits
                        accel = max(self.action_translator.max_decel,
                                    min(self.action_translator.max_accel, accel))
                        # Apply for exactly one step — then SUMO resumes control
                        traci.vehicle.setAcceleration(veh_id, accel, self.step_length)
                        acted_vehicles.add(veh_id)
                        if _detail_on:
                            self._action_detail[veh_id] = {
                                'commanded_accel': _orig_accel,
                                'yield_active': bool(yield_active),
                                'krauss_capped': abs(float(action_dict.get('acceleration', 0.0)) - _orig_accel) > 1e-6,
                                'final_accel': float(accel),
                            }

                    # Apply continuous lateral movement (sublane model)
                    # F5 SCALPEL (DMPO_LAT_HANDOFF=1): skip the per-step sublane command
                    # for STOPPED vehicles — the continuous changeSublane stream vetoes
                    # SUMO's own lane-change model, trapping followers behind blocked
                    # leaders. Releasing lateral (only while stopped) lets SUMO's LC
                    # model attempt the escape. Diagnostic gate; default off.
                    if os.environ.get('DMPO_LAT_HANDOFF', '0') == '1':
                        try:
                            if traci.vehicle.getSpeed(veh_id) < 0.5:
                                continue   # no sublane command this step for this vehicle
                        except Exception:
                            pass
                    if os.environ.get('DMPO_NO_LATERAL', '0') == '1':
                        continue   # RCA gate: policy lateral fully disabled
                    if 'lane_change' in action_dict:
                        lat_cmd = action_dict['lane_change']
                        # V9 LATERAL-CENTERING FIX: AI vehicles accumulate a
                        # ~0.7 m off-center drift (humans stay centered at lat≈0)
                        # which causes geometric overlap collisions even when the
                        # longitudinal gap is fine. Add a proportional restoring
                        # pull toward lane center based on the vehicle's current
                        # lateral lane position. Capped to half the per-step
                        # lateral budget so a DELIBERATE lane change is slowed but
                        # never blocked. Gated by DMPO_LATERAL_CENTERING_K
                        # (0/unset = off = exact V8 behavior).
                        cen_k = float(os.environ.get('DMPO_LATERAL_CENTERING_K', '0'))
                        if cen_k > 0:
                            try:
                                cur_lat = traci.vehicle.getLateralLanePosition(veh_id)
                            except Exception:
                                cur_lat = 0.0
                            max_step_lat = float(os.environ.get('DMPO_MAX_LAT_VELOCITY', '1.0')) * self.step_length
                            restore = -cen_k * cur_lat
                            cap = 0.5 * max_step_lat
                            restore = max(-cap, min(cap, restore))
                            lat_cmd = lat_cmd + restore
                            self._lat_centering_applied = getattr(self, '_lat_centering_applied', 0) + 1
                            self._lat_centering_drift_sum = getattr(self, '_lat_centering_drift_sum', 0.0) + abs(cur_lat)
                        traci.vehicle.changeSublane(veh_id, lat_cmd)

                except traci.exceptions.TraCIException:
                    # Vehicle may have left simulation
                    continue
    
    def _compute_rewards(self, observations: TypeDict[str, Any],
                        actions: TypeDict[str, TypeDict[str, np.ndarray]]) -> TypeDict[str, float]:
        """Compute rewards for all AI vehicles, keyed by vehicle ID.

        Uses local reward averaging (Zhou et al. 2022): each vehicle's
        final reward is the mean of its own reward and its κ-hop neighbors'
        rewards. This creates cooperative pressure — if your actions cause
        neighbors to get low reward, your averaged reward drops too.
        """
        # Share vehicle cache with reward computer
        self.reward_computer._vehicle_cache = self._vehicle_cache
        self.reward_computer._forefront_vehicles = self._forefront_vehicles
        self.reward_computer._forefront_tls_state = self._forefront_tls_state
        # Share the obs processor so conflict-point reward shaping (DMPO_CONFLICT_OBS) reuses
        # the SAME per-step _crossing_conflict signal the observation uses (consistency).
        self.reward_computer._obs_processor = self.observation_processor
        # OBS-V2 / REWARD-V2 contract: publish the shield-hold and stuck-behind-
        # collision sets the idle-yield exemptions read (rc reads both via getattr —
        # a no-op until published here). Same-step semantics: these are the holds of
        # the action phase whose outcome this reward scores.
        self.reward_computer._shield_held = self._shield_engaged_ids
        self.reward_computer._stuck_ai = set(self._stuck_ai.keys())

        # Step 1: Compute individual (ego) rewards for all vehicles
        ego_rewards = {}
        for veh_id in observations.keys():
            if veh_id in actions:
                reward = self.reward_computer.compute_reward(
                    veh_id, observations[veh_id], actions[veh_id],
                    self.prev_observations.get(veh_id, {}),
                    self.prev_actions.get(veh_id, {})
                )
                ego_rewards[veh_id] = float(reward)
            else:
                ego_rewards[veh_id] = 0.0

        # Step 2: Local reward averaging over κ-hop neighbors.
        # P1.2 / P3.5: now uses self.kappa_reach (transitive closure to κ hops)
        # instead of self.adjacency_matrix (1-hop). Previously this silently
        # averaged over 1-hop only despite the docstring and paper claims.
        ai_vehicles = sorted(self.ai_vehicle_ids)
        adj = getattr(self, 'kappa_reach', None)
        if adj is None or adj.size == 0:
            # Fallback for very-first-step or unusual states
            adj = self.adjacency_matrix

        if adj is not None and len(adj) > 0 and len(ai_vehicles) > 0:
            rewards = {}
            for i, veh_id in enumerate(ai_vehicles):
                if veh_id not in ego_rewards:
                    continue
                # Collect ego + neighbor rewards
                local_rewards = [ego_rewards[veh_id]]
                if i < len(adj):
                    for j in range(len(adj[i])):
                        if adj[i][j] > 0 and j != i and j < len(ai_vehicles):
                            neighbor_id = ai_vehicles[j]
                            if neighbor_id in ego_rewards:
                                local_rewards.append(ego_rewards[neighbor_id])
                # Average
                rewards[veh_id] = sum(local_rewards) / len(local_rewards)

            # Include any vehicles in ego_rewards but not in ai_vehicles
            # (e.g., completed vehicles with completion bonus)
            for veh_id in ego_rewards:
                if veh_id not in rewards:
                    rewards[veh_id] = ego_rewards[veh_id]
        else:
            rewards = ego_rewards

        return rewards
    
    def _compute_dones(self) -> TypeDict[str, bool]:
        """Compute done flags for all AI vehicles, keyed by vehicle ID.

        SLOT IDENTITY FIX (B1 proper): Returns dict keyed by vehicle ID (str).
        Per-vehicle done is True only if global episode done. Per-vehicle
        completion (vehicle exits the network) is signaled separately via
        the info['completed_vehicles'] set, NOT via dones — because returning
        done=True per agent would trigger global episode termination in the
        training loop.

        The training loop uses info['completed_vehicles'] to mark completed
        slots' last timestep with done=True for proper GAE bootstrapping.
        """
        dones = {}

        # Global done condition (episode termination)
        global_done = (self.current_step >= self.max_steps or
                      len(self.all_vehicle_ids) == 0)

        for veh_id in sorted(self.ai_vehicle_ids):
            dones[veh_id] = global_done

        return dones
    
    def _get_intersection_throughput(self) -> int:
        """Get number of vehicles that crossed intersection."""
        # Simplified metric - count vehicles on the single-intersection outgoing edges.
        # On non-'center' nets (grids, real cities like ATSC NY) these edges don't exist;
        # probe ONCE and skip thereafter, else 4 failed traci calls fire every step (slow + spammy).
        if getattr(self, '_center_out_edges', None) is None:
            try:
                _all = set(traci.edge.getIDList())
            except Exception:
                _all = set()
            self._center_out_edges = [e for e in
                                      ('center_to_north', 'center_to_south', 'center_to_east', 'center_to_west')
                                      if e in _all]
        throughput = 0
        for edge in self._center_out_edges:
            try:
                throughput += traci.edge.getLastStepVehicleNumber(edge)
            except traci.exceptions.TraCIException:
                pass
        return throughput
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get current communication adjacency matrix."""
        return self.adjacency_matrix if self.adjacency_matrix is not None else np.array([])
    
    def get_n_agents(self) -> int:
        """Get current number of AI agents."""
        return self.n_agents

    def get_traci_port(self) -> int:
        """Get the TraCI port for external connections."""
        return self.eval_proxy_port if self.enable_external_eval else None

    def _start_evaluation_proxy(self):
        """Start the evaluation proxy server in a separate thread."""
        def proxy_server():
            try:
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.bind(('127.0.0.1', self.eval_proxy_port))
                server_socket.listen(5)
                self.eval_proxy_server = server_socket

                print(f"📡 Evaluation proxy server listening on port {self.eval_proxy_port}")
                print(f"🔧 Test connection: telnet 127.0.0.1 {self.eval_proxy_port}")

                while True:
                    try:
                        client_socket, address = server_socket.accept()
                        print(f"📊 Evaluation client connected from {address}")

                        # Handle client in a separate thread
                        client_thread = threading.Thread(
                            target=self._handle_evaluation_client,
                            args=(client_socket,),
                            daemon=True
                        )
                        client_thread.start()
                    except Exception as e:
                        print(f"⚠️ Proxy server error: {e}")
                        break
            except Exception as e:
                print(f"❌ Failed to start evaluation proxy server: {e}")

        # Start proxy in background thread
        print(f"🚀 Starting evaluation proxy server on port {self.eval_proxy_port}...")
        self.eval_proxy_thread = threading.Thread(target=proxy_server, daemon=True)
        self.eval_proxy_thread.start()

        # Give it a moment to start
        time.sleep(0.5)
        print(f"✅ Evaluation proxy thread started")

    def _write_port_file(self):
        """Write evaluation proxy port to file for external scripts."""
        try:
            if self.eval_proxy_port is None:
                print(f"⚠️ Cannot write port file: eval_proxy_port is None")
                return

            port_file_path = os.path.join(os.getcwd(), '.dmpo_eval_port')
            with open(port_file_path, 'w') as f:
                f.write(str(self.eval_proxy_port))
            print(f"📁 Port file written: {port_file_path} (port: {self.eval_proxy_port})")
        except Exception as e:
            print(f"⚠️ Failed to write port file: {e}")

    def _handle_evaluation_client(self, client_socket):
        """Handle individual evaluation client connections."""
        try:
            while True:
                # Receive request from evaluation client
                data = client_socket.recv(4096)
                if not data:
                    break

                try:
                    request = json.loads(data.decode('utf-8'))
                    response = self._process_evaluation_request(request)

                    # Send response back to client
                    response_json = json.dumps(response, default=str)
                    client_socket.send(response_json.encode('utf-8'))
                except json.JSONDecodeError:
                    error_response = {"error": "Invalid JSON request"}
                    client_socket.send(json.dumps(error_response).encode('utf-8'))
                except Exception as e:
                    error_response = {"error": f"Request processing error: {e}"}
                    client_socket.send(json.dumps(error_response).encode('utf-8'))

        except Exception as e:
            print(f"⚠️ Client handler error: {e}")
        finally:
            client_socket.close()
            print("📊 Evaluation client disconnected")

    def _process_evaluation_request(self, request):
        """Process evaluation requests and return SUMO data."""
        command = request.get('command', '')

        try:
            if command == 'get_simulation_time':
                return {"result": traci.simulation.getTime()}

            elif command == 'get_vehicle_list':
                return {"result": list(traci.vehicle.getIDList())}

            elif command == 'get_vehicle_info':
                vehicle_id = request.get('vehicle_id')
                if vehicle_id:
                    info = {
                        'position': traci.vehicle.getPosition(vehicle_id),
                        'speed': traci.vehicle.getSpeed(vehicle_id),
                        'lane': traci.vehicle.getLaneID(vehicle_id),
                        'acceleration': traci.vehicle.getAcceleration(vehicle_id),
                    }
                    return {"result": info}
                else:
                    return {"error": "vehicle_id required"}

            elif command == 'get_simulation_stats':
                vehicles = list(traci.vehicle.getIDList())
                stats = {
                    'simulation_time': traci.simulation.getTime(),
                    'vehicle_count': len(vehicles),
                    'vehicles': vehicles,
                    'min_expected_vehicles': traci.simulation.getMinExpectedNumber()
                }
                return {"result": stats}

            elif command == 'simulation_step':
                # Note: We don't actually step here, just acknowledge
                # The training loop controls simulation stepping
                return {"result": "acknowledged"}

            else:
                return {"error": f"Unknown command: {command}"}

        except Exception as e:
            return {"error": f"TraCI error: {e}"}

    def close(self):
        """Close environment and cleanup proxy server."""
        # Stop evaluation proxy if running
        if self.eval_proxy_server:
            try:
                self.eval_proxy_server.close()
                print("📡 Evaluation proxy server stopped")
            except:
                pass

        # Clean up port file
        try:
            port_file_path = os.path.join(os.getcwd(), '.dmpo_eval_port')
            if os.path.exists(port_file_path):
                os.remove(port_file_path)
                print("📁 Port file cleaned up")
        except:
            pass

        # Close SUMO
        if USING_LIBSUMO:
            # libsumo: close only when truly done
            if self._libsumo_started:
                try:
                    traci.close()
                    self._libsumo_started = False
                except:
                    pass
        else:
            # TraCI: normal close
            if traci.isLoaded():
                traci.close()

        # Stop SUMO process if running (TraCI mode only)
        if hasattr(self, 'sumo_process') and self.sumo_process:
            try:
                self.sumo_process.terminate()
                self.sumo_process.wait(timeout=5)
            except:
                if hasattr(self, 'sumo_process'):
                    self.sumo_process.kill()


def create_intersection_env(config_path: str = None, **kwargs) -> SUMOIntersectionEnv:
    """Factory function to create intersection environment."""
    if config_path is None:
        # Default: the 4-lane lanefix benchmark scenario shipped with this repository
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                   'configs', 'sumo_configs', '4lane_allAI_unsig_poisson_lanefix', 'intersection.sumocfg')
    
    return SUMOIntersectionEnv(config_path, **kwargs)