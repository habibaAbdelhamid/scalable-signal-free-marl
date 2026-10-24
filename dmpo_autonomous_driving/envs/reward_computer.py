"""
Context-Aware Reward Computer for DMPO Intersection Environment

Computes multi-objective rewards using standard automotive metrics:
- Safety: Time headway (same-lane) + Post-encroachment time (cross-traffic)
- Progress: Continuous route progress + context-aware delay
- Comfort: ISO 2631 jerk thresholds with context
- Cooperation: Yielding + intersection blocking
"""

import numpy as np
import os
import logging
import math

logger = logging.getLogger(__name__)

if os.environ.get('DMPO_USE_TRACI', '0') == '1':
    import traci
else:
    try:
        import libsumo as traci
    except ImportError:
        import traci
from typing import Dict, Any, Optional


class RewardComputer:
    """
    Context-aware multi-objective driving rewards for DMPO.

    Weights (from literature):
    - Safety (40%): Collision avoidance, safe following, conflict avoidance
    - Progress (30%): Route completion, intersection efficiency
    - Comfort (20%): Smooth acceleration, human-like driving
    - Cooperation (10%): Yielding, traffic flow optimization
    """

    def __init__(self):
        self._vehicle_cache = None
        # Handle to the ObservationProcessor so conflict-point reward shaping reuses the
        # SAME _crossing_conflict signal the observation uses (consistency; no duplicate logic).
        # Wired by the env each step. None => conflict shaping is a no-op.
        self._obs_processor = None

        # Trial mode: simplified 3-term reward (no thresholds, no loopholes)
        import os as _os_w
        self._trial = os.environ.get('DMPO_REWARD_TRIAL', '0') == '1'
        # Literature-aligned reward (Liu et al. 2024, highway-env): 3 terms, ratio 1:2
        self._literature = os.environ.get('DMPO_REWARD_LITERATURE', '0') == '1'
        # Collision penalty override for sublane experiments (default -5, set higher for no-safety sublane)
        self._literature_collision = float(os.environ.get('DMPO_COLLISION_PENALTY', '5'))
        # Path D: Action-accountable reward (every action dim has a reward term)
        self._accountable = os.environ.get('DMPO_REWARD_ACCOUNTABLE', '0') == '1'
        # Path E: CaRL-style route progress + centering (CoRL 2025)
        # Path G (CARL_PLUS): same per-step reward as CaRL; the +200
        # completion bonus is injected by env.step(), not by this computer.
        self._carl = (os.environ.get('DMPO_REWARD_CARL', '0') == '1' or
                      os.environ.get('DMPO_REWARD_CARL_PLUS', '0') == '1')
        # Path F: Pure sparse reward — only completion + collision signals
        # The completion bonus is injected by the env (sumo_intersection_env.step),
        # since completed vehicles are no longer in observations when reward is computed.
        # This computer just produces 0 (or collision penalty) for every alive step.
        self._sparse = os.environ.get('DMPO_REWARD_SPARSE', '0') == '1'
        self._sparse_collision_penalty = float(os.environ.get('DMPO_SPARSE_COLLISION_PENALTY', '5'))
        # Track previous route fraction for delta computation
        self._prev_route_frac = {}
        # Track consecutive-stuck steps per vehicle so idle_penalty can ramp
        # with persistence (flat-gradient fix for CaRL gridlock)
        self._stuck_counter = {}
        # ---- REWARD V2 (Stage-3 rebalance; umbrella gate DMPO_REWARD_V2) ----
        # Default 0 = byte-identical legacy reward (legacy_r5_* profiles take
        # the exact pre-v2 code paths). Sub-flags are only consulted when the
        # umbrella is on; their defaults follow the v2 profile in
        # configs/training_block.sh but stay individually overridable.
        self._reward_v2 = os.environ.get('DMPO_REWARD_V2', '0') == '1'
        # §4: shared speed ramp replaces the 0.5 m/s movement/idle cliff
        self._v2_ramp = self._reward_v2 and os.environ.get('DMPO_MOVEMENT_RAMP', '1') == '1'
        self._v2_ramp_lo = float(os.environ.get('DMPO_MOVEMENT_RAMP_LO', '0.3'))
        self._v2_ramp_hi = float(os.environ.get('DMPO_MOVEMENT_RAMP_HI', '0.7'))
        # §6a: trapezoid route odometer (kills the rectangle-rule anti-braking bias)
        self._v2_trapezoid = self._reward_v2 and os.environ.get('DMPO_PROGRESS_TRAPEZOID', '1') == '1'
        # §5: yield-aware idle exemptions (all default OFF). The PET exemption
        # consumes obs_processor._crossing_conflict, which is an all-zero early
        # return unless the conflict geometry is live (DMPO_CONFLICT_OBS /
        # DMPO_GAP_SHIELD / the future obs-v2 DMPO_CONFLICT_GEOM) -> functional
        # no-op in the bare legacy rig even if the flag is set. The shield/stuck
        # exemptions read sets the env must publish onto this computer
        # (self._shield_held / self._stuck_ai — deferred to the obs-v2 env
        # batch); while unpublished they are a no-op.
        self._v2_idle_yield_exempt = self._reward_v2 and os.environ.get('DMPO_IDLE_YIELD_EXEMPT', '0') == '1'
        self._v2_idle_exempt_pet_s = float(os.environ.get('DMPO_IDLE_EXEMPT_PET_S', '1.5'))
        self._v2_idle_exempt_shield = self._reward_v2 and os.environ.get('DMPO_IDLE_EXEMPT_SHIELD', '0') == '1'
        self._v2_idle_exempt_stuck = self._reward_v2 and os.environ.get('DMPO_IDLE_EXEMPT_STUCK', '0') == '1'
        self._v2_idle_exempt_max_steps = int(os.environ.get('DMPO_IDLE_EXEMPT_MAX_STEPS', '50'))
        # v2 per-vehicle state: previous-step speed (trapezoid odometer) and
        # consecutive exempt-idle steps (anti-gridlock duration cap)
        self._prev_speed = {}
        self._exempt_counter = {}
        # Configurable overlap penalty (fires when leader within 0.5 m).
        # Default -0.05 preserves prior behavior; use -0.5 for stronger
        # anti-pileup signal in unsignalized coordination experiments.
        self._overlap_penalty_value = float(os.environ.get('DMPO_OVERLAP_PENALTY', '-0.05'))
        # Path B: Rebalanced 4-component (reduced penalties, higher completion bonus)
        self._rebalanced = os.environ.get('DMPO_REWARD_REBALANCED', '0') == '1'
        # Path C: Normalized non-negative reward (highway-env style, all rewards in [0,1])
        self._normalized = os.environ.get('DMPO_REWARD_NORMALIZED', '0') == '1'

        # Reward weights (used only in non-trial mode)
        # DMPO_PROGRESS_FIRST=1: swap safety/progress weights to incentivize movement
        _progress_first = os.environ.get('DMPO_PROGRESS_FIRST', '0') == '1'
        if _progress_first:
            self.weights = {
                'safety': 0.30,
                'progress': 0.40,
                'comfort': 0.20,
                'cooperation': 0.10
            }
        else:
            self.weights = {
                'safety': 0.40,
                'progress': 0.30,
                'comfort': 0.20,
                'cooperation': 0.10
            }

        # Phase 2 reward rebalancing (gated by env var DMPO_PHASE2=1)
        import os as _os
        self._phase2 = _os.environ.get('DMPO_PHASE2', '0') == '1'

        # Safety parameters
        if self._phase2:
            # PHASE 2: Rebalanced — collision penalty reduced from -50 to -20.
            # Intermediate penalties scaled by 0.4 (e.g., headway -5 → -2,
            # PET -10 → -4) to bring max penalty (~-8.7 weighted) closer to
            # max bonus (~+0.85 weighted) — 10x asymmetry instead of 25x.
            self.collision_penalty = -20.0
            self._penalty_scale = 0.4
            self._bonus_scale = 1.5  # Mildly boost bonuses too
        else:
            # PHASE 1 and earlier: original values
            self.collision_penalty = -50.0
            self._penalty_scale = 1.0
            self._bonus_scale = 1.0

        # Progress parameters
        # Default 16.67 m/s (60 km/h) matches training. Override via
        # DMPO_MAX_SPEED_MPS env var for eval-only OOD experiments.
        self.max_speed = float(os.environ.get('DMPO_MAX_SPEED_MPS', '16.67'))
        self.max_distance = 200.0   # m

        # State tracking
        self.prev_accelerations = {}  # vehicle_id -> prev acceleration
        self.prev_lane_positions = {}  # vehicle_id -> prev lane position
        self.vehicles_in_intersection = set()

        # Intersection center (detected at runtime)
        self.intersection_center = None

    def _cached(self, veh_id, attr):
        """Read vehicle attribute from cache, fall back to SUMO API."""
        if self._vehicle_cache and veh_id in self._vehicle_cache:
            return self._vehicle_cache[veh_id].get(attr)
        if attr == 'position':
            return traci.vehicle.getPosition(veh_id)
        elif attr == 'speed':
            return traci.vehicle.getSpeed(veh_id)
        elif attr == 'acceleration':
            return traci.vehicle.getAcceleration(veh_id)
        elif attr == 'angle':
            return traci.vehicle.getAngle(veh_id)
        elif attr == 'lane_id':
            return traci.vehicle.getLaneID(veh_id)
        elif attr == 'type_id':
            return traci.vehicle.getTypeID(veh_id)
        elif attr == 'lane_position':
            return traci.vehicle.getLanePosition(veh_id)
        elif attr == 'edge_id':
            state = self._vehicle_cache.get(veh_id, {}) if self._vehicle_cache else {}
            return state.get('edge_id', '')
        return None

    def _ramp_weight(self, speed):
        """REWARD V2 (§4): shared movement ramp weight — 0 at ≤lo m/s, 1 at ≥hi m/s.

        Used by BOTH movement_bonus (scaled by w) and idle_penalty (scaled by
        1-w) so the two terms cross-fade over ONE band instead of flipping in
        opposite directions at the same 0.5 m/s threshold (the legacy
        0.11–0.15/step reward cliff across an infinitesimal speed change).
        Only called when self._v2_ramp is on.
        """
        span = max(self._v2_ramp_hi - self._v2_ramp_lo, 1e-6)
        return min(max((speed - self._v2_ramp_lo) / span, 0.0), 1.0)

    def _idle_exempt(self, vehicle_id):
        """REWARD V2 (§5): is this stationary step a LEGITIMATE yield?

        Exemption sources (each flag-gated, ALL default OFF; red-light waits are
        handled separately in the caller with legacy semantics):
          - genuine crossing conflict where EGO must yield (ta > 0, i.e. ego
            arrives second) and the conflict is imminent (PET <
            DMPO_IDLE_EXEMPT_PET_S). Consumes the SAME _crossing_conflict
            signal as obs/shield; an all-zero no-op unless the conflict
            geometry is live (obs-v2 / shielded profiles).
          - scaffold hold: env-published set self._shield_held (obs-v2 env batch).
          - post-collision freeze: env-published set self._stuck_ai (obs-v2 env
            batch) — the vehicle is force-stopped by the env and already paid
            the collision penalty; fining idleness again is double-charging.
        Every exemption is duration-capped (DMPO_IDLE_EXEMPT_MAX_STEPS): past
        the cap the idle penalty resumes at full ramp. Without this cap an
        N-way priority cycle is reward-free forever and the r6 explicit-priority
        gridlock reproduces — the cap is the anti-gridlock guard.
        """
        exempt = False
        if self._v2_idle_yield_exempt and self._obs_processor is not None:
            try:
                # F1 fix (soundness review, HIGH): consume the SAME v2 conflict block
                # the policy sees, not the shield's _crossing_conflict. The shield
                # model computes TTCP from the ACTUAL speed, so a yielder that has
                # STOPPED erases the very conflict signal that justifies its stop —
                # the exemption structurally could not fire for the exact case it
                # exists for. The v2 block uses the symmetric counterfactual arrival
                # floor (F32), so a stopped yielder still reads a genuine conflict.
                # Falls back to the legacy signal only if the v2 block is unavailable.
                _op = self._obs_processor
                if getattr(_op, '_v2', False):
                    cb = _op._conflict_block_v2(vehicle_id)
                    # slot 0 = most urgent (overlap-first ordering):
                    #   [0] ttcp_ego  [2] occ_gap (/10 s, negative = overlap)
                    #   [6] yield_sign (+1 = ego arrives later ⇒ ego must yield)
                    #   [29] n_conflicts
                    gate = self._v2_idle_exempt_pet_s / 10.0     # same /10 normalization
                    if (float(cb[29]) > 0.0 and float(cb[6]) > 0.0
                            and float(cb[2]) < gate):
                        exempt = True
                else:
                    cc = _op._crossing_conflict(vehicle_id)
                    if (cc and cc.get('n', 0) > 0 and cc.get('ta', 0.0) > 0.0
                            and cc.get('pet', float('inf')) < self._v2_idle_exempt_pet_s):
                        exempt = True
            except Exception:
                pass
        if not exempt and self._v2_idle_exempt_shield:
            held = getattr(self, '_shield_held', None)
            if held and vehicle_id in held:
                exempt = True
        if not exempt and self._v2_idle_exempt_stuck:
            frozen = getattr(self, '_stuck_ai', None)
            if frozen and vehicle_id in frozen:
                exempt = True
        if exempt:
            cnt = self._exempt_counter.get(vehicle_id, 0) + 1
            self._exempt_counter[vehicle_id] = cnt
            if cnt >= self._v2_idle_exempt_max_steps:
                exempt = False  # cap exceeded: penalty resumes (gridlock guard)
        return exempt

    def compute_reward(self, vehicle_id: str, observation: Dict[str, np.ndarray],
                      action: Dict[str, np.ndarray], prev_observation: Dict[str, np.ndarray],
                      prev_action: Dict[str, np.ndarray]) -> float:
        """Compute reward for a vehicle.

        TRIAL MODE (DMPO_REWARD_TRIAL=1): Simplified 3-term reward.
        The complex 4-component reward with 12 thresholds creates loopholes
        that the policy exploits (crawling at 1.1 m/s to game thresholds).

        The simplified reward has NO thresholds to game:
          reward = speed/max_speed + 50*crossed - 20*collision

        SUMO's Krauss model (setSpeedMode=7) handles physics-level safety.
        The policy's job is coordination: when to go, when to yield.
        """
        try:
            if not vehicle_id:
                return 0.0

            if self._vehicle_cache and vehicle_id not in self._vehicle_cache:
                return 0.0

            # ── CURVE-SPEED term (gated, DMPO_CURVE_SPEED_K=0 default = OFF) ──
            # Severity forensics (2026-08-08): the two most severe residual events
            # are left-turn-across-path conflicts taken at essentially full speed;
            # nothing in the reward prices lateral acceleration, so turns are
            # taken flat-out. Price actual lateral acceleration a_lat = v·|Δψ|/dt
            # above a comfort bound (default 3.0 m/s^2). Physically principled;
            # zero geometry lookup; per-episode state reset with the epoch cache.
            _curve_k = float(os.environ.get('DMPO_CURVE_SPEED_K', '0'))
            _curve_pen = 0.0
            if _curve_k > 0.0:
                try:
                    _st = self._vehicle_cache.get(vehicle_id, {}) if self._vehicle_cache else {}
                    _ang = float(_st.get('angle', 0.0)); _spd = float(_st.get('speed', 0.0))
                    if not hasattr(self, '_curve_prev_ang'):
                        self._curve_prev_ang = {}
                    _pa = self._curve_prev_ang.get(vehicle_id)
                    self._curve_prev_ang[vehicle_id] = _ang
                    # v2 guards (curve-arm v1 forensics): spawn-settling exemption
                    # (insertion heading transients) and a per-step penalty cap so
                    # angle-jump artifacts cannot dominate the episode return.
                    if not hasattr(self, '_curve_seen_steps'):
                        self._curve_seen_steps = {}
                    _n_seen = self._curve_seen_steps.get(vehicle_id, 0) + 1
                    self._curve_seen_steps[vehicle_id] = _n_seen
                    if _pa is not None and _spd > 1.0 and _n_seen > 3:
                        import math as _m
                        _d = abs(_ang - _pa)
                        if _d > 180.0: _d = 360.0 - _d
                        _alat = _spd * _m.radians(_d) / 0.2
                        _amax = float(os.environ.get('DMPO_CURVE_ALAT_MAX', '3.0'))
                        if _alat > _amax:
                            _cap = float(os.environ.get('DMPO_CURVE_PEN_CAP', '10.0'))
                            _curve_pen = -min(_curve_k * (_alat - _amax) ** 2, _cap)
                except Exception:
                    _curve_pen = 0.0

            # Detect intersection center on first call
            if self.intersection_center is None:
                try:
                    self.intersection_center = traci.junction.getPosition('center')
                except Exception as e:
                    logger.warning(
                        "Intersection 'center' junction lookup failed (%s); "
                        "falling back to (0.0, 0.0). PET and distance-to-intersection "
                        "features will be wrong for non-center-named scenarios.",
                        e,
                    )
                    self.intersection_center = (0.0, 0.0)

            # SPARSE: pure completion-only reward (Path F).
            # Returns 0 every step; -collision_penalty on collision. The
            # completion bonus is injected by sumo_intersection_env.step()
            # because completed vehicles aren't in observations here.
            if self._sparse:
                return self._compute_sparse_reward(vehicle_id)

            # CARL: route progress + centering (Path E, CaRL CoRL 2025)
            if self._carl:
                return self._compute_carl_reward(vehicle_id) + self._gap_shaping(vehicle_id, observation) + _curve_pen

            # ACCOUNTABLE: every action dim has a reward term (Path D)
            if self._accountable:
                return self._compute_accountable_reward(vehicle_id)

            # LITERATURE: 3-term reward aligned with Liu et al. 2024, highway-env
            if self._literature:
                return self._compute_literature_reward(vehicle_id)

            # NORMALIZED: non-negative reward (highway-env style, all in [0,1])
            if self._normalized:
                return self._compute_normalized_reward(vehicle_id)

            # REBALANCED: 4-component with reduced penalties
            if self._rebalanced:
                return self._compute_rebalanced_reward(vehicle_id)

            # TRIAL: simplified 3-term reward
            if self._trial:
                return self._compute_simple_reward(vehicle_id)

            # Original: 4-component weighted reward
            safety_reward = self._compute_safety(vehicle_id)
            progress_reward = self._compute_progress(vehicle_id)
            comfort_reward = self._compute_comfort(vehicle_id)
            cooperation_reward = self._compute_cooperation(vehicle_id)

            total_reward = (
                self.weights['safety'] * safety_reward +
                self.weights['progress'] * progress_reward +
                self.weights['comfort'] * comfort_reward +
                self.weights['cooperation'] * cooperation_reward
            )

            return np.clip(total_reward + _curve_pen, -50.0, 50.0)

        except Exception as e:
            logger.debug(f"Reward computation failed for vehicle {vehicle_id}: {e}")
            return 0.0

    def _gap_shaping(self, vehicle_id, observation):
        """PART 3: light PATH-AWARE gap-acceptance shaping (weak point #1). Reads the SAME conflict_gap
        block the policy sees (path-aware pairwise TTCP, not the wrong Euclidean CROSS_NEARMISS). Softly
        penalizes a YIELDING vehicle (arrives later) for carrying speed into a GENUINE imminent conflict,
        so it learns to slow for real conflicts and flow otherwise. Gated DMPO_GAP_REWARD_K (default 0 =
        OFF). Deliberately NEVER compels a full stop and never penalizes the priority vehicle -> avoids
        the r6 hard-priority GRIDLOCK. Only a nudge; the collision penalty + clean obs do the heavy work."""
        K = float(os.environ.get('DMPO_GAP_REWARD_K', '0.0'))
        if K <= 0.0 or observation is None:
            return 0.0
        cg = observation.get('conflict_gap')
        if cg is None or len(cg) < 4:
            return 0.0
        if len(cg) >= 31:                             # OBS-V2 layout (31, or 47 with profile appendix; indices 0..30 preserved)
            ys0 = float(cg[6]); te0 = float(cg[0])    # yield_sign, ttcp_ego (/10s)
        else:                                         # v1 layout (3×4+2 = 14)
            ys0 = float(cg[3]); te0 = float(cg[1])    # yield_sign, ttcp_ego (normalized /10s)
        if ys0 <= 0.0 or te0 >= 0.5:                  # only when EGO must yield AND conflict <5s away
            return 0.0
        speed = self._cached(vehicle_id, 'speed') or 0.0
        return -K * (speed / 16.67) * (0.5 - te0)     # penalty grows with speed+imminence; 0 when slow

    # =========================================================================
    # SIMPLIFIED REWARD (trial mode) — 3 terms, no thresholds
    # =========================================================================

    def _compute_simple_reward(self, vehicle_id: str) -> float:
        """Simplified reward: speed + crossing bonus - collision penalty.

        No thresholds to game. No loopholes. SUMO handles safety physics.
        The policy's job is coordination (when to go, when to yield).

        Returns:
            float: reward for this step
        """
        try:
            speed = self._cached(vehicle_id, 'speed') or 0.0
            reward = 0.0

            # 1. Speed reward: continuous, proportional to speed limit
            #    Stopped = 0, speed limit = 1.0. No threshold.
            reward += speed / self.max_speed

            # 2. Crossing bonus: +50 when vehicle exits intersection
            #    Must track intersection entry/exit manually since the
            #    original _compute_progress (which tracked this) is skipped.
            if self._is_in_intersection(vehicle_id):
                self.vehicles_in_intersection.add(vehicle_id)
            if self._just_exited_intersection(vehicle_id):
                reward += 50.0

            # 3. Collision penalty: -20 for actual physical overlap
            if self._is_collision(vehicle_id):
                reward += -20.0

            return reward

        except Exception as e:
            logger.debug(f"Simple reward error for {vehicle_id}: {e}")
            return 0.0

    # =========================================================================
    # LITERATURE-ALIGNED REWARD — 3 terms, collision:completion = 1:2
    # Based on Liu et al. 2024, highway-env (Leurent), Risk-Aware 2025
    # =========================================================================

    def _compute_literature_reward(self, vehicle_id: str) -> float:
        """Literature-aligned reward: speed + completion - collision.

        Collision:completion ratio = 1:2 (matches published MARL intersection papers).
        Dense speed reward gives per-step gradient signal to move.
        No thresholds — SUMO's Krauss model handles physics safety.

        References:
            - Liu et al. 2024 (IEEE T-ITS): collision=-10, arrival=+5
            - highway-env: collision=-5, arrival=+1, speed in [0,1]
            - Risk-Aware 2025: collision=-50, goal=+100
        """
        try:
            speed = self._cached(vehicle_id, 'speed') or 0.0
            reward = 0.0

            # 1. Speed reward: dense, per-step, proportional to speed limit
            #    Stopped = 0, speed limit = 1.0
            reward += speed / self.max_speed

            # 2. Completion bonus: +10 when vehicle exits intersection
            if self._is_in_intersection(vehicle_id):
                self.vehicles_in_intersection.add(vehicle_id)
            if self._just_exited_intersection(vehicle_id):
                reward += 10.0

            # 3. Collision penalty for actual physical overlap
            if self._is_collision(vehicle_id):
                reward -= self._literature_collision

            return reward

        except Exception as e:
            logger.debug(f"Literature reward error for {vehicle_id}: {e}")
            return 0.0

    # =========================================================================
    # =========================================================================
    # PATH E: CaRL-STYLE ROUTE PROGRESS + CENTERING
    # Based on CaRL (CoRL 2025): reward = route progress, no speed farming
    # =========================================================================

    def _compute_carl_reward(self, vehicle_id: str) -> float:
        """CaRL-style reward: progress toward destination + centering.

        r = Δroute_fraction + 0.2 * centering - collision

        The ONLY way to get reward is to make progress along the route.
        Standing still = 0 reward. No speed farming possible.
        Total route progress sums to ~1.0 for a complete trip.
        Scaled by 10 so a full crossing ≈ +10 (comparable to other rewards).

        References:
            CaRL (Jaeger et al., CoRL 2025, arXiv 2504.17838)
        """
        try:
            reward = 0.0

            # 1. Route progress (longitudinal feedback)
            #    BUG FIX: traci.vehicle.getDistance() returns None under libsumo
            #    for AI vehicles, silently zeroing the progress reward across
            #    every phase before this fix. Use cached speed × step_length
            #    instead — robust, libsumo-safe, doesn't require special API.
            progress = 0.0
            try:
                spd = self._cached(vehicle_id, 'speed')
                if self._v2_trapezoid and spd is not None:
                    # REWARD V2 (§6a): trapezoid odometer 0.5·(v_prev+v)·dt matches
                    # SUMO's ballistic position update; the legacy rectangle rule
                    # (v·dt) under-credits hard braking by ½·|a|·dt² per step — a
                    # small but systematic reward-side bias AGAINST exactly the
                    # braking we want. Same +10.0 integral over a full trip.
                    step_length = float(os.environ.get('DMPO_STEP_LENGTH', '0.2'))
                    route_length = float(os.environ.get('DMPO_ROUTE_LENGTH_M', '380.0'))
                    prev_spd = self._prev_speed.get(vehicle_id, spd)
                    self._prev_speed[vehicle_id] = spd
                    delta_dist = 0.5 * (prev_spd + spd) * step_length
                    if delta_dist > 0:
                        progress = (delta_dist / route_length) * 10.0
                elif spd is not None and spd > 0:
                    # P1.3: previously hardcoded 0.2 — for variant E
                    # (DMPO_STEP_LENGTH=0.1, 10 Hz) this gave 2× the progress
                    # reward per sim-second relative to physical motion.
                    step_length = float(os.environ.get('DMPO_STEP_LENGTH', '0.2'))
                    delta_dist = spd * step_length
                    # Trip length (approach + exit) normalizes progress so a full
                    # crossing ≈ +10. Scenario-dependent: Bremen 2-lane 190m arms →
                    # ~380m; the 4-lane 200m arms → ~400m. Env-tunable so the reward
                    # magnitude is correct per geometry (DMPO_ROUTE_LENGTH_M).
                    route_length = float(os.environ.get('DMPO_ROUTE_LENGTH_M', '380.0'))
                    progress = (delta_dist / route_length) * 10.0
            except Exception:
                progress = 0.0

            # WAVE-4 COMFORT TERM (gated DMPO_COMFORT_DECEL_K, default 0 = OFF):
            # prices harsh deceleration so early-smooth yielding beats the
            # individually-optimal late-hard yield that manufactures the
            # stop-line rear-end class (55-75% of the champion's residual at
            # every density — the leader's braking style is the SOURCE; this
            # term rewires the leader's incentive). Penalty = K*(decel-THRESH)
            # per step above THRESH m/s²; the collision penalty (−400) still
            # dominates emergencies (max excess ~5 → −0.5/step at K=0.1).
            try:
                _ck = float(os.environ.get('DMPO_COMFORT_DECEL_K', '0') or 0)
                if _ck > 0.0:
                    _cth = float(os.environ.get('DMPO_COMFORT_DECEL_THRESH', '3.0'))
                    _sl = float(os.environ.get('DMPO_STEP_LENGTH', '0.2'))
                    _s = self._cached(vehicle_id, 'speed')
                    if _s is not None:
                        if not hasattr(self, '_comfort_prev'):
                            self._comfort_prev = {}
                        _p = self._comfort_prev.get(vehicle_id, _s)
                        self._comfort_prev[vehicle_id] = _s
                        _dec = (_p - _s) / max(_sl, 1e-6)
                        if _dec > _cth:
                            progress -= _ck * (_dec - _cth)
            except Exception:
                pass

            # PROGRESS+IDLE MODE: when DMPO_REWARD_PROGRESS_ONLY=1, return
            # progress + idle_penalty (when stopped) + nothing else.
            # Rationale: pure progress alone produces a 0-baseline value
            # function that drowns rare positives under PPO clipping.
            # Adding idle_penalty creates a clearly-negative baseline for
            # "stay still" so the policy gradient has a non-zero direction.
            if os.environ.get('DMPO_REWARD_PROGRESS_ONLY', '0') == '1':
                idle_pen = 0.0
                try:
                    spd = self._cached(vehicle_id, 'speed')
                    if spd is not None and spd < 0.5:
                        # Ramp from -0.01 (1 step stuck) to -0.05 (>=20 steps).
                        stuck = self._stuck_counter.get(vehicle_id, 0) + 1
                        self._stuck_counter[vehicle_id] = stuck
                        ramp = min(stuck / 20.0, 1.0)
                        idle_pen = -0.01 - 0.04 * ramp
                    else:
                        self._stuck_counter[vehicle_id] = 0
                except Exception:
                    pass
                return progress + idle_pen

            # 2. Lane centering (lateral feedback)
            #    0.01 coefficient — must be smaller than route progress
            centering = 0.0
            try:
                lat_pos = traci.vehicle.getLateralLanePosition(vehicle_id)
                lane_id = self._cached(vehicle_id, 'lane_id')
                if lane_id:
                    lane_width = traci.lane.getWidth(lane_id)
                    deviation = min(abs(lat_pos) / (lane_width / 2), 1.0)
                    centering = 0.01 * (1.0 - deviation)
                else:
                    centering = 0.005
            except Exception:
                centering = 0.005

            # 3. Idle penalty — ramps with consecutive stuck steps.
            #    Fires whenever the vehicle is below crawling speed AND is not
            #    legitimately waiting at a red light. Ramp makes the penalty
            #    grow with persistence so PPO has a gradient pointing toward
            #    "try to escape." Exempt only red-light waits within 50 m.
            idle_penalty = 0.0
            try:
                speed = self._cached(vehicle_id, 'speed')
                if self._v2_ramp:
                    # REWARD V2 (§4+§5): the idle penalty fades over the SAME
                    # ramp band the movement bonus rises over (no 0.5 m/s cliff)
                    # and a genuinely-yielding vehicle can be exempted (red
                    # light / imminent crossing conflict / scaffold hold /
                    # post-collision freeze), duration-capped in _idle_exempt.
                    if speed is None:
                        # unknown speed: mirror legacy (treated as not-idle —
                        # counter reset, no penalty)
                        self._stuck_counter[vehicle_id] = 0
                        w_move = 1.0
                    else:
                        w_move = self._ramp_weight(speed)
                    if w_move >= 1.0:
                        # Soundness review (reward MED): resetting on ANY w_move>0 let a
                        # creeping vehicle (inching above 0.3 m/s) defeat the anti-gridlock
                        # cap forever. Reset only on GENUINE movement (above the full ramp,
                        # >= 0.7 m/s sustained this step).
                        self._exempt_counter[vehicle_id] = 0
                    if w_move < 1.0:
                        at_red = False
                        try:
                            next_tls = traci.vehicle.getNextTLS(vehicle_id)
                            if next_tls:
                                tls_state = next_tls[0][3]
                                tls_dist = next_tls[0][2]
                                if tls_state in ('r', 'R') and tls_dist < 50.0:
                                    at_red = True
                        except Exception:
                            pass
                        if at_red:
                            # red-light wait keeps legacy semantics: exempt + reset ramp
                            self._stuck_counter[vehicle_id] = 0
                        elif not self._idle_exempt(vehicle_id):
                            stuck = self._stuck_counter.get(vehicle_id, 0) + 1
                            self._stuck_counter[vehicle_id] = stuck
                            # -0.01 at step 1, ramps to -0.05 by step 20; scaled
                            # by (1-w) so it fades out across the ramp band.
                            ramp = min(stuck / 20.0, 1.0)
                            idle_penalty = (-0.01 - 0.04 * ramp) * (1.0 - w_move)
                        # while exempt: the stuck ramp FREEZES (neither incremented
                        # nor reset) and resumes where it left off — anti-freeze
                        # pressure survives the exemption window.
                    else:
                        self._stuck_counter[vehicle_id] = 0
                elif speed is not None and speed < 0.5:
                    at_red = False
                    try:
                        next_tls = traci.vehicle.getNextTLS(vehicle_id)
                        if next_tls:
                            tls_state = next_tls[0][3]
                            tls_dist = next_tls[0][2]
                            if tls_state in ('r', 'R') and tls_dist < 50.0:
                                at_red = True
                    except Exception:
                        pass
                    if not at_red:
                        stuck = self._stuck_counter.get(vehicle_id, 0) + 1
                        self._stuck_counter[vehicle_id] = stuck
                        # -0.01 at step 1, ramps to -0.05 by step 20, then capped.
                        ramp = min(stuck / 20.0, 1.0)
                        idle_penalty = -0.01 - 0.04 * ramp
                    else:
                        self._stuck_counter[vehicle_id] = 0
                else:
                    self._stuck_counter[vehicle_id] = 0
            except Exception:
                pass

            # 4. Overlap penalty
            #    Vehicles pressed against their leader (gap <= 0.5 m).
            #    Fires even when commanded to move — teaches "don't shove
            #    into the car in front." Broader condition than strict
            #    overlap so the 0 m gridlock case gets penalized.
            overlap_penalty = 0.0
            try:
                leader = traci.vehicle.getLeader(vehicle_id)
                if leader is not None and leader[1] <= 0.5:
                    overlap_penalty = self._overlap_penalty_value
            except Exception:
                pass

            # 5. Proximity cost
            #    Graduated cost for being close to leader — smooth signal that
            #    teaches safe following distance BEFORE overlap occurs.
            #    Uses the same thresholds as collision_warning observation so
            #    the policy can correlate observation → cost.
            proximity_cost = 0.0
            try:
                leader = traci.vehicle.getLeader(vehicle_id)
                if leader is not None:
                    dist = leader[1]
                    if dist < 0:
                        proximity_cost = -0.02      # overlapping
                    elif dist < 2.0:
                        proximity_cost = -0.02      # imminent
                    elif dist < 5.0:
                        proximity_cost = -0.014     # danger
                    elif dist < 10.0:
                        proximity_cost = -0.008     # caution
                    elif dist < 20.0:
                        proximity_cost = -0.002     # awareness
            except Exception:
                pass

            # 5b. TTC-SHAPING COST (NEW — V6)
            #     The proximity_cost above is DISTANCE-based, miscalibrated at
            #     variable closing speeds. This term is TTC-based and an order
            #     of magnitude stronger, designed to drive the policy gradient
            #     toward active braking BEFORE the collision happens.
            #     Formula:
            #         ttc = distance / max(closing_speed, 0.1)
            #         if ttc < TTC_THRESH (default 3s):
            #             cost = -K * (TTC_THRESH - ttc) / TTC_THRESH
            #     K=0 disables (V2 and below). K=1.0 ≈ -1.0/step at ttc=0,
            #     -0.33/step at ttc=2s. Compounded over a 30-step cascade,
            #     -10 to -30 reward — much stronger than proximity_cost.
            #     Reuses leader from above to avoid extra traci call.
            ttc_shaping = 0.0
            ttc_K = float(os.environ.get('DMPO_TTC_SHAPING_K', '0.0'))
            if ttc_K > 0.0:
                try:
                    if leader is not None:
                        leader_id, dist = leader
                        if dist >= 0:
                            ttc_thresh = float(os.environ.get('DMPO_TTC_SHAPING_THRESH_S', '3.0'))
                            ego_speed = float(self._cached(vehicle_id, 'speed') or 0.0)
                            leader_speed = float(self._cached(leader_id, 'speed') or 0.0)
                            closing = max(ego_speed - leader_speed, 0.1)
                            ttc = dist / closing
                            if ttc < ttc_thresh:
                                ttc_shaping = -ttc_K * (ttc_thresh - ttc) / ttc_thresh
                except Exception:
                    pass

            # 7. Movement bonus
            #    Per-step positive reward when the AI is actually moving.
            #    Counters the "do nothing is optimal" attractor by giving a
            #    small but non-zero baseline reward for forward motion.
            #    Tuned to be SMALLER than progress reward so completion
            #    remains the dominant target, but large enough that
            #    standing still has a clear opportunity cost.
            movement_bonus = 0.0
            move_bonus_value = float(os.environ.get('DMPO_MOVEMENT_BONUS', '0.0'))
            if self._v2_ramp:
                # REWARD V2 (§4): bonus scales with the SAME ramp weight the
                # idle penalty fades over — no 0.5 m/s cliff. With the v2 value
                # (0.04) it stays < max progress/step (0.0834 @ 16.67 m/s) so
                # distance covered, not "being in motion", dominates.
                # DMPO_MOVEMENT_SPEED_THRESH is DEAD in this path (ramp replaces it).
                if move_bonus_value > 0:
                    cached_speed = self._cached(vehicle_id, 'speed')
                    if cached_speed is not None:
                        movement_bonus = move_bonus_value * self._ramp_weight(cached_speed)
            else:
                move_speed_thresh = float(os.environ.get('DMPO_MOVEMENT_SPEED_THRESH', '0.5'))
                if move_bonus_value > 0:
                    # P0.20: read speed from per-step cache instead of issuing an
                    # extra TraCI call. Cache is shared with env via
                    # self._vehicle_cache (set in env._compute_rewards).
                    cached_speed = self._cached(vehicle_id, 'speed')
                    if cached_speed is not None and cached_speed >= move_speed_thresh:
                        movement_bonus = move_bonus_value

            # 7b. CROSSING-aware near-miss penalty (V2 — targets the conflict-point floor)
            #     proximity_cost / ttc_shaping above are LEADER-based (same lane), so
            #     they miss CROSSING-traffic junction conflicts — the dominant collision
            #     mechanism (AI-AI cross_traffic_junction). This penalizes proximity to
            #     the nearest vehicle of ANY lane, but only inside the junction zone so it
            #     shapes who-goes-first coordination, not open-road following. Uses the
            #     shared per-step vehicle cache (no extra SUMO calls; early-exit within
            #     radius). Env-tunable; DMPO_CROSS_NEARMISS_K=0 (default) = exact prior behaviour.
            cross_nearmiss = 0.0
            cross_K = float(os.environ.get('DMPO_CROSS_NEARMISS_K', '0.0'))
            if cross_K > 0.0 and self._vehicle_cache:
                try:
                    ego_pos = self._cached(vehicle_id, 'position')
                    if ego_pos is not None:
                        zone = float(os.environ.get('DMPO_CROSS_NEARMISS_ZONE_M', '60.0'))
                        in_zone = True
                        if self.intersection_center is not None:
                            cx, cy = self.intersection_center
                            in_zone = ((ego_pos[0] - cx) ** 2 + (ego_pos[1] - cy) ** 2) <= zone * zone
                        if in_zone:
                            radius = float(os.environ.get('DMPO_CROSS_NEARMISS_M', '8.0'))
                            r2 = radius * radius
                            min_d2 = r2
                            for oid, ost in self._vehicle_cache.items():
                                if oid == vehicle_id:
                                    continue
                                opos = ost.get('position')
                                if opos is None:
                                    continue
                                d2 = (ego_pos[0] - opos[0]) ** 2 + (ego_pos[1] - opos[1]) ** 2
                                if d2 < min_d2:
                                    min_d2 = d2
                            if min_d2 < r2:
                                cross_nearmiss = -cross_K * (radius - (min_d2 ** 0.5)) / radius
                except Exception:
                    cross_nearmiss = 0.0

            # 7d. RSS CONTINUOUS RISK (Bet E) — Responsibility-Sensitive-Safety
            #     longitudinal safe distance to the same-lane leader. Smoother than
            #     bare TTC: penalizes being INSIDE the RSS safe gap
            #         d_safe = v·ρ + v²/(2b) − v_lead²/(2b)     (floored at 0.5 m)
            #     where b = DMPO_MAX_DECEL (the braking the policy actually has) and
            #     ρ = reaction time. risk = (d_safe−gap)/d_safe ∈ (0,1]; reward −= K·risk.
            #     0 when the gap is safe OR the ego is slower than the leader (not
            #     closing). DMPO_RSS_RISK_K=0 (default) disables. Reuses `leader`.
            rss_risk_term = 0.0
            rss_K = float(os.environ.get('DMPO_RSS_RISK_K', '0.0'))
            if rss_K > 0.0:
                try:
                    if leader is not None:
                        leader_id, gap = leader
                        if gap >= 0:
                            ego_speed = float(self._cached(vehicle_id, 'speed') or 0.0)
                            leader_speed = float(self._cached(leader_id, 'speed') or 0.0)
                            rho = float(os.environ.get('DMPO_RSS_REACTION_S', '0.3'))
                            b = max(float(os.environ.get('DMPO_MAX_DECEL', '4.5')), 0.1)
                            d_safe = (ego_speed * rho
                                      + (ego_speed * ego_speed) / (2.0 * b)
                                      - (leader_speed * leader_speed) / (2.0 * b))
                            d_safe = max(d_safe, 0.5)   # minimum clearance floor
                            if gap < d_safe:
                                rss_risk_term = -rss_K * (d_safe - gap) / d_safe
                except Exception:
                    rss_risk_term = 0.0

            # 7e. DELAY PENALTY (Round 3, arm res_delay) — cost ∝ speed DEFICIT.
            #     idle_penalty only fires when STOPPED (<0.5 m/s); the round-2
            #     always-yield hack MOVES (capped following), it is merely slow
            #     through conflicts. This term makes "always capped" strictly
            #     dominated by "cap only when needed":
            #         delay_pen = -K · (1 - v/v_max)   per step
            #     DMPO_DELAY_PENALTY_K=0 (default) = exact prior behavior.
            delay_pen = 0.0
            delay_K = float(os.environ.get('DMPO_DELAY_PENALTY_K', '0.0'))
            if delay_K > 0.0:
                try:
                    spd = float(self._cached(vehicle_id, 'speed') or 0.0)
                    delay_pen = -delay_K * (1.0 - min(spd / max(self.max_speed, 0.1), 1.0))
                except Exception:
                    delay_pen = 0.0

            # 7f. CONFLICT-POINT shaping (DMPO_CONFLICT_OBS) — reuses the SAME crossing-conflict
            #     signal as the observation (obs_processor._crossing_conflict) so reward and obs
            #     stay consistent. (a) PET shaping [BOTH arms]: penalize being on a collision
            #     course (small PET at the true conflict point) → pushes the pair to open a safe
            #     time gap; because both crossers see a small PET, it is implicitly team-priced.
            #     (b) priority-compliance [ARM A only]: reward GOING when it's your turn (ta<0)
            #     and SLOWING when you must yield (ta>0). OFF in Arm B → convention is learned.
            pet_shaping = 0.0
            priority_comply = 0.0
            if (os.environ.get('DMPO_CONFLICT_OBS', '0') == '1'
                    and self._obs_processor is not None):
                try:
                    cc = self._obs_processor._crossing_conflict(vehicle_id)
                    if cc['n'] > 0:
                        v_norm = min(float(self._cached(vehicle_id, 'speed') or 0.0)
                                     / max(self.max_speed, 0.1), 1.0)
                        pet_K = float(os.environ.get('DMPO_PET_SHAPING_K', '0.0'))
                        pet_safe = float(os.environ.get('DMPO_PET_SAFE_S', '2.0'))
                        if pet_K > 0.0 and cc['pet'] < pet_safe:
                            pet_shaping = -pet_K * (pet_safe - cc['pet']) / pet_safe
                        prio_K = float(os.environ.get('DMPO_PRIORITY_COMPLY_K', '0.0'))
                        if prio_K > 0.0 and os.environ.get('DMPO_EXPLICIT_PRIORITY', '0') == '1':
                            # ta>0: ego arrives second -> reward being SLOW (yield);
                            # ta<0: ego arrives first  -> reward keeping SPEED (go).
                            priority_comply = (prio_K * (1.0 - v_norm) if cc['ta'] > 0.0
                                               else prio_K * v_norm)
                except Exception:
                    pet_shaping = 0.0
                    priority_comply = 0.0

            # 8. Final reward
            #    progress: +0.08 for driving forward
            #    centering: +0.01 for lane centering
            #    idle_penalty: -0.01 for stopped when should move
            #    overlap_penalty: -0.05 for being inside another vehicle
            #    proximity_cost: -0.002 to -0.02 for being close to leader (distance-based, weak)
            #    ttc_shaping: -K·(thresh-ttc)/thresh when ttc < thresh (TTC-based, same-lane)
            #    cross_nearmiss: -K·(r-d)/r for nearest CROSS-lane vehicle in junction zone (V2)
            #    rss_risk_term: -K·(d_safe-gap)/d_safe — RSS continuous following risk (Bet E)
            #    movement_bonus: +0.1 (or env-set) for actually moving forward
            #    delay_pen: -K·(1-v/vmax) speed-deficit cost (Round 3)
            reward = (progress + centering + idle_penalty + overlap_penalty + proximity_cost
                      + ttc_shaping + cross_nearmiss + rss_risk_term + movement_bonus + delay_pen
                      + pet_shaping + priority_comply)

            return reward

        except Exception as e:
            logger.debug(f"CaRL reward error for {vehicle_id}: {e}")
            return 0.0

    def _reset_carl_state(self):
        """Reset CaRL route progress tracking."""
        self._prev_route_frac = {}
        self._stuck_counter = {}
        self._prev_speed = {}
        self._exempt_counter = {}

    # =========================================================================
    # PATH F: PURE SPARSE REWARD — completion-only
    # Eliminates the per-step progress reward entirely. The only positive
    # signal is a one-time completion bonus injected by env.step(). Collisions
    # incur a small penalty so crashing-to-terminate is net negative.
    #
    # Motivation: dense per-step progress rewards can be exploited by a
    # policy that maximises forward velocity without ever finishing the
    # route (the "drive forward, never finish" local optimum observed in
    # s1337). Sparse signals force the policy to learn the terminal goal.
    # =========================================================================

    def _compute_sparse_reward(self, vehicle_id: str) -> float:
        """Pure sparse reward: 0 per step.

        Completion bonus (+50) is added by sumo_intersection_env.step()
        because completed vehicles are no longer in observations when
        this runs. Collision penalty (if enabled) is also injected by
        env.step() using SUMO's getCollidingVehiclesIDList(), NOT the
        proximity heuristic in _is_collision (which fires every step a
        vehicle is within 2m and produces thousands of false positives
        in dense traffic).
        """
        return 0.0

    # =========================================================================
    # PATH D: ACTION-ACCOUNTABLE — every action dim gets a reward term
    # Based on ROAD 2025 (centering) + Liu et al. 2024 (speed + completion)
    # =========================================================================

    def _compute_accountable_reward(self, vehicle_id: str) -> float:
        """Action-accountable reward: every action dim has feedback.

        - longitudinal → speed/v_max (dense, 0 to 1.0)
        - lateral → lane centering (dense, 0 to 0.2)
        - completion → +10 (sparse goal signal)
        - collision → -5 (sparse safety signal)

        Total per-step range (no events): 0.0 to 1.2
        """
        try:
            speed = self._cached(vehicle_id, 'speed') or 0.0
            reward = 0.0

            # 1. Speed reward (longitudinal feedback)
            reward += speed / self.max_speed

            # 2. Lane centering reward (lateral feedback)
            #    Centered = 0.01, at lane edge = 0.0
            #    Must be SMALLER than route progress (0.088 at max speed)
            #    to avoid centering dominance (see D3 in issues doc)
            try:
                lat_pos = traci.vehicle.getLateralLanePosition(vehicle_id)
                lane_id = self._cached(vehicle_id, 'lane_id')
                if lane_id:
                    lane_width = traci.lane.getWidth(lane_id)
                    deviation = abs(lat_pos) / (lane_width / 2)  # 0=centered, 1=edge
                    deviation = min(deviation, 1.0)
                    reward += 0.01 * (1.0 - deviation)
                else:
                    reward += 0.005  # default if lane unknown
            except Exception:
                reward += 0.005  # default if lateral info unavailable

            # 3. Completion bonus
            if self._is_in_intersection(vehicle_id):
                self.vehicles_in_intersection.add(vehicle_id)
            if self._just_exited_intersection(vehicle_id):
                reward += 10.0

            # 4. Collision penalty
            if self._is_collision(vehicle_id):
                reward -= 5.0

            return reward

        except Exception as e:
            logger.debug(f"Accountable reward error for {vehicle_id}: {e}")
            return 0.0

    # =========================================================================
    # PATH B: REBALANCED 4-COMPONENT — reduced penalties, higher completion
    # =========================================================================

    def _compute_rebalanced_reward(self, vehicle_id: str) -> float:
        """4-component reward with rebalanced magnitudes.

        Changes from original:
        - Collision: -50 → -10
        - Headway penalties: -5/-10 → -1/-2
        - PET penalties: -3/-10 → -1/-3
        - Speed reward: 1.0 → 3.0
        - Completion bonus: +20 (was ~1.0)
        - Comfort penalties halved
        """
        try:
            speed = self._cached(vehicle_id, 'speed') or 0.0
            reward = 0.0

            # Safety (reduced penalties)
            if self._is_collision(vehicle_id):
                reward -= 10.0
            else:
                leader_info = traci.vehicle.getLeader(vehicle_id)
                if leader_info is not None:
                    leader_id, distance = leader_info
                    if distance >= 0 and speed > 0.5:
                        headway = distance / speed
                        if headway < 1.0:
                            reward -= 2.0
                        elif headway < 2.0:
                            reward -= 1.0

                if self._is_approaching_intersection(vehicle_id) or self._is_in_intersection(vehicle_id):
                    pet = self._estimate_pet(vehicle_id)
                    if pet is not None:
                        if pet < 1.5:
                            reward -= 3.0
                        elif pet < 3.0:
                            reward -= 1.0

            # Progress (boosted)
            speed_ratio = speed / self.max_speed
            reward += speed_ratio * 3.0

            if self._is_in_intersection(vehicle_id):
                self.vehicles_in_intersection.add(vehicle_id)
                reward += 0.5
            if self._just_exited_intersection(vehicle_id):
                reward += 20.0

            # Comfort (halved)
            accel = self._cached(vehicle_id, 'acceleration') or 0.0
            prev_accel = self.prev_accelerations.get(vehicle_id, accel)
            jerk = abs(accel - prev_accel) / 0.2
            self.prev_accelerations[vehicle_id] = accel
            if jerk > 5.0:
                reward -= 0.5
            elif jerk < 1.0:
                reward += 0.25

            return np.clip(reward, -20.0, 30.0)

        except Exception as e:
            logger.debug(f"Rebalanced reward error for {vehicle_id}: {e}")
            return 0.0

    # =========================================================================
    # PATH C: NORMALIZED NON-NEGATIVE REWARD — all rewards in [0, 1]
    # Based on highway-env: "negative rewards encourage ending episodes early"
    # =========================================================================

    def _compute_normalized_reward(self, vehicle_id: str) -> float:
        """Non-negative reward normalized to [0, 1].

        Maps: collision → 0.0, stopped → 0.71, speed limit → 0.86, arrived → 1.0
        The policy can never "win" by doing nothing (0.71 < 1.0).
        """
        try:
            speed = self._cached(vehicle_id, 'speed') or 0.0
            raw = 0.0

            # Speed: 0 to 1
            raw += speed / self.max_speed

            # Completion: +1
            if self._is_in_intersection(vehicle_id):
                self.vehicles_in_intersection.add(vehicle_id)
            if self._just_exited_intersection(vehicle_id):
                raw += 1.0

            # Collision: -5
            if self._is_collision(vehicle_id):
                raw -= 5.0

            # Normalize from [-5, 2] to [0, 1]
            reward = np.clip((raw - (-5.0)) / (2.0 - (-5.0)), 0.0, 1.0)
            return reward

        except Exception as e:
            logger.debug(f"Normalized reward error for {vehicle_id}: {e}")
            return 0.5  # neutral default

    # =========================================================================
    # SAFETY (0.40) — "Am I actually in danger?"
    # =========================================================================

    def _compute_safety(self, vehicle_id: str) -> float:
        """Context-aware safety reward using time headway and PET."""
        safety_reward = 0.0

        try:
            # 1. Collision check
            if self._is_collision(vehicle_id):
                return self.collision_penalty

            speed = self._cached(vehicle_id, 'speed')
            lane_id = self._cached(vehicle_id, 'lane_id')

            # 2. Same-lane following — Time Headway
            leader_info = traci.vehicle.getLeader(vehicle_id)
            if leader_info is not None:
                leader_id, distance = leader_info
                if distance >= 0 and speed > 0.5:
                    headway = distance / speed  # seconds

                    if headway < 1.0:
                        safety_reward += -5.0 * self._penalty_scale
                    elif headway < 2.0:
                        safety_reward += -1.0 * self._penalty_scale
                    elif headway < 4.0:
                        safety_reward += 1.0 * self._bonus_scale
                    else:
                        safety_reward += 0.5 * self._bonus_scale
                elif speed <= 0.5 and distance >= 0:
                    # Stopped behind leader — safe, neutral
                    safety_reward += 0.0
            else:
                # No leader — open road, safe
                safety_reward += 0.5 * self._bonus_scale

            # 3. Cross-traffic conflict — PET estimate
            if self._is_approaching_intersection(vehicle_id) or self._is_in_intersection(vehicle_id):
                pet = self._estimate_pet(vehicle_id)
                if pet is not None:
                    if pet < 1.5:
                        safety_reward += -10.0 * self._penalty_scale
                    elif pet < 3.0:
                        safety_reward += -3.0 * self._penalty_scale
                    elif pet < 6.0:
                        safety_reward += 0.5 * self._bonus_scale
                    # PET > 6.0: no interaction, no reward

            # 4. Rule compliance
            try:
                speed_limit = traci.lane.getMaxSpeed(lane_id)
                if speed > speed_limit * 1.1:
                    safety_reward += -2.0 * self._penalty_scale
            except:
                pass

        except Exception as e:
            logger.debug(f"Safety reward error for {vehicle_id}: {e}")

        return safety_reward

    def _is_collision(self, vehicle_id: str) -> bool:
        """Check for collision using physical overlap criteria.

        RED FLAG 9 FIX (applied in both Phase 1 and Phase 2): Previous version
        flagged ANY two vehicles within 1.5m as colliding, regardless of lane.
        This caused -50 penalties during normal dense traffic where vehicles
        in parallel lanes are 1.5m apart.

        New logic: collision requires vehicles to actually overlap in physical
        space, accounting for vehicle length (4.5m) and width (1.8m typical).
        Two vehicles collide only if their bounding boxes overlap, which is
        approximated by:
          - Same lane AND gap < 1.0m, OR
          - Cross lanes AND euclidean distance < 1.0m (intersection overlap)
        """
        try:
            position = self._cached(vehicle_id, 'position')
            ego_lane = self._cached(vehicle_id, 'lane_id')
            if not self._vehicle_cache:
                return False

            for other_id, other_state in self._vehicle_cache.items():
                if other_id == vehicle_id:
                    continue
                other_pos = other_state.get('position')
                if other_pos is None:
                    continue
                dx = position[0] - other_pos[0]
                dy = position[1] - other_pos[1]
                dist_sq = dx * dx + dy * dy

                # Strict collision: vehicles physically overlapping (< 1m)
                # 1m is half the vehicle length, so this requires actual contact
                if dist_sq < 1.0:
                    return True

                # Same-lane collision: closer than half the minimum gap
                other_lane = other_state.get('lane_id', '')
                if ego_lane and ego_lane == other_lane and dist_sq < 4.0:  # < 2m gap same lane
                    return True

            return False
        except:
            return False

    def _estimate_pet(self, vehicle_id: str) -> Optional[float]:
        """Estimate Post-Encroachment Time with nearest cross-traffic vehicle."""
        try:
            ego_pos = self._cached(vehicle_id, 'position')
            ego_speed = self._cached(vehicle_id, 'speed')
            ego_edge = self._cached(vehicle_id, 'edge_id')

            if ego_speed < 0.5:
                return None  # Stopped, no conflict

            cx, cy = self.intersection_center

            # Time for ego to reach intersection center
            ego_dist = math.sqrt((ego_pos[0] - cx)**2 + (ego_pos[1] - cy)**2)
            ego_time = ego_dist / max(ego_speed, 0.1)

            best_pet = None

            if self._vehicle_cache:
                for other_id, other_state in self._vehicle_cache.items():
                    if other_id == vehicle_id:
                        continue
                    other_edge = other_state.get('edge_id', '')

                    # Only check cross-traffic (different approach, both approaching)
                    if (other_edge == ego_edge or
                        not other_edge or not ego_edge or
                        'to_center' not in ego_edge or
                        'to_center' not in other_edge):
                        continue

                    other_pos = other_state['position']
                    other_speed = other_state.get('speed', 0)
                    if other_speed < 0.5:
                        continue

                    other_dist = math.sqrt((other_pos[0] - cx)**2 + (other_pos[1] - cy)**2)
                    other_time = other_dist / max(other_speed, 0.1)

                    pet = abs(ego_time - other_time)

                    if best_pet is None or pet < best_pet:
                        best_pet = pet

            return best_pet

        except:
            return None

    # =========================================================================
    # PROGRESS (0.30) — "Am I making progress toward my destination?"
    # =========================================================================

    def _compute_progress(self, vehicle_id: str) -> float:
        """Continuous route progress with context-aware delay penalty."""
        progress_reward = 0.0

        try:
            lane_pos = self._cached(vehicle_id, 'lane_position')
            speed = self._cached(vehicle_id, 'speed')
            lane_id = self._cached(vehicle_id, 'lane_id')

            # 1. Route progress — continuous per-step
            prev_lane_pos = self.prev_lane_positions.get(vehicle_id, lane_pos)
            delta = lane_pos - prev_lane_pos

            # Handle edge transitions (lane_pos resets when entering new edge)
            if delta < -50:
                # Likely edge transition — estimate progress from speed
                delta = speed * 0.2  # step_length = 0.2s

            progress_reward += (delta / self.max_distance) * 10.0 * self._bonus_scale

            # P0.21: removed the redundant speed_ratio bonus. At constant speed,
            # delta = speed * dt, so the delta-based progress already rewards
            # forward motion. Adding a separate `speed_ratio * 1.0` term
            # double-counts and dominates the safety/comfort terms.

            self.prev_lane_positions[vehicle_id] = lane_pos

            # 2. Delay penalty — context-aware
            if speed < 1.0:
                # Am I stopped for a good reason?
                justified = self._is_delay_justified(vehicle_id)
                if not justified:
                    # Stopped with clear road — penalize
                    progress_reward += -2.0 * self._penalty_scale
                # Justified stop (red light, queue) → 0.0, no penalty

            # 3. Intersection navigation — dense per-step gradient
            if self._is_approaching_intersection(vehicle_id):
                progress_reward += 0.2 * self._bonus_scale
            elif self._is_in_intersection(vehicle_id):
                progress_reward += 0.5 * self._bonus_scale
                self.vehicles_in_intersection.add(vehicle_id)
            elif self._just_exited_intersection(vehicle_id):
                progress_reward += 1.0 * self._bonus_scale

        except Exception as e:
            logger.debug(f"Progress reward error for {vehicle_id}: {e}")

        return progress_reward

    def _is_delay_justified(self, vehicle_id: str) -> bool:
        """Check if stopped/slow speed is justified (not the agent's fault)."""
        try:
            lane_id = self._cached(vehicle_id, 'lane_id')

            # In junction — may need to wait for crossing traffic
            if lane_id and lane_id.startswith(':'):
                return True

            # Approaching intersection (within 50m) — waiting for gap is justified
            if self._is_approaching_intersection(vehicle_id):
                return True

            # Red/yellow light ahead
            try:
                tl_ids = traci.trafficlight.getIDList()
                for tl_id in tl_ids:
                    state = traci.trafficlight.getRedYellowGreenState(tl_id)
                    if lane_id:
                        controlled = traci.trafficlight.getControlledLanes(tl_id)
                        if lane_id in controlled:
                            idx = controlled.index(lane_id)
                            if idx < len(state) and state[idx] in ('r', 'y'):
                                return True
            except:
                pass

            # Leader vehicle ahead within 30m — BUT only if leader is actually
            # moving. If leader is also stopped, both could move → not justified.
            # This prevents the "everyone stops because everyone else stopped"
            # deadlock where no vehicle is penalized for stopping.
            leader_info = traci.vehicle.getLeader(vehicle_id)
            if leader_info is not None:
                leader_id, distance = leader_info
                if distance < 30.0:
                    # Check if leader is moving
                    try:
                        leader_speed = self._cached(leader_id, 'speed')
                        if leader_speed is None:
                            leader_speed = traci.vehicle.getSpeed(leader_id)
                        if leader_speed > 1.0:
                            # Leader is moving but we're stopped — justified
                            # (we're in a queue that's progressing)
                            return True
                        # Leader is ALSO stopped — NOT justified
                        # Both should be moving
                    except Exception:
                        return True  # Can't check → assume justified

            return False
        except:
            return False  # Default: NOT justified (penalize if uncertain)

    def _is_approaching_intersection(self, vehicle_id: str) -> bool:
        """Check if vehicle is within 50m of intersection."""
        try:
            lane_id = self._cached(vehicle_id, 'lane_id')
            if not lane_id or lane_id.startswith(':'):
                return False
            if 'to_center' not in lane_id:
                return False
            lane_pos = self._cached(vehicle_id, 'lane_position')
            lane_length = traci.lane.getLength(lane_id)
            return (lane_length - lane_pos) < 50.0
        except:
            return False

    def _is_in_intersection(self, vehicle_id: str) -> bool:
        """Check if vehicle is in the intersection junction."""
        try:
            lane_id = self._cached(vehicle_id, 'lane_id')
            return lane_id and lane_id.startswith(':')
        except:
            return False

    def _just_exited_intersection(self, vehicle_id: str) -> bool:
        """Check if vehicle just exited the intersection."""
        try:
            lane_id = self._cached(vehicle_id, 'lane_id')
            was_in = vehicle_id in self.vehicles_in_intersection
            is_on_exit = lane_id and 'center_to_' in lane_id

            if was_in and is_on_exit:
                self.vehicles_in_intersection.discard(vehicle_id)
                return True
            if not (lane_id and lane_id.startswith(':')):
                self.vehicles_in_intersection.discard(vehicle_id)
            return False
        except:
            return False

    # =========================================================================
    # COMFORT (0.20) — "Would a passenger feel comfortable?"
    # =========================================================================

    def _compute_comfort(self, vehicle_id: str) -> float:
        """ISO 2631 inspired jerk thresholds with context awareness."""
        comfort_reward = 0.0

        try:
            accel = self._cached(vehicle_id, 'acceleration')
            prev_accel = self.prev_accelerations.get(vehicle_id, accel)

            # Jerk = rate of change of acceleration
            dt = 0.2  # SUMO step length
            jerk = abs(accel - prev_accel) / dt

            self.prev_accelerations[vehicle_id] = accel

            # Context: expected deceleration relaxes threshold
            approaching = self._is_approaching_intersection(vehicle_id)
            jerk_threshold_uncomfortable = 3.75 if approaching else 2.5
            jerk_threshold_harsh = 7.5 if approaching else 5.0

            # ISO 2631 inspired thresholds
            if jerk < 1.0:
                comfort_reward += 0.5 * self._bonus_scale  # Smooth
            elif jerk < jerk_threshold_uncomfortable:
                comfort_reward += 0.0  # Normal
            elif jerk < jerk_threshold_harsh:
                comfort_reward += -1.0 * self._penalty_scale  # Uncomfortable
            else:
                comfort_reward += -3.0 * self._penalty_scale  # Harsh

        except Exception as e:
            logger.debug(f"Comfort reward error for {vehicle_id}: {e}")

        return comfort_reward

    # =========================================================================
    # COOPERATION (0.10) — "Am I helping or hurting traffic flow?"
    # =========================================================================

    def _compute_cooperation(self, vehicle_id: str) -> float:
        """Yielding and intersection blocking assessment."""
        coop_reward = 0.0

        try:
            speed = self._cached(vehicle_id, 'speed')
            lane_id = self._cached(vehicle_id, 'lane_id')

            # 1. Blocking intersection — stopped inside junction
            if self._is_in_intersection(vehicle_id) and speed < 0.5:
                coop_reward += -5.0 * self._penalty_scale
            elif self._is_in_intersection(vehicle_id):
                # Moving through intersection — good
                coop_reward += 0.5 * self._bonus_scale

            # 2. Yielding assessment
            if self._is_approaching_intersection(vehicle_id):
                pet = self._estimate_pet(vehicle_id)
                if pet is not None and pet < 3.0:
                    # Conflicting traffic nearby
                    if speed < 2.0:
                        # Yielding — correct behavior
                        coop_reward += 2.0 * self._bonus_scale
                    else:
                        # Not yielding when should — dangerous
                        coop_reward += -3.0 * self._penalty_scale

        except Exception as e:
            logger.debug(f"Cooperation reward error for {vehicle_id}: {e}")

        return coop_reward

    # =========================================================================
    # UTILITY
    # =========================================================================

    def reset(self):
        """Reset state tracking between episodes."""
        self.prev_accelerations.clear()
        self.prev_lane_positions.clear()
        self.vehicles_in_intersection.clear()
        self._prev_route_frac.clear()
        self._stuck_counter.clear()
        self._prev_speed.clear()
        self._exempt_counter.clear()
        # WAVE-4 comfort term (F27 hazard class): flow ids restart from .0 after a
        # hard SUMO restart, so a stale prev-speed here would charge a phantom
        # harsh-decel penalty on the reused id's spawn step (verified bug 2026-07-31).
        if hasattr(self, '_comfort_prev'):
            self._comfort_prev.clear()

    def get_safety_metrics(self) -> Dict:
        """Return safety metrics for logging."""
        return {
            'collision_count': 0,
            'near_miss_count': 0,
            'emergency_brakes': 0,
            'total_interventions': 0,
            'vehicles_with_interventions': 0
        }
