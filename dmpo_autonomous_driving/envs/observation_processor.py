"""
Observation Processor for DMPO Intersection Environment

Processes SUMO vehicle data into κ-hop neighbor observations suitable for
Networked-MB-MARL DMPO algorithm.
"""

import numpy as np
import os

# SUMO interface selection - check environment variable
if os.environ.get('DMPO_USE_TRACI', '0') == '1':
    import traci
else:
    try:
        import libsumo as traci
    except ImportError:
        import traci
from typing import Dict, List, Set, Any, Optional
import math


class ObservationProcessor:
    """
    Processes raw SUMO data into structured observations for DMPO.
    
    Creates κ-hop neighbor observations following the paper's approach:
    - Ego vehicle state representation
    - κ-hop neighbor information
    - Infrastructure state (traffic lights, geometry)
    """
    
    def __init__(self, kappa: int = 2, max_neighbors: int = 10, communication_radius: float = 100.0):
        """
        Initialize observation processor.

        Args:
            kappa: κ-hop communication radius
            max_neighbors: Maximum number of neighbors to include
            communication_radius: Physical communication radius in meters
        """
        self.kappa = kappa
        # V7: env vars to override neighbor count + perception radius without
        # touching env.create_intersection_env() signature. Defaults match V2.
        env_max_neighbors = os.environ.get('DMPO_MAX_NEIGHBORS')
        if env_max_neighbors:
            try:
                max_neighbors = int(env_max_neighbors)
                print(f"  obs_processor: DMPO_MAX_NEIGHBORS={max_neighbors}")
            except Exception:
                pass
        env_perception_radius = os.environ.get('DMPO_PERCEPTION_RADIUS')
        if env_perception_radius:
            try:
                communication_radius = float(env_perception_radius)
                print(f"  obs_processor: DMPO_PERCEPTION_RADIUS={communication_radius}")
            except Exception:
                pass
        self.max_neighbors = max_neighbors
        self.communication_radius = communication_radius
        
        # FIX Issue 28: Normalization parameters - MUST match other modules
        # action_translator.py uses max_speed=18.0, reward_computer.py uses max_distance=200.0
        # SUMO config sets maxSpeed=18.0 for vehicles
        # Using consistent values across all modules prevents observation distribution shift
        # PHASE 1 FIX: All values must match SUMO vType for consistency
        # Default 16.67 m/s = 60 km/h (matches the trained-policy obs distribution).
        # Override via DMPO_MAX_SPEED_MPS env var for eval-only OOD experiments.
        # Caveat: changing this from training-time value shifts the obs distribution
        # the policy was trained on and may produce OOD behavior.
        self.max_speed = float(os.environ.get('DMPO_MAX_SPEED_MPS', '16.67'))
        self.max_accel = 2.6    # m/s² - matches SUMO vType accel
        self.max_distance = 200.0  # meters - road length
        # Note: Consistent normalization ensures neural network receives same scale inputs
        # during training and evaluation, preventing policy performance degradation
        
        # Intersection center coordinates (will be detected)
        self.intersection_center = None
        self._detect_intersection_center()

        # Optional per-step vehicle state cache (set by env before each step)
        self._vehicle_cache = None
        # Authoritative AI vehicle ordering set by env each step. Must match
        # the ordering used to build the adjacency matrix; otherwise neighbor
        # indices from the adjacency point to the wrong vehicles.
        self._ai_ordered = None
        # ── NEXT-JUNCTION OBS (DMPO_NEXT_JUNCTION_OBS=1): geometry-based obs relative to each
        # vehicle's NEXT downstream junction, so distance/approach/conflict work on ANY net
        # (multi-intersection grids, real cities) and don't rely on the single hard-coded
        # 'center' junction or 'north_to_center' edge-name parsing. Also fixes the
        # intersection_center=(0,0) bug on the training scenario. Maps built lazily once
        # traci is connected. Gated so existing checkpoints/evals keep the legacy obs.
        self._nj_on = os.environ.get('DMPO_NEXT_JUNCTION_OBS', '0') == '1'
        # Identity-only fix (heading-based approach one-hot). Independently toggleable so its
        # effect can be isolated from the distance/conflict changes. Implied by _nj_on.
        # This fix is IN-DISTRIBUTION: training saw a valid one-hot (edge names worked on the
        # 4-lane); on OSM nets it's dead (all-zero) — restoring it gives back the learned feature.
        self._identity_heading = (os.environ.get('DMPO_IDENTITY_HEADING', '0') == '1') or self._nj_on
        # Independently toggleable distance and conflict fixes (both implied by _nj_on), so their
        # effects can be isolated from each other and from identity.
        self._fix_distance = (os.environ.get('DMPO_FIX_DISTANCE', '0') == '1') or self._nj_on   # ego+neighbor dist -> real next junction
        self._fix_conflict = (os.environ.get('DMPO_FIX_CONFLICT', '0') == '1') or self._nj_on   # ctx_pet: same-junction grouping
        self._nj_built = False
        self._jpos = {}          # junction_id -> (x, y)  (non-internal junctions)
        self._edge_nextjxn = {}  # edge_id -> (jid, jx, jy)  (downstream junction id + position)
        # ── CROSSING-CONFLICT OBS (DMPO_CONFLICT_OBS=1): the salient, orientation-invariant
        # crossing-conflict signal the policy currently lacks. Computes the true CONFLICT POINT
        # (ray-intersection of two headings) → per-vehicle time-to-conflict-point (TTCP) → PET,
        # feeding (a) a crossing-aware collision_warning, (b) conflict-point PET into ctx, and
        # (c) an ego-frame direction rotation of neighbor pos/vel. Gated so legacy obs is untouched.
        self._conflict_obs = os.environ.get('DMPO_CONFLICT_OBS', '0') == '1'
        # Arm A vs Arm B: with explicit priority, ctx[3] carries Time-Advantage (whose-turn);
        # without it (Arm B), ctx[3]=0 and the cooperating agents must LEARN the yield convention.
        self._explicit_priority = os.environ.get('DMPO_EXPLICIT_PRIORITY', '0') == '1'
        # Gap-acceptance shield (DMPO_GAP_SHIELD=1) reuses the same conflict-point
        # geometry but is a PURE ACTION FILTER — it needs _crossing_conflict to be
        # COMPUTABLE even when conflict-obs is off, WITHOUT injecting any obs feature
        # (those stay gated on _conflict_obs only), so a plain policy's observations
        # are byte-identical and the shield still gets its conflict points.
        self._gap_shield = os.environ.get('DMPO_GAP_SHIELD', '0') == '1'
        self._cc_cache = {}      # per-step ego_id -> {urgency, pet, ta, n}
        self._cc_key = None      # invalidation key (id of current vehicle cache) to reset per step
        # CURVE-AWARE conflict geometry (DMPO_CURVE_CONFLICT=1): replace the straight heading-ray
        # conflict point + Euclidean distance with SUMO's ACTUAL curved internal-junction-lane
        # geometry (the fix for TURNING vehicles — a straight ray from the current heading mis-
        # locates a turner's curved path, so who-arrives-first is wrong). Built once per net.
        self._curve_conflict = os.environ.get('DMPO_CURVE_CONFLICT', '0') == '1'
        # BUMPER-TO-BUMPER occupancy window: a priority vehicle keeps physically occupying the
        # conflict zone until its REAR clears, not just until its front passes the conflict point.
        # Releasing the yielder at front-passes-point (dist_n<=0) lets it charge into the priority
        # vehicle's still-present body (the seed-202 collision). CLEAR_M = distance past the point
        # the priority vehicle must travel before it's treated as cleared (~veh length + zone width).
        # Default 0.0 = OFF (exactly the old front-passes-point behavior); gated for A/B.
        self._occ_clear_m = float(os.environ.get('DMPO_GAP_SHIELD_CLEAR_M', '0.0') or 0.0)
        # CONSISTENT two-sided occupancy-window model (DMPO_GAP_SHIELD_OCC_WINDOW=1): instead of a
        # point conflict with front-bumper pet, each vehicle OCCUPIES the conflict zone for a TIME
        # WINDOW [ (dist-Wz)/v , (dist+Wz+L)/v ] (front enters near edge -> rear clears far edge).
        # Two vehicles conflict iff their windows OVERLAP; priority = whoever's window STARTS first;
        # gap = separation BETWEEN windows. Applies bumper length to BOTH vehicles and to the
        # arrival/clear/priority/gap uniformly (fixes the one-sided CLEAR_M that shifted collisions).
        self._dump_ids = set(s for s in os.environ.get('DMPO_CONFLICT_DUMP', '').split(',') if s)
        self._occ_window = os.environ.get('DMPO_GAP_SHIELD_OCC_WINDOW', '0') == '1'
        # FIX #3 (grid detection gap): treat two internal lanes as conflicting if their centrelines
        # pass within this distance (~vehicle width + margin), not only if they exactly intersect.
        # Default 0.0 = OFF = exact-intersection-only (old behavior); gated for A/B.
        self._conflict_dist = float(os.environ.get('DMPO_GAP_SHIELD_CONFLICT_DIST', '0.0') or 0.0)
        # PART 3 (weak point #1): PATH-AWARE conflict/gap observation block. The clean pairwise
        # conflict signal the policy needs to time crossings (it was BLIND to gaps). Reuses the
        # curve-conflict machinery: per κ-hop crosser, the pairwise TTCP + gap; top-K salient block.
        # Gated DMPO_PATH_CONFLICT_OBS=1 (adds PATH_CONFLICT_BLOCK dims at the END of the flat obs,
        # so disabled = byte-identical to legacy). obs_dim recomputed; train/eval MUST agree.
        self._path_conflict_obs = os.environ.get('DMPO_PATH_CONFLICT_OBS', '0') == '1'
        self._occ_zone_w = float(os.environ.get('DMPO_GAP_SHIELD_ZONE_W', '3.2') or 3.2)   # half-width of the crossing zone (~lane width)
        self._occ_veh_len = float(os.environ.get('DMPO_GAP_SHIELD_VEH_LEN', '6.5') or 6.5)  # bumper length + safety buffer (4.5 veh + 2)
        self._jx_built = False
        self._jx_shapes = {}     # internal_lane -> [(x,y),...] curved shape
        self._jx_conn = {}       # (approach_lane, dest_edge) -> internal_lane (the connection a route uses)
        self._jx_conn_e = {}     # (from_edge, dest_edge) -> internal_lane (edge-level fallback: turns are
                                 # lane-specific, so a turner on a non-turn lane misses the exact-lane lookup)
        self._jx_confl = {}      # (internal_a, internal_b) -> (cp, arc_a, arc_b)  arc from each lane's START
        # ── F29/F16 (v2-PRIVATE, additive — legacy/shield never read these): SUMO splits
        # many connections into SEQUENTIAL internal lanes (split at an internal junction);
        # the left-turn-across-traffic crossing lies on the SECOND segment, which _jx_confl
        # keyed by single segments never attributes to an APPROACHING vehicle. These maps
        # chain the segments into FULL per-connection paths (built in _ensure_jx_conflict_map
        # under self._v2 only; consumed ONLY by _conflict_block_v2):
        self._jx_confl_v2 = {}   # (head_a, head_b) -> (cp, arc_a, arc_b) arcs along the FULL chained path
        self._jx_chain = {}      # head internal lane -> [seg0, seg1, ...] successor chain
        self._jx_chain_pos = {}  # any internal segment -> (head, arc offset of its start along the chain)
        self._veh_il_cache = {}  # (current_lane, next_edge) -> internal_lane (static per connection)
        # ── M12 PERF (pure, byte-identical): per-step memoization infrastructure.
        # The env publishes an explicit monotone _cache_epoch alongside _vehicle_cache;
        # _step_memo() invalidates ALL per-step memos on that token (never on id()).
        # DMPO_PERF_FIXES=0 is the master kill-switch restoring every legacy code path.
        self._cache_epoch = None       # set by env next to _vehicle_cache each step (M12-0)
        self._memo_epoch = None        # epoch the current memo container belongs to
        self._memo_cache_ref = None    # strong ref to the cache the memo was built from
        self._memo = {}
        self._perf = os.environ.get('DMPO_PERF_FIXES', '1') == '1'
        self._spatial_hash_on = os.environ.get('DMPO_OBS_SPATIAL_HASH', '1') == '1'  # M12-3 gate
        self._lane_enc_cache = {}      # E1: lane_id -> encoded value (static per net)
        self._edge_len_cache = {}      # E4: edge_id -> length (static per net)
        self._infra_cache = {}         # E5: lane_id -> (raw_max_speed, lane_count) (static per net)

        # ══ OBS-V2 (DMPO_OBS_V2=1 — Stage-3 checkpoint-breaking batch) ══
        # ONE umbrella flag. Under v2 the 14 legacy obs experiment flags are IGNORED
        # (anti-F6b): the semantics below are forced, the env vars are never consulted,
        # and scripts/assert_consistency.py refuses a launch that sets any of them to a
        # non-canonical value. obs_dim = 61 + 12*max_neighbors (241 at N=15).
        # VERIFIER CORRECTION (adversarial verify, 2026-07-25): the shield-shared attrs
        # _occ_window/_occ_clear_m/_conflict_dist/_occ_zone_w/_occ_veh_len are NOT
        # touched here — the gap shield's runtime semantics stay governed solely by the
        # DMPO_GAP_SHIELD_* block vars. The v2 conflict block computes its own two-sided
        # occupancy windows with v2-PRIVATE code (_conflict_block_v2), so flipping the
        # obs flag can never reconfigure the shield.
        # MATCHED MODEL LOSS (train-only, gated DMPO_MODEL_MATCHED_LOSS): export the
        # ordered neighbour vehicle-ids (slot k → id) beside each v2 obs dict so the
        # world-model loss can match slots by identity. v1 path never sets it; the
        # id tuple travels as an EXTRA dict key (never reaches flatten()/obs_dim).
        self._export_nb_ids = False
        self._nb_ids_scratch = None
        self._v2 = os.environ.get('DMPO_OBS_V2', '0') == '1'
        if self._v2:
            # Forced obs semantics (the anti-F6b hard-set; obs-only attrs ONLY):
            self._nj_on = True                 # always next-junction geometry
            self._identity_heading = True      # identity one-hot always heading-derived
            self._fix_distance = True
            self._fix_conflict = True
            self._curve_conflict = True        # curve machinery always on (§5.5)
            self._conflict_obs = False         # v2 has its own blocks; legacy ctx/rot paths off
            self._explicit_priority = False    # replaced by ctx arrival_rank (advice, not rule)
            self._path_conflict_obs = False    # v1 14-dim block replaced by _conflict_block_v2
            # ARM-B (user design 2026-07-26): endogenous driving-profile
            # distinguishability. Each vehicle's lived history (age, mean speed
            # ratio, mean |accel|, braking fraction) is appended to the ego block
            # and, as EGO-MINUS-OTHER deltas + the raw un-quantized arrival delta,
            # to each conflict slot. Pure information — no priority rule, no
            # convention encoded; the network is merely made able to DISTINGUISH
            # the two sides of a tie (breaks the shared-policy symmetry premise)
            # and to SEE the epsilon the ±0.2s whose-turn quantization erases.
            # Gated: OFF (default) leaves every dim and every value byte-identical.
            self._profile_obs = os.environ.get('DMPO_PROFILE_OBS', '0') == '1'
            # v2.1 conflict-visibility fix (F2 arm): see _conflict_block_v2's
            # same-edge branch. Default OFF = byte-identical legacy behavior.
            self._confl_same_edge = os.environ.get('DMPO_CONFL_SAME_EDGE', '0') == '1'
            # SENSING ROBUSTNESS (eval-only, gated; default off = byte-identical). Applied to
            # OTHER vehicles in the conflict block and arrival rank only (each car's own path
            # and position stay exact):
            #   DMPO_SENSE_INTENT_UNKNOWN=1  — a crosser still approaching the junction is
            #     assumed to go straight (its route is not communicated); a crosser already
            #     inside the junction keeps its visible path.
            #   DMPO_SENSE_LANEPOS_SIGMA=σ   — along-lane localization error of other vehicles:
            #     their distance to the junction += N(0, σ²) metres, one draw per vehicle per
            #     step (the same report for every receiver).
            self._intent_unknown = os.environ.get('DMPO_SENSE_INTENT_UNKNOWN', '0') == '1'
            self._lanepos_sigma = float(os.environ.get('DMPO_SENSE_LANEPOS_SIGMA', '0') or 0)
            self._straight_il_cache = {}
            self._lp_rng = None
            if self._profile_obs:
                # instance attrs shadow the class constants at every self.V2_* site
                # (zero-obs, block builders, all np.zeros sizing).
                self.V2_EGO_DIM = ObservationProcessor.V2_EGO_DIM + 4
                self.V2_CONFLICT_BLOCK = (ObservationProcessor.V2_CONFLICT_BLOCK
                                          + 4 * ObservationProcessor.V2_CONFLICT_K)
            self._veh_profiles = {}            # env publishes {vid: [n, sum_v, sum_|a|, n_brake]}
            self._export_nb_ids = os.environ.get('DMPO_MODEL_MATCHED_LOSS', '0') == '1'
            # v2 normalization constants — every slot bounded AT CONSTRUCTION (C3: no
            # z-filter, no ±5 clip; see trainer). Distances scale by the perception
            # radius; conflict scanning keeps a separate physical horizon.
            if not os.environ.get('DMPO_MAX_DECEL'):
                raise RuntimeError(
                    "DMPO_OBS_V2=1 requires DMPO_MAX_DECEL to be set explicitly: "
                    "v2 obs accel dims are normalized by it (asymmetric a/2.6 vs a/|MAX_DECEL|), "
                    "so an implicit default would silently mis-scale observations (H13/R9).")
            self.obs_dist_norm = max(float(self.communication_radius), 1.0)
            self.conflict_horizon_m = float(os.environ.get('DMPO_CONFLICT_HORIZON_M', '200.0'))
            self.acc_pos = 2.6                                        # SUMO vType accel
            self.acc_neg = abs(float(os.environ.get('DMPO_MAX_DECEL')))
            self.step_length = float(os.environ.get('DMPO_STEP_LENGTH', '0.2'))
            self.ttc_norm = 8.0        # s (W6 rescale; legacy nb_ttc divided by 30)
            self.ttcp_norm = 10.0      # s (conflict-block time normalizer)
            self.pet_gate_s = 1.5      # the VALIDATED genuine-conflict gate
            self.v_floor = 3.0         # m/s symmetric arrival-time floor (M8 fix)
            # v2 conflict-map pair-distance: SAME source as the shield (block-pinned var,
            # default 2.5 m) so obs and shield see one conflict map — never a private fork.
            if self._conflict_dist <= 0.0:
                self._conflict_dist = float(os.environ.get(
                    'DMPO_GAP_SHIELD_CONFLICT_DIST', '2.5') or 2.5)
            # Per-step state published by the env (Step-6 wiring):
            self._prev_actions = {}        # vid -> last action vector (prev_long dim)
            self._shield_engaged = set()   # vids the scaffold held on the PREVIOUS step
            self._zero_obs_v2 = None       # lazily built neutral (far/clear) zero-obs

    def _step_memo(self):
        """The ONE per-step memo container (M12-0), invalidated when the env swaps in a
        new vehicle cache: the env bumps _cache_epoch exactly once per fresh cache build
        (and once per episode reset). There is deliberately NO id()-based fallback —
        id() values get recycled and would poison memos. When no epoch is published
        (external harnesses/tests that set _vehicle_cache directly), the container is
        rebuilt on every call: memoization off, values unchanged. The cache-object
        identity check is an EXTRA invalidation trigger only — safe because we hold a
        strong reference (the old dict cannot be GC'd, so its identity cannot recycle).
        Epoch lifetime: one epoch spans the obs phase of step t and the action phase of
        step t+1 with no simulationStep() in between — the exact window the legacy
        _cc_cache id-trick already spanned."""
        ep = self._cache_epoch
        cache = self._vehicle_cache
        if ep is None or ep != self._memo_epoch or cache is not self._memo_cache_ref:
            self._memo_epoch = ep
            self._memo_cache_ref = cache
            self._memo = {'vi': {}, 'jb': None, 'sh': None, 'leader': {},
                          'route': {}, 'ri': {}, 'ctx': None, 'lanepos': None,
                          'tls': None, 'fcfs': None, 'cc': {}}
            # M12-5A: prune the (previously unbounded) speed-state dict once per epoch.
            # Only drops keys of vehicles absent from the live cache; a reappearing id
            # is a NEW vehicle whose counter should restart anyway.
            sst = getattr(self, '_speed_state_tracking', None)
            if sst is not None and len(sst) > 4 * len(cache or {}) + 64:
                live = cache or {}
                self._speed_state_tracking = {k: v for k, v in sst.items()
                                              if k[:-12] in live}   # strip '_speed_state'
        return self._memo

    def reset_episode_state(self):
        """Clear per-episode obs state at an episode boundary (M12-5B). Called by the
        env's reset() path ONLY — never from step() — and ONLY under
        DMPO_OBS_STATE_RESET=1 (checkpoint-affecting: context[9] speed-state counters
        restart for reused flow-derived vehicle ids)."""
        self._speed_state_tracking = {}
        self._memo = {}
        self._memo_epoch = None
        self._memo_cache_ref = None
        self._cc_cache = {}
        self._cc_key = None

    def on_vehicle_removed(self, vid):
        """Env hook (M12-1 exactness): a vehicle was force-removed mid-step
        (collision/stuck-timeout/clear-stopped) AFTER the obs phase warmed the memos.
        Scrub it so the shield's action-phase queries match legacy live-TraCI behavior:
        - approach vehicle: getRoute would raise -> _veh_internal = (None, None)
        - in-junction vehicle: getLanePosition would raise -> (cached_lane, 0.0)
        The cc memo is NOT touched: legacy kept obs-phase cc entries across the same
        window (same id-keyed cache)."""
        memo = self._memo
        if not memo:
            return
        lane = ''
        if self._vehicle_cache and vid in self._vehicle_cache:
            lane = self._vehicle_cache[vid].get('lane_id') or ''
        in_junction = lane.startswith(':')
        repl = (lane, 0.0) if in_junction else (None, None)
        vi = memo.get('vi')
        if vi is not None and vid in vi:
            vi[vid] = repl
        jb = memo.get('jb')
        if jb:
            for jid in list(jb.keys()):
                lst = jb[jid]
                if any(e[0] == vid for e in lst):
                    if in_junction:
                        jb[jid] = [(e[0], e[1], 0.0, e[3], e[4]) if e[0] == vid else e
                                   for e in lst]
                    else:
                        jb[jid] = [e for e in lst if e[0] != vid]

    def _ensure_nj_maps(self):
        """Build edge -> downstream-junction maps from the live network (once)."""
        if self._nj_built:
            return
        try:
            self._jpos = {j: traci.junction.getPosition(j)
                          for j in traci.junction.getIDList() if not j.startswith(':')}
            jids = list(self._jpos.keys())
            jarr = np.array([self._jpos[j] for j in jids], dtype=np.float64)  # (J, 2)
            for e in traci.edge.getIDList():
                if e.startswith(':'):
                    continue
                try:
                    end = traci.lane.getShape(e + '_0')[-1]   # where the approach meets the junction
                except Exception:
                    continue
                # nearest real junction to the approach end = the downstream junction (vectorized)
                k = int(np.argmin(((jarr[:, 0] - end[0]) ** 2) + ((jarr[:, 1] - end[1]) ** 2)))
                self._edge_nextjxn[e] = (jids[k], jarr[k, 0], jarr[k, 1])
            self._nj_built = True
            if os.environ.get('DMPO_DEBUG_CENTER'):
                _se = next(iter(self._edge_nextjxn.items()), None)
                print(f"🧭 NEXT-JXN maps: {len(self._jpos)} junctions, {len(self._edge_nextjxn)} edges. "
                      f"sample edge->jxn: {_se}", flush=True)
        except Exception as _e:
            self._nj_built = True   # don't retry every step if the net query fails
            # H15 FIX: latch stays (no per-step retry storm) but the failure is LOUD
            # and recorded — a silent failure here blinds every next-junction feature.
            self._nj_build_failed = True
            print(f"❌ ERROR (H15): NEXT-JXN map build FAILED — next-junction obs features "
                  f"will be DEGRADED for this run: {_e}", flush=True)

    def _next_junction_full(self, vehicle_id):
        """(jid, jx, jy) of the vehicle's next downstream junction, or None. Works on any net."""
        self._ensure_nj_maps()
        edge = self._cached(vehicle_id, 'edge_id') or ''
        if edge and not edge.startswith(':'):
            return self._edge_nextjxn.get(edge)
        if edge.startswith(':'):
            # inside a junction: internal edges are ':<junctionID>_<idx>' → O(1) lookup, no scan
            jid = edge[1:].rsplit('_', 1)[0]
            jp = self._jpos.get(jid)
            if jp is not None:
                return (jid, jp[0], jp[1])
        # unknown edge / parse miss: nearest junction to the current position (rare)
        try:
            px, py = self._cached(vehicle_id, 'position') or traci.vehicle.getPosition(vehicle_id)
        except Exception:
            return None
        best, bd = None, 1e18
        for jid, (jx, jy) in self._jpos.items():
            d = (jx - px) ** 2 + (jy - py) ** 2
            if d < bd:
                bd, best = d, (jid, jx, jy)
        return best

    def _next_junction(self, vehicle_id):
        """(jx, jy) of the vehicle's next downstream junction, or None."""
        f = self._next_junction_full(vehicle_id)
        return (f[1], f[2]) if f else None

    def _crossing_conflict(self, ego_id):
        """Salient crossing-conflict signal via ray-intersection CONFLICT POINT.

        For each AI neighbor heading to the SAME next junction from a DIFFERENT approach,
        intersect the two straight heading rays to get the conflict point and the along-
        heading distances t (ego) and s (neighbor) to it. TTCP = dist/speed; PET =
        |TTCP_ego - TTCP_nbr|. A conflict is DANGEROUS when both reach the point soon
        (small TTCP) AND close in time (small PET). Returns the most-imminent one:
          urgency  0/0.1/0.4/0.7/1.0  (same scale as the rear-end warning)
          pet      seconds (10=none)
          ta       TTCP_ego - TTCP_nbr  (<0 ego first=GO, >0 ego second=YIELD)
        Cached per step (keyed on the swapped-in vehicle-cache identity).
        """
        if self._perf:
            cc = self._step_memo()['cc']        # M12-0: ONE explicit invalidation point
        else:
            key = id(self._vehicle_cache)       # legacy id-trick (kill-switch path)
            if self._cc_key != key:
                self._cc_cache = {}
                self._cc_key = key
            cc = self._cc_cache
        if ego_id in cc:
            return cc[ego_id]
        res = {'urgency': 0.0, 'pet': 10.0, 'ta': 0.0, 'n': 0, 'cp': None}
        # OBS-V2: the conflict geometry is always live (the reward computer's PET
        # idle-yield exemption and any scaffold read it); legacy gate unchanged.
        if not ((self._conflict_obs or self._gap_shield or self._v2) and self._vehicle_cache):
            cc[ego_id] = res
            return res
        if self._curve_conflict:                        # CURVE-AWARE path (SUMO internal-lane geometry)
            res = self._crossing_conflict_curve(ego_id)
            cc[ego_id] = res
            return res
        ejf = self._next_junction_full(ego_id)
        E = self._cached(ego_id, 'position'); ea = self._cached(ego_id, 'angle')
        ev = self._cached(ego_id, 'speed') or 0.0
        ego_edge = self._cached(ego_id, 'edge_id') or ''
        if ejf is None or E is None or ea is None:
            cc[ego_id] = res
            return res
        ego_jid = ejf[0]
        ear = math.radians(ea)
        dex, dey = math.sin(ear), math.cos(ear)          # SUMO angle: 0=N,90=E -> vx=sin,vy=cos
        best_ttcp = float('inf'); best_nid = None; best_t = best_s = 0.0
        for nid, st in self._vehicle_cache.items():
            if nid == ego_id:
                continue
            vt = st.get('type_id', '') or ''
            if not (vt.startswith('ai_vehicle') and not vt.startswith('ai_human_')):
                continue
            n_edge = st.get('edge_id', '') or ''
            if n_edge == ego_edge:
                continue                                  # same approach = following, not crossing
            njf = self._next_junction_full(nid)
            if njf is None or njf[0] != ego_jid:
                continue                                  # not the same junction
            N = st.get('position'); na = st.get('angle'); nv = st.get('speed') or 0.0
            if N is None or na is None:
                continue
            nar = math.radians(na)
            dnx, dny = math.sin(nar), math.cos(nar)
            det = dex * (-dny) - (-dnx) * dey
            if abs(det) < 1e-6:
                continue                                  # parallel/anti-parallel -> no crossing pt
            rx, ry = N[0] - E[0], N[1] - E[1]
            t = (rx * (-dny) - (-dnx) * ry) / det         # dist along ego heading to conflict pt
            s = (dex * ry - dey * rx) / det               # dist along nbr heading to conflict pt
            if t <= 0.0 or s <= 0.0:
                continue                                  # conflict point behind one of them
            if t > self.max_distance or s > self.max_distance:
                continue                                  # too far to matter
            ttcp_e = t / max(ev, 0.5)
            ttcp_n = s / max(nv, 0.5)
            if ttcp_e < best_ttcp:
                best_ttcp = ttcp_e
                best_nid = nid
                best_t, best_s = t, s                      # ray-distances to the conflict point
                res['pet'] = min(abs(ttcp_e - ttcp_n), 10.0)
                res['ta'] = ttcp_e - ttcp_n
                res['cp'] = (E[0] + t * dex, E[1] + t * dey)
                res['nid'] = nid
            res['n'] += 1
        # FAIR symmetry-break: when arrivals ~tie (|TA| small) the raw signal can't say who
        # yields (the nt13 deadlock). Break it by DISTANCE to the conflict point — the
        # physically closer vehicle commits first. This is POSITION-based, so it rotates as
        # vehicles move and NO approach is permanently starved (the earlier lexicographic-id
        # tie-break starved 'ai_west' forever). Both agents see the same geometry -> same winner.
        if best_nid is not None and abs(res['ta']) < 0.5:
            res['ta'] = -0.5 if best_t <= best_s else 0.5
        if res['n'] > 0 and res['pet'] < 2.0:             # real conflict only if arrivals overlap
            if best_ttcp < 1.0:   res['urgency'] = 1.0
            elif best_ttcp < 2.0: res['urgency'] = 0.7
            elif best_ttcp < 3.0: res['urgency'] = 0.4
            elif best_ttcp < 5.0: res['urgency'] = 0.1
        cc[ego_id] = res
        return res

    # ── CURVE-AWARE conflict geometry (DMPO_CURVE_CONFLICT=1) ────────────────────────────
    @staticmethod
    def _seg_int(p1, p2, p3, p4):
        """Intersection point of segments p1p2 and p3p4, or None."""
        x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-9:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        return None

    def _first_poly_int(self, sa, sb):
        """First intersection point of two polylines sa, sb (or None)."""
        for i in range(len(sa) - 1):
            for j in range(len(sb) - 1):
                ip = self._seg_int(sa[i], sa[i + 1], sb[j], sb[j + 1])
                if ip is not None:
                    return ip
        return None

    @staticmethod
    def _seg_seg_min(p1, p2, p3, p4):
        """Min distance between segments p1p2 and p3p4, and the midpoint of the closest approach."""
        def clamp(v): return 0.0 if v < 0 else (1.0 if v > 1 else v)
        ax, ay = p1; bx, by = p2; cx, cy = p3; dx, dy = p4
        ux, uy = bx - ax, by - ay; vx, vy = dx - cx, dy - cy
        wx, wy = ax - cx, ay - cy
        a = ux*ux + uy*uy; b = ux*vx + uy*vy; c = vx*vx + vy*vy
        d = ux*wx + uy*wy; e = vx*wx + vy*wy
        den = a*c - b*b
        s = clamp((b*e - c*d) / den) if den > 1e-9 else 0.0
        t = (b*s + e) / c if c > 1e-9 else 0.0
        t = clamp(t); s = clamp((b*t - d) / a) if a > 1e-9 else 0.0
        p = (ax + s*ux, ay + s*uy); q = (cx + t*vx, cy + t*vy)
        return math.dist(p, q), ((p[0]+q[0])*0.5, (p[1]+q[1])*0.5)

    def _poly_close(self, sa, sb, thresh):
        """Closest-approach point of two polylines if they pass within `thresh` (else None).
        Catches crossings whose CENTRELINES don't intersect but whose bodies (vehicle width)
        overlap — common for merges / adjacent turns, which _first_poly_int misses."""
        best = thresh; cp = None
        for i in range(len(sa) - 1):
            for j in range(len(sb) - 1):
                dmin, mid = self._seg_seg_min(sa[i], sa[i + 1], sb[j], sb[j + 1])
                if dmin < best:
                    best = dmin; cp = mid
        return cp

    @staticmethod
    def _arc_to(shape, pt):
        """Arc length along polyline `shape` from its start to the point `pt` on it."""
        acc = 0.0
        for i in range(len(shape) - 1):
            a, b = shape[i], shape[i + 1]
            dab = math.dist(a, b)
            if dab > 1e-9 and abs(math.dist(a, pt) + math.dist(pt, b) - dab) < 0.5:
                return acc + math.dist(a, pt)
            acc += dab
        return acc

    @staticmethod
    def _il_jid(il):
        """Junction id of an internal lane (M12-2). MUST be the single shared parse for
        BOTH the conflict-map build gate and the bucket index — even a 'wrong' parse
        (junction id containing '_') stays consistent, hence the partition stays exact."""
        return il[1:].rsplit('_', 2)[0]      # ':center_5_0' -> 'center' ; ':nt13_5_0' -> 'nt13'

    def _ensure_jx_conflict_map(self):
        """Precompute SUMO's curved internal-junction-lane geometry ONCE: shapes, the
        (approach_lane, dest_edge)->internal_lane connection map, and pairwise curve-
        intersection conflict points with ARC-LENGTH along each internal lane. Any net."""
        if self._jx_built:
            return
        try:
            internal = [l for l in traci.lane.getIDList() if l.startswith(':')]
            self._jx_shapes = {l: traci.lane.getShape(l) for l in internal}
            for lane in traci.lane.getIDList():
                if lane.startswith(':'):
                    continue
                try:
                    from_edge = traci.lane.getEdgeID(lane)
                    for link in traci.lane.getLinks(lane):
                        dest, via = link[0], link[4]
                        if via:
                            dest_edge = traci.lane.getEdgeID(dest)
                            self._jx_conn[(lane, dest_edge)] = via
                            self._jx_conn_e.setdefault((from_edge, dest_edge), via)   # edge-level fallback
                except Exception:
                    pass

            _jid = self._il_jid                      # M12-2: ONE shared parse (see _il_jid)
            ils = list(self._jx_shapes.items())
            for i, (la, sa) in enumerate(ils):
                for lb, sb in ils[i + 1:]:
                    if _jid(la) != _jid(lb):
                        continue
                    cp = self._first_poly_int(sa, sb)
                    if cp is None and self._conflict_dist > 0.0:
                        # FIX #3: register close-passing crossings (bodies overlap though centrelines
                        # don't intersect) — the grid detection gap. Threshold ~vehicle width + margin.
                        cp = self._poly_close(sa, sb, self._conflict_dist)
                    if cp is not None:
                        # Two long vehicles' BODIES overlap ~a car-length BEFORE their centres reach the
                        # centreline crossing point (critical for two converging TURNS). Subtract a
                        # vehicle-length body buffer from the arc so the shield treats the conflict as
                        # starting where the bodies first meet, not where the centrelines cross.
                        buf = float(os.environ.get('DMPO_GAP_SHIELD_BODY_BUF', '0.0'))
                        aa = max(self._arc_to(sa, cp) - buf, 0.0)
                        ab = max(self._arc_to(sb, cp) - buf, 0.0)
                        self._jx_confl[(la, lb)] = (cp, aa, ab)
                        self._jx_confl[(lb, la)] = (cp, ab, aa)

            # ── F29 (BLOCKER) + F16: v2-PRIVATE successor-chained conflict map ─────
            # A split connection's FIRST internal lane ends at the internal-junction
            # wait point; the crossing with the oncoming stream lies on the SECOND
            # segment (getLinks(first)[4] = successor, verified empirically on
            # 4lane_allAI_train: 8 split left/U-turns). The legacy per-segment map
            # above (_jx_confl — SHIELD territory, byte-identical) can therefore
            # never show an approaching left-turner its across-traffic conflict.
            # Build a v2-only map over FULL concatenated paths, arcs measured along
            # the chained polyline from the FIRST segment's start. Deterministic
            # (sorted; hash-independent). One-time static build (DMPO_PERF pattern).
            if self._v2:
                def _plen(sh):
                    return sum(math.dist(sh[k], sh[k + 1]) for k in range(len(sh) - 1))
                succ = {}
                for il in self._jx_shapes:
                    try:
                        for link in traci.lane.getLinks(il):
                            via = link[4] if len(link) > 4 else ''
                            if via and via in self._jx_shapes:
                                succ[il] = via
                                break
                    except Exception:
                        pass
                _tails = set(succ.values())
                heads = sorted(l for l in self._jx_shapes if l not in _tails)
                chain_shape = {}
                for h in heads:
                    chain, seen, cur = [], set(), h
                    while cur is not None and cur in self._jx_shapes and cur not in seen:
                        seen.add(cur)
                        chain.append(cur)
                        cur = succ.get(cur)
                    self._jx_chain[h] = chain
                    shape = []
                    for seg in chain:
                        s = self._jx_shapes[seg]
                        if not shape:
                            off = 0.0
                            shape = list(s)
                        else:
                            gap = math.dist(shape[-1], s[0])
                            # arc of this segment's start along the CONCATENATED
                            # polyline (a tiny geometry gap becomes a bridge segment,
                            # so _arc_to over the chained shape stays consistent)
                            off = _plen(shape) + (gap if gap >= 1e-6 else 0.0)
                            shape.extend(list(s[1:]) if gap < 1e-6 else list(s))
                        # shared tails keep the first (sorted) head — mapping a
                        # vehicle PAST a merge is ambiguous but past the diverge
                        self._jx_chain_pos.setdefault(seg, (h, off))
                    chain_shape[h] = shape
                by_jid = {}
                for h in heads:
                    by_jid.setdefault(_jid(h), []).append(h)
                for hs in by_jid.values():
                    for i, ha in enumerate(hs):
                        sa = chain_shape[ha]
                        for hb in hs[i + 1:]:
                            sb = chain_shape[hb]
                            cp = self._first_poly_int(sa, sb)
                            if cp is None and self._conflict_dist > 0.0:
                                cp = self._poly_close(sa, sb, self._conflict_dist)
                            if cp is not None:
                                # NO body-buffer here (unlike the shield map): the v2
                                # window math applies Wz/vehicle lengths explicitly,
                                # so a buffer would double-count bodies.
                                aa = self._arc_to(sa, cp)
                                ab = self._arc_to(sb, cp)
                                self._jx_confl_v2[(ha, hb)] = (cp, aa, ab)
                                self._jx_confl_v2[(hb, ha)] = (cp, ab, aa)
            self._jx_built = True
        except Exception as _e:
            self._jx_built = True     # don't retry every step if the net query fails
            # H15 FIX: loud + recorded — a silent failure here means an EMPTY conflict
            # map: the shield/rule-controller silently never fires and conflict obs are zero.
            self._jx_build_failed = True
            print(f"❌ ERROR (H15): CURVE-CONFLICT map build FAILED — conflict map EMPTY; "
                  f"shield/conflict features are DEAD for this run: {_e}", flush=True)

    # ── M12 per-step / static memo helpers (all no-ops value-wise; DMPO_PERF_FIXES=0
    # routes every call straight to TraCI exactly as before) ─────────────────────────
    def _veh_route(self, vid):
        """getRoute memoized per epoch (E3). Raises through UNCACHED on TraCI error so
        every call site's own try/except sees the identical exception."""
        if not self._perf:
            return traci.vehicle.getRoute(vid)
        memo = self._step_memo()['route']
        r = memo.get(vid)
        if r is None:
            r = traci.vehicle.getRoute(vid)
            memo[vid] = r
        return r

    def _veh_route_index(self, vid):
        """getRouteIndex memoized per epoch (E3). Raises through uncached."""
        if not self._perf:
            return traci.vehicle.getRouteIndex(vid)
        memo = self._step_memo()['ri']
        r = memo.get(vid)
        if r is None:
            r = traci.vehicle.getRouteIndex(vid)
            memo[vid] = r
        return r

    def _veh_leader(self, vid):
        """getLeader (no-dist form) memoized per epoch (E2: obs computes it twice per
        ego — collision_warning and context[6]). None is a valid cached value."""
        if not self._perf:
            return traci.vehicle.getLeader(vid)
        memo = self._step_memo()['leader']
        if vid in memo:
            return memo[vid]
        res = traci.vehicle.getLeader(vid)
        memo[vid] = res
        return res

    def _edge_length(self, edge):
        """edge.getLength memoized permanently (E4) — edge lengths are static per net.
        Raises through uncached."""
        if not self._perf:
            return traci.edge.getLength(edge)
        v = self._edge_len_cache.get(edge)
        if v is None:
            v = traci.edge.getLength(edge)
            self._edge_len_cache[edge] = v
        return v

    def _veh_internal(self, vid):
        """(internal_lane, dist_to_conflict_base) — memoized per vehicle-cache epoch
        (M12-1). Valid because no traci.simulationStep() occurs inside an epoch, so
        route, routeIndex, laneID, laneLength and lanePosition are all constant within
        it. `hit is None` is an unambiguous miss test: every return path of the
        uncached body returns a 2-tuple, never a bare None."""
        if not self._perf:
            return self._veh_internal_uncached(vid)
        memo = self._step_memo()['vi']
        hit = memo.get(vid)
        if hit is None:
            hit = self._veh_internal_uncached(vid)
            memo[vid] = hit
        return hit

    def _veh_internal_uncached(self, vid):
        """(internal_lane, dist_to_conflict_base) for vehicle vid, where dist_to_conflict_base is
        distance from vid to the START of its internal lane: +remaining-approach if still on the
        approach, -(dist already travelled) if already inside the junction. None if no connection."""
        lane = self._cached(vid, 'lane_id') or ''
        if lane.startswith(':'):
            try:
                return lane, -traci.vehicle.getLanePosition(vid)
            except Exception:
                return lane, 0.0
        try:
            route = self._veh_route(vid); ri = self._veh_route_index(vid)
        except Exception:
            return None, None
        if ri < 0 or ri + 1 >= len(route):
            return None, None
        next_edge = route[ri + 1]
        key = (lane, next_edge)
        il = self._veh_il_cache.get(key, 0)
        if il == 0:
            # exact (lane, dest) first; fall back to edge-level (turn conn may be on another lane)
            il = self._jx_conn.get(key)
            if il is None:
                try:
                    il = self._jx_conn_e.get((traci.lane.getEdgeID(lane), next_edge))
                except Exception:
                    il = None
            self._veh_il_cache[key] = il
        if il is None:
            return None, None
        try:
            return il, (traci.lane.getLength(lane) - traci.vehicle.getLanePosition(vid))
        except Exception:
            return None, None

    def _junction_buckets(self):
        """{junction_id: [(vid, il, base, speed, is_ai), ...]} — ONE O(N) pass per epoch,
        shared by _crossing_conflict_curve and _conflict_gap_block (M12-2). Exact by the
        _jx_confl same-junction invariant: the conflict map is only ever populated for
        internal-lane pairs of the SAME junction (obs _ensure_jx_conflict_map continues
        on _il_jid mismatch), so restricting each ego's scan to its own junction's bucket
        drops only iterations that provably hit `info is None -> continue`. Built in
        _vehicle_cache iteration order so downstream first-wins tie-breaks (strict < /
        stable sort) are byte-identical to the old full-cache scans."""
        memo = self._step_memo()
        if memo['jb'] is not None:
            return memo['jb']
        self._ensure_jx_conflict_map()
        buckets = {}
        for vid, st in (self._vehicle_cache or {}).items():
            il, base = self._veh_internal(vid)            # memoized by M12-1
            if il is None or base is None:
                continue
            vt = st.get('type_id', '') or ''
            is_ai = vt.startswith('ai_vehicle') and not vt.startswith('ai_human_')
            buckets.setdefault(self._il_jid(il), []).append(
                (vid, il, base, st.get('speed') or 0.0, is_ai))
        memo['jb'] = buckets
        return buckets

    def _straight_il(self, vid):
        """Internal lane of the straight-through connection from vid's current approach lane
        (None when vid is already inside a junction or no connection exists). Static per lane."""
        lane = self._cached(vid, 'lane_id') or ''
        if not lane or lane.startswith(':'):
            return None
        if lane in self._straight_il_cache:
            return self._straight_il_cache[lane]
        il = None
        try:
            links = traci.lane.getLinks(lane, extended=True)
            for lk in links:
                if str(lk[6]).lower() == 's' and lk[4]:
                    il = lk[4]
                    break
            if il is None and links and links[0][4]:
                il = links[0][4]
        except Exception:
            il = None
        self._straight_il_cache[lane] = il
        return il

    def _crosser_view(self, nid, n_il, n_base):
        """What the ego is assumed to know about another vehicle's path and position under the
        gated sensing-robustness options (identity when both are off)."""
        if self._intent_unknown:
            _sil = self._straight_il(nid)
            if _sil is not None:
                n_il = _sil
        if self._lanepos_sigma > 0 and n_base is not None:
            memo = self._step_memo().setdefault('lpnoise', {}) if self._perf else {}
            if nid not in memo:
                if self._lp_rng is None:
                    _seed = int(os.environ.get('DMPO_SENSE_SEED', os.environ.get('DMPO_SUMO_SEED', '0')) or 0)
                    self._lp_rng = np.random.default_rng(_seed + 7919)
                memo[nid] = self._lp_rng.normal(0.0, self._lanepos_sigma)
            n_base = n_base + memo[nid]
        return n_il, n_base

    def _crossing_conflict_curve(self, ego_id):
        """Curve-aware version of _crossing_conflict: uses each vehicle's ACTUAL internal-junction-
        lane path (SUMO geometry) and ARC-LENGTH distance-to-conflict, so turning vehicles are
        handled correctly (unlike the straight heading-ray approximation)."""
        res = {'urgency': 0.0, 'pet': 10.0, 'ta': 0.0, 'n': 0, 'cp': None, 'nid': None}
        self._ensure_jx_conflict_map()
        e_il, e_base = self._veh_internal(ego_id)
        if e_il is None:
            return res
        ev = self._cached(ego_id, 'speed') or 0.0
        E = self._cached(ego_id, 'position'); ea = self._cached(ego_id, 'angle')
        # A vehicle at a busy junction has MULTIPLE conflicts; it must HOLD if it should yield to ANY
        # crosser (not just its nearest). So track (a) the most-urgent YIELD requirement and (b) the
        # nearest conflict as the fallback 'go' signal.
        best_ttcp = float('inf'); best = None                  # nearest conflict -> (nid,cp,ta,pet,ttcp)
        yield_pet = float('inf'); yielded = None               # most-urgent conflict where EGO yields
        _dump = [] if os.environ.get('DMPO_CONFLICT_DUMP') else None  # collect ALL conflicts for gap analysis
        # M12-2: scan only the ego's OWN junction bucket (exact — any other junction's
        # internal-lane pair provably misses _jx_confl and would `continue`); bucket
        # order preserves the full-cache iteration order, so first-wins tie-breaks are
        # unchanged. Legacy full scan kept verbatim under the kill-switch.
        if self._perf:
            cands = [(nid, n_il, n_base, nv)
                     for nid, n_il, n_base, nv, n_is_ai
                     in self._junction_buckets().get(self._il_jid(e_il), ())
                     if nid != ego_id and n_is_ai]
        else:
            cands = []
            for nid, st in (self._vehicle_cache or {}).items():
                if nid == ego_id:
                    continue
                vt = st.get('type_id', '') or ''
                if not (vt.startswith('ai_vehicle') and not vt.startswith('ai_human_')):
                    continue
                n_il, n_base = self._veh_internal(nid)
                if n_il is None or n_base is None:
                    continue
                cands.append((nid, n_il, n_base, st.get('speed') or 0.0))
        for nid, n_il, n_base, nv in cands:
            info = self._jx_confl.get((e_il, n_il))
            if info is None:
                continue                                  # their paths don't cross
            cp, arc_e, arc_n = info
            dist_e = e_base + arc_e; dist_n = n_base + arc_n
            if self._occ_window:
                # ── CONSISTENT TWO-SIDED OCCUPANCY-WINDOW MODEL ──────────────────────────────
                Wz = self._occ_zone_w; Le = self._occ_veh_len; Ln = self._occ_veh_len
                ve = max(ev, 0.5); vn = max(nv, 0.5)
                if dist_e <= -(Wz + Le):
                    continue                              # ego's REAR fully cleared the zone -> through
                if dist_n <= -(Wz + Ln):
                    continue                              # neighbor's REAR fully cleared -> gone
                if dist_e > self.max_distance or dist_n > self.max_distance:
                    continue
                # occupancy time-windows: front reaches near edge -> rear clears far edge
                e_in = (dist_e - Wz) / ve; e_out = (dist_e + Wz + Le) / ve
                n_in = (dist_n - Wz) / vn; n_out = (dist_n + Wz + Ln) / vn
                overlap = (e_in < n_out) and (n_in < e_out)
                ta = 0.5 if e_in >= n_in else -0.5        # yield if MY window starts later (symmetric across the pair)
                if overlap:
                    pet = 0.0                             # windows collide -> the later vehicle must hold
                else:
                    pet = (n_in - e_out) if e_out <= n_in else (e_in - n_out)  # >=0 separation between windows
                ttcp_e = max(e_in, 0.0)                   # time until ego reaches the zone (for urgency/reporting)
                ttcp_n = max(n_in, 0.0)
            else:
                if dist_e <= 0.0:
                    continue                              # EGO already through the conflict point
                # BUMPER-TO-BUMPER (one-sided CLEAR_M): priority veh occupies until its REAR clears.
                if dist_n <= -self._occ_clear_m:
                    continue                              # priority vehicle's rear has fully cleared
                if dist_e > self.max_distance or dist_n > self.max_distance:
                    continue
                ttcp_e = dist_e / max(ev, 0.5)
                if dist_n <= 0.0:
                    # Front past the point but BODY still blocks the zone -> hold (imminent yield).
                    ttcp_n = 0.0; ta = 0.5; pet = 0.0
                else:
                    ttcp_n = dist_n / max(nv, 0.5)
                    ta = ttcp_e - ttcp_n
                    if abs(ta) < 0.5:                      # fair, position-based tie-break (arc-closer commits)
                        ta = -0.5 if dist_e <= dist_n else 0.5
                    pet = abs(ttcp_e - ttcp_n)
            res['n'] += 1
            if _dump is not None:
                _dump.append((nid, dist_e, dist_n, ttcp_e, ttcp_n, ta, pet))
            if ttcp_e < best_ttcp:
                best_ttcp = ttcp_e; best = (nid, cp, ta, pet, ttcp_e)
            if ta > 0.0 and pet < 2.0 and ttcp_e < 8.0:   # EGO arrives later at a real, imminent conflict -> yield
                if pet < yield_pet:
                    yield_pet = pet; yielded = (nid, cp, ta, pet, ttcp_e)
        if _dump is not None and ego_id in self._dump_ids and _dump:
            # gap = ttcp_e - ttcp_n: >0 means ego arrives AFTER crosser (must yield); the SET of gaps
            # over all simultaneous crossers reveals whether a common opening exists to chase.
            _parts = " | ".join(f"{n.replace('ai_','')}(tn={tn:.2f}s,gap={te-tn:+.2f})"
                                for (n, de, dn, te, tn, ta_, pet_) in sorted(_dump, key=lambda z: z[3]))
            print(f"[CONFLICTS] ego={ego_id.replace('ai_','')} ev={ev:4.1f}m/s te={_dump[0][3]:.2f}s "
                  f"crossers[{len(_dump)}]: {_parts}", flush=True)
        chosen = yielded if yielded is not None else best   # HOLD on any yield need, else report nearest
        if chosen is None:
            return res
        nid, cp, ta, pet, ttcpe = chosen
        res['nid'] = nid; res['cp'] = cp; res['ta'] = ta; res['pet'] = min(pet, 10.0)
        if pet < 2.0:
            if   ttcpe < 1.0: res['urgency'] = 1.0
            elif ttcpe < 2.0: res['urgency'] = 0.7
            elif ttcpe < 3.0: res['urgency'] = 0.4
            elif ttcpe < 5.0: res['urgency'] = 0.1
        return res

    # PART 3: path-aware conflict/gap observation block. K most-urgent crossers, each as
    # [gap, ttcp_ego, ttcp_crosser, yield_sign], plus [n_conflicts, min_gap]. K=3 -> 14 dims.
    PATH_CONFLICT_K = 3
    PATH_CONFLICT_BLOCK = PATH_CONFLICT_K * 4 + 2

    def _conflict_gap_block(self, ego_id):
        """The CLEAN pairwise conflict signal the policy was missing (weak point #1). For each
        κ-hop crosser whose curve-aware path crosses the ego's, compute the pairwise time-to-
        conflict-point for both and the GAP = ttcp_ego - ttcp_crosser (>0 => ego arrives later =>
        must yield). Return the K tightest-gap conflicts as a fixed salient block so the policy can
        LEARN to preserve/chase gaps instead of charging blind. Path-aware (uses actual internal-
        lane geometry + the proximity fix), unlike the failed current-velocity DMPO_CONFLICT_OBS."""
        K = self.PATH_CONFLICT_K
        HOR = 10.0                                    # normalization horizon (s)
        block = np.zeros(self.PATH_CONFLICT_BLOCK, dtype=np.float32)
        if not (self._curve_conflict and self._vehicle_cache):
            return block
        self._ensure_jx_conflict_map()
        e_il, e_base = self._veh_internal(ego_id)
        if e_il is None:
            return block
        ev = self._cached(ego_id, 'speed') or 0.0
        conflicts = []
        # M12-2: same shared per-junction bucket as _crossing_conflict_curve (the two
        # consumers previously duplicated this full-cache scan). Exactness argument
        # identical; conflicts.sort() below is stable, and bucket order preserves the
        # cache iteration order, so ties resolve as before.
        if self._perf:
            cands = [(nid, n_il, n_base, nv)
                     for nid, n_il, n_base, nv, n_is_ai
                     in self._junction_buckets().get(self._il_jid(e_il), ())
                     if nid != ego_id and n_is_ai]
        else:
            cands = []
            for nid, st in (self._vehicle_cache or {}).items():
                if nid == ego_id:
                    continue
                vt = st.get('type_id', '') or ''
                if not (vt.startswith('ai_vehicle') and not vt.startswith('ai_human_')):
                    continue
                n_il, n_base = self._veh_internal(nid)
                if n_il is None or n_base is None:
                    continue
                cands.append((nid, n_il, n_base, st.get('speed') or 0.0))
        for nid, n_il, n_base, nv in cands:
            info = self._jx_confl.get((e_il, n_il))
            if info is None:
                continue
            cp, arc_e, arc_n = info
            dist_e = e_base + arc_e; dist_n = n_base + arc_n
            if dist_e <= 0.0 or dist_n <= 0.0:
                continue
            if dist_e > self.max_distance or dist_n > self.max_distance:
                continue
            ttcp_e = dist_e / max(ev, 0.5)
            ttcp_n = dist_n / max(nv, 0.5)
            conflicts.append((ttcp_e, ttcp_n, ttcp_e - ttcp_n))
        if not conflicts:
            return block
        conflicts.sort(key=lambda c: (abs(c[2]), c[0]))     # tightest gap first, then soonest ego arrival
        for i, (te, tn, gap) in enumerate(conflicts[:K]):
            ys = 1.0 if gap > 0.3 else (-1.0 if gap < -0.3 else 0.0)
            block[i * 4 + 0] = float(np.clip(gap / HOR, -1.0, 1.0))
            block[i * 4 + 1] = float(np.clip(te / HOR, 0.0, 1.0))
            block[i * 4 + 2] = float(np.clip(tn / HOR, 0.0, 1.0))
            block[i * 4 + 3] = ys
        block[K * 4 + 0] = min(len(conflicts) / 5.0, 1.0)
        block[K * 4 + 1] = float(np.clip(min(c[2] for c in conflicts) / HOR, -1.0, 1.0))
        return block

    def _cached(self, veh_id, attr):
        """Read vehicle attribute from cache, fall back to SUMO API."""
        if self._vehicle_cache and veh_id in self._vehicle_cache:
            return self._vehicle_cache[veh_id].get(attr)
        # Fallback to direct SUMO call
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
        # P0.23: whitelist lane_position and edge_id so callers that use the
        # `_cached(...) or 0.0` idiom don't silently mask valid values. These
        # are read from the per-step cache; if missing we go to TraCI.
        elif attr == 'lane_position':
            return traci.vehicle.getLanePosition(veh_id)
        elif attr == 'edge_id':
            lane_id = traci.vehicle.getLaneID(veh_id)
            return traci.lane.getEdgeID(lane_id) if lane_id else ''
        return None

    def _detect_intersection_center(self):
        """Detect intersection center coordinates."""
        try:
            # Try to get center junction position
            junction_ids = traci.junction.getIDList()
            if 'center' in junction_ids:
                self.intersection_center = traci.junction.getPosition('center')
            else:
                # Default to origin
                self.intersection_center = (0.0, 0.0)
        except Exception as _e:
            self.intersection_center = (0.0, 0.0)
        if os.environ.get('DMPO_DEBUG_CENTER'):
            try:
                _jids = traci.junction.getIDList()
            except Exception as _e2:
                _jids = f"<traci err: {_e2}>"
            print(f"🎯 DEBUG_CENTER: intersection_center={self.intersection_center}  junction_ids={_jids}", flush=True)

    def redetect_intersection_center(self):
        """C1 FIX: __init__ ran _detect_intersection_center
        BEFORE traci connected → exception → (0,0) fallback, never corrected. That stale (0,0)
        made the env adjacency's conflict-edge gate unsatisfiable (dist-to-(0,0) ≥ 200 m on
        (200,200)-centered nets vs the 100 m radius) → the κ-hop cooperation graph NEVER linked
        crossing vehicles in any run. Call this AFTER traci is live (env does it lazily, gated
        DMPO_CENTER_REDETECT). Returns the corrected center (or the existing one on failure)."""
        try:
            junction_ids = traci.junction.getIDList()
            if 'center' in junction_ids:
                self.intersection_center = traci.junction.getPosition('center')
            if os.environ.get('DMPO_DEBUG_CENTER'):
                print(f"🎯 CENTER_REDETECT: intersection_center={self.intersection_center}", flush=True)
        except Exception as _e:
            if os.environ.get('DMPO_DEBUG_CENTER'):
                print(f"🎯 CENTER_REDETECT failed: {_e}", flush=True)
        return self.intersection_center

    # ═════════════════════════ OBS-V2 (DMPO_OBS_V2=1) ═════════════════════════
    # Layout (N = max_neighbors; production N=15 → obs_dim 241 = 61 + 12N):
    #   ego 12 | infra 5 | neighbors N×12 | context 8 | identity 4 | temporal 1 |
    #   conflict 4×7+3 = 31
    # Every slot bounded AT CONSTRUCTION (no z-filter, no ±5 clip — the C3 fix).
    # Zero is MEANINGFUL in v2 ("at the junction / knife-edge"), so the zero-obs
    # and padding use the NEUTRAL far/clear vector (_zero_obs_v2), never zeros.
    V2_EGO_DIM = 12
    V2_INFRA_DIM = 5
    V2_NB_FEAT = 12
    V2_CTX_DIM = 8
    V2_IDENT_DIM = 4
    V2_TEMP_DIM = 1
    V2_CONFLICT_K = 4
    V2_CONFLICT_BLOCK = V2_CONFLICT_K * 7 + 3          # 31
    # THE canonical flat orders — the single source of truth for every flatten
    # site in the repo (train + all eval scripts delegate here; R4/Step-9 fix).
    FLAT_ORDER_V1 = ('ego', 'infrastructure', 'neighbors', 'context', 'identity',
                     'collision_warning', 'conflict_gap')
    FLAT_ORDER_V2 = ('ego', 'infrastructure', 'neighbors', 'context', 'identity',
                     'temporal', 'conflict_gap')

    @staticmethod
    def obs_dim(max_neighbors=None, v2=None) -> int:
        """THE one obs-dim arithmetic (R8). config/intersection_dmpo.py calls this —
        it must never restate the formula."""
        if max_neighbors is None:
            max_neighbors = int(os.environ.get('DMPO_MAX_NEIGHBORS', '10'))
        if v2 is None:
            v2 = os.environ.get('DMPO_OBS_V2', '0') == '1'
        if v2:
            d = 61 + 12 * int(max_neighbors)
            if os.environ.get('DMPO_PROFILE_OBS', '0') == '1':
                # ARM-B driving-profile appendix: +4 ego, +4 per conflict slot
                d += 4 + 4 * ObservationProcessor.V2_CONFLICT_K
            return d
        d = 30 + 10 * int(max_neighbors)
        if os.environ.get('DMPO_PATH_CONFLICT_OBS', '0') == '1':
            d += 14
        return d

    @staticmethod
    def flatten(obs: Dict[str, np.ndarray], v2=None) -> np.ndarray:
        """THE one flat concat (Step 9). v2: raises on a missing key — silent
        zero-padding of an absent block is exactly the R4 blind-policy hazard.
        v1: reproduces the legacy concat (train _flatten_observation_numpy /
        eval_verbose_collisions) byte-for-byte, including the .get defaults and
        the conflict_gap append-iff-present."""
        if v2 is None:
            v2 = os.environ.get('DMPO_OBS_V2', '0') == '1'
        if v2:
            parts = []
            for k in ObservationProcessor.FLAT_ORDER_V2:
                if k not in obs:
                    raise KeyError(
                        f"OBS-V2 flatten: required block '{k}' missing from the obs dict "
                        f"(present: {sorted(obs.keys())}). A script bypassing "
                        f"process_observation's v2 branch would silently evaluate a "
                        f"blind policy — refusing.")
                parts.append(np.asarray(obs[k], dtype=np.float32).flatten())
            return np.concatenate(parts)
        ego = obs.get('ego', np.zeros(10))
        infra = obs.get('infrastructure', np.zeros(5))
        neigh = obs.get('neighbors', np.zeros((10, 10))).flatten()
        ctx = obs.get('context', np.zeros(10))
        ident = obs.get('identity', np.zeros(4))
        cw = obs.get('collision_warning', np.zeros(1))
        _parts = [ego, infra, neigh, ctx, ident, cw]
        _cg = obs.get('conflict_gap')
        if _cg is not None:
            _parts.append(_cg)
        return np.concatenate(_parts).astype(np.float32)

    def _rot_to_ego(self, dx, dy, angle_deg):
        """(right, fwd) of vector (dx,dy) in the frame of a vehicle heading angle_deg.
        Verbatim the verified legacy rotation (SUMO 0°=N, 90°=E, cw):
        right = dx·cos−dy·sin, fwd = dx·sin+dy·cos."""
        r = math.radians(angle_deg)
        c, s = math.cos(r), math.sin(r)
        return (dx * c - dy * s, dx * s + dy * c)

    def _lane_static_v2(self, lane_id):
        """(raw_max_speed, raw_lane_count, lane_width) — static per lane, permanent
        memo. Separate from the E5 cache (which stores legacy-normalized values)."""
        if not hasattr(self, '_lane_static_cache'):
            self._lane_static_cache = {}
        hit = self._lane_static_cache.get(lane_id)
        if hit is None:
            try:
                ms = traci.lane.getMaxSpeed(lane_id)
            except Exception:
                ms = self.max_speed
            try:
                lc = float(traci.edge.getLaneNumber(traci.lane.getEdgeID(lane_id)))
            except Exception:
                lc = 1.0
            try:
                lw = traci.lane.getWidth(lane_id) or 3.2
            except Exception:
                lw = 3.2
            hit = (ms, lc, lw)
            self._lane_static_cache[lane_id] = hit
        return hit

    def _turn_angle(self, vid):
        """Signed entry→exit turn angle at the next junction, from the internal lane's
        ACTUAL start/end tangents (topology-only, any net; fixes the geometrically
        inverted legacy string labels, M6). Math convention: positive = CCW = LEFT
        turn under right-hand traffic. None if no connection (encoded as (0,0) =
        'unknown', off the sin/cos manifold, distinct from straight=(0,1))."""
        memo = self._step_memo().setdefault('turn', {}) if self._perf else {}
        if vid in memo:
            return memo[vid]
        self._ensure_jx_conflict_map()
        il, _base = self._veh_internal(vid)
        ang = None
        if il is not None:
            # Entry-lane END tangent -> DESTINATION-edge lane START tangent.
            # NOT the internal shape's own segments: (a) many turn lanes are stored
            # as 2-point straight shapes (start==end tangent -> Δθ=0 for a real
            # turner), and (b) SUMO splits turns across SEQUENTIAL internal lanes,
            # so the first internal's end tangent is still entry-aligned (both
            # failure modes caught by the S6 forensic-smoke check). Static per
            # (il, next_edge) -> permanent memo.
            try:
                route = self._veh_route(vid)
                ri = self._veh_route_index(vid)
                next_edge = route[ri + 1] if (0 <= ri and ri + 1 < len(route)) else None
            except Exception:
                next_edge = None
            if not hasattr(self, '_turn_static_cache'):
                self._turn_static_cache = {}
            ckey = (il, next_edge)
            ang = self._turn_static_cache.get(ckey, '?')
            if ang == '?':
                ang = None
                sa = self._jx_shapes.get(il)
                a0 = a1 = None
                a0_clean = False
                try:
                    lane = self._cached(vid, 'lane_id') or ''
                    if lane and not lane.startswith(':'):
                        es = traci.lane.getShape(lane)
                        if len(es) >= 2:
                            a0 = math.atan2(es[-1][1] - es[-2][1], es[-1][0] - es[-2][0])
                            a0_clean = True
                except Exception:
                    pass
                if next_edge is not None:
                    try:
                        xs = traci.lane.getShape(next_edge + '_0')
                        if len(xs) >= 2:
                            a1 = math.atan2(xs[1][1] - xs[0][1], xs[1][0] - xs[0][0])
                    except Exception:
                        pass
                if sa is not None and len(sa) >= 2:
                    if a0 is None:
                        a0 = math.atan2(sa[1][1] - sa[0][1], sa[1][0] - sa[0][0])
                    if a1 is None:
                        a1 = math.atan2(sa[-1][1] - sa[-2][1], sa[-1][0] - sa[-2][0])
                if a0 is not None and a1 is not None:
                    ang = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
                # cache only the CLEAN (approach-lane-based) value: an in-junction
                # vehicle's a0 falls back to a mid-turn internal tangent, and caching
                # that would poison the key for every later approaching vehicle
                if a0_clean:
                    self._turn_static_cache[ckey] = ang
        memo[vid] = ang
        return ang

    def _turn_sincos(self, vid):
        ang = self._turn_angle(vid)
        if ang is None:
            return 0.0, 0.0
        return math.sin(ang), math.cos(ang)

    def _next_tls_v2(self, vid):
        """(present, green_val, dist_m) via traci.vehicle.getNextTLS — the per-vehicle
        TL signal (H3 fix: legacy read tl_ids[0] = single-intersection assumption)."""
        memo = self._step_memo().setdefault('ntls', {}) if self._perf else {}
        if vid in memo:
            return memo[vid]
        res = (0.0, 0.0, float('inf'))
        try:
            nt = traci.vehicle.getNextTLS(vid)
            if nt:
                _tls, _idx, dist, state = nt[0][0], nt[0][1], nt[0][2], nt[0][3]
                if dist <= self.conflict_horizon_m:
                    g = 1.0 if state in ('g', 'G') else (0.5 if state in ('y', 'Y') else 0.0)
                    res = (1.0, g, float(dist))
        except Exception:
            pass
        memo[vid] = res
        return res

    def _veh_len_v2(self, vid):
        st = (self._vehicle_cache or {}).get(vid)
        if st is not None:
            return float(st.get('length', 4.5) or 4.5)
        return 4.5

    def _jx_counts_v2(self):
        """Per-epoch {jid: (n_inside, n_within_R)} — junction-scoped occupancy and
        local density (M13 fix: legacy counted internal lanes ANYWHERE in the net)."""
        memo = self._step_memo() if self._perf else {'jxc': None}
        if memo.get('jxc') is not None:
            return memo['jxc']
        self._ensure_nj_maps()
        inside = {}
        near = {}
        R2 = self.obs_dist_norm ** 2
        for vid, st in (self._vehicle_cache or {}).items():
            lane = st.get('lane_id') or ''
            if lane.startswith(':'):
                jid = lane[1:].rsplit('_', 2)[0]
                inside[jid] = inside.get(jid, 0) + 1
            p = st.get('position')
            if p is None:
                continue
            f = self._next_junction_full(vid)
            if f is not None:
                if (p[0] - f[1]) ** 2 + (p[1] - f[2]) ** 2 <= R2:
                    near[f[0]] = near.get(f[0], 0) + 1
        memo['jxc'] = (inside, near)
        return memo['jxc']

    def _prof(self, vid):
        """ARM-B: normalized driving profile [age, mean_speed_ratio, mean_|a|, brake_frac],
        each in [0,1], from the env-published lifetime stats. Zeros for unknown vehicles
        (neutral: 'no history'). Normalizers: age by 750 steps (150s — a long trip),
        |a| by 4 m/s² (comfortable-to-firm braking range)."""
        st = self._veh_profiles.get(vid)
        if not st or st[0] <= 0:
            return (0.0, 0.0, 0.0, 0.0)
        n = float(st[0])
        return (min(n / 750.0, 1.0),
                min(st[1] / (n * self.max_speed), 1.0),
                min(st[2] / (n * 4.0), 1.0),
                st[3] / n)

    def _ego_v2(self, vid):
        """Block A — 12 dims: [jx_right, jx_fwd, dist_jx, speed, accel, head_sin,
        head_cos, turn_sin, turn_cos, lane_frac, in_junction, lat_off]
        (+4 gated ARM-B profile dims: age, mean_speed_ratio, mean_|a|, brake_frac)."""
        e = np.zeros(self.V2_EGO_DIM, dtype=np.float32)
        pos = self._cached(vid, 'position')
        ang = self._cached(vid, 'angle')
        if pos is None or ang is None:
            z = self._zero_obs_neutral()
            return z['ego'].copy()
        R = self.obs_dist_norm
        lane_id = self._cached(vid, 'lane_id') or ''
        in_jx = lane_id.startswith(':')
        nj = self._next_junction(vid)
        if in_jx:
            e[2] = 0.0
            if nj is not None:
                r, f = self._rot_to_ego(nj[0] - pos[0], nj[1] - pos[1], ang)
                e[0] = np.clip(r / R, -1.0, 1.0)
                e[1] = np.clip(f / R, -1.0, 1.0)
        elif nj is not None:
            dxj, dyj = nj[0] - pos[0], nj[1] - pos[1]
            r, f = self._rot_to_ego(dxj, dyj, ang)
            e[0] = np.clip(r / R, -1.0, 1.0)
            e[1] = np.clip(f / R, -1.0, 1.0)
            e[2] = np.clip(math.sqrt(dxj * dxj + dyj * dyj) / R, 0.0, 1.0)
        else:
            e[2] = 1.0                                   # no junction ahead = far/neutral
        v = self._cached(vid, 'speed') or 0.0
        e[3] = np.clip(v / self.max_speed, 0.0, 1.2)
        a = self._cached(vid, 'acceleration') or 0.0
        e[4] = np.clip(a / self.acc_pos, 0.0, 1.2) if a >= 0 else np.clip(a / self.acc_neg, -1.2, 0.0)
        ar = math.radians(ang)
        e[5] = math.sin(ar)
        e[6] = math.cos(ar)
        e[7], e[8] = self._turn_sincos(vid)
        if not in_jx:
            _ms, lc, lw = self._lane_static_v2(lane_id)
            li = 0.0
            st = (self._vehicle_cache or {}).get(vid)
            if st is not None and st.get('lane_index') is not None:
                li = float(st['lane_index'])
            else:
                try:
                    li = float(traci.vehicle.getLaneIndex(vid))
                except Exception:
                    li = 0.0
            e[9] = np.clip(li / max(lc - 1.0, 1.0), 0.0, 1.0)
        e[10] = 1.0 if in_jx else 0.0
        try:
            lat = traci.vehicle.getLateralLanePosition(vid)
            lw = self._lane_static_v2(lane_id)[2] if lane_id else 3.2
            e[11] = np.clip(lat / (lw / 2.0), -1.0, 1.0)
        except Exception:
            e[11] = 0.0
        if self._profile_obs:
            e[12], e[13], e[14], e[15] = self._prof(vid)
        return e

    def _infra_v2(self, vid):
        """Block B — 5 dims: [tl_present, tl_green, tl_dist, speed_limit, lane_count]."""
        f = np.zeros(self.V2_INFRA_DIM, dtype=np.float32)
        present, green, dist = self._next_tls_v2(vid)
        f[0] = present
        f[1] = green
        f[2] = np.clip(dist / self.obs_dist_norm, 0.0, 1.0) if present else 1.0
        lane_id = self._cached(vid, 'lane_id') or ''
        ms, lc, _lw = self._lane_static_v2(lane_id) if lane_id else (self.max_speed, 1.0, 3.2)
        f[3] = np.clip(ms / self.max_speed, 0.0, 2.0)
        f[4] = np.clip(lc / 4.0, 0.0, 1.0)
        return f

    def _nb_v2(self, ego_id, nid):
        """Block C row — 12 dims: [rel_right, rel_fwd, rel_v_right, rel_v_fwd,
        rel_head_sin, rel_head_cos, is_ai, nb_dist_jx, nb_same_jx, nb_ttc,
        nb_stopped, nb_dv]. Rear-bumper corrected ALWAYS (cached length)."""
        n = np.zeros(self.V2_NB_FEAT, dtype=np.float32)
        pe = self._cached(ego_id, 'position')
        ae = self._cached(ego_id, 'angle')
        pn = self._cached(nid, 'position')
        an = self._cached(nid, 'angle')
        if pe is None or pn is None or ae is None or an is None:
            raise ValueError('nb state unavailable')
        R = self.obs_dist_norm
        # rear bumper of the neighbour (its position is the FRONT bumper in SUMO)
        Ln = self._veh_len_v2(nid)
        anr = math.radians(an)
        rx = pn[0] - Ln * math.sin(anr)
        ry = pn[1] - Ln * math.cos(anr)
        r, f = self._rot_to_ego(rx - pe[0], ry - pe[1], ae)
        n[0] = np.clip(r / R, -1.0, 1.0)
        n[1] = np.clip(f / R, -1.0, 1.0)
        ve = self._cached(ego_id, 'speed') or 0.0
        vn = self._cached(nid, 'speed') or 0.0
        aer = math.radians(ae)
        vex, vey = ve * math.sin(aer), ve * math.cos(aer)
        vnx, vny = vn * math.sin(anr), vn * math.cos(anr)
        rvr, rvf = self._rot_to_ego(vnx - vex, vny - vey, ae)
        n[2] = np.clip(rvr / self.max_speed, -2.0, 2.0)
        n[3] = np.clip(rvf / self.max_speed, -2.0, 2.0)
        dh = math.radians(an - ae)
        n[4] = math.sin(dh)
        n[5] = math.cos(dh)
        vt = self._cached(nid, 'type_id') or ''
        # F3 fix (soundness review): 'ai' prefix wrongly marked ai_human_vehicle
        # (Krauss-driven background traffic) as coordinatable AI — use the same
        # predicate as the junction buckets.
        n[6] = 1.0 if (vt.startswith('ai_vehicle') and not vt.startswith('ai_human_')) else 0.0
        fj = self._next_junction_full(nid)
        ej = self._next_junction_full(ego_id)
        if fj is not None:
            dj = math.sqrt((pn[0] - fj[1]) ** 2 + (pn[1] - fj[2]) ** 2)
            n[7] = np.clip(dj / R, 0.0, 1.0)
            n[8] = 1.0 if (ej is not None and ej[0] == fj[0]) else 0.0
            n[9] = np.clip((dj / max(vn, 0.5)) / self.ttc_norm, 0.0, 1.0)
        else:
            n[7] = 1.0
            n[9] = 1.0
        n[10] = 1.0 if vn < 0.5 else 0.0
        st = (self._vehicle_cache or {}).get(nid) or {}
        vprev = st.get('prev_speed')
        if vprev is not None:
            n[11] = np.clip((vn - float(vprev)) / (self.acc_neg * self.step_length), -1.0, 1.0)
        return n

    def _ctx_v2(self, vid, all_vehicles):
        """Block D — 8 dims, ALL scoped to the ego's own junction (M13):
        [tti, jx_occupancy, queue_ahead, local_density, leader_ttc,
        leader_present, arrival_rank, steps_in_state]."""
        c = np.zeros(self.V2_CTX_DIM, dtype=np.float32)
        pos = self._cached(vid, 'position')
        v = self._cached(vid, 'speed') or 0.0
        lane_id = self._cached(vid, 'lane_id') or ''
        in_jx = lane_id.startswith(':')
        ejf = self._next_junction_full(vid)
        # [0] tti — M8 fix: stopped vehicles use the distance-based floor, never 1.0
        if in_jx:
            c[0] = 0.0
        elif ejf is not None and pos is not None:
            d = math.sqrt((pos[0] - ejf[1]) ** 2 + (pos[1] - ejf[2]) ** 2)
            c[0] = np.clip((d / max(v, self.v_floor)) / self.ttc_norm, 0.0, 1.0)
        else:
            c[0] = 1.0
        inside, near = self._jx_counts_v2()
        if ejf is not None:
            c[1] = np.clip(inside.get(ejf[0], 0) / 4.0, 0.0, 1.0)
            c[3] = np.clip(near.get(ejf[0], 0) / 12.0, 0.0, 1.0)
        # [2] queue ahead on my lane (E6b per-epoch memo; lane_position is cached)
        if not in_jx and lane_id:
            my_lp = ((self._vehicle_cache or {}).get(vid) or {}).get('lane_position')
            if my_lp is not None:
                nq = sum(1 for ovid, olp in self._lane_positions().get(lane_id, ())
                         if ovid != vid and olp > my_lp)
                c[2] = np.clip(nq / 8.0, 0.0, 1.0)
        # [4]/[5] leader TTC + presence (rear-end role of the deleted collision_warning)
        c[4] = 1.0
        try:
            ld = self._veh_leader(vid)
            if ld is not None and ld[1] <= self.conflict_horizon_m:
                c[5] = 1.0
                gap = ld[1]
                if gap < 0:
                    c[4] = 0.0
                else:
                    vl = (self._vehicle_cache or {}).get(ld[0], {}).get('speed')
                    if vl is None:
                        vl = 0.0
                    c[4] = np.clip((gap / max(v - vl, 0.1)) / self.ttc_norm, 0.0, 1.0)
        except Exception:
            pass
        # [6] arrival rank among same-junction crossers — SYMMETRIC floors (M8 fix
        # for the legacy asymmetric ego max(v,3) vs other max(v,0.1))
        if ejf is not None:
            bucket = self._junction_buckets().get(ejf[0], ())
            mine = None
            for bvid, _il, base, spd, _ai in bucket:
                if bvid == vid:
                    mine = base / max(spd, self.v_floor)
                    break
            if mine is not None:
                if self._lanepos_sigma > 0:
                    others = [self._crosser_view(bvid, _il, base)[1] / max(spd, self.v_floor)
                              for bvid, _il, base, spd, _ai in bucket if bvid != vid]
                else:
                    others = [base / max(spd, self.v_floor)
                              for bvid, _il, base, spd, _ai in bucket if bvid != vid]
                if others:
                    c[6] = sum(1 for t in others if t < mine) / float(len(others))
        # [7] steps in speed-state (legacy counter semantics, /100)
        speed_key = f"{vid}_speed_state"
        cur = 'stopped' if v < 1.0 else 'moving'
        if not hasattr(self, '_speed_state_tracking'):
            self._speed_state_tracking = {}
        prev = self._speed_state_tracking.get(speed_key, (cur, 0))
        steps = prev[1] + 1 if prev[0] == cur else 1
        self._speed_state_tracking[speed_key] = (cur, steps)
        c[7] = min(steps / 100.0, 1.0)
        return c

    def _temporal_v2(self, vid):
        """Block F — 1 dim: previous longitudinal command (0.0 at reset/spawn).
        env.prev_actions holds the env-format action DICT per vehicle
        ({'longitudinal': array([a]), ...}); accept the raw-array form too
        (found by the S9 forensic-smoke check — the dict form silently read 0.0)."""
        t = np.zeros(self.V2_TEMP_DIM, dtype=np.float32)
        pa = getattr(self, '_prev_actions', None)
        if pa:
            act = pa.get(vid)
            if isinstance(act, dict):
                act = act.get('longitudinal')
            if act is not None:
                try:
                    t[0] = np.clip(float(np.asarray(act).flatten()[0]), -1.0, 1.0)
                except Exception:
                    pass
        return t

    def _conflict_block_v2(self, ego_id):
        """Block G — 31 dims. v2-PRIVATE two-sided occupancy windows over the
        v2-PRIVATE successor-CHAINED conflict map _jx_confl_v2 (F29: full
        concatenated internal paths, so left-turn-across-traffic conflicts on a
        SECOND internal segment are visible from the approach; the shield keeps
        its own per-segment _jx_confl untouched). Crossers = ALL vehicle types,
        per-vehicle lengths.
        VERIFIER CORRECTION: deliberately does NOT read _occ_window/_occ_clear_m/
        _occ_zone_w/_occ_veh_len — those govern the gap SHIELD; this block's window
        math is self-contained so DMPO_OBS_V2 can never reconfigure the shield.
        Empty slots read as far/clear (ttcp=ttcp_other=occ_gap=1.0), never imminent."""
        K = self.V2_CONFLICT_K
        block = np.zeros(self.V2_CONFLICT_BLOCK, dtype=np.float32)
        for i in range(K):
            b = 7 * i
            block[b] = 1.0        # ttcp_ego
            block[b + 1] = 1.0    # ttcp_other
            block[b + 2] = 1.0    # occ_gap
        self._ensure_jx_conflict_map()
        e_il, e_base = self._veh_internal(ego_id)
        if e_il is None or e_base is None:
            block[29] = 0.0
            block[30] = 1.0 if ego_id in getattr(self, '_shield_engaged', ()) else 0.0
            return block
        # F32 fix (soundness review, HIGH — the go/no-go saturation that reproduced
        # the 4-lane chain-hold blindness): arrival times must answer the
        # COUNTERFACTUAL "if I release NOW, when do I get there?", so they use the
        # symmetric v_floor (3.0 m/s, the M8 constant) for BOTH sides. Flooring
        # arrivals at 0.5 made a stopped yielder read dist/0.5 (15 m -> 23.6 s), which
        # clipped ttcp/occ_gap to 'far/clear', put the pet_urgency gate (e_in<8 s) out
        # of reach until <7 m, and pinned whose-turn to 'yield' for every crosser —
        # i.e. the release decision was made on saturated features. The floor stays
        # SYMMETRIC (same on both sides) on purpose: an asymmetric floor would let two
        # stopped vehicles each compute itself as arriving first -> mutual GO.
        # OCCUPANCY (clear-out) keeps the conservative 0.5 floor, so a stopped vehicle
        # still reads as blocking the zone for a long time.
        Wz = 3.2                                       # v2-private crossing half-width
        ve_raw = self._cached(ego_id, 'speed') or 0.0
        Le = self._veh_len_v2(ego_id)
        ve_arr = max(ve_raw, self.v_floor)             # arrival: counterfactual restart
        ve_occ = max(ve_raw, 0.5)                      # occupancy: conservative
        # F29 (BLOCKER): resolve the successor CHAIN — on split connections the
        # across-traffic crossing lies on the SECOND internal segment, invisible in
        # the per-segment map until the turner is already committed. Map ego (and
        # below, each crosser) to its chain HEAD and re-base its distance onto the
        # CONCATENATED path, so a conflict on ANY segment is visible from the
        # approach with the correct remaining arc; a vehicle already ON a later
        # segment re-bases via its stored chain offset.
        e_h, e_off = self._jx_chain_pos.get(e_il, (e_il, 0.0))
        e_base = e_base - e_off
        conflicts = []
        # F1 (soundness-review BLOCKER): same-approach diverging movements share their
        # internal-lane START point, which the conflict map registers as a "crossing"
        # at arc 0 — flooding the K slots and firing pet_urgency during ordinary
        # car-following (the exact over-firing class v2 exists to kill). Mirror the
        # legacy same-edge exclusion CONSUMER-side (the shared map itself is shield
        # territory and stays untouched): skip pairs whose connections originate from
        # the same approach edge — same-lane following is the leader channel's job.
        ego_edge = self._cached(ego_id, 'edge_id') or ''
        ego_lane_full = self._cached(ego_id, 'lane_id') or ''
        for nid, n_il, n_base, nv_raw, _is_ai in self._junction_buckets().get(self._il_jid(e_il), ()):
            if nid == ego_id:
                continue
            if self._intent_unknown or self._lanepos_sigma > 0:
                n_il, n_base = self._crosser_view(nid, n_il, n_base)
            n_edge = (self._vehicle_cache.get(nid) or {}).get('edge_id') or ''
            if ego_edge and not ego_edge.startswith(':') and n_edge == ego_edge:
                # v2.1 (DMPO_CONFL_SAME_EDGE=1, default OFF): the blanket same-EDGE
                # skip blinds the policy to same-road DIVERGING conflicts (different
                # turns from adjacent lanes meeting just inside the box — 22% of the
                # champion's residual, forensically attributed). v2.1 narrows the
                # skip to same-LANE only (true following = leader channel's job);
                # different-lane same-edge pairs stay in, protected against the
                # shared-start pseudo-conflict by the existing arc<0.05-both check.
                if not self._confl_same_edge:
                    continue                           # legacy: same approach edge = following
                n_lane_full = (self._vehicle_cache.get(nid) or {}).get('lane_id') or ''
                if n_lane_full == ego_lane_full:
                    continue                           # same physical lane = following
            # F29: chain-resolve the crosser too. Same-chain pairs (n_h == e_h) have
            # no map entry by construction — same-path following is the leader
            # channel's job (the old per-segment map even held a joint-point pseudo-
            # conflict between consecutive segments of ONE connection; gone here).
            n_h, n_off = self._jx_chain_pos.get(n_il, (n_il, 0.0))
            info = self._jx_confl_v2.get((e_h, n_h))
            if info is None:
                continue
            if info[1] < 0.05 and info[2] < 0.05:
                continue                               # shared-origin pseudo-conflict (arc≈0 both)
            _cp, arc_e, arc_n = info
            d_e = e_base + arc_e
            d_n = (n_base - n_off) + arc_n
            Ln = self._veh_len_v2(nid)
            if d_e <= -(Wz + Le) or d_n <= -(Wz + Ln):
                continue                               # a rear fully cleared the zone
            if max(d_e, d_n) > self.conflict_horizon_m:
                continue
            vn_arr = max(nv_raw, self.v_floor)         # symmetric arrival floor (F32)
            vn_occ = max(nv_raw, 0.5)                  # conservative occupancy floor
            e_in = (d_e - Wz) / ve_arr
            e_out = (d_e + Wz + Le) / ve_occ
            n_in = (d_n - Wz) / vn_arr
            n_out = (d_n + Wz + Ln) / vn_occ
            if e_out <= n_in:
                occ_gap_s = n_in - e_out
            elif n_out <= e_in:
                occ_gap_s = e_in - n_out
            else:
                occ_gap_s = -min(e_out - n_in, n_out - e_in)   # NEGATIVE = overlap size
            conflicts.append((0 if occ_gap_s < 0.0 else 1, e_in, nid,
                              n_in, occ_gap_s, nv_raw))
        # overlapping first, then soonest ego arrival, then id (hash-independent)
        conflicts.sort(key=lambda t: (t[0], t[1], t[2]))
        # ARM-B: ego profile computed once; the appendix lives AFTER the intact
        # legacy 31 dims (base = K*7+3), 4 per slot, so indices 0..30 are
        # byte-identical to the flag-off layout.
        _ep = self._prof(ego_id) if self._profile_obs else None
        _pb = K * 7 + 3
        for i, (_ov, e_in, nid, n_in, occ_gap_s, nv_raw) in enumerate(conflicts[:K]):
            b = 7 * i
            block[b] = np.clip(max(e_in, 0.0) / self.ttcp_norm, 0.0, 1.0)
            block[b + 1] = np.clip(max(n_in, 0.0) / self.ttcp_norm, 0.0, 1.0)
            block[b + 2] = np.clip(occ_gap_s / self.ttcp_norm, -1.0, 1.0)
            if self._intent_unknown and self._straight_il(nid) is not None:
                block[b + 3], block[b + 4] = 0.0, 1.0      # assumed straight (intent unknown)
            else:
                block[b + 3], block[b + 4] = self._turn_sincos(nid)
            block[b + 5] = 1.0 if nv_raw < 0.5 else 0.0
            de_in = e_in - n_in
            block[b + 6] = 1.0 if de_in > 0.2 else (-1.0 if de_in < -0.2 else 0.0)
            if _ep is not None:
                # Raw SIGNED arrival delta (positive = ego arrives FIRST), the
                # epsilon the ±0.2s quantization above erases — resolution, not
                # a rule — plus ego-minus-other profile deltas (distinguishability).
                # NOTE deliberate sign INVERSION vs block[b+6] above (+1 = ego
                # LATER there): readers/miners should expect anti-correlation.
                # Appendix validity is conditioned on the slot's legacy dims
                # (an EMPTY slot leaves the appendix 0.0, same as a perfect
                # no-history tie — the far/clear prefix disambiguates).
                _np_ = self._prof(nid)
                a = _pb + 4 * i
                block[a] = np.clip((n_in - e_in) / 4.0, -1.0, 1.0)
                block[a + 1] = np.clip(_ep[0] - _np_[0], -1.0, 1.0)
                block[a + 2] = np.clip(_ep[1] - _np_[1], -1.0, 1.0)
                block[a + 3] = np.clip(_ep[3] - _np_[3], -1.0, 1.0)
        # summary — pet_urgency implements the VALIDATED selective-stop semantics
        # (fires only for a GENUINE conflict: gap below the PET gate AND arrival soon)
        gating = [e_in for _ov, e_in, _nid, _nin, occ, _nv in conflicts
                  if occ < self.pet_gate_s and e_in < 8.0]
        block[28] = np.clip(1.0 - min(gating) / 8.0, 0.0, 1.0) if gating else 0.0
        block[29] = np.clip(len(conflicts) / 6.0, 0.0, 1.0)
        block[30] = 1.0 if ego_id in getattr(self, '_shield_engaged', ()) else 0.0
        # Forensic-only (DMPO_CONFL_DEBUG_SLOTS=1, default OFF; read-only): stash the
        # RANKED conflict list so an external logger can attribute each slot to a
        # vehicle and recover the TRUE crosser count that block[29] clips at 6.
        # Keyed by ego so a stale read from the early-return path is detectable.
        if os.environ.get('DMPO_CONFL_DEBUG_SLOTS', '0') == '1':
            if not hasattr(self, '_confl_debug_log'):
                self._confl_debug_log = []
            self._confl_debug_log.append({
                'ego': ego_id, 'n_total': len(conflicts),
                'slots': [(_nd, float(_e), float(_n)) for (_o, _e, _nd, _n, _g, _v)
                          in conflicts[:K]],
            })
        return block

    def _zero_obs_neutral(self):
        """The v2 zero-obs (R2): an ABSENT vehicle must read as far/clear, not as
        'at the junction, arriving now, knife-edge conflict' (which is what all-zeros
        means in v2). Built once, .copy()'d per use."""
        if self._zero_obs_v2 is None:
            ego = np.zeros(self.V2_EGO_DIM, dtype=np.float32)
            ego[2] = 1.0                                  # dist_jx = far
            infra = np.zeros(self.V2_INFRA_DIM, dtype=np.float32)
            infra[2] = 1.0                                # tl_dist = far
            nb = np.zeros((self.max_neighbors, self.V2_NB_FEAT), dtype=np.float32)
            nb[:, 7] = 1.0                                # nb_dist_jx = far
            nb[:, 9] = 1.0                                # nb_ttc = far
            ctx = np.zeros(self.V2_CTX_DIM, dtype=np.float32)
            ctx[0] = 1.0                                  # tti = far
            ctx[4] = 1.0                                  # leader_ttc = clear
            conflict = np.zeros(self.V2_CONFLICT_BLOCK, dtype=np.float32)
            for i in range(self.V2_CONFLICT_K):
                conflict[7 * i] = 1.0
                conflict[7 * i + 1] = 1.0
                conflict[7 * i + 2] = 1.0
            self._zero_obs_v2 = {
                'ego': ego,
                'infrastructure': infra,
                'neighbors': nb,
                'neighbor_mask': np.zeros(self.max_neighbors, dtype=np.float32),
                'context': ctx,
                'identity': np.zeros(self.V2_IDENT_DIM, dtype=np.float32),
                'temporal': np.zeros(self.V2_TEMP_DIM, dtype=np.float32),
                'conflict_gap': conflict,
            }
        return self._zero_obs_v2

    def _process_observation_v2(self, vehicle_id, all_vehicles, adjacency_matrix, vehicle_index):
        """The v2 dict assembly — no collision_warning (C5: its rear-end role is
        ctx leader_ttc, its crossing role is the conflict block's pet_urgency)."""
        try:
            neighbor_states, neighbor_mask = self._get_neighbor_states(
                vehicle_id, all_vehicles, adjacency_matrix, vehicle_index)
            obs = {
                'ego': self._ego_v2(vehicle_id),
                'infrastructure': self._infra_v2(vehicle_id),
                'neighbors': neighbor_states,
                'neighbor_mask': neighbor_mask,
                'context': self._ctx_v2(vehicle_id, all_vehicles),
                'identity': self._get_agent_identity(vehicle_id),
                'temporal': self._temporal_v2(vehicle_id),
                'conflict_gap': self._conflict_block_v2(vehicle_id),
            }
            if self._export_nb_ids:
                # MATCHED MODEL LOSS: extra key (like neighbor_mask) — flatten() walks
                # FLAT_ORDER_V2 only, so this never reaches the flat vector/obs_dim.
                obs['nb_ids'] = tuple(self._nb_ids_scratch)
            return obs
        except traci.exceptions.TraCIException:
            z = self._zero_obs_neutral()
            z_copy = {k: v.copy() for k, v in z.items()}
            if self._export_nb_ids:
                z_copy['nb_ids'] = (None,) * self.max_neighbors
            return z_copy
    # ═══════════════════════ end OBS-V2 section ═══════════════════════

    def process_observation(self, vehicle_id: str, all_vehicles: Set[str],
                          adjacency_matrix: np.ndarray, vehicle_index: int) -> Dict[str, np.ndarray]:
        """
        Process observation for a single vehicle.
        
        Args:
            vehicle_id: ID of ego vehicle
            all_vehicles: Set of all vehicle IDs in simulation
            adjacency_matrix: Communication adjacency matrix
            vehicle_index: Index of vehicle in adjacency matrix
            
        Returns:
            Dictionary containing structured observation
        """
        if self._v2:                                     # OBS-V2: whole-dict branch
            return self._process_observation_v2(
                vehicle_id, all_vehicles, adjacency_matrix, vehicle_index)
        try:
            # Get ego vehicle state
            ego_state = self._get_ego_state(vehicle_id)
            
            # Get infrastructure state
            infrastructure_state = self._get_infrastructure_state(vehicle_id)
            
            # Get κ-hop neighbors
            neighbor_states, neighbor_mask = self._get_neighbor_states(
                vehicle_id, all_vehicles, adjacency_matrix, vehicle_index
            )
            
            # Get derived context features
            context_features = self._get_context_features(vehicle_id, all_vehicles)

            # Agent identity: one-hot approach direction + route intent
            # This gives the shared policy network a strong, unambiguous
            # signal about WHERE this vehicle is, preventing correlated
            # collapse where all stopped vehicles get the same action.
            # (Christianos et al. 2020, "Shared Experience Actor-Critic")
            agent_identity = self._get_agent_identity(vehicle_id)

            # Collision warning: graduated signal based on distance to leader.
            # Instead of SUMO overriding the policy (Krauss mode) or hardcoding
            # a penalty, we give the vehicle INFORMATION and let it learn what
            # to do. The policy sees "you're about to collide" and must learn
            # to brake on its own through reward feedback.
            collision_warning = self._get_collision_warning(vehicle_id)

            observation = {
                'ego': ego_state,
                'infrastructure': infrastructure_state,
                'neighbors': neighbor_states,
                'neighbor_mask': neighbor_mask,
                'context': context_features,
                'identity': agent_identity,
                'collision_warning': collision_warning
            }
            if self._path_conflict_obs:                      # PART 3: appended at END of flat obs
                observation['conflict_gap'] = self._conflict_gap_block(vehicle_id)

            return observation

        except traci.exceptions.TraCIException:
            # Return zero observation if vehicle not found
            return self._get_zero_observation()
    
    def _get_ego_state(self, vehicle_id: str) -> np.ndarray:
        """
        Get ego vehicle state representation.

        Returns: [pos_x, pos_y, velocity, acceleration, heading, lane_id,
                 distance_to_intersection, route_direction, time_to_destination,
                 lateral_position]
        """
        try:
            # Position (from cache)
            position = self._cached(vehicle_id, 'position')
            pos_x = position[0] / self.max_distance  # Normalized
            pos_y = position[1] / self.max_distance

            # Velocity and acceleration (from cache)
            velocity = self._cached(vehicle_id, 'speed') / self.max_speed
            acceleration = self._cached(vehicle_id, 'acceleration') / self.max_accel

            # Heading (normalized to [-1, 1])
            angle = self._cached(vehicle_id, 'angle')
            heading = (angle % 360) / 180.0 - 1.0

            # Lane information (from cache)
            lane_id = self._cached(vehicle_id, 'lane_id')
            lane_index = self._encode_lane_id(lane_id)

            # Distance to intersection — to the vehicle's NEXT junction (any net) when
            # DMPO_NEXT_JUNCTION_OBS=1; else legacy single 'center'.
            _nj = self._next_junction(vehicle_id) if self._fix_distance else None
            if _nj is not None:
                dist_to_intersection = math.sqrt(
                    (position[0] - _nj[0])**2 + (position[1] - _nj[1])**2
                ) / self.max_distance
            elif self.intersection_center:
                dist_to_intersection = math.sqrt(
                    (position[0] - self.intersection_center[0])**2 +
                    (position[1] - self.intersection_center[1])**2
                ) / self.max_distance
            else:
                dist_to_intersection = 0.0

            # Route direction (simplified: -1=left, 0=straight, 1=right)
            route_direction = self._get_route_direction(vehicle_id)

            # Time to destination (simplified as remaining route length)
            time_to_dest = self._get_time_to_destination(vehicle_id)

            # Lateral lane position: offset from lane center, normalized to [-1, 1]
            # -1 = at right edge, 0 = centered, +1 = at left edge
            try:
                lat_pos = traci.vehicle.getLateralLanePosition(vehicle_id)
                lane_width = traci.lane.getWidth(lane_id) if lane_id else 3.2
                lateral_position = np.clip(lat_pos / (lane_width / 2), -1.0, 1.0)
            except Exception:
                lateral_position = 0.0

            ego_state = np.array([
                pos_x, pos_y, velocity, acceleration, heading,
                lane_index, dist_to_intersection, route_direction, time_to_dest,
                lateral_position
            ], dtype=np.float32)

            return ego_state

        except traci.exceptions.TraCIException:
            return np.zeros(10, dtype=np.float32)
    
    def _get_infrastructure_state(self, vehicle_id: str) -> np.ndarray:
        """
        Get infrastructure state representation.
        
        Returns: [traffic_light_phase, traffic_light_timer, speed_limit, lane_count, visibility]
        """
        try:
            # Traffic light state — DROPPED (zeroed) under DMPO_NEXT_JUNCTION_OBS: the TL-phase
            # feature is parsed with a single-4-approach assumption that is meaningless on real
            # multi-signal nets (ATSC NY), so feeding it garbage hurts more than helps. Zero it
            # rather than mis-parse; the policy relies on geometry + neighbors instead.
            if self._nj_on:
                tl_phase, tl_timer = 0.0, 0.0
            else:
                tl_phase, tl_timer = self._get_traffic_light_state(vehicle_id)

            # Speed limit
            lane_id = self._cached(vehicle_id, 'lane_id')
            if self._perf:
                # E5: (raw max-speed, lane_count) are static per lane — permanent memo.
                # A raising getMaxSpeed propagates UNCACHED so the outer except sees the
                # identical exception; the lane_count fallback (bare except -> 1.0) is
                # baked into the memoized value exactly as computed below.
                _infra = self._infra_cache.get(lane_id)
                if _infra is None:
                    _raw_ms = traci.lane.getMaxSpeed(lane_id)
                    try:
                        edge_id = traci.lane.getEdgeID(lane_id)
                        _lc = min(float(traci.edge.getLaneNumber(edge_id)) / 5.0, 1.0)
                    except:
                        _lc = 1.0
                    _infra = (_raw_ms, _lc)
                    self._infra_cache[lane_id] = _infra
                speed_limit = _infra[0] / self.max_speed
                lane_count = _infra[1]
            else:
                speed_limit = traci.lane.getMaxSpeed(lane_id) / self.max_speed

                # Lane count (number of lanes on current edge)
                try:
                    edge_id = traci.lane.getEdgeID(lane_id)
                    lane_count = min(float(traci.edge.getLaneNumber(edge_id)) / 5.0, 1.0)  # Normalized, supports up to 5 lanes
                except:
                    lane_count = 1.0  # Default value
            
            # Visibility (simplified as 1.0 for now)
            visibility = 1.0
            
            infrastructure_state = np.array([
                tl_phase, tl_timer, speed_limit, lane_count, visibility
            ], dtype=np.float32)
            
            return infrastructure_state
            
        except traci.exceptions.TraCIException:
            return np.zeros(5, dtype=np.float32)
    
    def _spatial_index(self, all_vehicles):
        """M12-3: uniform grid over vehicle positions, cell = perception radius, so any
        vehicle within the radius of a query point lies in the 3x3 cell block around the
        query cell (|dx| <= s => cell index differs by at most 1). ONE O(M) build per
        epoch instead of an O(N·M) rescan per ego. Positions come from the same _cached
        reads as the legacy loop."""
        memo = self._step_memo()
        if memo['sh'] is not None:
            return memo['sh']
        s = max(float(self.communication_radius), 1.0)
        grid = {}
        for veh in all_vehicles:
            try:
                p = self._cached(veh, 'position')
            except Exception:
                continue
            if p is None:
                continue
            grid.setdefault((int(math.floor(p[0] / s)), int(math.floor(p[1] / s))), []).append((veh, p))
        memo['sh'] = (s, grid)
        return memo['sh']

    def _get_neighbor_states(self, vehicle_id: str, all_vehicles: Set[str],
                           adjacency_matrix: np.ndarray, vehicle_index: int) -> tuple:
        """
        Get the ego's perceived neighbors.

        P0.3: REPLACED κ-hop-first + slot-starved-humans selection with
        DISTANCE-BASED nearest-N from any vehicle type. The κ-hop graph
        (built over AI-only in env._update_adjacency_matrix) remains the
        SOURCE OF TRUTH for V2V coordination (used by GCN value/policy
        aggregation), but the per-vehicle observation now includes the
        N nearest vehicles by Euclidean distance regardless of type so the
        policy can perceive humans without slot-competition with AI κ-hop.

        Each neighbor slot's feature vector still carries the is_ai flag
        (see _get_neighbor_info) so the MLP can learn to weight AI vs human
        appropriately.

        Returns:
            neighbor_states: Array of neighbor information [max_neighbors, 10]
            neighbor_mask: Binary mask indicating valid neighbors [max_neighbors]
        """
        _nfeat = self.V2_NB_FEAT if self._v2 else 10     # OBS-V2: 12-feature rows
        if self._v2:
            # F2 fix (soundness review): empty/failed slots must be the NEUTRAL row
            # (far/clear), never zeros — an all-zero v2 row reads as a phantom vehicle
            # AT the ego's position arriving NOW (nb_dist_jx=0, nb_ttc=0).
            neighbor_states = np.tile(self._zero_obs_neutral()['neighbors'][0],
                                      (self.max_neighbors, 1)).astype(np.float32)
        else:
            neighbor_states = np.zeros((self.max_neighbors, _nfeat), dtype=np.float32)
        neighbor_mask = np.zeros(self.max_neighbors, dtype=np.float32)
        if self._export_nb_ids:
            # MATCHED MODEL LOSS side-channel: slot k → vehicle id, recorded ONLY for
            # slots whose row was actually written (mask=1); every early return /
            # except path leaves the remaining slots None (= empty / neutral row).
            self._nb_ids_scratch = [None] * self.max_neighbors

        try:
            ego_position = self._cached(vehicle_id, 'position')
            if ego_position is None:
                return neighbor_states, neighbor_mask

            # P0.3: collect all vehicles within perception_radius, sort by distance,
            # take the nearest max_neighbors. NOTE: also removes the previous
            # `radius/2` halving for humans (Bug B) — all vehicles get the same
            # perception radius (= communication_radius, defaults to 100m).
            perception_radius_sq = self.communication_radius ** 2
            candidates = []  # list of (dist_sq, vehicle_id)
            if self._perf and self._spatial_hash_on:
                # M12-3 (gated DMPO_OBS_SPATIAL_HASH): query the shared per-epoch grid.
                # d2 uses the IDENTICAL expression/operand order as the legacy loop, and
                # the (d2, vehicle_id) key is a total order, so selection incl. ties is
                # unchanged. Sole documented divergence: a vehicle absent from the cache
                # whose live getPosition raises is skipped at build time instead of
                # zeroing the entire ego's neighbor block via the outer except.
                s, grid = self._spatial_index(all_vehicles)
                cx = int(math.floor(ego_position[0] / s))
                cy = int(math.floor(ego_position[1] / s))
                for gx in (cx - 1, cx, cx + 1):
                    for gy in (cy - 1, cy, cy + 1):
                        for veh, other_pos in grid.get((gx, gy), ()):
                            if veh == vehicle_id:
                                continue
                            dx = ego_position[0] - other_pos[0]
                            dy = ego_position[1] - other_pos[1]
                            d2 = dx * dx + dy * dy
                            if d2 <= perception_radius_sq:
                                candidates.append((d2, veh))
            else:
                for veh in all_vehicles:
                    if veh == vehicle_id:
                        continue
                    other_pos = self._cached(veh, 'position')
                    if other_pos is None:
                        continue
                    dx = ego_position[0] - other_pos[0]
                    dy = ego_position[1] - other_pos[1]
                    d2 = dx * dx + dy * dy
                    if d2 <= perception_radius_sq:
                        candidates.append((d2, veh))

            # Sort by distance ascending (PYTHONHASHSEED-independent because
            # tuple key (dist_sq, vehicle_id) tie-breaks lexicographically on the
            # vehicle ID string — deterministic).
            candidates.sort()
            neighbor_vehicles = [v for _, v in candidates[:self.max_neighbors]]

            # Process neighbor information
            for i, neighbor_id in enumerate(neighbor_vehicles):
                try:
                    neighbor_info = (self._nb_v2(vehicle_id, neighbor_id) if self._v2
                                     else self._get_neighbor_info(vehicle_id, neighbor_id))
                    neighbor_states[i] = neighbor_info
                    neighbor_mask[i] = 1.0
                    if self._export_nb_ids:
                        self._nb_ids_scratch[i] = neighbor_id
                except Exception:
                    continue

            return neighbor_states, neighbor_mask

        except Exception:
            return neighbor_states, neighbor_mask
    
    def _get_neighbor_info(self, ego_id: str, neighbor_id: str) -> np.ndarray:
        """
        Get information about a specific neighbor.

        Returns: [rel_x, rel_y, rel_velocity, rel_heading, lane_id, is_ai,
                  nb_dist_to_junction, nb_abs_vx, nb_abs_vy, nb_ttc]   (10 features)

        is_ai (binary): 1.0 if the neighbor is an AI-controlled vehicle
        (vType startswith 'ai'), 0.0 otherwise (humans, trucks, unknown).
        This is the critical signal that lets the policy distinguish
        controllable neighbors (AI, will respond to coordination signals)
        from uncontrollable ones (humans, ignore-the-policy Krauss drivers).

        Previously this slot held `type_encoded` which mapped vehicle type
        via _encode_vehicle_type — but that helper checked
        startswith('human_driver'), never matching the actual 'human_vehicle'
        vType used in cn_plus. Humans silently fell through to the
        "unknown type" branch (0.25), so the policy had no reliable way
        to tell humans from AI. Fixing that here with an unambiguous
        binary feature.
        """
        try:
            ego_pos = self._cached(ego_id, 'position')
            neighbor_pos = self._cached(neighbor_id, 'position')

            # Relative position
            rel_x = (neighbor_pos[0] - ego_pos[0]) / self.max_distance
            rel_y = (neighbor_pos[1] - ego_pos[1]) / self.max_distance

            # Relative velocity
            ego_vel = self._cached(ego_id, 'speed')
            neighbor_vel = self._cached(neighbor_id, 'speed')
            rel_velocity = (neighbor_vel - ego_vel) / self.max_speed

            # Relative heading
            ego_angle = self._cached(ego_id, 'angle')
            neighbor_angle = self._cached(neighbor_id, 'angle')
            rel_heading = ((neighbor_angle - ego_angle) % 360) / 180.0 - 1.0

            # Front-vs-rear bumper correction (Fix 3): SUMO getPosition returns the
            # FRONT bumper. A following AI collides with the leader's REAR bumper,
            # so the raw front-to-front rel_x overestimates the collision-relevant
            # gap by the leader's length (~4.5m). When DMPO_OBS_REAR_BUMPER=1, shift
            # the neighbor's reference point backward along its heading by its length
            # so rel_(x,y) reflects the neighbor's REAR bumper.
            # SUMO angle: 0°=N, 90°=E, clockwise; heading unit vec = (sin a, cos a);
            # rear = front - L * heading.
            if os.environ.get('DMPO_OBS_REAR_BUMPER', '0') == '1':
                try:
                    nb_len = float(traci.vehicle.getLength(neighbor_id))
                except Exception:
                    nb_len = 4.5
                nb_a = math.radians(neighbor_angle)
                _rx_before = rel_x
                rel_x -= (nb_len * math.sin(nb_a)) / self.max_distance
                rel_y -= (nb_len * math.cos(nb_a)) / self.max_distance
                if os.environ.get('DMPO_OBS_REAR_BUMPER_DEBUG', '0') == '1':
                    dx = -(nb_len * math.sin(nb_a)); dy = -(nb_len * math.cos(nb_a))
                    mag = (dx*dx + dy*dy) ** 0.5
                    print(f"[REAR_BUMPER] ego={ego_id} nbr={neighbor_id} nb_angle={neighbor_angle:.0f}° "
                          f"shift(dx={dx:+.2f}, dy={dy:+.2f}) |shift|={mag:.2f}m", flush=True)

            # Latency-aware predict-ahead (Fix 2): shift neighbor relative
            # position by Δt of constant-velocity extrapolation. Compensates
            # for the observation→action→effect delay (~step_length seconds).
            # Reference: Ramstedt & Pal NeurIPS 2019 ("Real-Time RL"); Derman
            # et al. ICLR 2021 ("Acting in Delayed Environments"); standard
            # motion-prediction practice in CARLA leaderboard winners.
            # Gated by DMPO_OBS_PREDICT_AHEAD_S (seconds). Default 0 = off.
            predict_ahead_s = float(os.environ.get('DMPO_OBS_PREDICT_AHEAD_S', '0.0'))
            if predict_ahead_s > 0:
                # SUMO angle: 0°=N, 90°=E, clockwise. pos = (x_east, y_north).
                # vx = speed·sin(angle), vy = speed·cos(angle).
                ego_a_rad = math.radians(ego_angle)
                nb_a_rad = math.radians(neighbor_angle)
                ego_vx = ego_vel * math.sin(ego_a_rad)
                ego_vy = ego_vel * math.cos(ego_a_rad)
                nb_vx = neighbor_vel * math.sin(nb_a_rad)
                nb_vy = neighbor_vel * math.cos(nb_a_rad)
                rel_x += ((nb_vx - ego_vx) * predict_ahead_s) / self.max_distance
                rel_y += ((nb_vy - ego_vy) * predict_ahead_s) / self.max_distance

            # Lane encoding (kept — may still carry weak signal for same-lane following)
            neighbor_lane = self._cached(neighbor_id, 'lane_id')
            lane_encoded = self._encode_lane_id(neighbor_lane)

            # Controllability flag: 1.0 if AI (policy can coordinate with it),
            # 0.0 otherwise. Handles SUMO's decorated type IDs like
            # 'ai_vehicle@specific_id' via startswith.
            # Must match the env's policy-tracking rule in _update_vehicle_tracking
            # so the policy doesn't mark background drivers (ai_human_vehicle) as
            # coordinatable when they're actually SUMO-controlled.
            veh_type = self._cached(neighbor_id, 'type_id') or ''
            is_ai = 1.0 if (veh_type.startswith('ai_vehicle') and not veh_type.startswith('ai_human_')) else 0.0

            # Distance from neighbor to the intersection — primary signal for
            # "who arrives first" FCFS coordination at unsignalized junctions.
            # When predict-ahead is on, shift neighbor's position by v·Δt
            # before computing dist-to-junction so this signal is also
            # latency-compensated.
            # Under DMPO_NEXT_JUNCTION_OBS, measure to the NEIGHBOR's own next junction (any net);
            # else legacy single 'center'.
            _nnj = self._next_junction(neighbor_id) if self._fix_distance else None
            _ctr = _nnj if _nnj is not None else self.intersection_center
            if _ctr is not None:
                cx, cy = _ctr
                nb_x = neighbor_pos[0]
                nb_y = neighbor_pos[1]
                if predict_ahead_s > 0:
                    nb_x += neighbor_vel * math.sin(math.radians(neighbor_angle)) * predict_ahead_s
                    nb_y += neighbor_vel * math.cos(math.radians(neighbor_angle)) * predict_ahead_s
                nb_dist_to_junction = math.hypot(nb_x - cx, nb_y - cy) / self.max_distance
            else:
                nb_dist_to_junction = 1.0  # far

            # Crossing-conflict features (added for the 4-lane all-AI run):
            # (a) neighbor ABSOLUTE velocity vector — encodes its direction of travel.
            #     rel_velocity above is only a scalar speed difference and loses
            #     direction, so the policy can't see whether a crossing neighbor is
            #     heading toward the conflict point. SUMO angle 0°=N,90°=E,cw →
            #     vx = v·sin(a), vy = v·cos(a). Normalized by max_speed.
            _nb_a_rad = math.radians(neighbor_angle)
            nb_abs_vx = (neighbor_vel * math.sin(_nb_a_rad)) / self.max_speed
            nb_abs_vy = (neighbor_vel * math.cos(_nb_a_rad)) / self.max_speed
            # (b) time-to-junction TTC (seconds, normalized): when this neighbor
            #     reaches the conflict point = raw distance / raw speed. /0 guarded
            #     on stopped vehicles; normalized by 30 s and clipped to [0,1].
            _raw_dist_m = nb_dist_to_junction * self.max_distance
            nb_ttc = min((_raw_dist_m / max(neighbor_vel, 0.5)) / 30.0, 1.0)

            # ── EGO-FRAME DIRECTION (DMPO_CONFLICT_OBS): rotate the neighbor's relative POSITION
            # and ABSOLUTE-VELOCITY vector into the ego's heading frame (x=right, y=forward) so
            # "a car crossing my path" looks identical regardless of the junction's global
            # orientation → generalizes across grids/cities. SUMO heading unit=(sin a, cos a),
            # right=(cos a, -sin a); forward'=g·heading, right'=g·right. Gated → legacy(global) off.
            if self._conflict_obs:
                _c = math.cos(math.radians(ego_angle)); _s = math.sin(math.radians(ego_angle))
                rel_x, rel_y = (rel_x * _c - rel_y * _s), (rel_x * _s + rel_y * _c)
                nb_abs_vx, nb_abs_vy = (nb_abs_vx * _c - nb_abs_vy * _s), (nb_abs_vx * _s + nb_abs_vy * _c)

            neighbor_info = np.array([
                rel_x, rel_y, rel_velocity, rel_heading,
                lane_encoded, is_ai, nb_dist_to_junction,
                nb_abs_vx, nb_abs_vy, nb_ttc
            ], dtype=np.float32)

            return neighbor_info

        except:
            return np.zeros(10, dtype=np.float32)
    
    def _get_kappa_hop_neighbors(self, adjacency_matrix: np.ndarray, 
                               vehicle_index: int) -> List[int]:
        """Get κ-hop neighbors from adjacency matrix."""
        if adjacency_matrix.size == 0:
            return []
            
        n = adjacency_matrix.shape[0]
        if vehicle_index >= n:
            return []
            
        # Compute κ-hop connectivity using matrix powers
        current_reach = adjacency_matrix.copy()
        total_reach = adjacency_matrix.copy()
        
        for hop in range(1, self.kappa):
            current_reach = np.dot(current_reach, adjacency_matrix)
            total_reach += current_reach
        
        # Get neighbors within κ hops
        neighbors = []
        for i in range(n):
            if i != vehicle_index and total_reach[vehicle_index, i] > 0:
                neighbors.append(i)
                
        return neighbors
    
    def _encode_lane_id(self, lane_id: str) -> float:
        """E1: permanent memo on the return value — a pure function of lane_id on a
        static net (called for ego AND every neighbor slot, every step). None is a
        valid dict key; the uncached body never raises Exception (catch-all 0.5), and
        non-Exception raises (KeyboardInterrupt) propagate uncached as before."""
        if not self._perf:
            return self._encode_lane_id_uncached(lane_id)
        hit = self._lane_enc_cache.get(lane_id)
        if hit is None:
            hit = self._encode_lane_id_uncached(lane_id)
            self._lane_enc_cache[lane_id] = hit
        return hit

    def _encode_lane_id_uncached(self, lane_id: str) -> float:
        """Encode lane ID as normalized float with direction AND lane index.

        Encoding scheme combines direction (4 values) and lane index (up to 4 lanes)
        to create a unique value for each lane in the network.

        Direction base values: N=0.0, S=0.25, E=0.5, W=0.75
        Lane offset: +0.0 to +0.2 based on lane index (normalized by max 4 lanes)

        This allows the policy to distinguish between lanes on the same edge,
        which is critical for proper lane discipline in multi-lane scenarios.
        """
        try:
            # Get direction component
            if 'north' in lane_id:
                direction_base = 0.0
            elif 'south' in lane_id:
                direction_base = 0.25
            elif 'east' in lane_id:
                direction_base = 0.5
            elif 'west' in lane_id:
                direction_base = 0.75
            elif ':center' in lane_id or 'center' in lane_id:
                # Inside junction - encode as special value
                return 0.9
            else:
                direction_base = 0.5  # Unknown direction

            # Extract lane index from lane_id (format: "edge_id_laneIndex")
            # e.g., "north_to_center_0" -> lane index 0
            lane_index = 0
            if '_' in lane_id:
                try:
                    lane_index = int(lane_id.split('_')[-1])
                except ValueError:
                    lane_index = 0

            # Normalize lane index using actual lane count from SUMO
            try:
                edge_id = traci.lane.getEdgeID(lane_id)
                num_lanes = traci.edge.getLaneNumber(edge_id)
            except Exception:
                num_lanes = 4
            lane_offset = (lane_index / max(num_lanes, 1)) * 0.2

            return direction_base + lane_offset

        except Exception:
            return 0.5  # Default fallback
    
    def _encode_vehicle_type(self, veh_type: str) -> float:
        """Encode vehicle type as float."""
        # NOTE: SUMO creates type variants like "ai_vehicle@veh_id" when
        # vehicle-specific attributes are set. Use startswith() to handle this.
        # Order matters: ai_human_vehicle starts with 'ai' but is background
        # SUMO-controlled traffic (used in high-AI-penetration scenarios),
        # so encode it as human-like, NOT as AI.
        if veh_type.startswith('ai_human_'):
            return 0.5
        elif veh_type.startswith('ai_vehicle'):
            return 0.0
        elif veh_type.startswith('human_vehicle') or veh_type.startswith('human_driver'):
            # P1.12: previously only 'human_driver' was matched, but the actual
            # vType in .rou.xml is 'human_vehicle'. Result: every human was
            # encoded as 0.25 (unknown), so the policy could not distinguish
            # humans from unknown vehicles in its observation. The 'human_driver'
            # alias is kept for legacy compatibility.
            return 0.5
        elif veh_type.startswith('truck'):
            return 1.0
        else:
            return 0.25
    
    def _get_route_direction(self, vehicle_id: str) -> float:
        """Get route direction at intersection."""
        try:
            route = self._veh_route(vehicle_id)              # E3: per-epoch memo
            route_index = self._veh_route_index(vehicle_id)

            if route_index < len(route) - 1:
                current_edge = route[route_index]
                next_edge = route[route_index + 1]
                
                # Determine turn direction based on edge names
                if ('north' in current_edge and 'south' in next_edge) or \
                   ('south' in current_edge and 'north' in next_edge) or \
                   ('east' in current_edge and 'west' in next_edge) or \
                   ('west' in current_edge and 'east' in next_edge):
                    return 0.0  # Straight
                elif ('north' in current_edge and 'east' in next_edge) or \
                     ('east' in current_edge and 'south' in next_edge) or \
                     ('south' in current_edge and 'west' in next_edge) or \
                     ('west' in current_edge and 'north' in next_edge):
                    return 1.0  # Right turn
                else:
                    return -1.0  # Left turn
            
            return 0.0  # Default straight
            
        except:
            return 0.0
    
    def _get_time_to_destination(self, vehicle_id: str) -> float:
        """Get estimated time to destination."""
        try:
            route = self._veh_route(vehicle_id)              # E3: per-epoch memo
            route_index = self._veh_route_index(vehicle_id)
            speed = max(traci.vehicle.getSpeed(vehicle_id), 1.0)

            # Estimate remaining distance
            remaining_distance = 0.0
            for i in range(route_index, len(route)):
                edge_length = self._edge_length(route[i])    # E4: static per-net memo
                remaining_distance += edge_length
            
            # Normalize time estimate
            time_estimate = remaining_distance / speed / 100.0  # Normalized
            return min(time_estimate, 1.0)
            
        except:
            return 0.5  # Default value
    
    def _get_traffic_light_state(self, vehicle_id: str) -> tuple:
        """Get traffic light state for vehicle's approach."""
        try:
            # Get controlled links for intersection
            tl_ids = traci.trafficlight.getIDList()
            if not tl_ids:
                return 0.5, 0.0  # No traffic light
                
            tl_id = tl_ids[0]  # Assume single intersection
            
            # Get current phase
            current_phase = traci.trafficlight.getPhase(tl_id)
            try:
                # Try new API first
                program = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            except AttributeError:
                # Fall back to deprecated API
                program = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
            
            # Phase encoding: 0=red, 0.5=yellow, 1=green
            phase_state = program.phases[current_phase].state

            # Map vehicle's lane to correct signal index
            # The phase state string length depends on lane count:
            #   2-lane: 20 chars (5 per approach), indices 0/5/10/15
            #   3-lane: 24 chars (6 per approach), indices 0/6/12/18
            #   N-lane: 4*(2N+1) chars, (2N+1) per approach
            # Auto-detect from phase_state length to support any lane count.

            vehicle_lane = traci.vehicle.getLaneID(vehicle_id)

            # Determine signal index based on approach direction
            signals_per_approach = len(phase_state) // 4 if len(phase_state) >= 4 else 5
            signal_index = 0
            if 'north_to_center' in vehicle_lane:
                signal_index = 0 * signals_per_approach
            elif 'east_to_center' in vehicle_lane:
                signal_index = 1 * signals_per_approach
            elif 'south_to_center' in vehicle_lane:
                signal_index = 2 * signals_per_approach
            elif 'west_to_center' in vehicle_lane:
                signal_index = 3 * signals_per_approach
            elif ':center' in vehicle_lane:
                # Vehicle is inside intersection - treat as green
                return 1.0, 0.0
            elif 'center_to_' in vehicle_lane:
                # Vehicle is exiting intersection - treat as green
                return 1.0, 0.0
            else:
                # Unknown lane - default to first signal
                signal_index = 0

            # Get signal for this approach
            if phase_state and len(phase_state) > signal_index:
                signal = phase_state[signal_index].lower()
                if signal == 'g':
                    phase_encoded = 1.0
                elif signal == 'y':
                    phase_encoded = 0.5
                else:
                    phase_encoded = 0.0
            else:
                phase_encoded = 0.0

            # Time remaining in phase
            next_switch = traci.trafficlight.getNextSwitch(tl_id)
            current_time = traci.simulation.getTime()
            time_remaining = (next_switch - current_time) / 60.0  # Normalized by 60s

            return phase_encoded, min(time_remaining, 1.0)
            
        except:
            return 0.5, 0.0
    
    def _get_agent_identity(self, vehicle_id: str) -> np.ndarray:
        """One-hot approach direction for agent identity.

        Gives the shared policy network an unambiguous signal about which
        direction this vehicle is approaching from. Without this, stopped
        vehicles from all directions look the same to the network, causing
        correlated collapse (all stopped vehicles get the same action).

        Returns: [from_north, from_east, from_south, from_west] — one-hot (4 dims)

        Reference: Christianos et al. 2020, "Shared Experience Actor-Critic"
        """
        identity = np.zeros(4, dtype=np.float32)
        if self._identity_heading:
            # GEOMETRY-BASED (any net): the compass direction the vehicle is coming FROM,
            # from its heading (angle: 0=N,90=E clockwise). A vehicle heading south (180)
            # approaches from the north. Binned to 4 sectors -> [from_N, from_E, from_S, from_W],
            # matching the legacy convention but working on OSM/grid edges too.
            try:
                angle = self._cached(vehicle_id, 'angle')
                if angle is None:
                    angle = traci.vehicle.getAngle(vehicle_id)
                from_dir = (float(angle) + 180.0) % 360.0
                sector = int(((from_dir + 45.0) % 360.0) // 90.0)  # 0=N,1=E,2=S,3=W
                identity[sector] = 1.0
            except Exception:
                pass
            return identity
        try:
            # Use the FIRST edge of the route (origin direction).
            # This is stable regardless of where the vehicle currently is
            # (approach road, junction, or exit road).
            route = self._veh_route(vehicle_id)              # E3: per-epoch memo
            if route:
                origin_edge = route[0]  # e.g., "north_to_center"
                if 'north' in origin_edge:
                    identity[0] = 1.0
                elif 'east' in origin_edge:
                    identity[1] = 1.0
                elif 'south' in origin_edge:
                    identity[2] = 1.0
                elif 'west' in origin_edge:
                    identity[3] = 1.0
        except Exception:
            pass
        return identity

    def _get_collision_warning(self, vehicle_id: str) -> np.ndarray:
        """Time-to-collision-based collision warning.

        P0.2: REPLACED distance-only thresholds (which were miscalibrated for
        variable speeds — at 13 m/s, the 20m threshold gave warning=0.1 even
        though the brake distance is ~19m and the AI should be at full
        warning) with TTC-based thresholds that adapt to closing speed.

        TTC = leader_distance / max(closing_speed, ε)
        where closing_speed = max(ego_speed - leader_speed, 0.1)

        Returns: [warning_level] — single float in [0, 1]:
            1.0 = TTC < 1.0 s (brake NOW)
            0.7 = TTC < 2.0 s (heavy brake)
            0.4 = TTC < 3.0 s (moderate brake)
            0.1 = TTC < 5.0 s (awareness)
            0.0 = TTC >= 5.0 s or leader moving away or no leader

        Real-world AEB / ACC systems use TTC because it correctly captures
        urgency at variable speeds. A leader at 20m is dangerous at 13 m/s
        (TTC ≈ 1.5s) but safe at 5 m/s (TTC ≈ 4s); distance-only thresholds
        cannot distinguish these.
        """
        warning = np.zeros(1, dtype=np.float32)
        try:
            leader = self._veh_leader(vehicle_id)            # E2: per-epoch memo
            if leader is not None:
                leader_id, distance = leader
                if distance < 0:
                    warning[0] = 1.0                    # already overlapping (rear-end)
                else:
                    ego_speed = float(self._cached(vehicle_id, 'speed') or 0.0)
                    leader_speed = float(self._cached(leader_id, 'speed') or 0.0)
                    closing_speed = max(ego_speed - leader_speed, 0.1)  # avoid /0
                    ttc = distance / closing_speed
                    if ttc < 1.0:   warning[0] = 1.0
                    elif ttc < 2.0: warning[0] = 0.7
                    elif ttc < 3.0: warning[0] = 0.4
                    elif ttc < 5.0: warning[0] = 0.1
                    # else: safe, stays 0.0
        except Exception:
            pass
        # CROSSING-AWARE (DMPO_CONFLICT_OBS): getLeader() only ever sees a same-lane
        # leader, so this scalar was structurally 0 for crossing/junction conflicts.
        # Combine it with the crossing-conflict urgency so the policy gets ONE salient
        # "brake" cue for both rear-end AND crossing. Gated → legacy obs when off.
        if self._conflict_obs:
            warning[0] = max(float(warning[0]),
                             float(self._crossing_conflict(vehicle_id)['urgency']))
        return warning

    def _ctx_totals(self):
        """E6: (n_in_junction, n_ai_total) — ego-independent, previously recomputed by a
        full-cache scan per ego. Counted once per epoch with the verbatim legacy loop."""
        memo = self._step_memo()
        if memo['ctx'] is None:
            n_in_junction = 0
            n_ai_total = 0
            if self._vehicle_cache:
                for vid, state in self._vehicle_cache.items():
                    if state.get('type_id', '').startswith('ai_vehicle'):
                        n_ai_total += 1
                        lane = state.get('lane_id', '')
                        if lane and lane.startswith(':'):
                            n_in_junction += 1
            memo['ctx'] = (n_in_junction, n_ai_total)
        return memo['ctx']

    def _lane_positions(self):
        """E6b: {lane_id: [(vid, lane_position), ...]} per epoch, for the same-lane
        queue-ahead count (strict > comparison preserved at the call site)."""
        memo = self._step_memo()
        lp = memo['lanepos']
        if lp is None:
            lp = {}
            for vid, state in (self._vehicle_cache or {}).items():
                lp.setdefault(state.get('lane_id', ''), []).append(
                    (vid, state.get('lane_position', 0)))
            memo['lanepos'] = lp
        return lp

    def _tls_snapshot(self):
        """E7: per-epoch snapshot of the first TL's RYG state + a FIRST-MATCH
        lane->index map (setdefault preserves list.index semantics — controlled lanes
        can repeat). Returns () when there is no TL. Exceptions propagate uncached so
        the caller's bare except sets 0.5 exactly as before."""
        memo = self._step_memo()
        tls = memo['tls']
        if tls is None:
            tl_ids = traci.trafficlight.getIDList()
            if tl_ids:
                tl_id = tl_ids[0]
                state_str = traci.trafficlight.getRedYellowGreenState(tl_id)
                controlled = traci.trafficlight.getControlledLanes(tl_id)
                lane_idx = {}
                for k, ln in enumerate(controlled):
                    lane_idx.setdefault(ln, k)
                tls = (state_str, lane_idx)
            else:
                tls = ()
            memo['tls'] = tls
        return tls

    def _fcfs_candidates(self, ego_jid, obs_stopped):
        """E8: per-epoch candidate prefilter for the FCFS/PET context scan
        (_fix_conflict branch ONLY; the legacy 'to_center' branch is not bucketable).
        Returns [(vid, state)]: vehicles whose approach edge's next junction is ego_jid,
        plus — when DMPO_OBS_STOPPED_CONFLICT — ALL internal-edge vehicles: the loop
        sets _onj=None for internal edges, so its stopped-conflict clause admits them
        WITHOUT any junction check and they must NOT be junction-bucketed. Every vehicle
        skipped here provably fails the loop's _other_conflict predicate (empty edge, or
        _onj None/mismatched junction, or internal without obs_stopped); the caller
        re-applies the full predicate, so this is a pure prefilter and the accumulators
        (min-PET, counts, sticky yield flag) are order-independent anyway."""
        memo = self._step_memo()
        fc = memo['fcfs']
        if fc is None:
            self._ensure_nj_maps()
            by_j = {}
            internal = []
            for vid, state in (self._vehicle_cache or {}).items():
                edge = state.get('edge_id', '')
                if not edge:
                    continue
                if edge.startswith(':'):
                    internal.append((vid, state))
                    continue
                rec = self._edge_nextjxn.get(edge)
                if rec is not None:
                    by_j.setdefault(rec[0], []).append((vid, state))
            fc = (by_j, internal)
            memo['fcfs'] = fc
        by_j, internal = fc
        cands = list(by_j.get(ego_jid, ())) if ego_jid is not None else []
        if obs_stopped:
            cands += internal
        return cands

    def _get_context_features(self, vehicle_id: str, all_vehicles: Set[str]) -> np.ndarray:
        """Compute derived context features from cached vehicle data.

        These pre-compute decision-relevant information that a small encoder
        would struggle to learn from raw features alone.

        Returns: [10] array of normalized context features
        """
        context = np.zeros(10, dtype=np.float32)

        try:
            ego_speed = self._cached(vehicle_id, 'speed') or 0.0
            ego_pos = self._cached(vehicle_id, 'position') or (0, 0)
            ego_lane = self._cached(vehicle_id, 'lane_id') or ''
            ego_lane_pos = self._cached(vehicle_id, 'lane_position') or 0.0

            # Center reference: ego's NEXT junction (any net) under DMPO_NEXT_JUNCTION_OBS,
            # else the legacy single 'center'. _ego_jid identifies it so conflicts can be
            # grouped by "same junction" without edge-name parsing.
            _ego_jid = None
            if self._fix_conflict:
                _ejf = self._next_junction_full(vehicle_id)
                if _ejf is not None:
                    _ego_jid, cx, cy = _ejf
                else:
                    cx, cy = self.intersection_center or (0, 0)
            else:
                cx, cy = self.intersection_center or (0, 0)
            dist_to_center = math.sqrt((ego_pos[0] - cx)**2 + (ego_pos[1] - cy)**2)

            # 1. Time to intersection (normalized by 30s)
            if ego_speed > 0.5:
                tti = dist_to_center / ego_speed
                context[0] = min(tti / 30.0, 1.0)
            else:
                context[0] = 1.0  # Stopped → "far" in time

            # 2. Intersection occupancy (vehicles in junction / max)
            if self._perf:
                n_in_junction, n_ai_total = self._ctx_totals()   # E6: once per epoch
            else:
                n_in_junction = 0
                n_ai_total = 0
                if self._vehicle_cache:
                    for vid, state in self._vehicle_cache.items():
                        if state.get('type_id', '').startswith('ai_vehicle'):
                            n_ai_total += 1
                            lane = state.get('lane_id', '')
                            if lane and lane.startswith(':'):
                                n_in_junction += 1
            max_agents = max(n_ai_total, 1)
            context[1] = min(n_in_junction / max_agents, 1.0)

            # 3. Queue position (same-lane vehicles ahead / max)
            n_ahead = 0
            if self._vehicle_cache:
                if self._perf:
                    # E6b: shared per-epoch lane buckets; identical strict-> count.
                    for _vid2, _lp2 in self._lane_positions().get(ego_lane, ()):
                        if _vid2 == vehicle_id:
                            continue
                        if _lp2 > ego_lane_pos:
                            n_ahead += 1
                else:
                    for vid, state in self._vehicle_cache.items():
                        if vid == vehicle_id:
                            continue
                        if state.get('lane_id', '') == ego_lane:
                            if state.get('lane_position', 0) > ego_lane_pos:
                                n_ahead += 1
            context[2] = min(n_ahead / max(max_agents, 1), 1.0)

            # 4. Right of way (approach-specific traffic light)
            try:
                if self._perf:
                    # E7: one TL snapshot per epoch instead of 3 TraCI calls + an O(L)
                    # list scan per ego; first-match map == controlled.index semantics.
                    _tls = self._tls_snapshot()
                    if _tls and ego_lane:
                        _state_str, _lane_idx = _tls
                        _tidx = _lane_idx.get(ego_lane)
                        if _tidx is not None and _tidx < len(_state_str):
                            sig = _state_str[_tidx].lower()
                            context[3] = 1.0 if sig == 'g' else (0.5 if sig == 'y' else 0.0)
                else:
                    tl_ids = traci.trafficlight.getIDList()
                    if tl_ids and ego_lane:
                        tl_id = tl_ids[0]
                        state_str = traci.trafficlight.getRedYellowGreenState(tl_id)
                        controlled = traci.trafficlight.getControlledLanes(tl_id)
                        if ego_lane in controlled:
                            idx = controlled.index(ego_lane)
                            if idx < len(state_str):
                                sig = state_str[idx].lower()
                                context[3] = 1.0 if sig == 'g' else (0.5 if sig == 'y' else 0.0)
            except:
                context[3] = 0.5

            # 5. Nearest conflict PET (normalized by 10s)
            # FCFS RIGHT-OF-WAY (Approach B, gated by DMPO_FCFS_PRIORITY=1):
            # while scanning crossing-conflict vehicles for PET, ALSO rank ego by
            # arrival time (time-to-junction) vs each conflicting vehicle. This is a
            # rule-based first-come-first-served priority — the explicit ordering the
            # policy currently has to infer. Reuses the same conflict definition
            # (vehicle on a DIFFERENT to_center edge = crossing path).
            best_pet = 10.0  # Default: no conflict
            fcfs_on = os.environ.get('DMPO_FCFS_PRIORITY', '0') == '1'
            n_conflicts = 0
            n_ego_first = 0   # # of conflicts ego reaches the junction BEFORE (ego has priority)
            yield_now = 0.0   # an earlier-arriving (higher-priority) conflict is imminent → yield
            # F3 (round-4, DMPO_OBS_STOPPED_CONFLICT=1): compute conflict/FCFS even when
            # STOPPED and when INSIDE the junction. Default-off gate preserves old
            # checkpoints' obs distribution. Fixes the freeze-blindness root cause:
            # a stopped/junction vehicle could not perceive the crossing conflict.
            _obs_stopped = os.environ.get('DMPO_OBS_STOPPED_CONFLICT', '0') == '1'
            _speed_ok = (ego_speed > 0.5) or _obs_stopped
            ego_edge = self._cached(vehicle_id, 'edge_id') or ''
            if self._fix_conflict:
                # any-net: ego is "on approach" if we resolved its next junction
                _on_approach = (_ego_jid is not None) and (not ego_edge.startswith(':'))
                _in_junction = ego_edge.startswith(':') if _obs_stopped else False
            else:
                _on_approach = 'to_center' in ego_edge
                _in_junction = ego_edge.startswith(':') if _obs_stopped else False
            if _speed_ok and self._vehicle_cache and (_on_approach or _in_junction):
                if True:
                    # realistic arrival-time floor: a stopped ego is NOT infinitely far
                    # in time — estimate as if it could accelerate (min 3 m/s), and an
                    # in-junction ego is already there (dist≈0).
                    _ego_v_eff = ego_speed if ego_speed > 3.0 else 3.0
                    ego_time = (0.0 if _in_junction else dist_to_center) / _ego_v_eff
                    # E8: under _fix_conflict, iterate the per-epoch junction prefilter
                    # instead of the full cache (pure prefilter — the body re-applies
                    # the complete _other_conflict predicate). Legacy 'to_center'
                    # branch and kill-switch keep the full scan.
                    if self._perf and self._fix_conflict:
                        _fcfs_iter = self._fcfs_candidates(_ego_jid, _obs_stopped)
                    else:
                        _fcfs_iter = self._vehicle_cache.items()
                    for vid, state in _fcfs_iter:
                        if vid == vehicle_id:
                            continue
                        other_edge = state.get('edge_id', '')
                        if self._fix_conflict:
                            # crossing conflict = heading to the SAME junction from a DIFFERENT
                            # approach edge (any net; no edge-name parsing).
                            _onj = self._edge_nextjxn.get(other_edge) if other_edge and not other_edge.startswith(':') else None
                            _other_conflict = ((_onj is not None and _ego_jid is not None
                                                and _onj[0] == _ego_jid and other_edge != ego_edge)
                                               or (_obs_stopped and other_edge.startswith(':')
                                                   and other_edge != ego_edge))
                        else:
                            _other_conflict = (('to_center' in other_edge and other_edge != ego_edge)
                                               or (_obs_stopped and other_edge.startswith(':')
                                                   and other_edge != ego_edge))
                        if other_edge and _other_conflict:
                            other_speed = state.get('speed', 0)
                            if other_speed > 0.5 or (_obs_stopped and state.get('edge_id','').startswith(':')):
                                other_pos = state['position']
                                other_dist = math.sqrt((other_pos[0]-cx)**2 + (other_pos[1]-cy)**2)
                                other_time = other_dist / max(other_speed, 0.1)
                                pet = abs(ego_time - other_time)
                                best_pet = min(best_pet, pet)
                                n_conflicts += 1
                                if ego_time <= other_time:
                                    n_ego_first += 1        # ego arrives first → ego has priority
                                elif other_dist < 80.0:
                                    yield_now = 1.0         # an earlier-arriving conflict is within range → ego should yield
            context[4] = min(best_pet / 10.0, 1.0)
            # Approach B: overwrite the (inert, no-traffic-light) right-of-way slot
            # context[3] with the FCFS priority fraction. Obs shape is UNCHANGED.
            # 1.0 = ego arrives before all conflicts (full priority); 0.0 = yields to all.
            if fcfs_on:
                context[3] = (n_ego_first / n_conflicts) if n_conflicts > 0 else 1.0

            # ── CROSSING-CONFLICT OVERRIDE (DMPO_CONFLICT_OBS). Replace the timing-coincidence
            # ctx_pet (context[4]) with PET at the TRUE conflict point; and (Arm A only) put the
            # whose-turn Time-Advantage into context[3]. Arm B leaves context[3]=0 so the
            # cooperating κ-hop agents must LEARN the yield convention from the team reward.
            if self._conflict_obs:
                cc = self._crossing_conflict(vehicle_id)
                context[4] = min(cc['pet'] / 10.0, 1.0)              # conflict-point PET (0..1)
                if self._explicit_priority:
                    context[3] = (float(np.clip(0.5 - cc['ta'] / 4.0, 0.0, 1.0))
                                  if cc['n'] > 0 else 1.0)           # >0.5 go, <0.5 yield, 1.0 clear
                else:
                    context[3] = 0.0                                 # Arm B: no explicit priority

            # 6. Traffic density (active agents / max_agents cap)
            context[5] = min(n_ai_total / max(self.max_neighbors, 8), 1.0)

            # 7. Is in queue (binary: leader < 30m and slow)
            try:
                leader_info = self._veh_leader(vehicle_id)   # E2: per-epoch memo
                if leader_info is not None:
                    leader_id, distance = leader_info
                    if distance < 30.0:
                        leader_speed = self._cached(leader_id, 'speed') or 0
                        if leader_speed is not None and leader_speed < 2.0:
                            context[6] = 1.0
            except:
                pass

            # 8. Intersection clear (binary: no vehicles in junction)
            context[7] = 1.0 if n_in_junction == 0 else 0.0

            # 9. Ego has priority (green AND intersection clear) — OR, under FCFS
            # (Approach B), repurpose this inert slot as a CONFLICT-ACTIVE flag:
            # 1.0 = ego currently has >=1 crossing conflict. This disambiguates
            # context[3]=1.0 (no conflict → free to go) from context[3]=1.0 with an
            # active conflict ego out-ranks (go assertively), and pairs with a low
            # context[3] (lower priority + conflict active → yield).
            if fcfs_on:
                context[8] = 1.0 if n_conflicts > 0 else 0.0
            else:
                context[8] = 1.0 if (context[3] >= 0.9 and n_in_junction == 0) else 0.0

            # 10. Time in current state (steps at same speed range / 100)
            speed_key = f"{vehicle_id}_speed_state"
            current_state = 'stopped' if ego_speed < 1.0 else 'moving'
            prev_state_info = getattr(self, '_speed_state_tracking', {}).get(speed_key, (current_state, 0))
            if prev_state_info[0] == current_state:
                steps = prev_state_info[1] + 1
            else:
                steps = 1
            if not hasattr(self, '_speed_state_tracking'):
                self._speed_state_tracking = {}
            self._speed_state_tracking[speed_key] = (current_state, steps)
            context[9] = min(steps / 100.0, 1.0)

        except Exception as e:
            pass  # Return zeros on error

        return context

    def _get_zero_observation(self) -> Dict[str, np.ndarray]:
        """Return zero observation when vehicle not found."""
        if self._v2:                                     # OBS-V2: neutral, never zeros (R2)
            return {k: v.copy() for k, v in self._zero_obs_neutral().items()}
        z = {
            'ego': np.zeros(10, dtype=np.float32),
            'infrastructure': np.zeros(5, dtype=np.float32),
            'neighbors': np.zeros((self.max_neighbors, 10), dtype=np.float32),
            'neighbor_mask': np.zeros(self.max_neighbors, dtype=np.float32),
            'context': np.zeros(10, dtype=np.float32),
            'identity': np.zeros(4, dtype=np.float32),
            'collision_warning': np.zeros(1, dtype=np.float32)
        }
        if self._path_conflict_obs:
            z['conflict_gap'] = np.zeros(self.PATH_CONFLICT_BLOCK, dtype=np.float32)
        return z