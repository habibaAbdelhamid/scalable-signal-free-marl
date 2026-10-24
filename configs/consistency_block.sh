#!/bin/bash
# ============================================================================
# CANONICAL CONSISTENCY BLOCK — Category-A env vars (train and eval MUST match).
# THE single source of truth. NEVER hand-type these in a launcher again
# (root cause C7 of the aborted 4-arm run).
#
# Usage:   export DMPO_BLOCK_PROFILE=legacy_r5_bare   # or legacy_r5_shielded
#          source configs/consistency_block.sh
# Then run scripts/assert_consistency.py before ANY train or eval — launchers
# that set DMPO_REQUIRE_BLOCK=1 make the engines refuse to start without it.
#
# Profiles:
#   legacy_r5_bare      — the validated faithful rig for ALL existing (R4/R5/R6/
#                         E-2200/s44-era) checkpoints, shield OFF (bare policy).
#   legacy_r5_shielded  — same rig + the rule-controller (ex-shield) engaged;
#                         used ONLY for the rule-controller baseline and legacy
#                         shielded re-scoring.
#   obs_v2_bare         — Stage-3 checkpoint-breaking rig (DMPO_OBS_V2=1, 241-dim
#                         obs, no z-filter), shield OFF. PRIMARY Stage-4 profile.
#   obs_v2_scaffold     — same + the annealed training scaffold (gap shield) ON.
#                         Scaffold arms ONLY; primary eval stays BARE.
# ============================================================================

export DMPO_BLOCK_PROFILE="${DMPO_BLOCK_PROFILE:-legacy_r5_bare}"
case "$DMPO_BLOCK_PROFILE" in
  obs_v2_*) export DMPO_BLOCK_VERSION="2.0-obsv2" ;;
  *)        export DMPO_BLOCK_VERSION="1.1-stage1" ;;
esac

# ---- C1 graph fix (Stage 1): action-neutral at eval (dynamic-mode policy never
# consumes the graph), restores the cooperation graph for training/team terms ----
export DMPO_CENTER_REDETECT=1
export DMPO_TOPO_CONFLICT_EDGES=1

# ---- OBS-V2 umbrella (Stage 3) — Category A: obs_dim/semantics change entirely ----
case "$DMPO_BLOCK_PROFILE" in
  obs_v2_*) export DMPO_OBS_V2=1 ;;
  *)        export DMPO_OBS_V2=0 ;;
esac
export DMPO_OBS_ZFILTER=0             # v2 ablation escape hatch — pinned OFF (C3)
export DMPO_PROFILE_OBS=0             # ARM-B driving-profile obs — pinned OFF everywhere;
                                      # ONLY the obs_v2_profile branch below flips it to 1
export DMPO_CONFL_SAME_EDGE=0         # v2.1 same-road diverging-conflict visibility (F2 arm)
                                      # — pinned OFF; ONLY obs_v2_profile21 flips it to 1
export DMPO_SUMO_HARDRESET_AT=27000   # consistency-critical; was unpinned (train
                                      # code default 300000 vs eval setdefault 27000 silently
                                      # diverged with GREEN gates). One canonical value; the
                                      # cold-start training arms override to 1 EXPLICITLY in
                                      # their launch command (single-episode evals never reach
                                      # any threshold, so eval behavior is unchanged).
export DMPO_CONFLICT_HORIZON_M=200    # v2 conflict-scan physical reach
# M12-5B episode-state reset: pinned 0 for legacy (checkpoint-affecting: ctx[9]
# counters historically LEAKED across episodes for reused flow ids); 1 for v2
# profiles (checkpoint-breaking anyway — per-episode reset is the correct semantics).
case "$DMPO_BLOCK_PROFILE" in
  obs_v2_*) export DMPO_OBS_STATE_RESET=1 ;;
  *)        export DMPO_OBS_STATE_RESET=0 ;;
esac
export DMPO_PERF_FIXES=1              # M12 memoization (value-identical; pinned so train==eval wall-clock behavior)
export DMPO_OBS_SPATIAL_HASH=1        # M12-3 (value-identical)

# ---- Category A: architecture / observation (identical in all legacy profiles) ----
export DMPO_DYNAMIC=1
case "$DMPO_BLOCK_PROFILE" in
  *_padded) export DMPO_DYNAMIC=0 ;;  # padded-ablation profiles: the SOLE architectural delta
esac
export DMPO_SHARED_POLICY=0
export DMPO_ACTION_VARIANT=yield_intent
export DMPO_TANH_POLICY=1
export DMPO_MAX_NEIGHBORS=15
export DMPO_PERCEPTION_RADIUS=150.0
export DMPO_KAPPA=2
export DMPO_NEXT_JUNCTION_OBS=1
export DMPO_CONFLICT_OBS=0            # disproven POC (r6) — pinned OFF
export DMPO_FCFS_PRIORITY=1           # legacy ckpts trained with it — obs slot ctx[3]/[8]
export DMPO_OBS_STOPPED_CONFLICT=1    # legacy ckpts trained with it — obs slot ctx[4]
export DMPO_PATH_CONFLICT_OBS=0       # v1 block not in legacy ckpts (obs_dim 180)
export DMPO_OBS_REAR_BUMPER=0
export DMPO_OBS_PREDICT_AHEAD_S=0.0
export DMPO_IDENTITY_HEADING=0        # NJ=1 implies heading identity anyway; pinned for clarity

# ---- Category A: action / physics ----
export DMPO_MAX_SPEED_MPS=16.67
export DMPO_MAX_DECEL=8               # NOT in the eval anchor (H13) — this block IS the anchor
export DMPO_MAX_LAT_VELOCITY=1.0
export DMPO_KRAUSS_CAP=0
export DMPO_SPEED_MODE=6
export DMPO_STEP_LENGTH=0.2
export DMPO_NO_LATERAL=1              # funnel fix — train AND eval
# LANE-CHANGE CONTROL CONDITION (2026-08-16): the
# *_nolc profile additionally switches SUMO's OWN lane-change model off
# (setLaneChangeMode 0) so the vehicle is under exactly one controller. Eval-only
# sensitivity profile; NO_LATERAL above already disables the POLICY's lateral
# channel in every profile. Consistency-critical (changes a fixed policy's
# outcomes) → lives here, never hand-typed.
if [ "$DMPO_BLOCK_PROFILE" = "obs_v2_profile_nolc" ]; then export DMPO_SET_LC_MODE=0; fi
export DMPO_RESIDUAL=1                # legacy action pipeline (R4/R5-era ckpts)
export DMPO_RESIDUAL_BASE=0.0,0.0,0.0,0.0
export DMPO_RESIDUAL_SCALE=1.0,0.3,0.3,0.6
export DMPO_KRAUSS_THIN_FLOOR_M=5

# ---- Category A: collision handling ----
export DMPO_CLEAR_STOPPED_STEPS=0     # audit F10: read site exists — pin the OFF default explicitly
export DMPO_COLLISION_ACTION=warn
export DMPO_COLLISION_CHECK_JUNCTIONS=true
export DMPO_COLLISION_STOPTIME=10
export DMPO_COLLISION_HALT_AI=0
export DMPO_COLLISION_REMOVE_AI=1
export DMPO_COLLISION_STUCK_MAX_STEPS=25
export DMPO_COLLISION_STUCK_REMOVE_ON_TIMEOUT=1
export DMPO_TIME_TO_TELEPORT=-1

# ---- Category A: rule-controller (ex-shield) — profile-dependent ----
# NOTE (verifier correction, 2026-07-25): the shield's semantics are governed ONLY
# by the DMPO_GAP_SHIELD_* vars below — DMPO_OBS_V2 never reconfigures them (the v2
# conflict block computes its own two-sided windows with v2-private code).
if [ "$DMPO_BLOCK_PROFILE" = "obs_v2_scaffold_sel" ]; then
  # SELECTIVE-TEACHER scaffold (Stage-4 lead arm; round-4 conditioning verdict):
  # the generic approach-zone speed cap taught "never accelerate" (uniform speed-
  # holding, mean≈0 everywhere — outcome-improving but plateaus at col≈5 and is
  # NOT the target skill). This profile teaches ONLY the target behavior: full
  # speed everywhere, hard PET-gated hold for a GENUINE conflict.
  export DMPO_RESIDUAL=0
  export DMPO_FORCE_YIELD_OFF=1
  export DMPO_GRAPH_FRESH=1
  export DMPO_GAP_SHIELD=1
  export DMPO_CURVE_CONFLICT=1
  export DMPO_GAP_SHIELD_CONFLICT_DIST=2.5
  export DMPO_GAP_SHIELD_FOLLOW=1
  export DMPO_GAP_SHIELD_APPROACH_ZONE_M=0     # NO generic slow-approach teaching
  export DMPO_GAP_SHIELD_APPROACH_MPS=6.0
  export DMPO_GAP_SHIELD_GENUINE_PET_S=1.5
  export DMPO_GAP_SHIELD_PET_S=4.0
  export DMPO_GAP_SHIELD_MARGIN_M=2.0
  export DMPO_GAP_SHIELD_CLEAR_M=0
  export DMPO_GAP_SHIELD_OCC_WINDOW=0
elif [ "$DMPO_BLOCK_PROFILE" = "obs_v2_scaffold" ]; then
  export DMPO_RESIDUAL=0   # fresh v2 arms train on the plain tanh head (see obs_v2_bare note)
  export DMPO_FORCE_YIELD_OFF=1  # longitudinal-only for real (see obs_v2_bare note)
  export DMPO_GRAPH_FRESH=1      # action-time graph (see obs_v2_bare note)
  # Stage-4 scaffold arms: the ANNEALED training scaffold = the validated curve-
  # aware gap shield (occ-window OFF — measured net-negative; Part-2 result).
  export DMPO_GAP_SHIELD=1
  export DMPO_CURVE_CONFLICT=1
  export DMPO_GAP_SHIELD_CONFLICT_DIST=2.5
  export DMPO_GAP_SHIELD_FOLLOW=1
  export DMPO_GAP_SHIELD_APPROACH_ZONE_M=30
  export DMPO_GAP_SHIELD_APPROACH_MPS=6
  export DMPO_GAP_SHIELD_GENUINE_PET_S=1.5
  export DMPO_GAP_SHIELD_PET_S=4.0
  export DMPO_GAP_SHIELD_MARGIN_M=2.0
  export DMPO_GAP_SHIELD_CLEAR_M=0
  export DMPO_GAP_SHIELD_OCC_WINDOW=0
elif [ "$DMPO_BLOCK_PROFILE" = "obs_v2_bare" ] || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_profile" ] \
  || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_profile_nolc" ] \
  || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_bare_cold" ] \
  || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_shielded" ] \
  || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_shielded_cold" ] \
  || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_shielded_follow" ] \
  || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_profile_padded" ] \
  || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_profile21" ] \
  || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_profile21_bare" ]; then
  # ARM-B (obs_v2_profile, user design 2026-07-26): obs_v2_bare + endogenous
  # driving-profile distinguishability (DMPO_PROFILE_OBS=1 → obs_dim 241→261:
  # +4 ego lifetime stats, +4 raw-delta dims per conflict slot). Pure
  # information — no priority rule; the tie-break convention must be LEARNED.
  # Checkpoint-breaking vs 241-dim checkpoints (load gate refuses mismatches).
  if [ "$DMPO_BLOCK_PROFILE" = "obs_v2_profile" ] || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_profile_nolc" ] || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_profile_padded" ] || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_profile21" ] \
    || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_profile21_bare" ]; then
    export DMPO_PROFILE_OBS=1
    if [ "$DMPO_BLOCK_PROFILE" = "obs_v2_profile21" ] || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_profile21_bare" ]; then
      export DMPO_CONFL_SAME_EDGE=1
    fi
  else
    export DMPO_PROFILE_OBS=0
  fi
  # Cold-start-every-episode IS the wave-2b training regime for BOTH cold
  # profiles (obs_v2_profile = +driving-profile obs; obs_v2_bare_cold = the
  # diversity-alone A2 CONTROL, 241-dim — same regime, no profile features,
  # the ablation that attributes B's effect to the profile). Block-recorded,
  # not hand-exported. Inert at single-episode eval (the FIRST reset
  # hard-starts; no later reset occurs within 700 steps), so train/eval stay
  # consistent under the same profile by construction.
  if [ "$DMPO_BLOCK_PROFILE" = "obs_v2_profile" ] || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_bare_cold" ] \
    || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_profile_padded" ] || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_profile_nolc" ] \
    || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_shielded_cold" ] || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_profile21" ] \
    || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_profile21_bare" ]; then
    export DMPO_SUMO_HARDRESET_AT=1
  fi
  # H11/4.7 ACTION DECISION (user, 2026-07-25): KEEP the 4-dim head
  # (ACTION_VARIANT=yield_intent above) with lateral frozen (NO_LATERAL=1) and
  # yield_intent inert — zero plumbing churn. RESIDUAL OFF for FRESH v2 arms:
  # the base+scale*tanh transform existed to serve legacy checkpoints; a fresh
  # policy trains on the plain tanh head (one less train/eval mirror surface).
  export DMPO_RESIDUAL=0
  # KAPPA-ADV package 1a (provenance-confirmed restoration): rebuild the comm
  # graph from the FRESH cache each step, so adjacency/kappa_reach describe the
  # action-time state (newly spawned vehicles were isolated nodes before).
  # Consistency-critical: the graph feeds reward averaging + team terms, which
  # eval also computes.
  export DMPO_GRAPH_FRESH=1
  # LONGITUDINAL-ONLY, FOR REAL (publication-audit finding, 2026-07-25): under
  # ACTION_VARIANT=yield_intent, dim 3 > 0 silently switches a per-vehicle
  # rule-based Krauss following cap ON (action_translator:146-148 ->
  # env:1939 `if krauss_cap_enabled or yield_active:`) even with KRAUSS_CAP=0 —
  # and the +0.5 init bias on dim 3 is never annealed, so every vehicle started
  # with the rule cap engaged (measured: dim-3 bias +0.5006->+0.4974 across 200
  # iters). That contradicts both the longitudinal-only design and the BARE
  # attribution claim. Pin the kill-switch: dims 1-3 are now truly inert.
  export DMPO_FORCE_YIELD_OFF=1
  # PRIMARY Stage-4 profile: bare policy, no shield. The obs conflict map still
  # uses CONFLICT_DIST (same source as the shield would — one conflict map, never
  # a private fork); CURVE_CONFLICT is forced on inside the processor under v2.
  export DMPO_GAP_SHIELD=0
  # TIE-HYPOTHESIS POSITIVE CONTROL (user PoC, 2026-07-27): *_shielded profiles
  # switch the curve-aware gap shield (rule-based conflict-point reservation =
  # the rule-based tie-breaker) ON, same params as below otherwise. DIAGNOSTIC
  # profiles only — never the paper headline (shield = teacher/baseline per
  # standing decision). If shield-on ≈ 0 crashes with throughput intact, the
  # residual IS tie-class failures; if crashes persist, the tie diagnosis is
  # incomplete and the survivors name what else is wrong.
  if [ "$DMPO_BLOCK_PROFILE" = "obs_v2_shielded" ] || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_shielded_cold" ] \
    || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_shielded_follow" ] \
  || [ "$DMPO_BLOCK_PROFILE" = "obs_v2_profile21" ]; then
    # NOTE (wave-5, 2026-08-01): obs_v2_profile21 keeps its historical shield-ON
    # record (f2 trained under it); the PURE-RL variant is obs_v2_profile21_bare,
    # which is deliberately ABSENT from this list (GAP_SHIELD stays 0).
    export DMPO_GAP_SHIELD=1
  fi
  export DMPO_CURVE_CONFLICT=1
  export DMPO_GAP_SHIELD_CONFLICT_DIST=2.5
  if [ "$DMPO_BLOCK_PROFILE" = "obs_v2_shielded_follow" ]; then
    # diagnostic ceiling variant: crossing rule + FOLLOW cap (rear-end coverage)
    export DMPO_GAP_SHIELD_FOLLOW=1
  else
    export DMPO_GAP_SHIELD_FOLLOW=0
  fi
  export DMPO_GAP_SHIELD_APPROACH_ZONE_M=0
  export DMPO_GAP_SHIELD_APPROACH_MPS=6.0
  export DMPO_GAP_SHIELD_GENUINE_PET_S=0
  export DMPO_GAP_SHIELD_PET_S=3.0
  export DMPO_GAP_SHIELD_MARGIN_M=2.0
  export DMPO_GAP_SHIELD_CLEAR_M=0
  export DMPO_GAP_SHIELD_OCC_WINDOW=0
elif [ "$DMPO_BLOCK_PROFILE" = "legacy_r5_shielded" ]; then
  export DMPO_GAP_SHIELD=1
  export DMPO_CURVE_CONFLICT=1
  export DMPO_GAP_SHIELD_CONFLICT_DIST=2.5
  export DMPO_GAP_SHIELD_FOLLOW=1
  export DMPO_GAP_SHIELD_APPROACH_ZONE_M=30
  export DMPO_GAP_SHIELD_APPROACH_MPS=6
  export DMPO_GAP_SHIELD_GENUINE_PET_S=1.5
  export DMPO_GAP_SHIELD_PET_S=4.0
  export DMPO_GAP_SHIELD_MARGIN_M=2.0
  export DMPO_GAP_SHIELD_CLEAR_M=0
  export DMPO_GAP_SHIELD_OCC_WINDOW=0
else  # legacy_r5_bare (default)
  export DMPO_GAP_SHIELD=0
  export DMPO_CURVE_CONFLICT=0
  export DMPO_GAP_SHIELD_CONFLICT_DIST=0.0
  export DMPO_GAP_SHIELD_FOLLOW=0
  export DMPO_GAP_SHIELD_APPROACH_ZONE_M=0
  export DMPO_GAP_SHIELD_APPROACH_MPS=6.0
  export DMPO_GAP_SHIELD_GENUINE_PET_S=0
  export DMPO_GAP_SHIELD_PET_S=3.0
  export DMPO_GAP_SHIELD_MARGIN_M=2.0
  export DMPO_GAP_SHIELD_CLEAR_M=0
  export DMPO_GAP_SHIELD_OCC_WINDOW=0
fi
