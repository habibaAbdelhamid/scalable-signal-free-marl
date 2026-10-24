#!/bin/bash
# ============================================================================
# TRAINING BLOCK — reward / curriculum / learner env vars.
# These MAY legitimately differ between training and evaluation (they do not
# change a FIXED policy's actions) — but they are still pinned HERE, never
# hand-typed, so every run is reproducible.
#
# NOTE (C4): the default section below is the LEGACY reward (completion 400 >
# collision 200 — the inverted exchange rate of the aborted launcher; the TRUE
# historical r4/r5/r6 regime was completion 200 / collision 400, ratio 2.0).
# Kept verbatim for legacy reproduction. The Stage-3 REWARD V2 profile is at
# the BOTTOM of this file, gated on DMPO_REWARD_V2=1. Do NOT retune ad hoc.
# ============================================================================

export DMPO_TRAINING_BLOCK_VERSION="1.0-stage0"

# ---- reward (legacy CARL+ profile) ----
export DMPO_REWARD_CARL_PLUS=1
export DMPO_COMPLETION_BONUS=400          # dual-default hazard — always explicit
export DMPO_COLLISION_PENALTY=200         # dual-default hazard — always explicit
export DMPO_MOVEMENT_BONUS=0.1
export DMPO_OVERLAP_PENALTY=-0.05
export DMPO_TEAM_BONUS=5.0
export DMPO_TEAM_COST=5.0
export DMPO_ROUTE_LENGTH_M=400            # dual-default hazard (380 code / 400 launcher)
export DMPO_TTC_SHAPING_K=0.0
export DMPO_CROSS_NEARMISS_K=0.0
export DMPO_GAP_REWARD_K=0                # gap-acceptance shaping OFF for first runs (decision)

# ---- policy init (training forensic 2026-07-25: was pinned NOWHERE — Run 1
# without it froze toward comp=0/stopped-rising; Run 2 with 0.3 learned
# comp 0.5→6.8 and reward −690→+907 in 80 iters. The documented anti-freeze
# init bias; re-applied on curriculum reinit.) ----
export DMPO_POLICY_BIAS_LONGITUDINAL=0.3
export DMPO_POLICY_BIAS_YIELD_INTENT=0.5
# ANNEAL the cold-start prior to EXACTLY zero well before convergence, so the
# converged/evaluated policy provably contains NO hand-set prior (publication
# guarantee; same anneal-to-zero rule we apply to the shield scaffold).
# Verified empirically 2026-07-25 (anneal_iters=5): 0.300→0.240→0.180→0.120→
# 0.060→0.000, then PINNED at 0.000 for the rest of training. 300 iters is a
# small fraction of a Stage-4 run (thousands) and far pre-convergence.
export DMPO_POLICY_BIAS_ANNEAL_ITERS=300

# ---- learner (H12: entropy EXPLICIT — train:301 silently defaults to 0.05!) ----
export DMPO_ENTROPY=0.0
export DMPO_MODEL_PROB=0                  # model-free ablation (honest naming)
# REMINDER: when MODEL_PROB=0 the
# launcher MUST also pass:  --n-model-update 0 --n-model-update-warmup 0
export DMPO_HUBER_VALUE=1
export DMPO_HUBER_DELTA=1.0
export DMPO_LOG_STD_MAX=0.5
export DMPO_LOG_STD_MIN=-2.0
export DMPO_N_UPDATE_PI=4
export DMPO_MODEL_BUFFER_SIZE=4
export DMPO_N_WARMUP=50
# ---- world-model loss (Category B, train-only: changes what the world model
# LEARNS, never a fixed policy's actions — hence pinned HERE, not in the
# consistency block; also the assert_consistency "known train var" whitelist /
# deploy EXTRA_OK_DIFF class, like DMPO_SAVE_DENSE). 1 = identity-matched loss
# for the nearest-first neighbour block (+ nearest-first re-sort of rollout
# states); requires DMPO_OBS_V2=1, DMPO_DYNAMIC=1, DMPO_SHARED_GCN=0 (hard
# errors otherwise) and only matters when world-model training runs
# (MODEL_PROB>0 or TRAIN_GATE=0). Sub-toggles pinned at their code defaults so
# assert_consistency's known-var check accepts them (an ablation arm overrides
# them explicitly): DMPO_MODEL_MATCHED_KEEP_EMPTY=1 ("empty stays empty" slots
# keep weight 1 → neutral row), DMPO_MODEL_MATCHED_DEBUG=1 (sparse slot-match
# census print). Model-based arms flip the main flag to 1 EXPLICITLY in their
# launch command.
export DMPO_MODEL_MATCHED_LOSS=0
export DMPO_MODEL_MATCHED_KEEP_EMPTY=1
export DMPO_MODEL_MATCHED_DEBUG=1

# ---- overload guard (F1, live-flow audit 2026-07-25): with GRIDLOCK_DETECT
# unset, the cap-overflow path silently RELEASES excess AI to SUMO/Krauss
# control every step — mixed-autonomy contamination of the all-AI rig. Harmless
# at curriculum scale 0.2 (peak 13/40) but plausible at scale 1.0. Pin the gate
# so overload TERMINATES the episode (a failure) instead of silently degrading. ----
export DMPO_GRIDLOCK_DETECT=1

# ---- curriculum (M17: mode EXPLICIT — code default is legacy 'cap') ----
export DMPO_CURRICULUM=1
export DMPO_CURRICULUM_MODE=scale
export DMPO_CURRICULUM_SCHEDULE="0:0.2,400:0.4,1000:0.6,1700:0.8,2300:1.0"
# ANTI-SATURATION CURRICULUM FLOOR (forensic 2026-07-25): at start-scale 0.2 the
# advantage landscape is almost pure GO-earns (≈10 completions vs ≈1.5 collisions
# per iter) — PPO saturates the head into constant-GO by iter 300, BEFORE the
# curriculum ever delivers collision pressure (a curriculum ORDERING trap on top
# of the representation trap). Under the v2 rig, start at collision-relevant
# density and reach eval density (0.6) early.
if [ "${DMPO_OBS_V2:-0}" = "1" ]; then
  export DMPO_CURRICULUM_SCHEDULE="0:0.4,300:0.6,900:0.8,1500:1.0"
fi
export DMPO_TRAFFIC_SCALE=1.0             # base; scale-curriculum overrides per stage

# ---- logging ----
export DMPO_QUIET_RESET=1

# ============================================================================
# REWARD V2 (Stage-3 rebalance — spec s3 + adversarial-verification corrections).
# GATED: activates ONLY when DMPO_REWARD_V2=1 is exported BEFORE sourcing this
# block. Sourcing with DMPO_REWARD_V2 unset/0 changes NOTHING (legacy profiles
# stay byte-identical). Values recomputed against the TRUE historical regime
# (completion 200 / collision 400, ratio 2.0 — r4/r5/r6 run logs), NOT the
# spec's wrong +400/−200 baseline.
# NOT re-exported here (Category-A, pinned in consistency_block.sh):
#   DMPO_COLLISION_REMOVE_AI  (=1 there; a second export would make the winner
#   depend on source order — the exact hand-typed-block failure mode).
# ============================================================================
if [ "${DMPO_REWARD_V2:-0}" = "1" ]; then
  export DMPO_TRAINING_BLOCK_VERSION="2.0-stage3-rewardv2"
  export DMPO_REWARD_PROFILE_VERSION="2.0"

  # ---- event terms (§1): KEEP HISTORICAL MAGNITUDES (user decision 2026-07-25) ----
  # No sudden magnitude changes: every empirical argument for a different ratio
  # (in EITHER direction — "raise it, the floor was tolerated" / "cap it, R6-A
  # gridlocked") comes from checkpoints trained with the severed graph + broken
  # obs + straight-ray geometry, so it is EVIDENCE-FREE for the fixed rig.
  # Logic gives only: ratio must stay >= 2 (assert), and the STRUCTURAL fixes
  # below (motion<progress, no yield-fine, no cliff) lower the charge/yield
  # break-even at FIXED penalty — so the penalty stays put and structure does
  # the work. Ratio is an explicit knob for a future ablation, not a default.
  export DMPO_COMPLETION_BONUS=200        # TRUE historical value
  export DMPO_COLLISION_PENALTY=400       # TRUE historical value (ratio 2.0)
  export DMPO_REWARD_EXCHANGE_ASSERT=1    # money guard — TRAINING entry only (never in env/eval)
  export DMPO_REWARD_MIN_RATIO=2.0        # historical ratio passes with equality

  # ---- dense terms (§4): shared ramp kills the 0.5 m/s cliff ----
  export DMPO_MOVEMENT_BONUS=0.04         # was 0.1 (> max progress 0.0834/step — inverted!); 0.04 ≈ half
  export DMPO_MOVEMENT_RAMP=1             # 0 = legacy threshold branch verbatim
  export DMPO_MOVEMENT_RAMP_LO=0.3
  export DMPO_MOVEMENT_RAMP_HI=0.7
  # DMPO_MOVEMENT_SPEED_THRESH is DEAD under RAMP=1 (only the legacy branch reads it)

  # ---- cooperation (§6c): KEEP HISTORICAL (user decision 2026-07-25) ----
  # The κ-graph was SEVERED in every historical run — there is ZERO valid
  # evidence about team-term magnitudes in either direction, so there is no
  # basis for changing them. First valid data arrives from the fixed-graph rig.
  export DMPO_TEAM_BONUS=5.0              # historical
  export DMPO_TEAM_COST=5.0               # historical

  # ---- progress (§6a): trapezoid odometer (removes anti-braking bias) ----
  export DMPO_PROGRESS_TRAPEZOID=1
  export DMPO_ROUTE_LENGTH_M=400          # keep explicit (code default is 380)

  # ---- idle / yield exemptions (§5) — ACTIVE with the obs-v2 batch ----
  # Logic-forced (never fine yielding-for-a-genuine-conflict). The PET exemption
  # consumes _crossing_conflict, which the obs-v2 batch keeps live under
  # DMPO_OBS_V2=1 (curve geometry always on); the shield/stuck exemptions read
  # the env-published _shield_held/_stuck_ai sets (obs-v2 env wiring). All are
  # functional no-ops under the legacy profiles (flag gated + geometry dark).
  export DMPO_IDLE_YIELD_EXEMPT=1
  export DMPO_IDLE_EXEMPT_PET_S=1.5
  case "${DMPO_BLOCK_PROFILE:-}" in
    obs_v2_scaffold*) export DMPO_IDLE_EXEMPT_SHIELD=1 ;;  # scaffold arms incl. _sel
    *)               export DMPO_IDLE_EXEMPT_SHIELD=0 ;;
  esac
  export DMPO_IDLE_EXEMPT_STUCK=1
  export DMPO_IDLE_EXEMPT_MAX_STEPS=50    # anti-gridlock cap

  # ---- DEFERRED (env/obs files frozen this stage; obs-v2 batch) ----
  #   DMPO_REWARD_KAPPA_AVG (§6b, needs env._compute_rewards), fault attribution
  #   (§2, needs env+obs), terminal progress credit (§9.2, needs env).
  # ---- SKIPPED (verifier): §3 event-window dedup (self-contradicting + dead
  #   under REMOVE_AI=1); anything on DMPO_GAP_SHIELD_OCC_WINDOW (pinned OFF,
  #   empirically net-negative).
fi

# ============================================================================
# LEARNER V2 (Stage-3 learner fixes — spec s3_learner-fixes + adversarial-
# verification corrections: C6 dynamic masks / H4b adjacency fiction / H5
# model-gate + KL early stop).
# GATED: activates ONLY when DMPO_LEARNER_V2=1 is exported BEFORE sourcing this
# block. Sourcing with DMPO_LEARNER_V2 unset/0 changes NOTHING — every flag
# below defaults OFF in code, so legacy training semantics stay byte-identical.
# All flags are TRAINING-side only (they do not change a fixed policy's actions
# at eval) — Category B, pinned here, never hand-typed.
# ============================================================================
if [ "${DMPO_LEARNER_V2:-0}" = "1" ]; then
  export DMPO_LEARNER_BLOCK_VERSION="2.0-stage3-learnerfix"

  # ---- C6: unmasked padding in dynamic mode (DPPO.py) ----
  export DMPO_DYNAMIC_MASK=1        # mask right-padded timesteps in PPO/GAE/value losses
  export DMPO_ADV_NORM_SCOPE=traj   # masked adv-norm axis; 'batch' = whole-batch z-score (ablation arm only)
  export DMPO_MASK_DEBUG=1          # 🎭 per-update mask stats + leak asserts (proof C6 fired)

  # ---- H5b/H5c(ii): KL-gated early stop (epoch loop + trainer inner loop) ----
  export DMPO_KL_EARLY_STOP=1       # freeze policy epochs at |KL| > 1.5*target_kl (value keeps training)

  # ---- H5a: skip world-model training when the model is never used ----
  export DMPO_MODEL_TRAIN_GATE=1    # model_prob=0 & no schedule → no model grad steps (≈94% wall-clock waste)
  # (supersedes the manual --n-model-update 0 --n-model-update-warmup 0 reminder above;
  #  keep passing those anyway — the gate makes forgetting them harmless)

  # ---- H5c(i): collapse the pointless model-free inner loop (40→4 epochs) ----
  export DMPO_INNER_COLLAPSE=1      # n_inner_iter→1 when model-free; DMPO_N_INNER_ITER=N overrides explicitly

  # ---- H4(b): adjacency-fiction removal (GATED on dynamic mode in code) ----
  export DMPO_ADJ_FIX=1             # dynamic: identity adj (fiction never consumed); static reinit: guarded
                                    # κ-cache invalidation; attention overflow rows extend with ZEROS not ones

  # ---- κ-hop advantage aggregation (restored; provenance-confirmed 2026-07-25:
  # the pristine original aggregates UNCONDITIONALLY — the dynamic bypass had no
  # commit history = dropped by mistake). Default OFF here; the κ-adv FORENSIC ARM
  # exports DMPO_KAPPA_ADV=1 (requires --n-episodes 1; DMPO_GRAPH_FRESH comes from
  # the consistency block). ADV_NORM arm overrides DMPO_ADV_NORM_SCOPE=batch. ----
  export DMPO_KAPPA_ADV=0

  # ---- Wave-2b pins (verify-workflow wf_86e4706c findings, 2026-07-26) ----
  # SEED_CYCLE: ARM-A data-diversity lever (per-hard-start SUMO seed rotation).
  # Pinned OFF here so it is block-recorded; the A/B launchers override to 1
  # EXPLICITLY in their launch command (visible in ps/logs). Eval engines
  # force-set it to 0 (eval pins one seed by definition).
  export DMPO_SEED_CYCLE=0
  # SKIP_HARD_RESET: skip-loop recovery threshold. The code default was an
  # implicit '8' — every launch silently had it. Pinned explicitly (same value,
  # zero behavior change for the running fleet) so the block records it.
  export DMPO_SKIP_HARD_RESET=8
  # GAE lambda — pinned at the original 0.5; the ARM-E credit-routing ablation
  # overrides to 0.9 explicitly in its launch command (E1/E5).
  export DMPO_GAE_LAMBDA=0.5
  # WAVE-4 comfort term — pinned OFF; the wave-4 arms export K=0.1 explicitly.
  export DMPO_COMFORT_DECEL_K=0
  export DMPO_COMFORT_DECEL_THRESH=3.0

  # ---- KL-anomaly fix (root cause found 2026-07-25 by the C6 masked estimator) ----
  # The buffer stored the ±1-CLIPPED env action while logp described the UNCLIPPED
  # sample → PPO ratios corrupted for every clipped draw (measured |KL|=0.348 at
  # epoch 0 ≈ 46× target; std e^0.5≈1.65 vs a ±1 box clips a large fraction).
  # Fix stores the RAW sample (env still gets the clipped action).
  export DMPO_STORE_RAW_ACTION=1
fi
