"""Frozen seed discipline (Stage 0).

PRIMARY EVALUATION PROTOCOL (user decision 2026-07-28, registered BEFORE any
wave-3 checkpoint was selected): the POISSON-arrival scenario
(4lane_allAI_unsig_poisson, scale 0.6, 700 steps, cap 40) is the primary
endpoint for selection, fresh-seed confirmation, and the one-shot TEST spend.
The uniform/metronome scenario (4lane_allAI_unsig) is retained as the
synchronized-arrival STRESS TEST (secondary; reported in the appendix).
Rationale: Poisson is the standard traffic-engineering arrival model for an
isolated unsignalized intersection; the metronome timetable manufactures
worst-case synchronized ties (documented mechanism study).

TRAIN: training realizations.
VAL:   checkpoint selection ONLY (all checkpoints, per the hard rule).
TEST:  report-only, touched ONCE per experiment via run_frozen_test.sh —
       any eval engine must REFUSE a TEST seed unless DMPO_FROZEN_TEST=1.

These frozensets are the mechanical enforcement of "checkpoint selection on
validation, reporting on held-out test" — the fix for the past invalidity of
selecting checkpoints on the same seeds used to report results.

DO NOT EDIT the TEST list (extend by APPENDING new seeds only; never reuse).
"""
import os

TRAIN_SEEDS = frozenset({23423})

# Wave-2b seed-cycling arms (A1/B/C) rotate the SUMO seed per hard restart:
# base + restart index. Their TRAIN ranges are the bases' neighborhoods —
# disjoint from VAL/TEST/FRESH by construction (23423+k and 61000+k stay far
# from the 1xxx / 9xxx / 7xxx families for any realistic k).
TRAIN_SEED_BASES = frozenset({23423, 61000, 51423, 151423})
# 51423 = κ-family / learning-baseline seed 1; 151423 = learning-baseline seed 2 (2026-10-11).
# Seed-cycle ranges ≈ base..base+30k — disjoint from each other and
# from VAL/TEST/FRESH.

VAL_SEEDS = frozenset({1101, 1207, 1303, 1409, 1511})

TEST_SEEDS = frozenset({9001, 9013, 9027, 9031, 9043, 9057, 9061, 9077, 9083, 9091})

# PRE-REGISTERED confirmation seeds (wave-2b selection ladder, registered
# 2026-07-26 BEFORE any wave-2b checkpoint was evaluated on them): checkpoints
# are SCREENED on VAL (cheap, reused, winner's-curse-biased), then the
# shortlist (within 2 total crashes of the screen minimum) is CONFIRMED on
# these 12 seeds, and the final selection statistic is the FRESH-seed mean —
# never the VAL screen value. TEST stays report-only, spent once.
# DO NOT EDIT once any wave-2b checkpoint has touched them.
# NETWORK CONFIRMATORY tier (pre-registered 2026-08-16, before any cell was
# run): 10 held-out grid realizations for the n>=10 expansion of the
# 25-junction matrix and saturation frontier. Paired across
# ALL controllers. Disjoint from every other tier. Never reuse; append only.
NETWORK_CONFIRM_SEEDS = frozenset({3011, 3023, 3037, 3041, 3053, 3067, 3079, 3083, 3097, 3109})
# batch 2 (pre-registered 2026-08-16, before any cell ran): raises network cells to n=20, paired.
NETWORK_CONFIRM_SEEDS_B = frozenset({3121, 3137, 3163, 3181, 3203, 3221, 3251, 3271, 3301, 3319})
# PROTOCOL-FACTORIAL tier (pre-registered 2026-08-16): 20 held-out single-junction
# realizations for the same-checkpoint accounting x shield factorial. Disjoint from all tiers.
PROTOCOL_FACTORIAL_SEEDS = frozenset({2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081,
                                      2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141})
# FRONTIER tier (pre-specified 2026-08-17, before any cell ran): 20 held-out
# single-junction realizations for the n=20 saturation-frontier sweep (90-200% demand,
# 1,000-step horizon, cap 100), paired across ALL controllers. Disjoint from every tier.
FRONTIER_CONFIRM_SEEDS = frozenset({4003, 4019, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093,
                                    4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177})
FRESH_CONFIRM_SEEDS = frozenset({7103, 7121, 7139, 7157, 7211, 7229,
                                 7247, 7265, 7321, 7339, 7357, 7411})
# FRESH tier of the operating benchmark (4lane_allAI_unsig_poisson_lanefix), pre-registered
# 2026-08-08 before any run on these seeds.
LANEFIX_FRESH_SEEDS = frozenset({4102, 4211, 4303, 4409, 4517, 4622, 4731, 4808, 4913, 5021,
                                 5137, 5248})
# RARE-EVENT tier (pre-registered 2026-10-16, before any cell ran): the first
# 100 primes above 5000, single-junction, report-only (never used for selection). The 5xxx
# family is unused by every other tier and by the training seed-cycle ranges.
# Correction 2026-10-17: seed 5021 also appears in the lanefix fresh-tier list
# (LANEFIX_FRESH_SEEDS; added to this file 2026-10-29), so 99 of the 100
# seeds are new; 5021 is kept (the set is report-only) and the overlap is disclosed.
# SENSING-ROBUSTNESS tier (pre-registered 2026-10-17, before any cell ran):
# the first 20 primes above 6000 (6007..6163), report-only.
SENSING_SEEDS = frozenset(
    n for n in range(6001, 6170) if all(n % d for d in range(2, int(n ** 0.5) + 1)))
# LANE-COUNT tier (pre-registered 2026-10-17, before any cell ran): the
# first 20 primes above 6500 (6521..6689), report-only. The 6xxx family is unused elsewhere.
LANE_COUNT_SEEDS = frozenset(
    n for n in range(6501, 6690) if all(n % d for d in range(2, int(n ** 0.5) + 1)))
RARE_EVENT_SEEDS = frozenset(
    n for n in range(5001, 6000) if all(n % d for d in range(2, int(n ** 0.5) + 1))
    and n <= 5851)


class FrozenTestSeedError(SystemExit):
    pass


def guard_seed(seed) -> None:
    """Refuse TEST seeds outside the frozen-test harness. Call from every eval engine."""
    try:
        s = int(seed)
    except (TypeError, ValueError):
        return
    if s in TEST_SEEDS and os.environ.get('DMPO_FROZEN_TEST', '0') != '1':
        raise FrozenTestSeedError(
            f'REFUSED: seed {s} is a FROZEN TEST seed. Test seeds are report-only and '
            f'must be run via run_frozen_test.sh (DMPO_FROZEN_TEST=1) AFTER checkpoint '
            f'selection on VAL seeds. Use VAL seeds {sorted(VAL_SEEDS)} for development.')
