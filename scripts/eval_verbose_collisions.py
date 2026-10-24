"""Verbose collision eval — dumps EVERY collision (no AI filter, no dedup) per step.

Goal: compare against what the GUI shows so we can reconcile mismatch between
the metric ('Collision events: 7') and visual count.

Reports four counters per step:
  - all_collisions_raw         = len(traci.simulation.getCollisions()) (no dedup, all pairs)
  - all_collision_pairs_new    = distinct pairs seen for first time, any vehicle types
  - ai_collision_pairs_new     = subset of above where collider OR victim is AI
  - human_human_pairs_new      = subset of above where BOTH are humans

Writes a per-event JSON with timestamp + IDs so you can scrub the GUI and check.

Usage:
  python3 scripts/eval_verbose_collisions.py \
      --checkpoint checkpoints/variantG_yield_remove_s42_g2/model_weights_2000.pt \
      --max-steps 7500 --out /tmp/collisions_verbose.json --gui
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Deterministic eval — see evaluate_phaseT_gui.py for rationale.
if os.environ.get('PYTHONHASHSEED') != '0':
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import torch

os.environ['DMPO_USE_TRACI'] = '1'
# Verify-workflow HIGH (wf_86e4706c): eval pins ONE traffic seed by definition —
# the ARM-A training-side seed-rotation lever must never fire here, and the gate
# is blind to inherited exports, so force it like USE_TRACI above.
os.environ['DMPO_SEED_CYCLE'] = '0'
# Same hazard class (verified bug 2026-07-31): a scenario-rotation export leaked
# from a sourced training shell would silently swap the evaluated scenario while
# every artifact records the requested one — force it off like SEED_CYCLE.
os.environ['DMPO_SCENARIO_CYCLE'] = ''
# Verify-workflow HIGH: the stochastic diagnostic switch must never touch the
# one-shot frozen TEST, and every artifact must record which mode produced it.
if (os.environ.get('DMPO_FROZEN_TEST', '0') == '1'
        and os.environ.get('DMPO_EVAL_STOCHASTIC', '0') == '1'):
    sys.exit('DMPO_EVAL_STOCHASTIC=1 is FORBIDDEN under DMPO_FROZEN_TEST=1: '
             'the frozen protocol is deterministic-mean only.')
os.environ.setdefault('DMPO_REWARD_CARL_PLUS', '1')
os.environ.setdefault('DMPO_DYNAMIC', '1')
os.environ.setdefault('DMPO_TANH_POLICY', '1')
os.environ.setdefault('DMPO_KRAUSS_CAP', '0')
os.environ.setdefault('DMPO_COLLISION_ACTION', 'warn')
os.environ.setdefault('DMPO_COLLISION_CHECK_JUNCTIONS', 'true')
os.environ.setdefault('DMPO_TIME_TO_TELEPORT', '-1')
os.environ.setdefault('DMPO_COLLISION_REMOVE_AI', '0')
os.environ.setdefault('DMPO_COLLISION_HALT_AI', '0')
os.environ.setdefault('DMPO_COLLISION_STUCK_RELEASE_GAP_M', '5.0')
os.environ.setdefault('DMPO_COLLISION_STUCK_MAX_STEPS', '5')
os.environ.setdefault('DMPO_ACTION_VARIANT', 'yield_intent')
os.environ.setdefault('DMPO_COMPLETION_BONUS', '200')
os.environ.setdefault('DMPO_TEAM_BONUS', '5.0')
os.environ.setdefault('DMPO_OVERLAP_PENALTY', '-0.05')
os.environ.setdefault('DMPO_MOVEMENT_BONUS', '0.1')
# Anchor aligned to the TRAINING value (historical 400; live-flow audit + clean
# penalty A/B 2026-07-25: magnitude is behaviorally neutral — 200 and 400 both
# converge to ~1.2 col/iter — so reported-reward comparability decides, and
# training uses 400). Pre-alignment frozen_stage2 JSONs computed reward at the
# old 200 fallback; reward is not a frozen-table headline metric.
os.environ.setdefault('DMPO_COLLISION_PENALTY', '400')
os.environ.setdefault('DMPO_POLICY_BIAS_LONGITUDINAL', '0.3')
os.environ.setdefault('DMPO_HUBER_VALUE', '1')
os.environ.setdefault('DMPO_HUBER_DELTA', '1.0')
os.environ.setdefault('DMPO_MAX_LAT_DIST', '0.5')
os.environ.setdefault('DMPO_SUMO_HARDRESET_AT', '27000')
# SEED PLUMBING (soundness-review BLOCKER F7): remember whether the CALLER exported
# the SUMO seed — if not, --seed must become the traffic seed (previously --seed fed
# only numpy/torch while traffic silently ran the 23423 setdefault = TRAIN seed).
_SUMO_SEED_EXPORTED = 'DMPO_SUMO_SEED' in os.environ
os.environ.setdefault('DMPO_SUMO_SEED', '23423')

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, os.environ.get('DMPO_FRAMEWORK_PATH', str(Path(__file__).parent.parent / "third_party" / "Networked-MB-MARL")))

import traci  # noqa: E402
from dmpo_autonomous_driving.envs.sumo_intersection_env import create_intersection_env  # noqa: E402
from dmpo_autonomous_driving.envs.observation_processor import ObservationProcessor  # noqa: E402
from dmpo_autonomous_driving.config.intersection_dmpo import create_intersection_dmpo_config  # noqa: E402
from algorithms.algo.agent.DMPO import DMPOAgent  # noqa: E402

_OBS_V2 = os.environ.get('DMPO_OBS_V2', '0') == '1'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=False, default=None)
    p.add_argument('--baseline', choices=['none', 'sumo', 'rule'], default='none',
                   help="FROZEN-PROTOCOL baselines: 'sumo'=SUMO-default driving (DMPO_SUMO_BASELINE), "
                        "'rule'=constant-go prior + the profile's caps (run under legacy_r5_shielded "
                        "= the rule-controller/ex-shield baseline). 'none'=evaluate --checkpoint.")
    p.add_argument('--scenario',
                   default='configs/sumo_configs/4lane_allAI_unsig_poisson_lanefix/intersection.sumocfg')
    p.add_argument('--max-steps', type=int, default=7500)
    p.add_argument('--max-ai-vehicles', type=int, default=5)
    p.add_argument('--out', default='/tmp/collisions_verbose.json')
    p.add_argument('--gui', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--target', default='', help='vehicle to flash RED + its nearest neighbour')
    p.add_argument('--obs-dim', type=int, default=0, help='force policy obs_dim (V7=100: N=15 built then truncated to 100)')
    args = p.parse_args()

    # ── STAGE-0 PROCESS ARMOR ──
    # (a) frozen-test seed guard — ALWAYS on (mechanical VAL/TEST discipline);
    # (b) canonical-block gate — refuses to run on env drift when DMPO_REQUIRE_BLOCK=1.
    try:
        from dmpo_autonomous_driving.eval_seeds import guard_seed
        # F7 fix: --seed IS the traffic seed unless the caller explicitly exported
        # DMPO_SUMO_SEED; if both are set and disagree, refuse (silent train-seed
        # traffic labeled as a VAL/TEST seed was the exact hazard).
        if not _SUMO_SEED_EXPORTED:
            os.environ['DMPO_SUMO_SEED'] = str(args.seed)
        elif int(os.environ['DMPO_SUMO_SEED']) != int(args.seed):
            sys.exit(f"SEED MISMATCH: caller exported DMPO_SUMO_SEED="
                     f"{os.environ['DMPO_SUMO_SEED']} but --seed={args.seed} — refusing "
                     f"(traffic would run one seed while results are labeled the other).")
        guard_seed(os.environ['DMPO_SUMO_SEED'])
        guard_seed(args.seed)
    except ImportError:
        print('WARNING: eval_seeds not importable — frozen-seed guard SKIPPED')
    if os.environ.get('DMPO_REQUIRE_BLOCK', '0') == '1':
        import subprocess as _sp
        _gate = _sp.run([sys.executable,
                         os.path.join(os.path.dirname(__file__), 'assert_consistency.py'),
                         '--scenario', args.scenario])
        if _gate.returncode != 0:
            sys.exit('REFUSED by assert_consistency (see violations above)')

    # ── FROZEN-PROTOCOL baselines (Stage 2) ──
    if args.baseline == 'none' and not args.checkpoint:
        sys.exit('--checkpoint is required unless --baseline is sumo/rule')
    if args.baseline == 'sumo':
        os.environ['DMPO_SUMO_BASELINE'] = '1'
        print('🚦 BASELINE=sumo: SUMO-default driving (policy commands ignored)')
    elif args.baseline == 'rule':
        print('📏 BASELINE=rule: constant-go prior + profile caps (rule-controller / ex-shield)')

    print(f"Loading policy from {args.checkpoint}, GUI={args.gui}")
    env = create_intersection_env(
        config_path=args.scenario,
        max_steps=args.max_steps,
        use_gui=args.gui,
        max_ai_vehicles=args.max_ai_vehicles,
        training_mode=True,
    )

    config = create_intersection_dmpo_config(args.scenario)
    config.agent_args.n_agent = args.max_ai_vehicles
    config.agent_args.dynamic = True
    config.agent_args.entropy_coeff = 0.05
    config.agent_args.grad_clip = 0.5
    config.agent_args.adj = np.ones((args.max_ai_vehicles, args.max_ai_vehicles), dtype=np.float32)
    if args.obs_dim > 0:
        if _OBS_V2:
            sys.exit('--obs-dim is FORBIDDEN under DMPO_OBS_V2=1: the conflict block is '
                     'the LAST 31 dims, so truncation would amputate exactly the signal '
                     'the v2 batch exists to add (Step 15/R5).')
        config.agent_args.observation_dim = args.obs_dim   # V7: obs built at N=15 (180) then truncated to 100

    import logging
    class StubLogger:
        def __init__(self, name='DMPO'):
            self.logger = logging.getLogger(name); self.metrics = {}
        def child(self, name): return StubLogger(name)
        def log(self, **kwargs): self.metrics.update(kwargs)
        def __getattr__(self, name): return getattr(self.logger, name, lambda *a, **kw: None)

    device = torch.device('cpu')
    # ── PADDED-CHECKPOINT SUPPORT (audit 2026-08-08, finding 1) ──
    # Padded (dynamic=False) checkpoints carry n_agent per-agent actors whose
    # shapes depend on the saved spatial adjacency; reconstruct the same agent.
    _ck_pre = None; _padded_ckpt = False; _pad_n = None
    if args.baseline == 'none':
        _ck_pre = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        _arch = _ck_pre.get('arch') or {}
        if _arch.get('dynamic') is False:
            _padded_ckpt = True
            _pad_n = int(_arch.get('n_agent') or args.max_ai_vehicles)
            _adj = _ck_pre.get('adjacency')
            if _adj is None:
                sys.exit('PADDED checkpoint lacks saved adjacency — cannot rebuild '
                         'per-agent actor shapes. Retrain with the adjacency-saving '
                         'trainer (2026-08-08+).')
            config.agent_args.dynamic = False
            config.agent_args.n_agent = _pad_n
            config.agent_args.adj = np.asarray(_adj, dtype=np.float32)
            # κ from the checkpoint's own CLI (the trainer sets radius_pi/v/p = --kappa):
            # per-slot actor input width = κ-hop degree × obs_dim, so a κ=0 padded arm
            # (matched padded twin) must be rebuilt at κ=0, not the config default 2.
            _argv = _ck_pre.get('argv') or []
            if '--kappa' in _argv:
                _k = int(_argv[_argv.index('--kappa') + 1])
                config.agent_args.radius_pi = config.agent_args.radius_v = config.agent_args.radius_p = _k
                print(f"🧱 PADDED checkpoint trained at kappa={_k}")
            print(f"🧱 PADDED checkpoint: n_agent={_pad_n}, dynamic=False "
                  f"(fixed-slot policy — capacity wall applies)")
    agent = DMPOAgent(StubLogger(), device, config.agent_args, config)
    # SET-ACTOR (learning baselines): rebuild the trunk the checkpoint was trained with
    # BEFORE the tanh patch + strict load; architecture comes from the ckpt itself.
    if _ck_pre is not None:
        from dmpo_autonomous_driving.utils import set_actor as _sa
        _sam = _sa.apply_from_checkpoint(agent.actors, _ck_pre,
                                         int(config.agent_args.observation_dim), device)
        if _sam != 'off':
            print(f"🧩 SET-ACTOR eval: trunk '{_sam}' (from checkpoint)")

    import types
    # RESIDUAL mirror (train/eval consistency): must match _apply_output_tanh in
    # dmpo_full_training.py exactly — mean = base + scale·tanh(head) when DMPO_RESIDUAL=1.
    _residual_on = os.environ.get('DMPO_RESIDUAL', '0') == '1'
    if _residual_on:
        _rbase = torch.tensor([float(x) for x in os.environ.get(
            'DMPO_RESIDUAL_BASE', '0.5,0.0,0.0,0.4').split(',')], dtype=torch.float32)
        _rscale = torch.tensor([float(x) for x in os.environ.get(
            'DMPO_RESIDUAL_SCALE', '0.5,0.3,0.3,0.6').split(',')], dtype=torch.float32)
        print(f"🧩 RESIDUAL eval: mean = base + scale·tanh(head); base={_rbase.tolist()} scale={_rscale.tolist()}")
    for actor in agent.actors:
        if hasattr(actor, 'action_head') and hasattr(actor, 'log_std'):
            def _tanh_forward(self_actor, obs, a=None):
                output = self_actor.network(obs)
                if _residual_on:
                    mean = _rbase + _rscale * torch.tanh(self_actor.action_head(output))
                else:
                    mean = torch.tanh(self_actor.action_head(output))
                clamped = torch.clamp(self_actor.log_std,
                                     self_actor.LOG_STD_MIN, self_actor.LOG_STD_MAX)
                std = torch.exp(clamped).expand(*mean.shape)
                if a is None: return mean, std
                from torch.distributions import Normal
                return Normal(mean, std).log_prob(a).sum(dim=-1)
            actor.forward = types.MethodType(_tanh_forward, actor)

    if args.baseline == 'none':
        ck = _ck_pre if _ck_pre is not None else torch.load(args.checkpoint, map_location=device, weights_only=False)
        # Verify-workflow LOW: explicit metadata gate (the implicit size-mismatch
        # refusal still backstops old ckpts without metadata, but this fails FAST
        # and with the trainer's message — e.g. a 241-dim ckpt on a 261 profile run).
        _ck_dim = ck.get('obs_dim')
        _run_dim = int(config.agent_args.observation_dim)
        if _ck_dim is not None and int(_ck_dim) != _run_dim:
            sys.exit(f'OBS DIM MISMATCH: checkpoint obs_dim={_ck_dim} vs run '
                     f'obs_dim={_run_dim} — REFUSING (train and eval must agree; '
                     f'check DMPO_OBS_V2/DMPO_PROFILE_OBS/DMPO_MAX_NEIGHBORS).')
        agent.actors.load_state_dict(ck['policy_net_legacy'])
        # MAPPO baseline: the centralized critic (input obs+G) is training-only and is
        # never consulted at execution — do not load it into the obs-width critic.
        if str(ck.get('central_critic', '0')) != '1':
            agent.vs.load_state_dict(ck['value_net_legacy'])
        try:
            agent.ps.load_state_dict(ck['model_net_legacy'])
        except Exception:
            pass
        obs_mean = ck.get('obs_norm_mean'); obs_var = ck.get('obs_norm_var')
        # ENSEMBLE MODE (gated DMPO_ENSEMBLE_CKPTS="p1,p2,...", default unset =
        # no-op): action-mean over additional band checkpoints. Motivated by the
        # per-checkpoint event forensics (2026-08-01): the band's residual is
        # checkpoint-idiosyncratic subsets of a solvable catalog (no event fails
        # >64% of checkpoints), so averaging actions across members should cover
        # each member's blind spots. Diagnostic harness; primary protocol is
        # single-checkpoint.
        _ens_extra = []
        _ens = os.environ.get('DMPO_ENSEMBLE_CKPTS', '').strip()
        if _ens:
            import copy as _copy
            for _p in [_x for _x in _ens.split(',') if _x.strip()]:
                _ck2 = torch.load(_p.strip(), map_location=device, weights_only=False)
                if _ck2.get('obs_dim') is not None and int(_ck2['obs_dim']) != _run_dim:
                    sys.exit(f'ENSEMBLE OBS DIM MISMATCH: {_p} has obs_dim={_ck2.get("obs_dim")}')
                _ac2 = _copy.deepcopy(agent.actors)
                _ac2.load_state_dict(_ck2['policy_net_legacy'])
                _ens_extra.append(_ac2)
            print(f"🎭 ENSEMBLE MODE: {1+len(_ens_extra)} members (action-mean)")
    else:
        # baselines never consult the network; keep the pipeline shape-intact
        obs_mean = None; obs_var = None
        _ens_extra = []

    obs_dim = config.agent_args.observation_dim
    action_dim = config.agent_args.action_space.shape[0]

    # === verbose collision tracking ===
    all_events = []                  # every single (collider, victim) seen, no dedup
    pair_first_seen = {}             # pair -> step it was first seen
    ai_pair_count = 0
    human_pair_count = 0
    mixed_pair_count = 0
    raw_record_count = 0
    completed_trips = 0              # AI vehicles that arrived at destination

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    observations = env.reset()
    _prev_partner = [None]
    # ── PROBE (DMPO_PROBE=1): per-step per-vehicle signals for the failure-mode
    # differential diagnosis (P1 deadlock / P2 over-stop / P3 mis-yield / P4 risky-entry
    # / P5 piling / P7 straddle / P10 bang-bang). One pass captures everything; the
    # analyzer (scripts/analyze_probe.py) turns it + the collision JSON into the table.
    _probe_on = os.environ.get('DMPO_PROBE', '0') == '1'
    _probe_rows = []
    # ── FROZEN-METRICS instrumentation (Stage 2) ──
    _step_len = float(os.environ.get('DMPO_STEP_LENGTH', '0.2') or 0.2)
    _t0_sim = None; _t1_sim = None                 # REAL sim time (fixes hardcoded max_steps*0.2)
    _controlled_ever = set()                        # H8: every id the POLICY ever controlled
    _ai_seen = set()                                # all ai-prefixed ids observed (contamination denom)
    _collided_ids = set()                           # ids in any AI-involving collision pair
    _depart_step, _arrive_step, _route_len = {}, {}, {}
    _edge_len_cache = {}
    _dl_run, _dl_events, _dl_steps = 0, 0, 0        # deadlock detector (≥150 steps mean v<0.5, no completions)
    if _probe_on:
        print(f"🔬 PROBE armed -> {args.out}.probe.csv")
    # ── DEADLOCK NUDGE (proof-of-concept intervention, DMPO_DEADLOCK_NUDGE=1) ──
    # Detect a junction standoff (≥2 AI stopped ≥5 s within 40 m of the junction) and
    # release exactly ONE vehicle (closest to center, tie-break by ID): force throttle
    # +1 / yield −1 for 15 steps. Tests whether the mutual-yield deadlock is the
    # throughput bottleneck: if freeing one leader un-jams the system, it is.
    _nudge_on = os.environ.get('DMPO_DEADLOCK_NUDGE', '0') == '1'
    _stop_counters, _nudge_left, _nudge_vid, _nudge_count, _cooldown, _last_activity = {}, 0, None, [0], {}, [0]
    _jx, _jy = 0.0, 0.0
    # RULE-POC state (strict FCFS role assignment; see block in the step loop)
    _poc_on = os.environ.get('DMPO_RULE_POC', '0') == '1'
    _poc_arrival = {}
    _pjx, _pjy = 200.0, 200.0
    if _poc_on:
        try:
            _pjx, _pjy = traci.junction.getPosition('center')
        except Exception:
            pass
        print(f"📏 RULE-POC armed: strict FCFS, junction at ({_pjx:.0f},{_pjy:.0f})")
    if _nudge_on:
        try:
            _jx, _jy = traci.junction.getPosition('center')
        except Exception:
            pass
        print(f"🔓 DEADLOCK-NUDGE armed (junction at {_jx:.0f},{_jy:.0f})")
    _pos_hist = {}   # vid -> (prev_xy, curr_xy) rolling two-frame cache for event delta-v
    _lane_hist = {}  # vid -> last lane index; counts simulator-initiated lane changes
    _lc_events = 0   # (quantify SUMO's lateral activity)
    # OBSERVATION LATENCY (eval-only, gated DMPO_OBS_DELAY_STEPS=k, default 0 = off): each
    # vehicle acts on its observation from k control steps earlier (its oldest one while it is
    # younger than k steps) — an end-to-end delay covering communication and processing, a
    # conservative superset of V2V delay (the car's own state is delayed too).
    _obs_delay = int(os.environ.get('DMPO_OBS_DELAY_STEPS', '0') or 0)
    _obs_hist = {}   # vid -> list of past flat obs rows (at most k+1)
    if _obs_delay > 0:
        print(f"⏱ OBSERVATION LATENCY: {_obs_delay} steps ({_obs_delay * 0.2:.1f} s)")
    # COMPUTE TIMING (eval-only, gated EVAL_TIMING=1, default off; not a DMPO_ var so the rig
    # hash is unchanged): per control step, wall-clock of the policy side (flatten, normalize,
    # inference, action dict) and of env.step (actions applied, simulation advance, graph
    # rebuild, next observations, rewards), with the controlled-vehicle count.
    _timing = os.environ.get('EVAL_TIMING', '0') == '1'
    _tlog = []   # (n_controlled, policy_ms, env_step_ms)
    if _timing:
        import time as _time
    for step in range(args.max_steps):
        try:
            for _vid in traci.vehicle.getIDList():
                _cur = traci.vehicle.getPosition(_vid)
                _pos_hist[_vid] = (_pos_hist.get(_vid, (None, _cur))[1], _cur)
                _li = traci.vehicle.getLaneIndex(_vid)
                _prev_li = _lane_hist.get(_vid)
                if _prev_li is not None and _li != _prev_li \
                        and not traci.vehicle.getLaneID(_vid).startswith(':'):
                    _lc_events += 1
                _lane_hist[_vid] = _li
        except Exception:
            pass
        if not observations:
            observations, _, _, info = env.step({})
            if info:
                completed_trips += len(info.get('completed_vehicles', set()))
            continue
        agent_ids = list(observations.keys())
        n = len(agent_ids)
        # ── frozen metrics: real sim clock + controlled-ever roster + trip bookkeeping ──
        if _t0_sim is None:
            try: _t0_sim = traci.simulation.getTime()
            except Exception: _t0_sim = 0.0
        for _vid in agent_ids:
            if _vid not in _controlled_ever:
                _controlled_ever.add(_vid); _depart_step[_vid] = step
                try:
                    _rl = 0.0
                    for _e in traci.vehicle.getRoute(_vid):
                        if _e not in _edge_len_cache:
                            try: _edge_len_cache[_e] = traci.lane.getLength(_e + '_0')
                            except Exception: _edge_len_cache[_e] = 0.0
                        _rl += _edge_len_cache[_e]
                    _route_len[_vid] = _rl
                except Exception: pass
        if step % 25 == 0:
            try:
                _ai_seen.update(v for v in traci.vehicle.getIDList()
                                if v.startswith('ai') and not v.startswith('ai_human_'))
            except Exception: pass
        if os.environ.get('DMPO_DEBUG_COUNT') and step % 50 == 0:
            with open('/tmp/dbg_count.txt','a') as _df:
                _df.write(f"step={step} policy_controls={n} sumo_ai_total={len([v for v in traci.vehicle.getIDList() if v.startswith('ai')])}\n")
        # FLOW TIME SERIES (DMPO_FLOW_SERIES=<path>, read-only getters; long
        # horizons): every 50 steps — vehicles in network, moving fraction (v>0.5 m/s),
        # vehicles stopped INSIDE a junction (box-blocking signature), cumulative arrivals.
        _fs = os.environ.get('DMPO_FLOW_SERIES')
        if _fs:
            _fs_arr = globals().get('_FS_ARR', 0) + traci.simulation.getArrivedNumber()
            globals()['_FS_ARR'] = _fs_arr
            if step % 50 == 0:
                _ids = traci.vehicle.getIDList()
                _sp = [traci.vehicle.getSpeed(_v) for _v in _ids]
                _inj = sum(1 for _v, _s in zip(_ids, _sp)
                           if _s < 0.1 and traci.vehicle.getRoadID(_v).startswith(':'))
                with open(_fs, 'a') as _ff:
                    if os.path.getsize(_fs) == 0 if os.path.exists(_fs) else True:
                        _ff.write("step,sim_s,in_network,moving_frac,stopped_in_junction,arrived_cum\n")
                    _ff.write(f"{step},{traci.simulation.getTime():.1f},{len(_ids)},"
                              f"{(sum(1 for _s in _sp if _s > 0.5) / len(_ids)) if _ids else 1.0:.3f},"
                              f"{_inj},{_fs_arr}\n")
        # RAW STATE DUMP (DMPO_DUMP_STATE=<path>): every AI vehicle's (x,y,speed,edge) per step,
        # so a corrected crossing-conflict feature can be recomputed OFFLINE and its predictive
        # lift compared against the live ctx_pet. Header written once at step 0.
        _ds = os.environ.get('DMPO_DUMP_STATE')
        if _ds:
            with open(_ds, 'a') as _sf:
                if step == 0 or (step == 1 and os.path.getsize(_ds) == 0):
                    _sf.write("step,vid,x,y,speed,edge\n")
                for _v in traci.vehicle.getIDList():
                    if not _v.startswith('ai'):
                        continue
                    try:
                        _p = traci.vehicle.getPosition(_v)
                        _sf.write(f"{step},{_v},{_p[0]:.2f},{_p[1]:.2f},{traci.vehicle.getSpeed(_v):.2f},{traci.vehicle.getRoadID(_v)}\n")
                    except Exception:
                        pass
        if _timing:
            _tp0 = _time.perf_counter()
        obs_array = np.zeros((n, obs_dim), dtype=np.float32)
        for i, aid in enumerate(agent_ids):
            obs = observations[aid]
            # THE canonical concat (Step 9): identical order/defaults to the legacy
            # inline code; the v2 branch RAISES on any missing block.
            flat = ObservationProcessor.flatten(obs)
            if _OBS_V2:
                # v2: pad/truncate would silently amputate or blind (R1) — hard assert.
                assert len(flat) == obs_dim, (
                    f'OBS-V2 dim mismatch: flatten={len(flat)} vs obs_dim={obs_dim}')
            else:
                if len(flat) < obs_dim: flat = np.pad(flat, (0, obs_dim - len(flat)))
                elif len(flat) > obs_dim: flat = flat[:obs_dim]
            obs_array[i] = np.clip(flat, -10, 10)
        if _obs_delay > 0:
            for i, aid in enumerate(agent_ids):
                _h = _obs_hist.setdefault(aid, [])
                _h.append(obs_array[i].copy())
                if len(_h) > _obs_delay + 1:
                    del _h[0]
                obs_array[i] = _h[0]
            for _gone in [v for v in _obs_hist if v not in set(agent_ids)]:
                del _obs_hist[_gone]
        # RAW OBS DUMP (DMPO_DUMP_OBS=<path>): per AI vehicle, the 180-dim obs BEFORE zfilter
        # normalization, so we can measure which dims are dead/constant vs carry signal.
        _do = os.environ.get('DMPO_DUMP_OBS')
        if _do:
            with open(_do, 'a') as _of:
                for i, aid in enumerate(agent_ids):
                    _of.write(f"{step},{aid}," + ",".join(f"{v:.4f}" for v in obs_array[i]) + "\n")
        _obs_raw = obs_array.copy() if os.environ.get('DMPO_TRACE_JXN') else None   # pre-norm snapshot for the junction trace only
        if obs_mean is not None:
            std = np.sqrt(obs_var + 1e-8)
            obs_array = ((obs_array - obs_mean.astype(np.float32)) / std.astype(np.float32))
            obs_array = np.clip(obs_array, -5, 5).astype(np.float32)
        if args.baseline != 'none':
            # constant-go prior: full throttle, no lateral, no yield toggle. Under the
            # shielded profile this IS the rule-controller; under 'sumo' actions are ignored.
            action_np = np.tile(np.array([1.0, 0.0, 0.0, -1.0], dtype=np.float32)[:action_dim], (n, 1))
        else:
            if _padded_ckpt:
                # scatter live agents into their fixed slots; zeros elsewhere
                _slotmap = getattr(env, '_vehicle_to_slot', {}) or {}
                _obs_pad = np.zeros((_pad_n, obs_array.shape[1]), dtype=np.float32)
                _gather_idx = np.zeros(len(agent_ids), dtype=np.int64)
                _gather_ok = np.zeros(len(agent_ids), dtype=bool)
                for _i, _aid in enumerate(agent_ids):
                    _sl = _slotmap.get(_aid)
                    if _sl is not None and _sl < _pad_n:
                        _obs_pad[_sl] = obs_array[_i]
                        _gather_idx[_i] = _sl; _gather_ok[_i] = True
                obs_tensor = torch.as_tensor(_obs_pad, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                dist = agent.act(obs_tensor)
                # DMPO_EVAL_STOCHASTIC=1: sample from the trained policy instead of
                # its mean — the sigma-hypothesis probe (training-time collision gains
                # that vanish under deterministic-mean eval may live in the sampling
                # noise). Diagnostic mode only; the frozen protocol stays mean-based.
                if os.environ.get('DMPO_EVAL_STOCHASTIC', '0') == '1':
                    action_tensor = dist.sample()
                else:
                    action_tensor = dist.mean
                if _ens_extra:
                    _means = [action_tensor]
                    _orig = agent.actors
                    for _ac in _ens_extra:
                        agent.actors = _ac
                        _means.append(agent.act(obs_tensor).mean)
                    agent.actors = _orig
                    _stack = torch.stack(_means)              # [k, 1, n, act_dim]
                    if os.environ.get('DMPO_ENSEMBLE_MODE', 'mean') == 'vote':
                        # MAJORITY-VOTE consolidation (2026-08-01): crossing decisions
                        # are bimodal (go/yield); action-MEAN of disagreeing members
                        # produces hesitant half-approaches (tested: VAL 3.4, worse
                        # than single). Instead: per agent, classify each member's
                        # longitudinal intent (brake if a0 < -0.15 ~= -1.2 m/s^2),
                        # take the MAJORITY mode, and execute the mean of the
                        # majority members only — decisive either way, never mushy.
                        _brake = (_stack[..., 0] < -0.15)     # [k, 1, n]
                        _maj_brake = _brake.float().mean(dim=0) > 0.5   # [1, n]
                        _sel = torch.where(_maj_brake.unsqueeze(0), _brake, ~_brake)  # [k,1,n]
                        _w = _sel.float().unsqueeze(-1)       # [k,1,n,1]
                        action_tensor = (_stack * _w).sum(dim=0) / _w.sum(dim=0).clamp(min=1.0)
                    else:
                        action_tensor = _stack.mean(dim=0)
            action_np = action_tensor.squeeze(0).cpu().numpy()
            if _padded_ckpt:
                _apad = action_np
                action_np = np.zeros((len(agent_ids), _apad.shape[-1]), dtype=np.float32)
                if _gather_ok.any():
                    action_np[_gather_ok] = _apad[_gather_idx[_gather_ok]]

        actions = {}
        for i, aid in enumerate(agent_ids):
            a = np.clip(action_np[i], -1.0, 1.0)
            actions[aid] = {
                'longitudinal': np.array([a[0]]),
                'lateral': np.array([a[1]]) if len(a) > 1 else np.array([0.0]),
                'lane_intention': np.array([a[2]]) if len(a) > 2 else np.array([0.0]),
                'yield_intent': np.array([a[3]]) if len(a) > 3 else np.array([0.0]),
            }
        if _timing:
            _tp1 = _time.perf_counter()

        # ── FOCUSED JUNCTION TRACE (DMPO_TRACE_JXN="x,y,R,outpath"): for every AI vehicle
        # within R metres of (x,y), log obs-features + policy action, to see whether the
        # OBSERVATION shows the crossing conflict and what the POLICY does about it.
        _tj = os.environ.get('DMPO_TRACE_JXN')
        if _tj:
            try:
                _tx, _ty, _tr, _tp = _tj.split(',')
                _tx = float(_tx); _ty = float(_ty); _tr = float(_tr)
                with open(_tp, 'a') as _tf:
                    for i, aid in enumerate(agent_ids):
                        try:
                            _px, _py = traci.vehicle.getPosition(aid)
                        except Exception:
                            continue
                        if (_px - _tx) ** 2 + (_py - _ty) ** 2 > _tr * _tr:
                            continue
                        _ob = obs_array[i]        # NORMALIZED (what the policy sees)
                        _rw = _obs_raw[i]         # RAW (pre-normalization)
                        try:
                            _sp = traci.vehicle.getSpeed(aid)
                        except Exception:
                            _sp = -1.0
                        if _OBS_V2:
                            # Step 16: layout-derived offsets (N from obs_dim = 61+12N;
                            # ARM-B profile: ego 12->16 and +4/slot conflict appendix,
                            # so subtract 20 before the //12 inversion and shift nb0).
                            _prof = os.environ.get('DMPO_PROFILE_OBS', '0') == '1'
                            _ego_d = 16 if _prof else 12
                            _n_nb = (obs_dim - 61 - (20 if _prof else 0)) // 12
                            _nb0 = _ego_d + 5
                            _nb_end = _nb0 + 12 * _n_nb
                            _dist = float(_rw[2])                    # ego dist_jx (/R)
                            _dist_n = float(_ob[2])
                            _ident = int(np.argmax(_rw[_nb_end + 8:_nb_end + 12]))
                            _cw = float(_rw[_nb_end + 12 + 1 + 28])  # pet_urgency
                            _nb = _rw[_nb0:_nb_end].reshape(_n_nb, 12)
                        else:
                            _dist = float(_rw[6])                    # RAW dist-to-next-junction (>=0)
                            _dist_n = float(_ob[6])                  # normalized dist (what policy sees)
                            _ident = int(np.argmax(_rw[175:179]))    # approach direction N/E/S/W
                            _cw = float(_rw[179])                    # RAW collision_warning
                            _nb = _rw[15:165].reshape(15, 10)
                        _nn = int((np.abs(_nb).sum(axis=1) > 1e-6).sum())  # RAW populated neighbors
                        _a = np.clip(action_np[i], -1.0, 1.0)
                        _long = float(_a[0]); _yi = float(_a[3]) if len(_a) > 3 else 0.0
                        # ALIGNMENT: RAW ego0 should == px/200 (proves raw obs is correct).
                        _ego = ",".join(f"{_rw[k]:+.3f}" for k in range(10))
                        _tf.write(f"{step},{aid},{_px:.1f},{_py:.1f},{_sp:.2f},{_dist:.3f},{_dist_n:.3f},"
                                  f"{_ident},{_cw:.3f},{_nn},{_long:+.3f},{_yi:+.3f},EGO,{_ego}\n")
            except Exception:
                pass

        # ── RULE-BASED POC (DMPO_RULE_POC=1): literature mechanism as eval-time rule.
        # FCFS role assignment at the junction: ONE prioritized vehicle (earliest zone
        # arrival; tie-break approach id, then vid) gets GO when the junction is clear
        # of conflicting traffic; all other queue heads HOLD. Policy keeps everything
        # else. Validates: role asymmetry + occupancy gating kills BOTH failure modes.
        if _poc_on:
            try:
                _inside, _zone = [], {}
                for vid in agent_ids:
                    try:
                        _x, _y = traci.vehicle.getPosition(vid)
                        _d2 = (_x - _pjx) ** 2 + (_y - _pjy) ** 2
                        _rd = traci.vehicle.getRoadID(vid)
                    except Exception:
                        continue
                    if _rd.startswith(':'):
                        _inside.append(vid)
                    elif _d2 < 3600.0 and 'to_center' in _rd:      # 60 m approach zone
                        _zone[vid] = _rd
                        _poc_arrival.setdefault(vid, step)
                for vid in list(_poc_arrival):
                    if vid not in _zone and vid not in _inside:
                        _poc_arrival.pop(vid, None)                 # left zone or entered junction
                # STRICT FCFS: earliest zone-arrival (tie: approach id, then vid) gets GO
                # when the junction is EMPTY; all other zone vehicles within 30 m HOLD.
                # GO also for same-approach followers of the leader (platoon release)
                # is deliberately NOT done in v1 — strictest, safest baseline first.
                _q = sorted(_zone, key=lambda v: (_poc_arrival.get(v, 1 << 30), _zone[v], v))
                # DEMONSTRATION MODE (DMPO_RULE_POC_KRAUSS=1): the rule acts through
                # Krauss — HOLD = setSpeed(0) (smooth stop at the line), GO =
                # setSpeed(-1)+speedMode 31 (gap-respecting entry). Combined with
                # DMPO_SUMO_BASELINE=1 (Krauss drives everyone), this is safe at BOTH
                # levels: demonstration-grade coordinated driving.
                _kr = os.environ.get('DMPO_RULE_POC_KRAUSS', '0') == '1'
                for i, vid in enumerate(_q):
                    if not _kr and vid not in actions:
                        continue
                    if i == 0 and not _inside:
                        if _kr:
                            traci.vehicle.setSpeedMode(vid, 31)
                            traci.vehicle.setSpeed(vid, -1.0)              # GO, Krauss-safe
                        else:
                            actions[vid]['longitudinal'] = np.array([1.0])
                            actions[vid]['yield_intent'] = np.array([-1.0])
                        if args.gui:
                            traci.vehicle.setColor(vid, (0, 255, 0, 255))   # GREEN = has the slot
                    else:
                        try:
                            _x, _y = traci.vehicle.getPosition(vid)
                            if (_x - _pjx) ** 2 + (_y - _pjy) ** 2 < 900.0:   # within 30 m
                                if _kr:
                                    traci.vehicle.setSpeedMode(vid, 31)
                                    traci.vehicle.setSpeed(vid, 0.0)          # HOLD, smooth stop
                                else:
                                    actions[vid]['longitudinal'] = np.array([-1.0])
                                    actions[vid]['yield_intent'] = np.array([1.0])
                                if args.gui:
                                    traci.vehicle.setColor(vid, (255, 0, 0, 255))  # RED = holding
                        except Exception:
                            pass
            except Exception:
                pass

        if _nudge_on:
            try:
                for vid in agent_ids:
                    try:
                        _sp = traci.vehicle.getSpeed(vid)
                        _px, _py = traci.vehicle.getPosition(vid)
                    except Exception:
                        continue
                    if _sp < 0.3 and (_px - _jx) ** 2 + (_py - _jy) ** 2 < 1600.0:
                        _stop_counters[vid] = _stop_counters.get(vid, 0) + 1
                    else:
                        _stop_counters[vid] = 0
                # v3: release = REAL Krauss (speedMode 31 restores gap/junction safety),
                # held until the vehicle clears the junction zone; strict 10 s standoff
                # detector (leaders only, >=2 approaches); 20 s per-vehicle cooldown.
                if _nudge_vid is not None:
                    try:
                        _px, _py = traci.vehicle.getPosition(_nudge_vid)
                        _sp = traci.vehicle.getSpeed(_nudge_vid)
                        if (_px - _jx) ** 2 + (_py - _jy) ** 2 > 2500.0 and _sp > 1.0:
                            traci.vehicle.setSpeedMode(_nudge_vid, 6)   # cleared: back to policy
                            _nudge_vid = None
                    except Exception:
                        _nudge_vid = None   # left the sim (completed)
                # v4: GLOBAL-STALL trigger — intervene only when the JUNCTION ITSELF has
                # had zero activity (no vehicle inside moving >1 m/s) for 50 steps (10 s).
                # A flowing junction with functional yielding waves never triggers.
                _active = False
                for _v2 in agent_ids:
                    try:
                        _vx, _vy = traci.vehicle.getPosition(_v2)
                        if (_vx - _jx) ** 2 + (_vy - _jy) ** 2 < 625.0 and traci.vehicle.getSpeed(_v2) > 1.0:
                            _active = True
                            break
                    except Exception:
                        continue
                if _active:
                    _last_activity[0] = step
                # DMPO_NUDGE_EARLY=1 reproduces the v3 early trigger (the proven rescue):
                # skip the global-stall wait and act on the per-vehicle standoff directly.
                _global_stall = ((step - _last_activity[0]) > 50
                                 or os.environ.get('DMPO_NUDGE_EARLY', '0') == '1')
                if _nudge_vid is None and _global_stall:
                    _stuck = [v for v, c in _stop_counters.items() if c >= 50 and v in actions
                              and step - _cooldown.get(v, -10**9) > 100]
                    _lead_per_approach = {}
                    for v in _stuck:
                        try:
                            x, y = traci.vehicle.getPosition(v)
                            appr = traci.vehicle.getRoadID(v)
                            d2v = (x - _jx) ** 2 + (y - _jy) ** 2
                        except Exception:
                            continue
                        if appr not in _lead_per_approach or d2v < _lead_per_approach[appr][1]:
                            _lead_per_approach[appr] = (v, d2v)
                    if len(_lead_per_approach) >= 2:
                        _nudge_vid = min((t[0] for t in _lead_per_approach.values()),
                                         key=lambda v: (min(t2[1] for a2, t2 in _lead_per_approach.items()
                                                            if t2[0] == v), v))
                        _nudge_count[0] += 1
                        _stop_counters[_nudge_vid] = 0
                        _cooldown[_nudge_vid] = step
                        try:
                            traci.vehicle.setSpeedMode(_nudge_vid, 31)  # FULL safety back on
                            traci.vehicle.setSpeed(_nudge_vid, -1.0)    # true Krauss control
                            # FULL handoff: also re-center laterally — releasing a vehicle
                            # that straddles lanes lets Krauss drive it forward mid-straddle.
                            traci.vehicle.changeSublane(_nudge_vid, -traci.vehicle.getLateralLanePosition(_nudge_vid))
                            traci.vehicle.setColor(_nudge_vid, (255, 0, 0, 255))  # RED = released
                        except Exception:
                            pass
                        print(f"  🔓 NUDGE #{_nudge_count[0]} @step {step}: {_nudge_vid} -> TRUE Krauss (until clear)")
                if _nudge_vid is not None:
                    actions.pop(_nudge_vid, None)   # policy hands off while released
            except Exception:
                pass

        if _probe_on:
            for aid in agent_ids:
                try:
                    _o = observations.get(aid, {})
                    _ctx = _o.get('context', np.zeros(10))
                    _act = actions.get(aid, {})
                    _ld = None
                    try:
                        _ld = traci.vehicle.getLeader(aid, 60.0)
                    except Exception:
                        pass
                    _spd = round(traci.vehicle.getSpeed(aid), 2)
                    _lat = round(traci.vehicle.getLateralLanePosition(aid), 2)
                    _inj = 1 if traci.vehicle.getRoadID(aid).startswith(':') else 0
                    _gap = round(_ld[1], 1) if _ld else -1
                    _ay = round(float(_act.get('yield_intent', [0])[0]), 3)
                    _cp = round(float(_ctx[3]), 3)
                    _cpet = round(float(_ctx[4]), 3)
                    # LC-theory signals: lane index, route-need (bestLaneOffset:
                    # 0 = current lane serves the route), and change feasibility.
                    _lidx, _boff, _cl, _cr = -1, 0, 0, 0
                    try:
                        _lidx = traci.vehicle.getLaneIndex(aid)
                        for _bl in traci.vehicle.getBestLanes(aid):
                            if _bl[0] == traci.vehicle.getLaneID(aid):
                                _boff = int(_bl[3]); break
                        _cl = 1 if traci.vehicle.couldChangeLane(aid, 1) else 0
                        _cr = 1 if traci.vehicle.couldChangeLane(aid, -1) else 0
                    except Exception:
                        pass
                    _probe_rows.append((
                        step, aid, _spd, _lat, _inj, _gap,
                        round(float(_act.get('longitudinal', [0])[0]), 3),
                        _ay, _cp, _cpet, _lidx, _boff, _cl, _cr,
                    ))
                    # ── LIVE HIGHLIGHTER (GUI validation): DMPO_HIGHLIGHT=P1|P2|P3|P5|P7
                    # colors a vehicle the moment its detector condition fires, so the
                    # user can visually confirm/reject each problem detector.
                    _hl = os.environ.get('DMPO_HIGHLIGHT', '')
                    if _hl and args.gui:
                        _colored = False
                        if _hl == 'P1' and _spd < 0.3 and not _inj and _gap < 0:
                            traci.vehicle.setColor(aid, (255, 0, 0, 255))      # RED stopped queue-leader
                            _colored = True
                        elif _hl == 'P2' and _spd < 0.5 and not _inj and (_gap < 0 or _gap > 30) and _cpet > 0.4:
                            traci.vehicle.setColor(aid, (255, 0, 255, 255))    # MAGENTA stopped, clear path
                            _colored = True
                        elif _hl == 'P3' and _ay > 0 and _cp > 0.9 and _cpet < 0.99:
                            traci.vehicle.setColor(aid, (255, 128, 0, 255))    # ORANGE yields despite priority
                            _colored = True
                        elif _hl == 'P3B':
                            # constant-yield visualization: RED = yield output ON,
                            # GREEN = proceeding. If freeze-thesis holds, ~all RED always.
                            if _ay > 0:
                                traci.vehicle.setColor(aid, (255, 0, 0, 255))
                            else:
                                traci.vehicle.setColor(aid, (0, 255, 0, 255))
                            _colored = True
                        elif _hl == 'P5' and _spd < 0.3 and not _inj and 0 <= _gap <= 15:
                            traci.vehicle.setColor(aid, (255, 255, 0, 255))    # YELLOW trapped follower
                            _colored = True
                        elif _hl == 'P7' and abs(_lat) > 0.8:
                            traci.vehicle.setColor(aid, (255, 0, 0, 255))      # RED lane-straddler
                            _colored = True
                        if not _colored:
                            traci.vehicle.setColor(aid, (0, 160, 255, 255))    # calm blue otherwise
                except Exception:
                    continue

        if _timing:
            _te0 = _time.perf_counter()
        new_obs, rewards, dones, info = env.step(actions)
        if _timing:
            _tlog.append((n, round((_tp1 - _tp0) * 1e3, 3), round((_time.perf_counter() - _te0) * 1e3, 3)))

        if os.environ.get('DMPO_DEBUG_EMERGENCY', '0') == '1':
            # Forensic: surface SUMO-internal interventions that are invisible
            # under --no-warnings (emergency stops bypass emergencyDecel).
            try:
                _t = traci.simulation.getTime()
                for _v in traci.simulation.getEmergencyStoppingVehiclesIDList():
                    print(f"[EMERGENCY-STOP] t={_t:.1f}s {_v}", flush=True)
                for _v in traci.simulation.getStopStartingVehiclesIDList():
                    print(f"[STOP-START] t={_t:.1f}s {_v}", flush=True)
            except Exception:
                pass

        if os.environ.get('DMPO_LANE_DUMP') and step > 0 and step % int(os.environ.get('DMPO_LANE_DUMP','120')) == 0:
            print(f"[LANEDUMP] step={step} — per-approach lane veh/halted (lane0=rightmost):", flush=True)
            for _eid in ['north_to_center','south_to_center','east_to_center','west_to_center']:
                _row=[]
                for _i in range(4):
                    try:
                        _n=traci.lane.getLastStepVehicleNumber(f"{_eid}_{_i}"); _h=traci.lane.getLastStepHaltingNumber(f"{_eid}_{_i}")
                    except Exception: _n=_h=-1
                    _row.append(f"L{_i}={_n}/{_h}")
                print(f"  {_eid:16s} {' '.join(_row)}", flush=True)

        if args.gui and args.target:
            try:
                _ids = traci.vehicle.getIDList()
                if args.target in _ids:
                    traci.vehicle.setColor(args.target, (255, 0, 0, 255))   # target = RED
                    _tp = traci.vehicle.getPosition(args.target); _near = None; _nd = 1e9
                    for _v in _ids:
                        if _v == args.target: continue
                        _vp = traci.vehicle.getPosition(_v)
                        _d = ((_tp[0]-_vp[0])**2 + (_tp[1]-_vp[1])**2) ** 0.5
                        if _d < _nd: _nd = _d; _near = _v
                    if _prev_partner[0] and _prev_partner[0] != _near:
                        try: traci.vehicle.setColor(_prev_partner[0], (0, 0, 255, 255))
                        except Exception: pass
                    if _near and _nd < 25:
                        traci.vehicle.setColor(_near, (255, 0, 0, 255)); _prev_partner[0] = _near
            except Exception:
                pass

        # === Verbose: read ALL new collision pairs (incl. human-human) ===
        new_pairs = info.get('all_new_collision_pair_records', [])
        for c in new_pairs:
            col = c['collider']; vic = c['victim']
            col_is_ai = c.get('col_is_ai', col.startswith('ai'))
            vic_is_ai = c.get('vic_is_ai', vic.startswith('ai'))
            if col_is_ai and vic_is_ai:
                kind = 'ai-ai'; ai_pair_count += 1
            elif col_is_ai or vic_is_ai:
                kind = 'ai-human'; mixed_pair_count += 1
            else:
                kind = 'human-human'; human_pair_count += 1
            # Mechanism: same heuristic as env (lane starting with ':' = junction internal)
            lane = c.get('lane', '') or ''
            mech = 'cross_traffic_junction' if lane.startswith(':') else 'rear_end_same_lane'
            ctype = c.get('type', '')   # SUMO collision type: collision/frontal/sharedLane/junction
            # Capture lateral positions to test the "off-center leader" hypothesis.
            # getLateralLanePosition: offset from lane center (m); ~0 = centered.
            def _lat(vid):
                try:
                    lp = traci.vehicle.getLateralLanePosition(vid)
                    ln = traci.vehicle.getLaneID(vid)
                    lw = traci.lane.getWidth(ln) if ln else None
                    return {'lane': ln, 'lat_offset_m': round(float(lp), 3),
                            'lane_width_m': round(float(lw), 2) if lw else None,
                            'frac_off_center': round(abs(lp) / (lw/2), 2) if lw else None}
                except Exception:
                    return {'lane': None, 'lat_offset_m': None}
            col_lat = _lat(col); vic_lat = _lat(vic)
            # Heading/angle at collision (from env trace state when DMPO_COLLISION_TRACE=1)
            cstate = c.get('collider_state') or {}
            vstate = c.get('victim_state') or {}
            col_ang = cstate.get('angle_deg'); vic_ang = vstate.get('angle_deg')
            # Live angles as a fallback
            try: col_ang_live = round(float(traci.vehicle.getAngle(col)), 1)
            except Exception: col_ang_live = None
            try: vic_ang_live = round(float(traci.vehicle.getAngle(vic)), 1)
            except Exception: vic_ang_live = None
            # Vector delta-v (severity): relative velocity magnitude at impact,
            # from the two-frame position cache. Scalar speed differences
            # understate angle collisions by up to an order of magnitude.
            _dv = None
            try:
                _pc, _cc = _pos_hist.get(col, (None, None))
                _pv, _cv = _pos_hist.get(vic, (None, None))
                if _pc and _cc and _pv and _cv:
                    _vcx, _vcy = (_cc[0]-_pc[0])/0.2, (_cc[1]-_pc[1])/0.2
                    _vvx, _vvy = (_cv[0]-_pv[0])/0.2, (_cv[1]-_pv[1])/0.2
                    _dv = round(((_vcx-_vvx)**2 + (_vcy-_vvy)**2)**0.5, 2)
            except Exception:
                pass
            all_events.append({
                'step': step, 'sim_time': step * 0.2,
                'delta_v': _dv,
                'collider': col, 'victim': vic,
                'kind': kind,
                'mechanism': mech,
                'sumo_type': str(ctype),
                'lane': lane,
                'pos': c.get('pos', None),
                'collider_lateral': col_lat,
                'victim_lateral': vic_lat,
                'collider_angle': col_ang, 'victim_angle': vic_ang,
                'collider_angle_live': col_ang_live, 'victim_angle_live': vic_ang_live,
            })
            _collided_ids.add(col); _collided_ids.add(vic)   # frozen metrics: success-rate numerator filter
            print(f"      angles: collider={col_ang_live}°  victim={vic_ang_live}°  "
                  f"(victim lane heading should be ~270° for east→center)")
            print(f"  [t={step*0.2:6.1f}s] {kind:9s} {mech:22s} sumo_type={str(ctype):10s} "
                  f"{col} (lat {col_lat.get('lat_offset_m')}, lane {col_lat.get('lane')}) <-> "
                  f"{vic} (lat {vic_lat.get('lat_offset_m')}, lane {vic_lat.get('lane')})")

        # also keep tally of all_collisions_raw via the sticky counter (best proxy)
        sticky = info.get('ai_collisions_this_step', set())
        raw_record_count += len(sticky)

        # Count AI trip completions this step (uses same key as training)
        _done_now = info.get('completed_vehicles', set())
        completed_trips += len(_done_now)
        for _vid in _done_now:
            if _vid in _controlled_ever and _vid not in _arrive_step:
                _arrive_step[_vid] = step
        # deadlock detector (controlled vehicles only; passive — never intervenes)
        try:
            _spds = [traci.vehicle.getSpeed(v) for v in agent_ids if v not in _done_now]
            _mean_v = (sum(_spds) / len(_spds)) if _spds else 99.0
        except Exception:
            _mean_v = 99.0
        if _mean_v < 0.5 and not _done_now and _spds:
            _dl_run += 1
        else:
            if _dl_run >= 150:
                _dl_events += 1; _dl_steps += _dl_run
            _dl_run = 0

        observations = new_obs

    if _dl_run >= 150:
        _dl_events += 1; _dl_steps += _dl_run
    try:
        _t1_sim = traci.simulation.getTime()
    except Exception:
        _t1_sim = None
    env.close()

    # Mechanism breakdown (AI-involving only)
    ai_events = [e for e in all_events if e['kind'] != 'human-human']
    rear_end_total = sum(1 for e in ai_events if e['mechanism'] == 'rear_end_same_lane')
    junction_total = sum(1 for e in ai_events if e['mechanism'] == 'cross_traffic_junction')
    # Mechanism × pair kind
    breakdown = {}
    for e in all_events:
        key = (e['kind'], e['mechanism'])
        breakdown[key] = breakdown.get(key, 0) + 1

    print()
    print("=" * 70)
    print(f"  VERBOSE COLLISION SUMMARY ({args.max_steps * 0.2:.0f}s sim)")
    print("=" * 70)
    print(f"  AI-AI new pairs:       {ai_pair_count}")
    print(f"  AI-human new pairs:    {mixed_pair_count}")
    print(f"  Human-human new pairs: {human_pair_count}  (NOT counted in eval metric!)")
    print(f"  AI involving total:    {ai_pair_count + mixed_pair_count}  (=eval 'Collision events')")
    print(f"  Vehicle-steps in AI collision (sticky): {raw_record_count}")
    print(f"  AI trip completions:                    {completed_trips}")
    print()
    print(f"  Total distinct pairs:  {ai_pair_count + mixed_pair_count + human_pair_count}")
    print()
    print(f"  --- By mechanism (AI-involving only) ---")
    print(f"  rear_end_same_lane:      {rear_end_total}")
    print(f"  cross_traffic_junction:  {junction_total}")
    print()
    print(f"  --- Breakdown (kind × mechanism) ---")
    for (k, m), n in sorted(breakdown.items()):
        print(f"    {k:14s} {m:24s}  {n}")

    # ── FROZEN METRICS (Stage 2): computed from the instrumented run ──
    real_sim_s = (_t1_sim - _t0_sim) if (_t0_sim is not None and _t1_sim is not None) \
        else args.max_steps * _step_len
    completed_ctrl = set(_arrive_step)
    collided_ctrl = _collided_ids & _controlled_ever
    success_set = completed_ctrl - _collided_ids
    known_outcomes = completed_ctrl | collided_ctrl
    success_rate = (len(success_set) / len(known_outcomes)) if known_outcomes else None
    censored = len(_controlled_ever) - len(known_outcomes)   # still driving at horizon
    throughput_vph = len(completed_ctrl) / real_sim_s * 3600.0 if real_sim_s > 0 else None
    _delays = []
    for _vid, _ar in _arrive_step.items():
        _rl = _route_len.get(_vid, 0.0)
        if _rl > 0:
            _dur = (_ar - _depart_step[_vid]) * _step_len
            _delays.append(_dur - _rl / 16.67)
    avg_delay_s = (sum(_delays) / len(_delays)) if _delays else None
    coll_per_trip = ((ai_pair_count + mixed_pair_count) / len(completed_ctrl)) if completed_ctrl else None
    tracked_fraction = (len(_controlled_ever) / len(_ai_seen | _controlled_ever)) \
        if (_ai_seen | _controlled_ever) else None
    # per-crash-EVENT clustering: pairs sharing a vehicle within 25 steps collapse transitively
    _ai_ev = sorted((e for e in all_events if e['kind'] != 'human-human'), key=lambda e: e['step'])
    _clusters = []   # each: {'ids': set, 'last_step': int, 'n': int}
    for _e in _ai_ev:
        _ids = {_e['collider'], _e['victim']}
        _home = None
        for _c in _clusters:
            if (_e['step'] - _c['last_step'] <= 25) and (_ids & _c['ids']):
                _home = _c; break
        if _home is None:
            _clusters.append({'ids': set(_ids), 'last_step': _e['step'], 'n': 1})
        else:
            _home['ids'] |= _ids; _home['last_step'] = _e['step']; _home['n'] += 1
    crash_events = len(_clusters)

    print(f"  --- FROZEN METRICS (controlled-only; pair-dedup definition DISCLOSED) ---")
    print(f"  real_sim_s={real_sim_s:.1f}  controlled_ever={len(_controlled_ever)}  "
          f"tracked_fraction={tracked_fraction if tracked_fraction is None else round(tracked_fraction,3)}")
    print(f"  crash_EVENTS={crash_events} (clustered)  success_rate={success_rate if success_rate is None else round(success_rate,3)}"
          f"  censored={censored}")
    print(f"  throughput={throughput_vph if throughput_vph is None else round(throughput_vph,1)} veh/h  "
          f"avg_delay={avg_delay_s if avg_delay_s is None else round(avg_delay_s,1)} s  "
          f"deadlock_events={_dl_events} ({_dl_steps} steps)")

    summary = {
        'checkpoint': args.checkpoint,
        'scenario': args.scenario,
        'baseline': args.baseline,
        'block_profile': os.environ.get('DMPO_BLOCK_PROFILE'),
        # artifact-hygiene (verification sweep): a result must be interpretable
        # without the launcher — record the condition + the exact rig hash.
        'traffic_scale': os.environ.get('DMPO_TRAFFIC_SCALE'),
        'eval_stochastic': os.environ.get('DMPO_EVAL_STOCHASTIC', '0'),
        'sense': {k: os.environ.get(k) for k in ('DMPO_SENSE_POS_SIGMA', 'DMPO_SENSE_SPD_SIGMA', 'DMPO_SENSE_DROP',
                                                 'DMPO_SENSE_LANEPOS_SIGMA', 'DMPO_SENSE_INTENT_UNKNOWN',
                                                 'DMPO_OBS_DELAY_STEPS') if os.environ.get(k)},
        'profile_obs': os.environ.get('DMPO_PROFILE_OBS', '0'),
        'max_ai_vehicles': args.max_ai_vehicles,
        'block_version': os.environ.get('DMPO_BLOCK_VERSION'),
        'dmpo_env_hash': __import__('hashlib').sha256(repr(sorted(
            (k, v) for k, v in os.environ.items() if k.startswith('DMPO_')
        )).encode()).hexdigest()[:16],
        'max_steps': args.max_steps,
        'sim_time': args.max_steps * 0.2,
        'real_sim_s': real_sim_s,
        'seed': os.environ.get('DMPO_SUMO_SEED'),
        'controlled_ever': len(_controlled_ever),
        'tracked_fraction': tracked_fraction,
        'completed_controlled': len(completed_ctrl),
        'success_rate': success_rate,
        'censored_at_horizon': censored,
        'crash_events_clustered': crash_events,
        'collisions_per_completed_trip': coll_per_trip,
        'throughput_veh_per_hour': throughput_vph,
        'lane_change_events': _lc_events,
        'avg_delay_s': avg_delay_s,
        'deadlock_events': _dl_events,
        'deadlock_steps': _dl_steps,
        'metric_definitions': {
            'collisions': 'distinct unordered AI-involving pairs, deduplicated over the whole run',
            'crash_events_clustered': 'pairs sharing a vehicle within 25 steps collapse transitively',
            'success_rate': '(completed∩controlled − collided)/known-outcome trips; censored reported',
            'avg_delay_s': 'control-span duration − routeLength/16.67 (completed controlled trips)',
        },
        'ai_trip_completions': completed_trips,
        'ai_ai_pairs': ai_pair_count,
        'ai_human_pairs': mixed_pair_count,
        'human_human_pairs': human_pair_count,
        'ai_involving_total': ai_pair_count + mixed_pair_count,
        'all_pairs_total': ai_pair_count + mixed_pair_count + human_pair_count,
        'rear_end_total': rear_end_total,
        'junction_total': junction_total,
        'breakdown_kind_x_mechanism': {f"{k}|{m}": n for (k, m), n in breakdown.items()},
        'events': all_events,
    }
    if _timing:
        summary['timing'] = {'fields': ['n_controlled', 'policy_ms', 'env_step_ms'], 'steps': _tlog}
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote events to {args.out}")
    if _probe_on and _probe_rows:
        import csv as _csv
        with open(args.out + '.probe.csv', 'w', newline='') as _f:
            _w = _csv.writer(_f)
            _w.writerow(['step', 'vid', 'speed', 'lat', 'in_junction',
                         'leader_gap', 'act_long', 'act_yield', 'ctx_priority', 'ctx_pet',
                         'lane_idx', 'best_offset', 'could_left', 'could_right'])
            _w.writerows(_probe_rows)
        print(f"🔬 Wrote {len(_probe_rows)} probe rows to {args.out}.probe.csv")


if __name__ == '__main__':
    main()
