"""Comprehensive deterministic evaluation: the full traffic-engineering metric set,
computed for REAL from the simulation (no stubbed zeros). Built on the verified
deterministic policy loading + deduplicated collision counting of eval_with_completions.

Metrics (per checkpoint):
  Travel Time Ratio (vs free-flow)   completed-vehicle travel_time / free_flow_time
  Average / Max Waiting Time (s)     time a vehicle spends at speed < WAIT_SPEED
  Average Speed (km/h)               mean per-step speed * 3.6
  Collision Rate                     AI collisions / AI vehicles seen
  Emergency Braking Events           rising-edges of accel < -EBRAKE_DECEL (REAL)
  Near-Miss Events (PET/TTC proxy)   rising-edges of min center-distance < NEARMISS_M (REAL)
  Safety Score (0-10)                documented formula over the REAL components
  Intersection Throughput (veh/h)    completions / sim-hours
  Capacity Utilization               throughput / CAPACITY_BASELINE
  Average Queue Length               mean #vehicles waiting (speed<WAIT_SPEED) per step
  Fairness Index (0-1)               Jain's index over per-vehicle waiting times
  Fuel Savings / CO2 Reduction (%)   simple speed->fuel MODEL (clearly an estimate)
  Total Episodes / Vehicles / Eval Time
"""
import argparse, json, os, sys, time

# ⚠ NON-CANONICAL ENGINE (close-out 2026-07-24).
# The frozen-protocol metrics (real sim_time, controlled-only accounting, success
# rate, deadlock, crash-event clustering, delay) and the --baseline modes live in
# scripts/eval_verbose_collisions.py, which is the CANONICAL engine. This script
# lacks the residual mirror and the TANH_POLICY guard, so it can silently evaluate
# a DIFFERENT policy than training produced. Use run_canonical_eval.sh instead.
# Kept for legacy comparisons only.
from pathlib import Path
if os.environ.get('PYTHONHASHSEED') != '0':
    os.environ['PYTHONHASHSEED'] = '0'; os.execv(sys.executable, [sys.executable] + sys.argv)
import numpy as np, torch
os.environ['DMPO_USE_TRACI'] = '1'
for k, v in {'DMPO_REWARD_CARL_PLUS':'1','DMPO_DYNAMIC':'1','DMPO_TANH_POLICY':'1','DMPO_KRAUSS_CAP':'0',
    'DMPO_COLLISION_ACTION':'warn','DMPO_COLLISION_CHECK_JUNCTIONS':'true','DMPO_TIME_TO_TELEPORT':'-1',
    'DMPO_COLLISION_REMOVE_AI':'0','DMPO_COLLISION_HALT_AI':'0','DMPO_ACTION_VARIANT':'yield_intent',
    'DMPO_COMPLETION_BONUS':'200','DMPO_TEAM_BONUS':'5.0','DMPO_POLICY_BIAS_LONGITUDINAL':'0.3',
    'DMPO_POLICY_BIAS_YIELD_INTENT':'0.5','DMPO_SUMO_HARDRESET_AT':'27000','DMPO_SUMO_SEED':'23423',
    'DMPO_MAX_NEIGHBORS':'15','DMPO_PERCEPTION_RADIUS':'150.0',
    'DMPO_MAX_DECEL':'8',            # H13: was MISSING from the anchor (default 4.5 vs trained 8)
    'DMPO_COLLISION_PENALTY':'400'}.items():  # Stage-3: was MISSING — env fell back to 200 while
    # historical r4/r5/r6 training used 400 (silent train/eval reward divergence). Exported
    # launcher/block values still win over this setdefault.
    os.environ.setdefault(k, v)
# Soundness review (protocol-armor HIGH): this engine has NO residual mirror — under
# DMPO_RESIDUAL=1 it would silently evaluate a DIFFERENT policy than training produced.
if os.environ.get('DMPO_RESIDUAL', '0') == '1':
    sys.exit('eval_comprehensive_metrics.py has no DMPO_RESIDUAL mirror — refusing under '
             'DMPO_RESIDUAL=1; use scripts/eval_verbose_collisions.py (canonical).')
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, os.environ.get('DMPO_FRAMEWORK_PATH', str(Path(__file__).parent.parent / "third_party" / "Networked-MB-MARL")))
import traci  # noqa
from dmpo_autonomous_driving.envs.sumo_intersection_env import create_intersection_env
from dmpo_autonomous_driving.config.intersection_dmpo import create_intersection_dmpo_config
from algorithms.algo.agent.DMPO import DMPOAgent

# --- documented metric thresholds ---
WAIT_SPEED = 2.0          # m/s, below this a vehicle is "waiting"
EBRAKE_DECEL = 4.5        # m/s^2, accel below -this is an emergency brake
NEARMISS_M = 5.0          # m, center distance below this (no collision) is a near-miss
FREE_FLOW_S = 28.0        # s, empirical free-flow trip time (from completion census)
CAPACITY_BASELINE = 1800  # veh/h per the traffic-engineering baseline
QUEUE_WAIT_S = 5.0        # s, waited longer than this counts toward queue length


def jain_fairness(x):
    x = np.asarray([v for v in x if v is not None], float)
    if x.size == 0 or x.sum() == 0: return 1.0
    return float((x.sum() ** 2) / (x.size * (x ** 2).sum()))


def fuel_rate_lphkm(speed_kmh):  # simple model -> L/100km (clearly an estimate)
    if speed_kmh < 10: return 12.0
    if speed_kmh < 50: return 8.0 + (50 - speed_kmh) * 0.1
    return 8.0 + (speed_kmh - 50) * 0.05


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--scenario', default='configs/sumo_configs/4lane_allAI_unsig_poisson_lanefix/intersection.sumocfg')
    p.add_argument('--max-steps', type=int, default=1500)
    p.add_argument('--max-ai-vehicles', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='/tmp/comprehensive_metrics.json')
    p.add_argument('--tag', default='')
    p.add_argument('--gui', action='store_true', help='launch SUMO GUI')
    args = p.parse_args()

    # ── STAGE-0 PROCESS ARMOR: frozen-seed guard (always) + canonical-block gate (DMPO_REQUIRE_BLOCK=1) ──
    try:
        from dmpo_autonomous_driving.eval_seeds import guard_seed
        guard_seed(os.environ.get('DMPO_SUMO_SEED', args.seed))
    except ImportError:
        print('WARNING: eval_seeds not importable — frozen-seed guard SKIPPED')
    if os.environ.get('DMPO_REQUIRE_BLOCK', '0') == '1':
        import subprocess as _sp
        _gate = _sp.run([sys.executable,
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assert_consistency.py'),
                         '--scenario', args.scenario])
        if _gate.returncode != 0:
            sys.exit('REFUSED by assert_consistency (see violations above)')

    t_start = time.time()
    env = create_intersection_env(config_path=args.scenario, max_steps=args.max_steps, use_gui=args.gui,
                                  max_ai_vehicles=args.max_ai_vehicles, training_mode=True)
    config = create_intersection_dmpo_config(args.scenario)
    config.agent_args.n_agent = args.max_ai_vehicles
    config.agent_args.dynamic = True; config.agent_args.entropy_coeff = 0.05; config.agent_args.grad_clip = 0.5
    config.agent_args.adj = np.ones((args.max_ai_vehicles, args.max_ai_vehicles), dtype=np.float32)
    import logging, types
    class StubLogger:
        def __init__(self, name='DMPO'): self.logger = logging.getLogger(name); self.metrics = {}
        def child(self, name): return StubLogger(name)
        def log(self, **kw): pass
        def __getattr__(self, name): return getattr(self.logger, name, lambda *a, **kw: None)
    device = torch.device('cpu')
    agent = DMPOAgent(StubLogger(), device, config.agent_args, config)
    for actor in agent.actors:
        if hasattr(actor, 'action_head') and hasattr(actor, 'log_std'):
            def _tanh(self_actor, obs, a=None):
                output = self_actor.network(obs); mean = torch.tanh(self_actor.action_head(output))
                clamped = torch.clamp(self_actor.log_std, self_actor.LOG_STD_MIN, self_actor.LOG_STD_MAX)
                std = torch.exp(clamped).expand(*mean.shape)
                if a is None: return mean, std
                from torch.distributions import Normal
                return Normal(mean, std).log_prob(a).sum(dim=-1)
            actor.forward = types.MethodType(_tanh, actor)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent.actors.load_state_dict(ck['policy_net_legacy']); agent.vs.load_state_dict(ck['value_net_legacy'])
    try: agent.ps.load_state_dict(ck['model_net_legacy'])
    except Exception: pass
    obs_mean = ck.get('obs_norm_mean'); obs_var = ck.get('obs_norm_var')
    obs_dim = config.agent_args.observation_dim

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    observations = env.reset()
    step_len = getattr(env, 'step_length', 0.2)

    # per-vehicle accumulators
    spd_sum, spd_cnt, wait_t = {}, {}, {}
    first_s, last_s = {}, {}
    ebrake_state, ebrake_cnt = {}, {}
    nearmiss_state, nearmiss_cnt = {}, {}
    lane_prev, lanechange_cnt = {}, {}      # (edge,laneIdx) per AI veh -> count true lane changes
    lc_dir = {'down': 0, 'up': 0}           # toward lane0 (down) vs toward outer lanes (up)
    _LC_DEBUG = os.environ.get('DMPO_LC_DEBUG') == '1'
    completed = set(); raw_pairs = set(); collisions = 0
    queue_per_step = []

    def tq(fn, *a, default=None):
        try: return fn(*a)
        except Exception: return default

    def collect_step():
        """Sample TraCI state for all AI vehicles this step."""
        ids = [v for v in tq(traci.vehicle.getIDList, default=[]) if v.startswith('ai')]
        pos = {v: tq(traci.vehicle.getPosition, v, default=None) for v in tq(traci.vehicle.getIDList, default=[])}
        n_wait = 0
        for v in ids:
            sp = tq(traci.vehicle.getSpeed, v, default=0.0)
            ac = tq(traci.vehicle.getAcceleration, v, default=0.0)
            spd_sum[v] = spd_sum.get(v, 0.0) + sp; spd_cnt[v] = spd_cnt.get(v, 0) + 1
            first_s.setdefault(v, step); last_s[v] = step
            # lane changes: laneIndex transitions on the SAME (non-internal) edge
            _edge = tq(traci.vehicle.getRoadID, v, default=None)
            _li = tq(traci.vehicle.getLaneIndex, v, default=None)
            if _edge and _li is not None and not _edge.startswith(':'):
                _prev = lane_prev.get(v)
                if _prev is not None and _prev[0] == _edge and _prev[1] != _li:
                    lanechange_cnt[v] = lanechange_cnt.get(v, 0) + 1
                    lc_dir['down' if _li < _prev[1] else 'up'] += 1
                    if _LC_DEBUG:
                        _lp = tq(traci.vehicle.getLateralLanePosition, v, default=0.0)
                        print(f"  LC v={v} step={step} lane {_prev[1]}->{_li} latpos={_lp:.2f}m edge={_edge}")
                        tq(traci.vehicle.setColor, v, (255, 0, 0, 255))   # GUI: turn lane-changers RED
                lane_prev[v] = (_edge, _li)
            if sp < WAIT_SPEED:
                wait_t[v] = wait_t.get(v, 0.0) + step_len; n_wait += 1
            # emergency braking: rising edge of hard decel
            hard = ac < -EBRAKE_DECEL
            if hard and not ebrake_state.get(v, False): ebrake_cnt[v] = ebrake_cnt.get(v, 0) + 1
            ebrake_state[v] = hard
            # near-miss: rising edge of close approach (not collision)
            pv = pos.get(v)
            close = False
            if pv is not None:
                for o, po in pos.items():
                    if o == v or po is None: continue
                    d = ((pv[0]-po[0])**2 + (pv[1]-po[1])**2) ** 0.5
                    if 0.1 < d < NEARMISS_M: close = True; break
            if close and not nearmiss_state.get(v, False): nearmiss_cnt[v] = nearmiss_cnt.get(v, 0) + 1
            nearmiss_state[v] = close
        queue_per_step.append(n_wait)

    def tally_collisions(info):
        nonlocal collisions
        for c in (info.get('all_new_collision_pair_records', []) if info else []):
            key = tuple(sorted([c['collider'], c['victim']]))
            if c.get('col_is_ai', c['collider'].startswith('ai')) or c.get('vic_is_ai', c['victim'].startswith('ai')):
                if key not in raw_pairs: raw_pairs.add(key); collisions += 1

    for step in range(args.max_steps):
        if observations:
            ids = list(observations.keys()); n = len(ids)
            arr = np.zeros((n, obs_dim), np.float32)
            for i, aid in enumerate(ids):
                ob = observations[aid]
                # Step 9: THE canonical concat (byte-identical v1 branch; v2 raises on
                # a missing block instead of silently zero-padding a blind policy).
                from dmpo_autonomous_driving.envs.observation_processor import ObservationProcessor as _OP
                flat = _OP.flatten(ob)
                if os.environ.get('DMPO_OBS_V2', '0') == '1':
                    assert len(flat) == obs_dim, (
                        f'OBS-V2 dim mismatch: flatten={len(flat)} vs obs_dim={obs_dim}')
                else:
                    if len(flat) < obs_dim: flat = np.pad(flat,(0,obs_dim-len(flat)))
                    elif len(flat) > obs_dim: flat = flat[:obs_dim]
                arr[i] = np.clip(flat,-10,10)
            if obs_mean is not None:
                std = np.sqrt(obs_var+1e-8)
                arr = np.clip((arr-obs_mean.astype(np.float32))/std.astype(np.float32),-5,5).astype(np.float32)
            with torch.no_grad():
                action_np = agent.act(torch.as_tensor(arr,dtype=torch.float32).unsqueeze(0)).mean.squeeze(0).cpu().numpy()
            actions = {}
            for i, aid in enumerate(ids):
                a = np.clip(action_np[i],-1,1)
                actions[aid] = {'longitudinal':np.array([a[0]]),
                    'lateral':np.array([a[1]]) if len(a)>1 else np.array([0.0]),
                    'lane_intention':np.array([a[2]]) if len(a)>2 else np.array([0.0]),
                    'yield_intent':np.array([a[3]]) if len(a)>3 else np.array([0.0])}
            collect_step()
            observations, _, _, info = env.step(actions)
        else:
            collect_step()
            observations, _, _, info = env.step({})
        tally_collisions(info)
        for cv in (info.get('completed_vehicles', set()) if info else set()):
            if str(cv).startswith('ai'): completed.add(str(cv))

    # ---- aggregate metrics ----
    all_v = set(first_s)
    n_veh = len(all_v); n_comp = len(completed)
    travel_times = [(last_s[v]-first_s[v])*step_len for v in completed if v in first_s]
    avg_speed_kmh = np.mean([spd_sum[v]/max(1,spd_cnt[v]) for v in all_v]) * 3.6 if all_v else 0.0
    waits = [wait_t.get(v, 0.0) for v in all_v]
    sim_hours = (args.max_steps * step_len) / 3600.0
    throughput = n_comp / sim_hours if sim_hours > 0 else 0.0
    n_ebrake = int(sum(ebrake_cnt.values())); n_nearmiss = int(sum(nearmiss_cnt.values()))
    n_lanechange = int(sum(lanechange_cnt.values())); n_lc_vehicles = int(sum(1 for c in lanechange_cnt.values() if c > 0))
    # fuel / emissions MODEL ESTIMATE
    fuel_rates = [fuel_rate_lphkm((spd_sum[v]/max(1,spd_cnt[v]))*3.6) for v in all_v]
    avg_fuel = np.mean(fuel_rates) if fuel_rates else 10.0
    base_fuel = 10.0  # baseline L/100km for "savings"
    fuel_savings = max(0.0, (base_fuel - avg_fuel)/base_fuel*100)
    co2_reduction = fuel_savings * 0.82  # CO2 tracks fuel
    safety_score = max(0.0, 10.0 - 5.0*(collisions/max(1,n_veh)) - 0.3*(n_nearmiss/max(1,n_veh)) - 0.2*(n_ebrake/max(1,n_veh)))

    M = {
        'tag': args.tag, 'checkpoint': args.checkpoint,
        'Travel Time Ratio (vs Free Flow)': (np.mean(travel_times)/FREE_FLOW_S) if travel_times else None,
        'Average Waiting Time (s)': float(np.mean(waits)) if waits else 0.0,
        'Max Waiting Time (s)': float(np.max(waits)) if waits else 0.0,
        'Average Speed (km/h)': float(avg_speed_kmh),
        'Collision Rate': float(collisions/max(1,n_veh)),
        'Collisions (total)': int(collisions),
        'Safety Score (0-10)': float(safety_score),
        'Emergency Braking Events': n_ebrake,
        'Near-Miss Events': n_nearmiss,
        'Lane Changes (total)': n_lanechange,
        'Lane Changes per veh': float(n_lanechange/max(1,n_veh)),
        'Vehicles that changed lane': n_lc_vehicles,
        'Lane Changes toward lane0 (down)': lc_dir['down'],
        'Lane Changes toward outer (up)': lc_dir['up'],
        'Intersection Throughput (veh/h)': float(throughput),
        'Capacity Utilization': float(min(1.0, throughput/CAPACITY_BASELINE)),
        'Average Queue Length': float(np.mean(queue_per_step)) if queue_per_step else 0.0,
        'Fairness Index (0-1)': jain_fairness(waits),
        'Fuel Savings (% MODEL-EST)': float(fuel_savings),
        'CO2 Reduction (% MODEL-EST)': float(co2_reduction),
        'Completions': n_comp,
        'Total Vehicles Evaluated': n_veh,
        'Evaluation Time (s)': round(time.time()-t_start, 1),
        '_thresholds': {'WAIT_SPEED': WAIT_SPEED, 'EBRAKE_DECEL': EBRAKE_DECEL,
                        'NEARMISS_M': NEARMISS_M, 'FREE_FLOW_S': FREE_FLOW_S},
    }
    json.dump(M, open(args.out, 'w'), indent=2)
    print(f"[{args.tag}] comp={n_comp} col={collisions} avgspd={avg_speed_kmh:.1f}km/h "
          f"wait={M['Average Waiting Time (s)']:.1f}s ebrake={n_ebrake} nearmiss={n_nearmiss} "
          f"lc={n_lanechange}(v{n_lc_vehicles}) lcdir={lc_dir['down']}d/{lc_dir['up']}u thr={throughput:.0f}veh/h ttr={M['Travel Time Ratio (vs Free Flow)']}")


if __name__ == '__main__':
    main()
