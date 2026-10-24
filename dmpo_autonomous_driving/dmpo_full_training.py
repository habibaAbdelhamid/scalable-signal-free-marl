#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full DMPO Neural Network Training for Autonomous Driving

Complete integration with Networked-MB-MARL DMPO neural networks,
training buffers, and learning algorithms.
"""

import os
import sys
import argparse
import torch
import numpy as np
import importlib.util
from typing import Dict, Any
import time
import json
from collections import deque

# Networked-MB-MARL framework location: DMPO_FRAMEWORK_PATH, default <repo>/third_party/Networked-MB-MARL.
# Put it on sys.path before the first `algorithms` import below.
_FRAMEWORK_PATH = os.environ.get('DMPO_FRAMEWORK_PATH', os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'third_party', 'Networked-MB-MARL'))
if _FRAMEWORK_PATH not in sys.path:
    sys.path.insert(0, _FRAMEWORK_PATH)
from algorithms.algo.normalization_utils import ZFilter

def load_networked_mb_marl_modules():
    """Attempt to load DMPO modules - gracefully falls back if not available."""
    print("🔧 Attempting to load Networked-MB-MARL DMPO modules...")

    try:
        # Add Networked-MB-MARL to Python path
        networked_mb_marl_path = _FRAMEWORK_PATH
        if networked_mb_marl_path not in sys.path:
            sys.path.insert(0, networked_mb_marl_path)

        # Try to import from external Networked-MB-MARL framework if available
        # This allows full DMPO functionality if user has the original framework
        from algorithms.algo.agent.DPPO import DPPOAgent
        from algorithms.algo.buffer import MultiCollect, TrajectoryBuffer, ModelBuffer, Trajectory
        from algorithms.algo.agent.DMPO import DMPOAgent, ModelBasedAgent
        from algorithms.models import GraphConvolutionalModel, MLP

        print("✅ Full DMPO modules loaded - advanced training available")

        return {
            'DMPOAgent': DMPOAgent,
            'DPPOAgent': DPPOAgent,
            'ModelBasedAgent': ModelBasedAgent,
            'MultiCollect': MultiCollect,
            'TrajectoryBuffer': TrajectoryBuffer,
            'ModelBuffer': ModelBuffer,
            'Trajectory': Trajectory,
            'GraphConvolutionalModel': GraphConvolutionalModel,
            'MLP': MLP
        }

    except ImportError:
        print("⚠️ Networked-MB-MARL modules not available")
        print("Will use simplified standalone training (still fully functional)")
        return None

from dmpo_autonomous_driving.envs.sumo_intersection_env import create_intersection_env
from dmpo_autonomous_driving.envs.observation_processor import ObservationProcessor
# MATCHED MODEL LOSS helpers (pure torch; gated DMPO_MODEL_MATCHED_LOSS at every call site)
from dmpo_autonomous_driving.utils import matched_model_loss as _mml
from dmpo_autonomous_driving.config.intersection_dmpo import (
    create_intersection_dmpo_config,
    EntropyScheduler,
    get_entropy_coefficient
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Full DMPO Neural Network Training')
    
    # Environment arguments
    parser.add_argument('--config-path', type=str, default=None,
                       help='SUMO configuration file path')
    parser.add_argument('--max-steps', type=int, default=300,
                       help='Maximum steps per episode')
    parser.add_argument('--max-ai-vehicles', type=int, default=0,
                       help='Cap AI vehicle count (0=no cap)')
    parser.add_argument('--n-agents', type=int, default=0,
                       help='Policy network capacity (number of agent slots). '
                            'If 0, inferred from --max-ai-vehicles or scenario. '
                            'Set this >= max concurrent AI vehicles in scenario.')
    parser.add_argument('--use-gui', action='store_true',
                       help='Use SUMO GUI for visualization')
    parser.add_argument('--enable-external-eval', action='store_true',
                       help='Enable external TraCI connections for real-time evaluation')
    
    # Training arguments  
    parser.add_argument('--n-iterations', type=int, default=1000,
                       help='Number of DMPO training iterations')
    parser.add_argument('--n-episodes', type=int, default=5,
                       help='Episodes per iteration for experience collection')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Training device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # DMPO specific arguments
    parser.add_argument('--kappa', type=int, default=2,
                       help='κ-hop communication radius')
    parser.add_argument('--communication-radius', type=float, default=100.0,
                       help='Physical communication radius in meters (default: 100.0)')
    parser.add_argument('--model-rollout-length', type=int, default=25,
                       help='Model rollout steps per trajectory (original: 25)')
    parser.add_argument('--model-rollout-batch', type=int, default=256,
                       help='Number of initial states to sample for model rollouts (original: 128-512)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate (original: 1e-4, same for all networks)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Training batch size (currently unused — PPO uses n_minibatch=4 from config)')
    
    # Logging and checkpoints
    parser.add_argument('--log-dir', type=str, default='logs/dmpo_full_training',
                       help='Directory for logs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/dmpo_full',
                       help='Directory for checkpoints')
    parser.add_argument('--save-interval', type=int, default=50,
                       help='Save checkpoint every N iterations')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Log progress every N iterations')

    # Resume training
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint')
    parser.add_argument('--resume-from', type=int, default=None,
                       help='Resume from specific iteration checkpoint')

    # Entropy scheduling for exploration vs exploitation
    # H12 CLEANUP: --entropy-schedule is a DEAD no-op (EntropyScheduler is constructed with
    # schedule='constant' hardcoded; this arg was never read). Kept ONLY so old scripts don't
    # crash; hidden from --help. Entropy is controlled explicitly via DMPO_ENTROPY
    # (pinned in configs/training_block.sh).
    parser.add_argument('--entropy-schedule', type=str, default='constant',
                        help=argparse.SUPPRESS)

    # Model training step overrides (for scaling by scenario size)
    parser.add_argument('--n-model-update', type=int, default=2000,
                       help='Gradient steps per model update (original: 2000)')
    parser.add_argument('--n-model-update-warmup', type=int, default=20000,
                       help='Gradient steps for post-warmup model pre-training (original: 20000)')

    return parser.parse_args()


class IntersectionDMPOTrainer:
    """Full DMPO trainer for intersection autonomous driving."""
    
    def __init__(self, args, env, dmpo_modules):
        self.args = args
        self.env = env
        self.modules = dmpo_modules
        self.device = torch.device(args.device)

        # Create DMPO configuration
        self.config = create_intersection_dmpo_config(args.config_path)
        self._update_config_from_args()
        
        # Initialize DMPO components
        self.agent = None
        self.trajectory_buffer = None
        self.model_buffer = None  # Initialized after DMPO modules are loaded
        self.multi_collect = None
        
        # Training metrics
        self.training_metrics = {
            'iteration_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'model_losses': [],
            'safety_metrics': [],
            'learning_curves': [],
            'entropy_coefficients': []  # Track entropy schedule
        }

        # Resume state
        self.start_iteration = 0

        # Initialize entropy scheduler
        # DMPO_ENTROPY env var overrides. Default 0.05 for continuous actions
        # (must exceed surrogate gradient ~0.05 to prevent std collapse).
        # Original framework uses 0.0 (power grid, discrete actions — std collapse N/A).
        _ent_env = os.environ.get('DMPO_ENTROPY', '')
        if _ent_env:
            _ent_coef = float(_ent_env)
        elif os.environ.get('DMPO_REWARD_TRIAL', '0') == '1':
            _ent_coef = 0.01
        else:
            _ent_coef = 0.05
        self.entropy_scheduler = EntropyScheduler(
            total_iterations=args.n_iterations,
            schedule='constant',
            start_coef=_ent_coef,
            end_coef=_ent_coef
        )
        print(f"📊 Entropy: constant {_ent_coef}")

        self._initialize_dmpo_components()

        # ZFilter: observation normalization matching original framework
        # Shape: (n_agent, obs_dim) — normalizes per-agent, per-feature
        # OBS-V2 (C3): with DMPO_OBS_V2=1 (and the DMPO_OBS_ZFILTER=0 default) there is
        # NO z-normalization anywhere — every v2 slot is bounded at construction, and
        # the ±5 z-clip measurably destroyed the braking signal beyond ≈5.5 m/s².
        self._obs_v2_nonorm = (os.environ.get('DMPO_OBS_V2', '0') == '1'
                               and os.environ.get('DMPO_OBS_ZFILTER', '0') != '1')
        n_agent = self.config.agent_args.n_agent
        obs_dim = self.config.agent_args.observation_dim
        if self._obs_v2_nonorm:
            self.running_state = None
            print(f"📊 OBS-V2: fixed-normalization obs ({obs_dim} dims) — ZFilter/z-clip DISABLED")
        else:
            self.running_state = ZFilter((n_agent, obs_dim), clip=5.0)
            print(f"📊 ZFilter initialized: shape=({n_agent}, {obs_dim}), clip=5.0")

        # ── MATCHED MODEL LOSS (gated DMPO_MODEL_MATCHED_LOSS, train-only) ──
        # Identity-matched world-model loss for the nearest-first neighbour block:
        # each reference slot is supervised against the SAME vehicle's next row
        # (slot→id lists captured by the processor), left/entered slots masked,
        # rollout states re-sorted to canonical nearest-first. Default '0' =
        # byte-identical legacy loss (no ids captured, no extra tensors).
        # Geometry is read from the LIVE processor — never restated as literals.
        self._matched_on = os.environ.get('DMPO_MODEL_MATCHED_LOSS', '0') == '1'
        self._nb_geom = None
        self._vid_intern = {}          # str vehicle id → int32 (reset per _prepare_dmpo_batch_dynamic call)
        if self._matched_on:
            # money guard: hard errors, never a silent no-op
            if not self._obs_v2_nonorm:
                raise RuntimeError(
                    "DMPO_MODEL_MATCHED_LOSS=1 requires DMPO_OBS_V2=1 and DMPO_OBS_ZFILTER!=1 "
                    "(re-sort / empty detector operate on RAW bounded v2 features)")
            if os.environ.get('DMPO_SHARED_GCN', '0') == '1':
                raise RuntimeError(
                    "DMPO_MODEL_MATCHED_LOSS=1 is only instrumented for the ensemble-dynamic "
                    "model loss (DMPO_SHARED_GCN must be 0)")
            if os.environ.get('DMPO_DYNAMIC', '0') != '1':
                raise RuntimeError("DMPO_MODEL_MATCHED_LOSS=1 requires DMPO_DYNAMIC=1")
            _op = self.env.observation_processor
            if not getattr(_op, '_export_nb_ids', False):
                raise RuntimeError("DMPO_MODEL_MATCHED_LOSS=1 but the ObservationProcessor did "
                                   "not enable nb-id export (flag must be set before env creation)")
            _neutral_row = np.asarray(_op._zero_obs_neutral()['neighbors'][0], dtype=np.float32)
            # RE-SORT KEY (review fix): the processor orders slots by FRONT-bumper
            # distance (_get_neighbor_states sorts on the cached SUMO 'position' =
            # front bumper) but stores the REAR-bumper offset in nb feats 0/1
            # (_nb_v2: rx = pn − Ln·sin an, ry = pn − Ln·cos an, then rotated to the
            # ego frame; feats 4/5 = (sin Δh, cos Δh)), normalised by R =
            # obs_dist_norm (= max(communication_radius, 1) = DMPO_PERCEPTION_RADIUS).
            # Front bumper (normalised) = feat0/1 + (Ln/R)·feat4/5. _nb_v2 uses a
            # PER-VEHICLE cached length (fallback 4.5), which the rollout state does
            # not carry, so a scenario-nominal length is used: DMPO_VEHICLE_LENGTH_M
            # if set, else the SUMO vType default 5.0 (NOTE: our generated scenarios'
            # vType length is 4.5 — export DMPO_VEHICLE_LENGTH_M=4.5 to match).
            _R = float(_op.obs_dist_norm)
            _L_nominal = float(os.environ.get('DMPO_VEHICLE_LENGTH_M', '5.0'))
            self._nb_geom = dict(
                off=int(_op.V2_EGO_DIM + _op.V2_INFRA_DIM),
                N=int(_op.max_neighbors),
                F=int(_op.V2_NB_FEAT),
                neutral=torch.as_tensor(_neutral_row, dtype=torch.float32, device=self.device),
                front_lam=_L_nominal / max(_R, 1e-6),
            )
            assert self._nb_geom['off'] + self._nb_geom['N'] * self._nb_geom['F'] <= obs_dim, \
                f"matched-loss nb block exceeds obs_dim: {self._nb_geom} vs {obs_dim}"
            assert len(_neutral_row) == self._nb_geom['F']
            print(f"📋 Matched model loss: ON nb_off={self._nb_geom['off']} N={self._nb_geom['N']} "
                  f"F={self._nb_geom['F']} obs_dim={obs_dim} "
                  f"keep_empty={os.environ.get('DMPO_MODEL_MATCHED_KEEP_EMPTY', '1')} "
                  f"neutral={_neutral_row.tolist()} | chain ref=sid[0]; rollout s1 re-sorted "
                  f"nearest-first (front-bumper key: front_lam=L/R={_L_nominal}/{_R}="
                  f"{self._nb_geom['front_lam']:.5f}) + empties snapped to neutral; "
                  f"rel-err NOT comparable to flag-off")

        # ───────────────────────────────────────────────────────────────────
        # DMPO-Safe v2: Initialize safety modules (BC anchor, safety shield)
        # All gated by env vars; default behavior unchanged when off.
        # ───────────────────────────────────────────────────────────────────
        self._safe_enabled = os.environ.get('DMPO_SAFE', '0') == '1'
        if self._safe_enabled:
            try:
                from dmpo_autonomous_driving.safety import BCAnchorBuffer, SafetyShield
                from dmpo_autonomous_driving.safety.safety_shield import ShieldConfig

                # BC anchor buffer
                bc_capacity = int(os.environ.get('DMPO_SAFE_BC_CAPACITY', '100'))
                self.bc_anchor = BCAnchorBuffer(
                    max_episodes=bc_capacity,
                    observation_dim=obs_dim,
                    action_dim=4,  # yield_intent variant
                )
                self.bc_loss_weight = float(os.environ.get('DMPO_SAFE_BC_WEIGHT', '0.1'))
                self.bc_batch_size = int(os.environ.get('DMPO_SAFE_BC_BATCH', '64'))

                # Safety shield
                shield_cfg = ShieldConfig(
                    rollout_horizon=int(os.environ.get('DMPO_SAFE_SHIELD_H', '10')),
                    activation_iter=int(os.environ.get('DMPO_SAFE_SHIELD_FROM_ITER', '100')),
                    cost_threshold=float(os.environ.get('DMPO_SAFE_SHIELD_COST_THRESH', '0.5')),
                    uncertainty_threshold=float(os.environ.get('DMPO_SAFE_SHIELD_UNC_THRESH', '1.0')),
                )
                self.safety_shield = SafetyShield(shield_cfg)
                print(f"🛡  DMPO-Safe enabled: BC capacity={bc_capacity}, "
                      f"BC weight={self.bc_loss_weight}, shield h={shield_cfg.rollout_horizon}, "
                      f"shield-from-iter={shield_cfg.activation_iter}")
            except Exception as e:
                print(f"⚠️  DMPO-Safe init failed: {e}; falling back to vanilla DMPO")
                self._safe_enabled = False
        else:
            self.bc_anchor = None
            self.safety_shield = None

        # Store resume request - checkpoint will be loaded AFTER agent reinitialization
        # This is critical because n_agent may change after first episode
        self._pending_resume = None
        if args.resume or args.resume_from is not None:
            self._pending_resume = args.resume_from
            print(f"📌 Resume requested - checkpoint will load after agent detection")
    
    def _update_config_from_args(self):
        """Update configuration with command line arguments."""
        self.config.n_iter = self.args.n_iterations
        self.config.model_traj_length = self.args.model_rollout_length
        self.config.agent_args.radius_v = self.args.kappa
        self.config.agent_args.radius_pi = self.args.kappa
        self.config.agent_args.radius_p = self.args.kappa

        # Learning rates: same for all networks (matching original framework)
        self.config.agent_args.lr = self.args.learning_rate
        self.config.agent_args.lr_v = self.args.learning_rate
        self.config.agent_args.lr_p = self.args.learning_rate
        self.config.model_batch_size = self.args.model_rollout_batch

        # Model-based settings (matching original framework).
        # model_prob and reward_coeff are configurable via env vars so we can
        # tune MBPO aggressiveness without editing code.
        self.config.model_prob = float(os.environ.get('DMPO_MODEL_PROB', '0.5'))
        self.config.model_based = True
        self.config.n_inner_iter = 10
        # n_warmup: collect-only iterations before the world model is pre-trained.
        # Default 50 (matches original). Env-tunable (DMPO_N_WARMUP) so a short
        # forensic smoke run can exercise the model path quickly without editing code.
        self.config.n_warmup = int(os.environ.get('DMPO_N_WARMUP', '50'))

        # DMPO-Safe: model-usage curriculum.
        # DMPO_USE_MODEL_FROM_ITER=3000 means: train model from iter 0 but only
        # USE it for rollouts starting iter 3000 (when model is mature).
        # DMPO_MODEL_PROB_SCHEDULE='3000:0.10,4000:0.20' overrides static model_prob
        # with a per-iteration schedule (becomes 10% at iter 3000, 20% at iter 4000).
        self.config.model_use_from_iter = int(os.environ.get('DMPO_USE_MODEL_FROM_ITER', '0'))
        _sched_str = os.environ.get('DMPO_MODEL_PROB_SCHEDULE', '')
        if _sched_str:
            self.config.model_prob_schedule = []
            for part in _sched_str.split(','):
                it_str, prob_str = part.split(':')
                self.config.model_prob_schedule.append((int(it_str), float(prob_str)))
        else:
            self.config.model_prob_schedule = None
        self.config.agent_args.grad_clip = 0.5  # Safety: not in original but prevents rare gradient spikes
        # Optional reward_coeff override (world-model reward-loss weight).
        _rc_env = os.environ.get('DMPO_REWARD_COEFF', '')
        if _rc_env:
            self.config.agent_args.p_args.reward_coeff = float(_rc_env)

        # Dynamic tensor mode: variable agent count, no padding, parameter sharing
        self.config.agent_args.dynamic = os.environ.get('DMPO_DYNAMIC', '0') == '1'

        # Entropy coefficient: DMPO_ENTROPY env var takes priority, then default 0.05
        # 0.05 balances surrogate gradient (~0.05) to prevent std collapse
        _ent_env = os.environ.get('DMPO_ENTROPY', '')
        if _ent_env:
            self.config.agent_args.entropy_coeff = float(_ent_env)
        else:
            self.config.agent_args.entropy_coeff = 0.05

        # Model training steps (CLI values)
        self.config.n_model_update = int(self.args.n_model_update)
        self.config.n_model_update_warmup = int(self.args.n_model_update_warmup)

        # Minibatch override (env var) — needed for large n_episodes so the
        # full-batch PPO backward fits the GPU. Default 0 = keep config (1=full-batch).
        _nmb = int(os.environ.get('DMPO_N_MINIBATCH', '0'))
        if _nmb > 0:
            self.config.agent_args.n_minibatch = _nmb
            print(f"  ⚙️  n_minibatch overridden to {_nmb} (DMPO_N_MINIBATCH) — splits the PPO update to fit GPU")

        # Print complete effective config for verification
        print(f"📋 Effective config:")
        print(f"  DMPO: model_prob={self.config.model_prob}, n_inner_iter={self.config.n_inner_iter}, n_warmup={self.config.n_warmup}")
        print(f"  Model: rollout_batch={self.config.model_batch_size}, rollout_length={self.config.model_traj_length}, n_update={self.config.n_model_update}, n_warmup_update={self.config.n_model_update_warmup}")
        print(f"  LR: policy={self.config.agent_args.lr}, value={self.config.agent_args.lr_v}, model={self.config.agent_args.lr_p}")
        print(f"  PPO: clip={self.config.agent_args.clip}, grad_clip={self.config.agent_args.grad_clip}, n_update_pi={self.config.agent_args.n_update_pi}, n_minibatch={self.config.agent_args.n_minibatch}")
        print(f"  GAE: lamda={self.config.agent_args.lamda}, use_gae_returns={self.config.agent_args.use_gae_returns}, entropy={self.config.agent_args.entropy_coeff}")
        print(f"  World model: reward_coeff={self.config.agent_args.p_args.reward_coeff}, GCN={self.config.agent_args.p_args.node_hidden_size}")
        print(f"  Env: max_steps={self.args.max_steps}, max_ai_vehicles={self.args.max_ai_vehicles}, kappa={self.args.kappa}")
        if self.config.agent_args.dynamic:
            print(f"  🔥 DYNAMIC TENSORS: shared policy/value, per-vehicle trajectories, no padding")
        # MATCHED MODEL LOSS flag (geometry line is printed by the __init__ block that
        # resolves it from the live processor — this dump runs before that block).
        if os.environ.get('DMPO_MODEL_MATCHED_LOSS', '0') == '1':
            print(f"  Matched model loss: ON (DMPO_MODEL_MATCHED_LOSS=1, "
                  f"keep_empty={os.environ.get('DMPO_MODEL_MATCHED_KEEP_EMPTY', '1')}) — "
                  f"model rel-err values NOT comparable to flag-off logs")
            if (float(self.config.model_prob) == 0.0
                    and getattr(self.config, 'model_prob_schedule', None) is None
                    and os.environ.get('DMPO_MODEL_TRAIN_GATE', '0') == '1'):
                print("  ⚠️ DMPO_MODEL_MATCHED_LOSS=1 is INERT: DMPO_MODEL_PROB=0 + "
                      "DMPO_MODEL_TRAIN_GATE=1 skip world-model training entirely")
        else:
            print(f"  Matched model loss: OFF")
        # Report reward function config
        import os as _os
        phase2 = _os.environ.get('DMPO_PHASE2', '0') == '1'
        trial = _os.environ.get('DMPO_REWARD_TRIAL', '0') == '1'
        progress_first = _os.environ.get('DMPO_PROGRESS_FIRST', '0') == '1'
        weights_str = "safety=30%,progress=40%" if progress_first else "safety=40%,progress=30%"
        # EFFECTIVE event penalties (CaRL+/sparse inject these via env.step, NOT the weighted-path
        # hardcoded −50/−20 which is only used by the legacy 4-component reward).
        _evt_col = _os.environ.get('DMPO_COLLISION_PENALTY', '200')
        _evt_comp = _os.environ.get('DMPO_COMPLETION_BONUS', '200')
        print(f"  Reward: {weights_str}, comfort=20%, cooperation=10%, "
              f"weighted_path_collision={'−20' if phase2 else '−50'} (legacy 4-comp only), "
              f"EVENT penalties (CaRL+/env-injected): collision=−{_evt_col}, completion=+{_evt_comp}, "
              f"speed_bonus=yes, delay_justified_fix=yes")
        # ---- REWARD V2 (Stage-3 rebalance): flags + derived exchange rate + money-guard.
        # TRAINING-ENTRY ONLY (verifier C3): the guard must NEVER live default-ON in the
        # env/reward computer — bare eval (COMPLETION_BONUS=200, penalty default) would
        # trip it and brick the whole eval/forensic toolchain. Default ON only under
        # DMPO_REWARD_V2=1; otherwise opt-in via DMPO_REWARD_EXCHANGE_ASSERT=1.
        _rv2 = _os.environ.get('DMPO_REWARD_V2', '0') == '1'
        _comp_f, _col_f = float(_evt_comp), float(_evt_col)
        _ratio = (_col_f / _comp_f) if _comp_f > 0 else float('inf')
        _dt = float(_os.environ.get('DMPO_STEP_LENGTH', '0.2'))
        _L = float(_os.environ.get('DMPO_ROUTE_LENGTH_M', '380.0'))
        _vmax = float(_os.environ.get('DMPO_MAX_SPEED_MPS', '16.67'))
        _gamma = float(getattr(self.config.agent_args, 'gamma', 0.99))
        _move_b = float(_os.environ.get('DMPO_MOVEMENT_BONUS', '0.0'))
        _team_b = float(_os.environ.get('DMPO_TEAM_BONUS', '5.0'))
        _team_c = float(_os.environ.get('DMPO_TEAM_COST', '5.0'))
        _remove_ai = _os.environ.get('DMPO_COLLISION_REMOVE_AI', '0') == '1'
        _prog_step = (_vmax * _dt / _L) * 10.0 if _L > 0 else 0.0   # dense progress/step @ v_max
        # Modeling assumptions (documented, not regime-derived): a junction collision/
        # yield decision happens ~25 steps (5 s) before completion; a yield delays 15
        # steps (3 s). TRUE historical regime (r4/r5/r6 run logs) = completion 200 /
        # collision 400, ratio 2.0 — NOT the +400/−200 inversion of the aborted launcher.
        _k_goal, _d_yield = 25, 15
        _g_goal = _gamma ** _k_goal
        # REMOVE_AI=1: the collider also forfeits its discounted completion (reward design)
        _c_eff = _col_f + _g_goal * _comp_f if _remove_ai else _col_f
        _yield_cost = _comp_f * _g_goal * (1.0 - _gamma ** _d_yield) + _move_b * _d_yield
        _be_a = (_yield_cost / _c_eff) if _c_eff > 0 else float('inf')       # completes either way
        _gain_b = _comp_f * _g_goal
        _be_b = _gain_b / (_gain_b + _c_eff) if (_gain_b + _c_eff) > 0 else float('inf')  # horizon bites
        _deg = 8.0  # measured-typical κ-hop degree assumption (dense 4-lane); re-measure per §6 smoke
        print(f"  REWARD V2: umbrella={'ON' if _rv2 else 'OFF (legacy byte-identical)'}"
              f" ramp={_os.environ.get('DMPO_MOVEMENT_RAMP', '1') if _rv2 else 'n/a'}"
              f"[{_os.environ.get('DMPO_MOVEMENT_RAMP_LO', '0.3')},{_os.environ.get('DMPO_MOVEMENT_RAMP_HI', '0.7')}]"
              f" trapezoid={_os.environ.get('DMPO_PROGRESS_TRAPEZOID', '1') if _rv2 else 'n/a'}"
              f" idle_exempt(conflict={_os.environ.get('DMPO_IDLE_YIELD_EXEMPT', '0')}"
              f"@PET<{_os.environ.get('DMPO_IDLE_EXEMPT_PET_S', '1.5')}s"
              f" shield={_os.environ.get('DMPO_IDLE_EXEMPT_SHIELD', '0')}"
              f" stuck={_os.environ.get('DMPO_IDLE_EXEMPT_STUCK', '0')}"
              f" cap={_os.environ.get('DMPO_IDLE_EXEMPT_MAX_STEPS', '50')})")
        print(f"  EXCHANGE RATE (derived; γ={_gamma:g}, dt={_dt:g}s, L={_L:g}m, v_max={_vmax:g}m/s):")
        print(f"    progress +{_prog_step:.4f}/step @v_max → +10.00/trip | movement +{_move_b:.4f}/step"
              f" → bonus/progress={_move_b / _prog_step if _prog_step > 0 else float('inf'):.2f} (must be <1)")
        print(f"    completion +{_comp_f:g}  collision −{_col_f:g}  ratio={_ratio:.2f}"
              f"  terminate_on_collision={'YES' if _remove_ai else 'NO (STUCK)'}"
              f"  effective_collider_loss=−{_c_eff:.1f} (assumes {_k_goal} steps to goal)")
        print(f"    team bonus +{_team_b:g}×deg cost −{_team_c:g}×deg (deg≈{_deg:g} → "
              f"{100.0 * _deg * _team_b / _comp_f if _comp_f > 0 else 0:.1f}% of completion / "
              f"{100.0 * _deg * _team_c / _col_f if _col_f > 0 else 0:.1f}% of collision)")
        print(f"    break-even p(collision): yield-3s-completes-anyway={100.0 * _be_a:.1f}%"
              f"  yield-forfeits-trip={100.0 * _be_b:.1f}%"
              f"  [historical r4/r5/r6 regime: completion 200 / collision 400, ratio 2.00]")
        _exch_on = _os.environ.get('DMPO_REWARD_EXCHANGE_ASSERT', '1' if _rv2 else '0') == '1'
        _min_ratio = float(_os.environ.get('DMPO_REWARD_MIN_RATIO', '2.0'))
        if _exch_on:
            if _col_f < _min_ratio * _comp_f:
                raise RuntimeError(
                    f"REWARD EXCHANGE-RATE GUARD (money guard): DMPO_COLLISION_PENALTY={_col_f:g} < "
                    f"DMPO_REWARD_MIN_RATIO={_min_ratio:g} × DMPO_COMPLETION_BONUS={_comp_f:g} "
                    f"(ratio {_ratio:.2f}). Inverted/mis-ratioed reward profile refused at the "
                    f"training entry. TRUE historical regime = completion 200 / collision 400 "
                    f"(ratio 2.0); the +400/−200 pair is the aborted-launcher inversion. "
                    f"Set DMPO_REWARD_EXCHANGE_ASSERT=0 only for a DELIBERATE legacy reproduction.")
            print(f"    ASSERT collision/completion = {_ratio:.2f} ≥ {_min_ratio:.2f} : OK")
        if trial:
            print(f"  TRIAL: simplified 3-term reward (speed + crossing_bonus=50 - collision=-20), entropy_coeff=0.01")
        literature = _os.environ.get('DMPO_REWARD_LITERATURE', '0') == '1'
        if literature:
            col_pen = _os.environ.get('DMPO_COLLISION_PENALTY', '5')
            print(f"  LITERATURE: 3-term reward (speed/v_max + completion=+10 - collision=-{col_pen}), ratio 1:{float(col_pen)/10:.0f}" if float(col_pen) != 0 else f"  LITERATURE: 2-term reward (speed/v_max + completion=+10, no collision penalty)")
        carl = _os.environ.get('DMPO_REWARD_CARL', '0') == '1'
        if carl:
            print(f"  CARL: Δroute_progress*10 + 0.01*centering, collision=multiplicative(zeroes progress), action_dim=2")
        sparse = _os.environ.get('DMPO_REWARD_SPARSE', '0') == '1'
        if sparse:
            comp_bonus = _os.environ.get('DMPO_COMPLETION_BONUS', '50')
            col_pen = _os.environ.get('DMPO_SPARSE_COLLISION_PENALTY', '5')
            print(f"  SPARSE: 0/step, +{comp_bonus} on completion, -{col_pen} on collision (Path F — basin-escape experiment)")
        carl_plus = _os.environ.get('DMPO_REWARD_CARL_PLUS', '0') == '1'
        if carl_plus:
            comp_bonus = _os.environ.get('DMPO_COMPLETION_BONUS', '200')
            print(f"  CARL_PLUS: CaRL dense (Δprogress*10 + 0.01*centering) + {comp_bonus} on completion (Path G — dense gradient + dominant goal signal)")
        accountable = _os.environ.get('DMPO_REWARD_ACCOUNTABLE', '0') == '1'
        if accountable:
            print(f"  ACCOUNTABLE: speed/v_max + 0.2*centering + completion=+10 - collision=-5, action_dim=2")
        rebalanced = _os.environ.get('DMPO_REWARD_REBALANCED', '0') == '1'
        if rebalanced:
            print(f"  REBALANCED: 4-component, collision=-10, completion=+20, speed*3, comfort halved")
        normalized = _os.environ.get('DMPO_REWARD_NORMALIZED', '0') == '1'
        if normalized:
            print(f"  NORMALIZED: non-negative [0,1] reward (collision=0, stopped=0.71, arrived=1.0)")

    def _patch_agent_value_loss(self):
        """Monkey-patch DPPO.updateAgent to fix value loss explosion.

        In dynamic mode: skip patching — no padding means no mask issues.
        The original DPPO.updateAgent works correctly with per-vehicle
        trajectories (n_agent=1, no padded agents).

        Three bugs in the original DPPO.updateAgent (fixed-mode only):
        1. Value loss ignores agent masks — padded agents (zero obs, zero rewards)
           contribute garbage gradients that corrupt the value network.
        2. Returns are unbounded — with rewards in [-50,+50] and gamma=0.99 over
           T=300 steps, cumulative returns reach thousands, while V(s) outputs ~O(1).
           MSE between them explodes.
        3. GAE bootstraps through padded agents if masks aren't propagated.

        Fix: Wrap updateAgent to (a) apply agent masks to value loss, and
        (b) normalize returns to prevent magnitude explosion.
        """
        if getattr(self.agent, 'dynamic', False):
            print("  - Dynamic mode: skipping updateAgent patch (no padding, no masks needed)")
            return
        if os.environ.get('DMPO_PAD_FIX', '0') == '1':
            # MATCHED PADDED ARM (gated DMPO_PAD_FIX=1, default 0 = legacy; padded learning
            # baseline): the legacy padded update below differs from the
            # verified dynamic learner in KL masking (empty slots inflate |KL| → PPO stops
            # at epoch 0 every iteration), critic targets (z-normed vs raw), advantage
            # normalisation and the logit penalty. With the flag, padded runs use the
            # FRAMEWORK update + GAE (DPPO.updateAgent / _process_traj) with the slot masks,
            # i.e. exactly the dynamic arms' masked learner. Neither legacy patch is installed.
            print("  - PAD-FIX: padded mode uses the framework's masked learner "
                  "(legacy updateAgent and _process_traj patches NOT installed)")
            return

        import types
        import torch.nn as nn

        agent = self.agent
        original_updateAgent = agent.updateAgent.__func__  # unbound method

        def patched_updateAgent(self_agent, trajs, clip=None):
            """Patched updateAgent with masked value loss and return normalization."""
            import time as _time
            from algorithms.algo.buffer import Trajectory

            time_t = _time.time()
            if clip is None:
                clip = self_agent.clip
            n_minibatch = self_agent.n_minibatch

            names = Trajectory.names()
            traj_all = {name: [] for name in names}

            # Extract agent masks
            has_masks = any(hasattr(t, 'has_mask') and t.has_mask() for t in trajs)
            mask_all = [] if has_masks else None

            max_traj_length = max([i.length for i in trajs])
            for traj in trajs:
                for name in names:
                    tensor_shape = traj[name].shape
                    full_part_shape = [max_traj_length - tensor_shape[0]] + list(tensor_shape[1:])
                    if name == 'd':
                        traj_all[name].append(torch.cat([traj[name], torch.ones(full_part_shape, dtype=torch.bool, device=self_agent.device)], dim=0))
                    else:
                        traj_all[name].append(torch.cat([traj[name], torch.zeros(full_part_shape, dtype=traj[name].dtype, device=self_agent.device)], dim=0))

                if has_masks:
                    if hasattr(traj, 'has_mask') and traj.has_mask():
                        mask = traj['mask']
                    else:
                        mask = torch.ones(traj['s'].shape[:-1], dtype=torch.float32, device=self_agent.device)
                    mask_shape = mask.shape
                    full_mask_shape = [max_traj_length - mask_shape[0]] + list(mask_shape[1:])
                    mask_all.append(torch.cat([mask, torch.zeros(full_mask_shape, dtype=torch.float32, device=self_agent.device)], dim=0))

            traj = {name: torch.stack(value, dim=0) for name, value in traj_all.items()}
            agent_mask = torch.stack(mask_all, dim=0) if has_masks else None

            # === BUG 1 FIX: Pass agent_mask into _process_traj ===
            # Previously, agent_mask was extracted at line 303 but never added back
            # to `traj`, so _process_traj always received mask=None. This made ALL
            # padded-agent masking in _process_traj dead code — padded agents'
            # garbage V(zeros) predictions contaminated real agents' GAE advantages.
            traj_for_gae = dict(traj)
            if agent_mask is not None:
                traj_for_gae['mask'] = agent_mask

            # H5b mirror (learner-fixes spec): kl_all scoped OVER the epoch loop so
            # the returned max(kl_all) is the max across epochs (was: last epoch
            # only); DMPO_KL_EARLY_STOP=1 freezes further POLICY epochs once
            # |KL| > 1.5*target_kl while the value net keeps training (SpinningUp
            # semantics). Default '0' = legacy control flow, byte-identical.
            kl_all = []
            _kl_early_stop = os.environ.get('DMPO_KL_EARLY_STOP', '0') == '1'
            pi_frozen = False

            for i_update in range(self_agent.n_update_pi):
                s, a, r, s1, d, logp = traj['s'], traj['a'], traj['r'], traj['s1'], traj['d'], traj['logp']
                s, a, r, s1, d, logp = [item.to(self_agent.device) for item in [s, a, r, s1, d, logp]]
                value_old, returns, advantages, reduced_advantages = self_agent._process_traj(**traj_for_gae)

                # === FIX: Normalize returns to prevent value loss explosion ===
                # Clamp returns first, then normalize with masked stats.
                # Skip normalization when too few active agents (< 4) — the stats
                # from 2 samples are too noisy and cause value loss explosion.
                # Clamp wide enough to admit completion bonuses up to 200; the
                # previous ±100 silently halved CARL_PLUS's terminal signal.
                # DMPO_HUBER_VALUE=1: value loss operates on RAW returns (matching
                # the dynamic arm's live masked-Huber-on-raw path); clamp+z-norm
                # applies only to the legacy MSE regime (audit 2026-08-08, finding 3).
                if os.environ.get('DMPO_HUBER_VALUE', '0') != '1':
                    returns = returns.clamp(-1000, 1000)
                    if agent_mask is not None:
                        mask_4d_ret = agent_mask.to(self_agent.device).unsqueeze(-1)  # [b, T, n, 1]
                        n_valid_ret = mask_4d_ret.sum().clamp(min=1.0)
                        if n_valid_ret > 4 * returns.shape[1]:  # at least 4 active agents on avg
                            ret_mean = (returns * mask_4d_ret).sum() / n_valid_ret
                            ret_std = ((returns - ret_mean) ** 2 * mask_4d_ret).sum() / n_valid_ret
                            ret_std = ret_std.sqrt().clamp(min=1.0)
                            returns = (returns - ret_mean) / ret_std
                        # else: skip normalization — too few samples, use raw clamped returns
                    else:
                        ret_mean = returns.mean()
                        ret_std = returns.std().clamp(min=1.0)
                        returns = (returns - ret_mean) / ret_std

                advantages_old = reduced_advantages if self_agent.use_reduced_v else advantages

                b, T_len, n, d_s = s.size()
                d_a = a.size()[-1]
                s = s.view(-1, n, d_s)
                a = a.view(-1, n, d_a)
                logp = logp.view(-1, n, d_a)
                advantages_old = advantages_old.view(-1, n, 1)
                returns = returns.view(-1, n, 1)
                value_old = value_old.view(-1, n, 1)

                if agent_mask is not None:
                    mask_flat = agent_mask.to(self_agent.device).view(-1, n, 1)
                else:
                    mask_flat = None

                batch_total = logp.size()[0]
                batch_size = int(batch_total / n_minibatch)

                i_pi = 0
                # H5b: once pi_frozen (KL early stop), skip the policy step for the
                # remaining epochs — the value block below still runs every epoch.
                for i_pi in (range(1) if not pi_frozen else range(0)):
                    batch_state, batch_action, batch_logp, batch_advantages_old = [s, a, logp, advantages_old]
                    batch_mask = mask_flat
                    if n_minibatch > 1:
                        idxs = np.random.choice(range(batch_total), size=batch_size, replace=False)
                        [batch_state, batch_action, batch_logp, batch_advantages_old] = [item[idxs] for item in [batch_state, batch_action, batch_logp, batch_advantages_old]]
                        if mask_flat is not None:
                            batch_mask = mask_flat[idxs]

                    batch_logp_new = self_agent.get_logp(batch_state, batch_action)
                    logp_diff = batch_logp_new - batch_logp
                    kl = logp_diff.mean()
                    ratio = torch.exp(batch_logp_new - batch_logp)

                    if batch_mask is not None:
                        mask_expanded = batch_mask.expand_as(ratio)
                        ratio = torch.where(mask_expanded > 0.5, ratio, torch.ones_like(ratio))

                    surr1 = ratio * batch_advantages_old
                    surr2 = ratio.clamp(1 - clip, 1 + clip) * batch_advantages_old
                    loss_surr = torch.min(surr1, surr2).mean()

                    # Compute TRUE policy entropy from the distribution's std,
                    # NOT from log_prob of old actions (which measures action compatibility,
                    # not exploration width). Gaussian entropy = 0.5*ln(2*pi*e*std^2).
                    # As std shrinks, this correctly decreases → SAC alpha increases.
                    true_entropy = torch.zeros(1, device=self_agent.device)
                    if hasattr(self_agent, 'actors'):
                        for actor in self_agent.actors:
                            clamped = torch.clamp(actor.log_std, actor.LOG_STD_MIN, actor.LOG_STD_MAX)
                            std = torch.exp(clamped)
                            actor_entropy = 0.5 * torch.log(2 * np.pi * np.e * std * std).mean()
                            true_entropy = true_entropy + actor_entropy
                        true_entropy = true_entropy / len(self_agent.actors)

                    # Use true entropy for the policy loss (pushes std UP when entropy is low)
                    loss_pi_entropy = -true_entropy

                    # SAC-style automatic entropy tuning (Haarnoja et al. 2018)
                    # Uses TRUE distribution entropy, not log-prob of old actions.
                    if hasattr(self_agent, '_log_alpha'):
                        alpha = self_agent._log_alpha.exp()
                        alpha_loss = (-alpha * (true_entropy - self_agent._target_entropy).detach()).mean()
                        self_agent._alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        self_agent._alpha_optimizer.step()
                        entropy_coeff = self_agent._log_alpha.exp().detach()
                    else:
                        entropy_coeff = self_agent.entropy_coeff

                    loss_pi = -loss_surr + entropy_coeff * loss_pi_entropy
                    self_agent.optimizer_pi.zero_grad()
                    loss_pi.backward()
                    if self_agent.grad_clip > 0:
                        # Clip gradients on the actual trained parameters
                        if hasattr(self_agent, '_policy_params'):
                            params_to_clip = list(self_agent._policy_params.parameters())
                            if hasattr(self_agent, '_shared_log_std'):
                                params_to_clip.append(self_agent._shared_log_std)
                            nn.utils.clip_grad_norm_(params_to_clip, self_agent.grad_clip)
                        else:
                            nn.utils.clip_grad_norm_(self_agent.actors.parameters(), self_agent.grad_clip)
                    self_agent.optimizer_pi.step()

                    self_agent.logger.log(surr_loss=loss_surr, entropy=true_entropy, kl_divergence=kl, pi_update=None)
                    kl_all.append(kl.abs().item())
                    if self_agent.target_kl is not None and kl.abs() > 1.5 * self_agent.target_kl:
                        if _kl_early_stop:
                            # H5b: freeze remaining POLICY epochs; value keeps training.
                            pi_frozen = True
                            print(f"⏹  PPO early stop: epoch {i_update}/{self_agent.n_update_pi} "
                                  f"|KL|={kl.abs().item():.5f} > 1.5*{self_agent.target_kl}", flush=True)
                            self_agent.logger.log(kl_early_stop_epoch=i_update)
                        break
                self_agent.logger.log(pi_update_step=i_update)

                # === FIX: Value update with agent mask ===
                for i_v in range(1):
                    batch_returns = returns
                    batch_state = s
                    batch_v_mask = mask_flat
                    if n_minibatch > 1:
                        idxs = np.random.randint(0, batch_total, size=batch_size)
                        [batch_returns, batch_state] = [item[idxs] for item in [batch_returns, batch_state]]
                        if mask_flat is not None:
                            batch_v_mask = mask_flat[idxs]
                    batch_v_new = self_agent._evalV(batch_state)

                    _hub = os.environ.get('DMPO_HUBER_VALUE', '0') == '1'
                    _hub_d = float(os.environ.get('DMPO_HUBER_DELTA', '1.0'))
                    if batch_v_mask is not None:
                        # Masked value loss: only train on real agents, ignore padded.
                        # Huber (on raw returns) when DMPO_HUBER_VALUE=1 to match the
                        # dynamic arm's live loss; legacy MSE otherwise.
                        if _hub:
                            v_error = torch.nn.functional.smooth_l1_loss(
                                batch_v_new, batch_returns, reduction='none', beta=_hub_d)
                        else:
                            v_error = (batch_v_new - batch_returns) ** 2
                        v_error_masked = v_error * batch_v_mask
                        n_valid = batch_v_mask.sum().clamp(min=1.0)
                        loss_v = v_error_masked.sum() / n_valid
                    else:
                        if _hub:
                            loss_v = torch.nn.functional.smooth_l1_loss(
                                batch_v_new, batch_returns, beta=_hub_d)
                        else:
                            loss_v = ((batch_v_new - batch_returns) ** 2).mean()

                    self_agent.optimizer_v.zero_grad()
                    loss_v.backward()
                    if self_agent.grad_clip > 0:
                        # Clip gradients on the actual trained parameters
                        if hasattr(self_agent, '_value_params'):
                            nn.utils.clip_grad_norm_(self_agent._value_params.parameters(), self_agent.grad_clip)
                        else:
                            nn.utils.clip_grad_norm_(self_agent.vs.parameters(), self_agent.grad_clip)
                    self_agent.optimizer_v.step()

                    var_v = ((batch_returns - batch_returns.mean()) ** 2).mean()
                    rel_v_loss = loss_v / (var_v + 1e-8)
                    self_agent.logger.log(v_loss=loss_v, v_update=None, v_var=var_v, rel_v_loss=rel_v_loss)
                    if rel_v_loss < self_agent.v_thres:
                        break
                self_agent.logger.log(v_update_step=i_update)
                self_agent.logger.log(update=None, reward=r, value=value_old, clip=clip, returns=returns, advantages=advantages_old.abs())
            self_agent.logger.log(agent_update_time=_time.time() - time_t)
            return [r.mean().item(), true_entropy.item(), max(kl_all)]

        # Bind patched method to agent instance
        agent.updateAgent = types.MethodType(patched_updateAgent, agent)
        print("  - Patched updateAgent: masked value loss + return normalization")

        # Also patch _process_traj to mask padded agents during GAE
        self._patch_process_traj()

    def _patch_process_traj(self):
        """Monkey-patch _process_traj to zero out padded agents during GAE.

        In dynamic mode: skip — no padded agents exist.
        """
        if getattr(self.agent, 'dynamic', False):
            print("  - Dynamic mode: skipping _process_traj patch (no padded agents)")
            return

        """
        Without this, padded agents (s=zeros, r=0) still get non-zero value
        predictions from V(zeros), which leak into GAE deltas/advantages.
        The graph convolution reduce_sum then propagates this noise into
        real agents' advantages, corrupting the policy gradient signal.
        """
        import types

        agent = self.agent
        original_process_traj = agent._process_traj.__func__

        def patched_process_traj(self_agent, s, a, r, s1, d, logp, **kwargs):
            """_process_traj with masked GAE for padded agents."""
            # Extract mask if passed via kwargs (from Trajectory dict unpacking)
            mask = kwargs.get('mask', None)

            b, T, n, dim_s = s.shape
            s, a, r, s1, d, logp = [item.to(self_agent.device) for item in [s, a, r, s1, d, logp]]

            if mask is not None:
                mask = mask.to(self_agent.device)
                # mask shape: [b, T, n] or [T, n] — expand to [b, T, n, 1]
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0).expand(b, -1, -1)
                mask_4d = mask.unsqueeze(-1)  # [b, T, n, 1]
                # Zero out rewards for padded agents
                r = r * mask_4d

            value = self_agent._evalV(s.view(-1, n, dim_s)).view(b, T, n, -1)

            if mask is not None:
                # Zero out value predictions for padded agents
                value = value * mask_4d

            returns = torch.zeros(value.size(), device=self_agent.device)
            deltas, advantages = torch.zeros_like(returns), torch.zeros_like(returns)
            prev_value = self_agent._evalV(s1.select(1, T - 1))

            if mask is not None:
                # prev_value shape: [b, n, 1], mask shape: [b, T, n]
                # Use select (not slice) to drop T dim: mask[:, -1, :] → [b, n]
                prev_value = prev_value * mask[:, -1, :].unsqueeze(-1)  # [b, n, 1]

            if not self_agent.use_rtg:
                prev_return = prev_value
            else:
                prev_return = torch.zeros_like(prev_value)
            prev_advantage = torch.zeros_like(prev_return)
            d_mask = d.float()

            for t in reversed(range(T)):
                deltas[:, t, :, :] = r.select(1, t) + self_agent.gamma * (1 - d_mask.select(1, t)) * prev_value - value.select(1, t).detach()
                advantages[:, t, :, :] = deltas.select(1, t) + self_agent.gamma * self_agent.lamda * (1 - d_mask.select(1, t)) * prev_advantage
                if self_agent.use_gae_returns:
                    returns[:, t, :, :] = value.select(1, t).detach() + advantages.select(1, t)
                else:
                    returns[:, t, :, :] = r.select(1, t) + self_agent.gamma * (1 - d_mask.select(1, t)) * prev_return

                prev_return = returns.select(1, t)
                prev_value = value.select(1, t)
                prev_advantage = advantages.select(1, t)

            if mask is not None:
                # Zero out advantages/returns for padded agents before aggregation
                advantages = advantages * mask_4d
                returns = returns * mask_4d

            # Aggregate advantages across κ-hop neighbors.
            # P3.1: previously this used raw self_agent.adj (1-hop) despite the
            # comment claiming κ-hop. Now uses matrix_power(adj, radius_v) so
            # the receptive field actually matches the configured κ.
            if hasattr(self_agent, '_shared_encoder'):
                # Shared attention mode: aggregate via κ-hop adjacency mean
                if not hasattr(self_agent, '_adj_kappa_v') and hasattr(self_agent, 'adj'):
                    _kv = getattr(self_agent, 'radius_v', 2)
                    self_agent._adj_kappa_v = (torch.matrix_power(self_agent.adj.float(), _kv) > 0)
                adj_kappa_v = getattr(self_agent, '_adj_kappa_v', None)
                adv_flat = advantages.view(-1, n, 1)  # [b*T, n, 1]
                reduced_list = []
                for i in range(n):
                    if adj_kappa_v is not None and adj_kappa_v.shape[0] > i:
                        neighbor_mask_i = adj_kappa_v[i]
                        neighbor_advs = adv_flat[:, neighbor_mask_i, :]  # [b*T, k, 1]
                        reduced_list.append(neighbor_advs.mean(dim=1))  # [b*T, 1]
                    else:
                        reduced_list.append(adv_flat[:, i, :])
                reduced_advantages = torch.stack(reduced_list, dim=1).view(advantages.size())
            else:
                # Per-agent mode: use original MultiCollect (already κ-hop via
                # matrix_power inside its constructor).
                reduced_advantages = self_agent.collect_v.reduce_sum(advantages.view(-1, n, 1)).view(advantages.size())

            if self_agent.advantage_norm and reduced_advantages.size()[1] > 1:
                reduced_advantages = (reduced_advantages - reduced_advantages.mean(dim=1, keepdim=True)) / (reduced_advantages.std(dim=1, keepdim=True) + 1e-5)
                advantages = (advantages - advantages.mean(dim=1, keepdim=True)) / (advantages.std(dim=1, keepdim=True) + 1e-5)

            return value.detach(), returns, advantages.detach(), reduced_advantages.detach()

        agent._process_traj = types.MethodType(patched_process_traj, agent)
        print("  - Patched _process_traj: masked GAE for padded agents")

    def _patch_shared_attention_policy(self):
        """Replace per-agent policy/value networks with shared networks + attention.

        Instead of n separate GaussianActors (one per slot), use ONE shared actor
        that processes each agent with attention over its κ-hop neighbors.

        This enables the policy to handle any number of agents naturally —
        no padding, no slot assignment, no out-of-distribution issues when
        agent count changes.

        Reference: DGN (Jiang et al., ICLR 2020) — attention-based graph convolution
        """
        import types
        import torch.nn as nn
        from torch.distributions.normal import Normal

        agent = self.agent
        obs_dim = agent.observation_dim
        action_dim = agent.action_dim
        device = self.device

        # Create shared encoder + attention + policy head
        embed_dim = 64

        shared_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU()
        ).to(device)

        # Attention: query and key projections
        attn_query = nn.Linear(embed_dim, embed_dim).to(device)
        attn_key = nn.Linear(embed_dim, embed_dim).to(device)

        # Policy head: takes (my_embedding + context) → action mean
        shared_action_head = nn.Linear(embed_dim * 2, action_dim).to(device)
        shared_log_std = nn.Parameter(-0.5 * torch.ones(action_dim, device=device))

        # Value network: separate encoder + head (avoids in-place gradient conflict with policy)
        value_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU()
        ).to(device)

        value_attn_query = nn.Linear(embed_dim, embed_dim).to(device)
        value_attn_key = nn.Linear(embed_dim, embed_dim).to(device)

        shared_value_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)

        # Store on agent for parameter access
        agent._shared_encoder = shared_encoder
        agent._attn_query = attn_query
        agent._attn_key = attn_key
        agent._shared_action_head = shared_action_head
        agent._shared_log_std = shared_log_std
        # Env-configurable log_std bounds (V2): a too-permissive MAX let std run
        # away to the ceiling (≈random policy). Read once here so the per-step act
        # forward doesn't hit os.environ. Mirrors models.py GaussianActor bounds.
        agent._log_std_min = float(os.environ.get('DMPO_LOG_STD_MIN', '-2.0'))
        agent._log_std_max = float(os.environ.get('DMPO_LOG_STD_MAX', '1.0'))
        agent._value_encoder = value_encoder
        agent._value_attn_query = value_attn_query
        agent._value_attn_key = value_attn_key
        agent._shared_value_head = shared_value_head
        agent._embed_dim = embed_dim

        # Register as parameters so optimizer can find them
        agent._policy_params = nn.ModuleDict({
            'encoder': shared_encoder,
            'attn_query': nn.ModuleList([attn_query]),
            'attn_key': nn.ModuleList([attn_key]),
            'action_head': nn.ModuleList([shared_action_head]),
        }).to(device)

        agent._value_params = nn.ModuleDict({
            'encoder': value_encoder,
            'attn_query': nn.ModuleList([value_attn_query]),
            'attn_key': nn.ModuleList([value_attn_key]),
            'value_head': shared_value_head
        }).to(device)

        # Separate optimizers — no shared parameters between policy and value
        policy_params = list(agent._policy_params.parameters()) + [shared_log_std]
        agent.optimizer_pi = torch.optim.Adam(policy_params, lr=agent.lr)
        agent.optimizer_v = torch.optim.Adam(agent._value_params.parameters(), lr=agent.lr_v)

        def _attend(self_agent, s_full, agent_idx, all_embedded=None, real_agent_mask=None):
            """Compute attention-aggregated context for agent_idx.

            Args:
                s_full: [batch, n_agent, obs_dim] — all agents' observations
                agent_idx: which agent to compute for
                all_embedded: pre-computed embeddings (avoids recomputing for each agent)
                real_agent_mask: [n_agent] bool — True for real agents, False for padded
                    If None, derived from observations (padded agents have all-zero obs)

            Returns:
                combined: [batch, embed_dim * 2] — (my_embedding, context)
                all_embedded: cached embeddings for reuse
                real_agent_mask: cached mask for reuse
            """
            b, n, d = s_full.shape
            # Encode all agents with shared encoder (compute once, reuse)
            if all_embedded is None:
                all_embedded = self_agent._shared_encoder(s_full.reshape(-1, d)).reshape(b, n, self_agent._embed_dim)

            # Detect padded agents from observations (all-zero = padded)
            if real_agent_mask is None:
                # An agent is real if any observation value is non-zero (across batch)
                real_agent_mask = (s_full.abs().sum(dim=-1).sum(dim=0) > 0)  # [n_agent]

            my_embed = all_embedded[:, agent_idx, :]  # [batch, embed_dim]

            # Get κ-hop neighbor indices from adjacency, excluding padded agents.
            # P3.2: previously read raw self_agent.adj (1-hop). Now uses the
            # cached κ-hop closure (computed via matrix_power(adj, radius_pi))
            # so the policy attention actually spans κ hops as designed.
            # Dynamic-mode shape fix kept: n (current agent count) can be
            # smaller than the κ-hop matrix; slice and extend as needed.
            if not hasattr(self_agent, '_adj_kappa_pi') and hasattr(self_agent, 'adj'):
                _kpi = getattr(self_agent, 'radius_pi', 2)
                self_agent._adj_kappa_pi = (torch.matrix_power(self_agent.adj.float(), _kpi) > 0)
            adj_kappa_pi = getattr(self_agent, '_adj_kappa_pi', None)
            if adj_kappa_pi is not None and adj_kappa_pi.shape[0] > agent_idx:
                adj_row = adj_kappa_pi[agent_idx]
                adj_n = adj_row.shape[0]
                if adj_n >= n:
                    neighbor_mask = adj_row[:n].clone()
                else:
                    # H4(b) gated: legacy extends overflow agents with ONES
                    # (= fully-connected fiction); DMPO_ADJ_FIX=1 extends with
                    # ZEROS (no invented edges for unknown agents).
                    if os.environ.get('DMPO_ADJ_FIX', '0') == '1':
                        extension = torch.zeros(n - adj_n, dtype=torch.bool, device=adj_row.device)
                    else:
                        extension = torch.ones(n - adj_n, dtype=torch.bool, device=adj_row.device)
                    neighbor_mask = torch.cat([adj_row.clone(), extension])
                neighbor_mask[agent_idx] = False  # exclude self
                neighbor_mask = neighbor_mask & real_agent_mask  # exclude padded agents
            else:
                neighbor_mask = torch.zeros(n, dtype=torch.bool, device=s_full.device)

            if neighbor_mask.any():
                neighbor_embeds = all_embedded[:, neighbor_mask, :]  # [batch, k_neighbors, embed_dim]

                # Attention weights
                q = self_agent._attn_query(my_embed).unsqueeze(1)  # [batch, 1, embed_dim]
                k = self_agent._attn_key(neighbor_embeds)  # [batch, k_neighbors, embed_dim]
                attn_scores = (q * k).sum(dim=-1) / (self_agent._embed_dim ** 0.5)  # [batch, k_neighbors]
                attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)  # [batch, k_neighbors, 1]

                context = (attn_weights * neighbor_embeds).sum(dim=1)  # [batch, embed_dim]
            else:
                context = torch.zeros_like(my_embed)

            return torch.cat([my_embed, context], dim=-1), all_embedded, real_agent_mask  # [batch, embed_dim * 2]

        agent._attend = types.MethodType(_attend, agent)

        # Patch act() — shared policy for all agents
        def shared_act(self_agent, s, if_test=False, requires_log=False):
            with torch.no_grad():
                dim = s.dim()
                while s.dim() <= 2:
                    s = s.unsqueeze(0)
                s = s.to(self_agent.device)
                b, n, d = s.shape

                means, stds = [], []
                cached_embeds = None
                cached_mask = None
                for i in range(n):
                    combined, cached_embeds, cached_mask = self_agent._attend(s, i, cached_embeds, cached_mask)
                    mean = self_agent._shared_action_head(combined)
                    clamped_log_std = torch.clamp(self_agent._shared_log_std, self_agent._log_std_min, self_agent._log_std_max)
                    std = torch.exp(clamped_log_std).expand_as(mean)
                    means.append(mean)
                    stds.append(std)

                means = torch.stack(means, dim=1)
                stds = torch.stack(stds, dim=1)
                while means.dim() > dim:
                    means = means.squeeze(0)
                    stds = stds.squeeze(0)
                return Normal(means, stds)

        agent.act = types.MethodType(shared_act, agent)

        # Patch get_logp() — shared policy log probability
        def shared_get_logp(self_agent, s, a):
            s = torch.as_tensor(s, dtype=torch.float32, device=self_agent.device)
            while s.dim() <= 2:
                s = s.unsqueeze(0)
                a = a.unsqueeze(0)
            while a.dim() < s.dim():
                a = a.unsqueeze(-1)
            b, n, d = s.shape

            # Compute embeddings once, reuse for all agents
            all_embedded = self_agent._shared_encoder(s.reshape(-1, d)).reshape(b, n, self_agent._embed_dim)

            log_probs = []
            cached_mask = None
            for i in range(n):
                combined, _, cached_mask = self_agent._attend(s, i, all_embedded, cached_mask)
                mean = self_agent._shared_action_head(combined)
                clamped_log_std = torch.clamp(self_agent._shared_log_std, self_agent._log_std_min, self_agent._log_std_max)
                std = torch.exp(clamped_log_std).expand_as(mean)
                dist = Normal(mean, std)
                log_probs.append(dist.log_prob(a.select(dim=1, index=i)).sum(dim=-1))

            log_prob = torch.stack(log_probs, dim=1)
            while log_prob.dim() < 3:
                log_prob = log_prob.unsqueeze(-1)
            return log_prob

        agent.get_logp = types.MethodType(shared_get_logp, agent)

        # Patch _evalV() — separate value network (own encoder, no gradient conflict)
        def shared_evalV(self_agent, s):
            s = torch.as_tensor(s, dtype=torch.float32, device=self_agent.device)
            while s.dim() <= 2:
                s = s.unsqueeze(0)
            b, n, d = s.shape

            # Detect padded agents from observations (all-zero = padded)
            real_agent_mask = (s.abs().sum(dim=-1).sum(dim=0) > 0)  # [n_agent]

            # Value network uses its OWN encoder (separate from policy)
            all_embedded = self_agent._value_encoder(s.reshape(-1, d)).reshape(b, n, self_agent._embed_dim)

            values = []
            for i in range(n):
                my_embed = all_embedded[:, i, :]

                # Value attention (separate from policy attention), excluding padded agents.
                # P3.3: same κ-hop fix as policy _attend. Use the cached
                # adj_kappa_v (matrix_power(adj, radius_v)) instead of raw 1-hop adj.
                if not hasattr(self_agent, '_adj_kappa_v') and hasattr(self_agent, 'adj'):
                    _kv = getattr(self_agent, 'radius_v', 2)
                    self_agent._adj_kappa_v = (torch.matrix_power(self_agent.adj.float(), _kv) > 0)
                adj_kappa_v = getattr(self_agent, '_adj_kappa_v', None)
                if adj_kappa_v is not None and adj_kappa_v.shape[0] > i:
                    adj_row = adj_kappa_v[i]
                    adj_n = adj_row.shape[0]
                    if adj_n >= n:
                        neighbor_mask = adj_row[:n].clone()
                    else:
                        # H4(b) gated: same ones→zeros overflow-extension fix as
                        # the policy attention path above.
                        if os.environ.get('DMPO_ADJ_FIX', '0') == '1':
                            extension = torch.zeros(n - adj_n, dtype=torch.bool, device=adj_row.device)
                        else:
                            extension = torch.ones(n - adj_n, dtype=torch.bool, device=adj_row.device)
                        neighbor_mask = torch.cat([adj_row.clone(), extension])
                    neighbor_mask[i] = False
                    neighbor_mask = neighbor_mask & real_agent_mask  # exclude padded
                else:
                    neighbor_mask = torch.zeros(n, dtype=torch.bool, device=s.device)

                if neighbor_mask.any():
                    neighbor_embeds = all_embedded[:, neighbor_mask, :]
                    q = self_agent._value_attn_query(my_embed).unsqueeze(1)
                    k = self_agent._value_attn_key(neighbor_embeds)
                    attn_scores = (q * k).sum(dim=-1) / (self_agent._embed_dim ** 0.5)
                    attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)
                    context = (attn_weights * neighbor_embeds).sum(dim=1)
                else:
                    context = torch.zeros_like(my_embed)

                combined = torch.cat([my_embed, context], dim=-1)
                v = self_agent._shared_value_head(combined)
                values.append(v)

            return torch.stack(values, dim=1)  # [batch, n_agent, 1]

        agent._evalV = types.MethodType(shared_evalV, agent)

        # Count parameters
        shared_param_count = sum(p.numel() for p in agent._policy_params.parameters()) + sum(p.numel() for p in agent._value_params.parameters()) + shared_log_std.numel()
        old_actor_count = sum(p.numel() for p in agent.actors.parameters()) if hasattr(agent, 'actors') else 0
        old_vs_count = sum(p.numel() for p in agent.vs.parameters()) if hasattr(agent, 'vs') else 0

        print(f"  - Shared attention policy: {shared_param_count:,} params (replaces {old_actor_count:,} actor + {old_vs_count:,} value params)")

    def _setup_model_ensemble(self):
        """Create ensemble of 5 GCNs and patch model_step with uncertainty penalty.

        Each GCN is trained on bootstrap samples from the ModelBuffer.
        During model_step, all 5 predict → mean prediction used, reward
        penalized by ensemble disagreement (std across predictions).

        Reference: MOPO (Yu et al., NeurIPS 2020)
        """
        import types
        from algorithms.models import GraphConvolutionalModel, DynamicGraphConvolutionalModel
        from torch.optim import Adam

        agent = self.agent
        n_ensemble = 5
        uncertainty_penalty = 1.0  # λ coefficient for reward penalty

        # Create 4 additional GCNs (agent already has 1 as self.ps)
        ensemble_models = [agent.ps]  # First model is the existing one
        ensemble_optimizers = [agent.optimizer_p]

        for i in range(n_ensemble - 1):
            if getattr(agent, 'dynamic', False):
                # Dynamic mode: use DynamicGraphConvolutionalModel (shared weights,
                # handles variable n_agent). Per-vehicle trajectories have n_agent=1.
                model = DynamicGraphConvolutionalModel(
                    agent.logger, agent.observation_dim, agent.action_dim,
                    agent.p_args
                ).to(self.device)
            else:
                model = GraphConvolutionalModel(
                    agent.logger, agent.adj,
                    agent.observation_dim, agent.action_dim,
                    agent.n_agent, agent.p_args
                ).to(self.device)
            optimizer = Adam(model.parameters(), lr=agent.lr_p)
            ensemble_models.append(model)
            ensemble_optimizers.append(optimizer)

        agent._ensemble_models = ensemble_models
        agent._ensemble_optimizers = ensemble_optimizers
        agent._ensemble_size = n_ensemble
        agent._uncertainty_penalty = uncertainty_penalty

        # Patch updateModel to train all 5 GCNs on bootstrap samples
        original_updateModel = agent.updateModel

        def ensemble_updateModel(self_agent, trajs, length=1):
            """Train each ensemble member on a bootstrap sample."""
            import time as _time
            time_t = _time.time()

            ss, actions, rs, s1s, ds = [], [], [], [], []
            for traj in trajs:
                s, a, r, s1, d = traj["s"], traj["a"], traj["r"], traj["s1"], traj["d"]
                s, a, r, s1, d = [torch.as_tensor(item, device=self_agent.device) for item in [s, a, r, s1, d]]
                ss.append(s)
                actions.append(a)
                rs.append(r)
                s1s.append(s1)
                ds.append(d)
            ss, actions, rs, s1s, ds = [torch.stack(item, dim=0) for item in [ss, actions, rs, s1s, ds]]

            total_rel_error = 0.0
            batch_size = ss.size(0)

            for i, (model, optimizer) in enumerate(zip(
                self_agent._ensemble_models, self_agent._ensemble_optimizers
            )):
                # Bootstrap sample: random indices with replacement
                boot_idx = torch.randint(0, batch_size, (batch_size,), device=self_agent.device)
                boot_ss = ss[boot_idx]
                boot_actions = actions[boot_idx]
                boot_rs = rs[boot_idx]
                boot_s1s = s1s[boot_idx]
                boot_ds = ds[boot_idx]

                # DynamicGraphConvolutionalModel uses train_model() to avoid
                # conflict with nn.Module.train(mode).
                if hasattr(model, 'train_model'):
                    loss, rel_error = model.train_model(boot_ss, boot_actions, boot_rs, boot_s1s, boot_ds, length)
                else:
                    loss, rel_error = model.train(boot_ss, boot_actions, boot_rs, boot_s1s, boot_ds, length)
                optimizer.zero_grad()
                loss.sum().backward()
                optimizer.step()
                total_rel_error += rel_error.item()

            avg_rel_error = total_rel_error / self_agent._ensemble_size
            self_agent.logger.log(p_loss_total=loss.sum(), p_update=None)
            self_agent.logger.log(model_update_time=_time.time() - time_t)
            return avg_rel_error

        agent.updateModel = types.MethodType(ensemble_updateModel, agent)

        # Patch validateModel similarly. Dynamic-mode safe: iterates per
        # trajectory instead of stacking, so variable T doesn't crash. Each
        # trajectory contributes to the rel_error weighted by its length.
        def ensemble_validateModel(self_agent, trajs, length=1):
            with torch.no_grad():
                if not trajs:
                    return 0.0

                # MATCHED MODEL LOSS (gated): per-traj matched targets, index-aligned
                # with `prepared` (None entries when the flag is off).
                #   chain : ref = sid[0]  → for the L-step autoregressive rel-err
                #   one   : ref = sid     → for the one-step predict() paths below
                _matched_on = getattr(self, '_matched_on', False)
                _geom = getattr(self, '_nb_geom', None) if _matched_on else None
                prepared_m = []

                # Prepare each trajectory's tensors once (shared across ensemble members)
                prepared = []
                for traj in trajs:
                    s, a, r, s1, d = traj["s"], traj["a"], traj["r"], traj["s1"], traj["d"]
                    s, a, r, s1, d = [torch.as_tensor(item, device=self_agent.device)
                                      for item in [s, a, r, s1, d]]
                    prepared.append((s, a, r, s1, d))
                    if _matched_on:
                        _td = traj.dict if hasattr(traj, 'dict') else traj
                        if 'slot_ids' not in _td or 'slot_ids_next' not in _td:
                            raise RuntimeError("DMPO_MODEL_MATCHED_LOSS=1 but a validation "
                                               "trajectory carries no slot_ids/slot_ids_next")
                        _sid = torch.as_tensor(_td['slot_ids'], device=self_agent.device)
                        _sidn = torch.as_tensor(_td['slot_ids_next'], device=self_agent.device)
                        _L = min(length, s.shape[0])
                        _s1m_c, _w_c = _mml.build_matched_target(
                            s1[:_L], _sid[:_L], _sidn[:_L], ref=_sid[0:1],
                            off=_geom['off'], N=_geom['N'], F=_geom['F'],
                            neutral=_geom['neutral'])
                        _s1m_o, _w_o = _mml.build_matched_target(
                            s1, _sid, _sidn, ref=_sid,
                            off=_geom['off'], N=_geom['N'], F=_geom['F'],
                            neutral=_geom['neutral'])
                        prepared_m.append({'s1m_chain': _s1m_c, 'w_chain': _w_c,
                                           's1m_one': _s1m_o, 'w_one': _w_o})
                    else:
                        prepared_m.append(None)

                total_rel_error = 0.0
                valid_models = 0
                for model in self_agent._ensemble_models:
                    model_rel_error = 0.0
                    weight_sum = 0.0
                    for _pi, (s, a, r, s1, d) in enumerate(prepared):
                        T = s.shape[0]
                        t_length = min(length, T)
                        if t_length == 0:
                            continue
                        if prepared_m[_pi] is not None:
                            # MATCHED: same L-step chain as train_model, matched
                            # target/weight (masked MSE / pooled var — same units as
                            # the framework's rel_state_loss). NOTE: train_model's
                            # state_loss/reward_loss/done_loss logger keys are NOT
                            # emitted on this branch.
                            rel_error = _mml.chain_state_rel_err(
                                model, s, a, prepared_m[_pi]['s1m_chain'],
                                prepared_m[_pi]['w_chain'], length)
                            model_rel_error += rel_error.item() * t_length
                            weight_sum += t_length
                            continue
                        # Add batch dim (batch=1) so per-trajectory call matches
                        # the model's expected [batch, T, n_agents, ...] shape.
                        s_b = s[:t_length].unsqueeze(0)
                        a_b = a[:t_length].unsqueeze(0)
                        r_b = r[:t_length].unsqueeze(0)
                        s1_b = s1[:t_length].unsqueeze(0)
                        d_b = d[:t_length].unsqueeze(0)
                        if hasattr(model, 'train_model'):
                            _, rel_error = model.train_model(s_b, a_b, r_b, s1_b, d_b, length)
                        else:
                            _, rel_error = model.train(s_b, a_b, r_b, s1_b, d_b, length)
                        # Length-weighted average — longer trajectories contribute more
                        model_rel_error += rel_error.item() * t_length
                        weight_sum += t_length
                    if weight_sum > 0:
                        total_rel_error += model_rel_error / weight_sum
                        valid_models += 1

                # ── STABLE rel-err (2026-08-14): the legacy metric divides each
                # batch's MSE by THAT BATCH's target variance (+1e-7), so a
                # homogeneous validation batch detonates the ratio regardless of
                # model quality (observed 0.2→15 swings, one 1e22 spike). Here:
                # pool squared error and target variance across the WHOLE
                # validation sample (member 0), then divide once. Same units,
                # conditioned denominator.
                try:
                    _m0 = self_agent._ensemble_models[0]
                    _num = 0.0
                    _s1_all = []
                    _w_all = []          # MATCHED only (stays empty when off)
                    _preds = []
                    for _pi, (s, a, r, s1, d) in enumerate(prepared):
                        _pr, _ps1, _pd = _m0.predict(s, a)
                        if prepared_m[_pi] is not None:
                            # MATCHED: one-step matched target/weight; explicit shapes
                            # (no .squeeze() — it collapses the agent axis when L=1).
                            _D = s1.shape[-1]
                            _tm = prepared_m[_pi]['s1m_one'].reshape(-1, _D)
                            _wm = prepared_m[_pi]['w_one'].reshape(-1, _D)
                            _preds.append(((_ps1.reshape(-1, _D) - _tm).pow(2) * _wm).sum().item())
                            _s1_all.append(_tm)
                            _w_all.append(_wm)
                            continue
                        _preds.append((_ps1.squeeze() - s1.squeeze()).pow(2).sum().item())
                        _s1_all.append(s1.reshape(-1, s1.shape[-1]))
                    _num = sum(_preds)
                    _cat = torch.cat(_s1_all, dim=0)
                    if _w_all:
                        # MATCHED: pooled var over the SAME unmasked features as _num,
                        # centred on the w-WEIGHTED per-feature mean (an unweighted
                        # mean would let masked rows — left/entered/absent — pull the
                        # centre; same definition as chain_state_rel_err's "legacy").
                        _wcat = torch.cat(_w_all, dim=0)
                        _mu = (_cat * _wcat).sum(0, keepdim=True) / _wcat.sum(0, keepdim=True).clamp_min(1)
                        _den = ((_cat - _mu).pow(2) * _wcat).sum().item()
                    else:
                        _den = (_cat - _cat.mean(dim=0, keepdim=True)).pow(2).sum().item()
                    _stable = _num / max(_den, 1e-7)
                    _legacy = total_rel_error / max(valid_models, 1)
                    print(f"  📐 model-val: stable={_stable:.4f} legacy={_legacy:.4f} "
                          f"(pooled err={_num:.3e}, pooled var={_den:.3e})")
                    self_agent._last_stable_rel_error = _stable
                except Exception as _e:
                    print(f"  ⚠️ stable rel-err failed: {_e}")

                # ── Gated per-dimension error dump (DMPO_MODEL_ERR_DUMP=path).
                # Diagnostic only: accumulates Σ|Δ| and Σ|target| per obs dim
                # (plus the reward channel) across validation transitions, so
                # the aggregate rel-err can be decomposed offline. Default off.
                _dump = os.environ.get('DMPO_MODEL_ERR_DUMP', '')
                if _dump:
                    try:
                        import numpy as _np
                        _m = self_agent._ensemble_models[0]
                        _es = _eb = None
                        _rs_e = _rs_b = 0.0
                        for _pi, (s, a, r, s1, d) in enumerate(prepared):
                            _pr, _ps1, _pd = _m.predict(s, a)
                            if prepared_m[_pi] is not None:
                                # MATCHED: decompose the same one-step matched metric
                                _D = s1.shape[-1]
                                _tm = prepared_m[_pi]['s1m_one'].reshape(-1, _D)
                                _wm = prepared_m[_pi]['w_one'].reshape(-1, _D)
                                _err = ((_ps1.reshape(-1, _D) - _tm).abs() * _wm).sum(dim=0)
                                _base = (_tm.abs() * _wm).sum(dim=0)
                            else:
                                _err = (_ps1.squeeze() - s1.squeeze()).abs().sum(dim=tuple(range(_ps1.squeeze().dim() - 1)))
                                _base = s1.squeeze().abs().sum(dim=tuple(range(s1.squeeze().dim() - 1)))
                            _es = _err if _es is None else _es + _err
                            _eb = _base if _eb is None else _eb + _base
                            _rs_e += (_pr.squeeze() - r.squeeze()).abs().sum().item()
                            _rs_b += r.squeeze().abs().sum().item()
                        if _es is not None:
                            _es = _es.cpu().numpy(); _eb = _eb.cpu().numpy()
                            try:
                                _old = _np.load(_dump)
                                _es = _es + _old['err']; _eb = _eb + _old['base']
                                _rs_e += float(_old['r_err']); _rs_b += float(_old['r_base'])
                            except Exception:
                                pass
                            _np.savez(_dump, err=_es, base=_eb,
                                      r_err=_rs_e, r_base=_rs_b)
                    except Exception as _e:
                        print(f"  ⚠️ MODEL_ERR_DUMP failed: {_e}")

                return total_rel_error / max(valid_models, 1)

        agent.validateModel = types.MethodType(ensemble_validateModel, agent)

        # Patch model_step to use ensemble mean + uncertainty penalty
        def ensemble_model_step(self_agent, s, a):
            """Predict with all ensemble members, penalize reward by uncertainty."""
            with torch.no_grad():
                while s.dim() <= 2:
                    s = s.unsqueeze(0)
                    a = a.unsqueeze(0)
                while a.dim() <= 2:
                    a = a.unsqueeze(-1)
                s = s.to(self_agent.device)
                a = a.to(self_agent.device)

                all_rs, all_s1s, all_ds = [], [], []
                for model in self_agent._ensemble_models:
                    rs, s1s, ds = model.predict(s, a)
                    all_rs.append(rs)
                    all_s1s.append(s1s)
                    all_ds.append(ds)

                # Stack: [n_ensemble, batch, n_agent, dim]
                all_rs = torch.stack(all_rs, dim=0)
                all_s1s = torch.stack(all_s1s, dim=0)
                all_ds = torch.stack([d.float() for d in all_ds], dim=0)  # Bool→Float for mean

                # Mean predictions
                mean_rs = all_rs.mean(dim=0)
                mean_s1s = all_s1s.mean(dim=0)
                mean_ds = all_ds.mean(dim=0)

                # Uncertainty: std of reward predictions across ensemble
                reward_std = all_rs.std(dim=0)

                # Penalize reward by uncertainty (MOPO-style)
                penalized_rs = mean_rs - self_agent._uncertainty_penalty * reward_std

                return penalized_rs.detach(), mean_s1s.detach(), mean_ds.detach(), s.detach()

        agent.model_step = types.MethodType(ensemble_model_step, agent)

        # ── Proposal 1+2: Patch GCN forward to zero padded edges + mask model loss ──

        def _make_masked_forward(model):
            """Wrap GCN forward to zero edge outputs involving padded agents."""
            original_forward = model.forward

            def masked_forward(s, a, agent_mask=None):
                """GCN forward with optional agent mask to prevent pad corruption.

                agent_mask: [batch_size, n_agent] — 1.0 for real, 0.0 for padded.
                When provided, edge outputs involving any padded agent are zeroed
                before entering the NodeNetwork sum, preventing garbage from leaking
                into real agents' representations.

                For DynamicGraphConvolutionalModel: delegates to its own forward()
                which handles variable agent counts natively (no masking needed).
                """
                from algorithms.models import DynamicGraphConvolutionalModel
                if isinstance(model, DynamicGraphConvolutionalModel):
                    # Dynamic GCN handles variable agents natively
                    return model.forward(s, a)

                # Original GCN with per-agent networks
                embedding = model.node_embedding(s)
                for _ in range(model.n_conv):
                    edge_info_of_nodes = [[] for __ in range(model.n_agent)]
                    for edge_net in model.edge_nets:
                        edge_info = edge_net(embedding)
                        # Zero edge output if either endpoint is padded
                        if agent_mask is not None:
                            mask_i = agent_mask[:, edge_net.i].unsqueeze(-1)  # [batch, 1]
                            mask_j = agent_mask[:, edge_net.j].unsqueeze(-1)  # [batch, 1]
                            edge_info = edge_info * mask_i * mask_j
                        edge_info_of_nodes[edge_net.i].append(edge_info)
                        edge_info_of_nodes[edge_net.j].append(edge_info)
                    node_preds = []
                    for i in range(model.n_agent):
                        node_net = model.node_nets[i]
                        node_pred = node_net(edge_info_of_nodes[i], a.select(dim=1, index=i))
                        node_preds.append(node_pred)
                    embedding = torch.stack(node_preds, dim=1)
                state_pred = model.state_head(embedding)
                if model.residual:
                    state_pred += s
                reward_pred = model.reward_head(embedding)
                done_pred = model.done_head(embedding)
                return reward_pred, state_pred, done_pred

            model._masked_forward = masked_forward

        for m in ensemble_models:
            _make_masked_forward(m)

        # Patch predict to also use masked forward
        def _make_masked_predict(model):
            def masked_predict(s, a, agent_mask=None):
                with torch.no_grad():
                    r1, s1, d1 = model._masked_forward(s, a, agent_mask=agent_mask)
                    done = torch.clamp(d1, 0., 1.)
                    done = torch.cat([1 - done, done], dim=-1)
                    from torch.distributions import Categorical
                    done = Categorical(done).sample() > 0
                    return r1, s1, done
            model._masked_predict = masked_predict

        for m in ensemble_models:
            _make_masked_predict(m)

        # Override ensemble_updateModel with dynamic (no-padding) masked loss.
        # Trajectories have variable T in dynamic mode; we iterate per-trajectory
        # and accumulate masked element-wise losses instead of stacking into
        # [batch, T, ...]. Memory scales with sum-of-trajectory-lengths rather
        # than max_T × batch_size.
        def ensemble_updateModel_dynamic(self_agent, trajs, length=1):
            """Train each ensemble member on variable-length trajectories."""
            import time as _time
            time_t = _time.time()

            if not trajs:
                return 0.0

            # MATCHED MODEL LOSS flag/geometry (trainer attrs; False when off)
            _matched_on = getattr(self, '_matched_on', False)
            _geom = getattr(self, '_nb_geom', None) if _matched_on else None
            _dbg_on = _matched_on and os.environ.get('DMPO_MODEL_MATCHED_DEBUG', '1') == '1'
            _stat_acc = ({k: torch.zeros((), dtype=torch.long, device=self_agent.device)
                          for k in _mml.STAT_KEYS} if _matched_on else None)

            # Prepare trajectory tensors once (shared across ensemble members)
            prepared = []
            for traj in trajs:
                s, a, r, s1, d = traj["s"], traj["a"], traj["r"], traj["s1"], traj["d"]
                s, a, r, s1, d = [torch.as_tensor(item, device=self_agent.device) for item in [s, a, r, s1, d]]
                _has_mask = ("mask" in traj.dict) if hasattr(traj, 'dict') else ("mask" in traj)
                if _has_mask:
                    _m = traj.dict["mask"] if hasattr(traj, 'dict') else traj["mask"]
                    m = torch.as_tensor(_m, device=self_agent.device)
                else:
                    m = torch.ones(s.shape[0], s.shape[1], device=self_agent.device)
                _entry = {"s": s, "a": a, "r": r, "s1": s1, "d": d, "mask": m}
                # MATCHED MODEL LOSS (gated DMPO_MODEL_MATCHED_LOSS): build the
                # identity-matched target ONCE per trajectory (model-independent —
                # outside the member/bootstrap loop). Chain reference = sid[0] of the
                # TRUNCATED window: the autoregressive chain feeds s0 = s_ back, so
                # step k's prediction lives in s[0]'s slot order; its target is "the
                # s[0]-slot vehicles' features at t=k+1" (matched against sidn[k]).
                if _matched_on:
                    _td = traj.dict if hasattr(traj, 'dict') else traj
                    if 'slot_ids' not in _td or 'slot_ids_next' not in _td:
                        raise RuntimeError(
                            "DMPO_MODEL_MATCHED_LOSS=1 but a real trajectory carries no "
                            "slot_ids/slot_ids_next (a silent slot-wise fallback would "
                            "defeat the run)")
                    _sid = torch.as_tensor(_td['slot_ids'], device=self_agent.device)
                    _sidn = torch.as_tensor(_td['slot_ids_next'], device=self_agent.device)
                    _L = min(length, s.shape[0])
                    _s1m, _w, _st = _mml.build_matched_target(
                        s1[:_L], _sid[:_L], _sidn[:_L], ref=_sid[0:1],
                        off=_geom['off'], N=_geom['N'], F=_geom['F'], return_stats=True,
                        neutral=_geom['neutral'])
                    _entry['s1m'] = _s1m            # [L,1,D]
                    _entry['w'] = _w                # [L,1,D] ∈ {0,1}
                    for _k in _mml.STAT_KEYS:
                        _stat_acc[_k] = _stat_acc[_k] + _st[_k]
                prepared.append(_entry)

            if _matched_on and _dbg_on:
                # C5 diagnostic: slot-match census of THIS call's batch (single sync);
                # printed sparsely (every 500th call) — the smoke-test evidence line.
                _n_call = getattr(self_agent, '_matched_dbg_calls', 0)
                self_agent._matched_dbg_calls = _n_call + 1
                self_agent._last_matched_stats = {k: int(v.item()) for k, v in _stat_acc.items()}
                if _n_call % 500 == 0:
                    print(f"  🔗 matched-loss census (updateModel call {_n_call}, "
                          f"{len(prepared)} trajs, L≤{length}): "
                          f"{_mml.format_match_stats(self_agent._last_matched_stats)}")

            batch_size = len(prepared)
            total_rel_error = 0.0
            total_loss = 0.0
            last_loss = None

            for i, (model, optimizer) in enumerate(zip(
                self_agent._ensemble_models, self_agent._ensemble_optimizers
            )):
                # Bootstrap-sample trajectory indices (with replacement)
                boot_idx = torch.randint(0, batch_size, (batch_size,), device=self_agent.device).tolist()

                # Accumulators (element-wise sums, normalized at end)
                state_loss_num = torch.zeros((), device=self_agent.device)
                reward_loss_num = torch.zeros((), device=self_agent.device)
                done_loss_num = torch.zeros((), device=self_agent.device)
                mask_elem_total = 0.0  # sum of valid (T, n_agent) entries
                state_w_total = torch.zeros((), device=self_agent.device)  # MATCHED: Σ(w·mask), single sync-free accumulator

                # For variance normalization of relative error
                all_s1_flat = []

                for j in boot_idx:
                    tr = prepared[j]
                    T = tr["s"].shape[0]
                    t_length = min(length, T)
                    if t_length == 0:
                        continue

                    # Per-trajectory rollout, batch dim = 1
                    s0 = tr["s"][0:1]  # [1, n_agents, obs_dim]
                    pred_r_list, pred_s_list, pred_d_list = [], [], []
                    for t in range(t_length):
                        a_t = tr["a"][t:t+1]       # [1, n_agents, act_dim]
                        m_t = tr["mask"][t:t+1]    # [1, n_agents]
                        r_, s_, d_ = model._masked_forward(s0, a_t, agent_mask=m_t)
                        pred_r_list.append(r_)
                        pred_s_list.append(s_)
                        pred_d_list.append(d_)
                        s0 = s_  # auto-regressive

                    # Concat across time (batch dim 0 is 1 — cat on dim 0 gives t_length)
                    state_pred = torch.cat(pred_s_list, dim=0)   # [t_length, n_agents, obs_dim]
                    reward_pred = torch.cat(pred_r_list, dim=0)
                    done_pred = torch.cat(pred_d_list, dim=0)

                    s1_target = tr["s1"][:t_length]
                    r_target = tr["r"][:t_length]
                    d_target = tr["d"][:t_length].float()
                    mask_t_slice = tr["mask"][:t_length]           # [t_length, n_agents]
                    mask_expanded = mask_t_slice.unsqueeze(-1)     # [t_length, n_agents, 1]

                    if 's1m' in tr:
                        # MATCHED MODEL LOSS: state term against the identity-matched
                        # target, weighted by w (0 on left/entered nb slots) AND the
                        # agent mask; reward/done are per-ego scalars — unchanged.
                        # s1m/w are [L,1,D] with L = min(length, T) == t_length.
                        s1_target = tr['s1m'][:t_length]
                        _w_t = tr['w'][:t_length]
                        state_loss_num = state_loss_num + (model.MSE(state_pred, s1_target) * _w_t * mask_expanded).sum()
                        state_w_total = state_w_total + (_w_t * mask_expanded).sum().detach()
                    else:
                        state_loss_num = state_loss_num + (model.MSE(state_pred, s1_target) * mask_expanded).sum()
                    reward_loss_num = reward_loss_num + (model.MSE(reward_pred, r_target) * mask_expanded).sum()
                    done_loss_num = done_loss_num + (model.BCE(done_pred, d_target) * mask_expanded).sum()
                    mask_elem_total += mask_expanded.sum().item()

                    all_s1_flat.append(s1_target.reshape(-1, s1_target.shape[-1]))

                if mask_elem_total == 0:
                    continue

                # Normalize by (valid_entries × feature_dim)
                obs_dim = prepared[0]["s"].shape[-1]
                reward_dim = prepared[0]["r"].shape[-1] if prepared[0]["r"].dim() >= 3 else 1
                done_dim = prepared[0]["d"].shape[-1] if prepared[0]["d"].dim() >= 3 else 1
                if _matched_on:
                    # MATCHED: mean over the UNMASKED state features (Σ w·mask), not
                    # valid_entries × obs_dim; reward/done denominators unchanged.
                    state_loss = state_loss_num / (state_w_total + 1e-8)
                else:
                    state_loss = state_loss_num / (mask_elem_total * obs_dim + 1e-8)
                reward_loss = reward_loss_num / (mask_elem_total * reward_dim + 1e-8)
                done_loss = done_loss_num / (mask_elem_total * done_dim + 1e-8)

                loss = state_loss + model.reward_coeff * reward_loss + done_loss

                # Relative error: state_loss / state_variance
                if all_s1_flat:
                    s1_cat = torch.cat(all_s1_flat, dim=0)
                    state_var = model.MSE(s1_cat, s1_cat.mean(dim=0, keepdim=True).expand_as(s1_cat)).mean()
                    rel_error = (state_loss / (state_var + 1e-7)).detach()
                else:
                    rel_error = torch.zeros((), device=self_agent.device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_rel_error += rel_error.item()
                total_loss += float(loss.item())
                last_loss = loss

            avg_rel_error = total_rel_error / self_agent._ensemble_size
            avg_loss = total_loss / self_agent._ensemble_size
            if last_loss is not None:
                self_agent.logger.log(p_loss_total=last_loss, p_update=None)
            self_agent.logger.log(model_update_time=_time.time() - time_t)
            # LOGGING FIX: return the ACTUAL training loss (state+reward+done MSE),
            # not rel_error. updateModel's return is logged as "Model loss"; returning
            # rel_error made it read ~0.000000 and look like a dead world model.
            self_agent._last_model_rel_error = avg_rel_error
            return avg_loss

        agent.updateModel = types.MethodType(ensemble_updateModel_dynamic, agent)

        # Override model_step to pass agent mask during rollouts
        def ensemble_model_step_masked(self_agent, s, a, agent_mask=None):
            """Predict with ensemble, using agent mask to prevent pad corruption."""
            with torch.no_grad():
                while s.dim() <= 2:
                    s = s.unsqueeze(0)
                    a = a.unsqueeze(0)
                while a.dim() <= 2:
                    a = a.unsqueeze(-1)
                s = s.to(self_agent.device)
                a = a.to(self_agent.device)
                if agent_mask is not None:
                    agent_mask = agent_mask.to(self_agent.device)

                all_rs, all_s1s, all_ds = [], [], []
                for model in self_agent._ensemble_models:
                    rs, s1s, ds = model._masked_predict(s, a, agent_mask=agent_mask)
                    all_rs.append(rs)
                    all_s1s.append(s1s)
                    all_ds.append(ds)

                all_rs = torch.stack(all_rs, dim=0)
                all_s1s = torch.stack(all_s1s, dim=0)
                all_ds = torch.stack([d.float() for d in all_ds], dim=0)

                mean_rs = all_rs.mean(dim=0)
                mean_s1s = all_s1s.mean(dim=0)
                mean_ds = all_ds.mean(dim=0)

                reward_std = all_rs.std(dim=0)
                penalized_rs = mean_rs - self_agent._uncertainty_penalty * reward_std

                return penalized_rs.detach(), mean_s1s.detach(), mean_ds.detach(), s.detach()

        agent.model_step = types.MethodType(ensemble_model_step_masked, agent)

        print(f"  - Model ensemble: {n_ensemble} GCNs, uncertainty_penalty={uncertainty_penalty}")
        print(f"  - Proposal 1+2: GCN edges masked for padded agents, model loss masked")

    def _setup_shared_weight_gcn(self):
        """Proposal 5: Replace per-agent GCN with shared-weight dynamic GCN.

        The original GCN has separate networks per agent slot (NodeWiseEmbedding[i],
        NodeNetwork[i], EdgeNetwork[i,j]). These are positional — slot 0's weights
        differ from slot 1's. This breaks with variable agent counts because:
        - Padded slots have their own weights that learn garbage
        - Agent identity is tied to slot position, not learned

        Fix: ONE shared embedding, ONE shared edge network, ONE shared node network.
        The GCN processes any number of agents using the same weights.
        The adjacency is rebuilt each step from the environment.
        """
        import types
        import torch
        import torch.nn as nn
        from torch.optim import Adam

        agent = self.agent
        device = self.device
        state_dim = agent.observation_dim
        action_dim = agent.action_dim
        n_agent = agent.n_agent
        p_args = agent.p_args

        node_embed_dim = p_args.node_embed_dim
        edge_embed_dim = p_args.edge_embed_dim
        edge_hidden = p_args.edge_hidden_size
        node_hidden = p_args.node_hidden_size
        reward_coeff = p_args.reward_coeff
        residual = p_args.residual

        class SharedGCN(nn.Module):
            """GCN with shared weights across all agent slots."""

            def __init__(self):
                super().__init__()
                # One shared embedding for all agents
                self.node_embedding = nn.Sequential(
                    nn.Linear(state_dim, node_embed_dim), nn.ReLU()
                )
                # One shared edge network
                edge_sizes = [node_embed_dim * 2] + edge_hidden + [edge_embed_dim]
                edge_layers = []
                for t in range(len(edge_sizes) - 1):
                    edge_layers.append(nn.Linear(edge_sizes[t], edge_sizes[t+1]))
                    if t < len(edge_sizes) - 2:
                        edge_layers.append(nn.ReLU())
                self.edge_net = nn.Sequential(*edge_layers)

                # One shared node network
                node_input_dim = edge_embed_dim + action_dim
                node_sizes = [node_input_dim] + node_hidden + [node_embed_dim]
                node_layers = []
                for t in range(len(node_sizes) - 1):
                    node_layers.append(nn.Linear(node_sizes[t], node_sizes[t+1]))
                    if t < len(node_sizes) - 2:
                        node_layers.append(nn.ReLU())
                self.node_net = nn.Sequential(*node_layers)

                # Shared prediction heads
                self.state_head = nn.Linear(node_embed_dim, state_dim)
                self.reward_head = nn.Linear(node_embed_dim, 1)
                self.done_head = nn.Sequential(
                    nn.Linear(node_embed_dim, 1), nn.Sigmoid()
                )

                self.residual = residual
                self.reward_coeff = reward_coeff
                self.n_agent = n_agent
                self.n_conv = agent.p_args.n_conv  # P3.4: effective receptive field
                self.MSE = nn.MSELoss(reduction='none')
                self.BCE = nn.BCELoss(reduction='none')

            def forward(self, s, a, adj=None, agent_mask=None):
                """Forward pass with dynamic adjacency.

                s: [batch, n_agent, state_dim]
                a: [batch, n_agent, action_dim]
                adj: [n_agent, n_agent] or None (use full connectivity)
                agent_mask: [batch, n_agent] — 1 for real, 0 for padded
                """
                batch, n, _ = s.shape
                if adj is None:
                    adj = torch.ones(n, n, device=s.device)
                    adj.fill_diagonal_(0)

                # P3.7: raise adj to κ via matrix_power so the effective
                # receptive field matches the configured κ. Previously the
                # edge check `adj[i, j] > 0` ran on raw 1-hop adjacency
                # regardless of self.n_conv. Cheap: n_agent ≤ ~30.
                if int(self.n_conv) > 1:
                    adj_kappa = (torch.matrix_power(adj.float(), int(self.n_conv)) > 0).to(adj.dtype)
                else:
                    adj_kappa = adj

                # Shared embedding for all agents
                embedding = self.node_embedding(s)  # [batch, n, node_embed_dim]

                # Edge computation — shared network for all edges
                edge_info_of_nodes = [[] for _ in range(n)]
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj_kappa[i, j] > 0:
                            si = embedding[:, i, :]  # [batch, embed_dim]
                            sj = embedding[:, j, :]
                            edge_input = torch.cat([si, sj], dim=-1)
                            edge_out = self.edge_net(edge_input)  # [batch, edge_embed_dim]
                            # Zero if either endpoint is padded
                            if agent_mask is not None:
                                mask_ij = (agent_mask[:, i] * agent_mask[:, j]).unsqueeze(-1)
                                edge_out = edge_out * mask_ij
                            edge_info_of_nodes[i].append(edge_out)
                            edge_info_of_nodes[j].append(edge_out)

                # Node update — shared network for all agents
                node_preds = []
                for i in range(n):
                    if edge_info_of_nodes[i]:
                        edge_sum = torch.stack(edge_info_of_nodes[i], dim=0).sum(dim=0)
                    else:
                        edge_sum = torch.zeros(batch, edge_embed_dim, device=s.device)
                    node_input = torch.cat([edge_sum, a[:, i, :]], dim=-1)
                    node_pred = self.node_net(node_input)
                    node_preds.append(node_pred)
                embedding = torch.stack(node_preds, dim=1)  # [batch, n, node_embed_dim]

                state_pred = self.state_head(embedding)
                if self.residual:
                    state_pred += s
                reward_pred = self.reward_head(embedding)
                done_pred = self.done_head(embedding)
                return reward_pred, state_pred, done_pred

            def predict(self, s, a, adj=None, agent_mask=None):
                with torch.no_grad():
                    r1, s1, d1 = self.forward(s, a, adj=adj, agent_mask=agent_mask)
                    done = torch.clamp(d1, 0., 1.)
                    done = torch.cat([1 - done, done], dim=-1)
                    from torch.distributions import Categorical
                    done = Categorical(done).sample() > 0
                    return r1, s1, done

            def train_step(self, s, a, r, s1, d, length=1, agent_mask=None, adj=None):
                """Train with masked loss."""
                pred_s_list, pred_r_list, pred_d_list = [], [], []
                s0 = s.select(dim=1, index=0)
                length = min(length, s.shape[1])

                for t in range(length):
                    mask_t = agent_mask[:, min(t, agent_mask.shape[1]-1), :] if agent_mask is not None else None
                    r_, s_, d_ = self.forward(s0, a.select(dim=1, index=t), adj=adj, agent_mask=mask_t)
                    pred_r_list.append(r_)
                    pred_s_list.append(s_)
                    pred_d_list.append(d_)
                    s0 = s_

                state_pred = torch.stack(pred_s_list, dim=1)
                reward_pred = torch.stack(pred_r_list, dim=1)
                done_pred = torch.stack(pred_d_list, dim=1)

                if agent_mask is not None:
                    mask_exp = agent_mask[:, :length, :].unsqueeze(-1)
                    state_loss = (self.MSE(state_pred, s1[:, :length]) * mask_exp).sum() / (mask_exp.sum() * state_pred.shape[-1] + 1e-8)
                    reward_loss = (self.MSE(reward_pred, r[:, :length]) * mask_exp).sum() / (mask_exp.sum() + 1e-8)
                    done_loss = (self.BCE(done_pred, d[:, :length].float()) * mask_exp).sum() / (mask_exp.sum() + 1e-8)
                else:
                    state_loss = self.MSE(state_pred, s1[:, :length]).mean()
                    reward_loss = self.MSE(reward_pred, r[:, :length]).mean()
                    done_loss = self.BCE(done_pred, d[:, :length].float()).mean()

                loss = state_loss + self.reward_coeff * reward_loss + done_loss

                s1_view = s1[:, :length].reshape(-1, s1.shape[-1])
                state_var = self.MSE(s1_view, s1_view.mean(dim=0, keepdim=True).expand_as(s1_view)).mean()
                rel_error = state_loss / (state_var + 1e-7)

                return loss, rel_error

        # Create ensemble of 5 SharedGCNs
        n_ensemble = 5
        uncertainty_penalty = 1.0
        ensemble_models = []
        ensemble_optimizers = []

        for i in range(n_ensemble):
            model = SharedGCN().to(device)
            optimizer = Adam(model.parameters(), lr=agent.lr_p)
            ensemble_models.append(model)
            ensemble_optimizers.append(optimizer)

        agent._ensemble_models = ensemble_models
        agent._ensemble_optimizers = ensemble_optimizers
        agent._ensemble_size = n_ensemble
        agent._uncertainty_penalty = uncertainty_penalty

        # Get current adjacency
        current_adj = torch.as_tensor(agent.adj, dtype=torch.float32, device=device)

        # Patch updateModel
        def shared_gcn_updateModel(self_agent, trajs, length=1):
            import time as _time
            time_t = _time.time()

            ss, actions, rs, s1s, ds, masks = [], [], [], [], [], []
            for traj in trajs:
                s, a, r, s1, d = traj["s"], traj["a"], traj["r"], traj["s1"], traj["d"]
                s, a, r, s1, d = [torch.as_tensor(item, device=self_agent.device) for item in [s, a, r, s1, d]]
                ss.append(s); actions.append(a); rs.append(r); s1s.append(s1); ds.append(d)
                _has_mask = ("mask" in traj.dict) if hasattr(traj, 'dict') else ("mask" in traj)
                if _has_mask:
                    _m = traj.dict["mask"] if hasattr(traj, 'dict') else traj["mask"]
                    masks.append(torch.as_tensor(_m, device=self_agent.device))
                else:
                    masks.append(torch.ones(s.shape[0], s.shape[1], device=self_agent.device))
            ss, actions, rs, s1s, ds = [torch.stack(item, dim=0) for item in [ss, actions, rs, s1s, ds]]
            masks = torch.stack(masks, dim=0)

            total_rel_error = 0.0
            total_loss = 0.0
            batch_size = ss.size(0)
            adj = torch.as_tensor(self_agent.adj, dtype=torch.float32, device=self_agent.device)

            for model, optimizer in zip(self_agent._ensemble_models, self_agent._ensemble_optimizers):
                boot_idx = torch.randint(0, batch_size, (batch_size,), device=self_agent.device)
                loss, rel_error = model.train_step(
                    ss[boot_idx], actions[boot_idx], rs[boot_idx], s1s[boot_idx], ds[boot_idx],
                    length=length, agent_mask=masks[boot_idx], adj=adj
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_rel_error += rel_error.item()
                total_loss += float(loss.item())

            avg_rel_error = total_rel_error / self_agent._ensemble_size
            avg_loss = total_loss / self_agent._ensemble_size
            self_agent.logger.log(p_loss_total=loss, p_update=None)
            self_agent.logger.log(model_update_time=_time.time() - time_t)
            # LOGGING FIX: return the ACTUAL training loss (not rel_error). The
            # training loop logs this as "Model loss"; returning rel_error made
            # it read ~0.000000 and look like a dead model. Expose both.
            self_agent._last_model_rel_error = avg_rel_error
            return avg_loss

        agent.updateModel = types.MethodType(shared_gcn_updateModel, agent)

        # Patch validateModel
        def shared_gcn_validateModel(self_agent, trajs, length=1):
            with torch.no_grad():
                ss, actions, rs, s1s, ds, masks = [], [], [], [], [], []
                for traj in trajs:
                    s, a, r, s1, d = traj["s"], traj["a"], traj["r"], traj["s1"], traj["d"]
                    s, a, r, s1, d = [torch.as_tensor(item, device=self_agent.device) for item in [s, a, r, s1, d]]
                    ss.append(s); actions.append(a); rs.append(r); s1s.append(s1); ds.append(d)
                    _has_mask = ("mask" in traj.dict) if hasattr(traj, 'dict') else ("mask" in traj)
                    if _has_mask:
                        _m = traj.dict["mask"] if hasattr(traj, 'dict') else traj["mask"]
                        masks.append(torch.as_tensor(_m, device=self_agent.device))
                    else:
                        masks.append(torch.ones(s.shape[0], s.shape[1], device=self_agent.device))
                ss, actions, rs, s1s, ds = [torch.stack(item, dim=0) for item in [ss, actions, rs, s1s, ds]]
                masks = torch.stack(masks, dim=0)
                adj = torch.as_tensor(self_agent.adj, dtype=torch.float32, device=self_agent.device)

                total_rel_error = 0.0
                for model in self_agent._ensemble_models:
                    _, rel_error = model.train_step(ss, actions, rs, s1s, ds, length=length, agent_mask=masks, adj=adj)
                    total_rel_error += rel_error.item()
                return total_rel_error / self_agent._ensemble_size

        agent.validateModel = types.MethodType(shared_gcn_validateModel, agent)

        # Patch model_step
        def shared_gcn_model_step(self_agent, s, a, agent_mask=None):
            with torch.no_grad():
                while s.dim() <= 2:
                    s = s.unsqueeze(0)
                    a = a.unsqueeze(0)
                while a.dim() <= 2:
                    a = a.unsqueeze(-1)
                s = s.to(self_agent.device)
                a = a.to(self_agent.device)
                adj = torch.as_tensor(self_agent.adj, dtype=torch.float32, device=self_agent.device)
                if agent_mask is not None:
                    agent_mask = agent_mask.to(self_agent.device)

                all_rs, all_s1s, all_ds = [], [], []
                for model in self_agent._ensemble_models:
                    rs, s1s, ds = model.predict(s, a, adj=adj, agent_mask=agent_mask)
                    all_rs.append(rs)
                    all_s1s.append(s1s)
                    all_ds.append(ds)

                all_rs = torch.stack(all_rs, dim=0)
                all_s1s = torch.stack(all_s1s, dim=0)
                all_ds = torch.stack([d.float() for d in all_ds], dim=0)

                mean_rs = all_rs.mean(dim=0)
                mean_s1s = all_s1s.mean(dim=0)
                mean_ds = all_ds.mean(dim=0)
                reward_std = all_rs.std(dim=0)
                penalized_rs = mean_rs - self_agent._uncertainty_penalty * reward_std

                return penalized_rs.detach(), mean_s1s.detach(), mean_ds.detach(), s.detach()

        agent.model_step = types.MethodType(shared_gcn_model_step, agent)

        param_count = sum(p.numel() for p in ensemble_models[0].parameters())
        print(f"  - Proposal 5: SharedGCN ensemble ({n_ensemble} models, {param_count} params each)")
        print(f"  - Shared weights: 1 embedding, 1 edge net, 1 node net (permutation invariant)")

    def _get_n_agents_from_config(self):
        """Determine number of agents from config path or CLI.

        Priority:
          1. CLI --max-ai-vehicles (if set > 0) — explicit user override
          2. Path-based parsing (e.g., '1lane_8veh_100ai' -> 8)
          3. Lane-count heuristic
          4. Default 8

        NOTE: This returns the CONFIGURED count. Actual active agents may differ.
        Use _actual_n_agents for the runtime-detected count after first episode.
        """
        # --n-agents: explicit network capacity (highest priority)
        if hasattr(self.args, 'n_agents') and self.args.n_agents > 0:
            return int(self.args.n_agents)

        # --max-ai-vehicles: if set, network matches env cap
        if hasattr(self.args, 'max_ai_vehicles') and self.args.max_ai_vehicles > 0:
            return int(self.args.max_ai_vehicles)

        import re
        config_path = self.args.config_path

        # Try to extract vehicle count from path (e.g., "8veh", "30veh")
        match = re.search(r'(\d+)veh', config_path)
        if match:
            return int(match.group(1))

        # Fallback: check for lane-based defaults
        if '1lane' in config_path:
            return 8
        elif '2lane' in config_path:
            return 30
        elif '3lane' in config_path:
            return 40
        elif '4lane' in config_path:
            return 50

        # Default fallback
        return 8

    def _detect_actual_n_agents(self):
        """Determine the policy network's n_agent capacity.

        SLOT IDENTITY FIX: Returns the configured cap (max_ai_vehicles or
        scenario default), NOT the runtime count. With per-vehicle slot
        assignment, the SAME vehicle ID can occupy slot 3 across many steps,
        but DIFFERENT vehicle IDs occupy different slots over time. The
        policy network needs enough slots to accommodate the maximum number
        of *concurrently active* vehicles, which equals max_ai_vehicles.

        Counting unique vehicle IDs across ALL timesteps would over-count —
        a 1500-step episode might see 50+ unique vehicles total but only 16
        active at any moment.
        """
        return self._get_n_agents_from_config()

    def _reinitialize_dmpo_for_n_agents(self, n_agents: int):
        """Reinitialize DMPO agent with correct number of agents.

        Called after first episode when we know the actual number of active agents.
        This ensures n_agent matches reality, preventing zero-padding issues.

        Args:
            n_agents: Actual number of active agents detected
        """
        if hasattr(self, '_dmpo_reinitialized') and self._dmpo_reinitialized:
            return  # Only reinitialize once

        current_n_agent = self.agent.n_agent if hasattr(self.agent, 'n_agent') else 0

        if current_n_agent == n_agents:
            print(f"✅ DMPO n_agent ({current_n_agent}) matches actual agents ({n_agents})")
            self._dmpo_reinitialized = True
            return

        print(f"🔄 Reinitializing DMPO: n_agent {current_n_agent} → {n_agents} (actual active agents)")

        # Use env's sparse road-topology adjacency (built from actual vehicle positions)
        if self.env.adjacency_matrix is not None and self.env.adjacency_matrix.size > 0 and self.env.adjacency_matrix.shape[0] == n_agents:
            adj_matrix = self.env.adjacency_matrix
            print(f"  - Using env sparse adjacency: {int(np.sum(adj_matrix > 0))} connections, avg {np.sum(adj_matrix > 0)/n_agents:.1f} neighbors/agent")
        else:
            communication_radius = self.args.communication_radius
            adj_matrix = self._create_spatial_adjacency_matrix(n_agents, communication_radius)
            print(f"  - Fallback to spatial adjacency: {int(np.sum(adj_matrix > 0))} connections")

        # Update config
        self.config.agent_args.adj = adj_matrix
        self.config.agent_args.n_agent = n_agents

        # Reinitialize DMPO agent
        import logging

        class DMPOLogger:
            def __init__(self, name='DMPO'):
                self.logger = logging.getLogger(name)
                self.logger.setLevel(logging.INFO)
                self.metrics = {}
            def child(self, name):
                return DMPOLogger(f"{self.logger.name}.{name}")
            def log(self, **kwargs):
                self.metrics.update(kwargs)
            def __getattr__(self, name):
                return getattr(self.logger, name)

        logger = DMPOLogger('DMPO')

        # Create new agent with correct n_agent
        self.agent = self.modules['DMPOAgent'](
            logger,
            self.device,
            self.config.agent_args,
            self.config
        )

        # H4(b): invalidate any frozen κ-closure caches so they are recomputed
        # from the NEW adjacency. A fresh agent never has these attributes
        # (they are lazily built inside the patched attention paths), so this is
        # hasattr-GUARDED — an unguarded del would raise AttributeError here
        # (verifier correction #4). Gated: legacy behaviour untouched.
        if os.environ.get('DMPO_ADJ_FIX', '0') == '1':
            for _cache_attr in ('_adj_kappa_pi', '_adj_kappa_v'):
                if hasattr(self.agent, _cache_attr):
                    delattr(self.agent, _cache_attr)

        # Re-apply value loss fix after reinitialization
        self._patch_agent_value_loss()

        # Shared attention policy (optional — default is per-agent networks from original framework)
        if os.environ.get('DMPO_SHARED_POLICY', '0') == '1':
            self._patch_shared_attention_policy()

        # Re-setup model ensemble
        if os.environ.get('DMPO_SHARED_GCN', '0') == '1':
            self._setup_shared_weight_gcn()
        else:
            self._setup_model_ensemble()

        # Reinitialize ZFilter for new agent count (stays None under OBS-V2 no-norm)
        obs_dim = self.config.agent_args.observation_dim
        if getattr(self, '_obs_v2_nonorm', False):
            self.running_state = None
        else:
            self.running_state = ZFilter((n_agents, obs_dim), clip=5.0)
            print(f"  - ZFilter reinitialized: shape=({n_agents}, {obs_dim})")

        # SAC-style automatic entropy tuning (Haarnoja et al. 2018)
        # log_alpha is a learnable parameter; target_entropy = -action_dim
        action_dim = self.config.agent_args.action_space.shape[0]
        self.agent._log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.agent._target_entropy = -float(action_dim)  # -2 for 2-dim actions
        self.agent._alpha_optimizer = torch.optim.Adam([self.agent._log_alpha], lr=3e-4)
        print(f"  - SAC entropy: log_alpha=0.0 (alpha=1.0), target_entropy={self.agent._target_entropy}")

        # Re-apply initial-policy bias on the freshly-instantiated actors
        # (curriculum reinitialization replaces self.agent.actors). Without
        # this re-application the bias would only affect iter 0; subsequent
        # curriculum stages would start from unbiased random init.
        self._apply_pad_shared()
        self._apply_set_actor()
        self._apply_central_critic()
        self._apply_policy_bias()
        self._apply_output_tanh()
        self._apply_weight_decay()

        # WARM-START (gated DMPO_INIT_FROM=<ckpt.pt>, default unset = no-op;
        # g1b 2026-08-06): load policy/value weights from an existing checkpoint
        # AFTER reinitialization + bias/tanh patches, so the loaded weights are
        # the true starting point (fresh optimizer = warm-start semantics, not
        # resume). Loud-fail on obs-dim mismatch via load_state_dict shapes.
        self._apply_warm_start()

        print(f"✅ DMPO agent reinitialized with n_agent={n_agents}")
        self._dmpo_reinitialized = True

    def _apply_warm_start(self):
        """WARM-START (gated DMPO_INIT_FROM=<ckpt.pt>, default unset = no-op).

        Loads policy/value weights from an existing checkpoint with a FRESH
        optimizer (warm start, not resume). Called from BOTH agent-construction
        paths: the main initializer and the curriculum reinitializer. Idempotent
        via _warm_started, so a later curriculum reinit does not silently revert
        the loaded weights to random ones.
        """
        _wf = os.environ.get('DMPO_INIT_FROM', '').strip()
        if not _wf or getattr(self, '_warm_started', False):
            return
        _ck = torch.load(_wf, map_location=self.device, weights_only=False)
        _cd = _ck.get('obs_dim')
        _rd = int(self.config.agent_args.observation_dim)
        if _cd is not None and int(_cd) != _rd:
            raise RuntimeError(f'DMPO_INIT_FROM obs_dim {_cd} != run obs_dim {_rd}')
        self.agent.actors.load_state_dict(_ck['policy_net_legacy'])
        _v = _ck.get('value_net_legacy')
        if _v:
            try:
                self.agent.vs.load_state_dict(_v)
            except Exception as _e:
                print(f"⚠️ DMPO_INIT_FROM: value net not loaded ({_e}); actors loaded")
        self._warm_started = True
        print(f"🔥 WARM-START: policy initialized from {_wf} (obs_dim {_rd})", flush=True)

    def _create_spatial_adjacency_matrix(self, n_agents: int, communication_radius: float) -> np.ndarray:
        """Create sparse adjacency matrix based on spatial proximity.

        FIX Issue 30: Instead of fully-connected graph, create meaningful
        local neighborhoods based on vehicle spawn positions.

        For intersection scenarios, vehicles spawn on 4 approach roads.
        Agents on the same road or nearby roads should be connected.

        Args:
            n_agents: Number of AI agents
            communication_radius: Max distance (meters) for 1-hop neighbors

        Returns:
            Sparse adjacency matrix where adj[i,j] = 1 if agents i,j are within radius
        """
        # Initialize with self-connections (diagonal)
        adj_matrix = np.eye(n_agents, dtype=np.float32)

        # For intersection scenarios, we create structured neighborhoods
        # based on road approach. In a 4-way intersection with n_agents:
        # - Roughly n_agents/4 vehicles per approach road
        # - Vehicles on same approach road are connected (same "lane group")
        # - Vehicles on adjacent approaches (turning conflict) are connected

        agents_per_approach = max(1, n_agents // 4)

        for i in range(n_agents):
            approach_i = i // agents_per_approach  # 0=North, 1=South, 2=East, 3=West

            for j in range(n_agents):
                if i == j:
                    continue

                approach_j = j // agents_per_approach

                # Same approach road - definitely connected (convoy)
                if approach_i == approach_j:
                    adj_matrix[i, j] = 1.0

                # Adjacent approaches (potential conflict at intersection)
                # N(0) conflicts with E(2), W(3); S(1) conflicts with E(2), W(3)
                # E(2) conflicts with N(0), S(1); W(3) conflicts with N(0), S(1)
                elif abs(approach_i - approach_j) != 2:  # Not opposite approaches
                    # Probabilistic connection for adjacent approaches
                    # Closer agent indices on different approaches = more likely connected
                    position_diff = abs((i % agents_per_approach) - (j % agents_per_approach))
                    if position_diff <= 2:  # Within 2 positions
                        adj_matrix[i, j] = 1.0

        # Ensure symmetry
        adj_matrix = np.maximum(adj_matrix, adj_matrix.T)

        # Log adjacency statistics
        connections = adj_matrix.sum() - n_agents  # Exclude self-connections
        avg_neighbors = connections / n_agents
        print(f"  - Sparse adjacency: {int(connections)} connections, avg {avg_neighbors:.1f} neighbors/agent")

        return adj_matrix

    def _initialize_dmpo_components(self):
        """Initialize DMPO neural network components."""
        print("🧠 Initializing DMPO neural network components...")

        # Determine correct n_agents from config path
        n_agents = self._get_n_agents_from_config()
        print(f"  - Scenario vehicle count: {n_agents}")

        # FIX Issue 30: Create sparse adjacency matrix based on κ-hop communication radius
        # Previous fully-connected (np.ones) defeated κ-hop purpose:
        # - Every agent saw every other agent at 1-hop
        # - 10-neighbor limit caused random subset selection each episode
        # - Policy learned for "10 random neighbors" not "κ-hop spatial neighbors"
        #
        # Solution: Create a sparse adjacency based on initial vehicle positions
        # Each agent connects to agents within communication_radius (meters)
        # This creates meaningful local neighborhoods that κ-hop can expand
        communication_radius = self.args.communication_radius
        if self.config.agent_args.dynamic and os.environ.get('DMPO_ADJ_FIX', '0') == '1':
            # H4(b) (learner-fixes spec, GATED on dynamic mode per verifier
            # correction #1): in dynamic mode the policy/value use IdentityCollect
            # and _process_traj does no cross-agent reduction, so the index-
            # arithmetic "spatial" adjacency below is fiction that is never a real
            # communication graph (slot index != approach; no slots at all in
            # dynamic mode). Use the identity — DPPOAgent still needs a 2-D
            # tensor for self.adj, and any DMPO_SHARED_GCN=1 world model then
            # sees each node predicting from itself instead of a fictional graph.
            # STATIC mode keeps _create_spatial_adjacency_matrix untouched: there
            # adj sizes every per-agent network via MultiCollect degrees, and an
            # eye would break all legacy static checkpoints.
            adj_matrix = np.eye(n_agents, dtype=np.float32)
            print(f"  - H4 ADJ FIX: dynamic mode → identity adjacency "
                  f"(index-arithmetic spatial fiction disabled)")
        else:
            adj_matrix = self._create_spatial_adjacency_matrix(n_agents, communication_radius)

        self.config.agent_args.adj = adj_matrix
        self.config.agent_args.n_agent = adj_matrix.shape[0]

        print(f"  - Adjacency matrix shape: {adj_matrix.shape}")
        print(f"  - Number of agents: {self.config.agent_args.n_agent}")
        print(f"  - Communication: sparse spatial adjacency (κ-hop expands local neighborhoods)")
        
        try:
            # Create a compatible logger for DMPO agent (required by Networked-MB-MARL)
            import logging

            class DMPOLogger:
                """Logger wrapper that supports child() method required by DMPO."""
                def __init__(self, name='DMPO'):
                    self.logger = logging.getLogger(name)
                    self.logger.setLevel(logging.INFO)
                    self.metrics = {}

                def child(self, name):
                    """Create a child logger."""
                    return DMPOLogger(f"{self.logger.name}.{name}")

                def log(self, **kwargs):
                    """Log metrics."""
                    self.metrics.update(kwargs)

                def __getattr__(self, name):
                    """Forward other calls to underlying logger."""
                    return getattr(self.logger, name)

            logger = DMPOLogger('DMPO')

            # Initialize DMPO agent with neural networks
            # Signature: __init__(self, logger, device, agent_args, env_args, **kwargs)
            self.agent = self.modules['DMPOAgent'](
                logger,                  # logger (required)
                self.device,             # device
                self.config.agent_args,  # agent_args
                self.config              # env_args
            )

            print("✅ DMPO agent initialized with neural networks")

            # Patch updateAgent to fix value loss explosion
            self._patch_agent_value_loss()

            # Shared attention policy (optional — default is per-agent networks from original framework)
            if os.environ.get('DMPO_SHARED_POLICY', '0') == '1':
                self._patch_shared_attention_policy()

            # Setup model ensemble (5 GCNs) with uncertainty penalty
            if os.environ.get('DMPO_SHARED_GCN', '0') == '1':
                self._setup_shared_weight_gcn()
            else:
                self._setup_model_ensemble()

            # Initialize proper neural networks
            self._initialize_policy_networks()
            self._initialize_value_networks()
            self._initialize_model_networks()
            
            # Initialize training buffers
            self._initialize_training_buffers()
            
            # Initialize multi-agent communication
            self._initialize_communication_networks(adj_matrix)
            
            self._apply_warm_start()

            print("✅ All DMPO components initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize DMPO components: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"DMPO agent initialization failed - cannot proceed: {e}")
    
    def _initialize_policy_networks(self):
        """Verify policy networks are available via DMPO agent.

        DMPO agent contains self.actors (nn.ModuleList of GaussianActor networks).
        These are the ONLY policy networks - no standalone networks needed.
        """
        print("🎯 Verifying policy networks via DMPO agent...")

        if not self.agent:
            raise RuntimeError("DMPO agent not initialized - cannot verify policy networks")

        if not hasattr(self.agent, 'actors'):
            raise RuntimeError("DMPO agent missing 'actors' (policy networks)")

        n_actors = len(self.agent.actors)
        print(f"  - DMPO policy networks (actors): {n_actors}")
        print(f"  - Policy learning rate: {self.config.agent_args.lr}")
        print(f"  - Policy optimizer: {type(self.agent.optimizer_pi).__name__}")
        print("✅ Policy networks verified via DMPO agent")

        # Apply optional initial-policy bias (DMPO_POLICY_BIAS_LONGITUDINAL)
        # — see _apply_policy_bias docstring for rationale.
        self._apply_pad_shared()
        self._apply_set_actor()
        self._apply_central_critic()
        self._apply_policy_bias()
        self._apply_output_tanh()
        self._apply_weight_decay()

    def _apply_pad_shared(self):
        """MATCHED PADDED ARM (gated DMPO_PAD_SHARED=1, default 0 = no-op; padded learning baseline):
        in padded mode (DMPO_DYNAMIC=0) the framework builds one actor and one critic PER
        SLOT — so the legacy padded twin differed from padding-free in parameter sharing and
        capacity, not only in padding. This ties every slot to slot 0's actor and critic
        (one shared network each, as in the dynamic formulation) and rebuilds both
        optimizers on the tied parameters. At --kappa 0 each slot reads only its own
        observation, so the input width also matches. The ONLY remaining delta is the
        fixed-capacity padded tensors + masks."""
        if os.environ.get('DMPO_PAD_SHARED', '0') != '1' or not hasattr(self.agent, 'actors'):
            return
        if os.environ.get('DMPO_DYNAMIC', '0') == '1':
            raise RuntimeError("DMPO_PAD_SHARED applies to the padded path only (DMPO_DYNAMIC=0)")
        import torch.nn as nn
        a0, v0 = self.agent.actors[0], self.agent.vs[0]
        shapes = {tuple(p.shape for p in a.parameters()) for a in self.agent.actors}
        if len(shapes) != 1:
            raise RuntimeError("DMPO_PAD_SHARED needs identical slot networks (run with --kappa 0)")
        n = len(self.agent.actors)
        self.agent.actors = nn.ModuleList([a0] * n)
        self.agent.vs = nn.ModuleList([v0] * len(self.agent.vs))
        self.agent.optimizer_pi = torch.optim.Adam(self.agent.actors.parameters(), lr=self.agent.lr)
        self.agent.optimizer_v = torch.optim.Adam(self.agent.vs.parameters(), lr=self.agent.lr_v)
        _np = sum(p.numel() for p in self.agent.actors.parameters())
        print(f"🔗 PAD-SHARED: {n} padded slots tied to ONE actor ({_np:,} params) and ONE critic; "
              f"optimizers rebuilt")

    def _apply_set_actor(self):
        """SET-ACTOR BASELINES (gated DMPO_SET_ACTOR=attn|mean, default 'off' = no-op;
        learning baselines): swap the actor trunk for ego-attention
        (Leurent & Mercat 2019) or Deep Sets pooling over the perceived-neighbour set.
        Head/log_std/tanh/bias/critic untouched — see utils/set_actor.py. Must run
        BEFORE bias/tanh/weight-decay; the policy optimizer is rebuilt on the new
        parameters (the agent's own was built over the replaced MLP trunk)."""
        from dmpo_autonomous_driving.utils import set_actor as _sa
        _mode = _sa.mode_from_env()
        if _mode == 'off' or not hasattr(self.agent, 'actors'):
            return
        _n = _sa.apply(self.agent.actors, int(self.config.agent_args.observation_dim),
                       self.device, _mode)
        self.agent.optimizer_pi = torch.optim.Adam(self.agent.actors.parameters(), lr=self.agent.lr)
        print(f"🧩 SET-ACTOR '{_mode}': trunk swapped on {len(self.agent.actors)} actors "
              f"({_n:,} trunk params each); optimizer_pi rebuilt")

    def _apply_central_critic(self):
        """MAPPO BASELINE (gated DMPO_CENTRAL_CRITIC=1, default 0 = no-op; learning
        baseline): critic V(o_i, g_t) on the agent-specific global state, actor unchanged
        (reads only o_i). See utils/central_critic.py. Runs AFTER _apply_set_actor so
        the input slice wraps the final trunk. MAPPO has no credit routing → --kappa 0
        (the κ-adv operator is disabled at κ=0, so DMPO_KAPPA_ADV is inert there)."""
        from dmpo_autonomous_driving.utils import central_critic as _cc
        if not _cc.enabled() or not hasattr(self.agent, 'vs'):
            return
        if int(getattr(self.args, 'kappa', 2)) != 0:
            raise RuntimeError("DMPO_CENTRAL_CRITIC (MAPPO baseline) requires --kappa 0")
        _d = int(self.config.agent_args.observation_dim)
        _n = _cc.apply(self.agent, _d, self.device)
        print(f"🌐 CENTRAL CRITIC (MAPPO): V(o_i, g_t) input {_d}+{_cc.g_dim(_d)}, "
              f"{_n:,} params; optimizer_v rebuilt; actors sliced to first {_d} dims")

    def _apply_output_tanh(self):
        """Bound the GaussianActor's mean output to [-1, 1] via tanh.

        Without this, nothing in the network prevents the mean from growing
        arbitrarily large. Since the env clips the executed action to [-1, 1],
        the clipped action is monotonic in the reward signal (more throttle
        → more completion bonus) — PPO's gradient therefore pushes
        action_head weights toward infinity, producing a degenerate policy
        whose raw output is 10^3 to 10^4 but always clips to ±1 with a fixed
        sign pattern regardless of observation.

        Diagnostic (after B/C-v2 runs) showed both policies produced:
            raw mean = [+1094, +1188, -1421]  (zero obs)
            raw mean = [+3179, +3453, -4129]  (typical obs)
        All clipped identically to [+1, +1, -1]. Policy is observation-
        INVARIANT after clipping — no actual learning of coordination,
        just reward hacking (accelerate always, rely on teleport recovery).

        Applying tanh bounds the mean natively, so the gradient has to
        produce meaningful within-range discrimination instead of grinding
        weights toward saturation. Controlled by env var DMPO_TANH_POLICY
        (default '1' = on). Set '0' to disable.
        """
        if os.environ.get('DMPO_TANH_POLICY', '1') != '1':
            return
        if not hasattr(self.agent, 'actors'):
            return
        import types as _types
        # ── RESIDUAL-FROM-PRIOR (round 3): mean = base + scale·tanh(head) ──
        # Round-2 forensic: outcomes were decided by the INIT constant and frozen by
        # tanh saturation; the reward gradient never changed driving behavior. So make
        # the safe prior the FIXED base and learn only a bounded deviation Δ. Per-dim
        # base/scale chosen so base+scale ≤ 1 (mean stays in [-1,1] natively) and the
        # yield dim CAN cross 0 (else the shield could never disengage — the whole point).
        _residual_on = os.environ.get('DMPO_RESIDUAL', '0') == '1'
        if _residual_on:
            _base = torch.tensor([float(x) for x in os.environ.get(
                'DMPO_RESIDUAL_BASE', '0.5,0.0,0.0,0.4').split(',')],
                dtype=torch.float32, device=self.device)
            _scale = torch.tensor([float(x) for x in os.environ.get(
                'DMPO_RESIDUAL_SCALE', '0.5,0.3,0.3,0.6').split(',')],
                dtype=torch.float32, device=self.device)
            assert bool(((_base.abs() + _scale) <= 1.0 + 1e-6).all()), \
                "residual base+scale must be ≤1 per dim (mean must stay in [-1,1])"
        n_patched = 0
        for actor in self.agent.actors:
            if not hasattr(actor, 'action_head') or not hasattr(actor, 'log_std'):
                continue
            if _residual_on:
                # small-init the head so Δ≈0 at start → policy begins EXACTLY at the
                # measured base-null; also keeps the head far from tanh saturation.
                with torch.no_grad():
                    actor.action_head.weight.mul_(0.01)
                    actor.action_head.bias.zero_()
            # Capture the original forward for log_prob path; only touch 'acting'.
            def _tanh_forward(self_actor, obs, a=None):
                output = self_actor.network(obs)
                if _residual_on:
                    mean = _base + _scale * torch.tanh(self_actor.action_head(output))
                else:
                    mean = torch.tanh(self_actor.action_head(output))
                clamped = torch.clamp(self_actor.log_std,
                                      self_actor.LOG_STD_MIN,
                                      self_actor.LOG_STD_MAX)
                std = torch.exp(clamped).expand(*mean.shape)
                if a is None:
                    return mean, std
                from torch.distributions import Normal
                return Normal(mean, std).log_prob(a).sum(dim=-1)
            actor.forward = _types.MethodType(_tanh_forward, actor)
            n_patched += 1
        if _residual_on:
            print(f"🧩 RESIDUAL policy: mean = base + scale·tanh(head) on {n_patched} actors; "
                  f"base={_base.tolist()} scale={_scale.tolist()} (head small-init ×0.01)")
        else:
            print(f"🎛 Output tanh applied: policy mean now bounded to [-1, 1] on {n_patched} actors")

    def _apply_weight_decay(self):
        """L2 weight decay on the policy optimizer — stops the action-head pre-tanh from
        blowing up to ±30 (the saturation collapse that made the policy observation-invariant).
        Env: DMPO_WEIGHT_DECAY (default 0 = off)."""
        wd = float(os.environ.get('DMPO_WEIGHT_DECAY', '0'))
        if wd > 0 and self.agent is not None and hasattr(self.agent, 'optimizer_pi'):
            lr = getattr(self.agent, 'lr', 1e-4)
            params = [p for g in self.agent.optimizer_pi.param_groups for p in g['params']]
            self.agent.optimizer_pi = torch.optim.Adam(params, lr=lr, weight_decay=wd)
            print(f"🧲 Policy weight decay applied: weight_decay={wd} (anti-saturation, {len(params)} param tensors)")

    def _apply_policy_bias(self):
        """Bias the GaussianActor's longitudinal output toward driving forward.

        Diagnostic showed that with default Linear-layer init, different
        random seeds produce different action-mean signs on action[0]
        (longitudinal). For seeds whose mean happens to be slightly
        negative, the policy initially BRAKES, vehicles never reach the
        intersection, no completions happen, and PPO has no positive
        learning signal to escape. This is a per-seed local-optimum
        lock-in ("Deviation 11").

        Fix: add a domain prior — bias action[0] (longitudinal) toward
        positive values so that, regardless of seed, the initial policy
        defaults to driving forward. This is a standard initial-policy
        biasing technique in robotics RL.

        Controlled by env var DMPO_POLICY_BIAS_LONGITUDINAL (default 0.0
        = no bias). A value like 0.3 corresponds to a target acceleration
        of ~+0.78 m/s² (positive accel).
        """
        if os.environ.get('DMPO_RESIDUAL', '0') == '1':
            # Residual mode: the prior lives in DMPO_RESIDUAL_BASE, and the head is
            # small-init'd to Δ≈0. Adding head bias here would displace the policy
            # from the measured base-null — skip.
            print("🧩 policy-bias skipped (residual mode: prior carried by DMPO_RESIDUAL_BASE)")
            return
        bias_long = float(os.environ.get('DMPO_POLICY_BIAS_LONGITUDINAL', '0.0'))
        # Additional prior on the yield_intent action dim (action[3], yield_intent
        # variant only). Used by G_yield_bias to encourage the policy to start
        # from a yield-prone init instead of the random-init zero-mean.
        bias_yield = float(os.environ.get('DMPO_POLICY_BIAS_YIELD_INTENT', '0.0'))
        if bias_long == 0.0 and bias_yield == 0.0:
            return  # No bias — preserve original behavior
        if not hasattr(self.agent, 'actors'):
            return
        with torch.no_grad():
            n_biased = 0
            for actor in self.agent.actors:
                if hasattr(actor, 'action_head') and hasattr(actor.action_head, 'bias'):
                    # action_head: nn.Linear(output_size, action_dim). bias[0]
                    # is the longitudinal output bias. Zero out the existing
                    # (random-init) bias and add our prior so the seed-induced
                    # variation on dim 0 is replaced by a deterministic prior.
                    if bias_long != 0.0:
                        actor.action_head.bias[0] = float(bias_long)
                    if bias_yield != 0.0 and actor.action_head.bias.shape[0] >= 4:
                        # action[3] is yield_intent under DMPO_ACTION_VARIANT=yield_intent
                        actor.action_head.bias[3] = float(bias_yield)
                    n_biased += 1
        print(f"🎯 Initial policy bias applied: long={bias_long:+.3f} "
              f"yield={bias_yield:+.3f} on {n_biased} actors")

    def _apply_bias_anneal(self, iteration):
        """V8: anneal the longitudinal cold-start bias toward 0 over training.

        The bias (Deviation 11 fix) is needed EARLY to escape the per-seed
        "brake forever, never finish" basin. But a *permanent* forward bias
        prevents the policy from ever learning to brake (V7's residual creep).

        Resolution: treat action_head.bias[0] as a SCHEDULED exploration prior,
        not a learned parameter. Each iteration we re-set it to:
            bias(it) = start * max(0, 1 - it/anneal_iters)
        The network learns the real longitudinal response on top of the prior
        (through action_head.weight and every upstream layer); as the prior
        decays to 0, the learned response takes over and the policy is free to
        command negative longitudinal (brake).

        PRECISE BEHAVIOR (empirically verified 2026-07-25, anneal_iters=5:
        0.300→0.240→0.180→0.120→0.060→0.000): for iteration >= anneal_iters the
        schedule value is 0.0 and this function KEEPS RE-SETTING bias[0]=0.0 on
        every subsequent iteration — the bias term stays PERMANENTLY PINNED at
        zero for the rest of training. That is deliberate and desirable: the
        converged policy then provably contains NO hand-set prior at all (the
        publication guarantee), and expressiveness is not lost because the
        longitudinal response is still free through action_head.weight and the
        upstream biases. Earlier wording here implied bias[0] becomes a learned
        parameter after the window; it does not.

        Gated by DMPO_POLICY_BIAS_ANNEAL_ITERS (0/unset = no annealing = constant
        bias = exact V7 behavior). Returns the current bias value for logging.
        """
        anneal_iters = int(os.environ.get('DMPO_POLICY_BIAS_ANNEAL_ITERS', '0'))
        if anneal_iters <= 0:
            return None  # annealing disabled — bias stays as initialized (V7)
        start_long = float(os.environ.get('DMPO_POLICY_BIAS_LONGITUDINAL', '0.0'))
        if start_long == 0.0:
            return None
        cur = start_long * max(0.0, 1.0 - iteration / anneal_iters)
        if not hasattr(self.agent, 'actors'):
            return None
        with torch.no_grad():
            for actor in self.agent.actors:
                if hasattr(actor, 'action_head') and hasattr(actor.action_head, 'bias'):
                    actor.action_head.bias[0] = float(cur)
        return cur

    def _initialize_value_networks(self):
        """Verify value networks are available via DMPO agent.

        DMPO agent contains self.vs (nn.ModuleList of MLP value networks).
        These are the ONLY value networks - no standalone networks needed.
        """
        print("💎 Verifying value networks via DMPO agent...")

        if not self.agent:
            raise RuntimeError("DMPO agent not initialized - cannot verify value networks")

        if not hasattr(self.agent, 'vs'):
            raise RuntimeError("DMPO agent missing 'vs' (value networks)")

        n_vs = len(self.agent.vs)
        print(f"  - DMPO value networks (vs): {n_vs}")
        print(f"  - Value learning rate: {self.config.agent_args.lr_v}")
        print(f"  - Value optimizer: {type(self.agent.optimizer_v).__name__}")
        print("✅ Value networks verified via DMPO agent")
    
    def _initialize_model_networks(self):
        """Verify environment model networks are available via DMPO agent.

        DMPO agent contains self.ps (GraphConvolutionalModel for world model).
        This is the ONLY model network - no standalone networks needed.
        """
        print("🌍 Verifying environment model networks via DMPO agent...")

        if not self.agent:
            raise RuntimeError("DMPO agent not initialized - cannot verify model networks")

        if not hasattr(self.agent, 'ps'):
            raise RuntimeError("DMPO agent missing 'ps' (world model network)")

        print(f"  - DMPO world model (ps): GraphConvolutionalModel")
        print(f"  - Model learning rate: {self.config.agent_args.lr_p}")
        print(f"  - Model optimizer: {type(self.agent.optimizer_p).__name__}")
        print("✅ Environment model verified via DMPO agent")
    
    def _initialize_communication_networks(self, adj_matrix):
        """Verify communication networks are set up via DMPO agent.

        DMPO uses collect_pi and collect_v (MultiCollect) for κ-hop communication.
        These handle neighbor observation gathering internally.
        """
        print("📡 Verifying communication networks via DMPO agent...")

        if not self.agent:
            raise RuntimeError("DMPO agent not initialized - cannot verify communication")

        # Store adjacency for reference
        self.adjacency_matrix = torch.FloatTensor(adj_matrix).to(self.device)
        self.communication_radius = self.config.agent_args.radius_v

        # DMPO handles communication through MultiCollect
        if hasattr(self.agent, 'collect_pi'):
            print(f"  - Policy communication (collect_pi): MultiCollect")
        if hasattr(self.agent, 'collect_v'):
            print(f"  - Value communication (collect_v): MultiCollect")

        print(f"  - κ-hop radius: {self.communication_radius}")
        print(f"  - Adjacency matrix shape: {adj_matrix.shape}")
        print("✅ Communication verified via DMPO agent")
    
    def _initialize_training_buffers(self):
        """Initialize proper DMPO training buffers."""
        print("💾 Initializing DMPO training buffers...")
        
        # Experience buffer for trajectory data
        # SLOT IDENTITY FIX (B1 proper): Added 'vehicle_slots' — per-timestep
        # mapping of vehicle_id → slot index, used by _prepare_dmpo_batch to
        # place each vehicle's data in its assigned (stable) slot.
        self.experience_buffer = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'advantages': [],
            'returns': [],
            'dones': [],
            'vehicle_slots': [],  # list of dict[vehicle_id, slot_idx] per timestep
            'costs': [],          # constrained-RL: per-step binary collision indicator
            'cost_values': [],    # constrained-RL: V_C(s) predictions
            # KAPPA-ADV (gated DMPO_KAPPA_ADV): the ACTION-TIME κ-graph per step —
            # adjacency captured BEFORE env.step (rows ordered by adj_ids) so the
            # learner can κ-hop-aggregate advantages (the original framework's
            # reduce_sum, restored for the dynamic graph).
            'adj_pre': [],
            'adj_ids': [],
            # MATCHED MODEL LOSS (gated DMPO_MODEL_MATCHED_LOSS): per-step dict
            # vid → tuple[str|None] of neighbour ids in slot order (None when off).
            'slot_ids': [],
        }
        self._kappa_adv_on = os.environ.get('DMPO_KAPPA_ADV', '0') == '1'

        # Constrained-RL state: Lagrangian dual variable (PPO-Lagrangian, Ray et al. 2019).
        # λ ≥ 0 is updated after each iteration based on (mean_cost - cost_budget).
        # Default cost_budget=0 → strict zero-collision constraint.
        self._constrained_rl = os.environ.get('DMPO_CONSTRAINED_RL', '0') == '1'
        self._cost_budget = float(os.environ.get('DMPO_COST_BUDGET', '0.0'))
        self._lambda_lr = float(os.environ.get('DMPO_LAMBDA_LR', '0.05'))
        self._lambda_max = float(os.environ.get('DMPO_LAMBDA_MAX', '100.0'))
        self._lambda_var = float(os.environ.get('DMPO_LAMBDA_INIT', '0.0'))
        
        # Model buffer for environment model learning
        self.model_experience = {
            'states': [],
            'actions': [],
            'next_states': [], 
            'rewards': [],
            'dones': []
        }
        
        # Communication buffer for κ-hop information sharing
        # H4 NOTE (learner-fixes spec, verified 2026-07): comm_buffer is
        # DIAGNOSTIC-ONLY — it has ZERO readers (only init/clear/append exist).
        # The env's real per-step adjacency stored here never reaches the
        # learner; consuming it for κ-hop advantage aggregation is the deferred
        # H4(a) Stage-4 arm (DMPO_KAPPA_ADV), not implemented here.
        self.comm_buffer = {
            'neighbor_obs': [],
            'adjacency_matrices': [],
            'communication_masks': []
        }
        
        # Training statistics
        self.training_stats = {
            'policy_losses': [],
            'value_losses': [],
            'model_losses': [],
            'rewards': [],
            'episode_lengths': []
        }

        # Initialize ModelBuffer for accumulating real trajectories (Paper Alg 2, D_E)
        # This is the environment buffer that stores real trajectories across iterations
        # for model training. The model ONLY trains on data from this buffer.
        model_buffer_size = getattr(self.config, 'model_buffer_size', 20)
        # GPU-MEMORY override: ModelBuffer stores trajectories on GPU. At large
        # n_episodes, 20 epochs of data exceeds GPU memory (OOM). DMPO_MODEL_BUFFER_SIZE
        # caps it so the on-GPU buffer fits (e.g. 3 epochs for n_episodes=500).
        _mbs = int(os.environ.get('DMPO_MODEL_BUFFER_SIZE', '0'))
        if _mbs > 0:
            model_buffer_size = _mbs
            print(f"  ⚙️  ModelBuffer capacity overridden to {_mbs} epochs (DMPO_MODEL_BUFFER_SIZE) — fits GPU at high n_episodes")
        if 'ModelBuffer' in self.modules:
            self.model_buffer = self.modules['ModelBuffer'](model_buffer_size)
            print(f"📦 ModelBuffer initialized (capacity: {model_buffer_size} epochs)")
        else:
            print("⚠️ ModelBuffer not available — model will train on current iteration data only")

        print("✅ Training buffers initialized")
    

    def _apply_bc_anchor_loss(self) -> None:
        """Apply a one-step BC gradient step pulling policy toward anchor actions.

        Implementation note: rather than entangle with the framework's PPO update
        (which has its own optimizer state), we run a separate small SGD step on
        the actor only, scaled by self.bc_loss_weight. This is functionally
        equivalent to adding the BC term to the PPO loss and simpler to integrate.
        """
        if self.bc_anchor is None or len(self.bc_anchor.episodes) == 0:
            return
        # Find the actor(s) — dynamic mode uses agent.actors as a Sequential of one
        actor = None
        if hasattr(self.agent, 'actors') and len(self.agent.actors) > 0:
            actor = self.agent.actors[0]
        elif hasattr(self.agent, 'actor'):
            actor = self.agent.actor
        if actor is None:
            return

        # Sample BC batch
        obs_b, act_b = self.bc_anchor.sample(self.bc_batch_size, device=self.device)
        if obs_b is None:
            return

        # Forward: get action means
        try:
            # Different actor APIs — try a few common ones
            if hasattr(actor, 'action_head'):
                features = actor.network(obs_b) if hasattr(actor, 'network') else actor(obs_b)
                if isinstance(features, tuple):
                    features = features[0]
                if hasattr(actor, 'action_head'):
                    predicted_mean = actor.action_head(features)
                else:
                    predicted_mean = features
            else:
                out = actor(obs_b)
                if hasattr(out, 'mean'):
                    predicted_mean = out.mean
                elif isinstance(out, tuple):
                    predicted_mean = out[0]
                else:
                    predicted_mean = out
        except Exception:
            return

        if predicted_mean.shape != act_b.shape:
            return  # shape mismatch — skip
        loss = self.bc_loss_weight * ((predicted_mean - act_b) ** 2).mean()
        # Backward + single step using actor's existing optimizer if available
        if hasattr(self.agent, 'optimizer_a') and self.agent.optimizer_a is not None:
            try:
                self.agent.optimizer_a.zero_grad()
                loss.backward()
                self.agent.optimizer_a.step()
            except Exception:
                pass

    def _apply_safety_shield(self, actions: dict, observations: dict) -> dict:
        """DMPO-Safe: filter per-agent actions through the safety shield.

        For each tracked AI vehicle, the shield uses the world model to predict
        the next h steps under the proposed action. If a collision is predicted
        with high confidence, the action is overridden with a safe (braking) one.

        Falls back to the original action on any error or when the shield is
        inactive (e.g., during early training before world model is mature).

        Args:
            actions: dict mapping vehicle_id → action tensor/array
            observations: dict mapping vehicle_id → observation tensor

        Returns:
            possibly-modified actions dict
        """
        if not actions or not observations:
            return actions
        cur_iter = getattr(self, '_current_iteration', 0)
        if not self.safety_shield.is_active(cur_iter):
            return actions

        # World model handle: from agent if available
        ensemble = None
        if hasattr(self.agent, '_ensemble_models'):
            ensemble = self.agent
        elif hasattr(self.agent, 'ps'):
            ensemble = self.agent.ps

        if ensemble is None:
            return actions

        adjacency = None
        if hasattr(self.env, 'adjacency_matrix') and self.env.adjacency_matrix is not None:
            try:
                adjacency = torch.from_numpy(self.env.adjacency_matrix).float().to(self.device)
            except Exception:
                adjacency = None

        out = {}
        for vid, action in actions.items():
            try:
                obs_v = observations.get(vid)
                if obs_v is None:
                    out[vid] = action
                    continue
                # Tensorize
                obs_t = torch.as_tensor(obs_v, dtype=torch.float32, device=self.device).unsqueeze(0)
                act_t = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
                # Shield decides
                exec_act, was_override, info = self.safety_shield.filter_action(
                    proposed_action=act_t,
                    state=obs_t,
                    ensemble_model=ensemble,
                    adjacency=adjacency,
                    current_iter=cur_iter,
                )
                if was_override:
                    out[vid] = exec_act.squeeze(0).detach().cpu().numpy()
                else:
                    out[vid] = action
            except Exception:
                out[vid] = action
        return out

    def _safe_tensor_to_numpy(self, tensor):
        """Safely convert tensor to numpy, handling CUDA tensors."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)

    def _safe_tensor_item(self, tensor):
        """Safely get scalar value from tensor, handling CUDA tensors."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().item()
        return float(tensor)
    def collect_experience(self, n_episodes: int):
        """Collect experience using current policy with proper buffering."""
        
        # Initialize buffers if not done
        if not hasattr(self, 'experience_buffer'):
            self._initialize_training_buffers()
        
        # Clear previous experience
        self._clear_buffers()
        
        total_reward = 0
        total_steps = 0
        total_completions = 0
        # P0.1: per-iter real collision counters
        iter_ai_collisions = 0
        iter_collision_breakdown = {}
        iter_mechanism_breakdown = {}

        # Use existing environment state or step once to get current observations
        if hasattr(self, '_initial_observations'):
            observations = self._initial_observations
            self._initial_observations = None  # Use it once
        else:
            observations, _, _, _ = self.env.step({})
        
        for episode in range(n_episodes):
            # Reset environment at the start of each episode (except first which uses _initial_observations)
            if episode > 0 or observations is None or not observations:
                observations = self.env.reset()

            episode_reward = 0
            episode_length = 0
            episode_experiences = []
            episode_completions = 0

            for step in range(self.args.max_steps):
                episode_length = step
                total_steps += 1

                if not observations:
                    # If somehow observations are empty, reset and try again
                    observations = self.env.reset()
                    continue

                # Get actions from DMPO policy
                actions, values, log_probs = self._get_dmpo_actions(observations)

                # DMPO-Safe: safety shield can override actions before execution
                if self._safe_enabled and self.safety_shield is not None:
                    try:
                        actions = self._apply_safety_shield(actions, observations)
                    except Exception as _e:
                        # Shield failure should never crash training — log and continue
                        pass

                # KAPPA-ADV: snapshot the ACTION-TIME κ-graph (the adjacency currently
                # held by the env was built from the state these actions were computed
                # on — under DMPO_GRAPH_FRESH it is exactly A(s_t)). Rows are ordered
                # by sorted vehicle id, the same ordering the env's graph builder uses.
                if self._kappa_adv_on:
                    _adj_pre = self.env.get_adjacency_matrix()
                    _adj_pre = _adj_pre.copy() if _adj_pre.size else None
                    _adj_ids = sorted(observations.keys())
                else:
                    _adj_pre, _adj_ids = None, None

                # Environment step
                new_observations, rewards, dones, info = self.env.step(actions)

                # Track trip completions
                completed = info.get('completed_vehicles', set())
                episode_completions += len(completed)

                # Store experience in episode buffer
                # KL-ANOMALY FIX (gated DMPO_STORE_RAW_ACTION): store the UNCLIPPED
                # sampled action so the PPO update recomputes logp on the SAME action
                # the stored logp describes (the env still received the clipped one).
                _acts_processed = self._process_actions(actions)
                if os.environ.get('DMPO_STORE_RAW_ACTION', '0') == '1':
                    _raw = getattr(self, '_raw_action_store', {})
                    for _aid in list(_acts_processed.keys()):
                        if _aid in _raw:
                            _acts_processed[_aid] = _raw[_aid]
                experience = {
                    'obs': self._process_observations(observations),
                    'actions': _acts_processed,
                    'rewards': rewards,
                    'values': values,
                    'log_probs': log_probs,
                    'next_obs': self._process_observations(new_observations) if new_observations else None,
                    'dones': dones,
                    'adjacency': self.env.get_adjacency_matrix(),
                    'info': info,
                    # Constrained-RL: per-step cost (0 or 1) per vehicle, from env info.
                    # Empty when not in constrained mode (downstream code skips cost path).
                    'costs': info.get('ai_costs_this_step', {}) if self._constrained_rl else {},
                    # KAPPA-ADV: action-time graph snapshot (None unless gated on)
                    'adj_pre': _adj_pre,
                    'adj_ids': _adj_ids,
                    # MATCHED MODEL LOSS: per-vehicle ordered neighbour vehicle-ids
                    # (tuple[str|None] len N) carried INSIDE the raw pre-step obs dict
                    # (processor key 'nb_ids'), so they are index-locked with 'obs'
                    # by construction. None unless gated on.
                    'slot_ids': ({vid: o.get('nb_ids') for vid, o in observations.items()
                                  if isinstance(o, dict)}
                                 if getattr(self, '_matched_on', False) else None),
                }

                episode_experiences.append(experience)

                # Update metrics
                step_reward = sum(rewards.values()) if rewards else 0
                episode_reward += step_reward
                observations = new_observations

                # P0.1: per-step real collision count + breakdown rollup
                iter_ai_collisions += int(info.get('ai_collision_new_pairs_this_step', 0))
                for rec in info.get('ai_new_collision_pair_records', []):
                    col_is_ai = rec.get('col_is_ai', False)
                    vic_is_ai = rec.get('vic_is_ai', False)
                    if col_is_ai and vic_is_ai:
                        pk = 'AI-AI'
                    elif col_is_ai != vic_is_ai:
                        pk = 'AI-human(AI_is_collider)' if col_is_ai else 'AI-human(AI_is_victim)'
                    else:
                        pk = 'human-human'
                    iter_collision_breakdown[pk] = iter_collision_breakdown.get(pk, 0) + 1
                    lane = rec.get('lane', '') or ''
                    mech = 'cross_traffic_junction' if lane.startswith(':') else 'rear_end_same_lane'
                    iter_mechanism_breakdown[mech] = iter_mechanism_breakdown.get(mech, 0) + 1

                # Check if episode should end — GLOBAL done ONLY.
                # TRAINING-FORENSIC BLOCKER (2026-07-25): this was
                # `any(dones.values())`, but the env sets a PER-VEHICLE
                # dones[vid]=True for every REMOVED vehicle (collision-remove
                # env:905 — live under the pinned DMPO_COLLISION_REMOVE_AI=1 —
                # plus stuck-timeout env:973 and clear-stopped env:1001). With
                # `any()`, the FIRST AI collision therefore ended the entire
                # multi-agent episode: exactly what the env's own
                # _compute_dones docstring forbids ("a per-vehicle done would
                # terminate the entire episode on every completion"). Genuine
                # global termination (max_steps, empty network, gridlock,
                # overload) marks EVERY vehicle done, so `all()` is the correct
                # predicate; per-vehicle flags still reach GAE via the stored
                # experience, unchanged.
                if bool(dones) and all(dones.values()):
                    break

            # Process episode for training
            self._process_episode_for_training(episode_experiences, episode_length)
            total_reward += episode_reward
            total_completions += episode_completions

        avg_reward = total_reward / max(1, n_episodes)
        avg_length = total_steps / max(1, n_episodes)
        avg_completions = total_completions / max(1, n_episodes)
        
        return {
            'total_steps': total_steps,
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'avg_completions': avg_completions,
            'buffer_size': len(self.experience_buffer['observations']),
            # P0.1: real collision metric for this iter (replaces the stale
            # safety_monitor.collision_count that always reported 0).
            'iter_ai_collisions': iter_ai_collisions,
            'iter_collision_breakdown': dict(iter_collision_breakdown),
            'iter_mechanism_breakdown': dict(iter_mechanism_breakdown),
        }
    
    def _clear_buffers(self):
        """Clear experience buffers for fresh collection."""
        for key in self.experience_buffer:
            self.experience_buffer[key] = []
        for key in self.model_experience:
            self.model_experience[key] = []
        for key in self.comm_buffer:
            self.comm_buffer[key] = []
        # Clear training stats to prevent unbounded growth
        if hasattr(self, 'training_stats'):
            for key in self.training_stats:
                self.training_stats[key] = []
    
    def _process_observations(self, observations):
        """Process observations for neural network training."""
        if not observations:
            return {}
        
        processed = {}
        for agent_id, obs in observations.items():
            if isinstance(obs, dict):
                # Flatten dict observation for neural networks
                processed[agent_id] = self._flatten_observation(obs)
            else:
                processed[agent_id] = obs
        return processed
    
    def _process_actions(self, actions):
        """Process actions for neural network training.

        Must match the network's action_dim:
          2-dim (Path D accountable): longitudinal + lateral
          3-dim (default):            + lane_intention
          4-dim throttle_brake:       (throttle, brake, lateral, lane_intention)
          4-dim yield_intent:         (longitudinal, lateral, lane_intention, yield_intent)
        """
        if not actions:
            return {}

        # Determine action_dim from config
        action_dim = self.config.agent_args.action_space.shape[0] if hasattr(self, 'config') else 3
        variant = os.environ.get('DMPO_ACTION_VARIANT', 'default')

        processed = {}
        for agent_id, action in actions.items():
            if isinstance(action, dict):
                if action_dim == 4 and variant == 'throttle_brake':
                    action_vec = np.concatenate([
                        action.get('throttle', [0.0]),
                        action.get('brake', [0.0]),
                        action.get('lateral', [0.0]),
                        action.get('lane_intention', [0.0]),
                    ])
                elif action_dim == 4 and variant == 'yield_intent':
                    action_vec = np.concatenate([
                        action.get('longitudinal', [0.0]),
                        action.get('lateral', [0.0]),
                        action.get('lane_intention', [0.0]),
                        action.get('yield_intent', [0.0]),
                    ])
                elif action_dim == 2:
                    action_vec = np.concatenate([
                        action.get('longitudinal', [0.0]),
                        action.get('lateral', [0.0])
                    ])
                else:
                    action_vec = np.concatenate([
                        action.get('longitudinal', [0.0]),
                        action.get('lateral', [0.0]),
                        action.get('lane_intention', [0.0])
                    ])
                processed[agent_id] = action_vec
            else:
                processed[agent_id] = action
        return processed
    
    def _process_episode_for_training(self, episode_experiences, episode_length):
        """Process episode experiences into training buffers.

        B1 REVERTED: Per-vehicle done=True at trip completion is REMOVED.
        It caused a value cliff that made the policy avoid completing trips.
        Instead, done=global_done only (all agents done at episode end).
        The mask infrastructure handles exited vehicles (mask=0, padded).

        COMPLETION BONUS: When a vehicle completes its trip, add a bonus
        to its last-step reward. This compensates for the value function
        overestimating future returns for about-to-exit agents. The bonus
        makes the advantage at exit ≈ neutral instead of negative.

        Bonus magnitude: estimated average remaining discounted return.
        With per-step reward ~0.7 and γ=0.99 over ~100 remaining steps:
        bonus ≈ 0.7 × (1 - 0.99^100) / (1 - 0.99) ≈ 45
        We use a conservative fixed value of 30.
        """
        # B1 REVERTED: no per-vehicle done patching.
        # Crossing bonus is handled directly in the reward function
        # (_compute_simple_reward gives +50 when vehicle exits intersection).

        for exp in episode_experiences:
            if exp['obs'] and exp['actions']:
                # Add to experience buffer
                self.experience_buffer['observations'].append(exp['obs'])
                self.experience_buffer['actions'].append(exp['actions'])
                self.experience_buffer['rewards'].append(exp['rewards'])
                self.experience_buffer['values'].append(exp['values'])
                self.experience_buffer['log_probs'].append(exp['log_probs'])
                self.experience_buffer['dones'].append(exp['dones'])
                # SLOT IDENTITY FIX: store per-timestep vehicle→slot mapping
                vehicle_slots = (exp.get('info') or {}).get('vehicle_slots', {})
                self.experience_buffer['vehicle_slots'].append(vehicle_slots)
                # Constrained-RL: store per-step costs alongside rewards
                self.experience_buffer['costs'].append(exp.get('costs', {}))
                # KAPPA-ADV: keep the graph series index-locked with observations
                self.experience_buffer['adj_pre'].append(exp.get('adj_pre'))
                self.experience_buffer['adj_ids'].append(exp.get('adj_ids'))
                # MATCHED MODEL LOSS: keep slot→id lists index-locked with observations
                self.experience_buffer['slot_ids'].append(exp.get('slot_ids'))

                # Add to model experience for environment model learning
                if exp['next_obs'] is not None:
                    self.model_experience['states'].append(exp['obs'])
                    self.model_experience['actions'].append(exp['actions'])
                    self.model_experience['next_states'].append(exp['next_obs'])
                    self.model_experience['rewards'].append(exp['rewards'])
                    self.model_experience['dones'].append(exp['dones'])

                # Add communication information
                if exp['adjacency'].size > 0:
                    self.comm_buffer['adjacency_matrices'].append(exp['adjacency'])
        
        # Update training statistics
        self.training_stats['episode_lengths'].append(episode_length)

    # NOTE: _compute_advantages_and_returns removed - DMPO handles GAE internally via _process_traj
    # The removed method was buggy (used current value as next_value instead of V(s_{t+1}))
    # DMPO's DPPOAgent._process_traj computes proper GAE with correct value bootstrapping

    def _get_dmpo_actions(self, observations):
        """Get actions from DMPO policy networks.

        DMPO expects input shape: [batch_size, n_agent, obs_dim]
        Returns distribution from agent.act(s), sample with .sample()

        Note: DMPO agent has fixed n_agent networks. We must match observation
        tensor size to agent.n_agent, padding with zeros or truncating as needed.

        NO FALLBACKS: If agent is None or action fails, we raise an error.
        This ensures training always uses the neural network.
        """
        if self.agent is None:
            raise RuntimeError("DMPO agent not initialized - cannot get actions without neural network!")

        actions = {}
        values = {}
        log_probs = {}

        try:
            obs_dim = self.agent.observation_dim

            if getattr(self.agent, 'dynamic', False):
                # ============================================================
                # DYNAMIC MODE: No slots, no padding. Build observation tensor
                # directly from the observations dict with exactly n_current
                # agents. The shared network handles any count.
                # ============================================================
                agent_ids = list(observations.keys())
                n_current = len(agent_ids)

                if n_current == 0:
                    return actions, values, log_probs

                obs_array = np.zeros((n_current, obs_dim), dtype=np.float32)
                for i, agent_id in enumerate(agent_ids):
                    obs_flat = self._flatten_observation_numpy(observations[agent_id])
                    # R1 (obs-v2): pad/truncate would amputate the conflict block or blind
                    # the policy — under DMPO_OBS_V2=1 any mismatch is a hard error.
                    if os.environ.get('DMPO_OBS_V2', '0') == '1' and len(obs_flat) != obs_dim:
                        raise RuntimeError(f'OBS-V2 dim mismatch at pad site: '
                                           f'{len(obs_flat)} vs obs_dim={obs_dim}')
                    if len(obs_flat) < obs_dim:
                        obs_flat = np.pad(obs_flat, (0, obs_dim - len(obs_flat)))
                    elif len(obs_flat) > obs_dim:
                        obs_flat = obs_flat[:obs_dim]
                    obs_array[i] = obs_flat

                # ZFilter: normalize (use first n_current rows)
                # Dynamic ZFilter handles variable sizes by normalizing per-feature
                obs_array = self._normalize_obs_dynamic(obs_array)

                obs_tensor = torch.as_tensor(
                    obs_array, dtype=torch.float32, device=self.device
                ).unsqueeze(0)  # [1, n_current, obs_dim]

                with torch.no_grad():
                    dist = self.agent.act(obs_tensor)
                    action_tensor = dist.sample()       # [1, n_current, action_dim]
                    log_prob_tensor = dist.log_prob(action_tensor)
                    if os.environ.get('DMPO_CENTRAL_CRITIC', '0') == '1':
                        from dmpo_autonomous_driving.utils import central_critic as _cc
                        _g = torch.as_tensor(_cc.global_feat(obs_array), device=self.device)
                        value_tensor = self.agent._evalV(torch.cat(
                            [obs_tensor, _g.expand(1, obs_tensor.shape[1], -1)], dim=-1))
                    else:
                        value_tensor = self.agent._evalV(obs_tensor)  # [1, n_current, 1]

                action_np = action_tensor.squeeze(0).cpu().numpy()
                log_prob_np = log_prob_tensor.squeeze(0).cpu().numpy()
                value_np = value_tensor.squeeze(0).cpu().numpy()

                for i, agent_id in enumerate(agent_ids):
                    actions[agent_id] = self._tensor_to_env_action_from_array(action_np[i])
                    values[agent_id] = float(value_np[i].mean())
                    log_probs[agent_id] = float(log_prob_np[i].sum())
                    # KL-ANOMALY FIX (found 2026-07-25, gated DMPO_STORE_RAW_ACTION):
                    # log_probs above is the density of the UNCLIPPED sample, but the
                    # buffer historically stored the ±1-CLIPPED env action — so the
                    # PPO update recomputed logp on a DIFFERENT action (measured
                    # |KL|=0.348 at epoch 0, ≈46× target; ratios corrupted for every
                    # clipped sample). Standard resolution: store the RAW sample so
                    # stored-logp and recomputed-logp refer to the same action; the
                    # env still receives the clipped action (part of the dynamics).
                    if os.environ.get('DMPO_STORE_RAW_ACTION', '0') == '1':
                        if not hasattr(self, '_raw_action_store'):
                            self._raw_action_store = {}
                        self._raw_action_store[agent_id] = action_np[i].copy()

            else:
                # ============================================================
                # ORIGINAL MODE: Fixed n_agent with slot mapping + padding
                # ============================================================
                n_agent_fixed = self.agent.n_agent
                env_slot_map = getattr(self.env, '_vehicle_to_slot', None) or {}
                obs_array = np.zeros((n_agent_fixed, obs_dim), dtype=np.float32)

                agent_id_to_slot = {}
                slot_to_agent_id = {}

                for agent_id, obs in observations.items():
                    if agent_id in env_slot_map:
                        slot = env_slot_map[agent_id]
                        if slot >= n_agent_fixed:
                            continue
                    else:
                        continue

                    obs_flat = self._flatten_observation_numpy(obs)
                    # R1 (obs-v2): pad/truncate would amputate the conflict block or blind
                    # the policy — under DMPO_OBS_V2=1 any mismatch is a hard error.
                    if os.environ.get('DMPO_OBS_V2', '0') == '1' and len(obs_flat) != obs_dim:
                        raise RuntimeError(f'OBS-V2 dim mismatch at pad site: '
                                           f'{len(obs_flat)} vs obs_dim={obs_dim}')
                    if len(obs_flat) < obs_dim:
                        obs_flat = np.pad(obs_flat, (0, obs_dim - len(obs_flat)))
                    elif len(obs_flat) > obs_dim:
                        obs_flat = obs_flat[:obs_dim]
                    obs_array[slot] = obs_flat
                    agent_id_to_slot[agent_id] = slot
                    slot_to_agent_id[slot] = agent_id

                agent_ids = list(observations.keys())
                if self.running_state is not None:       # None under OBS-V2 no-norm
                    obs_array = self.running_state(obs_array, update=True)

                obs_tensor = torch.as_tensor(
                    obs_array, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                with torch.no_grad():
                    dist = self.agent.act(obs_tensor)
                    action_tensor = dist.sample()
                    log_prob_tensor = dist.log_prob(action_tensor)
                    value_tensor = self.agent._evalV(obs_tensor)

                action_np = action_tensor.squeeze(0).cpu().numpy()
                log_prob_np = log_prob_tensor.squeeze(0).cpu().numpy()
                value_np = value_tensor.squeeze(0).cpu().numpy()

                self._pad_agent_steps = getattr(self, '_pad_agent_steps', 0) + len(agent_ids)
                for agent_id in agent_ids:
                    if agent_id in agent_id_to_slot:
                        slot = agent_id_to_slot[agent_id]
                        actions[agent_id] = self._tensor_to_env_action_from_array(action_np[slot])
                        values[agent_id] = float(value_np[slot].mean())
                        log_probs[agent_id] = float(log_prob_np[slot].sum())
                        # Padded-mode mirror of the dynamic-branch raw-action store
                        # (KL/ratio fix): without this, DMPO_STORE_RAW_ACTION was a
                        # silent no-op in padded mode (audit 2026-08-08, finding 2).
                        if os.environ.get('DMPO_STORE_RAW_ACTION', '0') == '1':
                            if not hasattr(self, '_raw_action_store'):
                                self._raw_action_store = {}
                            self._raw_action_store[agent_id] = action_np[slot].copy()
                    else:
                        # Agent observed but holding no env slot → no policy action this step.
                        # Cumulative audit (matched padded arm), printed at 1,10,100,...
                        self._pad_unslotted = getattr(self, '_pad_unslotted', 0) + 1
                        _u = self._pad_unslotted
                        if _u in (1, 10, 100, 1000, 10000, 100000, 1000000):
                            print(f"⚠️ padded: {_u} agent-steps without a slot (no policy action) "
                                  f"of {self._pad_agent_steps} agent-steps so far "
                                  f"(capacity {n_agent_fixed}, {len(agent_ids)} observed now)")

        except Exception as e:
            print(f"❌ Neural network action FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"DMPO action generation failed: {e}") from e

        return actions, values, log_probs

    def _get_heuristic_single_action(self, obs):
        """Get heuristic action for a single agent.

        DEPRECATED: This function is no longer used in the main training path.
        DMPO actions are now always generated by the neural network (no fallbacks).
        Kept for backwards compatibility with validation tests.
        """
        if os.environ.get('DMPO_OBS_V2', '0') == '1':
            raise NotImplementedError(
                'heuristic actions index the LEGACY obs layout (neighbors[:,0], '
                'infrastructure[0]) whose semantics changed under DMPO_OBS_V2 (Step 14)')
        ego = obs.get('ego', np.zeros(10))
        neighbors = obs.get('neighbors', np.zeros((10, 10)))
        infrastructure = obs.get('infrastructure', np.zeros(5))

        # Longitudinal control
        if len(neighbors) > 0 and np.any(neighbors[:, 0] > 0):
            min_distance = np.min(neighbors[neighbors[:, 0] > 0, 0])
            if min_distance < 0.1:
                longitudinal = -0.5
            elif min_distance < 0.3:
                longitudinal = -0.2
            else:
                longitudinal = 0.1
        else:
            longitudinal = np.clip(np.random.normal(0.2, 0.1), -1, 1)

        # Traffic light compliance
        traffic_phase = infrastructure[0] if len(infrastructure) > 0 else 0.5
        if traffic_phase < 0.3:
            longitudinal = min(longitudinal, -0.3)

        lateral = np.clip(np.random.normal(0.0, 0.03), -0.2, 0.2)
        lane_intention = np.clip(np.random.normal(0.0, 0.05), -0.3, 0.3)

        action = {
            'longitudinal': np.array([longitudinal]),
            'lateral': np.array([lateral]),
            'lane_intention': np.array([lane_intention])
        }
        value = np.random.normal(0, 1)
        log_prob = np.random.normal(0, 0.1)

        return action, value, log_prob

    def _normalize_obs_dynamic(self, obs_array, update=True):
        """Normalize observations for dynamic mode (variable agent count).

        Uses running mean/std statistics (updated online) instead of ZFilter
        which requires fixed shape. ObservationProcessor already normalizes
        most values, so this is a lightweight pass.

        When called from act-time (collect_experience), pass update=True so
        running stats track the obs distribution. When called from train-time
        (_prepare_dmpo_batch_dynamic) on the SAME buffered raw obs, pass
        update=False — otherwise stats get updated twice with the same data
        and act-time vs train-time normalizations diverge, breaking PPO's
        importance-ratio assumption that obs(act)==obs(train).
        """
        # OBS-V2 (C3 fix): NO z-filter, NO running stats, NO ±5 clip. Every v2 slot is
        # analytically bounded at construction (widest ±2), and the measured damage of
        # the z-path was exactly the regime that decides collisions (running std 0.42
        # on ego-accel → every braking beyond ≈5.5 m/s² clipped indistinguishable).
        # DMPO_OBS_ZFILTER=1 is the gated ablation escape hatch (keeps this z-path).
        if getattr(self, '_obs_v2_nonorm', False):
            return np.asarray(obs_array, dtype=np.float32)
        if not hasattr(self, '_obs_running_mean'):
            obs_dim = obs_array.shape[-1]
            self._obs_running_mean = np.zeros(obs_dim, dtype=np.float64)
            self._obs_running_var = np.ones(obs_dim, dtype=np.float64)
            self._obs_count = 1e-4

        if update:
            # Update running stats with all rows
            batch_mean = obs_array.mean(axis=0)
            batch_var = obs_array.var(axis=0)
            batch_count = obs_array.shape[0]

            delta = batch_mean - self._obs_running_mean
            tot_count = self._obs_count + batch_count
            self._obs_running_mean += delta * batch_count / tot_count
            m_a = self._obs_running_var * self._obs_count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta**2 * self._obs_count * batch_count / tot_count
            self._obs_running_var = M2 / tot_count
            self._obs_count = tot_count

        # Normalize and clip (always, regardless of update flag)
        std = np.sqrt(self._obs_running_var + 1e-8)
        normalized = (obs_array - self._obs_running_mean.astype(np.float32)) / std.astype(np.float32)
        return np.clip(normalized, -5.0, 5.0).astype(np.float32)

    def _flatten_observation_numpy(self, obs):
        """Flatten observation dictionary to numpy array for DMPO input.

        NOTE: ObservationProcessor already normalizes all values:
        - Positions: divided by max_distance (200m)
        - Velocities: divided by max_speed (18 m/s)
        - Infrastructure: already in [0,1]
        - Neighbors: already normalized

        DO NOT normalize again here - that was causing double normalization
        which made values extremely small (e.g., velocity / 18 / 20 = velocity / 360).
        """
        # THE canonical concat (Step 9 / R4): ObservationProcessor.flatten() is the ONE
        # flat-order authority for train + every eval script. Its v1 branch reproduces
        # the old inline concat here byte-for-byte (same .get defaults, same order,
        # conflict_gap appended iff present); its v2 branch RAISES on a missing block.
        obs_flat = ObservationProcessor.flatten(obs)

        # Final clip kept as a tripwire only (values are already in range by construction)
        obs_flat = np.clip(obs_flat, -10.0, 10.0)

        return obs_flat

    def _tensor_to_env_action_from_array(self, action_array):
        """Convert action array to environment action dict.

        Path D (DMPO_REWARD_ACCOUNTABLE): 2-dim actions — longitudinal + lateral.
        lane_intention is always 0 (redundant with continuous sublane).
        Default: 3-dim (longitudinal, lateral, lane_intention).
        DMPO_ACTION_VARIANT=throttle_brake: 4-dim (throttle, brake, lateral, lane_intention).
        DMPO_ACTION_VARIANT=yield_intent:   4-dim (longitudinal, lateral, lane_intention, yield_intent).
        """
        action_array = np.clip(action_array, -1.0, 1.0)

        variant = os.environ.get('DMPO_ACTION_VARIANT', 'default')
        if variant == 'throttle_brake':
            return {
                'throttle': np.array([action_array[0]]),
                'brake': np.array([action_array[1]]),
                'lateral': np.array([action_array[2]]) if len(action_array) > 2 else np.array([0.0]),
                'lane_intention': np.array([action_array[3]]) if len(action_array) > 3 else np.array([0.0]),
            }
        if variant == 'yield_intent':
            return {
                'longitudinal': np.array([action_array[0]]),
                'lateral': np.array([action_array[1]]) if len(action_array) > 1 else np.array([0.0]),
                'lane_intention': np.array([action_array[2]]) if len(action_array) > 2 else np.array([0.0]),
                'yield_intent': np.array([action_array[3]]) if len(action_array) > 3 else np.array([0.0]),
            }

        return {
            'longitudinal': np.array([action_array[0]]),
            'lateral': np.array([action_array[1]]) if len(action_array) > 1 else np.array([0.0]),
            'lane_intention': np.array([action_array[2]]) if len(action_array) > 2 else np.array([0.0])
        }
    
    def _get_heuristic_actions(self, observations):
        """Enhanced heuristic actions for fallback.

        DEPRECATED: This function is no longer used in the main training path.
        DMPO actions are now always generated by the neural network (no fallbacks).
        Kept for backwards compatibility with validation tests that monitor fallback counts.
        """
        if os.environ.get('DMPO_OBS_V2', '0') == '1':
            raise NotImplementedError(
                'heuristic actions index the LEGACY obs layout whose semantics '
                'changed under DMPO_OBS_V2 (Step 14)')
        actions = {}
        values = {}
        log_probs = {}
        
        for agent_id, obs in observations.items():
            ego_state = obs.get('ego', np.zeros(10))
            neighbors = obs.get('neighbors', np.zeros((10, 10)))
            infrastructure = obs.get('infrastructure', np.zeros(5))
            
            # Smart action selection based on intersection context
            
            # Longitudinal control
            if len(neighbors) > 0 and np.any(neighbors[:, 0] > 0):
                # Vehicle ahead - adjust based on distance
                min_distance = np.min(neighbors[neighbors[:, 0] > 0, 0])
                if min_distance < 0.1:  # Very close
                    longitudinal = -0.5  # Strong deceleration
                elif min_distance < 0.3:  # Close
                    longitudinal = -0.2  # Mild deceleration
                else:
                    longitudinal = 0.1   # Slight acceleration
            else:
                longitudinal = np.clip(np.random.normal(0.2, 0.1), -1, 1)
            
            # Traffic light compliance
            traffic_phase = infrastructure[0] if len(infrastructure) > 0 else 0.5
            if traffic_phase < 0.3:  # Red light
                longitudinal = min(longitudinal, -0.3)
            
            # Lateral control - avoid conflicts
            lateral = np.clip(np.random.normal(0.0, 0.03), -0.2, 0.2)
            
            # Lane intention - conservative
            lane_intention = np.clip(np.random.normal(0.0, 0.05), -0.3, 0.3)
            
            actions[agent_id] = {
                'longitudinal': np.array([longitudinal]),
                'lateral': np.array([lateral]),
                'lane_intention': np.array([lane_intention])
            }
            
            # Mock values and log_probs for consistency
            values[agent_id] = np.random.normal(0, 1)
            log_probs[agent_id] = np.random.normal(0, 0.1)
        
        return actions, values, log_probs
    
    def _flatten_observation(self, obs):
        """Flatten observation dictionary for neural network input.

        NOTE: Uses _flatten_observation_numpy to ensure consistent normalization
        with the training path. Previously this method was unnormalized causing
        train/inference mismatch.
        """
        # Use the same normalization as _flatten_observation_numpy
        obs_flat = self._flatten_observation_numpy(obs)
        return torch.FloatTensor(obs_flat).unsqueeze(0).to(self.device)
    
    def _tensor_to_env_action(self, action_tensor):
        """Convert neural network action tensor to environment action."""
        action_np = action_tensor.cpu().numpy().flatten()
        
        # Assume 3D continuous action space
        return {
            'longitudinal': np.array([action_np[0]]),
            'lateral': np.array([action_np[1]]),  
            'lane_intention': np.array([action_np[2]])
        }
    
    def _update_entropy_coefficient(self, new_coeff: float):
        """Update the entropy coefficient in the DMPO agent.

        This dynamically adjusts exploration vs exploitation during training.
        High entropy = more exploration (early training)
        Low entropy = more exploitation (late training)

        Args:
            new_coeff: New entropy coefficient value
        """
        if self.agent is not None and hasattr(self.agent, 'entropy_coeff'):
            self.agent.entropy_coeff = new_coeff

        # Also update in config for consistency
        if hasattr(self.config, 'agent_args'):
            self.config.agent_args.entropy_coeff = new_coeff

    def _log_training_dynamics(self, iteration, value_loss, avg_reward, model_loss, entropy_coeff):
        """Log training dynamics: compact summary + detailed per-vehicle trace at key iterations."""
        vl = self.training_metrics['value_losses']
        rl = self.training_metrics['iteration_rewards']
        n_agents = self.env.n_agents
        max_agents = self.env.max_ai_vehicles or n_agents

        # Trend arrows
        v_trend = ""
        if len(vl) >= 2:
            v_trend = "^" if vl[-1] > vl[-2] * 1.1 else "v" if vl[-1] < vl[-2] * 0.9 else "="
        r_trend = ""
        if len(rl) >= 2:
            r_trend = "^" if rl[-1] > rl[-2] else "v" if rl[-1] < rl[-2] else "="

        # Completions
        completions = self.training_metrics.get('completions', [])
        comp_str = f" comp={completions[-1]:.1f}" if completions else ""

        # F2 (live-flow audit, HIGH): per-iteration collision count in the >> series.
        # It was printed only every 10th iteration, so the paper's collision learning
        # curve was unrecoverable from run artifacts (ai_collisions IS already stored
        # per-iteration in training_metrics — this just surfaces it per line).
        _cols = self.training_metrics.get('ai_collisions', [])
        col_str = f" col={_cols[-1]}" if _cols else ""

        # Scaffold kill-gate metric (audit: _gap_shield_fired was written, never
        # read): per-iteration shield interventions — must decay to ~0 under the
        # anneal, else the arm is a crutch and gets dropped.
        shd_str = ""
        if os.environ.get('DMPO_GAP_SHIELD', '0') == '1':
            _tot = int(getattr(self.env, '_gap_shield_fired', 0) or 0)
            _prev = int(getattr(self, '_shield_fired_prev', 0))
            self._shield_fired_prev = _tot
            self.training_metrics.setdefault('shield_fired', []).append(_tot - _prev)
            shd_str = f" shd={_tot - _prev}"

        # Audit F-hygiene: a no-trajectory iteration must not masquerade as a real
        # update with v=0.00 (it polluted trend miners as a fake loss minimum).
        if getattr(self, '_last_iter_skipped', False):
            self._last_iter_skipped = False
            print(f"  >> SKIPPED (no trajectory data — losses not updated)"
                  f"{comp_str}{col_str}")
            return

        # Policy std
        std_str = ""
        if hasattr(self.agent, 'actors') and len(self.agent.actors) > 0:
            avg_std = sum(torch.exp(torch.clamp(a.log_std, a.LOG_STD_MIN, a.LOG_STD_MAX)).mean().item() for a in self.agent.actors) / len(self.agent.actors)
            std_str = f" std={avg_std:.3f}"

        # Action head bias (the drift indicator)
        bias_str = ""
        if hasattr(self.agent, 'actors') and len(self.agent.actors) > 0:
            bias_val = self.agent.actors[0].action_head.bias.data[0].item()
            bias_str = f" bias={bias_val:+.3f}"

        # Count stopped vehicles
        stopped = 0
        try:
            cache = getattr(self.env, '_vehicle_cache', {})
            ai_vehs = list(self.env.ai_vehicle_ids)
            if ai_vehs and cache:
                speeds = [cache.get(v, {}).get('speed', 0) for v in ai_vehs if v in cache]
                stopped = sum(1 for s in speeds if s < 0.5)
        except Exception:
            pass

        # Enhanced summary line
        print(f"  >> v={value_loss:.2f}({v_trend}) r={avg_reward:.0f}({r_trend}) agents={n_agents}/{max_agents}{comp_str}{col_str}{shd_str}{std_str}{bias_str} stopped={stopped}")

        # =====================================================================
        # DETAILED DIAGNOSTIC at key iterations: 1,2,3,5,10,15,20,30,50,100
        # Traces every AI vehicle's state to identify exactly when/where
        # the policy diverges between CPU and GPU runs.
        # =====================================================================
        diagnostic_iters = {1, 2, 3, 5, 10, 15, 20, 30, 50, 100}
        if iteration in diagnostic_iters:
            try:
                cache = getattr(self.env, '_vehicle_cache', {})
                ai_vehs = sorted(self.env.ai_vehicle_ids)
                print(f"  📋 DIAGNOSTIC iter={iteration} ({len(ai_vehs)} vehicles):")
                # Get traci module (libsumo or traci)
                try:
                    import libsumo as _traci
                except ImportError:
                    import traci as _traci

                n_warnings = {0: 0, 0.1: 0, 0.4: 0, 0.7: 0, 1.0: 0}
                for v in ai_vehs[:15]:  # cap at 15 to avoid log explosion
                    state = cache.get(v, {})
                    speed = state.get('speed', -1)
                    lane = state.get('lane_id', '?')
                    edge = state.get('edge_id', '?')
                    lane_short = lane.split('_')[-1] if lane else '?'
                    direction = 'N' if 'north' in str(edge) else 'E' if 'east' in str(edge) else 'S' if 'south' in str(edge) else 'W' if 'west' in str(edge) else 'J' if ':' in str(lane) else '?'

                    # Get leader + collision warning
                    leader_str = ""
                    warn_val = 0.0
                    try:
                        leader = _traci.vehicle.getLeader(v)
                        if leader:
                            lid, ldist = leader
                            leader_str = f" ldr={lid[-8:]}@{ldist:.0f}m"
                            # Compute same warning as observation
                            if ldist < 0:
                                warn_val = 1.0
                            elif ldist < 2:
                                warn_val = 1.0
                            elif ldist < 5:
                                warn_val = 0.7
                            elif ldist < 10:
                                warn_val = 0.4
                            elif ldist < 20:
                                warn_val = 0.1
                    except Exception:
                        pass

                    # Track warning distribution
                    if warn_val >= 1.0:
                        n_warnings[1.0] += 1
                    elif warn_val >= 0.7:
                        n_warnings[0.7] += 1
                    elif warn_val >= 0.4:
                        n_warnings[0.4] += 1
                    elif warn_val >= 0.1:
                        n_warnings[0.1] += 1
                    else:
                        n_warnings[0] += 1

                    status = "🟢" if speed > 0.5 else "🔴"
                    warn_str = f" ⚠{warn_val:.1f}" if warn_val > 0 else ""
                    print(f"    {status} {v[-12:]}: spd={speed:.1f} dir={direction} {lane_short}{leader_str}{warn_str}")

                # Warning distribution summary
                print(f"    ⚠️ Warnings: safe={n_warnings[0]} caution={n_warnings[0.1]}+{n_warnings[0.4]} danger={n_warnings[0.7]} imminent={n_warnings[1.0]}")

                # Policy stats
                if hasattr(self.agent, 'actors') and len(self.agent.actors) > 0:
                    actor = self.agent.actors[0]
                    bias = actor.action_head.bias.data.cpu().tolist()
                    log_std = actor.log_std.data.cpu().tolist()
                    w_norm = actor.action_head.weight.data.norm().item()
                    print(f"    📊 Policy: bias={[f'{b:+.4f}' for b in bias]} log_std={[f'{s:.3f}' for s in log_std]} w_norm={w_norm:.4f}")
            except Exception as e:
                print(f"    📋 DIAGNOSTIC error: {e}")

    def update_dmpo(self, episodes_data=None):
        """Update DMPO neural networks with collected experience.

        Uses DMPO agent's updateAgent (policy+value) and updateModel (world model).
        NO fallbacks - DMPO agent must be properly initialized.

        KEY FIX: On first update, detect actual number of active agents and
        reinitialize DMPO if needed. This prevents zero-padding issues where
        DMPO was configured for 30 agents but only 16 are active.
        """
        # (removed per-iteration "Updating" print — info in >> summary line)

        if not hasattr(self, 'experience_buffer') or not self.experience_buffer['observations']:
            print("⚠️ No experience data available for training")
            return 0.0, 0.0, 0.0

        if self.agent is None:
            raise RuntimeError("DMPO agent not initialized - cannot update networks")

        # CRITICAL FIX: Detect and fix n_agent mismatch on first update
        # In dynamic mode: no reinitialization needed (shared networks handle any count)
        if not getattr(self.agent, 'dynamic', False):
            if not hasattr(self, '_dmpo_reinitialized') or not self._dmpo_reinitialized:
                if os.environ.get('DMPO_CURRICULUM', '0') == '1':
                    actual_n_agents = self.env.max_ai_vehicles
                else:
                    actual_n_agents = self._detect_actual_n_agents()
                self._reinitialize_dmpo_for_n_agents(actual_n_agents)

                # NOW load checkpoint after agent reinitialization (architecture matches)
                if hasattr(self, '_pending_resume') and self._pending_resume is not None:
                    print(f"📂 Loading checkpoint now that agent count is correct ({actual_n_agents})...")
                    self._load_checkpoint(self._pending_resume)
                    self._pending_resume = None  # Clear pending resume
        else:
            # Dynamic mode: shared networks are already sized correctly. Still
            # need to load resume checkpoint here (the if-branch above is
            # bypassed entirely in dynamic mode). BUG: prior code never loaded
            # weights on resume in dynamic mode, silently training from scratch.
            if hasattr(self, '_pending_resume') and self._pending_resume is not None:
                print(f"📂 Loading checkpoint (dynamic mode) from iter {self._pending_resume}...")
                self._load_checkpoint(self._pending_resume)
                self._pending_resume = None

        # Use the real DMPO agent update
        return self._update_dmpo_agent()
    
    def _rollout_model(self, real_trajs, length=0):
        """Generate synthetic trajectories using learned world model (MBPO-style).

        Uses BREADTH over DEPTH: sample many initial states from real data,
        predict only 1 step ahead from each. This avoids error compounding
        while generating useful synthetic data to augment real experience.

        With depth=15 (old approach), model errors compound over 15 steps,
        producing out-of-distribution states that destroy the value network.
        With depth=1, each prediction starts from a REAL state, so the
        residual connection (state_pred += s) keeps predictions close to reality.

        Args:
            real_trajs: Real trajectories to sample initial states from
            length: Rollout horizon (default: 1 for MBPO-style short rollouts)

        Returns:
            List of synthetic Trajectory objects from model rollouts
        """
        # MBPO-style: short rollouts to prevent error compounding.
        # The benefit comes from BREADTH (many starting states), not DEPTH.
        # Controlled via --model-rollout-length CLI arg (default: 1).
        if length <= 0:
            length = getattr(self.config, 'model_traj_length', 1)

        # Sample MANY initial states for breadth (MBPO approach)
        n_traj = getattr(self.config, 'model_batch_size', 256)

        # Extract states from real trajectories.
        # Dynamic mode: trajectories have variable T, so torch.stack fails.
        # We concatenate along the time axis instead — we only need a pool of
        # (timestep, n_agents, obs_dim) slices to sample n_traj starting points
        # from, and the trajectory boundaries don't matter for single-step
        # rollouts (length=1 MBPO).
        try:
            s_list = []
            mask_list = []
            for traj in real_trajs:
                if isinstance(traj, dict):
                    s_list.append(traj['s'])
                    if 'mask' in traj:
                        mask_list.append(traj['mask'])
                elif hasattr(traj, 'dict'):
                    s_list.append(traj.dict['s'])
                    if 'mask' in traj.dict:
                        mask_list.append(traj.dict['mask'])
                elif hasattr(traj, '__getitem__'):
                    s_list.append(traj['s'])
                    try:
                        mask_list.append(traj['mask'])
                    except (KeyError, TypeError):
                        pass
                else:
                    print("⚠️ Cannot extract states from trajectories for model rollout (no valid access pattern)")
                    return real_trajs

            if not s_list:
                return real_trajs

            # Ensure tensors (lists from buffer may be numpy arrays)
            s_list = [torch.as_tensor(x, device=self.device) for x in s_list]
            s_flat = torch.cat(s_list, dim=0)  # [sum_T, n_agent, dim]
            total_steps = s_flat.shape[0]
            if total_steps == 0:
                return real_trajs

            # Sample n_traj random initial states from the pool (with replacement) —
            # matches original framework's sampling; duplicates are fine because
            # each gets a fresh policy action with its own noise.
            idxs = torch.randint(low=0, high=total_steps, size=(n_traj,), device=self.device)
            s = s_flat.index_select(dim=0, index=idxs)  # [n_traj, n_agent, dim]

            # Agent masks along the same flat axis, if present
            rollout_agent_mask = None
            if mask_list:
                try:
                    mask_list = [torch.as_tensor(m, device=self.device) for m in mask_list]
                    m_flat = torch.cat(mask_list, dim=0)  # [sum_T, n_agent]
                    if m_flat.shape[0] == total_steps:
                        rollout_agent_mask = m_flat.index_select(dim=0, index=idxs)
                except Exception:
                    pass

        except Exception as e:
            print(f"⚠️ Cannot extract states from trajectories for model rollout: {e}")
            return real_trajs

        # Use TrajectoryBuffer like the original implementation
        if 'TrajectoryBuffer' in self.modules:
            traj_buffer = self.modules['TrajectoryBuffer'](device=self.device)

            for _ in range(length):
                # Get action from policy
                dist = self.agent.act(s)
                a = dist.sample()
                logp = dist.log_prob(a)

                # Use world model to predict next state and reward
                # Pass agent mask to prevent padded-agent corruption (Proposal 1+2)
                r, s1, d, _ = self.agent.model_step(s, a, agent_mask=rollout_agent_mask)

                # MATCHED MODEL LOSS (gated): canonicalise the predicted neighbour
                # block to nearest-first (empties snapped to the exact neutral row)
                # BEFORE it is stored (→ _evalV(s1)) and fed back as the next s
                # (→ agent.act / get_logp / _evalV). Real states are already canonical.
                if getattr(self, '_matched_on', False):
                    _g = self._nb_geom
                    s1 = _mml.resort_nb_slots(s1, off=_g['off'], N=_g['N'], F=_g['F'],
                                              neutral=_g['neutral'],
                                              front_lam=_g['front_lam'])

                # Protect against NaN values from model predictions
                if not (torch.isfinite(s1).all() and torch.isfinite(r).all()):
                    # NaN OR ±inf: torch.isnan(inf) is False, and an infinite reward
                    # sailed through the old check into advantage normalization
                    # (inf - inf = NaN) → the C6.8 mask assert. Root cause of both
                    # 2026-08-14/15 deaths; the batch-level finite gate is the
                    # second layer.
                    print("⚠️ non-finite value in model rollout step, truncating")
                    break

                # Store transition in buffer
                traj_buffer.store(s, a, r, s1, d, logp)

                # Move to next state
                s = s1

            # Retrieve returns list of Trajectory objects with [T, n_agent, dim] shape
            synthetic_trajs = traj_buffer.retrieve()

            if not synthetic_trajs or len(synthetic_trajs) == 0:
                print("⚠️ No valid model rollouts generated, using real trajectories")
                return real_trajs

            # FIX: Inject agent masks from real trajectories into synthetic ones.
            # Without masks, padded agents (zero obs) produce garbage model predictions
            # that corrupt the value network via unmasked MSE loss.
            if real_trajs and hasattr(real_trajs[0], 'has_mask') and real_trajs[0].has_mask():
                real_mask = real_trajs[0]['mask']  # [T_real, n_agent]
                for syn_traj in synthetic_trajs:
                    syn_T = syn_traj.length
                    # Agent layout is fixed across timesteps — use first timestep's mask
                    mask = real_mask[0:1].expand(syn_T, -1).clone()
                    syn_traj.dict['mask'] = mask
                    syn_traj.names = ["s", "a", "r", "s1", "d", "logp", "mask"]

            return synthetic_trajs

        else:
            print("⚠️ TrajectoryBuffer not available, using real trajectories")
            return real_trajs

    def _truncate_trajs_for_model(self, trajs, model_update_length):
        """Truncate trajectories to model_update_length for model training.

        Always returns plain dicts (not Trajectory objects) so that the mask key
        is preserved. Trajectory.getFraction() drops keys not in self.names.
        """
        truncated = []
        for traj in trajs:
            traj_dict = traj.dict if hasattr(traj, 'dict') else traj
            t_dict = {}
            # 'slot_ids'/'slot_ids_next' exist only under DMPO_MODEL_MATCHED_LOSS=1
            # (prefix truncation keeps ref = sid[0] == the trajectory start).
            for key in ['s', 'a', 'r', 's1', 'd', 'mask', 'slot_ids', 'slot_ids_next']:
                if key in traj_dict:
                    t = traj_dict[key]
                    t_dict[key] = t[:model_update_length] if t.shape[0] > model_update_length else t
            if t_dict:
                truncated.append(t_dict)
        return truncated

    def _update_dmpo_agent(self):
        """Update using true DMPO agent from Networked-MB-MARL.

        Matches Algorithm 2 from the paper (Nature Machine Intelligence, 2024):
          Line 5-6: Collect real trajectories → store in environment buffer D_E
          Line 7:   Train model p^ψ on real data from D_E (BEFORE inner loop)
          Line 8-17: Inner loop: model rollouts → policy/value updates

        Key differences from previous implementation:
          - Model ALWAYS trains on real data from ModelBuffer (not leaked `trajs`)
          - ModelBuffer accumulates real trajectories across iterations
          - Multiple model gradient steps per update (n_model_update from config)
          - Model update happens BEFORE policy inner loop
          - Model validation with early stopping
          - Dedicated warmup phase with model pre-training
        """
        # Prepare data for DMPO agent in Trajectory format
        real_trajs = self._prepare_dmpo_batch()

        if not real_trajs:
            print("⚠️ No trajectory data available for DMPO update")
            # Audit F-hygiene: mark the iteration so the >> series prints SKIPPED
            # instead of a fake v=0.00 datapoint (miners must also skip the 0.0
            # placeholders appended to the loss metrics for such iterations).
            self._last_iter_skipped = True
            # SKIP-LOOP RECOVERY (Stage-4 overnight finding): K consecutive
            # skipped updates means the env is stuck in a self-perpetuating jam
            # (soft resets inherit the jammed net; scaffsel_s7 burned 5000 such
            # iterations). Force ONE hard SUMO restart to clear it.
            _k = int(os.environ.get('DMPO_SKIP_HARD_RESET', '8') or 0)
            self._consec_skips = getattr(self, '_consec_skips', 0) + 1
            if _k > 0 and self._consec_skips >= _k:
                print(f"🔁 {self._consec_skips} consecutive skipped updates — forcing "
                      f"a HARD env reset to clear the jammed network")
                self.env._force_hard_next = True
                self._consec_skips = 0
            return 0.0, 0.0, 0.0

        # DMPO-Safe: apply model-usage curriculum
        # If a schedule is defined, model_prob = piecewise(current_iteration)
        # If DMPO_USE_MODEL_FROM_ITER is set, model usage gated until that iter
        current_iter_for_curriculum = getattr(self, '_current_iteration', 0)
        model_use_from_iter = getattr(self.config, 'model_use_from_iter', 0)
        model_prob_schedule = getattr(self.config, 'model_prob_schedule', None)
        if model_prob_schedule is not None:
            model_prob = 0.0
            for threshold, prob in model_prob_schedule:
                if current_iter_for_curriculum >= threshold:
                    model_prob = prob
        else:
            model_prob = getattr(self.config, 'model_prob', 0.0)
        # Gate by use-from-iter
        if current_iter_for_curriculum < model_use_from_iter:
            model_prob = 0.0
        # BUFFER-FLOOR gate (2026-08-14): never mix rollouts from a near-empty
        # buffer. On resume the ModelBuffer restarts empty; rollouts drawn from
        # a thin buffer against a mature policy produced extreme rollout rewards
        # → NaN advantages → the C6.8 mask assert killed the run (resumed model
        # arm @3770). Standard MBPO hygiene; model TRAINING is unaffected.
        _mix_floor = int(os.environ.get('DMPO_MODEL_MIX_MIN_TRAJS', '16'))
        if model_prob > 0.0 and self.model_buffer is not None \
                and getattr(self.model_buffer, 'count', 0) < _mix_floor:
            if not getattr(self, '_mix_floor_logged', False):
                print(f"  ⏳ model-mix gated: buffer {getattr(self.model_buffer, 'count', 0)} "
                      f"< floor {_mix_floor} trajs (post-resume refill)")
                self._mix_floor_logged = True
            model_prob = 0.0
        # Gate by world-model CONVERGENCE (DMPO_MODEL_CONVERGE_THRESH > 0):
        # keep model-free (model_prob=0) until the world model is accurate enough
        # (rel-err < thresh), THEN switch to the configured model_prob — true DMPO,
        # but never feeding a garbage model into the policy. Latches once converged
        # so a later rel-err spike won't flip it back off. The model still TRAINS
        # while gated (n_model_update), so it can reach the threshold.
        if model_prob > 0.0:
            _conv_thresh = float(os.environ.get('DMPO_MODEL_CONVERGE_THRESH', '0.0'))
            if _conv_thresh > 0.0:
                if not getattr(self, '_model_converged', False):
                    _rel = getattr(self.agent, '_last_model_rel_error', float('inf'))
                    _stable_need = int(os.environ.get('DMPO_MODEL_STABLE_CHECKS', '1'))
                    # Count only FRESH rel-errs (a new model update happened) and require
                    # N CONSECUTIVE sub-threshold updates so a single lucky dip can't latch
                    # (converge AND stabilise). Any spike resets the streak.
                    if _rel != getattr(self, '_last_checked_rel_error', None):
                        self._last_checked_rel_error = _rel
                        if _rel < _conv_thresh:
                            self._model_stable_count = getattr(self, '_model_stable_count', 0) + 1
                        else:
                            self._model_stable_count = 0
                    if getattr(self, '_model_stable_count', 0) >= _stable_need:
                        self._model_converged = True
                        print(f"✅ World model CONVERGED+STABLE (rel-err {_rel:.4f} < {_conv_thresh} for "
                              f"{_stable_need} consecutive updates) — enabling model_prob={model_prob} at iter {current_iter_for_curriculum}")
                    else:
                        model_prob = 0.0   # still model-free until converged AND stable
        model_based = getattr(self.config, 'model_based', True)

        # H5 (learner-fixes spec): does the CONFIGURED INTENT ever use the world
        # model? Keyed off config intent, NOT the per-iteration effective
        # model_prob — the convergence gate above deliberately holds the
        # effective value at 0 while the model trains toward its threshold, and
        # gating on it would deadlock that feature. (model_use_from_iter only
        # delays usage of the same configured model_prob, so it is not part of
        # the intent predicate.)
        _model_ever_used = (
            float(getattr(self.config, 'model_prob', 0.0)) > 0.0
            or getattr(self.config, 'model_prob_schedule', None) is not None
        )

        n_inner_iter = getattr(self.config, 'n_inner_iter', 10)
        # H5c(i): with model_prob=0 the inner loop's only purpose (interleaving
        # model batches) is gone — it is a pure 10x repetition on the identical
        # batch (10 x n_update_pi = 40 epochs). DMPO_N_INNER_ITER (any explicit
        # value, 0 allowed) overrides; else DMPO_INNER_COLLAPSE=1 collapses to 1
        # on model-free runs. Both default to legacy behaviour.
        _inner_env = os.environ.get('DMPO_N_INNER_ITER', '').strip()
        if _inner_env != '':
            n_inner_iter = int(_inner_env)
        elif os.environ.get('DMPO_INNER_COLLAPSE', '0') == '1' and not _model_ever_used:
            n_inner_iter = 1
        if not getattr(self, '_logged_ppo_budget', False):
            print(f"  PPO budget: inner={n_inner_iter} x n_update_pi="
                  f"{getattr(self.config.agent_args, 'n_update_pi', '?')} epochs on the real batch "
                  f"(model_ever_used={_model_ever_used})", flush=True)
            self._logged_ppo_budget = True

        if not hasattr(self.agent, 'updateAgent'):
            raise RuntimeError("DMPO agent missing 'updateAgent' method")

        n_warmup = getattr(self.config, 'n_warmup', 50)
        current_iteration = getattr(self, '_current_iteration', 0)
        in_warmup = current_iteration < n_warmup

        # =============================================================================
        # STEP 1: Store real trajectories in ModelBuffer (Paper Alg 2, line 6)
        # =============================================================================
        if self.model_buffer is not None:
            if os.environ.get('DMPO_CENTRAL_CRITIC', '0') == '1':
                # MAPPO baseline: the world model sees o_i only, never [o_i | g_t].
                from algorithms.algo.buffer import Trajectory
                _D = int(self.config.agent_args.observation_dim)
                _mb = []
                for _tr in real_trajs:
                    _c = Trajectory(**{k: (v[..., :_D] if k in ('s', 's1') else v)
                                       for k, v in _tr.dict.items()})
                    for _k, _v in _tr.dict.items():
                        _c.dict.setdefault(_k, _v)
                    _mb.append(_c)
                self.model_buffer.storeTrajs(_mb)
            else:
                self.model_buffer.storeTrajs(real_trajs)

        # =============================================================================
        # STEP 2: Train world model on REAL data from buffer (Paper Alg 2, line 7)
        #
        # Matches original main.py warmup + training structure:
        #   - Warmup (iter 0 to n_warmup-1): ONLY collect data into buffer, NO model training
        #   - First post-warmup (iter == n_warmup): 1500 model gradient steps (pre-training)
        #   - After warmup (iter > n_warmup): every 10 iterations, 2000 model gradient steps
        # =============================================================================
        model_loss = 0.0
        self._last_did_model_update = False  # set True below only when a model update actually runs

        # H5a (learner-fixes spec): when the run will NEVER use the world model
        # (model_prob=0 and no schedule), every model gradient step is discarded
        # work (~94% of wall clock on past runs).
        # DMPO_MODEL_TRAIN_GATE=1 skips model training entirely on such runs;
        # default '0' = legacy (model trains regardless). Covers BOTH branches
        # below (main chain AND the model_buffer-is-None fallback — verifier fix).
        _skip_model_training = (
            os.environ.get('DMPO_MODEL_TRAIN_GATE', '0') == '1' and not _model_ever_used
        )
        if _skip_model_training and not getattr(self, '_logged_model_gate', False):
            print("🛑 Model-free run (model_prob=0, no schedule) — world-model training SKIPPED "
                  f"(saves {getattr(self.config, 'n_model_update', 2000)} grad steps / 10 iters "
                  f"+ {getattr(self.config, 'n_model_update_warmup', 20000)} at iter {n_warmup})",
                  flush=True)
            self._logged_model_gate = True

        if not _skip_model_training and hasattr(self.agent, 'updateModel') and self.model_buffer is not None:
            model_update_length = getattr(self.config, 'model_update_length', 4)
            model_batch_size = getattr(self.config, 'model_batch_size', 256)
            model_validate_interval = getattr(self.config, 'model_validate_interval', 5)
            model_error_thres = getattr(self.config, 'model_error_thres', 1e-8)

            # Determine if we should train the model this iteration
            if in_warmup:
                # During warmup: just collect data, no model training
                # (matching original: 50 iters of rollout_env() + storeTrajs() only)
                do_model_update = False
            elif current_iteration == n_warmup:
                # First post-warmup iteration: do the big pre-training batch
                do_model_update = True
                n_model_steps = getattr(self.config, 'n_model_update_warmup', 1500)
                print(f"🔥 Warmup complete — pre-training world model ({n_model_steps} gradient steps, buffer has {self.model_buffer.count} trajectories)")
            elif current_iteration % 10 == 0:
                # Regular model update every 10 iterations
                do_model_update = True
                n_model_steps = getattr(self.config, 'n_model_update', 2000)
            else:
                do_model_update = False

            self._last_did_model_update = do_model_update
            if do_model_update and self.model_buffer.count > 0:
                # Cap batch size to buffer contents to avoid excessive duplication
                effective_batch = min(model_batch_size, self.model_buffer.count)
                for i_model_step in range(n_model_steps):
                    # Sample from buffer — ALWAYS real data (Paper Alg 2, line 7)
                    sampled_trajs = self.model_buffer.sampleTrajs(effective_batch)

                    # getFraction truncates to model_update_length from random start
                    truncated_trajs = self._truncate_trajs_for_model(sampled_trajs, model_update_length)

                    if truncated_trajs:
                        model_loss = self.agent.updateModel(truncated_trajs, model_update_length)

                    # Validation with early stopping (matching original main.py)
                    if i_model_step % model_validate_interval == 0 and hasattr(self.agent, 'validateModel'):
                        val_trajs = self.model_buffer.sampleTrajs(effective_batch)
                        val_truncated = self._truncate_trajs_for_model(val_trajs, model_update_length)
                        if val_truncated:
                            rel_error = self.agent.validateModel(val_truncated, length=model_update_length)
                            if rel_error < model_error_thres:
                                print(f"  🎯 Model converged at step {i_model_step}/{n_model_steps} (rel_error={rel_error:.2e})")
                                break

        elif not _skip_model_training and hasattr(self.agent, 'updateModel') and self.model_buffer is None:
            # Fallback: no ModelBuffer available — train on current real_trajs directly
            # (H5a gate covers this branch too — verifier correction #2)
            if not in_warmup:
                model_update_length = getattr(self.config, 'model_update_length', 4)
                model_trajs = self._truncate_trajs_for_model(real_trajs, model_update_length)
                if model_trajs:
                    model_loss = self.agent.updateModel(model_trajs, model_update_length)

        # =============================================================================
        # STEP 3: Inner loop — policy/value updates (Paper Alg 2, lines 8-17)
        #         With model_prob probability, use model rollouts for policy updates
        # =============================================================================
        policy_loss = 0.0
        value_loss = 0.0
        model_data_count = 0
        real_data_count = 0

        for inner in range(n_inner_iter):
            if model_based and model_prob > 0 and hasattr(self.agent, 'model_step') and not in_warmup:
                use_model = np.random.uniform() < model_prob
                if use_model:
                    try:
                        trajs = self._rollout_model(real_trajs)
                        # FINITE-ROLLOUT GATE (2026-08-15): a rollout batch with any
                        # non-finite value poisons the advantage normalization → NaN
                        # advantages → the C6.8 mask assert kills the run (killed the
                        # resumed arm @3770 AND the fresh arm @688 with a full buffer,
                        # so this is a general hazard, not a resume artifact). Reject
                        # the batch at the source; fall back to real data this update.
                        _bad = False
                        for _tr in (trajs if isinstance(trajs, (list, tuple)) else [trajs]):
                            _d = _tr if isinstance(_tr, dict) else getattr(_tr, '__dict__', {})
                            for _k, _v in _d.items():
                                if torch.is_tensor(_v):
                                    if not torch.isfinite(_v.float()).all():
                                        _bad = True; break
                                elif isinstance(_v, np.ndarray) and _v.dtype.kind == 'f':
                                    if not np.isfinite(_v).all():
                                        _bad = True; break
                            if _bad: break
                        if _bad:
                            self._nonfinite_rollouts = getattr(self, '_nonfinite_rollouts', 0) + 1
                            print(f"⚠️ non-finite model rollout rejected "
                                  f"(#{self._nonfinite_rollouts}); using real trajectories")
                            trajs = real_trajs
                            real_data_count += 1
                        else:
                            model_data_count += 1
                    except Exception as e:
                        print(f"⚠️ Model rollout failed: {e}, using real trajectories")
                        trajs = real_trajs
                        real_data_count += 1
                else:
                    trajs = real_trajs
                    real_data_count += 1
            else:
                trajs = real_trajs
                real_data_count += 1

            try:
                update_result = self.agent.updateAgent(trajs)
            except Exception as e:
                if trajs is not real_trajs:
                    print(f"⚠️ Update with model rollouts failed: {e}, retrying with real data")
                    update_result = self.agent.updateAgent(real_trajs)
                else:
                    raise

            # H5c(ii): KL-guard the inner loop too (gated with H5b's flag).
            # updateAgent returns [r_mean, entropy, max(kl_all)] — max over epochs
            # once H5b re-scopes kl_all. Guard on `trajs is real_trajs` so a bad
            # model rollout never aborts the real-data passes. Default off = legacy.
            # Audit F-hygiene: persist the per-iteration max KL (previously the only
            # audit trail was the ABSENCE of a breach print).
            if update_result and len(update_result) > 2 and trajs is real_trajs:
                try:
                    self._last_kl_max = float(update_result[2])
                except Exception:
                    pass
            if (os.environ.get('DMPO_KL_EARLY_STOP', '0') == '1'
                    and update_result and len(update_result) > 2
                    and trajs is real_trajs
                    and getattr(self.agent, 'target_kl', None) is not None
                    and update_result[2] > 1.5 * self.agent.target_kl):
                print(f"⏹  inner-loop early stop at inner={inner + 1}/{n_inner_iter} "
                      f"(|KL|={update_result[2]:.5f})", flush=True)
                break

            # DMPO-Safe: BC anchor loss — pull policy toward past-best behavior.
            # Adds an MSE between policy output and stored (obs, action) anchors.
            # Only fires if the buffer has data (populated by periodic eval).
            if self._safe_enabled and self.bc_anchor is not None and len(self.bc_anchor.episodes) > 0:
                try:
                    self._apply_bc_anchor_loss()
                except Exception as _bc_e:
                    # Soft failure — log once per iter at most
                    if inner == 0:
                        print(f"  ⚠️ BC anchor loss skipped: {_bc_e}")

        # Get losses from logger
        if hasattr(self.agent, 'logger') and hasattr(self.agent.logger, 'metrics'):
            policy_loss = self.agent.logger.metrics.get('surr_loss', 0.0)
            value_loss = self.agent.logger.metrics.get('v_loss', 0.0)
            if hasattr(policy_loss, 'item'):
                policy_loss = policy_loss.item()
            if hasattr(value_loss, 'item'):
                value_loss = value_loss.item()

        # (removed per-iteration "update completed" print — info in >> summary line)
        return float(policy_loss), float(value_loss), float(model_loss)

    # NOTE: Standalone network methods removed - all training must go through DMPO agent
    # The following methods were removed because they trained unused networks:
    # - _update_standalone_networks
    # - _update_policy_networks (standalone version)
    # - _update_value_networks (standalone version)
    # - _update_model_networks (standalone version)
    # - _update_fallback_networks

    def _prepare_dmpo_batch(self):
        """Prepare batch data in DMPO Trajectory format.

        In dynamic mode: builds per-vehicle Trajectory objects with shape
        [T_vehicle, 1, dim]. Each vehicle that appeared during the episode
        gets its own trajectory spanning the timesteps it was alive.

        In original mode: builds [T, n_agent_fixed, dim] with padding.
        """
        if not hasattr(self, 'experience_buffer') or not self.experience_buffer['observations']:
            return []

        obs_dim = self.config.agent_args.observation_dim if hasattr(self.config.agent_args, 'observation_dim') else 130
        action_dim = self.config.agent_args.action_space.shape[0] if hasattr(self.config.agent_args, 'action_space') else 3
        T = len(self.experience_buffer['observations'])

        if T < 2:
            return []

        if getattr(self.agent, 'dynamic', False):
            return self._prepare_dmpo_batch_dynamic(T, obs_dim, action_dim)

        # ============================================================
        # ORIGINAL MODE: Fixed n_agent with slot mapping + padding
        # ============================================================
        n_agent_fixed = self.agent.n_agent if hasattr(self.agent, 'n_agent') else 8

        if T < 2:
            return []

        # SLOT IDENTITY FIX (B1 proper): Use per-step vehicle_slots mapping from
        # the env. Each vehicle has a stable slot for its lifetime; slots are
        # reused after a vehicle leaves. This makes trajectory tensors correctly
        # represent each vehicle's transitions.
        slot_buffer = self.experience_buffer.get('vehicle_slots', [])
        has_slots = len(slot_buffer) == T

        # For backwards compatibility / debug logging:
        all_agent_ids = set()
        for obs_dict in self.experience_buffer['observations']:
            all_agent_ids.update(obs_dict.keys())
        n_agents_actual = len(all_agent_ids)

        # Log batch info (only once per training run)
        if not hasattr(self, '_logged_nagent_info'):
            print(f"📊 DMPO batch preparation:")
            print(f"   - DMPO n_agent (fixed): {n_agent_fixed}")
            print(f"   - Unique agents across episode: {n_agents_actual}")
            print(f"   - Slot identity: {'per-step vehicle_slots (stable)' if has_slots else 'fallback alphabetical'}")
            self._logged_nagent_info = True

        # Build trajectory arrays [T, n_agent_fixed, dim]
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        log_probs = []

        # =============================================================================
        # Agent masks track which slots are real vs padded at each timestep.
        # MAPPO-style masking prevents:
        # 1. Zero log_prob explosion in PPO ratio (exp(logp_new - 0) = huge)
        # 2. Value function training on garbage (zero) observations
        # =============================================================================
        agent_masks = []  # 1.0 for real, 0.0 for padded — per timestep

        # Fallback agent ordering (alphabetical) used only when vehicle_slots
        # mapping is unavailable. This is the legacy behavior.
        agent_ids_fallback = sorted(all_agent_ids)

        # RCPO: pull costs buffer + current λ for reward shaping (r' = r - λ·c).
        cost_buffer_static = self.experience_buffer.get('costs', [])
        lambda_now_static = float(getattr(self, '_lambda_var', 0.0)) if self._constrained_rl else 0.0

        for t in range(T - 1):  # T-1 because we need next_state
            obs_t = self.experience_buffer['observations'][t]
            obs_t1 = self.experience_buffer['observations'][t + 1]
            act_t = self.experience_buffer['actions'][t]
            rew_t = self.experience_buffer['rewards'][t]
            done_t = self.experience_buffer['dones'][t]
            logp_t = self.experience_buffer['log_probs'][t]
            cost_t = cost_buffer_static[t] if t < len(cost_buffer_static) else {}

            # Initialize arrays for EXACTLY n_agent_fixed agents
            s_t = np.zeros((n_agent_fixed, obs_dim), dtype=np.float32)
            a_t = np.zeros((n_agent_fixed, action_dim), dtype=np.float32)
            r_t = np.zeros(n_agent_fixed, dtype=np.float32)
            s1_t = np.zeros((n_agent_fixed, obs_dim), dtype=np.float32)
            d_t = np.zeros(n_agent_fixed, dtype=bool)
            lp_t = np.zeros((n_agent_fixed, action_dim), dtype=np.float32)
            mask_t = np.zeros(n_agent_fixed, dtype=np.float32)

            # SLOT IDENTITY FIX: build (slot, vehicle_id) pairs for this timestep
            # using the env's stable slot assignment. Falls back to alphabetical
            # ordering if slot mapping is unavailable.
            if has_slots and slot_buffer[t]:
                # Use env's per-step vehicle_slots mapping
                slot_pairs = [
                    (slot, vid) for vid, slot in slot_buffer[t].items()
                    if slot < n_agent_fixed and vid in obs_t
                ]
            else:
                # Fallback: alphabetical assignment within first n_agent_fixed
                slot_pairs = [
                    (i, agent_ids_fallback[i])
                    for i in range(min(n_agent_fixed, len(agent_ids_fallback)))
                    if agent_ids_fallback[i] in obs_t
                ]

            # Fill in data for active vehicles at their assigned slots
            for idx, agent_id in slot_pairs:
                # States - handle both tensor and dict formats
                obs = obs_t.get(agent_id, None)
                if obs is None:
                    # Agent not present at this timestep - keep mask=0 (inactive)
                    obs_flat = np.zeros(obs_dim, dtype=np.float32)
                    # mask_t[idx] stays 0.0 - agent is inactive/padded
                elif isinstance(obs, torch.Tensor):
                    # TRUE FIX: Agent HAS data - set mask=1.0 (active)
                    mask_t[idx] = 1.0
                    obs_flat = obs.cpu().numpy().flatten().astype(np.float32)
                elif isinstance(obs, dict):
                    # TRUE FIX: Agent HAS data - set mask=1.0 (active)
                    mask_t[idx] = 1.0
                    obs_flat = self._flatten_observation_numpy(obs)
                else:
                    # TRUE FIX: Agent HAS data - set mask=1.0 (active)
                    mask_t[idx] = 1.0
                    obs_flat = np.atleast_1d(obs).astype(np.float32)

                # Ensure correct dimension (pad or truncate)
                # R1 (obs-v2): pad/truncate would amputate the conflict block or blind
                # the policy — under DMPO_OBS_V2=1 any mismatch is a hard error.
                if os.environ.get('DMPO_OBS_V2', '0') == '1' and len(obs_flat) != obs_dim:
                    raise RuntimeError(f'OBS-V2 dim mismatch at pad site: '
                                       f'{len(obs_flat)} vs obs_dim={obs_dim}')
                if len(obs_flat) < obs_dim:
                    obs_flat = np.pad(obs_flat, (0, obs_dim - len(obs_flat)))
                elif len(obs_flat) > obs_dim:
                    obs_flat = obs_flat[:obs_dim]
                s_t[idx] = obs_flat

                # Actions - convert dict to flat array
                act = act_t.get(agent_id, {})
                if isinstance(act, torch.Tensor):
                    act_flat = act.cpu().numpy().flatten().astype(np.float32)
                elif isinstance(act, dict):
                    _variant = os.environ.get('DMPO_ACTION_VARIANT', 'default')
                    if action_dim == 4 and _variant == 'throttle_brake':
                        act_flat = np.concatenate([
                            np.atleast_1d(act.get('throttle', [0.0])),
                            np.atleast_1d(act.get('brake', [0.0])),
                            np.atleast_1d(act.get('lateral', [0.0])),
                            np.atleast_1d(act.get('lane_intention', [0.0])),
                        ]).astype(np.float32)
                    elif action_dim == 4 and _variant == 'yield_intent':
                        act_flat = np.concatenate([
                            np.atleast_1d(act.get('longitudinal', [0.0])),
                            np.atleast_1d(act.get('lateral', [0.0])),
                            np.atleast_1d(act.get('lane_intention', [0.0])),
                            np.atleast_1d(act.get('yield_intent', [0.0])),
                        ]).astype(np.float32)
                    elif action_dim == 2:
                        act_flat = np.concatenate([
                            np.atleast_1d(act.get('longitudinal', [0.0])),
                            np.atleast_1d(act.get('lateral', [0.0]))
                        ]).astype(np.float32)
                    else:
                        act_flat = np.concatenate([
                            np.atleast_1d(act.get('longitudinal', [0.0])),
                            np.atleast_1d(act.get('lateral', [0.0])),
                            np.atleast_1d(act.get('lane_intention', [0.0]))
                        ]).astype(np.float32)
                else:
                    act_flat = np.atleast_1d(act).astype(np.float32)

                # Ensure correct action dimension
                if len(act_flat) < action_dim:
                    act_flat = np.pad(act_flat, (0, action_dim - len(act_flat)))
                elif len(act_flat) > action_dim:
                    act_flat = act_flat[:action_dim]
                a_t[idx] = act_flat

                # Rewards (RCPO: shape with -λ·c when constrained_rl is on)
                r_raw = float(rew_t.get(agent_id, 0.0))
                if lambda_now_static > 0.0:
                    c = float(cost_t.get(agent_id, 0.0))
                    r_t[idx] = r_raw - lambda_now_static * c
                else:
                    r_t[idx] = r_raw

                # Next states - handle both tensor and dict formats
                obs1 = obs_t1.get(agent_id, None)
                if obs1 is None:
                    obs1_flat = np.zeros(obs_dim, dtype=np.float32)
                elif isinstance(obs1, torch.Tensor):
                    obs1_flat = obs1.cpu().numpy().flatten().astype(np.float32)
                elif isinstance(obs1, dict):
                    obs1_flat = self._flatten_observation_numpy(obs1)
                else:
                    obs1_flat = np.atleast_1d(obs1).astype(np.float32)

                # Ensure correct dimension
                if os.environ.get('DMPO_OBS_V2', '0') == '1' and len(obs1_flat) != obs_dim:
                    raise RuntimeError(f'OBS-V2 dim mismatch at s1 pad site: '
                                       f'{len(obs1_flat)} vs obs_dim={obs_dim}')
                if len(obs1_flat) < obs_dim:
                    obs1_flat = np.pad(obs1_flat, (0, obs_dim - len(obs1_flat)))
                elif len(obs1_flat) > obs_dim:
                    obs1_flat = obs1_flat[:obs_dim]
                s1_t[idx] = obs1_flat

                # Dones
                d_t[idx] = bool(done_t.get(agent_id, False))
                # PAD-FIX (gated): a vehicle absent at t+1 has ended — its slot may be
                # refilled by a new vehicle, and without this the two returns chain.
                # (The dynamic builder applies the same rule.)
                if os.environ.get('DMPO_PAD_FIX', '0') == '1' and agent_id not in obs_t1:
                    d_t[idx] = True

                # Log probs — store the JOINT logp in all action_dim slots.
                # DPPO's get_logp returns [b, n, 1] (joint sum); subtracting [b, n, action_dim]
                # broadcasts to [b, n, action_dim]. With each slot = joint logp (NOT joint/action_dim),
                # the per-dim ratio = exp(lp_new_joint - lp_old_joint) = correct PPO ratio.
                # The previous lp/action_dim split made each per-dim "ratio" be
                # exp(lp_new_joint - lp_old_joint/action_dim), which biases by exp((K-1)/K * lp_old_joint)
                # per sample — sample-dependent corruption of the importance ratio.
                lp = logp_t.get(agent_id, 0.0)
                if np.isscalar(lp):
                    lp_t[idx] = np.full(action_dim, float(lp), dtype=np.float32)
                else:
                    lp_arr = np.atleast_1d(lp).astype(np.float32)
                    if len(lp_arr) < action_dim:
                        lp_arr = np.pad(lp_arr, (0, action_dim - len(lp_arr)))
                    elif len(lp_arr) > action_dim:
                        lp_arr = lp_arr[:action_dim]
                    lp_t[idx] = lp_arr

            # SLOT IDENTITY FIX: Mark padded slots (those not occupied by any
            # vehicle this timestep) as done=True. This prevents GAE corruption.
            # mask_t[idx] is set to 1.0 above only for slots that received data.
            for slot_idx in range(n_agent_fixed):
                if mask_t[slot_idx] < 0.5:
                    d_t[slot_idx] = True  # padded → done (no bootstrapping)

            # ZFilter: normalize observations (update=False — stats already collected during act())
            if self.running_state is not None:           # None under OBS-V2 no-norm
                s_t = self.running_state(s_t, update=False)
                s1_t = self.running_state(s1_t, update=False)

            states.append(s_t)      # [n_agent_fixed, obs_dim]
            actions.append(a_t)     # [n_agent_fixed, action_dim]
            agent_masks.append(mask_t)  # TRUE FIX: [n_agent_fixed] - 1.0 for real, 0.0 for padded
            rewards.append(r_t)     # [n_agent_fixed]
            next_states.append(s1_t)  # [n_agent_fixed, obs_dim]
            dones.append(d_t)       # [n_agent_fixed]
            log_probs.append(lp_t)  # [n_agent_fixed, action_dim]

        # Stack into [T, n_agent_fixed, dim] tensors
        # Trajectory class expects [T, n_agent, dim] - batch dimension is added by updateAgent
        s = torch.as_tensor(np.stack(states, axis=0), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.stack(actions, axis=0), dtype=torch.float32, device=self.device)
        r = torch.as_tensor(np.stack(rewards, axis=0), dtype=torch.float32, device=self.device)
        s1 = torch.as_tensor(np.stack(next_states, axis=0), dtype=torch.float32, device=self.device)
        d = torch.as_tensor(np.stack(dones, axis=0), dtype=torch.bool, device=self.device)
        logp = torch.as_tensor(np.stack(log_probs, axis=0), dtype=torch.float32, device=self.device)

        # =============================================================================
        # TRUE FIX: Stack agent masks [T, n_agent_fixed]
        # =============================================================================
        # Mask enables MAPPO-style loss computation that excludes padded agents
        mask = torch.as_tensor(np.stack(agent_masks, axis=0), dtype=torch.float32, device=self.device)
        # Log mask statistics on first batch
        if not hasattr(self, '_logged_mask_info'):
            active_pct = mask.mean().item() * 100
            print(f"   🎭 Agent mask: {active_pct:.1f}% active agents (real), {100-active_pct:.1f}% padded")
            self._logged_mask_info = True

        # =============================================================================
        # FIX: Return Normalization enabled in DPPO.py
        # =============================================================================
        # DPPO.py now normalizes returns using running mean/std (VecNormalize-style).
        # This prevents value loss explosion by keeping value targets in [-3, 3] range.
        # No reward scaling needed here - DPPO handles normalization internally.

        # r and d need trailing dimension for DPPO: [T, n_agent] -> [T, n_agent, 1]
        r = r.unsqueeze(-1)  # [T, n_agent, 1]
        d = d.unsqueeze(-1)  # [T, n_agent, 1]

        # Create Trajectory object if available - now WITH agent mask
        if 'Trajectory' in self.modules:
            # TRUE FIX: Pass mask to Trajectory for MAPPO-style loss masking
            traj = self.modules['Trajectory'](s=s, a=a, r=r, s1=s1, d=d, logp=logp, mask=mask)
            return [traj]
        else:
            # Return as dict format for model update (include mask)
            return [{"s": s, "a": a, "r": r, "s1": s1, "d": d, "logp": logp, "mask": mask}]
    
    def _intern_ids(self, ids, N):
        """MATCHED MODEL LOSS: tuple[str|None] (len N) → np.int32[N] of interned
        vehicle ids (−1 for None / when the tuple itself is None or short).
        Interning is monotone within one _prepare_dmpo_batch_dynamic call (dict
        insertion order; the dict is reset at the top of each call — ids only
        need to be consistent between sid and sidn of the SAME batch) so equal
        ints ⇔ equal vehicle-id strings within a call."""
        out = np.full(N, -1, dtype=np.int32)
        if not ids:
            return out
        intern = self._vid_intern
        for k, v in enumerate(ids[:N]):
            if v is not None:
                out[k] = intern.setdefault(v, len(intern))
        return out

    def _prepare_dmpo_batch_dynamic(self, T, obs_dim, action_dim):
        """Build per-vehicle Trajectory objects for dynamic tensor mode.

        Instead of a single [T, n_agent_fixed, dim] tensor with padding,
        each vehicle that appeared during the episode gets its own
        Trajectory with shape [T_vehicle, 1, dim].

        This eliminates padding, slot assignment, and agent masks entirely.
        Each trajectory represents one vehicle's full lifespan in the episode.
        """
        from algorithms.algo.buffer import Trajectory

        # Step 1: Collect per-vehicle data across all timesteps
        # vehicle_data[veh_id] = {'s': [...], 'a': [...], 'r': [...], ...}
        vehicle_data = {}
        self._kappa_kept_ids = []      # KAPPA-ADV: kept-trajectory order for the S operator
        if getattr(self, '_matched_on', False):
            # MATCHED MODEL LOSS: interned ids need only be consistent WITHIN this
            # call (sid vs sidn of the same trajectories) — reset so the dict never
            # grows for the life of the run.
            self._vid_intern = {}

        # RCPO: pull costs buffer + current λ for reward shaping (r' = r - λ·c).
        cost_buffer_dyn = self.experience_buffer.get('costs', [])
        lambda_now_dyn = float(getattr(self, '_lambda_var', 0.0)) if self._constrained_rl else 0.0

        for t in range(T - 1):  # T-1 because we need next_state
            obs_t = self.experience_buffer['observations'][t]
            obs_t1 = self.experience_buffer['observations'][t + 1]
            act_t = self.experience_buffer['actions'][t]
            rew_t = self.experience_buffer['rewards'][t]
            done_t = self.experience_buffer['dones'][t]
            logp_t = self.experience_buffer['log_probs'][t]
            cost_t_dyn = cost_buffer_dyn[t] if t < len(cost_buffer_dyn) else {}

            for veh_id in obs_t:
                if veh_id not in vehicle_data:
                    vehicle_data[veh_id] = {
                        's': [], 'a': [], 'r': [], 's1': [], 'd': [], 'logp': [],
                        't_ep': [],  # KAPPA-ADV: episode step of each local index
                        'sid': [], 'sidn': []   # MATCHED MODEL LOSS: slot ids at t / t+1
                    }
                vehicle_data[veh_id]['t_ep'].append(t)

                # State
                obs = obs_t[veh_id]
                if isinstance(obs, torch.Tensor):
                    obs_flat = obs.cpu().numpy().flatten().astype(np.float32)
                elif isinstance(obs, dict):
                    obs_flat = self._flatten_observation_numpy(obs)
                else:
                    obs_flat = np.atleast_1d(obs).astype(np.float32)
                # R1 (obs-v2): pad/truncate would amputate the conflict block or blind
                # the policy — under DMPO_OBS_V2=1 any mismatch is a hard error.
                if os.environ.get('DMPO_OBS_V2', '0') == '1' and len(obs_flat) != obs_dim:
                    raise RuntimeError(f'OBS-V2 dim mismatch at pad site: '
                                       f'{len(obs_flat)} vs obs_dim={obs_dim}')
                if len(obs_flat) < obs_dim:
                    obs_flat = np.pad(obs_flat, (0, obs_dim - len(obs_flat)))
                elif len(obs_flat) > obs_dim:
                    obs_flat = obs_flat[:obs_dim]

                # Next state
                obs1 = obs_t1.get(veh_id, None)
                if obs1 is None:
                    obs1_flat = np.zeros(obs_dim, dtype=np.float32)
                elif isinstance(obs1, torch.Tensor):
                    obs1_flat = obs1.cpu().numpy().flatten().astype(np.float32)
                elif isinstance(obs1, dict):
                    obs1_flat = self._flatten_observation_numpy(obs1)
                else:
                    obs1_flat = np.atleast_1d(obs1).astype(np.float32)
                if os.environ.get('DMPO_OBS_V2', '0') == '1' and len(obs1_flat) != obs_dim:
                    raise RuntimeError(f'OBS-V2 dim mismatch at s1 pad site: '
                                       f'{len(obs1_flat)} vs obs_dim={obs_dim}')
                if len(obs1_flat) < obs_dim:
                    obs1_flat = np.pad(obs1_flat, (0, obs_dim - len(obs1_flat)))
                elif len(obs1_flat) > obs_dim:
                    obs1_flat = obs1_flat[:obs_dim]

                # Action
                act = act_t.get(veh_id, {})
                if isinstance(act, torch.Tensor):
                    act_flat = act.cpu().numpy().flatten().astype(np.float32)
                elif isinstance(act, dict):
                    _variant = os.environ.get('DMPO_ACTION_VARIANT', 'default')
                    if action_dim == 4 and _variant == 'throttle_brake':
                        act_flat = np.concatenate([
                            np.atleast_1d(act.get('throttle', [0.0])),
                            np.atleast_1d(act.get('brake', [0.0])),
                            np.atleast_1d(act.get('lateral', [0.0])),
                            np.atleast_1d(act.get('lane_intention', [0.0])),
                        ]).astype(np.float32)
                    elif action_dim == 4 and _variant == 'yield_intent':
                        act_flat = np.concatenate([
                            np.atleast_1d(act.get('longitudinal', [0.0])),
                            np.atleast_1d(act.get('lateral', [0.0])),
                            np.atleast_1d(act.get('lane_intention', [0.0])),
                            np.atleast_1d(act.get('yield_intent', [0.0])),
                        ]).astype(np.float32)
                    else:
                        act_flat = np.concatenate([
                            np.atleast_1d(act.get('longitudinal', [0.0])),
                            np.atleast_1d(act.get('lateral', [0.0])),
                        ] + ([np.atleast_1d(act.get('lane_intention', [0.0]))] if action_dim == 3 else [])
                        ).astype(np.float32)
                else:
                    act_flat = np.atleast_1d(act).astype(np.float32)
                if len(act_flat) < action_dim:
                    act_flat = np.pad(act_flat, (0, action_dim - len(act_flat)))
                elif len(act_flat) > action_dim:
                    act_flat = act_flat[:action_dim]

                # Reward (scalar) — RCPO shaping when constrained_rl is on.
                r_raw = float(rew_t.get(veh_id, 0.0))
                if lambda_now_dyn > 0.0:
                    c = float(cost_t_dyn.get(veh_id, 0.0))
                    r = r_raw - lambda_now_dyn * c
                else:
                    r = r_raw

                # Done
                d = bool(done_t.get(veh_id, False))
                # If vehicle disappears next step, treat as done for GAE bootstrap
                if veh_id not in obs_t1:
                    d = True

                # Log prob — store JOINT logp (NOT joint/action_dim) so the
                # broadcasted ratio = exp(lp_new_joint - lp_old_joint) is correct.
                lp = logp_t.get(veh_id, 0.0)
                if np.isscalar(lp):
                    lp_arr = np.full(action_dim, float(lp), dtype=np.float32)
                else:
                    lp_arr = np.atleast_1d(lp).astype(np.float32)
                    if len(lp_arr) < action_dim:
                        lp_arr = np.pad(lp_arr, (0, action_dim - len(lp_arr)))
                    elif len(lp_arr) > action_dim:
                        lp_arr = lp_arr[:action_dim]

                # Append with agent dim = 1: [1, dim]
                vehicle_data[veh_id]['s'].append(obs_flat.reshape(1, -1))
                vehicle_data[veh_id]['a'].append(act_flat.reshape(1, -1))
                vehicle_data[veh_id]['r'].append(np.array([[r]], dtype=np.float32))
                vehicle_data[veh_id]['s1'].append(obs1_flat.reshape(1, -1))
                vehicle_data[veh_id]['d'].append(np.array([[d]], dtype=np.bool_))
                vehicle_data[veh_id]['logp'].append(lp_arr.reshape(1, -1))

                # MATCHED MODEL LOSS: interned slot ids at t and t+1 (agent dim = 1).
                # slot_ids_next comes from buffer index t+1 — the SAME condition that
                # makes s1 = zeros / d = True when the vehicle is absent there.
                # Absent ego at t+1 is encoded EXPLICITLY as −2 in every slot (−1 =
                # empty slot of a PRESENT ego) so build_matched_target can mask the
                # whole nb block on those rows.
                if getattr(self, '_matched_on', False):
                    _sl_t = self.experience_buffer['slot_ids'][t] or {}
                    _sl_t1 = self.experience_buffer['slot_ids'][t + 1] or {}
                    _N = self._nb_geom['N']
                    vehicle_data[veh_id]['sid'].append(
                        self._intern_ids(_sl_t.get(veh_id), _N).reshape(1, _N))
                    if veh_id not in obs_t1:
                        vehicle_data[veh_id]['sidn'].append(
                            np.full((1, _N), -2, dtype=np.int32))
                    else:
                        vehicle_data[veh_id]['sidn'].append(
                            self._intern_ids(_sl_t1.get(veh_id), _N).reshape(1, _N))

        # MAPPO BASELINE (gated DMPO_CENTRAL_CRITIC): global feature g_t over ALL
        # vehicles present at each buffered step (incl. t = T-1 for s1).
        _cc_on = os.environ.get('DMPO_CENTRAL_CRITIC', '0') == '1'
        if _cc_on:
            from dmpo_autonomous_driving.utils import central_critic as _cc
            def _flat_any(o):
                if isinstance(o, torch.Tensor):
                    return o.cpu().numpy().flatten().astype(np.float32)
                if isinstance(o, dict):
                    return self._flatten_observation_numpy(o)
                return np.atleast_1d(o).astype(np.float32)
            _gfeat = []
            for _t in range(T):
                _ob = self.experience_buffer['observations'][_t]
                _rows = [_flat_any(_o) for _o in _ob.values()]
                _gfeat.append(_cc.global_feat(np.stack(_rows) if _rows
                                              else np.zeros((0, obs_dim), np.float32)))
            _gfeat = torch.as_tensor(np.stack(_gfeat), dtype=torch.float32, device=self.device)

        # Step 2: Convert to Trajectory objects [T_vehicle, 1, dim]
        trajectories = []
        min_length = 2  # Need at least 2 timesteps for GAE

        for veh_id, data in vehicle_data.items():
            if len(data['s']) < min_length:
                continue

            s = torch.as_tensor(np.stack(data['s']), dtype=torch.float32, device=self.device)
            a = torch.as_tensor(np.stack(data['a']), dtype=torch.float32, device=self.device)
            r = torch.as_tensor(np.stack(data['r']), dtype=torch.float32, device=self.device)
            s1 = torch.as_tensor(np.stack(data['s1']), dtype=torch.float32, device=self.device)
            d = torch.as_tensor(np.stack(data['d']), dtype=torch.bool, device=self.device)
            logp = torch.as_tensor(np.stack(data['logp']), dtype=torch.float32, device=self.device)

            # Normalize observations using stats from act-time
            # (update=False: act-time already updated stats while collecting,
            # re-updating here would diverge act-time and train-time
            # normalizations and corrupt PPO's importance ratio).
            s_np = s.cpu().numpy().reshape(-1, obs_dim)
            s_np = self._normalize_obs_dynamic(s_np, update=False)
            s = torch.as_tensor(s_np.reshape(s.shape), dtype=torch.float32, device=self.device)
            s1_np = s1.cpu().numpy().reshape(-1, obs_dim)
            s1_np = self._normalize_obs_dynamic(s1_np, update=False)
            s1 = torch.as_tensor(s1_np.reshape(s1.shape), dtype=torch.float32, device=self.device)
            if _cc_on:
                _te = torch.as_tensor(data['t_ep'], dtype=torch.long, device=self.device)
                s = _cc.extend(s, _gfeat[_te])
                s1 = _cc.extend(s1, _gfeat[_te + 1])

            traj = Trajectory(s=s, a=a, r=r, s1=s1, d=d, logp=logp)
            if getattr(self, '_matched_on', False):
                # Stored in traj.dict ONLY (never appended to traj.names): DPPO's
                # updateAgent stacks Trajectory.names() = the 6 base names, so the
                # extra keys are ignored there; the model loss reads traj.dict.get().
                # NOT normalized (ids), NOT z-filtered. Shape [T_v, 1, N] int32.
                traj.dict['slot_ids'] = torch.as_tensor(
                    np.stack(data['sid']), dtype=torch.int32, device=self.device)
                traj.dict['slot_ids_next'] = torch.as_tensor(
                    np.stack(data['sidn']), dtype=torch.int32, device=self.device)
                assert traj.dict['slot_ids'].shape[0] == s.shape[0]
            trajectories.append(traj)
            self._kappa_kept_ids.append(veh_id)

        if not trajectories:
            return []

        # ── KAPPA-ADV (gated DMPO_KAPPA_ADV): build the per-batch κ-hop neighbor-sum
        # operator S over (trajectory, local-timestep) slots, from the ACTION-TIME
        # graphs captured at collect. This restores the ORIGINAL framework's
        # `reduced_advantages = collect_v.reduce_sum(advantages)` (pristine DPPO:363,
        # unconditional there) for the dynamic graph — the credit-assignment term the
        # dynamic adaptation silently dropped. DPPO's dynamic branch consumes S via
        # torch.sparse.mm at the exact bypass site.
        kept_ids = getattr(self, '_kappa_kept_ids', [])
        self._kappa_kept_ids = []                       # reset for the next batch
        self.agent._kappa_adv_S = None                  # never reuse a stale operator
        if self._kappa_adv_on and trajectories:
            if int(getattr(self.args, 'n_episodes', 1) or 1) > 1:
                if not getattr(self, '_kappa_adv_warned', False):
                    print("⚠️ KAPPA_ADV disabled: --n-episodes > 1 (flow-id reuse across "
                          "episodes would merge different vehicles; pin --n-episodes 1)")
                    self._kappa_adv_warned = True
            else:
                T_max = max(tr.length for tr in trajectories)
                b_of = {vid: i for i, vid in enumerate(kept_ids)}
                loc = {}
                for vid in kept_ids:
                    for lt, tep in enumerate(vehicle_data[vid]['t_ep']):
                        loc[(vid, tep)] = lt
                # CLI --kappa is authoritative (argparse default 2). The old
                # `args.kappa or env or 2` fallback swallowed κ=0 (falsy in the
                # or-chain) and fell through to the manifest's DMPO_KAPPA=2, so
                # the κ=0 ablation arm silently trained with κ=2 credit routing
                # (compounded by max(kappa,1) below). κ=0 now means what it
                # says: no operator, independent advantages.
                kappa = int(getattr(self.args, 'kappa', 2))
                rows, cols = [], []
                if kappa == 0:
                    adjs, idls = [], []
                    if not getattr(self, '_kappa_adv_logged', False):
                        print("🕸️ KAPPA_ADV: κ=0 — operator disabled "
                              "(independent advantages)")
                        self._kappa_adv_logged = True
                else:
                    adjs = self.experience_buffer.get('adj_pre', [])
                    idls = self.experience_buffer.get('adj_ids', [])
                skipped_steps = 0
                for tep in range(min(len(adjs), len(idls))):
                    A, ids = adjs[tep], idls[tep]
                    if A is None or ids is None or getattr(A, 'size', 0) == 0 \
                            or A.shape[0] != len(ids):
                        skipped_steps += 1
                        continue
                    M = (A > 0).astype(np.float32)
                    np.fill_diagonal(M, 1.0)            # self-loop ⇒ reach = ≤κ hops
                    R = np.linalg.matrix_power(M, max(kappa, 1)) > 0
                    for i, vi in enumerate(ids):
                        bi = b_of.get(vi)
                        li = loc.get((vi, tep))
                        if bi is None or li is None:
                            continue
                        ri = bi * T_max + li
                        for j in np.nonzero(R[i])[0]:
                            bj = b_of.get(ids[j])
                            lj = loc.get((ids[j], tep))
                            if bj is None or lj is None:
                                continue
                            rows.append(ri)
                            cols.append(bj * T_max + lj)
                if rows:
                    n_slots = len(kept_ids) * T_max
                    vals = torch.ones(len(rows), dtype=torch.float32)
                    if os.environ.get('DMPO_KAPPA_ADV_MEAN', '0') == '1':
                        # MEAN aggregation (round-4 arm): degree-normalize each row.
                        # The original's SUM is scale-consistent only for fixed-degree
                        # grids; on our variable-degree dynamic graph it multiplies
                        # the collision fear by neighborhood size (~8x at dense
                        # moments) — the measured over-caution driver. Mean keeps
                        # the cooperative direction with degree-invariant magnitude.
                        import collections as _cA
                        _deg = _cA.Counter(rows)
                        vals = torch.tensor([1.0 / _deg[r] for r in rows],
                                            dtype=torch.float32)
                    S = torch.sparse_coo_tensor(
                        torch.tensor([rows, cols], dtype=torch.long),
                        vals,
                        size=(n_slots, n_slots)).coalesce().to(self.device)
                    self.agent._kappa_adv_S = S
                    if not getattr(self, '_kappa_adv_logged', False):
                        print(f"🕸️ KAPPA_ADV: S built — {len(rows)} neighbor-sum entries "
                              f"over {n_slots} slots (κ={kappa}, "
                              f"{skipped_steps} graph-less steps skipped)")
                        self._kappa_adv_logged = True

        if not hasattr(self, '_logged_dynamic_batch'):
            n_vehs = len(trajectories)
            avg_len = np.mean([t.length for t in trajectories])
            print(f"📊 Dynamic batch: {n_vehs} vehicle trajectories, avg length {avg_len:.0f}")
            self._logged_dynamic_batch = True

        return trajectories

    def _print_fixes_banner(self):
        """Print a self-verifying banner of which V8/V9 fixes are ACTIVE.

        Reads the env vars directly so the training log proves, at startup,
        exactly which fixes are live. Prevents the silent-no-op problem where a
        fix is assumed on but was never actually applied. Each ✅/⬜ reflects the
        real os.environ value the env/policy will use this run.
        """
        import os as _o
        def _f(v, d='0'): return float(_o.environ.get(v, d))
        def _on(cond): return "✅ ON " if cond else "⬜ off"
        max_decel   = _f('DMPO_MAX_DECEL', '4.5')
        bias_long   = _f('DMPO_POLICY_BIAS_LONGITUDINAL', '0.0')
        anneal_it   = int(_f('DMPO_POLICY_BIAS_ANNEAL_ITERS', '0'))
        thin_floor  = _f('DMPO_KRAUSS_THIN_FLOOR_M', '0')
        rear_bumper = _o.environ.get('DMPO_OBS_REAR_BUMPER', '0') == '1'
        lateral_k   = _f('DMPO_LATERAL_CENTERING_K', '0')
        cmdp        = _o.environ.get('DMPO_CONSTRAINED_RL', '0') == '1'
        ttc_k       = _f('DMPO_TTC_SHAPING_K', '0.0')
        print("\n" + "="*70)
        print("  🔧 ACTIVE FIXES (self-verification — confirm before long runs)")
        print("="*70)
        print(f"  V8 braking cap (max_decel):     {_on(max_decel != 4.5)}  {max_decel:.1f} m/s²  (V7=4.5)")
        print(f"  V8 bias anneal:                 {_on(anneal_it > 0)}  start={bias_long:+.2f} -> 0 over {anneal_it} iters")
        print(f"  V8 thin emergency floor:        {_on(thin_floor > 0)}  gap<{thin_floor:.1f}m engages Krauss (else policy drives)")
        print(f"  V9 rear-bumper perception:      {_on(rear_bumper)}  (obs uses leader REAR bumper)")
        print(f"  V9 lateral-centering fix:       {_on(lateral_k > 0)}  K={lateral_k:.2f}  (pull toward lane center)")
        print(f"  --- KNOWN-BAD levers (should be OFF) ---")
        print(f"  CMDP/RCPO (collapses):          {_on(cmdp)}  {'⚠ WARNING: ON' if cmdp else ''}")
        print(f"  TTC shaping (confounded):       {_on(ttc_k > 0)}  K={ttc_k:.1f}  {'⚠' if ttc_k > 0 else ''}")
        print("="*70 + "\n", flush=True)

    def train(self):
        """Run full DMPO training loop."""
        self._print_fixes_banner()
        # Eager resume for dynamic mode: shared networks are sized correctly at
        # __init__, so we can load the checkpoint NOW (before the for-loop
        # range is captured) so the iter counter starts at the resumed value.
        # Without this, the for-loop's `range(self.start_iteration, ...)` is
        # evaluated when start_iteration is still 0, and the late load inside
        # _update_dmpo_agent updates start_iteration but the loop has already
        # been bound — so G2 would train 2000 fresh iters instead of (2000 -
        # resumed). Non-dynamic mode still resumes lazily because it needs
        # _detect_actual_n_agents() to size the agent first.
        if getattr(self.agent, 'dynamic', False):
            if hasattr(self, '_pending_resume') and self._pending_resume is not None:
                print(f"📂 [Eager resume, dynamic mode] Loading checkpoint from iter {self._pending_resume}...")
                self._load_checkpoint(self._pending_resume)
                self._pending_resume = None

        if self.start_iteration > 0:
            print(f"🔄 Resuming DMPO Training from iteration {self.start_iteration}")
        else:
            print("🚀 Starting Full DMPO Neural Network Training")
        print("="*70)

        # Initialize environment once at the start of training
        print("🔄 Initializing environment for training...")
        self._initial_observations = self.env.reset()
        self._env_initialized = True
        print("✅ Environment initialized - SUMO GUI should now be visible and stable")

        start_time = time.time()

        for iteration in range(self.start_iteration, self.args.n_iterations):
            iteration_start = time.time()

            # Track current iteration for model-based policy optimization warmup check
            self._current_iteration = iteration

            # V8: anneal the longitudinal cold-start bias toward 0 (no-op if disabled)
            self._cur_bias_anneal = self._apply_bias_anneal(iteration)

            # ── ARM-C EXPLORATION ANNEAL (gated DMPO_STD_ANNEAL="start:end:target_log_std",
            # e.g. "5000:9000:-1.6" ⇒ std 0.58→0.20 linearly over iters 5000-9000;
            # default unset = no-op). Pure-RL exploration schedule (action-std decay,
            # standard continuous-control practice): late-only so skill formation keeps
            # full exploration, then the noise crutch (ties resolved by jitter for free;
            # gap margins learned under ±0.58 action noise) is withdrawn WHILE learning
            # can still adapt — training behavior converges to the deterministic-mean
            # behavior that eval grades. Implemented by lowering the actors' log-std
            # clamp CEILING (the learned log_std parameter is clamped at forward time),
            # never touching the floor (DMPO_LOG_STD_MIN).
            _sa = os.environ.get('DMPO_STD_ANNEAL', '')
            if _sa:
                try:
                    _s0, _s1, _tgt = _sa.split(':')
                    _s0, _s1, _tgt = int(_s0), int(_s1), float(_tgt)
                    _base = float(os.environ.get('DMPO_LOG_STD_MAX', '1.0'))
                    if iteration >= _s0 and _s1 > _s0:
                        _f = min(1.0, (iteration - _s0) / float(_s1 - _s0))
                        _cur = _base + (_tgt - _base) * _f
                        for _a in getattr(self.agent, 'actors', []):
                            if hasattr(_a, 'LOG_STD_MAX'):
                                _a.LOG_STD_MAX = _cur
                        if hasattr(self.agent, '_log_std_max'):
                            self.agent._log_std_max = _cur
                except (ValueError, AttributeError) as _e:
                    print(f"⚠️ DMPO_STD_ANNEAL malformed/inapplicable ({_sa!r}): {_e}")

            # ── WAVE-4 LR DECAY (gated DMPO_LR_DECAY="start:end", default unset =
            # no-op): linearly decay every optimizer's learning rate to ZERO between
            # the given iterations, so the run CONVERGES to a stable endpoint instead
            # of oscillating forever at constant LR (the select-and-freeze pathology;
            # SWA showed the band's center ≈ honest level — this makes the final
            # policy BE that center). Original LRs are cached on first application.
            _ld = os.environ.get('DMPO_LR_DECAY', '')
            if _ld:
                try:
                    _d0, _d1 = (int(x) for x in _ld.split(':'))
                    if iteration >= _d0 and _d1 > _d0:
                        _f = max(0.0, 1.0 - (iteration - _d0) / float(_d1 - _d0))
                        # Attribute scan misses list-held optimizers (verified bug
                        # 2026-07-31): the world-model ensemble keeps members 1-4
                        # only in this list; include them so "every optimizer"
                        # holds if model training is ever active alongside decay.
                        _opts = [getattr(self.agent, _a, None) for _a in dir(self.agent)]
                        _opts += list(getattr(self.agent, '_ensemble_optimizers', None) or [])
                        for _o in _opts:
                            if isinstance(_o, torch.optim.Optimizer):
                                for _g in _o.param_groups:
                                    if '_lr0' not in _g:
                                        _g['_lr0'] = _g['lr']
                                    _g['lr'] = _g['_lr0'] * _f
                        if (iteration - _d0) % 1000 == 0:
                            print(f"📉 LR-decay factor {_f:.2f} at iter {iteration}")
                except ValueError:
                    print(f"⚠️ DMPO_LR_DECAY malformed ({_ld!r}) — expected 'start:end'")

            # ── SHIELD-SCAFFOLD ANNEAL (gated DMPO_SHIELD_ANNEAL_ITERS; portfolio
            # arm 4, user-approved 2026-07-25): scale the scaffold's engagement
            # envelope (PET window + approach zone) linearly to ZERO by N iters, so
            # the policy trains on shield-shaped experience early (worked examples
            # of conflict-conditioned braking; shield = environment dynamics, the
            # buffer stores the policy's own action) and finishes with the scaffold
            # provably gone — the teacher-not-deployment attribution story. The
            # shield reads these env vars at use-time each step, so a per-iteration
            # rewrite takes effect immediately. Kill-gate metric: shield_fired/iter
            # (surfaced in the >> line) must decay to ~0 before convergence.
            _sa_iters = int(os.environ.get('DMPO_SHIELD_ANNEAL_ITERS', '0') or 0)
            if _sa_iters > 0 and os.environ.get('DMPO_GAP_SHIELD', '0') == '1':
                if not hasattr(self, '_shield_anneal_base'):
                    self._shield_anneal_base = (
                        float(os.environ.get('DMPO_GAP_SHIELD_PET_S', '4.0')),
                        float(os.environ.get('DMPO_GAP_SHIELD_APPROACH_ZONE_M', '30')),
                        float(os.environ.get('DMPO_GAP_SHIELD_GENUINE_PET_S', '1.5')))
                _f = max(0.0, 1.0 - iteration / _sa_iters)
                _b = self._shield_anneal_base
                os.environ['DMPO_GAP_SHIELD_PET_S'] = str(_b[0] * _f)
                os.environ['DMPO_GAP_SHIELD_APPROACH_ZONE_M'] = str(_b[1] * _f)
                os.environ['DMPO_GAP_SHIELD_GENUINE_PET_S'] = str(_b[2] * _f)

            # Update entropy coefficient based on schedule (exploration → exploitation)
            current_entropy_coeff = self.entropy_scheduler.get_coefficient(iteration)
            if os.environ.get('DMPO_ENTROPY_SUSTAIN', '0') == '1':
                current_entropy_coeff = float(os.environ.get('DMPO_ENTROPY', '0.05'))  # hold exploration, no decay
            self._update_entropy_coefficient(current_entropy_coeff)
            if 'entropy_coefficients' not in self.training_metrics:
                self.training_metrics['entropy_coefficients'] = []
            self.training_metrics['entropy_coefficients'].append(current_entropy_coeff)

            # Curriculum: gradually increase traffic density.
            # Mode 'scale' (V2): ramp SUMO inflow, cap fixed above peak → every
            #   vehicle stays policy-controlled at all densities (no Krauss release),
            #   and n_agent never changes so there is no reinit/weight-transfer churn.
            # Mode 'cap' (legacy): ramp max_ai_vehicles + reinit DMPO each stage.
            if (os.environ.get('DMPO_CURRICULUM', '0') == '1'
                    and os.environ.get('DMPO_CURRICULUM_MODE', 'cap') == 'scale'):
                # DMPO_CURRICULUM_SCHEDULE format here: "iter1:scale1,iter2:scale2,..."
                sched_env = os.environ.get('DMPO_CURRICULUM_SCHEDULE',
                                           '0:0.2,400:0.4,1000:0.6,1700:0.8,2300:1.0')
                sched = []
                for part in sched_env.split(','):
                    it_str, sc_str = part.split(':')
                    sched.append((int(it_str), float(sc_str)))
                new_scale = sched[0][1]
                for threshold, sc in reversed(sched):
                    if iteration >= threshold:
                        new_scale = sc
                        break
                old_scale = getattr(self.env, '_traffic_scale', 1.0)
                if iteration == 0 or abs(new_scale - old_scale) > 1e-9:
                    if abs(new_scale - old_scale) > 1e-9:
                        print(f"📈 Curriculum (inflow scale): {old_scale} → {new_scale} (iter {iteration})")
                    self.env.set_traffic_scale(new_scale)
            elif os.environ.get('DMPO_CURRICULUM', '0') == '1':
                # DMPO_CURRICULUM_SCHEDULE format: "iter1:cap1,iter2:cap2,..."
                # Default: ramp from 5 to 30 over 5000 iters for V7 scale-up.
                schedule_env = os.environ.get('DMPO_CURRICULUM_SCHEDULE',
                                              '0:5,500:10,1500:15,2500:20,3500:30')
                curriculum_schedule = []
                for part in schedule_env.split(','):
                    it_str, cap_str = part.split(':')
                    curriculum_schedule.append((int(it_str), int(cap_str)))
                new_cap = curriculum_schedule[0][1]  # initial cap
                for threshold, n_veh in reversed(curriculum_schedule):
                    if iteration >= threshold:
                        new_cap = n_veh
                        break

                old_cap = self.env.max_ai_vehicles
                if new_cap != old_cap:
                    print(f"📈 Curriculum: {old_cap} → {new_cap} vehicles (iter {iteration})")
                    # Save old weights before reinitialization
                    old_actors_state = self.agent.actors.state_dict() if hasattr(self.agent, 'actors') else None
                    old_vs_state = self.agent.vs.state_dict() if hasattr(self.agent, 'vs') else None

                    # Update env cap
                    self.env.max_ai_vehicles = new_cap
                    self.env._max_ai_vehicles = new_cap

                    # Reinitialize DMPO with new n_agent = new_cap
                    self._reinitialize_dmpo_for_n_agents(new_cap)

                    # Transfer weights from old actors to new (overlapping slots)
                    if old_actors_state is not None and hasattr(self.agent, 'actors'):
                        new_state = self.agent.actors.state_dict()
                        for key in old_actors_state:
                            if key in new_state and old_actors_state[key].shape == new_state[key].shape:
                                new_state[key] = old_actors_state[key]
                            elif key in new_state:
                                # Partial transfer for size-mismatched layers (e.g., input layer changes with degree)
                                old_shape = old_actors_state[key].shape
                                new_shape = new_state[key].shape
                                min_shape = tuple(min(o, n) for o, n in zip(old_shape, new_shape))
                                slices = tuple(slice(0, m) for m in min_shape)
                                new_state[key][slices] = old_actors_state[key][slices]
                        self.agent.actors.load_state_dict(new_state)
                        print(f"  - Transferred actor weights (overlapping slots)")

                    if old_vs_state is not None and hasattr(self.agent, 'vs'):
                        new_state = self.agent.vs.state_dict()
                        for key in old_vs_state:
                            if key in new_state and old_vs_state[key].shape == new_state[key].shape:
                                new_state[key] = old_vs_state[key]
                            elif key in new_state:
                                old_shape = old_vs_state[key].shape
                                new_shape = new_state[key].shape
                                min_shape = tuple(min(o, n) for o, n in zip(old_shape, new_shape))
                                slices = tuple(slice(0, m) for m in min_shape)
                                new_state[key][slices] = old_vs_state[key][slices]
                        self.agent.vs.load_state_dict(new_state)
                        print(f"  - Transferred value weights (overlapping slots)")

                elif iteration == 0:
                    # First iteration: set initial cap
                    self.env.max_ai_vehicles = new_cap
                    self.env._max_ai_vehicles = new_cap

            # Collect experience using current policy
            t_collect = time.time()
            collection_results = self.collect_experience(self.args.n_episodes)
            t_collect = time.time() - t_collect

            # Update DMPO neural networks
            t_update = time.time()
            policy_loss, value_loss, model_loss = self.update_dmpo()
            t_update = time.time() - t_update
            print(f"  ⏱️ collect={t_collect:.1f}s, update={t_update:.1f}s")

            # RCPO Lagrangian dual update: λ ← max(0, λ + η · (µ_cost - d)).
            # Computed AFTER the policy update so the next iteration's data
            # collection sees the new λ. Mean cost is per-episode total.
            if self._constrained_rl:
                cost_buf = self.experience_buffer.get('costs', [])
                ep_total_cost = 0.0
                for c_dict in cost_buf:
                    if isinstance(c_dict, dict):
                        ep_total_cost += sum(float(v) for v in c_dict.values())
                n_eps = max(int(self.args.n_episodes), 1)
                mean_cost = ep_total_cost / n_eps
                # Gradient ascent on λ when cost > budget; descent when below.
                self._lambda_var = max(
                    0.0,
                    min(self._lambda_max,
                        self._lambda_var + self._lambda_lr * (mean_cost - self._cost_budget))
                )
                if 'lambda_history' not in self.training_metrics:
                    self.training_metrics['lambda_history'] = []
                    self.training_metrics['mean_cost_history'] = []
                self.training_metrics['lambda_history'].append(self._lambda_var)
                self.training_metrics['mean_cost_history'].append(mean_cost)
                print(f"  🔒 RCPO: mean_cost={mean_cost:.2f} (budget={self._cost_budget}) "
                      f"→ λ={self._lambda_var:.3f}")
            
            # Get metrics from collection results
            avg_reward = collection_results['avg_reward']
            avg_length = collection_results['avg_length']
            
            # Store metrics (defensive after resume — keys may be missing if
            # resumed checkpoint didn't track them)
            for _k in ['iteration_rewards', 'episode_lengths', 'policy_losses',
                       'value_losses', 'model_losses', 'safety_metrics']:
                if _k not in self.training_metrics:
                    self.training_metrics[_k] = []
            self.training_metrics['iteration_rewards'].append(avg_reward)
            self.training_metrics['episode_lengths'].append(avg_length)
            self.training_metrics['policy_losses'].append(policy_loss)
            self.training_metrics['value_losses'].append(value_loss)
            self.training_metrics['model_losses'].append(model_loss)
            # Audit F-hygiene: per-iteration max KL series (index-locked with losses)
            self.training_metrics.setdefault('kl_max', []).append(
                getattr(self, '_last_kl_max', None))
            avg_completions = collection_results.get('avg_completions', 0)
            if 'completions' not in self.training_metrics:
                self.training_metrics['completions'] = []
            self.training_metrics['completions'].append(avg_completions)
            
            # P0.1: REAL AI collision count for this iter — accumulated from
            # info['ai_collisions_episode'] during data collection (see
            # _collect_experience). Also captures pair_kind / mechanism
            # breakdowns when DMPO_COLLISION_TRACE=1.
            iter_ai_collisions = collection_results.get('iter_ai_collisions', 0)
            iter_collision_breakdown = collection_results.get('iter_collision_breakdown', {})
            iter_mechanism_breakdown = collection_results.get('iter_mechanism_breakdown', {})

            # Safety metrics — the legacy safety_monitor dict was STALE during
            # training (filter_action short-circuits in training_mode;
            # check_collision_occurred is never called) so it always read 0.
            # We now store the REAL per-iter safety summary under the same key
            # for back-compat, and still reset the monitor so it never leaks a
            # fake status into any future read.
            self.training_metrics['safety_metrics'].append({
                'ai_collisions': iter_ai_collisions,
                'completions': avg_completions,
                'collision_breakdown': dict(iter_collision_breakdown),
                'mechanism_breakdown': dict(iter_mechanism_breakdown),
            })
            self.env.safety_monitor.reset_metrics()
            if 'ai_collisions' not in self.training_metrics:
                self.training_metrics['ai_collisions'] = []
                self.training_metrics['ai_collision_breakdown'] = []
                self.training_metrics['ai_mechanism_breakdown'] = []
            self.training_metrics['ai_collisions'].append(iter_ai_collisions)
            self.training_metrics['ai_collision_breakdown'].append(dict(iter_collision_breakdown))
            self.training_metrics['ai_mechanism_breakdown'].append(dict(iter_mechanism_breakdown))

            # Dynamics monitor — compact one-line summary + warnings
            self._log_training_dynamics(iteration, value_loss, avg_reward, model_loss, current_entropy_coeff)

            # Logging
            if (iteration + 1) % self.args.log_interval == 0:
                elapsed = time.time() - start_time
                remaining = elapsed / (iteration + 1) * (self.args.n_iterations - iteration - 1)
                entropy_phase = self.entropy_scheduler.get_phase(iteration)

                print(f"\n📊 Iteration {iteration + 1}/{self.args.n_iterations}")
                print(f"  - Average reward: {avg_reward:.3f}")
                print(f"  - Episode length: {avg_length:.1f}")
                print(f"  - Trip completions/episode: {avg_completions:.1f}")
                # NOTE: this is the PPO surrogate OBJECTIVE (higher is better), logged as
                # 'surr_loss' upstream — it is not a minimized loss. Labelled accordingly.
                print(f"  - Surrogate objective: {policy_loss:.6f}")
                print(f"  - Value loss: {value_loss:.6f}")
                # Policy exploration width — THE convergence signal. Should DECREASE
                # from the init (std≈0.61) as the policy commits; a value pinned at the
                # ceiling (std=exp(LOG_STD_MAX)) means it never learned (≈random).
                try:
                    if hasattr(self.agent, '_shared_log_std'):       # attention policy
                        _ls = self.agent._shared_log_std.detach().cpu()
                    else:                                            # plain-MLP actor
                        _ls = self.agent.actors[0].log_std.detach().cpu()
                    import torch as _t
                    print(f"  - Policy std (exploration): {_t.exp(_ls).mean().item():.4f}  (log_std={_ls.mean().item():+.3f})")
                except Exception:
                    pass
                # Model loss is only meaningful on iterations where a world-model update
                # actually ran (warmup-end + every 10th iter). On other iters it is 0 by
                # init, which previously printed a misleading 'Model loss: 0.000000'.
                if getattr(self, '_last_did_model_update', False):
                    print(f"  - Model loss: {model_loss:.6f}  (world-model rel-err: "
                          f"{getattr(self.agent, '_last_model_rel_error', float('nan')):.4f})")
                else:
                    print(f"  - Model loss: n/a (no world-model update this iter)  "
                          f"(last rel-err: {getattr(self.agent, '_last_model_rel_error', float('nan')):.4f})")
                print(f"  - Entropy coeff: {current_entropy_coeff:.4f} ({entropy_phase})")
                # P0.1: replace the stale Safety dict with the REAL collision
                # numbers. The legacy dict is kept on the metrics object for
                # back-compat but no longer printed (it always said 0).
                print(f"  - AI collisions (this iter): {iter_ai_collisions}")
                if iter_collision_breakdown:
                    bd_str = ', '.join(f'{k}={v}' for k, v in
                                       sorted(iter_collision_breakdown.items(), key=lambda x: -x[1]))
                    print(f"      by pair_kind:  {bd_str}")
                if iter_mechanism_breakdown:
                    mb_str = ', '.join(f'{k}={v}' for k, v in
                                       sorted(iter_mechanism_breakdown.items(), key=lambda x: -x[1]))
                    print(f"      by mechanism:  {mb_str}")
                print(f"  - Time: {time.time() - iteration_start:.2f}s")
                print(f"  - ETA: {remaining/3600:.2f} hours")
                # V8/V9 fix RUNTIME confirmation — proves fixes are actually firing,
                # not just configured. Resets the per-iter counters after reading.
                _bias = getattr(self, '_cur_bias_anneal', None)
                _floor_fired = getattr(self.env, '_krauss_floor_fired', 0)
                _floor_skipped = getattr(self.env, '_krauss_floor_skipped', 0)
                _bias_live = self.agent.actors[0].action_head.bias.data[0].item() \
                    if hasattr(self.agent, 'actors') and len(self.agent.actors) else None
                _fix_bits = []
                if _bias is not None:
                    _fix_bits.append(f"bias_anneal={_bias:+.3f}")
                if _bias_live is not None:
                    _fix_bits.append(f"bias_live={_bias_live:+.3f}")
                if (_floor_fired + _floor_skipped) > 0:
                    _fix_bits.append(f"thin_floor[fired={_floor_fired},skipped={_floor_skipped}]")
                if os.environ.get('DMPO_OBS_REAR_BUMPER', '0') == '1':
                    _fix_bits.append("rear_bumper=ON")
                _lat_n = getattr(self.env, '_lat_centering_applied', 0)
                if _lat_n > 0:
                    _lat_drift = getattr(self.env, '_lat_centering_drift_sum', 0.0) / max(1, _lat_n)
                    _fix_bits.append(f"lat_center[n={_lat_n},avg_drift={_lat_drift:.2f}m]")
                if _fix_bits:
                    print(f"  - 🔧 fixes firing: {' | '.join(_fix_bits)}")
                # reset per-iter firing counters
                self.env._krauss_floor_fired = 0
                self.env._krauss_floor_skipped = 0
                self.env._lat_centering_applied = 0
                self.env._lat_centering_drift_sum = 0.0

            # Save checkpoints at regular intervals only
            # NOTE: Early checkpoint saving (every 10 iters) disabled - it consumed
            # 14 GB of disk on 12 parallel runs (each checkpoint ~170-200 MB)
            regular_save = (iteration + 1) % self.args.save_interval == 0
            # WAVE-5 dense-save window (gated DMPO_SAVE_DENSE="start:end:step",
            # default unset = no-op): gem checkpoints cluster in a narrow
            # post-breakaway window while adjacent 50-grid screens differ by 1-3
            # crashes — the grid subsamples an oscillating trajectory. Denser
            # UNBIASED saving there (screens still judge) is the evidence-backed
            # form of metric-triggered saving: rolling training col/comp proved
            # uncorrelated with screen quality (|rho|<=0.03; gems captured 0/4).
            _sd = os.environ.get('DMPO_SAVE_DENSE', '')
            if _sd and not regular_save:
                try:
                    _s0, _s1, _st = (int(x) for x in _sd.split(':'))
                    if _s0 <= iteration + 1 <= _s1 and (iteration + 1) % _st == 0:
                        regular_save = True
                except ValueError:
                    print(f"⚠️ DMPO_SAVE_DENSE malformed ({_sd!r}) — expected 'start:end:step'")
            if regular_save:
                self._save_checkpoint(iteration + 1)
        
        # Final results
        self._save_final_results(start_time)
        print("🎉 Full DMPO training completed successfully!")
    
    def _load_checkpoint(self, iteration=None):
        """Load training checkpoint to resume training.

        Args:
            iteration: Specific iteration to load, or None to find latest
        """
        import glob

        # Find checkpoint to load
        if iteration is not None:
            checkpoint_path = os.path.join(self.args.checkpoint_dir, f'dmpo_checkpoint_{iteration}.json')
            model_path = os.path.join(self.args.checkpoint_dir, f'model_weights_{iteration}.pt')
        else:
            # Find latest checkpoint
            pattern = os.path.join(self.args.checkpoint_dir, 'dmpo_checkpoint_*.json')
            checkpoints = glob.glob(pattern)
            if not checkpoints:
                print("⚠️  No checkpoints found to resume from")
                return

            # P0.12: filter to well-formed names BEFORE selecting max. Previously
            # non-matching files mapped to 0 via fallback and could be silently
            # picked when no valid checkpoint exists. Now we filter the glob
            # first; if nothing remains, we treat it as no checkpoints.
            def get_iter(path):
                import re
                match = re.search(r'dmpo_checkpoint_(\d+)\.json', path)
                return int(match.group(1)) if match else None

            valid_checkpoints = [(p, get_iter(p)) for p in checkpoints
                                 if get_iter(p) is not None]
            if not valid_checkpoints:
                print("⚠️  Checkpoint dir contains no well-formed dmpo_checkpoint_<N>.json files")
                return

            latest, iteration = max(valid_checkpoints, key=lambda pi: pi[1])
            checkpoint_path = latest
            model_path = os.path.join(self.args.checkpoint_dir, f'model_weights_{iteration}.pt')

        if not os.path.exists(checkpoint_path):
            # HARD ABORT — an explicit resume that cannot find its checkpoint
            # must never degrade into silent fresh training (a "resumed" arm
            # would quietly become a from-scratch run; caught 2026-08-12 when
            # a diagnostic resume pointed at a never-saved iteration).
            raise SystemExit(f"❌ RESUME REFUSED: checkpoint not found: {checkpoint_path} "
                             f"— refusing to fall back to fresh training.")

        print(f"📂 Loading checkpoint from iteration {iteration}...")

        # Load metrics
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)

        self.start_iteration = checkpoint_data.get('iteration', 0)
        # Merge loaded metrics with defaults (don't overwrite — preserve all default keys)
        loaded_metrics = checkpoint_data.get('metrics', {})
        for _k, _v in loaded_metrics.items():
            if _v:  # only override default empty list if loaded list has content
                self.training_metrics[_k] = _v

        print(f"  - Resuming from iteration {self.start_iteration}")
        print(f"  - Loaded {len(self.training_metrics.get('iteration_rewards', []))} reward records")

        # Load model weights if available
        if os.path.exists(model_path) and self.agent is not None:
            # ── OBS VERSION GATE (Step 12 / R6) — OUTSIDE the try below, so the
            # swallow-and-continue handler can never turn a shape-mismatched resume
            # into silent training from random weights (today's behavior). Defensive:
            # an UNREADABLE file skips the gate (the legacy handler owns corruption),
            # and a stubbed trainer without config skips the dim check — the gate
            # raises ONLY on a genuinely declared mismatch.
            _run_v = 2 if os.environ.get('DMPO_OBS_V2', '0') == '1' else 1
            try:
                _run_dim = int(self.config.agent_args.observation_dim)
            except Exception:
                _run_dim = None
            try:
                _ck_head = torch.load(model_path, map_location='cpu', weights_only=False)
            except Exception:
                _ck_head = None
            if isinstance(_ck_head, dict):
                _ck_v = int(_ck_head.get('obs_version', 1))
                _ck_dim = _ck_head.get('obs_dim')
                if (_ck_v != _run_v) or (_run_dim is not None and _ck_dim is not None
                                         and int(_ck_dim) != _run_dim):
                    raise RuntimeError(
                        f"OBS VERSION MISMATCH: checkpoint obs_version={_ck_v} obs_dim={_ck_dim} "
                        f"vs run obs_version={_run_v} obs_dim={_run_dim} — REFUSING to resume. "
                        f"(A mismatched resume used to be swallowed and silently trained from "
                        f"random weights.) Use a matching checkpoint, or start a fresh run.")
                # soundness review (train-plumbing HIGH): also refuse action/architecture
                # drift when the checkpoint declares it (keys absent in old ckpts = skip).
                for _key, _run_val in (
                        ('action_variant', os.environ.get('DMPO_ACTION_VARIANT', 'default')),
                        ('shared_policy', os.environ.get('DMPO_SHARED_POLICY', '0'))):
                    _ck_val = _ck_head.get(_key)
                    if _ck_val is not None and str(_ck_val) != str(_run_val):
                        raise RuntimeError(
                            f"ARCH MISMATCH: checkpoint {_key}={_ck_val!r} vs run "
                            f"{_run_val!r} — REFUSING to resume.")
            del _ck_head
            try:
                model_checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

                # ── Load shared attention policy/value (new format) ──
                if 'shared_policy_params' in model_checkpoint and hasattr(self.agent, '_policy_params'):
                    self.agent._policy_params.load_state_dict(model_checkpoint['shared_policy_params'])
                    print("  - Shared policy params loaded")

                if 'shared_log_std' in model_checkpoint and hasattr(self.agent, '_shared_log_std'):
                    with torch.no_grad():
                        self.agent._shared_log_std.copy_(
                            model_checkpoint['shared_log_std'].to(self.agent._shared_log_std.device)
                        )
                    print("  - Shared log_std loaded")

                if 'shared_value_params' in model_checkpoint and hasattr(self.agent, '_value_params'):
                    self.agent._value_params.load_state_dict(model_checkpoint['shared_value_params'])
                    print("  - Shared value params loaded")

                # ── Load ensemble world model ──
                if 'ensemble_models' in model_checkpoint and hasattr(self.agent, '_ensemble_models'):
                    for i, sd in enumerate(model_checkpoint['ensemble_models']):
                        if i < len(self.agent._ensemble_models):
                            self.agent._ensemble_models[i].load_state_dict(sd)
                    print(f"  - Ensemble models loaded ({len(model_checkpoint['ensemble_models'])} GCNs)")

                # ── Load ZFilter stats (RunningStat uses _M, _S, _n) ──
                if 'zfilter_M' in model_checkpoint and hasattr(self, 'running_state') and self.running_state is not None:
                    import numpy as np
                    self.running_state.rs._M = np.asarray(model_checkpoint['zfilter_M'])
                    self.running_state.rs._S = np.asarray(model_checkpoint['zfilter_S'])
                    self.running_state.rs._n = int(model_checkpoint.get('zfilter_n', 0))
                    print("  - ZFilter stats loaded")

                # ── Legacy fallback for old checkpoints ──
                # Save format writes under *_legacy keys (see _save_checkpoint
                # line ~4110); the old `policy_net` check below never matched
                # for current-format checkpoints. Check both keys.
                policy_key = (
                    'policy_net_legacy' if 'policy_net_legacy' in model_checkpoint
                    else ('policy_net' if 'policy_net' in model_checkpoint else None)
                )
                if policy_key and model_checkpoint[policy_key] and hasattr(self.agent, 'actors'):
                    try:
                        self.agent.actors.load_state_dict(model_checkpoint[policy_key])
                        print(f"  - actors loaded from {policy_key}")
                    except Exception as e:
                        print(f"  ⚠️ actors load failed from {policy_key}: {e}")

                value_key = (
                    'value_net_legacy' if 'value_net_legacy' in model_checkpoint
                    else ('value_net' if 'value_net' in model_checkpoint else None)
                )
                if value_key and model_checkpoint[value_key] and hasattr(self.agent, 'vs'):
                    try:
                        self.agent.vs.load_state_dict(model_checkpoint[value_key])
                        print(f"  - vs loaded from {value_key}")
                    except Exception as e:
                        print(f"  ⚠️ vs load failed from {value_key}: {e}")

                model_key = (
                    'model_net_legacy' if 'model_net_legacy' in model_checkpoint
                    else ('model_net' if 'model_net' in model_checkpoint else None)
                )
                if model_key and model_checkpoint[model_key] and hasattr(self.agent, 'ps'):
                    try:
                        self.agent.ps.load_state_dict(model_checkpoint[model_key])
                        print(f"  - ps loaded from {model_key}")
                    except Exception as e:
                        print(f"  ⚠️ ps load failed from {model_key}: {e}")

                # ── Dynamic-mode observation normalization stats (mirrors
                # the save block in _save_checkpoint that writes obs_norm_*) ──
                if getattr(self, '_obs_v2_nonorm', False):
                    # OBS-V2 no-norm: stats are identity by construction — never
                    # restore whatever is in the file (Step 12).
                    pass
                elif 'obs_norm_mean' in model_checkpoint:
                    import numpy as np
                    self._obs_running_mean = np.asarray(model_checkpoint['obs_norm_mean'])
                    self._obs_running_var = np.asarray(model_checkpoint['obs_norm_var'])
                    self._obs_count = float(model_checkpoint.get('obs_norm_count', 1.0))
                    print(f"  - obs_norm stats loaded (count={self._obs_count:.0f})")

                # ── Optimizer + RNG restore (load side of the states the save
                # path has always written; without this, Adam moments reset at
                # resume and the post-seam updates are transiently noisier) ──
                if 'optimizer_pi_state' in model_checkpoint and hasattr(self.agent, 'optimizer_pi'):
                    try:
                        self.agent.optimizer_pi.load_state_dict(model_checkpoint['optimizer_pi_state'])
                        print("  - optimizer_pi state restored")
                    except Exception as e:
                        print(f"  ⚠️ optimizer_pi restore failed: {e}")
                if 'optimizer_v_state' in model_checkpoint and hasattr(self.agent, 'optimizer_v'):
                    try:
                        self.agent.optimizer_v.load_state_dict(model_checkpoint['optimizer_v_state'])
                        print("  - optimizer_v state restored")
                    except Exception as e:
                        print(f"  ⚠️ optimizer_v restore failed: {e}")
                if 'torch_rng_state' in model_checkpoint:
                    try:
                        torch.set_rng_state(model_checkpoint['torch_rng_state'].cpu())
                        print("  - torch RNG state restored")
                    except Exception as e:
                        print(f"  ⚠️ torch RNG restore failed: {e}")
                if 'numpy_rng_state' in model_checkpoint:
                    try:
                        import numpy as np
                        np.random.set_state(model_checkpoint['numpy_rng_state'])
                        print("  - numpy RNG state restored")
                    except Exception as e:
                        print(f"  ⚠️ numpy RNG restore failed: {e}")

                print(f"✅ Model weights restored from {model_path}")
            except Exception as e:
                import traceback
                print(f"⚠️  Could not load model weights: {e}")
                traceback.print_exc()
                print("  - Continuing with fresh networks (metrics preserved)")
        else:
            print("  - No model weights found, starting with fresh networks")

        print(f"✅ Resume checkpoint loaded - will continue from iteration {self.start_iteration}")

    def _save_checkpoint(self, iteration):
        """Save training checkpoint with both metrics and model weights."""
        # 1. Save metrics (JSON) - existing functionality
        checkpoint_path = os.path.join(self.args.checkpoint_dir, f'dmpo_checkpoint_{iteration}.json')

        checkpoint_data = {
            'iteration': iteration,
            'metrics': self.training_metrics,
            'config': vars(self.args),
            # P0.4: persist κ so eval knows what κ-hop the policy was trained with.
            # Previously eval defaulted DMPO_KAPPA=2; if you trained with --kappa 3
            # and forgot to set DMPO_KAPPA=3, the eval policy operated on the wrong κ.
            'kappa': int(getattr(self.args, 'kappa', 2)),
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        # 2. Save model weights (PyTorch) - for post-training evaluation
        # CRITICAL FIX: Save the ACTUAL trained networks, not the unused shells.
        # The original code saved self.agent.actors / .vs / .ps which are the
        # per-agent networks created at init but NEVER trained — we replaced them
        # with shared attention policy and SharedGCN/ensemble model.
        if self.agent is not None:
            model_path = os.path.join(self.args.checkpoint_dir, f'model_weights_{iteration}.pt')

            try:
                model_checkpoint = {
                    'iteration': iteration,
                    'config': vars(self.args),
                    # P0.4: persist κ inside the .pt as well so eval scripts
                    # that only load model_weights_*.pt get it for free.
                    'kappa': int(getattr(self.args, 'kappa', 2)),
                }

                # ── Shared attention policy (the actually trained policy) ──
                if hasattr(self.agent, '_policy_params'):
                    model_checkpoint['shared_policy_params'] = self.agent._policy_params.state_dict()
                if hasattr(self.agent, '_shared_log_std'):
                    model_checkpoint['shared_log_std'] = self.agent._shared_log_std.detach().cpu()
                if hasattr(self.agent, '_value_params'):
                    model_checkpoint['shared_value_params'] = self.agent._value_params.state_dict()

                # ── World model: ensemble (P1+2 path) or SharedGCN (P5 path) ──
                if hasattr(self.agent, '_ensemble_models'):
                    model_checkpoint['ensemble_models'] = [
                        m.state_dict() for m in self.agent._ensemble_models
                    ]
                    model_checkpoint['ensemble_size'] = len(self.agent._ensemble_models)

                # ── ZFilter normalization stats (RunningStat uses _M, _S, _n) ──
                if hasattr(self, 'running_state') and self.running_state is not None:
                    rs = self.running_state.rs
                    model_checkpoint['zfilter_M'] = rs._M.copy()
                    model_checkpoint['zfilter_S'] = rs._S.copy()
                    model_checkpoint['zfilter_n'] = int(rs._n)

                # ── OBS VERSION METADATA (Step 11) — every checkpoint declares its obs
                # contract so the loader gate (R6) and every eval engine can refuse a
                # mismatched pairing instead of silently truncating/padding.
                # (getattr-guarded: unit-test stubs save without a config object.)
                import hashlib as _hashlib
                _obs_v2_on = os.environ.get('DMPO_OBS_V2', '0') == '1'
                try:
                    _obs_dim_ck = int(self.config.agent_args.observation_dim)
                except Exception:
                    _obs_dim_ck = None
                if _obs_dim_ck is not None:
                    _flat_order = (ObservationProcessor.FLAT_ORDER_V2 if _obs_v2_on
                                   else ObservationProcessor.FLAT_ORDER_V1)
                    model_checkpoint['obs_version'] = 2 if _obs_v2_on else 1
                    model_checkpoint['obs_dim'] = _obs_dim_ck
                    model_checkpoint['obs_max_neighbors'] = int(os.environ.get('DMPO_MAX_NEIGHBORS', '10'))
                    model_checkpoint['obs_block_hash'] = _hashlib.sha256(
                        repr((_flat_order, _obs_dim_ck)).encode()).hexdigest()[:16]
                    # soundness review (train-plumbing HIGH): the obs gate alone leaves
                    # action/architecture drift silently loadable — declare those too.
                    model_checkpoint['action_variant'] = os.environ.get('DMPO_ACTION_VARIANT', 'default')
                    try:
                        model_checkpoint['action_dim'] = int(self.config.agent_args.action_space.shape[0])
                    except Exception:
                        pass
                    model_checkpoint['shared_policy'] = os.environ.get('DMPO_SHARED_POLICY', '0')
                    model_checkpoint['set_actor'] = os.environ.get('DMPO_SET_ACTOR', 'off').strip().lower() or 'off'
                    model_checkpoint['central_critic'] = os.environ.get('DMPO_CENTRAL_CRITIC', '0')
                    model_checkpoint['pad_shared'] = os.environ.get('DMPO_PAD_SHARED', '0')

                # ── REPRODUCIBILITY SNAPSHOT (audit gap: the paper's repro section was
                # unwritable from artifacts — no env vars, no argv, no RNG/optimizer
                # state in any checkpoint) ──
                try:
                    _dmpo_env = {k: v for k, v in sorted(os.environ.items())
                                 if k.startswith('DMPO_')}
                    model_checkpoint['env_snapshot'] = _dmpo_env
                    model_checkpoint['env_snapshot_hash'] = _hashlib.sha256(
                        repr(_dmpo_env).encode()).hexdigest()[:16]
                    model_checkpoint['argv'] = list(sys.argv)
                    model_checkpoint['torch_rng_state'] = torch.get_rng_state()
                    import numpy as _np_snap   # later branch-local `import numpy as np`
                    # statements make bare `np` function-local here (UnboundLocalError
                    # — found by the verification sweep; snapshot silently skipped 108x)
                    model_checkpoint['numpy_rng_state'] = _np_snap.random.get_state()
                    if hasattr(self.agent, 'optimizer_pi'):
                        model_checkpoint['optimizer_pi_state'] = self.agent.optimizer_pi.state_dict()
                    if hasattr(self.agent, 'optimizer_v'):
                        model_checkpoint['optimizer_v_state'] = self.agent.optimizer_v.state_dict()
                except Exception as _snap_e:
                    print(f"⚠️ repro-snapshot skipped: {_snap_e}")

                # ── Dynamic normalization stats (_normalize_obs_dynamic) ──
                # In dynamic mode, ZFilter is not used. Instead we maintain
                # running mean/var in _obs_running_mean/_obs_running_var.
                # These MUST be saved for evaluation to produce correct inputs.
                if getattr(self, '_obs_v2_nonorm', False) and _obs_dim_ck is not None:
                    # OBS-V2 no-norm: save IDENTITY stats (NOT omission!). Every eval
                    # script replays (x-mean)/sqrt(var) then clips ±5 — with mean=0,
                    # var=1 that is an exact no-op for features bounded at ±2, so even
                    # UNPATCHED eval scripts normalize correctly by construction.
                    # (Omission would take their `obs_mean is None` branch — the H13
                    # raw-obs hazard path.) VERIFIER CORRECTION: conditional on the
                    # no-norm flag, NOT on DMPO_OBS_V2 alone — the DMPO_OBS_ZFILTER=1
                    # ablation keeps REAL running stats and must save them below.
                    import numpy as np
                    model_checkpoint['obs_norm_mean'] = np.zeros(_obs_dim_ck)
                    model_checkpoint['obs_norm_var'] = np.ones(_obs_dim_ck)
                    model_checkpoint['obs_norm_count'] = 0.0
                elif hasattr(self, '_obs_running_mean'):
                    model_checkpoint['obs_norm_mean'] = self._obs_running_mean.copy()
                    model_checkpoint['obs_norm_var'] = self._obs_running_var.copy()
                    model_checkpoint['obs_norm_count'] = float(self._obs_count)

                # ── Adjacency matrix (frozen at init for current architecture) ──
                if hasattr(self.agent, 'adj'):
                    import numpy as np
                    adj = self.agent.adj
                    if hasattr(adj, 'cpu'):
                        adj = adj.cpu().numpy()
                    elif not isinstance(adj, np.ndarray):
                        adj = np.array(adj)
                    model_checkpoint['adjacency'] = adj

                # ── SAC entropy tuning state ──
                if hasattr(self.agent, '_log_alpha'):
                    model_checkpoint['log_alpha'] = self.agent._log_alpha.detach().cpu()
                if hasattr(self.agent, '_target_entropy'):
                    model_checkpoint['target_entropy'] = float(self.agent._target_entropy)

                # ── Architecture metadata for reconstruction ──
                model_checkpoint['arch'] = {
                    'n_agent': getattr(self.agent, 'n_agent', None),
                    'observation_dim': getattr(self.agent, 'observation_dim', None),
                    'action_dim': getattr(self.agent, 'action_dim', None),
                    'embed_dim': getattr(self.agent, '_embed_dim', None),
                    'shared_gcn': os.environ.get('DMPO_SHARED_GCN', '0') == '1',
                    'phase2': os.environ.get('DMPO_PHASE2', '0') == '1',
                    'dynamic': getattr(self.agent, 'dynamic', False),
                    'speed_mode': int(os.environ.get('DMPO_SPEED_MODE', '6')),
                }
                # Padded-eval reconstruction: per-agent actor shapes depend on the
                # spatial adjacency; save it so evaluation can rebuild identically.
                try:
                    model_checkpoint['adjacency'] = np.asarray(self.config.agent_args.adj)
                except Exception:
                    pass

                # ── Legacy fallback (kept for backwards compat with old eval scripts) ──
                if hasattr(self.agent, 'actors'):
                    model_checkpoint['policy_net_legacy'] = self.agent.actors.state_dict()
                if hasattr(self.agent, 'vs'):
                    model_checkpoint['value_net_legacy'] = self.agent.vs.state_dict()
                if hasattr(self.agent, 'ps'):
                    model_checkpoint['model_net_legacy'] = self.agent.ps.state_dict()

                torch.save(model_checkpoint, model_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
                print(f"💾 Model weights saved: {model_path}")
                print(f"   - shared_policy_params: {'shared_policy_params' in model_checkpoint}")
                print(f"   - ensemble_models: {len(model_checkpoint.get('ensemble_models', []))}")
                print(f"   - zfilter: {'zfilter_M' in model_checkpoint}")
            except Exception as e:
                import traceback
                print(f"⚠️  Could not save model weights: {e}")
                traceback.print_exc()
                print(f"💾 Checkpoint saved: {checkpoint_path}")
        else:
            print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def _save_final_results(self, start_time):
        """Save final training results."""
        total_time = time.time() - start_time
        
        final_results = {
            'training_completed': True,
            'total_iterations': self.args.n_iterations,
            'total_time_hours': total_time / 3600,
            'final_metrics': self.training_metrics,
            'config': vars(self.args)
        }
        
        results_path = os.path.join(self.args.log_dir, 'dmpo_final_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"💾 Final results saved: {results_path}")


def main():
    """Main training function."""
    args = parse_args()

    # ── STAGE-0 PROCESS ARMOR: canonical-block gate (DMPO_REQUIRE_BLOCK=1) ──
    # Refuses to train on any drift from configs/consistency_block.sh — the C7 fix.
    if os.environ.get('DMPO_REQUIRE_BLOCK', '0') == '1':
        import subprocess as _sp
        _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _cmd = [sys.executable, os.path.join(_repo, 'scripts', 'assert_consistency.py')]
        if args.config_path:
            _cmd += ['--scenario', args.config_path]
        if _sp.run(_cmd).returncode != 0:
            sys.exit('REFUSED by assert_consistency (see violations above)')

    # ── UMBRELLA COUPLING (soundness review, goal-blockers HIGH): a TRAINING run on
    # the obs-v2 rig with the reward/learner umbrellas forgotten would train the
    # fixed obs against the LEGACY reward + LEGACY learner defects silently. The
    # three umbrellas are one Stage-4 design — refuse the partial combination
    # unless explicitly overridden for a deliberate ablation arm.
    if (os.environ.get('DMPO_OBS_V2', '0') == '1'
            and os.environ.get('DMPO_UMBRELLA_SPLIT_OK', '0') != '1'):
        _missing = [u for u in ('DMPO_REWARD_V2', 'DMPO_LEARNER_V2')
                    if os.environ.get(u, '0') != '1']
        if _missing:
            sys.exit(f'REFUSED: DMPO_OBS_V2=1 training without {_missing}. The Stage-3 '
                     f'umbrellas ship together (obs + reward + learner). For a deliberate '
                     f'ablation arm export DMPO_UMBRELLA_SPLIT_OK=1.')

    print("🚗🤖 Full DMPO Neural Network Training")
    print("="*70)
    print(f"Configuration: {args.config_path or 'default'}")
    print(f"Device: {args.device}")
    print(f"Iterations: {args.n_iterations}")
    print(f"Episodes per iteration: {args.n_episodes}")
    print(f"κ-hop radius: {args.kappa}")
    print(f"GUI enabled: {args.use_gui}")
    print(f"External evaluation: {args.enable_external_eval}")
    print("="*70)
    
    # Setup directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load DMPO modules
    dmpo_modules = load_networked_mb_marl_modules()
    
    # Initialize environment
    print("🚗 Initializing intersection environment...")
    env = create_intersection_env(
        config_path=args.config_path,
        max_steps=args.max_steps,
        use_gui=args.use_gui,
        enable_external_eval=args.enable_external_eval,
        communication_radius=args.communication_radius,
        max_ai_vehicles=args.max_ai_vehicles,
        training_mode=True,  # PHASE 3 FIX: explicit training mode (SafetyMonitor pass-through)
        kappa=args.kappa,    # P3.6: forward CLI κ to env so obs/CARL+ team_bonus respect it
    )
    
    # Initialize trainer
    trainer = IntersectionDMPOTrainer(args, env, dmpo_modules)

    # Print TraCI port info for external evaluation
    if args.enable_external_eval:
        traci_port = env.get_traci_port()
        if traci_port:
            print(f"🔗 External evaluation available on TraCI port: {traci_port}")
            print(f"📊 Run evaluation with: python scripts/evaluate_dmpo_performance.py --sumo-port {traci_port}")
        else:
            print("⚠️ External evaluation not available (environment not yet reset)")
    
    try:
        # Run training
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        
    finally:
        print("🔄 Cleaning up...")
        env.close()
        print("✅ Environment closed")


if __name__ == "__main__":
    main()