"""
DMPO Configuration for 4-Lane Intersection Autonomous Driving

Adapts Networked-MB-MARL DMPO configuration for intersection driving scenarios.
Based on Catchup_DMPO.py from the original framework.
"""

from gym.spaces import Box, Dict
import torch.nn
import numpy as np
import math
import sys
import os


# =========================
# ENTROPY SCHEDULING FOR EXPLORATION VS EXPLOITATION
# =========================

def get_entropy_coefficient(iteration: int, total_iterations: int,
                           schedule: str = "linear_warmup") -> float:
    """
    Get entropy coefficient based on training progress.

    Implements adaptive exploration-exploitation balance:
    - Early training: High entropy for broad exploration
    - Mid training: Gradual decay to balance exploration/exploitation
    - Late training: Low entropy for fine-tuning exploitation

    Args:
        iteration: Current training iteration
        total_iterations: Total number of training iterations
        schedule: Scheduling strategy - "linear_warmup", "cosine", "exponential", or "constant"

    Returns:
        Entropy coefficient for current iteration

    References:
        - axPPO (arXiv:2405.04664): Adaptive exploration based on performance
        - PPO Implementation Details (costa.sh): Standard entropy values
        - Entropy Scheduling in RL (OpenReview): Schedule comparison
    """
    START_COEF = 0.02    # Moderate exploration
    END_COEF = 0.001     # Low exploitation at end
    WARMUP_FRAC = 0.05   # P0.14: 5% warmup so the 'linear_warmup' name matches
                         # behavior. Previously 0.0 made it linear_decay in disguise.
    ANNEAL_FRAC = 0.80   # 80% annealing phase

    if schedule == "constant":
        # P0.13: return START_COEF so direct callers get the start value
        # (was END_COEF — silently overrode any caller's intent). The
        # EntropyScheduler wrapper short-circuits 'constant' separately,
        # so this only affects direct callers of the helper.
        return START_COEF

    elif schedule == "linear_warmup":
        # Phase 1 (0-WARMUP_FRAC): hold at START_COEF
        # Phase 2 (WARMUP..WARMUP+ANNEAL): linear decay from START_COEF to END_COEF
        # Phase 3 (else): exploitation at END_COEF
        progress = iteration / total_iterations

        if progress < WARMUP_FRAC:
            return START_COEF
        elif progress < (WARMUP_FRAC + ANNEAL_FRAC):
            anneal_progress = (progress - WARMUP_FRAC) / ANNEAL_FRAC
            return START_COEF * (1 - anneal_progress) + END_COEF * anneal_progress
        else:
            return END_COEF

    elif schedule == "cosine":
        # Smooth cosine annealing (avoids sudden transitions)
        progress = iteration / total_iterations
        return END_COEF + 0.5 * (START_COEF - END_COEF) * (1 + math.cos(math.pi * progress))

    elif schedule == "exponential":
        # Exponential decay (faster initial drop, slower later)
        progress = iteration / total_iterations
        decay_rate = 3.0  # Controls decay speed
        return END_COEF + (START_COEF - END_COEF) * math.exp(-decay_rate * progress)

    else:
        # Fallback to constant
        return 0.02


class EntropyScheduler:
    """
    Entropy coefficient scheduler for DMPO training.

    Tracks training progress and provides appropriate entropy coefficients
    for exploration-exploitation balance throughout training.
    """

    def __init__(self, total_iterations: int, schedule: str = "linear_warmup",
                 start_coef: float = 0.05, end_coef: float = 0.005):
        """
        Initialize entropy scheduler.

        Args:
            total_iterations: Total training iterations
            schedule: Scheduling strategy
            start_coef: Initial entropy coefficient (high exploration)
            end_coef: Final entropy coefficient (exploitation)
        """
        self.total_iterations = total_iterations
        self.schedule = schedule
        self.start_coef = start_coef
        self.end_coef = end_coef
        self.current_iteration = 0
        self.history = []

    def get_coefficient(self, iteration: int = None) -> float:
        """Get entropy coefficient for given or current iteration."""
        if iteration is None:
            iteration = self.current_iteration

        if self.start_coef == 0.0 and self.end_coef == 0.0:
            coef = 0.0
        elif self.schedule == 'constant':
            # Constant schedule: use the start_coef directly, don't call
            # get_entropy_coefficient which has hardcoded END_COEF=0.001
            coef = self.start_coef
        else:
            coef = get_entropy_coefficient(
                iteration, self.total_iterations, self.schedule
            )
        self.history.append((iteration, coef))
        return coef

    def step(self) -> float:
        """Advance iteration and return new entropy coefficient."""
        self.current_iteration += 1
        return self.get_coefficient()

    def get_phase(self, iteration: int = None) -> str:
        """Get current training phase description."""
        if iteration is None:
            iteration = self.current_iteration
        progress = iteration / self.total_iterations

        if progress < 0.10:
            return "warmup (exploration)"
        elif progress < 0.80:
            return "annealing (balanced)"
        else:
            return "final (exploitation)"

# Ensure Networked-MB-MARL is on sys.path before importing
_networked_path = os.environ.get('DMPO_FRAMEWORK_PATH', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'third_party', 'Networked-MB-MARL'))
if os.path.isdir(_networked_path) and _networked_path not in sys.path:
    sys.path.insert(0, _networked_path)

# Import with fallback handling
try:
    from algorithms.models import MLP
    from algorithms.utils import Config
except ImportError:
    # Create minimal fallback classes if DMPO modules not available
    import torch.nn as nn

    class MLP(nn.Module):
        """Minimal MLP implementation for standalone mode."""
        def __init__(self, input_dim, output_dim, hidden_dims=[64, 64]):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    class Config:
        """Minimal config class for standalone mode."""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def _toDict(self, recursive=False):
            pr = {}
            for name in vars(self):
                value = getattr(self, name)
                if not name.startswith('_') and not name.endswith('_'):
                    if isinstance(value, Config) and recursive:
                        value = value._toDict(recursive)
                    pr[name] = value
            return pr


def getArgs(radius_p, radius_v, radius_pi, env):
    """
    Get DMPO arguments for intersection autonomous driving.
    
    Args:
        radius_p: Environment model communication radius (κ-hop for model learning)
        radius_v: Value function communication radius (κ-hop for value estimation)  
        radius_pi: Policy communication radius (κ-hop for policy optimization)
        env: Intersection environment instance
        
    Returns:
        Algorithm configuration for DMPO intersection training
    """
    
    # =========================
    # ALGORITHM CONFIGURATION
    # =========================
    alg_args = Config()
    
    # Training iterations (based on DMPO original - 25k iterations)
    alg_args.n_iter = 25000          # Total training iterations (increased from 20k)
    alg_args.n_inner_iter = 10       # PPO inner iterations (match DMPO original)
    alg_args.n_warmup = 50           # Warmup iterations (match DMPO original)

    # Model-based learning parameters (κ-hop environment modeling)
    alg_args.n_model_update = int(2000)        # Model updates per iteration (match DMPO original)
    alg_args.n_model_update_warmup = int(1500) # Model updates during warmup (match DMPO original)
    alg_args.model_validate_interval = 5       # Model validation frequency
    
    # Testing and evaluation
    alg_args.n_test = 10            # Number of test episodes
    alg_args.test_interval = 50     # Test every N iterations
    # FIX Issue 35 (Alternative D): Restore 400 steps for proper credit assignment
    # Evidence: 400 steps achieved +97.8 avg reward vs -315.7 with 200 steps
    # Root cause: 200 steps insufficient for vehicles to complete intersection crossing
    # Speed optimizations applied elsewhere (n_update_pi, n_update_v, model_traj_length)
    # Literature: PPO rollout length significantly affects performance (ICLR 2022)
    alg_args.rollout_length = 400   # Episode length (80 seconds at 0.2s steps)
    alg_args.test_length = 400      # Test episode length
    alg_args.max_episode_len = 400  # Maximum episode length
    
    # Model-based specific settings
    alg_args.model_based = True                    # Enable model-based learning
    alg_args.load_pretrained_model = False        # No pretrained models for driving
    alg_args.pretrained_model = None
    
    # Experience collection and model training
    alg_args.n_traj = 2048              # Trajectory batch size (match DMPO original)
    # SPEED OPTIMIZATION (Alternative D): Reduced model rollout from 25 to 15
    # Shorter model rollouts = faster model updates with acceptable accuracy
    alg_args.model_traj_length = 15     # Model rollout horizon (was 25)
    alg_args.model_error_thres = 1e-5   # Model validation threshold (original: 1e-5)
    # Model-based policy optimization probability
    # With this probability, policy updates use synthetic model rollouts instead of real experience
    # This is the key DMPO feature - enables sample-efficient learning by training on imagined data
    # Set to 0.5 for balanced mix of real and model-based experience (as in original DMPO)
    alg_args.model_prob = 0.5           # 50% real experience, 50% model-generated
    alg_args.model_batch_size = 256     # Model training batch size
    alg_args.model_buffer_size = 20     # Model experience buffer size
    alg_args.model_update_length = 4    # Model update sequence length
    alg_args.model_length_schedule = None # No length scheduling
    
    # =========================
    # AGENT CONFIGURATION
    # =========================
    agent_args = Config()
    
    # Communication topology (κ-hop neighborhoods)
    agent_args.adj = env.get_adjacency_matrix() if hasattr(env, 'get_adjacency_matrix') else np.eye(8)
    agent_args.n_agent = agent_args.adj.shape[0] if agent_args.adj.size > 0 else 8
    
    # Handle dynamic agent count for intersection
    if agent_args.n_agent == 0:
        agent_args.n_agent = 8  # Default for initialization
        agent_args.adj = np.eye(8)
    
    # PPO hyperparameters (matching original DMPO framework)
    agent_args.gamma = 0.99             # Discount factor
    # GAE lambda (original: 0.5 across all domains). ARM-E1 (credit-routing
    # ablation, 2026-07-27): env-gated override — at lambda=0.5 a terminal
    # collision surprise reaches the decision 11 steps earlier at (gamma*lambda)^11
    # ~ 5e-4 of its size unless the critic anticipates it (probe-verified blind);
    # the old "lambda=0.95 was unstable" lesson predates the repaired stack.
    # Default preserves 0.5 exactly.
    agent_args.lamda = float(os.environ.get('DMPO_GAE_LAMBDA', '0.5'))
    agent_args.clip = 0.2               # PPO clip parameter
    agent_args.target_kl = 7.5e-3       # Target KL divergence
    agent_args.v_coeff = 1.0            # Value function coefficient
    agent_args.v_thres = 0.             # Value function threshold
    agent_args.entropy_coeff = 0.0      # Original uses no entropy bonus

    # Learning rates (original: equal LR for all networks)
    agent_args.lr = 1e-4
    agent_args.lr_v = 1e-4
    agent_args.lr_p = 1e-4

    # Training parameters (matching original)
    agent_args.n_update_v = 10      # Value function updates per iteration
    # n_update_pi: PPO gradient epochs per update. Default 1 (prevents mean
    # convergence on stale on-policy data); the parent Grid_DMPO used 10. Now
    # env-tunable (DMPO_N_UPDATE_PI) so we can lift it without a code edit — the
    # cheapest lever against under-training (more gradient updates, no extra data).
    agent_args.n_update_pi = int(os.environ.get('DMPO_N_UPDATE_PI', '1'))
    agent_args.n_minibatch = 1      # Full-batch updates (original: 1)

    # DMPO specific settings (matching original)
    agent_args.use_reduced_v = True         # Use reduced value function
    agent_args.use_rtg = False              # Don't use reward-to-go
    agent_args.use_gae_returns = False       # Monte Carlo returns (original: False)
    agent_args.advantage_norm = True         # Normalize advantages
    
    # Observation and action spaces (from intersection environment)
    agent_args.observation_space = env.observation_space

    # DMPO expects Box action space, but our env has Dict action space
    # Convert Dict action space to flattened Box for DMPO compatibility
    if hasattr(env.action_space, 'spaces'):  # Dict action space
        from gym.spaces import Box
        total_action_dim = sum(
            space.shape[0] if hasattr(space, 'shape') else 1
            for space in env.action_space.spaces.values()
        )
        # Path D/E (accountable/CaRL): 2-dim actions (longitudinal + lateral only)
        import os as _os_act
        if _os_act.environ.get('DMPO_REWARD_ACCOUNTABLE', '0') == '1' or _os_act.environ.get('DMPO_REWARD_CARL', '0') == '1':
            total_action_dim = 2
        # Action-space variants for the next experiment phase:
        #   throttle_brake: (throttle, brake, lateral, lane_intent) — separate accel/decel modes
        #   yield_intent:   (longitudinal, lateral, lane_intent, yield_intent) — explicit yield primitive
        _action_variant = _os_act.environ.get('DMPO_ACTION_VARIANT', 'default')
        if _action_variant in ('throttle_brake', 'yield_intent'):
            total_action_dim = 4
        # Create equivalent Box space: -1 to 1 for all actions
        agent_args.action_space = Box(low=-1.0, high=1.0, shape=(total_action_dim,), dtype=np.float32)
    else:
        agent_args.action_space = env.action_space
    
    # State representation for intersection driving.
    # observation_dim is COMPUTED from DMPO_MAX_NEIGHBORS (same default 10 as the
    # ObservationProcessor) so the two can NEVER drift and cause silent truncation:
    #   10 ego + 5 infra + (max_neighbors × 10 feats) + 10 context + 4 identity + 1 collision_warning
    # max_neighbors=10 -> 130 ; max_neighbors=15 -> 180.
    agent_args.hidden_state_dim = 64       # Higher dim for driving complexity
    # OBS-V2 / legacy: the arithmetic lives in ONE place — ObservationProcessor.obs_dim()
    # (R8: config must CALL it, never restate the formula). Legacy: 30 + 10N (+14 under
    # DMPO_PATH_CONFLICT_OBS). v2 (DMPO_OBS_V2=1): 61 + 12N (N=15 -> 241).
    try:
        from dmpo_autonomous_driving.envs.observation_processor import ObservationProcessor as _OP
        agent_args.observation_dim = _OP.obs_dim()
    except ImportError:                      # standalone/config-only contexts
        _max_nb = int(os.environ.get('DMPO_MAX_NEIGHBORS', '10'))
        if os.environ.get('DMPO_OBS_V2', '0') == '1':
            agent_args.observation_dim = 61 + _max_nb * 12
            if os.environ.get('DMPO_PROFILE_OBS', '0') == '1':
                agent_args.observation_dim += 20   # ARM-B appendix (4 ego + 4×4 slots)
        else:
            agent_args.observation_dim = 30 + _max_nb * 10
            if os.environ.get('DMPO_PATH_CONFLICT_OBS', '0') == '1':
                agent_args.observation_dim += 14

    # Embedding network for heterogeneous observations
    agent_args.embedding_sizes = [
        agent_args.observation_dim,  # Input dimension (130)
        128,                         # Hidden layer
        agent_args.hidden_state_dim  # Output embedding (64)
    ]
    
    # Communication radii (κ-hop parameters)
    agent_args.radius_v = radius_v      # Value function κ-hop radius
    agent_args.radius_pi = radius_pi    # Policy κ-hop radius  
    agent_args.radius_p = radius_p      # Model κ-hop radius
    agent_args.squeeze = True           # Squeeze singleton dimensions
    
    # =========================
    # ENVIRONMENT MODEL CONFIGURATION (Graph Convolutional Networks)
    # =========================
    p_args = Config()
    
    # GCN architecture (matching original DMPO scale)
    # P3.4: n_conv = radius_p (= CLI κ) so the world model's effective
    # receptive field equals the configured κ. Previously hardcoded to 1,
    # giving a 1-hop world model regardless of κ-hop framework claims.
    p_args.n_conv = max(1, int(radius_p))
    p_args.n_embedding = 0          # 0 for continuous actions (uses Identity, not Embedding lookup)
    p_args.residual = True          # Residual connections
    p_args.edge_embed_dim = 12      # Edge embedding dimension (original: 12)
    p_args.node_embed_dim = 16      # Node embedding dimension (original: 16)

    # Network sizes (matching original Monaco/Pandemic scale)
    p_args.edge_hidden_size = [32, 32]   # Edge network layers
    p_args.node_hidden_size = [32, 32]   # Node network layers
    p_args.reward_coeff = 10.0           # Reward prediction weight (original: 10.0 everywhere)
    
    agent_args.p_args = p_args
    
    # =========================
    # VALUE FUNCTION CONFIGURATION
    # =========================
    v_args = Config()
    v_args.network = MLP
    v_args.activation = torch.nn.ReLU
    v_args.sizes = [-1, 128, 128, 64, 1]  # Value network architecture
    agent_args.v_args = v_args
    
    # =========================
    # POLICY NETWORK CONFIGURATION  
    # =========================
    pi_args = Config()
    pi_args.network = MLP
    pi_args.activation = torch.nn.ReLU
    
    # Handle different action space types
    if hasattr(env.action_space, 'spaces'):  # Dict action space
        # Calculate total action dimension
        total_action_dim = sum(
            space.shape[0] if hasattr(space, 'shape') else 1
            for space in env.action_space.spaces.values()
        )
    else:  # Box action space
        total_action_dim = env.action_space.shape[0]

    # Path D/E (accountable/CaRL): 2-dim actions (longitudinal + lateral only, no lane_intention)
    import os as _os_cfg
    if _os_cfg.environ.get('DMPO_REWARD_ACCOUNTABLE', '0') == '1' or _os_cfg.environ.get('DMPO_REWARD_CARL', '0') == '1':
        total_action_dim = 2
    # Action-space variants (mirrors block above for action_space definition).
    if _os_cfg.environ.get('DMPO_ACTION_VARIANT', 'default') in ('throttle_brake', 'yield_intent'):
        total_action_dim = 4

    pi_args.sizes = [-1, 128, 128, 64, total_action_dim]  # Policy network architecture
    pi_args.squash = False  # No action squashing (handled by environment)
    agent_args.pi_args = pi_args

    # Add agent args to algorithm args
    alg_args.agent_args = agent_args

    # Add required fields for DMPO agent initialization
    alg_args.env = "intersection"  # Environment name
    alg_args.algo = "DMPO"  # Algorithm name

    return alg_args


def get_intersection_dmpo_config(env, kappa=2):
    """
    Get DMPO configuration for intersection autonomous driving.
    
    Args:
        env: Intersection environment instance
        kappa: κ-hop communication radius
        
    Returns:
        Complete DMPO configuration
    """
    
    # Set κ-hop communication radii
    # FIX: Use same radius for policy and value to ensure actor-critic consistency
    # Previously: radius_pi=1, radius_v=kappa caused mismatch where critic saw
    # more neighbors than actor, leading to biased gradient estimates
    # Now: Both actor and critic see the same κ-hop neighborhood
    radius_v = kappa     # Value function: κ-hop for intersection coordination
    radius_pi = kappa    # Policy: κ-hop (FIXED: was 1, now matches value)
    radius_p = kappa     # Environment model: κ-hop for traffic prediction

    return getArgs(radius_p, radius_v, radius_pi, env)


def create_intersection_dmpo_config(config_path=None):
    """
    Create DMPO configuration for intersection training.
    
    Args:
        config_path: Optional SUMO configuration path
        
    Returns:
        DMPO algorithm configuration
    """
    # Import here to avoid circular dependencies
    from dmpo_autonomous_driving.envs.sumo_intersection_env import create_intersection_env
    
    # Create environment to get spaces and adjacency matrix
    if config_path is None:
        # Use relative path to sumo_configs in the package
        import os
        package_dir = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(package_dir, "sumo_configs", "dmpo_intersection.sumocfg")
    
    # Create temporary environment for configurationdmpo_modules
    temp_env = create_intersection_env(config_path)
    
    try:
        # Get configuration with κ=2 for intersection scenarios
        config = get_intersection_dmpo_config(temp_env, kappa=2)
        return config
    finally:
        # Clean up temporary environment
        temp_env.close()


# Default configuration for intersection DMPO
def get_default_intersection_config():
    """Get default intersection DMPO configuration."""
    return create_intersection_dmpo_config()