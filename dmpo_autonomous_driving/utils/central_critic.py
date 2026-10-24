"""MAPPO-style centralized critic for the padding-free dynamic mode — the MAPPO
learning baseline. Gated DMPO_CENTRAL_CRITIC=1,
default 0 = no-op.

MAPPO (Yu et al., NeurIPS 2022) = parameter-shared PPO whose value function sees a
global state during training (CTDE); execution uses the decentralized actor only.
With a variable vehicle set there is no fixed global vector, so the critic gets
Yu et al.'s *agent-specific* global state with a permutation-invariant global part:

    V(o_i, g_t),   g_t = [ mean_j o_j , max_j o_j , n_t / DMPO_CC_COUNT_NORM ]

over ALL vehicles present at step t (centralized information the decentralized
critic never sees). Width G = 2·obs_dim + 1.

The actor is untouched in what it sees: trajectories carry the extended state
[o_i | g_t] so the critic can read it, and every actor reads only the first obs_dim
columns (slice patched onto actor.network.forward — no new parameters, checkpoint
keys unchanged). Trajectories stored to the world-model buffer are the un-extended
o_i. Recorded in the checkpoint ('central_critic'); the critic is not used at eval.
"""
import os

import numpy as np
import torch
import torch.nn as nn


def enabled() -> bool:
    return os.environ.get('DMPO_CENTRAL_CRITIC', '0') == '1'


def g_dim(obs_dim: int) -> int:
    return 2 * obs_dim + 1


def global_feat(flat: np.ndarray) -> np.ndarray:
    """flat: [n, obs_dim] — every vehicle present at one step. → [2·obs_dim + 1]."""
    norm = float(os.environ.get('DMPO_CC_COUNT_NORM', '40'))
    flat = np.asarray(flat, dtype=np.float32)
    if flat.shape[0] == 0:
        return np.zeros(2 * flat.shape[1] + 1, dtype=np.float32)
    return np.concatenate([flat.mean(0), flat.max(0),
                           np.array([flat.shape[0] / norm], dtype=np.float32)])


def _slice_forward(net: nn.Module, d: int):
    inner = net.forward

    def fwd(x, _inner=inner, _d=d):
        return _inner(x[..., :_d])
    net.forward = fwd


def apply(agent, obs_dim: int, device) -> int:
    """Rebuild the shared critic on obs_dim + G inputs (same hidden sizes/activation
    as the configured v_args) + rebuild optimizer_v; slice every actor's input.
    Returns critic param count (0 = no-op)."""
    if not enabled():
        return 0
    if os.environ.get('DMPO_DYNAMIC', '0') != '1':
        raise RuntimeError("DMPO_CENTRAL_CRITIC requires DMPO_DYNAMIC=1")
    if os.environ.get('DMPO_SHARED_POLICY', '0') == '1':
        raise RuntimeError("DMPO_CENTRAL_CRITIC is incompatible with DMPO_SHARED_POLICY=1")
    from algorithms.models import MLP
    va = agent.v_args
    sizes = [obs_dim + g_dim(obs_dim)] + list(va.sizes[1:])
    agent.vs = nn.ModuleList([MLP(sizes=sizes, activation=va.activation)
                              for _ in range(len(agent.vs))]).to(device)
    agent.optimizer_v = torch.optim.Adam(agent.vs.parameters(), lr=agent.lr_v)
    for actor in agent.actors:
        _slice_forward(actor.network, obs_dim)
    return sum(p.numel() for p in agent.vs[0].parameters())


def extend(s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """s: [T, 1, obs_dim], g: [T, G] → [T, 1, obs_dim + G]."""
    return torch.cat([s, g.unsqueeze(1).to(s)], dim=-1)
