"""Set-structured policy trunks — the learning-based baselines.
Gated DMPO_SET_ACTOR, default 'off'.

Replaces ONLY the GaussianActor's feature trunk (`actor.network`); the action head,
log_std, output-tanh, policy bias, critic, κ routing, reward and training loop are
untouched — so the comparison against the plain-MLP actor is matched in observation,
reward, critic, budget and evaluation by construction.

  attn : ego-attention over the perceived-neighbour set (Leurent & Mercat 2019,
         "Social Attention for Autonomous Decision-Making in Dense Traffic"):
         the ego embedding queries {ego} ∪ {valid neighbours}; 2 heads.
  mean : Deep Sets (Zaheer et al. 2017): ρ(ego, mean_j φ(neighbour_j)).

Both are permutation-invariant over neighbours and indifferent to their number;
the MLP reads the same 15 slots in fixed nearest-first order.

Empty neighbour slots are the v2 NEUTRAL row (all 0 except feat 7 = feat 9 = 1,
observation_processor._zero_obs_neutral) and are masked out. This requires the raw
v2 vector reach the actor (OBS-V2 = no normalisation) — enforced below.

The architecture is recorded in the checkpoint ('set_actor'); eval engines take it
from there (apply_from_checkpoint), so train and eval cannot disagree.
"""
import os

import torch
import torch.nn as nn

MODES = ('off', 'attn', 'mean')


def mode_from_env() -> str:
    m = os.environ.get('DMPO_SET_ACTOR', 'off').strip().lower() or 'off'
    if m not in MODES:
        raise ValueError(f"DMPO_SET_ACTOR={m!r} — expected one of {MODES}")
    return m


def nb_layout(obs_dim: int):
    """(start, n_slots, n_feat) of the neighbour block inside the flat v2 vector.
    Offsets come from ObservationProcessor's own constants (single source of truth)."""
    from dmpo_autonomous_driving.envs.observation_processor import ObservationProcessor as OP
    n_slots = int(os.environ['DMPO_MAX_NEIGHBORS'])
    ego = OP.V2_EGO_DIM + (4 if os.environ.get('DMPO_PROFILE_OBS', '0') == '1' else 0)
    start = ego + OP.V2_INFRA_DIM
    if OP.obs_dim(n_slots, True) != obs_dim:
        raise RuntimeError(f"set_actor: obs_dim {obs_dim} != v2 layout "
                           f"{OP.obs_dim(n_slots, True)} — refusing a mis-sliced trunk")
    return start, n_slots, OP.V2_NB_FEAT


class SetTrunk(nn.Module):
    CHUNK = 16384          # rows per attention call (×2 heads stays under 65,535)

    def __init__(self, obs_dim: int, out_size: int, mode: str, emb: int = 64, heads: int = 2):
        super().__init__()
        assert mode in ('attn', 'mean')
        self.mode = mode
        self.start, self.n_slots, self.n_feat = nb_layout(obs_dim)
        self.end = self.start + self.n_slots * self.n_feat
        rest = obs_dim - (self.end - self.start)
        neutral = torch.zeros(self.n_feat)
        neutral[7] = 1.0
        neutral[9] = 1.0
        self.register_buffer('neutral', neutral, persistent=False)
        self.ego_enc = nn.Sequential(nn.Linear(rest, 128), nn.ReLU(), nn.Linear(128, emb), nn.ReLU())
        self.phi = nn.Sequential(nn.Linear(self.n_feat, emb), nn.ReLU(), nn.Linear(emb, emb), nn.ReLU())
        if mode == 'attn':
            self.attn = nn.MultiheadAttention(emb, heads, batch_first=True)
        self.rho = nn.Sequential(nn.Linear(2 * emb, 64), nn.ReLU(), nn.Linear(64, out_size))

    def forward(self, s):
        lead = s.shape[:-1]
        s = s.reshape(-1, s.shape[-1])
        nb = s[:, self.start:self.end].reshape(-1, self.n_slots, self.n_feat)
        rest = torch.cat([s[:, :self.start], s[:, self.end:]], dim=-1)
        valid = (nb - self.neutral).abs().sum(-1) > 1e-6            # [B, K]
        e = self.ego_enc(rest)                                      # [B, emb]
        h = self.phi(nb)                                            # [B, K, emb]
        if self.mode == 'mean':
            w = valid.unsqueeze(-1).to(h.dtype)
            pooled = (h * w).sum(1) / w.sum(1).clamp(min=1.0)       # 0 when no neighbours
        else:
            # ego is itself an entity (Leurent & Mercat) → never an all-masked row
            keys = torch.cat([e.unsqueeze(1), h], dim=1)            # [B, 1+K, emb]
            pad = torch.cat([torch.zeros_like(valid[:, :1]), ~valid], dim=1)
            # Row-chunked: the fused SDPA kernel cannot launch when rows×heads exceeds
            # the CUDA grid limit (65,535) — PPO full-batch passes ~70k padded rows.
            # Rows are independent, so chunking is exact.
            q = e.unsqueeze(1)
            pooled = torch.cat([
                self.attn(q[i:i + self.CHUNK], keys[i:i + self.CHUNK], keys[i:i + self.CHUNK],
                          key_padding_mask=pad[i:i + self.CHUNK], need_weights=False)[0]
                for i in range(0, q.shape[0], self.CHUNK)], dim=0).squeeze(1)
        out = self.rho(torch.cat([e, pooled], dim=-1))
        return out.reshape(*lead, -1)


def apply(actors, obs_dim: int, device, mode: str) -> int:
    """Swap every actor's trunk for a SetTrunk. Returns #params per trunk (0 = no-op)."""
    if mode == 'off':
        return 0
    if os.environ.get('DMPO_OBS_V2', '0') != '1' or os.environ.get('DMPO_DYNAMIC', '0') != '1':
        raise RuntimeError("DMPO_SET_ACTOR requires DMPO_OBS_V2=1 and DMPO_DYNAMIC=1")
    if os.environ.get('DMPO_SHARED_POLICY', '0') == '1':
        raise RuntimeError("DMPO_SET_ACTOR is incompatible with DMPO_SHARED_POLICY=1")
    n = 0
    for actor in actors:
        actor.network = SetTrunk(obs_dim, actor.output_size, mode).to(device)
        n = sum(p.numel() for p in actor.network.parameters())
    return n


def apply_from_checkpoint(actors, ck: dict, obs_dim: int, device) -> str:
    """Eval side: the checkpoint is the source of truth. An env value, if set, must agree."""
    ck_mode = ck.get('set_actor', 'off')
    env = os.environ.get('DMPO_SET_ACTOR')
    if env is not None and env.strip().lower() not in ('', ck_mode):
        raise RuntimeError(f"DMPO_SET_ACTOR={env!r} but checkpoint was trained with {ck_mode!r}")
    apply(actors, obs_dim, device, ck_mode)
    return ck_mode
