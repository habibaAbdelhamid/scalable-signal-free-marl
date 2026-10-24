"""Identity-matched world-model loss for the OBS-V2 neighbour block.

Gated by ``DMPO_MODEL_MATCHED_LOSS`` (train-only; default OFF = the trainer never
imports these paths at runtime beyond the module import).

Problem being solved (world-model loss on a slot-ordered neighbour block): the
v2 observation stores the N nearest vehicles nearest-first, so a slot-wise MSE
between the model's prediction for slot k and s1's slot k compares DIFFERENT
vehicles whenever the ordering changes between t and t+1 (overtakes, births,
deaths, entrants). These helpers build a target where each reference slot is
matched to the SAME vehicle's next-step row (found via the captured slot→id
lists), mask the slots that cannot be supervised (left the perception set /
entered it), and canonicalise synthetic rollout states back to nearest-first.

Pure torch — no SUMO / framework imports — so it is unit-testable in
``tests/unit/test_matched_model_loss.py``. The trainer only calls into it.

Conventions (agent axis ``n`` always explicit; NO ``.squeeze()`` anywhere):
    s1, s1m, w : [L, n, D]  float
    sid, sidn  : [L, n, N]  int   (interned vehicle id per neighbour slot; −1 = empty;
                                   sidn == −2 in EVERY slot = ego ABSENT at t+1)
    ref        : [1, n, N] (chain mode = sid[0:1]) or [L, n, N] (one-step mode = sid)
    neighbour block = flat[off : off + N*F], slot k = flat[off + F*k : off + F*k + F]
"""
import os

import torch
import torch.nn.functional as _F

__all__ = [
    'keep_empty_enabled',
    'build_matched_target',
    'match_stats',
    'format_match_stats',
    'STAT_KEYS',
    'resort_nb_slots',
    'masked_state_terms',
    'chain_state_rel_err',
]


def keep_empty_enabled() -> bool:
    """``DMPO_MODEL_MATCHED_KEEP_EMPTY`` (default '1'): an empty reference slot whose
    next slot k is ALSO empty keeps weight 1 with s1's own (neutral) row as target
    ("empty stays empty"). '0' = the literal masking of empty ref slots (ablation)."""
    return os.environ.get('DMPO_MODEL_MATCHED_KEEP_EMPTY', '1') == '1'


def _match(ref, sidn):
    """Return (eq, found, j, ref_b) — eq [L,n,N,N] (dim −2 = ref slot k, dim −1 =
    next slot j), found [L,n,N], j [L,n,N] (argmax over the one-hot row), and the
    reference ids broadcast to sidn's shape [L,n,N]."""
    ref = ref.to(sidn.device)
    ref_b = torch.broadcast_to(ref, sidn.shape)                          # [L,n,N]
    # [1|L,n,N,1] vs [L,n,1,N] → [L,n,N,N]; empty ref slots (−1) never match
    eq = (ref.unsqueeze(-1) == sidn.unsqueeze(-2)) & (ref.unsqueeze(-1) >= 0)
    found = eq.any(-1)                               # [L,n,N]
    j = eq.to(torch.int8).argmax(-1)                 # [L,n,N] (ids unique per list ⇒ ≤1 hit)
    return eq, found, j, ref_b


def build_matched_target(s1, sid, sidn, ref, off, N, F, keep_empty=None,
                         return_stats=False, neutral=None):
    """Build the identity-matched next-state target and its feature weight.

    Args:
        s1   [L,n,D] float — slot-wise next-state target (unchanged outside the nb block)
        sid  [L,n,N] int   — slot ids at t (kept for signature symmetry / stats)
        sidn [L,n,N] int   — slot ids at t+1 (−1 = empty; −2 in EVERY slot = the ego
             itself is ABSENT at t+1 → the whole nb block is unsupervisable, w=0)
        ref  [1,n,N] or [L,n,N] int — reference slot ids: ``sid[0:1]`` for the
             autoregressive chain (step k's prediction lives in s[0]'s slot order),
             ``sid`` for one-step matching.
        off, N, F — neighbour-block geometry (read from the live processor).
        keep_empty — None → ``keep_empty_enabled()``.
        neutral — optional [F] float: the processor's empty-slot row. When given it
             is the target of KEPT-EMPTY slots ("empty stays empty" → predict the
             neutral row, not whatever s1 happens to hold there); None keeps s1's
             own slot-k row as the fallback (identical for real data, where empty
             slots ARE neutral rows).
    Returns:
        (s1m, w) — s1m [L,n,D] (== s1 except at matched-and-moved nb rows and at
        kept-empty rows when ``neutral`` differs from s1's row), w [L,n,D] float
        ∈ {0,1} (== 1 outside the nb block; == 0 over the WHOLE nb block on
        ego-absent rows).  With ``return_stats=True`` a third element: dict of raw
        slot counts for the C5 diagnostic line.
    """
    if keep_empty is None:
        keep_empty = keep_empty_enabled()
    s1 = torch.as_tensor(s1)
    sidn = torch.as_tensor(sidn, device=s1.device)
    ref = torch.as_tensor(ref, device=s1.device)
    L, n, D = s1.shape
    lo, hi = off, off + N * F
    assert hi <= D, f'nb block [{lo}:{hi}] exceeds obs_dim {D}'
    nb1 = s1[..., lo:hi].reshape(L, n, N, F)

    eq, found, j, ref_b = _match(ref, sidn)          # [L,n,N,N], [L,n,N], [L,n,N], [L,n,N]
    tgt = nb1.gather(2, j.unsqueeze(-1).expand(-1, -1, -1, F))   # [L,n,N,F]
    empty_ref = ref_b < 0
    empty_next_k = sidn < 0
    # ego absent at t+1 (explicit −2 encoding from _prepare_dmpo_batch_dynamic):
    # nothing in the nb block can be supervised → no kept empties, w=0 everywhere
    absent = (sidn == -2).all(-1, keepdim=True).expand_as(found)   # [L,n,N]
    keep_empty_mask = (empty_ref & empty_next_k & ~absent) if keep_empty else torch.zeros_like(found)
    w_nb = (found | keep_empty_mask).to(s1.dtype)   # [L,n,N]  (found is False on absent rows)
    # unmatched / empty slots fall back to s1's own slot-k row so s1m ≠ s1 only at
    # matched-and-moved rows; kept empties target the exact neutral row when given
    if neutral is not None:
        neutral = torch.as_tensor(neutral, dtype=s1.dtype, device=s1.device).reshape(F)
        fallback = torch.where(keep_empty_mask.unsqueeze(-1), neutral.expand_as(nb1), nb1)
    else:
        fallback = nb1
    tgt = torch.where(found.unsqueeze(-1), tgt, fallback)

    s1m = s1.clone()
    s1m[..., lo:hi] = tgt.reshape(L, n, N * F)
    w = torch.ones_like(s1)
    w[..., lo:hi] = w_nb.unsqueeze(-1).expand(-1, -1, -1, F).reshape(L, n, N * F)
    if not return_stats:
        return s1m, w
    return s1m, w, _stats_tensors(found, j, ref_b, sidn, keep_empty_mask)


STAT_KEYS = ('ref_real', 'found', 'moved', 'lost', 'empty_ref', 'empty_kept', 'entrant_masked')


def _stats_tensors(found, j, ref_b, sidn, kept):
    """Slot-count diagnostics as 0-dim int64 tensors ON DEVICE (no .item() sync —
    the trainer accumulates over a batch and syncs once). Keys = STAT_KEYS:
    ref_real (real reference slots), found (matched at t+1), moved (found & j≠k),
    lost (real ref, left the set), empty_ref, empty_kept (weight-1 empties),
    entrant_masked (empty ref, occupied next → masked)."""
    N = sidn.shape[-1]
    k_idx = torch.arange(N, device=sidn.device).view(1, 1, N).expand_as(found)
    empty_ref = ref_b < 0
    empty_next_k = sidn < 0
    real_ref = ~empty_ref
    return {
        'ref_real': real_ref.sum(),
        'found': (found & real_ref).sum(),
        'moved': (found & real_ref & (j != k_idx)).sum(),
        'lost': (real_ref & ~found).sum(),
        'empty_ref': empty_ref.sum(),
        'empty_kept': kept.sum(),
        'entrant_masked': (empty_ref & ~empty_next_k).sum(),
    }


def match_stats(ref, sidn, keep_empty=None):
    """Slot-count diagnostics only (python ints; no target tensors built): same
    numbers as ``build_matched_target(..., return_stats=True)[2]``."""
    if keep_empty is None:
        keep_empty = keep_empty_enabled()
    sidn = torch.as_tensor(sidn)
    ref = torch.as_tensor(ref, device=sidn.device)
    _, found, j, ref_b = _match(ref, sidn)
    empty_ref = ref_b < 0
    empty_next_k = sidn < 0
    absent = (sidn == -2).all(-1, keepdim=True).expand_as(found)
    kept = (empty_ref & empty_next_k & ~absent) if keep_empty else torch.zeros_like(found)
    return {k: int(v.item()) for k, v in _stats_tensors(found, j, ref_b, sidn, kept).items()}


def format_match_stats(acc):
    """One diagnostic line from an accumulated stats dict (ints or 0-dim tensors)."""
    g = {k: int(acc[k].item()) if torch.is_tensor(acc[k]) else int(acc[k]) for k in STAT_KEYS}
    rr = max(g['ref_real'], 1)
    er = max(g['empty_ref'], 1)
    return (f"ref_real={g['ref_real']} found={g['found']/rr:.1%} moved={g['moved']/rr:.1%} "
            f"lost={g['lost']/rr:.1%} | empty_ref={g['empty_ref']} kept={g['empty_kept']/er:.1%} "
            f"entrant_masked={g['entrant_masked']/er:.1%}")


def resort_nb_slots(s, off, N, F, neutral, tau=0.25, snap_empty=True, front_lam=0.0):
    """Canonicalise a (predicted) state's neighbour block to nearest-first.

    s       [..., D] float (any leading dims, e.g. [n_traj, n_agent, D])
    neutral [F] float — the processor's empty-slot row (v2: [0,0,0,0,0,0,0,1,0,1,0,0])
    front_lam — L_nominal / R (vehicle length over the processor's distance
            normaliser). The processor orders slots by FRONT-bumper Euclidean
            distance (``_get_neighbor_states`` sorts on the cached SUMO 'position'
            = front bumper) but ``_nb_v2`` stores the REAR-bumper offset
            (rx = pn − Ln·sin an, ry = pn − Ln·cos an, rotated to the ego frame).
            Rotating the length vector into the ego frame gives exactly
            Ln·(sin Δh, cos Δh) = Ln·(feat4, feat5), so the front bumper is
            (feat0 + front_lam·feat4, feat1 + front_lam·feat5). front_lam=0 keeps
            the (wrong-key) rear-bumper ordering — tests document the difference.
    Empty detector: ``||row − neutral||² < tau``  OR  heading-pair norm² < 0.25
    (a real v2 row always carries a UNIT heading pair (rel_head_sin² +
    rel_head_cos² ≈ 1) where the neutral AND the all-zero row have 0 there — so a
    zero row is empty even though ||0 − neutral||² = 2 ≥ tau)  OR  the predicted
    front-bumper distance exceeds 1.1 (beyond the perception radius: snapped to
    neutral, never promoted to a real slot).
    Sort key = predicted planar FRONT-bumper distance + 10·empty, so real rows
    come first (nearest-first) and empties trail; ``stable=True`` keeps ties
    deterministic. ``snap_empty`` replaces empties by the EXACT neutral row.
    Every dim outside the nb block is returned untouched.
    """
    s = torch.as_tensor(s)
    lead = s.shape[:-1]
    D = s.shape[-1]
    lo, hi = off, off + N * F
    assert hi <= D, f'nb block [{lo}:{hi}] exceeds obs_dim {D}'
    neutral = torch.as_tensor(neutral, dtype=s.dtype, device=s.device).reshape(F)
    nb = s[..., lo:hi].reshape(*lead, N, F)
    # front-bumper planar offset (see front_lam above); == rear-bumper when front_lam=0
    dx = nb[..., 0] + front_lam * nb[..., 4]
    dy = nb[..., 1] + front_lam * nb[..., 5]
    dist = torch.sqrt(dx ** 2 + dy ** 2)                                 # [..., N]
    empty = ((nb - neutral) ** 2).sum(-1) < tau                          # [..., N]
    empty |= (nb[..., 4] ** 2 + nb[..., 5] ** 2) < 0.25                  # no unit heading pair
    empty |= dist > 1.1                                                  # far-clip (beyond R)
    key = dist + empty.to(s.dtype) * 10.0
    order = torch.argsort(key, dim=-1, stable=True)                      # [..., N]
    nb_sorted = nb.gather(-2, order.unsqueeze(-1).expand(*order.shape, F))
    if snap_empty:
        empty_sorted = empty.gather(-1, order)
        nb_sorted = torch.where(empty_sorted.unsqueeze(-1), neutral.expand_as(nb_sorted), nb_sorted)
    out = s.clone()
    out[..., lo:hi] = nb_sorted.reshape(*lead, N * F)
    return out


def masked_state_terms(state_pred, s1m, w, mask_expanded):
    """Numerator / denominator of the masked state loss.

    w=None reproduces the pre-flag expression bit-for-bit:
        num = (MSE(pred, s1) * mask).sum(),   den = mask.sum() * D
    w given (flag on):
        num = (MSE(pred, s1m) * w * mask).sum(),   den = (w * mask).sum()
    (MSE = torch.nn.MSELoss(reduction='none') == F.mse_loss(..., reduction='none').)
    """
    mse = _F.mse_loss(state_pred, s1m, reduction='none')
    if w is None:
        num = (mse * mask_expanded).sum()
        den = mask_expanded.sum() * s1m.shape[-1]
        return num, den
    wm = w * mask_expanded
    num = (mse * wm).sum()
    den = wm.sum()
    return num, den


def chain_state_rel_err(model, s, a, s1m, w, length, forward=None):
    """Matched relative state error of the L-step autoregressive chain (no grad).

    Reruns the chain exactly as the framework's ``train_model`` (models.py
    L636-645): s0 = s[0:1]; for t: _, s_, _ = model.forward(s0, a[t:t+1]); s0 = s_.
    Returns   (Σ (pred − s1m)²·w / Σ w) / (Σ w·(s1m − μ_w)² / Σ w + 1e-7), with μ_w the
    w-WEIGHTED per-feature mean of the s1m rows — the same units as the framework's
    ``rel_state_loss`` (masked MSE / pooled var) and the SAME weighted-pooled
    definition as the trainer's "stable" rel-err print under the flag (unweighted
    var would count masked (left/entered/absent) rows in the denominator only).
    ``forward`` overrides the callable (default ``model.forward``); it must return
    a (reward, state, done) tuple.
    """
    fwd = forward if forward is not None else model.forward
    with torch.no_grad():
        L = min(int(length), s.shape[0], s1m.shape[0])
        s0 = s[0:1]
        preds = []
        for t in range(L):
            _, s_, _ = fwd(s0, a[t:t + 1])
            preds.append(s_)
            s0 = s_
        state_pred = torch.cat(preds, dim=0)                            # [L,n,D]
        tgt = s1m[:L]
        wt = w[:L]
        ones = torch.ones(tgt.shape[:-1] + (1,), dtype=tgt.dtype, device=tgt.device)
        num, den = masked_state_terms(state_pred, tgt, wt, ones)
        state_loss = num / (den + 1e-8)
        rows = tgt.reshape(-1, tgt.shape[-1])
        wrows = wt.reshape(-1, wt.shape[-1])
        mu_w = (rows * wrows).sum(0, keepdim=True) / wrows.sum(0, keepdim=True).clamp_min(1)
        state_var = ((rows - mu_w).pow(2) * wrows).sum() / wrows.sum().clamp_min(1)
        return state_loss / (state_var + 1e-7)
