#!/usr/bin/env python3
"""Pre-launch consistency gate (Stage 0).

Diffs the LIVE environment against the canonical consistency block
(configs/consistency_block.sh, profile-resolved) and REFUSES (exit 1) on any
missing or mismatched Category-A var. Root cause being killed: hand-typed env
blocks silently dropping consistency-critical vars (C7 — the aborted 4-arm run).

Also (C2): with --scenario <sumocfg>, asserts every vType decel in the loaded
route files == abs(DMPO_MAX_DECEL) — the turn60 half-braking bug class.

Usage:
    source configs/consistency_block.sh            # (with DMPO_BLOCK_PROFILE set)
    python3 scripts/assert_consistency.py [--scenario path/to/x.sumocfg] [--quiet]

Exit 0 = green (prints the resolved block hash for the effective-config dump).
Exit 1 = refused, with the exact list of violations.
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCK = os.path.join(REPO, 'configs', 'consistency_block.sh')


def resolve_block(profile: str) -> dict:
    """Resolve the block exactly as bash would, for the given profile."""
    out = subprocess.run(
        ['bash', '-c',
         f'export DMPO_BLOCK_PROFILE={profile}; source {BLOCK} >/dev/null 2>&1; env'],
        capture_output=True, text=True, check=True).stdout
    block = {}
    for line in out.splitlines():
        m = re.match(r'(DMPO_[A-Z_0-9]+)=(.*)$', line)
        if m and m.group(1) not in ('DMPO_BLOCK_PROFILE',):
            block[m.group(1)] = m.group(2)
    return block


def check_vtypes(sumocfg: str, max_decel: float, problems: list):
    """C2: every vType decel in the scenario's route files must equal |DMPO_MAX_DECEL|."""
    cfg_dir = os.path.dirname(os.path.abspath(sumocfg))
    try:
        root = ET.parse(sumocfg).getroot()
        route_files = []
        for inp in root.iter('route-files'):
            route_files += [v for v in inp.get('value', '').split(',') if v]
    except Exception as e:
        problems.append(f'C2: cannot parse sumocfg {sumocfg}: {e}')
        return
    for rf in route_files:
        path = os.path.join(cfg_dir, rf)
        try:
            rroot = ET.parse(path).getroot()
        except Exception as e:
            problems.append(f'C2: cannot parse route file {path}: {e}')
            continue
        for vt in rroot.iter('vType'):
            decel = vt.get('decel')
            if decel is None:
                problems.append(
                    f"C2: vType '{vt.get('id')}' in {rf} has NO decel attr "
                    f"(SUMO default 4.5 != required {max_decel})")
            elif abs(float(decel) - max_decel) > 1e-6:
                problems.append(
                    f"C2: vType '{vt.get('id')}' in {rf} decel={decel} != "
                    f"|DMPO_MAX_DECEL|={max_decel} (half-braking bug class)")
            ed = vt.get('emergencyDecel')
            if ed is None and float(decel or 4.5) > 4.5:
                problems.append(
                    f"C2: vType '{vt.get('id')}' in {rf} sets decel={decel} but no "
                    f"emergencyDecel (SUMO default 9.0 may clamp differently)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default=None, help='sumocfg to physics-check (C2)')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    profile = os.environ.get('DMPO_BLOCK_PROFILE', 'legacy_r5_bare')
    block = resolve_block(profile)
    if not block:
        print('ASSERT-CONSISTENCY: could not resolve the block — refusing.')
        sys.exit(1)

    problems, missing, mismatched = [], [], []
    for var, want in sorted(block.items()):
        have = os.environ.get(var)
        if have is None:
            missing.append(f'  MISSING    {var} (block wants {want!r})')
        elif have != want:
            mismatched.append(f'  MISMATCH   {var}: env={have!r} block={want!r}')
    problems += missing + mismatched

    # unknown DMPO_ vars in env but in neither block (warning, not fatal):
    # they may be training-block or debug vars — surfaced so nothing hides.
    tb = {}
    tb_path = os.path.join(REPO, 'configs', 'training_block.sh')
    if os.path.exists(tb_path):
        for line in open(tb_path):
            m = re.match(r'\s*export\s+(DMPO_[A-Z_0-9]+)=', line)
            if m:
                tb[m.group(1)] = True
    unknown = [v for v in os.environ
               if v.startswith('DMPO_') and v not in block and v not in tb
               and v not in ('DMPO_BLOCK_PROFILE', 'DMPO_BLOCK_VERSION',
                             'DMPO_TRAINING_BLOCK_VERSION', 'DMPO_REQUIRE_BLOCK',
                             'DMPO_SUMO_SEED', 'DMPO_USE_TRACI', 'DMPO_FROZEN_TEST')]

    if args.scenario:
        check_vtypes(args.scenario, abs(float(block.get('DMPO_MAX_DECEL', '4.5'))), problems)

    blob = '\n'.join(f'{k}={v}' for k, v in sorted(block.items()))
    block_hash = hashlib.sha256(blob.encode()).hexdigest()[:12]

    if problems:
        print(f'ASSERT-CONSISTENCY: REFUSED (profile={profile}, {len(problems)} violation(s)):')
        for p in problems:
            print(p)
        sys.exit(1)

    if not args.quiet:
        print(f'ASSERT-CONSISTENCY: GREEN  profile={profile}  block_hash={block_hash}  '
              f'vars={len(block)}  entropy={os.environ.get("DMPO_ENTROPY", "(unset->0.05 TRAP H12)")}  '
              f'curriculum_mode={os.environ.get("DMPO_CURRICULUM_MODE", "(unset->cap TRAP M17)")}')
        if os.environ.get('DMPO_OBS_V2', '0') == '1':
            # OBS-V2: print the ONE obs_dim arithmetic so the pre-flight eyeball
            # sees 241 (N=15). Anti-F6b overrides are refused by the generic diff
            # above (every ignored-legacy flag is pinned in the block).
            _n = int(os.environ.get('DMPO_MAX_NEIGHBORS', '10'))
            _d = 61 + 12 * _n
            _prof = os.environ.get('DMPO_PROFILE_OBS', '0') == '1'
            if _prof:
                # derive from the authoritative constant, never restate (R8);
                # fallback 20 only if the package is unimportable in this shell
                try:
                    from dmpo_autonomous_driving.envs.observation_processor import (
                        ObservationProcessor as _OP)
                    _d += 4 + 4 * _OP.V2_CONFLICT_K
                except Exception:
                    _d += 20
            print(f'  OBS-V2 ACTIVE: obs_dim = 61 + 12*{_n}{" + 20 (ARM-B profile)" if _prof else ""} = {_d}  '
                  f'(no z-filter; ignored legacy obs flags pinned canonical)')
        if unknown:
            print(f'  WARNING: {len(unknown)} DMPO_ vars in env not covered by any block: '
                  f'{", ".join(sorted(unknown)[:12])}')
    sys.exit(0)


if __name__ == '__main__':
    main()
