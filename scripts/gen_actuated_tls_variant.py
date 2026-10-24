#!/usr/bin/env python
"""Derive an ACTUATED-signal variant of a signalized SUMO scenario (rationale:
fixed-time signals are the weakest signal comparator).

The variant is byte-identical to the source except that every <tlLogic type="static">
becomes type="actuated" with minDur/maxDur on green phases (yellow/all-red phases
keep fixed duration). SUMO's actuated controller auto-generates induction loops on
the incoming lanes (default max-gap 3.0 s, detector-gap 2.0 s, passing-time 1.9 s).

Usage:
  python scripts/gen_actuated_tls_variant.py SRC_DIR DST_DIR SUMOCFG_NAME [--min-green 5] [--max-green MAX]
MAX defaults to the phase's static duration (the actuated program can only shorten
greens, never extend beyond the fixed plan) unless --max-green is given.
"""
import re, sys, os, shutil, argparse
ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('sumocfg')
ap.add_argument('--min-green', type=float, default=5.0)
ap.add_argument('--max-green', type=float, default=None)
a = ap.parse_args()
os.makedirs(a.dst, exist_ok=True)
cfg = open(os.path.join(a.src, a.sumocfg)).read()
net = re.search(r'net-file value="([^"]+)"', cfg).group(1)
for f in os.listdir(a.src):
    if f.endswith('.rou.xml') or f.endswith('.add.xml'):
        shutil.copy(os.path.join(a.src, f), os.path.join(a.dst, f))
xml = open(os.path.join(a.src, net)).read()
n_tl = 0; n_ph = 0
def fix_logic(m):
    global n_tl, n_ph
    n_tl += 1
    head, body, tail = m.group(1), m.group(2), m.group(3)
    head = head.replace('type="static"', 'type="actuated"')
    def fix_phase(pm):
        global n_ph
        dur = float(pm.group(1)); state = pm.group(2)
        if 'y' in state and 'G' not in state and 'g' not in state:   # yellow / clearance: fixed
            return pm.group(0)
        n_ph += 1
        mx = a.max_green if a.max_green is not None else dur
        mn = min(a.min_green, mx)
        return f'<phase duration="{dur:g}" minDur="{mn:g}" maxDur="{mx:g}" state="{state}"/>'
    body = re.sub(r'<phase duration="([\d.]+)"\s+state="([^"]+)"/>', fix_phase, body)
    return head + body + tail
xml2 = re.sub(r'(<tlLogic[^>]*>)(.*?)(</tlLogic>)', fix_logic, xml, flags=re.S)
assert n_tl > 0, 'no tlLogic found'
open(os.path.join(a.dst, net), 'w').write(xml2)
open(os.path.join(a.dst, a.sumocfg), 'w').write(cfg)
print(f'wrote {a.dst}: {n_tl} tlLogic -> actuated, {n_ph} green phases with minDur={a.min_green:g} maxDur={"static duration" if a.max_green is None else a.max_green}')
