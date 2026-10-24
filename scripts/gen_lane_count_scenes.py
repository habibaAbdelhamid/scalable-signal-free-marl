#!/usr/bin/env python
"""Build the 1-, 2- and 3-lane versions of the lanefix operating benchmark.

Each scene is the 4-lane benchmark (configs/sumo_configs/4lane_allAI_unsig_poisson_lanefix) with only
the lane count changed:
  - network: scripts/generate_intersection.py with --arm-length 200 (the same generator reproduces the
    4-lane benchmark network exactly: edges, lengths, speeds, right-before-left centre, connections);
  - demand: the same per-turn Poisson streams, with every rate scaled by L/4, so the demand PER LANE
    equals the 4-lane benchmark at the same scale factor s; right turns enter lane 0, left turns
    lane L-1, through traffic a random lane (every lane connects straight);
  - vehicle type and the .sumocfg (collision warn + junction checking, step 0.2 s) unchanged.

Usage: python scripts/gen_lane_count_scenes.py   (writes configs/sumo_configs/{L}lane_allAI_unsig_poisson_lanefix)
"""
import os, re, shutil, subprocess, sys, tempfile
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, 'configs/sumo_configs/4lane_allAI_unsig_poisson_lanefix')
GEN = os.path.join(REPO, 'scripts/generate_intersection.py')


def build(L):
    dst = os.path.join(REPO, f'configs/sumo_configs/{L}lane_allAI_unsig_poisson_lanefix')
    os.makedirs(dst, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, GEN, '--lanes', str(L), '--arm-length', '200', '--output', tmp],
                       check=True, capture_output=True, text=True)
        shutil.copy(os.path.join(tmp, 'intersection.net.xml'), os.path.join(dst, 'intersection.net.xml'))
    shutil.copy(os.path.join(SRC, 'intersection.sumocfg'), os.path.join(dst, 'intersection.sumocfg'))
    rou = open(os.path.join(SRC, 'intersection.rou.xml')).read()

    def scale(m):
        return f'period="exp({float(m.group(1)) * L / 4.0:.6f})"'
    rou = re.sub(r'period="exp\(([0-9.]+)\)"', scale, rou)
    rou = rou.replace('departLane="3"', f'departLane="{L - 1}"')
    rou = rou.replace('<!-- LANEFIX:', f'<!-- {L}-LANE VERSION of the lanefix benchmark (rates x {L}/4 = same demand per lane;'
                      f' left turns -> lane {L - 1}). Original note - LANEFIX:')
    open(os.path.join(dst, 'intersection.rou.xml'), 'w').write(rou)
    return dst


def check(dst, L):
    n = ET.parse(os.path.join(dst, 'intersection.net.xml')).getroot()
    edges = [e for e in n.findall('edge') if e.get('function') != 'internal']
    assert all(len(e.findall('lane')) == L for e in edges), 'lane count'
    lens = {round(float(e.find('lane').get('length')), 1) for e in edges}
    con = [(c.get('fromLane'), c.get('dir')) for c in n.findall('connection') if c.get('from') == 'north_to_center']
    right = sorted(int(fl) for fl, d in con if d == 'r'); left = sorted(int(fl) for fl, d in con if d == 'l')
    straight = sorted(int(fl) for fl, d in con if d == 's')
    assert right == [0] and left == [L - 1] and straight == list(range(L)), f'connections {con}'
    rou = open(os.path.join(dst, 'intersection.rou.xml')).read()
    rates = [float(x) for x in re.findall(r'period="exp\(([0-9.]+)\)"', rou)]
    print(f'{L}-lane: lanes ok, lengths {lens}, right {right} left {left} straight {straight}, '
          f'demand {sum(rates) * 3600:.0f} veh/h at s=1 ({sum(rates) * 3600 / 4 / L:.1f} per lane per approach)')


if __name__ == '__main__':
    for L in (1, 2, 3):
        check(build(L), L)
