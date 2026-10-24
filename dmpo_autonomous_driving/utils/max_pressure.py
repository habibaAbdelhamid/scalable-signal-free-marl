"""Max-pressure signal control — network-level signal baseline.
Gated DMPO_TLS_CONTROLLER=maxpressure, default
unset = no-op (SUMO runs the net's own programs).

Varaiya (2013), "Max pressure control of a network of signalized intersections",
Transp. Res. C 36: at each decision instant every junction activates the green stage
with the largest pressure  P(s) = Σ_{movements (i→o) green in s} (q_i − q_o),
q = vehicles on the incoming / outgoing lane. Practical form used in the ATSC
literature (e.g. PressLight, Wei et al. KDD 2019 baseline): decision every
DMPO_MP_DT s (default 5), minimum green DMPO_MP_MIN_GREEN s (default 5), and the
net's own yellow phase (its programmed duration) between stages.

Stages = the program's green phases (contain G/g, no y); the yellow of a stage is
the phase that follows it in the program when that phase contains 'y'. Junctions
switched off by DMPO_TLS_OFF_LIST are left alone (unsignalized cells).
"""
import os

HOLD = 1e5  # phase duration that keeps SUMO from auto-advancing a held phase


class MaxPressureController:
    def __init__(self, traci):
        self.t = traci
        self.dt = float(os.environ.get('DMPO_MP_DT', '5'))
        self.min_green = float(os.environ.get('DMPO_MP_MIN_GREEN', '5'))
        off = {x.strip() for x in os.environ.get('DMPO_TLS_OFF_LIST', '').split(',') if x.strip()}
        self.j = {}
        for tls in traci.trafficlight.getIDList():
            if tls in off:
                continue
            prog = traci.trafficlight.getProgram(tls)
            logic = next((l for l in traci.trafficlight.getAllProgramLogics(tls)
                          if l.programID == prog), None)
            if logic is None:
                continue
            ph = list(logic.phases)
            greens = [i for i, p in enumerate(ph)
                      if any(c in 'Gg' for c in p.state) and 'y' not in p.state]
            if len(greens) < 2:
                continue
            yellow = {}
            for g in greens:
                n = (g + 1) % len(ph)
                yellow[g] = (n, float(ph[n].duration)) if 'y' in ph[n].state else None
            links = traci.trafficlight.getControlledLinks(tls)
            moves = {g: sorted({(lk[0][0], lk[0][1]) for k, lk in enumerate(links)
                                if lk and k < len(ph[g].state) and ph[g].state[k] in 'Gg'})
                     for g in greens}
            cur = traci.trafficlight.getPhase(tls)
            if cur not in greens:
                cur = greens[0]
            traci.trafficlight.setPhase(tls, cur)
            traci.trafficlight.setPhaseDuration(tls, HOLD)
            self.j[tls] = {'greens': greens, 'yellow': yellow, 'moves': moves,
                           'cur': cur, 'mode': 'green', 'since': -1e9, 'until': 0.0,
                           'target': None, 'last': -1e9}
        print(f"🚦 MAX-PRESSURE: {len(self.j)} junction(s) under control "
              f"(dt={self.dt}s, min_green={self.min_green}s)")

    def _pressure(self, moves, cache):
        lane_n = self.t.lane.getLastStepVehicleNumber
        p = 0.0
        for i, o in moves:
            if i not in cache:
                cache[i] = lane_n(i)
            if o not in cache:
                cache[o] = lane_n(o)
            p += cache[i] - cache[o]
        return p

    def step(self):
        tl = self.t.trafficlight
        now = self.t.simulation.getTime()
        cache = {}
        for tls, s in self.j.items():
            if s['mode'] == 'yellow':
                if now >= s['until']:
                    tl.setPhase(tls, s['target'])
                    tl.setPhaseDuration(tls, HOLD)
                    s.update(cur=s['target'], mode='green', since=now, target=None)
                continue
            if now - s['since'] < self.min_green or now - s['last'] < self.dt:
                continue
            s['last'] = now
            pr = {g: self._pressure(s['moves'][g], cache) for g in s['greens']}
            best = max(s['greens'], key=lambda g: (pr[g], g == s['cur']))
            if best == s['cur'] or pr[best] <= pr[s['cur']]:
                continue
            y = s['yellow'][s['cur']]
            if y is None:
                tl.setPhase(tls, best)
                tl.setPhaseDuration(tls, HOLD)
                s.update(cur=best, since=now)
            else:
                tl.setPhase(tls, y[0])
                tl.setPhaseDuration(tls, y[1])
                s.update(mode='yellow', until=now + y[1], target=best)


def make(traci):
    """Controller for the current SUMO run, or None when the flag is unset."""
    mode = os.environ.get('DMPO_TLS_CONTROLLER', '').strip().lower()
    if not mode:
        return None
    if mode != 'maxpressure':
        raise ValueError(f"DMPO_TLS_CONTROLLER={mode!r} — only 'maxpressure' is implemented")
    return MaxPressureController(traci)
