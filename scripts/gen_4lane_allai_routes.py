"""Generate a 4-lane ALL-AI route file with lane-spread, seeded continuous demand.

Unlike generate_intersection.py (departLane="best" → concentrates) and the Bremen
2-lane builder, this:
  - spreads departs across all 4 lanes (departLane="random"),
  - sets demand from the REAL 4-lane trip time (not the 60s assumption),
  - is reproducible (SUMO --seed controls flow stochasticity; this only writes rates),
  - keeps turn diversity (40% straight, 30/30 turns) for conflict richness.

Concurrent count target N (Little's law): concurrent ≈ arrival_rate × trip_time.
  per-approach vehsPerHour = N / trip_s * 3600 / 4 (4 approaches).
"""
import argparse, os

ROUTES = {
    'north': [('north_to_center center_to_south', 0.4), ('north_to_center center_to_east', 0.3), ('north_to_center center_to_west', 0.3)],
    'south': [('south_to_center center_to_north', 0.4), ('south_to_center center_to_east', 0.3), ('south_to_center center_to_west', 0.3)],
    'east':  [('east_to_center center_to_west', 0.4),  ('east_to_center center_to_north', 0.3), ('east_to_center center_to_south', 0.3)],
    'west':  [('west_to_center center_to_east', 0.4),  ('west_to_center center_to_north', 0.3), ('west_to_center center_to_south', 0.3)],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, help='output .rou.xml path')
    p.add_argument('--target-concurrent', type=int, default=30, help='~simultaneous AI vehicles')
    p.add_argument('--trip-seconds', type=float, default=28.0, help='free-flow trip time (4-lane 200m≈28s)')
    p.add_argument('--max-speed', type=float, default=16.67)
    p.add_argument('--decel', type=float, default=8.0)
    p.add_argument('--emergency-decel', type=float, default=9.0)
    p.add_argument('--end', type=int, default=360000, help='continuous spawning horizon')
    args = p.parse_args()

    # per-approach vehsPerHour to maintain ~N concurrent
    vph = args.target_concurrent / args.trip_seconds * 3600.0 / 4.0

    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
             f'  <vType id="ai_vehicle" vClass="passenger" color="0,0,255" accel="2.6" decel="{args.decel}" '
             f'emergencyDecel="{args.emergency_decel}" sigma="0.0" length="4.5" maxSpeed="{args.max_speed}" minGap="2.0"/>']
    for approach, routes in ROUTES.items():
        lines.append(f'  <flow id="ai_{approach}" type="ai_vehicle" begin="0" end="{args.end}" '
                     f'vehsPerHour="{vph:.2f}" departLane="random" departSpeed="max">')
        lines.append('    <routeDistribution>')
        for edges, prob in routes:
            lines.append(f'      <route probability="{prob}" edges="{edges}"/>')
        lines.append('    </routeDistribution>')
        lines.append('  </flow>')
    lines.append('</routes>')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"wrote {args.out}")
    print(f"  target_concurrent={args.target_concurrent}  trip_s={args.trip_seconds}  -> vehsPerHour/approach={vph:.1f}  (total {vph*4:.0f} veh/h)")


if __name__ == '__main__':
    main()
