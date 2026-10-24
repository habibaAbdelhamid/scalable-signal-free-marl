#!/usr/bin/env python3
"""
Generate SUMO intersection configurations for DMPO experiments.

Supports parameterized generation of 1-4 lane intersections with
configurable traffic flow, AI penetration, and vehicle types.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_intersection(num_lanes: int,
                         num_vehicles: int,
                         ai_penetration: float,
                         output_dir: str,
                         arm_length: int = 600,
                         speed_limit: float = 16.67,
                         truck_percentage: float = 0.02) -> str:
    """
    Generate complete SUMO configuration for intersection.

    Args:
        num_lanes: Number of lanes per direction (1-4)
        num_vehicles: Target number of vehicles in simulation
        ai_penetration: Fraction of AI vehicles (0.0-1.0)
        output_dir: Output directory for SUMO files
        arm_length: Length of each road arm in meters
        speed_limit: Speed limit in m/s
        truck_percentage: Percentage of truck traffic

    Returns:
        Path to generated .sumocfg file
    """
    print(f"🏗️ Generating {num_lanes}-lane intersection...")
    print(f"   Output: {output_dir}")
    print(f"   Vehicles: {num_vehicles} ({int(ai_penetration*100)}% AI)")

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Generate nodes
    nodes_file = generate_nodes(output_dir, arm_length)
    print(f"   ✅ Nodes: {nodes_file}")

    # Step 2: Generate edges
    edges_file = generate_edges(output_dir, num_lanes, speed_limit)
    print(f"   ✅ Edges: {edges_file}")

    # Step 3: Generate network using netconvert
    network_file = generate_network(output_dir, nodes_file, edges_file)
    print(f"   ✅ Network: {network_file}")

    # Step 4: Generate routes and vehicle flows
    routes_file = generate_routes(output_dir, num_vehicles, ai_penetration, truck_percentage)
    print(f"   ✅ Routes: {routes_file}")

    # Step 5: Generate SUMO configuration file
    sumocfg_file = generate_sumocfg(output_dir, network_file, routes_file)
    print(f"   ✅ Config: {sumocfg_file}")

    print(f"✅ Generated {num_lanes}-lane intersection successfully!")
    print()

    return sumocfg_file


def generate_nodes(output_dir: str, arm_length: int) -> str:
    """Generate nodes XML for intersection.

    Args:
        output_dir: Output directory
        arm_length: Length of each road arm

    Returns:
        Path to generated .nod.xml file
    """
    nodes_file = os.path.join(output_dir, "intersection.nod.xml")

    content = f'''<?xml version="1.0" encoding="UTF-8"?>
<nodes version="1.9" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">

    <!-- Center intersection node with traffic light -->
    <node id="center" x="0" y="0" type="right_before_left"/>

    <!-- Entry/exit nodes for each direction -->
    <node id="north" x="0" y="{arm_length}" type="priority"/>
    <node id="south" x="0" y="-{arm_length}" type="priority"/>
    <node id="east" x="{arm_length}" y="0" type="priority"/>
    <node id="west" x="-{arm_length}" y="0" type="priority"/>

</nodes>
'''

    with open(nodes_file, 'w') as f:
        f.write(content)

    return nodes_file


def generate_edges(output_dir: str, num_lanes: int, speed_limit: float) -> str:
    """Generate edges XML for intersection.

    Args:
        output_dir: Output directory
        num_lanes: Number of lanes per direction
        speed_limit: Speed limit in m/s

    Returns:
        Path to generated .edg.xml file
    """
    edges_file = os.path.join(output_dir, "intersection.edg.xml")

    content = f'''<?xml version="1.0" encoding="UTF-8"?>
<edges version="1.9" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">

    <!-- Incoming edges (approach to intersection) -->
    <edge id="north_to_center" from="north" to="center" numLanes="{num_lanes}" speed="{speed_limit:.2f}"/>
    <edge id="south_to_center" from="south" to="center" numLanes="{num_lanes}" speed="{speed_limit:.2f}"/>
    <edge id="east_to_center" from="east" to="center" numLanes="{num_lanes}" speed="{speed_limit:.2f}"/>
    <edge id="west_to_center" from="west" to="center" numLanes="{num_lanes}" speed="{speed_limit:.2f}"/>

    <!-- Outgoing edges (exit from intersection) -->
    <edge id="center_to_north" from="center" to="north" numLanes="{num_lanes}" speed="{speed_limit:.2f}"/>
    <edge id="center_to_south" from="center" to="south" numLanes="{num_lanes}" speed="{speed_limit:.2f}"/>
    <edge id="center_to_east" from="center" to="east" numLanes="{num_lanes}" speed="{speed_limit:.2f}"/>
    <edge id="center_to_west" from="center" to="west" numLanes="{num_lanes}" speed="{speed_limit:.2f}"/>

</edges>
'''

    with open(edges_file, 'w') as f:
        f.write(content)

    return edges_file


def generate_network(output_dir: str, nodes_file: str, edges_file: str) -> str:
    """Generate network using SUMO netconvert.

    Args:
        output_dir: Output directory
        nodes_file: Path to nodes XML
        edges_file: Path to edges XML

    Returns:
        Path to generated .net.xml file
    """
    network_file = os.path.join(output_dir, "intersection.net.xml")

    cmd = [
        'netconvert',
        '--node-files', nodes_file,
        '--edge-files', edges_file,
        '--output-file', network_file,
        '--no-turnarounds', 'false',
        '--no-warnings', 'true'
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running netconvert: {e.stderr}")
        raise

    return network_file


def generate_routes(output_dir: str,
                   num_vehicles: int,
                   ai_penetration: float,
                   truck_percentage: float = 0.02) -> str:
    """Generate routes XML with vehicle types and flows.

    Args:
        output_dir: Output directory
        num_vehicles: Target number of vehicles
        ai_penetration: Fraction of AI vehicles (0.0-1.0)
        truck_percentage: Fraction of trucks

    Returns:
        Path to generated .rou.xml file
    """
    routes_file = os.path.join(output_dir, "intersection.rou.xml")

    # Calculate vehicle type counts
    n_ai = int(num_vehicles * ai_penetration)
    n_truck = int(num_vehicles * truck_percentage)
    n_human = num_vehicles - n_ai - n_truck

    # Calculate spawn rate (vehicles per hour) for deterministic spawning
    # For continuous training: maintain target number of vehicles in steady state
    # Assume average trip duration ~60 seconds through intersection
    # To maintain N vehicles: spawn rate = N vehicles / 60s * 3600s/hr / 4 directions
    trip_duration = 60  # Average time a vehicle spends in simulation (seconds)
    ai_vph = (n_ai / trip_duration * 3600 / 4) if n_ai > 0 else 0
    human_vph = (n_human / trip_duration * 3600 / 4) if n_human > 0 else 0
    truck_vph = (n_truck / trip_duration * 3600 / 4) if n_truck > 0 else 0

    # Spawn vehicles continuously throughout training (matches sumocfg end time)
    sim_duration = 360000  # 100 hours of continuous spawning for soft reset training

    content = '''<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">

    <!-- Vehicle Types -->
    <vType id="ai_vehicle" vClass="passenger" color="0,0,255" accel="2.6" decel="4.5" sigma="0.0" length="4.5" maxSpeed="16.67" minGap="2.0"/>
    <vType id="human_vehicle" vClass="passenger" color="255,255,0" accel="2.0" decel="4.0" sigma="0.5" length="4.5" maxSpeed="16.67" minGap="2.5"/>
    <vType id="truck" vClass="truck" color="128,128,128" accel="1.3" decel="3.5" sigma="0.3" length="8.0" maxSpeed="13.89" minGap="3.0"/>

    <!-- Routes: All possible paths through intersection -->
    <route id="north_south" edges="north_to_center center_to_south"/>
    <route id="north_east" edges="north_to_center center_to_east"/>
    <route id="north_west" edges="north_to_center center_to_west"/>

    <route id="south_north" edges="south_to_center center_to_north"/>
    <route id="south_east" edges="south_to_center center_to_east"/>
    <route id="south_west" edges="south_to_center center_to_west"/>

    <route id="east_west" edges="east_to_center center_to_west"/>
    <route id="east_north" edges="east_to_center center_to_north"/>
    <route id="east_south" edges="east_to_center center_to_south"/>

    <route id="west_east" edges="west_to_center center_to_east"/>
    <route id="west_north" edges="west_to_center center_to_north"/>
    <route id="west_south" edges="west_to_center center_to_south"/>

'''

    # Add AI vehicle flows
    if ai_vph > 0:
        content += f'''    <!-- AI Vehicle Flows -->
    <flow id="ai_north" type="ai_vehicle" begin="0" end="{sim_duration}" vehsPerHour="{ai_vph:.2f}" departLane="best" departSpeed="max">
        <routeDistribution>
            <route probability="0.4" edges="north_to_center center_to_south"/>
            <route probability="0.3" edges="north_to_center center_to_east"/>
            <route probability="0.3" edges="north_to_center center_to_west"/>
        </routeDistribution>
    </flow>
    <flow id="ai_south" type="ai_vehicle" begin="0" end="{sim_duration}" vehsPerHour="{ai_vph:.2f}" departLane="best" departSpeed="max">
        <routeDistribution>
            <route probability="0.4" edges="south_to_center center_to_north"/>
            <route probability="0.3" edges="south_to_center center_to_east"/>
            <route probability="0.3" edges="south_to_center center_to_west"/>
        </routeDistribution>
    </flow>
    <flow id="ai_east" type="ai_vehicle" begin="0" end="{sim_duration}" vehsPerHour="{ai_vph:.2f}" departLane="best" departSpeed="max">
        <routeDistribution>
            <route probability="0.4" edges="east_to_center center_to_west"/>
            <route probability="0.3" edges="east_to_center center_to_north"/>
            <route probability="0.3" edges="east_to_center center_to_south"/>
        </routeDistribution>
    </flow>
    <flow id="ai_west" type="ai_vehicle" begin="0" end="{sim_duration}" vehsPerHour="{ai_vph:.2f}" departLane="best" departSpeed="max">
        <routeDistribution>
            <route probability="0.4" edges="west_to_center center_to_east"/>
            <route probability="0.3" edges="west_to_center center_to_north"/>
            <route probability="0.3" edges="west_to_center center_to_south"/>
        </routeDistribution>
    </flow>

'''

    # Add human vehicle flows
    if human_vph > 0:
        content += f'''    <!-- Human Vehicle Flows -->
    <flow id="human_north" type="human_vehicle" begin="0" end="{sim_duration}" vehsPerHour="{human_vph:.2f}" departLane="random" departSpeed="random">
        <routeDistribution>
            <route probability="0.4" edges="north_to_center center_to_south"/>
            <route probability="0.3" edges="north_to_center center_to_east"/>
            <route probability="0.3" edges="north_to_center center_to_west"/>
        </routeDistribution>
    </flow>
    <flow id="human_south" type="human_vehicle" begin="0" end="{sim_duration}" vehsPerHour="{human_vph:.2f}" departLane="random" departSpeed="random">
        <routeDistribution>
            <route probability="0.4" edges="south_to_center center_to_north"/>
            <route probability="0.3" edges="south_to_center center_to_east"/>
            <route probability="0.3" edges="south_to_center center_to_west"/>
        </routeDistribution>
    </flow>
    <flow id="human_east" type="human_vehicle" begin="0" end="{sim_duration}" vehsPerHour="{human_vph:.2f}" departLane="random" departSpeed="random">
        <routeDistribution>
            <route probability="0.4" edges="east_to_center center_to_west"/>
            <route probability="0.3" edges="east_to_center center_to_north"/>
            <route probability="0.3" edges="east_to_center center_to_south"/>
        </routeDistribution>
    </flow>
    <flow id="human_west" type="human_vehicle" begin="0" end="{sim_duration}" vehsPerHour="{human_vph:.2f}" departLane="random" departSpeed="random">
        <routeDistribution>
            <route probability="0.4" edges="west_to_center center_to_east"/>
            <route probability="0.3" edges="west_to_center center_to_north"/>
            <route probability="0.3" edges="west_to_center center_to_south"/>
        </routeDistribution>
    </flow>

'''

    # Add truck flows
    if truck_vph > 0:
        content += f'''    <!-- Truck Flows -->
    <flow id="truck_north" type="truck" begin="0" end="{sim_duration}" vehsPerHour="{truck_vph:.2f}" departLane="random" departSpeed="random">
        <routeDistribution>
            <route probability="0.5" edges="north_to_center center_to_south"/>
            <route probability="0.5" edges="north_to_center center_to_east"/>
        </routeDistribution>
    </flow>
    <flow id="truck_south" type="truck" begin="0" end="{sim_duration}" vehsPerHour="{truck_vph:.2f}" departLane="random" departSpeed="random">
        <routeDistribution>
            <route probability="0.5" edges="south_to_center center_to_north"/>
            <route probability="0.5" edges="south_to_center center_to_west"/>
        </routeDistribution>
    </flow>

'''

    content += '</routes>\n'

    with open(routes_file, 'w') as f:
        f.write(content)

    return routes_file


def generate_sumocfg(output_dir: str, network_file: str, routes_file: str) -> str:
    """Generate SUMO configuration file.

    Args:
        output_dir: Output directory
        network_file: Path to network XML
        routes_file: Path to routes XML

    Returns:
        Path to generated .sumocfg file
    """
    sumocfg_file = os.path.join(output_dir, "intersection.sumocfg")

    # Use relative paths in config
    net_rel = os.path.basename(network_file)
    rou_rel = os.path.basename(routes_file)

    content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="{net_rel}"/>
        <route-files value="{rou_rel}"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="360000"/>
        <step-length value="0.1"/>
    </time>

    <processing>
        <collision.action value="warn"/>
        <collision.check-junctions value="true"/>
    </processing>

    <report>
        <no-warnings value="true"/>
        <no-step-log value="true"/>
    </report>

</configuration>
'''

    with open(sumocfg_file, 'w') as f:
        f.write(content)

    return sumocfg_file


def main():
    parser = argparse.ArgumentParser(
        description='Generate SUMO intersection configuration for DMPO experiments'
    )
    parser.add_argument('--lanes', type=int, required=True, choices=[1, 2, 3, 4],
                       help='Number of lanes per direction')
    parser.add_argument('--vehicles', type=int, default=20,
                       help='Target number of vehicles (default: 20)')
    parser.add_argument('--ai-penetration', type=float, default=1.0,
                       help='AI penetration rate 0.0-1.0 (default: 1.0)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for SUMO files')
    parser.add_argument('--arm-length', type=int, default=600,
                       help='Road arm length in meters (default: 600)')
    parser.add_argument('--speed-limit', type=float, default=16.67,
                       help='Speed limit in m/s (default: 16.67 = 60 km/h)')
    parser.add_argument('--truck-percentage', type=float, default=0.02,
                       help='Truck percentage (default: 0.02 = 2%%)')

    args = parser.parse_args()

    # Validate arguments
    if args.ai_penetration < 0.0 or args.ai_penetration > 1.0:
        parser.error("ai-penetration must be between 0.0 and 1.0")

    if args.vehicles < 1:
        parser.error("vehicles must be at least 1")

    # Generate intersection
    sumocfg_file = generate_intersection(
        num_lanes=args.lanes,
        num_vehicles=args.vehicles,
        ai_penetration=args.ai_penetration,
        output_dir=args.output,
        arm_length=args.arm_length,
        speed_limit=args.speed_limit,
        truck_percentage=args.truck_percentage
    )

    print(f"🎉 Ready to use! Run with:")
    print(f"   sumo-gui -c {sumocfg_file}")
    print()


if __name__ == '__main__':
    main()
