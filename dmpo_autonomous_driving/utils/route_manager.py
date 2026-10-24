"""
Route Manager for DMPO Intersection Environment

Manages vehicle routing, navigation goals, and route planning for intersection scenarios.
"""

import numpy as np
import os

# SUMO interface selection - check environment variable
if os.environ.get('DMPO_USE_TRACI', '0') == '1':
    import traci
else:
    try:
        import libsumo as traci
    except ImportError:
        import traci
from typing import Dict, List, Tuple, Optional
import random


class RouteManager:
    """
    Manages vehicle routes and navigation for intersection scenarios.
    
    Provides:
    - Route assignment and management
    - Navigation goal tracking
    - Route efficiency metrics
    - Dynamic rerouting capabilities
    """
    
    def __init__(self):
        """Initialize route manager."""
        # Available routes for 4-lane intersection
        self.available_routes = {
            # Straight movements
            'north_to_south': ['north_to_center', 'center_to_south'],
            'south_to_north': ['south_to_center', 'center_to_north'],
            'east_to_west': ['east_to_center', 'center_to_west'],
            'west_to_east': ['west_to_center', 'center_to_east'],
            
            # Left turns
            'north_to_west': ['north_to_center', 'center_to_west'],
            'north_to_east': ['north_to_center', 'center_to_east'],
            'south_to_west': ['south_to_center', 'center_to_west'],
            'south_to_east': ['south_to_center', 'center_to_east'],
            'east_to_north': ['east_to_center', 'center_to_north'],
            'east_to_south': ['east_to_center', 'center_to_south'],
            'west_to_north': ['west_to_center', 'center_to_north'],
            'west_to_south': ['west_to_center', 'center_to_south']
        }
        
        # Route priorities (higher = more priority)
        self.route_priorities = {
            'north_to_south': 1.0,
            'south_to_north': 1.0,
            'east_to_west': 1.0,
            'west_to_east': 1.0,
            'north_to_west': 0.8,
            'north_to_east': 0.8,
            'south_to_west': 0.8,
            'south_to_east': 0.8,
            'east_to_north': 0.8,
            'east_to_south': 0.8,
            'west_to_north': 0.8,
            'west_to_south': 0.8
        }
        
        # Vehicle route tracking
        self.vehicle_routes = {}
        self.route_completion_times = {}
        
    def assign_route(self, vehicle_id: str) -> Optional[str]:
        """
        Assign route to vehicle based on current position.
        
        Args:
            vehicle_id: Vehicle ID
            
        Returns:
            Route name or None if cannot determine
        """
        try:
            current_route = traci.vehicle.getRouteID(vehicle_id)
            if current_route in self.available_routes:
                self.vehicle_routes[vehicle_id] = current_route
                return current_route
            
            # Try to infer route from current edges
            route_edges = traci.vehicle.getRoute(vehicle_id)
            inferred_route = self._infer_route_from_edges(route_edges)
            
            if inferred_route:
                self.vehicle_routes[vehicle_id] = inferred_route
                return inferred_route
            
            return None
            
        except:
            return None
    
    def _infer_route_from_edges(self, edges: List[str]) -> Optional[str]:
        """Infer route name from edge sequence."""
        edge_str = '_'.join(edges)
        
        for route_name, route_edges in self.available_routes.items():
            if all(edge in edge_str for edge in route_edges):
                return route_name
                
        return None
    
    def get_route_progress(self, vehicle_id: str) -> float:
        """
        Get route completion progress (0.0 to 1.0).
        
        Args:
            vehicle_id: Vehicle ID
            
        Returns:
            Progress ratio
        """
        try:
            route = traci.vehicle.getRoute(vehicle_id)
            route_index = traci.vehicle.getRouteIndex(vehicle_id)
            
            if len(route) == 0:
                return 0.0
                
            return route_index / len(route)
            
        except:
            return 0.0
    
    def get_remaining_distance(self, vehicle_id: str) -> float:
        """
        Get remaining distance to destination.
        
        Args:
            vehicle_id: Vehicle ID
            
        Returns:
            Remaining distance in meters
        """
        try:
            route = traci.vehicle.getRoute(vehicle_id)
            route_index = traci.vehicle.getRouteIndex(vehicle_id)
            current_position = traci.vehicle.getLanePosition(vehicle_id)
            
            remaining_distance = 0.0
            
            # Add remaining distance on current edge
            if route_index < len(route):
                current_edge = route[route_index]
                edge_length = traci.edge.getLength(current_edge)
                remaining_distance += edge_length - current_position
            
            # Add length of remaining edges
            for i in range(route_index + 1, len(route)):
                edge_length = traci.edge.getLength(route[i])
                remaining_distance += edge_length
            
            return remaining_distance
            
        except:
            return 0.0
    
    def get_route_direction_at_intersection(self, vehicle_id: str) -> str:
        """
        Get intended direction at intersection.
        
        Args:
            vehicle_id: Vehicle ID
            
        Returns:
            Direction: 'straight', 'left', 'right', or 'unknown'
        """
        try:
            route = traci.vehicle.getRoute(vehicle_id)
            route_index = traci.vehicle.getRouteIndex(vehicle_id)
            
            if route_index >= len(route) - 1:
                return 'unknown'
            
            current_edge = route[route_index]
            next_edge = route[route_index + 1]
            
            return self._determine_turn_direction(current_edge, next_edge)
            
        except:
            return 'unknown'
    
    def _determine_turn_direction(self, current_edge: str, next_edge: str) -> str:
        """Determine turn direction based on edge transition."""
        # Extract direction from edge names
        current_dir = self._extract_direction(current_edge)
        next_dir = self._extract_direction(next_edge)
        
        if current_dir == 'unknown' or next_dir == 'unknown':
            return 'unknown'
        
        # Determine turn based on direction change
        direction_map = {'north': 0, 'east': 1, 'south': 2, 'west': 3}
        
        if current_dir in direction_map and next_dir in direction_map:
            current_idx = direction_map[current_dir]
            next_idx = direction_map[next_dir]
            
            diff = (next_idx - current_idx) % 4
            
            if diff == 0 or diff == 2:  # Same or opposite
                return 'straight'
            elif diff == 1:  # Clockwise
                return 'right'
            elif diff == 3:  # Counter-clockwise
                return 'left'
        
        return 'unknown'
    
    def _extract_direction(self, edge_name: str) -> str:
        """Extract direction from edge name."""
        edge_lower = edge_name.lower()
        
        if 'north' in edge_lower:
            return 'north'
        elif 'south' in edge_lower:
            return 'south'
        elif 'east' in edge_lower:
            return 'east'
        elif 'west' in edge_lower:
            return 'west'
        else:
            return 'unknown'
    
    def is_approaching_intersection(self, vehicle_id: str, distance_threshold: float = 50.0) -> bool:
        """
        Check if vehicle is approaching intersection.
        
        Args:
            vehicle_id: Vehicle ID
            distance_threshold: Distance threshold in meters
            
        Returns:
            True if approaching intersection
        """
        try:
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            lane_length = traci.lane.getLength(lane_id)
            vehicle_position = traci.vehicle.getLanePosition(vehicle_id)
            
            distance_to_end = lane_length - vehicle_position
            
            # Check if approaching intersection center
            return distance_to_end <= distance_threshold and 'center' not in lane_id
            
        except:
            return False
    
    def is_in_intersection(self, vehicle_id: str) -> bool:
        """Check if vehicle is currently in intersection."""
        try:
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            return 'center' in lane_id
        except:
            return False
    
    def has_completed_route(self, vehicle_id: str) -> bool:
        """Check if vehicle has completed its route."""
        try:
            route = traci.vehicle.getRoute(vehicle_id)
            route_index = traci.vehicle.getRouteIndex(vehicle_id)
            
            return route_index >= len(route) - 1
            
        except:
            return False
    
    def get_route_efficiency_score(self, vehicle_id: str) -> float:
        """
        Calculate route efficiency score (0.0 to 1.0).
        
        Args:
            vehicle_id: Vehicle ID
            
        Returns:
            Efficiency score
        """
        try:
            if vehicle_id not in self.vehicle_routes:
                return 0.5  # Neutral score
            
            route_name = self.vehicle_routes[vehicle_id]
            
            # Base efficiency from route priority
            base_efficiency = self.route_priorities.get(route_name, 0.5)
            
            # Adjust for progress speed
            progress = self.get_route_progress(vehicle_id)
            travel_time = traci.simulation.getTime()  # Simplified
            
            if travel_time > 0:
                progress_rate = progress / travel_time
                # Normalize progress rate (higher is better)
                progress_bonus = min(progress_rate * 100, 0.3)
            else:
                progress_bonus = 0.0
            
            efficiency_score = base_efficiency + progress_bonus
            return np.clip(efficiency_score, 0.0, 1.0)
            
        except:
            return 0.5
    
    def get_optimal_speed_for_intersection(self, vehicle_id: str) -> float:
        """
        Get optimal speed for intersection approach.
        
        Args:
            vehicle_id: Vehicle ID
            
        Returns:
            Optimal speed in m/s
        """
        try:
            if self.is_in_intersection(vehicle_id):
                # In intersection - maintain steady speed
                return 10.0
            elif self.is_approaching_intersection(vehicle_id):
                # Approaching intersection - check traffic light
                return self._get_traffic_light_optimal_speed(vehicle_id)
            else:
                # Normal driving - use speed limit
                lane_id = traci.vehicle.getLaneID(vehicle_id)
                return traci.lane.getMaxSpeed(lane_id)
                
        except:
            return 10.0  # Default speed
    
    def _get_traffic_light_optimal_speed(self, vehicle_id: str) -> float:
        """Get optimal speed based on traffic light timing."""
        try:
            tl_ids = traci.trafficlight.getIDList()
            if not tl_ids:
                return 10.0
            
            tl_id = tl_ids[0]
            
            # Get current phase and timing
            current_phase = traci.trafficlight.getPhase(tl_id)
            next_switch = traci.trafficlight.getNextSwitch(tl_id)
            current_time = traci.simulation.getTime()
            time_to_switch = next_switch - current_time
            
            # Get distance to intersection
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            lane_length = traci.lane.getLength(lane_id)
            vehicle_pos = traci.vehicle.getLanePosition(vehicle_id)
            distance_to_intersection = lane_length - vehicle_pos
            
            # Calculate optimal speed to reach intersection at good time
            if time_to_switch > 0 and distance_to_intersection > 0:
                optimal_speed = distance_to_intersection / time_to_switch
                return np.clip(optimal_speed, 5.0, 15.0)
            
            return 10.0
            
        except:
            return 10.0
    
    def should_yield_at_intersection(self, vehicle_id: str) -> bool:
        """
        Check if vehicle should yield at intersection.
        
        Args:
            vehicle_id: Vehicle ID
            
        Returns:
            True if should yield
        """
        try:
            # Check traffic light
            if self._is_at_red_light(vehicle_id):
                return True
            
            # Check for priority traffic (simplified)
            if self._has_higher_priority_traffic(vehicle_id):
                return True
            
            return False
            
        except:
            return False
    
    def _is_at_red_light(self, vehicle_id: str) -> bool:
        """Check if vehicle is at red light."""
        try:
            tl_ids = traci.trafficlight.getIDList()
            if not tl_ids:
                return False
            
            tl_id = tl_ids[0]
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            current_lane = traci.vehicle.getLaneID(vehicle_id)
            
            if current_lane in controlled_lanes:
                state = traci.trafficlight.getRedYellowGreenState(tl_id)
                lane_index = controlled_lanes.index(current_lane)
                
                if lane_index < len(state):
                    return state[lane_index].lower() == 'r'
            
            return False
            
        except:
            return False
    
    def _has_higher_priority_traffic(self, vehicle_id: str) -> bool:
        """Check for higher priority traffic at intersection."""
        try:
            if not self.is_approaching_intersection(vehicle_id):
                return False
            
            ego_route = self.vehicle_routes.get(vehicle_id, '')
            ego_priority = self.route_priorities.get(ego_route, 0.5)
            
            # Check other vehicles at intersection
            for other_id in traci.vehicle.getIDList():
                if other_id == vehicle_id:
                    continue
                
                if self.is_approaching_intersection(other_id) or self.is_in_intersection(other_id):
                    other_route = self.vehicle_routes.get(other_id, '')
                    other_priority = self.route_priorities.get(other_route, 0.5)
                    
                    # Check if other vehicle has higher priority
                    if other_priority > ego_priority:
                        # Check if paths conflict
                        if self._routes_conflict(ego_route, other_route):
                            return True
            
            return False
            
        except:
            return False
    
    def _routes_conflict(self, route1: str, route2: str) -> bool:
        """Check if two routes conflict at intersection."""
        if not route1 or not route2:
            return False
        
        # Get route directions
        route1_parts = route1.split('_to_')
        route2_parts = route2.split('_to_')
        
        if len(route1_parts) != 2 or len(route2_parts) != 2:
            return False
        
        # Check for crossing paths (simplified)
        route1_from, route1_to = route1_parts
        route2_from, route2_to = route2_parts
        
        # Opposing directions conflict
        if ((route1_from == 'north' and route1_to == 'south' and 
             route2_from == 'south' and route2_to == 'north') or
            (route1_from == 'east' and route1_to == 'west' and 
             route2_from == 'west' and route2_to == 'east')):
            return True
        
        # Crossing paths conflict
        crossing_pairs = [
            (['north', 'south'], ['east', 'west']),
            (['east', 'west'], ['north', 'south'])
        ]
        
        for pair1, pair2 in crossing_pairs:
            if ((route1_from in pair1 and route1_to in pair1) and
                (route2_from in pair2 and route2_to in pair2)):
                return True
        
        return False
    
    def get_route_statistics(self) -> Dict:
        """Get route management statistics."""
        return {
            'active_routes': len(self.vehicle_routes),
            'completed_routes': len(self.route_completion_times),
            'route_distribution': self._get_route_distribution(),
            'average_completion_time': self._get_average_completion_time()
        }
    
    def _get_route_distribution(self) -> Dict[str, int]:
        """Get distribution of active routes."""
        distribution = {}
        for route in self.vehicle_routes.values():
            distribution[route] = distribution.get(route, 0) + 1
        return distribution
    
    def _get_average_completion_time(self) -> float:
        """Get average route completion time."""
        if not self.route_completion_times:
            return 0.0
        return np.mean(list(self.route_completion_times.values()))
    
    def cleanup_completed_vehicles(self):
        """Clean up data for vehicles that have left the simulation."""
        current_vehicles = set(traci.vehicle.getIDList())
        
        # Remove data for vehicles no longer in simulation
        vehicles_to_remove = []
        for vehicle_id in self.vehicle_routes.keys():
            if vehicle_id not in current_vehicles:
                vehicles_to_remove.append(vehicle_id)
        
        for vehicle_id in vehicles_to_remove:
            if vehicle_id in self.vehicle_routes:
                del self.vehicle_routes[vehicle_id]
            if vehicle_id in self.route_completion_times:
                del self.route_completion_times[vehicle_id]