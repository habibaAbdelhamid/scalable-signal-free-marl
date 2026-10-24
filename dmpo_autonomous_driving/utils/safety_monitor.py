"""
Safety Monitor for DMPO Intersection Environment

Monitors vehicle safety and provides safety constraints for actions.
Can override unsafe actions while preserving learning process.
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
from typing import Dict, List, Optional, Any
import math


class SafetyMonitor:
    """
    Monitors vehicle safety and enforces safety constraints.

    Provides:
    - Action filtering to prevent unsafe maneuvers
    - Collision detection and avoidance
    - Emergency braking capabilities
    - Safety metrics tracking

    Training mode: Less restrictive to allow policy learning from mistakes
    Evaluation mode: Full safety enforcement
    """

    def __init__(self, training_mode: bool = True):
        """Initialize safety monitor with constraints and thresholds.

        Args:
            training_mode: If True, use relaxed constraints to allow learning.
                          If False, use strict constraints for evaluation.
        """
        self.training_mode = training_mode

        # Safety thresholds - VERY RELAXED for training, STRICT for evaluation
        # Training needs to allow policy to make mistakes and learn from negative rewards
        # Too much safety filtering breaks credit assignment (policy sees modified actions)
        if training_mode:
            # Training: Minimal safety intervention - let policy learn from consequences
            # Only intervene for imminent collisions (< 0.5m or < 0.3s TTC)
            self.min_following_distance = 0.5  # meters - only very close
            self.critical_ttc = 0.3           # seconds - only critical TTC
            self.emergency_ttc = 0.2          # seconds - only emergency
            self.max_safe_speed = 30.0        # m/s - allow higher speeds
            self.min_gap_lane_change = 1.5    # meters - allow tighter gaps
        else:
            # Evaluation: Full safety
            self.min_following_distance = 1.5  # meters
            self.critical_ttc = 1.0           # seconds
            self.emergency_ttc = 0.5          # seconds
            self.max_safe_speed = 20.0        # m/s
            self.min_gap_lane_change = 3.0    # meters

        # Match the policy's braking capability (DMPO_MAX_DECEL). A residual 4.5
        # here under-models how hard the vehicle can actually brake, so the safe-
        # distance calc and the emergency floor were too weak — a known rear-end
        # cause. Tie both to DMPO_MAX_DECEL (default 4.5) so the V3 decel fix (8)
        # propagates through the safety layer too.
        import os as _os_sm
        _md = -abs(float(_os_sm.environ.get('DMPO_MAX_DECEL', '4.5')))
        self.emergency_decel = _md        # m/s² — matches policy braking cap
        self.max_decel = _md              # m/s² — matches policy braking cap
        self.max_accel = 2.6              # m/s² — matches vType

        # Safety state tracking
        self.collision_count = 0
        self.near_miss_count = 0
        self.emergency_brakes = 0
        self.safety_violations = {}

        # Vehicle safety history
        self.vehicle_history = {}
        
    def filter_action(self, vehicle_id: str, action: Dict[str, np.ndarray],
                     observation: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Filter action to ensure safety constraints.

        PHASE 3 FIX (RF2): In training mode, return the raw action unchanged.
        Filtering corrupts credit assignment — the policy never sees its own
        actions executed, so it can't learn from consequences. The collision
        penalty in reward_computer (-20) and SUMO's physics constraints
        (setSpeedMode bits 0+1+2) are sufficient signals.

        In evaluation mode (training_mode=False), filtering is fully active.

        Args:
            vehicle_id: Vehicle ID
            action: Raw action from DMPO
            observation: Current observation

        Returns:
            Safety-filtered action (training: identity, eval: full filter)
        """
        if self.training_mode:
            # PHASE 3: pass-through for credit assignment
            return action

        try:
            # Create filtered action copy
            filtered_action = {
                key: value.copy() for key, value in action.items()
            }
            
            # Apply longitudinal safety constraints
            filtered_action['longitudinal'] = self._filter_longitudinal_action(
                vehicle_id, filtered_action['longitudinal'][0]
            )
            
            # Apply lateral safety constraints  
            filtered_action['lateral'] = self._filter_lateral_action(
                vehicle_id, filtered_action['lateral'][0]
            )
            
            # Apply lane change safety constraints
            filtered_action['lane_intention'] = self._filter_lane_change_action(
                vehicle_id, filtered_action['lane_intention'][0]
            )
            
            # Check for emergency override
            if self._needs_emergency_intervention(vehicle_id):
                filtered_action = self._apply_emergency_override(vehicle_id)
                self.emergency_brakes += 1
            
            # Update safety history
            self._update_safety_history(vehicle_id, action, filtered_action)
            
            return {k: np.array([v]) for k, v in filtered_action.items()}
            
        except:
            # Return safe default action
            return {
                'longitudinal': np.array([0.0]),
                'lateral': np.array([0.0]), 
                'lane_intention': np.array([0.0])
            }
    
    def _filter_longitudinal_action(self, vehicle_id: str, action: float) -> float:
        """Filter longitudinal action for safety."""
        try:
            current_speed = traci.vehicle.getSpeed(vehicle_id)
            
            # Speed limit constraint
            if current_speed >= self.max_safe_speed and action > 0:
                action = 0.0
            
            # Following distance constraint
            leader_info = traci.vehicle.getLeader(vehicle_id)
            if leader_info is not None:
                _, distance = leader_info
                
                if distance < self.min_following_distance:
                    # Force deceleration if too close
                    action = min(action, -0.5)
                elif distance < self.min_following_distance * 2:
                    # Limit acceleration if close
                    action = min(action, 0.0)
            
            # Time-to-collision constraint
            ttc = self._calculate_ttc(vehicle_id)
            if ttc is not None:
                if ttc < self.critical_ttc:
                    # Force strong deceleration
                    action = min(action, -0.8)
                elif ttc < self.critical_ttc * 2:
                    # Force moderate deceleration
                    action = min(action, -0.3)
            
            # Physical limits
            action = np.clip(action, -1.0, 1.0)

            return float(action)

        except Exception:
            return 0.0

    def _filter_lateral_action(self, vehicle_id: str, action: float) -> float:
        """Filter lateral action for safety."""
        try:
            # Limit steering at high speeds
            current_speed = traci.vehicle.getSpeed(vehicle_id)
            if current_speed > 15.0:  # High speed
                max_steering = 0.3
                action = np.clip(action, -max_steering, max_steering)
            
            # Check for safe lateral space
            if not self._has_safe_lateral_space(vehicle_id, action):
                action = 0.0  # No steering if unsafe
            
            return float(action)
            
        except:
            return 0.0
    
    def _filter_lane_change_action(self, vehicle_id: str, action: float) -> float:
        """Filter lane change action for safety."""
        try:
            # Only allow lane changes if safe
            if abs(action) > 0.3:  # Lane change intended
                if not self._is_lane_change_safe(vehicle_id, action):
                    action = 0.0  # Block unsafe lane change
            
            return float(action)
            
        except:
            return 0.0
    
    def _needs_emergency_intervention(self, vehicle_id: str) -> bool:
        """Check if emergency intervention is needed."""
        try:
            # Immediate collision risk
            if self._is_immediate_collision_risk(vehicle_id):
                return True
            
            # Critical time-to-collision
            ttc = self._calculate_ttc(vehicle_id)
            if ttc is not None and ttc < self.emergency_ttc:
                return True
            
            # Very close following distance
            leader_info = traci.vehicle.getLeader(vehicle_id)
            if leader_info is not None:
                _, distance = leader_info
                if distance < 1.0:  # Less than 1 meter
                    return True
            
            return False
            
        except:
            return False
    
    def _apply_emergency_override(self, vehicle_id: str) -> Dict[str, float]:
        """Apply emergency action override."""
        return {
            'longitudinal': -1.0,  # Maximum deceleration
            'lateral': 0.0,        # No steering
            'lane_intention': 0.0  # No lane change
        }
    
    def _calculate_ttc(self, vehicle_id: str) -> Optional[float]:
        """Calculate time to collision with leader."""
        try:
            leader_info = traci.vehicle.getLeader(vehicle_id)
            if leader_info is None:
                return None
                
            leader_id, distance = leader_info
            if distance <= 0:
                return None
            
            ego_speed = traci.vehicle.getSpeed(vehicle_id)
            leader_speed = traci.vehicle.getSpeed(leader_id)
            
            relative_speed = ego_speed - leader_speed
            
            if relative_speed <= 0:  # Not approaching
                return None
            
            return distance / relative_speed
            
        except:
            return None
    
    def _is_immediate_collision_risk(self, vehicle_id: str) -> bool:
        """Check for immediate collision risk."""
        try:
            position = traci.vehicle.getPosition(vehicle_id)
            
            # Check nearby vehicles for overlap risk
            for other_id in traci.vehicle.getIDList():
                if other_id == vehicle_id:
                    continue
                    
                other_pos = traci.vehicle.getPosition(other_id)
                distance = math.sqrt(
                    (position[0] - other_pos[0])**2 + 
                    (position[1] - other_pos[1])**2
                )
                
                if distance < 2.0:  # Very close
                    # Check if on collision course
                    if self._are_on_collision_course(vehicle_id, other_id):
                        return True
            
            return False
            
        except:
            return False
    
    def _are_on_collision_course(self, vehicle1: str, vehicle2: str) -> bool:
        """Check if two vehicles are on collision course."""
        try:
            # Get positions and velocities
            pos1 = traci.vehicle.getPosition(vehicle1)
            pos2 = traci.vehicle.getPosition(vehicle2)
            
            speed1 = traci.vehicle.getSpeed(vehicle1)
            speed2 = traci.vehicle.getSpeed(vehicle2)
            
            angle1 = traci.vehicle.getAngle(vehicle1)
            angle2 = traci.vehicle.getAngle(vehicle2)
            
            # Calculate velocity vectors
            # M5 FIX: SUMO's angle convention is 0deg=NORTH, 90deg=EAST, CLOCKWISE
            # => vx = v*sin(a), vy = v*cos(a). The previous code used the MATH
            # convention (0=East, CCW: vx=cos, vy=sin), which reflects every heading
            # across the NE diagonal (north<->east, south<->west) and made this
            # collision-course extrapolation wrong for every vehicle. Matches
            # observation_processor.py (nb_abs_vx/vy) which had it right.
            vel1_x = speed1 * math.sin(math.radians(angle1))
            vel1_y = speed1 * math.cos(math.radians(angle1))
            vel2_x = speed2 * math.sin(math.radians(angle2))
            vel2_y = speed2 * math.cos(math.radians(angle2))
            
            # Predict positions in 1 second
            future_pos1 = (pos1[0] + vel1_x, pos1[1] + vel1_y)
            future_pos2 = (pos2[0] + vel2_x, pos2[1] + vel2_y)
            
            # Check if future positions are close
            future_distance = math.sqrt(
                (future_pos1[0] - future_pos2[0])**2 + 
                (future_pos1[1] - future_pos2[1])**2
            )
            
            return future_distance < 3.0
            
        except:
            return False
    
    def _has_safe_lateral_space(self, vehicle_id: str, steering_action: float) -> bool:
        """Check if there's safe lateral space for steering."""
        try:
            if abs(steering_action) < 0.1:  # No significant steering
                return True
            
            # Get vehicle dimensions and position
            position = traci.vehicle.getPosition(vehicle_id)
            angle = traci.vehicle.getAngle(vehicle_id)
            
            # Check for lateral conflicts
            for other_id in traci.vehicle.getIDList():
                if other_id == vehicle_id:
                    continue
                
                other_pos = traci.vehicle.getPosition(other_id)
                distance = math.sqrt(
                    (position[0] - other_pos[0])**2 + 
                    (position[1] - other_pos[1])**2
                )
                
                if distance < 10.0:  # Close vehicles
                    # Check lateral separation
                    lateral_distance = self._calculate_lateral_distance(
                        position, other_pos, angle
                    )
                    
                    if abs(lateral_distance) < 4.0:  # Too close laterally
                        return False
            
            return True
            
        except:
            return False
    
    def _calculate_lateral_distance(self, pos1: tuple, pos2: tuple, angle: float) -> float:
        """Calculate lateral distance between vehicles."""
        # Vector from pos1 to pos2
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        # Rotate to vehicle coordinate system
        # M5 FIX: SUMO angle 0deg=N clockwise => heading unit vector is (sin a, cos a)
        # and the RIGHT-hand (lateral) unit vector is (cos a, -sin a). The old formula
        # (-dx*sin + dy*cos) is the projection onto the HEADING under this convention,
        # i.e. it returned longitudinal distance labelled as lateral.
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))

        # Lateral component = projection onto the vehicle's right-hand axis
        lateral = dx * cos_a - dy * sin_a
        return lateral
    
    def _is_lane_change_safe(self, vehicle_id: str, lane_intention: float) -> bool:
        """Check if lane change is safe."""
        try:
            current_lane_idx = traci.vehicle.getLaneIndex(vehicle_id)
            edge_id = traci.lane.getEdgeID(traci.vehicle.getLaneID(vehicle_id))
            num_lanes = traci.edge.getLaneNumber(edge_id)
            
            # Determine target lane
            # SUMO lane indexing: 0 = rightmost lane, higher index = more left
            if lane_intention < -0.3:
                # Move left (higher lane index in SUMO)
                target_lane = current_lane_idx + 1
            elif lane_intention > 0.3:
                # Move right (lower lane index in SUMO)
                target_lane = current_lane_idx - 1
            else:
                return True
            
            # Check lane bounds
            if target_lane < 0 or target_lane >= num_lanes:
                return False
            
            # Check for vehicles in target lane
            target_lane_id = f"{edge_id}_{target_lane}"
            vehicles_in_lane = traci.lane.getLastStepVehicleIDs(target_lane_id)
            
            ego_pos = traci.vehicle.getLanePosition(vehicle_id)
            ego_speed = traci.vehicle.getSpeed(vehicle_id)
            
            for other_veh in vehicles_in_lane:
                other_pos = traci.vehicle.getLanePosition(other_veh)
                other_speed = traci.vehicle.getSpeed(other_veh)
                
                # Check gap requirements
                gap = abs(other_pos - ego_pos)
                relative_speed = ego_speed - other_speed
                
                required_gap = max(self.min_gap_lane_change, 
                                 abs(relative_speed) * 2.0)
                
                if gap < required_gap:
                    return False
            
            return True
            
        except:
            return False
    
    def _update_safety_history(self, vehicle_id: str, original_action: Dict,
                              filtered_action: Dict):
        """Update safety history for vehicle."""
        if vehicle_id not in self.vehicle_history:
            self.vehicle_history[vehicle_id] = {
                'interventions': 0,
                'last_intervention': None
            }
        
        # Check if action was modified
        action_modified = False
        for key in original_action:
            if abs(original_action[key][0] - filtered_action[key]) > 0.01:
                action_modified = True
                break
        
        if action_modified:
            self.vehicle_history[vehicle_id]['interventions'] += 1
            self.vehicle_history[vehicle_id]['last_intervention'] = traci.simulation.getTime()
    
    def check_collision_occurred(self) -> bool:
        """Check if a collision has occurred."""
        try:
            # Simple collision detection based on vehicle overlap
            vehicles = list(traci.vehicle.getIDList())
            
            for i, veh1 in enumerate(vehicles):
                pos1 = traci.vehicle.getPosition(veh1)
                
                for veh2 in vehicles[i+1:]:
                    pos2 = traci.vehicle.getPosition(veh2)
                    distance = math.sqrt(
                        (pos1[0] - pos2[0])**2 + 
                        (pos1[1] - pos2[1])**2
                    )
                    
                    if distance < 1.0:  # Collision threshold
                        self.collision_count += 1
                        return True
            
            return False
            
        except:
            return False
    
    def check_near_miss(self, vehicle_id: str) -> bool:
        """Check if vehicle had a near miss."""
        try:
            ttc = self._calculate_ttc(vehicle_id)
            if ttc is not None and ttc < 2.0:
                self.near_miss_count += 1
                return True
            return False
        except:
            return False
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get safety performance metrics."""
        return {
            'collision_count': self.collision_count,
            'near_miss_count': self.near_miss_count,
            'emergency_brakes': self.emergency_brakes,
            'total_interventions': sum(
                history['interventions'] 
                for history in self.vehicle_history.values()
            ),
            'vehicles_with_interventions': len(self.vehicle_history)
        }
    
    def get_collision_count(self) -> int:
        """Get total collision count."""
        return self.collision_count
    
    def reset_metrics(self):
        """Reset safety metrics."""
        self.collision_count = 0
        self.near_miss_count = 0
        self.emergency_brakes = 0
        self.vehicle_history.clear()
        self.safety_violations.clear()
    
    def is_vehicle_safe(self, vehicle_id: str) -> bool:
        """Check if vehicle is currently in safe state."""
        try:
            # Check basic safety conditions
            if self._is_immediate_collision_risk(vehicle_id):
                return False
            
            ttc = self._calculate_ttc(vehicle_id)
            if ttc is not None and ttc < 2.0:
                return False
            
            leader_info = traci.vehicle.getLeader(vehicle_id)
            if leader_info is not None:
                _, distance = leader_info
                if distance < self.min_following_distance:
                    return False
            
            return True
            
        except:
            return True  # Default to safe if unsure