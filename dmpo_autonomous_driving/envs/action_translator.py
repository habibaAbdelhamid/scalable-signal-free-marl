"""
Action Translator for DMPO Intersection Environment

Translates DMPO continuous actions to SUMO vehicle control commands
for intersection navigation.
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
from typing import Dict, Any, Optional
import math


class ActionTranslator:
    """
    Translates DMPO actions to SUMO vehicle commands.
    
    Converts normalized continuous actions to:
    - Vehicle acceleration/deceleration
    - Lane change maneuvers
    - Intersection navigation commands
    """
    
    def __init__(self):
        """Initialize action translator with driving constraints.

        PHASE 1 FIX: All limits MUST match SUMO vType values, otherwise the
        policy never sees its actions executed verbatim. SUMO clamps to vType
        when setSpeedMode bits are set.

        SUMO vType ai_vehicle: accel=2.6, decel=4.5, maxSpeed=16.67
        """
        # PHASE 1 FIX: Match SUMO vType exactly (was max_accel=5.0, max_decel=-8.0)
        self.max_accel = 2.6     # m/s² — matches vType
        # DMPO_MAX_DECEL: env-var override for the braking cap (magnitude, m/s²).
        # Default 4.5 = vType decel. Eval-only ablation: raise to e.g. 8.0 to test
        # whether collisions are caused by insufficient deceleration. NOTE: SUMO
        # also enforces vType decel / emergencyDecel, so the scenario vType decel
        # (and --default.emergencydecel) must allow the higher value to take effect.
        self.max_decel = -abs(float(os.environ.get('DMPO_MAX_DECEL', '4.5')))
        # DMPO_MAX_SPEED_MPS: env-var override for the policy's commanded-speed cap.
        # Default 16.67 m/s = 60 km/h (matches the trained-policy distribution).
        # Bumping this for eval experiments (e.g. 27.78 = 100 km/h) is OUT OF
        # DISTRIBUTION for the trained policy — it may behave erratically.
        self.max_speed = float(os.environ.get('DMPO_MAX_SPEED_MPS', '16.67'))
        self.min_speed = 0.0     # m/s

        # Lane change parameters
        self.lane_change_duration = 3.0  # seconds
        self.min_gap = 2.0              # minimum gap for lane changes

        # Action smoothing — PHASE 1 FIX: disabled for credit assignment
        # Was 0.3 (70% old + 30% new) which broke credit assignment.
        # Policy must learn smoothness via comfort reward, not via action mutation.
        self.prev_actions = {}
        self.smoothing_factor = 0.0

        # Emergency constraints (only used if _apply_safety_constraints is called).
        # Tie the emergency floor to DMPO_MAX_DECEL so the <1m forced brake is as
        # hard as the vehicle can actually brake (V3 decel fix), not a stale 4.5.
        self.emergency_decel = -abs(float(os.environ.get('DMPO_MAX_DECEL', '4.5')))  # m/s²
        self.safety_time_headway = 1.5  # seconds
    
    def translate(self, vehicle_id: str, action: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Translate DMPO action to SUMO commands.
        
        Args:
            vehicle_id: ID of the vehicle
            action: Dictionary with 'longitudinal', 'lateral', 'lane_intention' actions
            
        Returns:
            Dictionary with SUMO commands
        """
        # P0.9: validate DMPO_ACTION_VARIANT BEFORE the traci try-block so
        # the ValueError propagates to the caller (not silently caught by
        # the TraCIException handler). Previously typos (e.g. 'throttle-brake'
        # with a hyphen) silently fell through to 'default', causing a paper
        # experiment to run the wrong variant.
        variant = os.environ.get('DMPO_ACTION_VARIANT', 'default')
        _VALID_VARIANTS = ('default', 'throttle_brake', 'yield_intent')
        if variant not in _VALID_VARIANTS:
            raise ValueError(
                f"Unknown DMPO_ACTION_VARIANT={variant!r}; expected one of "
                f"{_VALID_VARIANTS}. Check launcher script env vars."
            )

        try:
            # Get current vehicle state
            current_speed = traci.vehicle.getSpeed(vehicle_id)
            current_lane = traci.vehicle.getLaneID(vehicle_id)

            # Extract longitudinal command per variant.
            #  default / yield_intent: single 'longitudinal' dim in [-1, 1].
            #  throttle_brake: separate (throttle, brake) ∈ [-1, 1] each. After
            #    tanh, throttle>0 => accel, brake>0 => decel. Negatives are
            #    interpreted as zero (rectified). Combined longitudinal is
            #    used downstream so the existing scaling pipeline applies.
            if variant == 'throttle_brake':
                throttle = float(action.get('throttle', [0.0])[0])
                brake = float(action.get('brake', [0.0])[0])
                throttle_pos = max(throttle, 0.0)
                brake_pos = max(brake, 0.0)
                # Map to [-1, 1] in the same scale as the default longitudinal:
                # accel max corresponds to longitudinal=+1; decel max to -1.
                longitudinal = throttle_pos - brake_pos
            else:
                longitudinal = float(action.get('longitudinal', [0.0])[0])

            lateral = float(action.get('lateral', [0.0])[0])
            lane_intention = float(action.get('lane_intention', [0.0])[0])

            # Apply action smoothing
            longitudinal = self._smooth_action(vehicle_id, 'longitudinal', longitudinal)
            lateral = self._smooth_action(vehicle_id, 'lateral', lateral)

            # Translate longitudinal action to acceleration
            target_accel = self._translate_longitudinal(vehicle_id, longitudinal, current_speed)

            # Translate lateral action to lane change
            lane_change_command = self._translate_lateral(
                vehicle_id, lateral, lane_intention, current_lane
            )

            # Build SUMO command
            sumo_command = {
                'acceleration': target_accel
            }

            # Add lane change if requested
            if lane_change_command is not None:
                sumo_command['lane_change'] = lane_change_command

            # Variant C: pass yield_intent through so the env can apply a
            # per-vehicle Krauss-cap. yield_intent in [-1, 1]; active when > 0
            # (after tanh, "is the policy choosing to yield this step?").
            if variant == 'yield_intent':
                yi = float(action.get('yield_intent', [0.0])[0])
                sumo_command['yield_intent_active'] = yi > 0.0

            return sumo_command
            
        except traci.exceptions.TraCIException:
            # Vehicle not found, return safe default
            return {'acceleration': 0.0}
    
    def _smooth_action(self, vehicle_id: str, action_type: str, current_action: float) -> float:
        """Apply temporal smoothing to actions.

        BUG FIX (B12): The original formula was inverted:
          smoothed = (1 - factor) * prev + factor * current
        With factor=0.0 this returned 100% prev, ignoring current entirely.
        Every vehicle's action was locked to its first-ever action forever.

        Fixed formula:
          smoothed = (1 - factor) * current + factor * prev
        With factor=0.0: returns current (no smoothing). Correct.
        With factor=0.5: 50/50 blend. Also correct.
        """
        if vehicle_id not in self.prev_actions:
            self.prev_actions[vehicle_id] = {}

        if action_type in self.prev_actions[vehicle_id]:
            prev_action = self.prev_actions[vehicle_id][action_type]
            smoothed = (1 - self.smoothing_factor) * current_action + \
                      self.smoothing_factor * prev_action
        else:
            smoothed = current_action

        self.prev_actions[vehicle_id][action_type] = smoothed
        return smoothed
    
    def _translate_longitudinal(self, vehicle_id: str, longitudinal_action: float, 
                              current_speed: float) -> float:
        """
        Translate longitudinal action to acceleration.
        
        Args:
            vehicle_id: Vehicle ID
            longitudinal_action: Normalized action [-1, 1]
            current_speed: Current vehicle speed
            
        Returns:
            Target acceleration in m/s²
        """
        # Map normalized action to acceleration range
        if longitudinal_action > 0:
            target_accel = longitudinal_action * self.max_accel
        else:
            target_accel = longitudinal_action * abs(self.max_decel)

        # Apply speed constraints
        if current_speed >= self.max_speed and target_accel > 0:
            target_accel = 0.0  # Don't accelerate beyond max speed
        elif current_speed <= self.min_speed and target_accel < 0:
            target_accel = 0.0  # Don't decelerate below min speed

        # PHASE 1 FIX: Removed _apply_safety_constraints call.
        # SafetyMonitor.filter_action handles safety filtering; double-filtering
        # broke credit assignment.

        # Clamp to physical limits
        target_accel = np.clip(target_accel, self.max_decel, self.max_accel)

        return float(target_accel)
    
    def _translate_lateral(self, vehicle_id: str, lateral_action: float,
                         lane_intention: float, current_lane: str) -> Optional[float]:
        """
        Translate lateral action to continuous sublane offset.

        With SUMO's sublane model (--lateral-resolution), vehicles can move
        continuously within and across lanes using changeSublane(latDist).

        Args:
            vehicle_id: Vehicle ID
            lateral_action: Steering input [-1, 1], maps to lateral velocity
            lane_intention: Lane change intention [-1, 1] (combined with lateral)
            current_lane: Current lane ID

        Returns:
            Lateral distance offset in meters, or None if negligible
        """
        # Combine lateral action and lane intention for smooth control
        combined = 0.7 * lateral_action + 0.3 * lane_intention

        # P1.10: removed the 0.02 deadband. At converged policy std it fired up
        # to 65% of steps, silently zeroing lateral commands the policy emitted
        # — broke credit assignment for small steering corrections. SUMO
        # handles tiny lateral input fine; no filtering needed at this layer.

        # P1.11: lateral offset is now derived programmatically from
        # DMPO_MAX_LAT_VELOCITY (m/s) and the env's step_length (s) so that
        #   max commanded velocity = DMPO_MAX_LAT_VELOCITY
        # regardless of step length. Default 1.0 m/s matches SUMO's
        # setMaxSpeedLat cap; commands therefore stay below the cap rather
        # than being silently clipped. Replaces the old DMPO_MAX_LAT_DIST
        # which was step-length-blind (variant E silently asked for 5 m/s
        # lateral velocity, all clipped to 1 m/s by SUMO).
        max_lat_velocity = float(os.environ.get('DMPO_MAX_LAT_VELOCITY', '1.0'))
        step_length = float(os.environ.get('DMPO_STEP_LENGTH', '0.2'))
        max_lat_dist = max_lat_velocity * step_length
        lat_dist = combined * max_lat_dist

        return float(lat_dist)

    def _apply_safety_constraints(self, vehicle_id: str, target_accel: float) -> float:
        """Apply minimal safety constraints to acceleration.

        NOTE: SafetyMonitor already handles most safety filtering.
        This method now only handles extreme cases to avoid double filtering.
        Double filtering breaks credit assignment (policy doesn't see its actions executed).
        """
        try:
            # Get following information
            leader_info = traci.vehicle.getLeader(vehicle_id)
            if leader_info is not None:
                leader_id, distance = leader_info

                if distance < 0:  # Invalid distance
                    return target_accel

                # Only intervene if VERY close (< 1m) - SafetyMonitor handles larger gaps
                if distance < 1.0:
                    target_accel = min(target_accel, self.emergency_decel)
                elif distance < 2.0:
                    # Limit max acceleration, but allow mild acceleration
                    target_accel = min(target_accel, 0.5)

            # Only brake hard for stopped vehicle very close ahead (< 5m)
            if self._has_stopped_vehicle_ahead(vehicle_id):
                # Check actual distance before forcing deceleration
                leader_info = traci.vehicle.getLeader(vehicle_id)
                if leader_info and leader_info[1] < 5.0:
                    target_accel = min(target_accel, -1.0)

            return target_accel

        except:
            return target_accel
    
    def _is_lane_change_safe(self, vehicle_id: str, target_lane: int) -> bool:
        """Check if lane change to target lane is safe."""
        try:
            current_speed = traci.vehicle.getSpeed(vehicle_id)
            current_pos = traci.vehicle.getLanePosition(vehicle_id)
            
            # Get vehicles in target lane
            edge_id = traci.lane.getEdgeID(traci.vehicle.getLaneID(vehicle_id))
            target_lane_id = f"{edge_id}_{target_lane}"
            
            # Check for conflicts in target lane
            vehicles_in_lane = traci.lane.getLastStepVehicleIDs(target_lane_id)
            
            for other_veh in vehicles_in_lane:
                other_pos = traci.vehicle.getLanePosition(other_veh)
                other_speed = traci.vehicle.getSpeed(other_veh)
                
                # Calculate relative position and speed
                pos_diff = other_pos - current_pos
                speed_diff = other_speed - current_speed
                
                # Check if there's sufficient gap
                if abs(pos_diff) < self.min_gap * max(current_speed, other_speed, 5.0):
                    # Check if vehicles are converging
                    if (pos_diff > 0 and speed_diff < -2.0) or \
                       (pos_diff < 0 and speed_diff > 2.0):
                        return False
            
            return True
            
        except:
            return False
    
    def _has_stopped_vehicle_ahead(self, vehicle_id: str) -> bool:
        """Check if there's a stopped vehicle ahead."""
        try:
            # Check vehicles on same lane
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            vehicles_in_lane = traci.lane.getLastStepVehicleIDs(lane_id)
            
            current_pos = traci.vehicle.getLanePosition(vehicle_id)
            
            for other_veh in vehicles_in_lane:
                if other_veh == vehicle_id:
                    continue
                    
                other_pos = traci.vehicle.getLanePosition(other_veh)
                other_speed = traci.vehicle.getSpeed(other_veh)
                
                # Check if vehicle is ahead and stopped/very slow
                if (other_pos > current_pos and 
                    other_pos - current_pos < 30.0 and  # Within 30m
                    other_speed < 1.0):  # Very slow/stopped
                    return True
            
            return False
            
        except:
            return False
    
    def _get_intersection_navigation_command(self, vehicle_id: str, 
                                           route_intention: float) -> Optional[str]:
        """Get intersection-specific navigation command."""
        try:
            # Check if vehicle is approaching intersection
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            
            # If approaching center, determine turn direction
            if 'center' in lane_id or self._is_approaching_intersection(vehicle_id):
                route = traci.vehicle.getRoute(vehicle_id)
                route_index = traci.vehicle.getRouteIndex(vehicle_id)
                
                if route_index < len(route) - 1:
                    current_edge = route[route_index]
                    next_edge = route[route_index + 1]
                    
                    # Return turn command based on route
                    turn_direction = self._get_turn_direction(current_edge, next_edge)
                    return turn_direction
            
            return None
            
        except:
            return None
    
    def _is_approaching_intersection(self, vehicle_id: str) -> bool:
        """Check if vehicle is approaching intersection."""
        try:
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            lane_length = traci.lane.getLength(lane_id)
            vehicle_pos = traci.vehicle.getLanePosition(vehicle_id)
            
            # Check if within 50m of lane end (intersection)
            distance_to_end = lane_length - vehicle_pos
            return distance_to_end < 50.0
            
        except:
            return False
    
    def _get_turn_direction(self, current_edge: str, next_edge: str) -> str:
        """Determine turn direction based on edge transition."""
        # Simplified turn detection based on edge names
        if ('north' in current_edge and 'south' in next_edge) or \
           ('south' in current_edge and 'north' in next_edge) or \
           ('east' in current_edge and 'west' in next_edge) or \
           ('west' in current_edge and 'east' in next_edge):
            return 'straight'
        elif ('north' in current_edge and 'east' in next_edge) or \
             ('east' in current_edge and 'south' in next_edge) or \
             ('south' in current_edge and 'west' in next_edge) or \
             ('west' in current_edge and 'north' in next_edge):
            return 'right'
        else:
            return 'left'
    
    def reset_vehicle_history(self, vehicle_id: str):
        """Reset action history for a vehicle."""
        if vehicle_id in self.prev_actions:
            del self.prev_actions[vehicle_id]
    
    def get_action_limits(self) -> Dict[str, tuple]:
        """Get action limits for each action dimension."""
        return {
            'longitudinal': (-1.0, 1.0),  # Maps to [max_decel, max_accel]
            'lateral': (-1.0, 1.0),       # Steering input
            'lane_intention': (-1.0, 1.0) # Lane change intention
        }