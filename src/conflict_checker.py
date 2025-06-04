"""
Conflict Checker Module for UAV Strategic Deconfliction System

This module provides conflict detection functionality for drone missions
by analyzing 3D spatial separation and temporal overlap between flights.
"""

from typing import List, Dict, Any, Tuple, Union
from dataclasses import dataclass
import math
from src.mission_parser import DroneMission, Waypoint, calculate_distance_3d


@dataclass
class ConflictDetail:
    """Represents a detected conflict between two drones."""
    time: int
    location: Waypoint
    conflicting_drone_id: str
    distance: float
    primary_position: Waypoint
    conflicting_position: Waypoint
    
    def __str__(self):
        return (f"Conflict at t={self.time}s: {self.conflicting_drone_id} "
                f"({self.distance:.2f}m separation) at "
                f"({self.location.x:.1f}, {self.location.y:.1f}, {self.location.z:.1f})")


class ConflictResult:
    """Result of conflict checking analysis."""
    
    def __init__(self, status: str, conflicts: List[ConflictDetail] = None):
        self.status = status  # "clear" or "conflict"
        self.conflicts = conflicts or []
    
    def __str__(self):
        if self.status == "clear":
            return "Mission Status: CLEAR - No conflicts detected"
        else:
            return (f"Mission Status: CONFLICT - {len(self.conflicts)} conflicts detected\n" +
                   "\n".join(str(c) for c in self.conflicts))
    
    def get_conflict_summary(self) -> Dict[str, Any]:
        """Returns a summary of conflicts grouped by drone."""
        if self.status == "clear":
            return {"status": "clear", "total_conflicts": 0}
        
        summary = {
            "status": "conflict",
            "total_conflicts": len(self.conflicts),
            "conflicting_drones": {},
            "time_range": {
                "first_conflict": min(c.time for c in self.conflicts),
                "last_conflict": max(c.time for c in self.conflicts)
            },
            "min_separation": min(c.distance for c in self.conflicts),
            "max_separation": max(c.distance for c in self.conflicts)
        }
        
        # Group conflicts by drone
        for conflict in self.conflicts:
            drone_id = conflict.conflicting_drone_id
            if drone_id not in summary["conflicting_drones"]:
                summary["conflicting_drones"][drone_id] = {
                    "count": 0,
                    "times": [],
                    "min_distance": float('inf'),
                    "max_distance": 0
                }
            
            drone_data = summary["conflicting_drones"][drone_id]
            drone_data["count"] += 1
            drone_data["times"].append(conflict.time)
            drone_data["min_distance"] = min(drone_data["min_distance"], conflict.distance)
            drone_data["max_distance"] = max(drone_data["max_distance"], conflict.distance)
        
        return summary


def check_conflicts(primary_mission: DroneMission, 
                   simulated_flights: List[DroneMission], 
                   buffer: float = 5.0,
                   time_step: int = 1) -> ConflictResult:
    """
    Check for conflicts between primary mission and simulated flights.
    
    Args:
        primary_mission (DroneMission): The primary drone mission
        simulated_flights (List[DroneMission]): List of simulated drone missions
        buffer (float): Minimum safe separation distance in meters (default: 5.0)
        time_step (int): Time step for checking in seconds (default: 1)
        
    Returns:
        ConflictResult: Result object containing conflict status and details
    """
    conflicts = []
    
    # Determine the overall time range to check
    start_time = primary_mission.start_time
    end_time = primary_mission.end_time
    
    # Extend time range if simulated flights go beyond primary mission
    for flight in simulated_flights:
        start_time = min(start_time, flight.start_time)
        end_time = max(end_time, flight.end_time)
    
    # Iterate through time steps
    for current_time in range(start_time, end_time + 1, time_step):
        # Skip if primary mission is not active at this time
        if not (primary_mission.start_time <= current_time <= primary_mission.end_time):
            continue
        
        # Get primary drone position at current time
        primary_position = primary_mission.get_waypoint_at_time(current_time)
        
        # Check each simulated flight
        for simulated_flight in simulated_flights:
            # Skip if simulated flight is not active at this time
            if not (simulated_flight.start_time <= current_time <= simulated_flight.end_time):
                continue
            
            # Get simulated drone position at current time
            simulated_position = simulated_flight.get_waypoint_at_time(current_time)
            
            # Calculate 3D distance between drones
            distance = calculate_distance_3d(primary_position, simulated_position)
            
            # Check for conflict
            if distance < buffer:
                # Calculate midpoint for conflict location
                conflict_location = Waypoint(
                    x=(primary_position.x + simulated_position.x) / 2,
                    y=(primary_position.y + simulated_position.y) / 2,
                    z=(primary_position.z + simulated_position.z) / 2,
                    time=current_time
                )
                
                conflict = ConflictDetail(
                    time=current_time,
                    location=conflict_location,
                    conflicting_drone_id=simulated_flight.drone_id,
                    distance=distance,
                    primary_position=primary_position,
                    conflicting_position=simulated_position
                )
                
                conflicts.append(conflict)
    
    # Return result
    if conflicts:
        return ConflictResult("conflict", conflicts)
    else:
        return ConflictResult("clear")


def check_closest_approach(primary_mission: DroneMission, 
                          simulated_flights: List[DroneMission],
                          time_step: int = 1) -> Dict[str, Dict[str, Any]]:
    """
    Find the closest approach between primary mission and each simulated flight.
    
    Args:
        primary_mission (DroneMission): The primary drone mission
        simulated_flights (List[DroneMission]): List of simulated drone missions
        time_step (int): Time step for checking in seconds (default: 1)
        
    Returns:
        Dict: Dictionary with closest approach data for each simulated drone
    """
    closest_approaches = {}
    
    for simulated_flight in simulated_flights:
        min_distance = float('inf')
        closest_time = None
        closest_primary_pos = None
        closest_simulated_pos = None
        
        # Find overlapping time range
        overlap_start = max(primary_mission.start_time, simulated_flight.start_time)
        overlap_end = min(primary_mission.end_time, simulated_flight.end_time)
        
        if overlap_start <= overlap_end:
            # Check each time step in the overlap period
            for current_time in range(overlap_start, overlap_end + 1, time_step):
                primary_pos = primary_mission.get_waypoint_at_time(current_time)
                simulated_pos = simulated_flight.get_waypoint_at_time(current_time)
                
                distance = calculate_distance_3d(primary_pos, simulated_pos)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_time = current_time
                    closest_primary_pos = primary_pos
                    closest_simulated_pos = simulated_pos
        
        closest_approaches[simulated_flight.drone_id] = {
            "min_distance": min_distance if min_distance != float('inf') else None,
            "time": closest_time,
            "primary_position": closest_primary_pos,
            "simulated_position": closest_simulated_pos,
            "has_temporal_overlap": overlap_start <= overlap_end
        }
    
    return closest_approaches


def analyze_mission_safety(primary_mission: DroneMission, 
                          simulated_flights: List[DroneMission],
                          buffer: float = 5.0,
                          warning_buffer: float = 10.0,
                          time_step: int = 1) -> Dict[str, Any]:
    """
    Comprehensive safety analysis of the mission.
    
    Args:
        primary_mission (DroneMission): The primary drone mission
        simulated_flights (List[DroneMission]): List of simulated drone missions
        buffer (float): Critical conflict distance in meters
        warning_buffer (float): Warning distance in meters
        time_step (int): Time step for analysis in seconds
        
    Returns:
        Dict: Comprehensive safety analysis results
    """
    # Check for critical conflicts
    conflict_result = check_conflicts(primary_mission, simulated_flights, buffer, time_step)
    
    # Check for warnings (closer than warning buffer but not critical)
    warning_result = check_conflicts(primary_mission, simulated_flights, warning_buffer, time_step)
    
    # Get closest approaches
    closest_approaches = check_closest_approach(primary_mission, simulated_flights, time_step)
    
    # Calculate warning-only events (in warning zone but not conflict zone)
    warning_only_conflicts = []
    if warning_result.status == "conflict":
        conflict_times = set()
        if conflict_result.status == "conflict":
            conflict_times = {(c.time, c.conflicting_drone_id) for c in conflict_result.conflicts}
        
        warning_only_conflicts = [
            w for w in warning_result.conflicts 
            if (w.time, w.conflicting_drone_id) not in conflict_times
        ]
    
    return {
        "safety_status": conflict_result.status,
        "critical_conflicts": {
            "count": len(conflict_result.conflicts),
            "details": conflict_result.conflicts
        },
        "warnings": {
            "count": len(warning_only_conflicts),
            "details": warning_only_conflicts
        },
        "closest_approaches": closest_approaches,
        "summary": conflict_result.get_conflict_summary(),
        "recommendations": _generate_safety_recommendations(
            conflict_result, warning_only_conflicts, closest_approaches, buffer
        )
    }


def _generate_safety_recommendations(conflict_result: ConflictResult,
                                   warnings: List[ConflictDetail],
                                   closest_approaches: Dict[str, Dict[str, Any]],
                                   buffer: float) -> List[str]:
    """Generate safety recommendations based on analysis results."""
    recommendations = []
    
    if conflict_result.status == "conflict":
        recommendations.append(f"CRITICAL: {len(conflict_result.conflicts)} conflicts detected below {buffer}m separation")
        recommendations.append("Consider rescheduling mission or adjusting flight path")
        
        # Identify most problematic drones
        drone_conflicts = {}
        for conflict in conflict_result.conflicts:
            drone_id = conflict.conflicting_drone_id
            drone_conflicts[drone_id] = drone_conflicts.get(drone_id, 0) + 1
        
        most_problematic = max(drone_conflicts.items(), key=lambda x: x[1])
        recommendations.append(f"Primary conflict source: {most_problematic[0]} ({most_problematic[1]} conflicts)")
    
    if warnings:
        recommendations.append(f"WARNING: {len(warnings)} close approaches detected")
        recommendations.append("Monitor these situations closely during flight")
    
    # Check for very close approaches
    for drone_id, approach_data in closest_approaches.items():
        if approach_data["min_distance"] and approach_data["min_distance"] < buffer * 2:
            recommendations.append(f"Close approach with {drone_id}: {approach_data['min_distance']:.1f}m minimum separation")
    
    if not conflict_result.conflicts and not warnings:
        recommendations.append("Mission appears safe with current traffic")
        recommendations.append("Continue monitoring for any changes in simulated flight plans")
    
    return recommendations


# Example usage and testing
if __name__ == "__main__":
    from mission_parser import load_primary_mission, load_simulated_flights
    
    try:
        # Load missions
        primary = load_primary_mission("primary_mission.json")
        simulated = load_simulated_flights("simulated_flights.json")
        
        print("=== UAV Conflict Detection Analysis ===\n")
        
        # Basic conflict check
        result = check_conflicts(primary, simulated, buffer=50.0)  # Using larger buffer for demo
        print("Basic Conflict Check:")
        print(result)
        print()
        
        # Comprehensive safety analysis
        safety_analysis = analyze_mission_safety(primary, simulated, buffer=50.0, warning_buffer=75.0)
        print("=== Comprehensive Safety Analysis ===")
        print(f"Safety Status: {safety_analysis['safety_status'].upper()}")
        print(f"Critical Conflicts: {safety_analysis['critical_conflicts']['count']}")
        print(f"Warnings: {safety_analysis['warnings']['count']}")
        print()
        
        # Show recommendations
        print("Safety Recommendations:")
        for i, rec in enumerate(safety_analysis['recommendations'], 1):
            print(f"{i}. {rec}")
        print()
        
        # Show closest approaches
        print("Closest Approaches:")
        for drone_id, approach in safety_analysis['closest_approaches'].items():
            if approach['min_distance']:
                print(f"  {drone_id}: {approach['min_distance']:.1f}m at t={approach['time']}s")
            else:
                print(f"  {drone_id}: No temporal overlap")
        
    except Exception as e:
        print(f"Error during conflict analysis: {e}")

def query_conflict_status(
    primary: DroneMission,
    simulated: List[DroneMission],
    buffer: float = 50.0
) -> Tuple[str, List[ConflictDetail]]:
    """
    Returns status ("clear" or "conflict detected") and list of ConflictDetail objects.
    """
    result = check_conflicts(primary, simulated, buffer)
    if result.status.lower() == "conflict":
        return "conflict detected", result.conflicts
    return "clear", []