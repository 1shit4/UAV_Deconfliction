"""
Mission Parser Module for UAV Strategic Deconfliction System

This module provides functions to load and parse mission data for primary
and simulated drone flights from JSON files.
"""

import json
from typing import Dict, List, Any, NamedTuple
from pathlib import Path


class Waypoint(NamedTuple):
    """Represents a waypoint with 3D coordinates and timestamp."""
    x: float
    y: float
    z: float
    time: int


class DroneMission:
    """Represents a complete drone mission with waypoints."""
    
    def __init__(self, drone_id: str, start_time: int, end_time: int, waypoints: List[Waypoint]):
        self.drone_id = drone_id
        self.start_time = start_time
        self.end_time = end_time
        self.waypoints = waypoints
    
    def __repr__(self):
        return f"DroneMission(id={self.drone_id}, start={self.start_time}, end={self.end_time}, waypoints={len(self.waypoints)})"
    
    def get_duration(self) -> int:
        """Returns the total mission duration in seconds."""
        return self.end_time - self.start_time
    
    def get_waypoint_at_time(self, time: int) -> Waypoint:
        """
        Returns the waypoint at a specific time, or interpolates between waypoints.
        If time is before mission start or after mission end, returns the nearest waypoint.
        """
        if time <= self.start_time:
            return self.waypoints[0]
        if time >= self.end_time:
            return self.waypoints[-1]
        
        # Find the two waypoints to interpolate between
        for i in range(len(self.waypoints) - 1):
            if self.waypoints[i].time <= time <= self.waypoints[i + 1].time:
                wp1, wp2 = self.waypoints[i], self.waypoints[i + 1]
                
                # Linear interpolation
                if wp2.time == wp1.time:
                    return wp1
                
                ratio = (time - wp1.time) / (wp2.time - wp1.time)
                return Waypoint(
                    x=wp1.x + ratio * (wp2.x - wp1.x),
                    y=wp1.y + ratio * (wp2.y - wp1.y),
                    z=wp1.z + ratio * (wp2.z - wp1.z),
                    time=time
                )
        
        return self.waypoints[-1]


def load_primary_mission(filepath: str) -> DroneMission:
    """
    Load the primary drone mission from a JSON file.
    
    Args:
        filepath (str): Path to the primary mission JSON file
        
    Returns:
        DroneMission: The loaded primary mission object
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        KeyError: If required fields are missing from the JSON
        ValueError: If data types are incorrect
    """
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        
        # Validate required fields
        required_fields = ['drone_id', 'start_time', 'end_time', 'waypoints']
        for field in required_fields:
            if field not in data:
                raise KeyError(f"Missing required field: {field}")
        
        # Parse waypoints
        waypoints = []
        for wp_data in data['waypoints']:
            waypoint_fields = ['x', 'y', 'z', 'time']
            for field in waypoint_fields:
                if field not in wp_data:
                    raise KeyError(f"Missing waypoint field: {field}")
            
            waypoints.append(Waypoint(
                x=float(wp_data['x']),
                y=float(wp_data['y']),
                z=float(wp_data['z']),
                time=int(wp_data['time'])
            ))
        
        # Sort waypoints by time to ensure proper ordering
        waypoints.sort(key=lambda wp: wp.time)
        
        return DroneMission(
            drone_id=str(data['drone_id']),
            start_time=int(data['start_time']),
            end_time=int(data['end_time']),
            waypoints=waypoints
        )
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Primary mission file not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in primary mission file: {e}")


def load_simulated_flights(filepath: str) -> List[DroneMission]:
    """
    Load simulated drone flights from a JSON file.
    
    Args:
        filepath (str): Path to the simulated flights JSON file
        
    Returns:
        List[DroneMission]: List of simulated drone mission objects
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        KeyError: If required fields are missing from the JSON
        ValueError: If data types are incorrect
    """
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        
        if not isinstance(data, list):
            raise ValueError("Simulated flights file must contain a JSON array")
        
        missions = []
        for i, flight_data in enumerate(data):
            try:
                # Validate required fields
                required_fields = ['drone_id', 'start_time', 'end_time', 'waypoints']
                for field in required_fields:
                    if field not in flight_data:
                        raise KeyError(f"Missing required field: {field}")
                
                # Parse waypoints
                waypoints = []
                for wp_data in flight_data['waypoints']:
                    waypoint_fields = ['x', 'y', 'z', 'time']
                    for field in waypoint_fields:
                        if field not in wp_data:
                            raise KeyError(f"Missing waypoint field: {field}")
                    
                    waypoints.append(Waypoint(
                        x=float(wp_data['x']),
                        y=float(wp_data['y']),
                        z=float(wp_data['z']),
                        time=int(wp_data['time'])
                    ))
                
                # Sort waypoints by time
                waypoints.sort(key=lambda wp: wp.time)
                
                missions.append(DroneMission(
                    drone_id=str(flight_data['drone_id']),
                    start_time=int(flight_data['start_time']),
                    end_time=int(flight_data['end_time']),
                    waypoints=waypoints
                ))
                
            except (KeyError, ValueError) as e:
                raise ValueError(f"Error parsing flight {i}: {e}")
        
        return missions
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Simulated flights file not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in primary mission file: {e}")



def get_all_active_drones_at_time(primary_mission: DroneMission, 
                                 simulated_flights: List[DroneMission], 
                                 time: int) -> List[DroneMission]:
    """
    Get all drones that are active at a specific time.
    
    Args:
        primary_mission (DroneMission): The primary mission
        simulated_flights (List[DroneMission]): List of simulated flights
        time (int): Time in seconds
        
    Returns:
        List[DroneMission]: List of active drones at the specified time
    """
    active_drones = []
    
    # Check primary mission
    if primary_mission.start_time <= time <= primary_mission.end_time:
        active_drones.append(primary_mission)
    
    # Check simulated flights
    for flight in simulated_flights:
        if flight.start_time <= time <= flight.end_time:
            active_drones.append(flight)
    
    return active_drones


def calculate_distance_3d(waypoint1: Waypoint, waypoint2: Waypoint) -> float:
    """
    Calculate the 3D Euclidean distance between two waypoints.
    
    Args:
        waypoint1 (Waypoint): First waypoint
        waypoint2 (Waypoint): Second waypoint
        
    Returns:
        float: 3D distance between the waypoints
    """
    return ((waypoint1.x - waypoint2.x) ** 2 + 
            (waypoint1.y - waypoint2.y) ** 2 + 
            (waypoint1.z - waypoint2.z) ** 2) ** 0.5


# Example usage and testing
if __name__ == "__main__":
    try:
        # Load missions
        primary = load_primary_mission("primary_mission.json")
        simulated = load_simulated_flights("simulated_flights.json")
        
        print(f"Loaded primary mission: {primary}")
        print(f"Loaded {len(simulated)} simulated flights:")
        for flight in simulated:
            print(f"  - {flight}")
        
        # Example: Check for active drones at time 200 seconds
        active_at_200 = get_all_active_drones_at_time(primary, simulated, 200)
        print(f"\nActive drones at t=200 seconds: {len(active_at_200)}")
        
        # Example: Get positions at a specific time
        for drone in active_at_200:
            position = drone.get_waypoint_at_time(200)
            print(f"  {drone.drone_id} position at t=200: ({position.x:.1f}, {position.y:.1f}, {position.z:.1f})")
        
    except Exception as e:
        print(f"Error: {e}")