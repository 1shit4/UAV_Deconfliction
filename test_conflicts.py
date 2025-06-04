"""
Comprehensive Test Suite for UAV Strategic Deconfliction System

This module provides comprehensive unit tests for the conflict detection system,
covering utility functions, interpolation, conflict detection, and edge cases.
"""

import unittest
import math
import json
import tempfile
import os
from typing import List

# Import the modules we're testing
from src.mission_parser import (
    DroneMission, 
    Waypoint, 
    load_primary_mission, 
    load_simulated_flights,
    get_all_active_drones_at_time,
    calculate_distance_3d
)
from src.conflict_checker import (
    ConflictDetail,
    ConflictResult,
    check_conflicts,
    check_closest_approach,
    analyze_mission_safety
)


class TestUtilityFunctions(unittest.TestCase):
    """Test basic utility functions."""
    
    def test_calculate_distance_3d_same_point(self):
        """Test distance calculation between identical points."""
        wp1 = Waypoint(x=100.0, y=200.0, z=50.0, time=0)
        wp2 = Waypoint(x=100.0, y=200.0, z=50.0, time=10)
        
        distance = calculate_distance_3d(wp1, wp2)
        self.assertAlmostEqual(distance, 0.0, places=5)
    
    def test_calculate_distance_3d_known_values(self):
        """Test distance calculation with known values."""
        # Test a 3-4-5 right triangle in 3D space
        wp1 = Waypoint(x=0.0, y=0.0, z=0.0, time=0)
        wp2 = Waypoint(x=3.0, y=4.0, z=0.0, time=10)
        
        distance = calculate_distance_3d(wp1, wp2)
        self.assertAlmostEqual(distance, 5.0, places=5)
    
    def test_calculate_distance_3d_with_altitude(self):
        """Test distance calculation including altitude difference."""
        wp1 = Waypoint(x=0.0, y=0.0, z=0.0, time=0)
        wp2 = Waypoint(x=3.0, y=4.0, z=12.0, time=10)
        
        # Should form a 3-4-12 right triangle with hypotenuse = 13
        expected_distance = math.sqrt(3**2 + 4**2 + 12**2)
        distance = calculate_distance_3d(wp1, wp2)
        self.assertAlmostEqual(distance, expected_distance, places=5)
        self.assertAlmostEqual(distance, 13.0, places=5)


class TestWaypointInterpolation(unittest.TestCase):
    """Test waypoint interpolation functionality."""
    
    def setUp(self):
        """Set up test drone mission for interpolation tests."""
        waypoints = [
            Waypoint(x=0.0, y=0.0, z=50.0, time=0),
            Waypoint(x=100.0, y=100.0, z=100.0, time=100),
            Waypoint(x=200.0, y=0.0, z=50.0, time=200)
        ]
        self.mission = DroneMission("TEST-001", 0, 200, waypoints)
    
    def test_interpolation_at_waypoint(self):
        """Test interpolation exactly at waypoint times."""
        pos_at_0 = self.mission.get_waypoint_at_time(0)
        self.assertEqual(pos_at_0.x, 0.0)
        self.assertEqual(pos_at_0.y, 0.0)
        self.assertEqual(pos_at_0.z, 50.0)
        
        pos_at_100 = self.mission.get_waypoint_at_time(100)
        self.assertEqual(pos_at_100.x, 100.0)
        self.assertEqual(pos_at_100.y, 100.0)
        self.assertEqual(pos_at_100.z, 100.0)
    
    def test_interpolation_midpoint(self):
        """Test interpolation at midpoint between waypoints."""
        pos_at_50 = self.mission.get_waypoint_at_time(50)
        
        # Should be exactly halfway between first two waypoints
        self.assertAlmostEqual(pos_at_50.x, 50.0, places=5)
        self.assertAlmostEqual(pos_at_50.y, 50.0, places=5)
        self.assertAlmostEqual(pos_at_50.z, 75.0, places=5)
        self.assertEqual(pos_at_50.time, 50)
    
    def test_interpolation_quarter_point(self):
        """Test interpolation at quarter point between waypoints."""
        pos_at_25 = self.mission.get_waypoint_at_time(25)
        
        # Should be 1/4 of the way from first to second waypoint
        self.assertAlmostEqual(pos_at_25.x, 25.0, places=5)
        self.assertAlmostEqual(pos_at_25.y, 25.0, places=5)
        self.assertAlmostEqual(pos_at_25.z, 62.5, places=5)
    
    def test_interpolation_before_mission_start(self):
        """Test interpolation before mission start time."""
        pos_before = self.mission.get_waypoint_at_time(-10)
        
        # Should return first waypoint
        self.assertEqual(pos_before.x, 0.0)
        self.assertEqual(pos_before.y, 0.0)
        self.assertEqual(pos_before.z, 50.0)
    
    def test_interpolation_after_mission_end(self):
        """Test interpolation after mission end time."""
        pos_after = self.mission.get_waypoint_at_time(250)
        
        # Should return last waypoint
        self.assertEqual(pos_after.x, 200.0)
        self.assertEqual(pos_after.y, 0.0)
        self.assertEqual(pos_after.z, 50.0)


class TestConflictDetection(unittest.TestCase):
    """Test conflict detection logic."""
    
    def setUp(self):
        """Set up test missions for conflict detection."""
        # Create a simple primary mission
        primary_waypoints = [
            Waypoint(x=0.0, y=0.0, z=50.0, time=0),
            Waypoint(x=100.0, y=0.0, z=50.0, time=100)
        ]
        self.primary_mission = DroneMission("PRIMARY", 0, 100, primary_waypoints)
    
    def test_no_conflicts_different_altitudes(self):
        """Test no conflicts when drones are at different altitudes."""
        # Simulated drone flying same path but at different altitude
        sim_waypoints = [
            Waypoint(x=0.0, y=0.0, z=150.0, time=0),  # 100m higher
            Waypoint(x=100.0, y=0.0, z=150.0, time=100)
        ]
        simulated_mission = DroneMission("SIM-001", 0, 100, sim_waypoints)
        
        result = check_conflicts(self.primary_mission, [simulated_mission], buffer=50.0)
        
        self.assertEqual(result.status, "clear")
        self.assertEqual(len(result.conflicts), 0)
    
    def test_no_conflicts_different_times(self):
        """Test no conflicts when drones fly same path at different times."""
        # Simulated drone flying same path but starting after primary ends
        sim_waypoints = [
            Waypoint(x=0.0, y=0.0, z=50.0, time=150),
            Waypoint(x=100.0, y=0.0, z=50.0, time=250)
        ]
        simulated_mission = DroneMission("SIM-002", 150, 250, sim_waypoints)
        
        result = check_conflicts(self.primary_mission, [simulated_mission], buffer=5.0)
        
        self.assertEqual(result.status, "clear")
        self.assertEqual(len(result.conflicts), 0)
    
    def test_conflict_same_path_same_time(self):
        """Test conflict when drones fly identical paths at same time."""
        # Simulated drone flying exactly same path at same time
        sim_waypoints = [
            Waypoint(x=0.0, y=0.0, z=50.0, time=0),
            Waypoint(x=100.0, y=0.0, z=50.0, time=100)
        ]
        simulated_mission = DroneMission("SIM-003", 0, 100, sim_waypoints)
        
        result = check_conflicts(self.primary_mission, [simulated_mission], buffer=5.0)
        
        self.assertEqual(result.status, "conflict")
        self.assertGreater(len(result.conflicts), 0)
        
        # All conflicts should be with the same drone
        for conflict in result.conflicts:
            self.assertEqual(conflict.conflicting_drone_id, "SIM-003")
            self.assertLess(conflict.distance, 5.0)
    
    def test_conflict_just_under_buffer(self):
        """Test conflict detection just under buffer distance."""
        # Simulated drone flying parallel path 4m away (buffer is 5m)
        sim_waypoints = [
            Waypoint(x=0.0, y=4.0, z=50.0, time=0),
            Waypoint(x=100.0, y=4.0, z=50.0, time=100)
        ]
        simulated_mission = DroneMission("SIM-004", 0, 100, sim_waypoints)
        
        result = check_conflicts(self.primary_mission, [simulated_mission], buffer=5.0)
        
        self.assertEqual(result.status, "conflict")
        self.assertGreater(len(result.conflicts), 0)
        
        # Check that all detected distances are indeed under buffer
        for conflict in result.conflicts:
            self.assertLess(conflict.distance, 5.0)
            self.assertAlmostEqual(conflict.distance, 4.0, places=1)
    
    def test_no_conflict_just_over_buffer(self):
        """Test no conflict when distance is just over buffer."""
        # Simulated drone flying parallel path 6m away (buffer is 5m)
        sim_waypoints = [
            Waypoint(x=0.0, y=6.0, z=50.0, time=0),
            Waypoint(x=100.0, y=6.0, z=50.0, time=100)
        ]
        simulated_mission = DroneMission("SIM-005", 0, 100, sim_waypoints)
        
        result = check_conflicts(self.primary_mission, [simulated_mission], buffer=5.0)
        
        self.assertEqual(result.status, "clear")
        self.assertEqual(len(result.conflicts), 0)
    
    def test_partial_time_overlap_conflict(self):
        """Test conflict with partial time overlap."""
        # Primary mission: 0-100s
        # Simulated mission: 50-150s, overlapping path during 50-100s
        sim_waypoints = [
            Waypoint(x=50.0, y=0.0, z=50.0, time=50),  # Starts at primary's midpoint
            Waypoint(x=150.0, y=0.0, z=50.0, time=150)
        ]
        simulated_mission = DroneMission("SIM-006", 50, 150, sim_waypoints)
        
        result = check_conflicts(self.primary_mission, [simulated_mission], buffer=5.0)
        
        self.assertEqual(result.status, "conflict")
        self.assertGreater(len(result.conflicts), 0)
        
        # All conflicts should occur during overlap period (50-100s)
        for conflict in result.conflicts:
            self.assertGreaterEqual(conflict.time, 50)
            self.assertLessEqual(conflict.time, 100)
    
    def test_multiple_drones_single_conflict(self):
        """Test scenario with multiple drones but only one conflict."""
        # Safe drone at different altitude
        safe_waypoints = [
            Waypoint(x=0.0, y=0.0, z=150.0, time=0),
            Waypoint(x=100.0, y=0.0, z=150.0, time=100)
        ]
        safe_mission = DroneMission("SAFE", 0, 100, safe_waypoints)
        
        # Conflicting drone at same altitude
        conflict_waypoints = [
            Waypoint(x=0.0, y=2.0, z=50.0, time=0),  # 2m away, under 5m buffer
            Waypoint(x=100.0, y=2.0, z=50.0, time=100)
        ]
        conflict_mission = DroneMission("CONFLICT", 0, 100, conflict_waypoints)
        
        result = check_conflicts(self.primary_mission, [safe_mission, conflict_mission], buffer=5.0)
        
        self.assertEqual(result.status, "conflict")
        self.assertGreater(len(result.conflicts), 0)
        
        # All conflicts should be with the conflicting drone only
        for conflict in result.conflicts:
            self.assertEqual(conflict.conflicting_drone_id, "CONFLICT")


class TestClosestApproach(unittest.TestCase):
    """Test closest approach calculation."""
    
    def setUp(self):
        """Set up test missions for closest approach tests."""
        primary_waypoints = [
            Waypoint(x=0.0, y=0.0, z=50.0, time=0),
            Waypoint(x=100.0, y=0.0, z=50.0, time=100)
        ]
        self.primary_mission = DroneMission("PRIMARY", 0, 100, primary_waypoints)
    
    def test_closest_approach_parallel_paths(self):
        """Test closest approach calculation for parallel paths."""
        # Drone flying parallel path 10m away
        sim_waypoints = [
            Waypoint(x=0.0, y=10.0, z=50.0, time=0),
            Waypoint(x=100.0, y=10.0, z=50.0, time=100)
        ]
        simulated_mission = DroneMission("SIM", 0, 100, sim_waypoints)
        
        approaches = check_closest_approach(self.primary_mission, [simulated_mission])
        
        self.assertIn("SIM", approaches)
        self.assertAlmostEqual(approaches["SIM"]["min_distance"], 10.0, places=1)
        self.assertTrue(approaches["SIM"]["has_temporal_overlap"])
    
    def test_closest_approach_crossing_paths(self):
        """Test closest approach for crossing paths."""
        # Drone crossing primary's path at 90 degrees
        sim_waypoints = [
            Waypoint(x=50.0, y=-20.0, z=50.0, time=0),
            Waypoint(x=50.0, y=20.0, z=50.0, time=100)
        ]
        simulated_mission = DroneMission("SIM", 0, 100, sim_waypoints)
        
        approaches = check_closest_approach(self.primary_mission, [simulated_mission])
        
        self.assertIn("SIM", approaches)
        # Should approach very close when paths cross at (50, 0, 50)
        self.assertLess(approaches["SIM"]["min_distance"], 1.0)
        self.assertEqual(approaches["SIM"]["time"], 50)  # Should cross at t=50
    
    def test_closest_approach_no_temporal_overlap(self):
        """Test closest approach with no temporal overlap."""
        # Drone flying after primary mission ends
        sim_waypoints = [
            Waypoint(x=0.0, y=0.0, z=50.0, time=150),
            Waypoint(x=100.0, y=0.0, z=50.0, time=250)
        ]
        simulated_mission = DroneMission("SIM", 150, 250, sim_waypoints)
        
        approaches = check_closest_approach(self.primary_mission, [simulated_mission])
        
        self.assertIn("SIM", approaches)
        self.assertIsNone(approaches["SIM"]["min_distance"])
        self.assertFalse(approaches["SIM"]["has_temporal_overlap"])


class TestSafetyAnalysis(unittest.TestCase):
    """Test comprehensive safety analysis functionality."""
    
    def setUp(self):
        """Set up test missions for safety analysis."""
        primary_waypoints = [
            Waypoint(x=0.0, y=0.0, z=50.0, time=0),
            Waypoint(x=100.0, y=0.0, z=50.0, time=100)
        ]
        self.primary_mission = DroneMission("PRIMARY", 0, 100, primary_waypoints)
    
    def test_safety_analysis_all_clear(self):
        """Test safety analysis with no conflicts or warnings."""
        # Safe drone at different altitude
        safe_waypoints = [
            Waypoint(x=0.0, y=0.0, z=150.0, time=0),
            Waypoint(x=100.0, y=0.0, z=150.0, time=100)
        ]
        safe_mission = DroneMission("SAFE", 0, 100, safe_waypoints)
        
        analysis = analyze_mission_safety(
            self.primary_mission, 
            [safe_mission], 
            buffer=5.0, 
            warning_buffer=15.0
        )
        
        self.assertEqual(analysis["safety_status"], "clear")
        self.assertEqual(analysis["critical_conflicts"]["count"], 0)
        self.assertEqual(analysis["warnings"]["count"], 0)
        self.assertIn("Mission appears safe", " ".join(analysis["recommendations"]))
    
    def test_safety_analysis_warnings_only(self):
        """Test safety analysis with warnings but no critical conflicts."""
        # Drone close enough for warning but not critical conflict
        warning_waypoints = [
            Waypoint(x=0.0, y=8.0, z=50.0, time=0),  # 8m away
            Waypoint(x=100.0, y=8.0, z=50.0, time=100)
        ]
        warning_mission = DroneMission("WARNING", 0, 100, warning_waypoints)
        
        analysis = analyze_mission_safety(
            self.primary_mission, 
            [warning_mission], 
            buffer=5.0,      # Critical: <5m
            warning_buffer=10.0  # Warning: <10m
        )
        
        self.assertEqual(analysis["safety_status"], "clear")  # No critical conflicts
        self.assertEqual(analysis["critical_conflicts"]["count"], 0)
        self.assertGreater(analysis["warnings"]["count"], 0)
        self.assertIn("WARNING", " ".join(analysis["recommendations"]))
    
    def test_safety_analysis_critical_conflicts(self):
        """Test safety analysis with critical conflicts."""
        # Drone too close - critical conflict
        critical_waypoints = [
            Waypoint(x=0.0, y=3.0, z=50.0, time=0),  # 3m away, under 5m buffer
            Waypoint(x=100.0, y=3.0, z=50.0, time=100)
        ]
        critical_mission = DroneMission("CRITICAL", 0, 100, critical_waypoints)
        
        analysis = analyze_mission_safety(
            self.primary_mission, 
            [critical_mission], 
            buffer=5.0, 
            warning_buffer=10.0
        )
        
        self.assertEqual(analysis["safety_status"], "conflict")
        self.assertGreater(analysis["critical_conflicts"]["count"], 0)
        self.assertIn("CRITICAL", " ".join(analysis["recommendations"]))
        self.assertIn("rescheduling", " ".join(analysis["recommendations"]).lower())


class TestConflictResult(unittest.TestCase):
    """Test ConflictResult class functionality."""
    
    def test_conflict_result_clear(self):
        """Test ConflictResult with clear status."""
        result = ConflictResult("clear")
        
        self.assertEqual(result.status, "clear")
        self.assertEqual(len(result.conflicts), 0)
        self.assertIn("CLEAR", str(result))
        
        summary = result.get_conflict_summary()
        self.assertEqual(summary["status"], "clear")
        self.assertEqual(summary["total_conflicts"], 0)
    
    def test_conflict_result_with_conflicts(self):
        """Test ConflictResult with conflicts."""
        conflict1 = ConflictDetail(
            time=50,
            location=Waypoint(25.0, 25.0, 50.0, 50),
            conflicting_drone_id="TEST-1",
            distance=3.0,
            primary_position=Waypoint(25.0, 23.0, 50.0, 50),
            conflicting_position=Waypoint(25.0, 27.0, 50.0, 50)
        )
        
        conflict2 = ConflictDetail(
            time=75,
            location=Waypoint(50.0, 25.0, 50.0, 75),
            conflicting_drone_id="TEST-2",
            distance=4.5,
            primary_position=Waypoint(50.0, 23.0, 50.0, 75),
            conflicting_position=Waypoint(50.0, 27.5, 50.0, 75)
        )
        
        result = ConflictResult("conflict", [conflict1, conflict2])
        
        self.assertEqual(result.status, "conflict")
        self.assertEqual(len(result.conflicts), 2)
        self.assertIn("CONFLICT", str(result))
        
        summary = result.get_conflict_summary()
        self.assertEqual(summary["status"], "conflict")
        self.assertEqual(summary["total_conflicts"], 2)
        self.assertEqual(summary["min_separation"], 3.0)
        self.assertEqual(summary["max_separation"], 4.5)
        self.assertIn("TEST-1", summary["conflicting_drones"])
        self.assertIn("TEST-2", summary["conflicting_drones"])


class TestFileLoading(unittest.TestCase):
    """Test JSON file loading functionality."""
    
    def setUp(self):
        """Set up temporary files for testing."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
    
    def test_load_valid_primary_mission(self):
        """Test loading a valid primary mission file."""
        mission_data = {
            "drone_id": "TEST-001",
            "start_time": 0,
            "end_time": 100,
            "waypoints": [
                {"x": 0.0, "y": 0.0, "z": 50.0, "time": 0},
                {"x": 100.0, "y": 100.0, "z": 75.0, "time": 100}
            ]
        }
        
        filepath = os.path.join(self.temp_dir, "test_primary.json")
        with open(filepath, 'w') as f:
            json.dump(mission_data, f)
        
        mission = load_primary_mission(filepath)
        
        self.assertEqual(mission.drone_id, "TEST-001")
        self.assertEqual(mission.start_time, 0)
        self.assertEqual(mission.end_time, 100)
        self.assertEqual(len(mission.waypoints), 2)
    
    def test_load_missing_file(self):
        """Test loading a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_primary_mission("nonexistent_file.json")
    
    def test_load_invalid_json(self):
        """Test loading a file with invalid JSON."""
        filepath = os.path.join(self.temp_dir, "invalid.json")
        with open(filepath, 'w') as f:
            f.write("{invalid json")
        
        with self.assertRaises(ValueError):
            load_primary_mission(filepath)
    
    def test_load_simulated_flights(self):
        """Test loading simulated flights file."""
        flights_data = [
            {
                "drone_id": "SIM-001",
                "start_time": 10,
                "end_time": 110,
                "waypoints": [
                    {"x": 10.0, "y": 10.0, "z": 60.0, "time": 10},
                    {"x": 110.0, "y": 110.0, "z": 80.0, "time": 110}
                ]
            },
            {
                "drone_id": "SIM-002",
                "start_time": 20,
                "end_time": 120,
                "waypoints": [
                    {"x": 20.0, "y": 20.0, "z": 70.0, "time": 20},
                    {"x": 120.0, "y": 120.0, "z": 90.0, "time": 120}
                ]
            }
        ]
        
        filepath = os.path.join(self.temp_dir, "test_simulated.json")
        with open(filepath, 'w') as f:
            json.dump(flights_data, f)
        
        flights = load_simulated_flights(filepath)
        
        self.assertEqual(len(flights), 2)
        self.assertEqual(flights[0].drone_id, "SIM-001")
        self.assertEqual(flights[1].drone_id, "SIM-002")


class TestActiveDrones(unittest.TestCase):
    """Test active drone detection functionality."""
    
    def setUp(self):
        """Set up test missions."""
        primary_waypoints = [
            Waypoint(x=0.0, y=0.0, z=50.0, time=0),
            Waypoint(x=100.0, y=0.0, z=50.0, time=100)
        ]
        self.primary_mission = DroneMission("PRIMARY", 0, 100, primary_waypoints)
        
        sim1_waypoints = [
            Waypoint(x=0.0, y=10.0, z=60.0, time=50),
            Waypoint(x=100.0, y=10.0, z=60.0, time=150)
        ]
        self.sim1_mission = DroneMission("SIM-1", 50, 150, sim1_waypoints)
        
        sim2_waypoints = [
            Waypoint(x=0.0, y=20.0, z=70.0, time=200),
            Waypoint(x=100.0, y=20.0, z=70.0, time=300)
        ]
        self.sim2_mission = DroneMission("SIM-2", 200, 300, sim2_waypoints)
    
    def test_active_drones_primary_only(self):
        """Test active drones when only primary is active."""
        active = get_all_active_drones_at_time(
            self.primary_mission, 
            [self.sim1_mission, self.sim2_mission], 
            25
        )
        
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].drone_id, "PRIMARY")
    
    def test_active_drones_multiple(self):
        """Test active drones when multiple drones are active."""
        active = get_all_active_drones_at_time(
            self.primary_mission, 
            [self.sim1_mission, self.sim2_mission], 
            75
        )
        
        self.assertEqual(len(active), 2)
        drone_ids = {drone.drone_id for drone in active}
        self.assertIn("PRIMARY", drone_ids)
        self.assertIn("SIM-1", drone_ids)
    
    def test_active_drones_none(self):
        """Test active drones when no drones are active."""
        active = get_all_active_drones_at_time(
            self.primary_mission, 
            [self.sim1_mission, self.sim2_mission], 
            400
        )
        
        self.assertEqual(len(active), 0)


if __name__ == "__main__":
    # Create a test suite with all test cases
    unittest.main(verbosity=2)