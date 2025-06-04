"""
Visualizer Module for UAV Strategic Deconfliction System

This module provides 3D visualization capabilities for drone missions and conflicts
using matplotlib for static and animated plots.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
from typing import List, Optional, Dict, Tuple
import matplotlib.colors as mcolors
from src.mission_parser import DroneMission, Waypoint
from src.conflict_checker import ConflictResult, ConflictDetail


class DroneVisualizer:
    """Main class for visualizing drone missions and conflicts."""
    
    def __init__(self, figsize=(16, 12)):
        """
        Initialize the visualizer.
        
        Args:
            figsize (tuple): Figure size for plots (width, height)
        """
        self.figsize = figsize
        self.colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        self.primary_color = 'red'
        self.conflict_color = 'red'
        self.conflict_size = 100
        
    def plot_static_missions(self, 
                           primary_mission: DroneMission,
                           simulated_flights: List[DroneMission],
                           conflict_result: Optional[ConflictResult] = None,
                           show_waypoints: bool = True,
                           show_grid: bool = True,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a static 3D plot of all drone missions and conflicts.
        
        Args:
            primary_mission (DroneMission): The primary drone mission
            simulated_flights (List[DroneMission]): List of simulated drone missions
            conflict_result (Optional[ConflictResult]): Conflict detection results
            show_waypoints (bool): Whether to show individual waypoints
            show_grid (bool): Whether to show grid lines
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot primary mission
        self._plot_mission_path(ax, primary_mission, self.primary_color, 
                              'Primary Mission', show_waypoints, linewidth=3)
        
        # Plot simulated flights
        for i, flight in enumerate(simulated_flights):
            color = self.colors[i % len(self.colors)]
            self._plot_mission_path(ax, flight, color, f'Simulated {flight.drone_id}', 
                                  show_waypoints, alpha=0.7)
        
        # Plot conflicts if provided
        if conflict_result and conflict_result.status == "conflict":
            self._plot_conflicts(ax, conflict_result.conflicts)
        
        # Customize the plot
        self._customize_3d_plot(ax, show_grid)
        
        # Add title with conflict information
        title = f"UAV Mission Visualization ({len(simulated_flights) + 1} drones)"
        if conflict_result and conflict_result.status == "conflict":
            title += f" - {len(conflict_result.conflicts)} Conflicts Detected"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return fig
    
    def animate_missions(self,
                        primary_mission: DroneMission,
                        simulated_flights: List[DroneMission],
                        conflict_result: Optional[ConflictResult] = None,
                        time_step: int = 1,
                        interval: int = 100,
                        show_trails: bool = True,
                        trail_length: int = 30,
                        save_path: Optional[str] = None) -> Tuple[plt.Figure, animation.FuncAnimation]:
        """
        Create an animated 3D plot showing drone movement over time.
        
        Args:
            primary_mission (DroneMission): The primary drone mission
            simulated_flights (List[DroneMission]): List of simulated drone missions
            conflict_result (Optional[ConflictResult]): Conflict detection results
            time_step (int): Time step between animation frames
            interval (int): Animation interval in milliseconds
            show_trails (bool): Whether to show movement trails
            trail_length (int): Length of trails in time steps
            save_path (Optional[str]): Path to save the animation (as GIF)
            
        Returns:
            Tuple[plt.Figure, animation.FuncAnimation]: Figure and animation objects
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate time range
        all_missions = [primary_mission] + simulated_flights
        start_time = min(mission.start_time for mission in all_missions)
        end_time = max(mission.end_time for mission in all_missions)
        time_range = range(start_time, end_time + 1, time_step)
        
        # Plot static paths (faded)
        self._plot_mission_path(ax, primary_mission, self.primary_color, 
                              'Primary Path', False, alpha=0.3, linewidth=1)
        for i, flight in enumerate(simulated_flights):
            color = self.colors[i % len(self.colors)]
            self._plot_mission_path(ax, flight, color, f'{flight.drone_id} Path', 
                                  False, alpha=0.3, linewidth=1)
        
        # Initialize animation elements
        drone_points = []
        drone_trails = []
        conflict_points = []
        
        # Create drone markers
        primary_point, = ax.plot([], [], [], 'o', color=self.primary_color, 
                               markersize=10, label='Primary Drone')
        drone_points.append((primary_mission, primary_point, self.primary_color))
        
        for i, flight in enumerate(simulated_flights):
            color = self.colors[i % len(self.colors)]
            point, = ax.plot([], [], [], 'o', color=color, markersize=8, 
                           label=f'{flight.drone_id}')
            drone_points.append((flight, point, color))
        
        # Create trail lines if requested
        if show_trails:
            primary_trail, = ax.plot([], [], [], '-', color=self.primary_color, 
                                   alpha=0.6, linewidth=2)
            drone_trails.append((primary_mission, primary_trail, self.primary_color, []))
            
            for i, flight in enumerate(simulated_flights):
                color = self.colors[i % len(self.colors)]
                trail, = ax.plot([], [], [], '-', color=color, alpha=0.6, linewidth=1.5)
                drone_trails.append((flight, trail, color, []))
        
        # Create conflict markers
        if conflict_result and conflict_result.status == "conflict":
            conflict_scatter = ax.scatter([], [], [], c=self.conflict_color, 
                                        s=self.conflict_size, marker='X', 
                                        alpha=0.8, label='Conflicts')
            conflict_points.append(conflict_scatter)
        
        # Customize plot
        self._customize_3d_plot(ax, True)
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        # Animation function
        def animate(frame):
            current_time = time_range[frame]
            
            # Update drone positions
            for mission, point, color in drone_points:
                if mission.start_time <= current_time <= mission.end_time:
                    pos = mission.get_waypoint_at_time(current_time)
                    point.set_data([pos.x], [pos.y])
                    point.set_3d_properties([pos.z])
                    point.set_alpha(1.0)
                else:
                    point.set_alpha(0.0)  # Hide inactive drones
            
            # Update trails
            if show_trails:
                for mission, trail, color, positions in drone_trails:
                    if mission.start_time <= current_time <= mission.end_time:
                        pos = mission.get_waypoint_at_time(current_time)
                        positions.append((pos.x, pos.y, pos.z))
                        
                        # Limit trail length
                        if len(positions) > trail_length:
                            positions.pop(0)
                        
                        if len(positions) > 1:
                            xs, ys, zs = zip(*positions)
                            trail.set_data(xs, ys)
                            trail.set_3d_properties(zs)
            
            # Update conflicts
            if conflict_result and conflict_result.status == "conflict":
                current_conflicts = [c for c in conflict_result.conflicts if c.time == current_time]
                if current_conflicts and conflict_points:
                    xs = [c.location.x for c in current_conflicts]
                    ys = [c.location.y for c in current_conflicts]
                    zs = [c.location.z for c in current_conflicts]
                    conflict_points[0]._offsets3d = (xs, ys, zs)
                else:
                    if conflict_points:
                        conflict_points[0]._offsets3d = ([], [], [])
            
            # Update title with current time
            ax.set_title(f"UAV Mission Animation - Time: {current_time}s", 
                        fontsize=14, fontweight='bold')
            
            return [point for _, point, _ in drone_points] + \
                   [trail for _, trail, _, _ in drone_trails] + conflict_points
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(time_range),
                                     interval=interval, blit=False, repeat=True)
        
        # Save animation if requested
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000//interval)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=1000//interval)
        
        plt.tight_layout()
        return fig, anim
    
    def plot_conflict_heatmap(self,
                             primary_mission: DroneMission,
                             simulated_flights: List[DroneMission],
                             conflict_result: ConflictResult,
                             grid_resolution: int = 50) -> plt.Figure:
        """
        Create a 2D heatmap showing conflict density in X-Y space.
        
        Args:
            primary_mission (DroneMission): The primary drone mission
            simulated_flights (List[DroneMission]): List of simulated drone missions
            conflict_result (ConflictResult): Conflict detection results
            grid_resolution (int): Resolution of the heatmap grid
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        if conflict_result.status == "clear" or not conflict_result.conflicts:
            raise ValueError("No conflicts to visualize in heatmap")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Flight paths with conflicts
        self._plot_2d_paths(ax1, primary_mission, simulated_flights, conflict_result)
        ax1.set_title("Flight Paths and Conflicts (Top View)", fontweight='bold')
        
        # Plot 2: Conflict density heatmap
        conflicts = conflict_result.conflicts
        xs = [c.location.x for c in conflicts]
        ys = [c.location.y for c in conflicts]
        
        # Create heatmap
        hist, xedges, yedges = np.histogram2d(xs, ys, bins=grid_resolution)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        im = ax2.imshow(hist.T, extent=extent, origin='lower', cmap='Reds', alpha=0.7)
        ax2.scatter(xs, ys, c='red', s=20, alpha=0.8)
        ax2.set_title("Conflict Density Heatmap", fontweight='bold')
        ax2.set_xlabel("X Position (m)")
        ax2.set_ylabel("Y Position (m)")
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, label='Conflict Density')
        
        plt.tight_layout()
        return fig
    
    def _plot_mission_path(self, ax, mission: DroneMission, color: str, 
                          label: str, show_waypoints: bool, **kwargs):
        """Helper method to plot a single mission path."""
        waypoints = mission.waypoints
        xs = [wp.x for wp in waypoints]
        ys = [wp.y for wp in waypoints]
        zs = [wp.z for wp in waypoints]
        
        # Plot path
        ax.plot(xs, ys, zs, color=color, label=label, **kwargs)
        
        # Plot waypoints
        if show_waypoints:
            ax.scatter(xs, ys, zs, c=color, s=30, alpha=0.8)
            
        # Mark start and end points
        ax.scatter([xs[0]], [ys[0]], [zs[0]], c=color, s=100, marker='^', 
                  edgecolors='black', linewidths=1)  # Start
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], c=color, s=100, marker='v', 
                  edgecolors='black', linewidths=1)  # End
    
    def _plot_conflicts(self, ax, conflicts: List[ConflictDetail]):
        """Helper method to plot conflict markers."""
        if not conflicts:
            return
            
        xs = [c.location.x for c in conflicts]
        ys = [c.location.y for c in conflicts]
        zs = [c.location.z for c in conflicts]
        
        ax.scatter(xs, ys, zs, c=self.conflict_color, s=self.conflict_size, 
                  marker='X', alpha=0.8, label=f'Conflicts ({len(conflicts)})',
                  edgecolors='black', linewidths=1)
    
    def _plot_2d_paths(self, ax, primary_mission: DroneMission, 
                      simulated_flights: List[DroneMission], 
                      conflict_result: ConflictResult):
        """Helper method to plot 2D paths for heatmap."""
        # Plot primary mission
        waypoints = primary_mission.waypoints
        xs = [wp.x for wp in waypoints]
        ys = [wp.y for wp in waypoints]
        ax.plot(xs, ys, color=self.primary_color, linewidth=3, 
               label='Primary Mission', alpha=0.7)
        
        # Plot simulated flights
        for i, flight in enumerate(simulated_flights):
            color = self.colors[i % len(self.colors)]
            waypoints = flight.waypoints
            xs = [wp.x for wp in waypoints]
            ys = [wp.y for wp in waypoints]
            ax.plot(xs, ys, color=color, linewidth=2, 
                   label=f'{flight.drone_id}', alpha=0.7)
        
        # Plot conflicts
        if conflict_result.conflicts:
            xs = [c.location.x for c in conflict_result.conflicts]
            ys = [c.location.y for c in conflict_result.conflicts]
            ax.scatter(xs, ys, c=self.conflict_color, s=50, marker='X', 
                      alpha=0.8, label='Conflicts')
        
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _customize_3d_plot(self, ax, show_grid: bool):
        """Helper method to customize 3D plot appearance."""
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_zlabel("Z Position (m)")
        
        if show_grid:
            ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,0.5])
        
        # Improve viewing angle
        ax.view_init(elev=30, azim=135)


# Convenience functions for easy use
def plot_missions(primary_mission: DroneMission,
                 simulated_flights: List[DroneMission],
                 conflict_result: Optional[ConflictResult] = None,
                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Convenience function to create a static 3D plot of missions.
    
    Args:
        primary_mission (DroneMission): The primary drone mission
        simulated_flights (List[DroneMission]): List of simulated drone missions
        conflict_result (Optional[ConflictResult]): Conflict detection results
        save_path (Optional[str]): Path to save the plot
        
    Returns:
        plt.Figure: The matplotlib figure object
    """
    visualizer = DroneVisualizer()
    return visualizer.plot_static_missions(primary_mission, simulated_flights, 
                                         conflict_result, save_path=save_path)


def animate_missions(primary_mission: DroneMission,
                    simulated_flights: List[DroneMission],
                    conflict_result: Optional[ConflictResult] = None,
                    save_path: Optional[str] = None) -> Tuple[plt.Figure, animation.FuncAnimation]:
    """
    Convenience function to create an animated 3D plot of missions.
    
    Args:
        primary_mission (DroneMission): The primary drone mission
        simulated_flights (List[DroneMission]): List of simulated drone missions
        conflict_result (Optional[ConflictResult]): Conflict detection results
        save_path (Optional[str]): Path to save the animation
        
    Returns:
        Tuple[plt.Figure, animation.FuncAnimation]: Figure and animation objects
    """
    visualizer = DroneVisualizer()
    return visualizer.animate_missions(primary_mission, simulated_flights, 
                                     conflict_result, save_path=save_path)


# Example usage and testing
if __name__ == "__main__":
    from mission_parser import load_primary_mission, load_simulated_flights
    from conflict_checker import check_conflicts
    
    try:
        # Load missions
        primary = load_primary_mission("primary_mission.json")
        simulated = load_simulated_flights("simulated_flights.json")
        
        # Check for conflicts (using larger buffer for demo)
        conflicts = check_conflicts(primary, simulated, buffer=50.0)
        
        print("=== UAV Mission Visualization ===")
        print(f"Loaded {len(simulated) + 1} missions")
        print(f"Conflict status: {conflicts.status}")
        if conflicts.status == "conflict":
            print(f"Found {len(conflicts.conflicts)} conflicts")
        
        # Create visualizer
        visualizer = DroneVisualizer()
        
        # Static plot
        print("\nCreating static 3D plot...")
        fig1 = visualizer.plot_static_missions(primary, simulated, conflicts)
        
        # Animated plot
        print("Creating animated 3D plot...")
        fig2, anim = visualizer.animate_missions(primary, simulated, conflicts, 
                                               interval=200, show_trails=True)
        global _anim
        _anim = anim
        
        # Conflict heatmap (if conflicts exist)
        if conflicts.status == "conflict":
            print("Creating conflict heatmap...")
            fig3 = visualizer.plot_conflict_heatmap(primary, simulated, conflicts)
        
        # Show plots
        plt.show()
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()