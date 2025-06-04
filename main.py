from src.mission_parser import load_primary_mission, load_simulated_flights
from src.conflict_checker import check_conflicts
from src.visualizer import DroneVisualizer
from src.drone_plotly_viz import (
    plot_static_missions_interactive,
    animate_mission_interactive,
    plot_conflict_heatmap_3d
)
import matplotlib.pyplot as plt


def main():
    print("UAV Deconfliction System - Plotly Visualization")

    # Load missions
    print("Loading mission data...")
    primary = load_primary_mission("data/primary_mission.json")
    simulated = load_simulated_flights("data/simulated_flights.json")

    # Perform conflict check
    print("Running conflict analysis...")
    conflict_result = check_conflicts(primary, simulated, buffer=50.0)

    # Display conflict summary
    print(f"\nMission Status: {conflict_result.status.upper()} - "
          f"{len(conflict_result.conflicts)} conflicts detected")
    for conflict in conflict_result.conflicts:
        print(f"Time: {conflict.time}s | Drone: {conflict.conflicting_drone_id} | "
              f"Distance: {conflict.distance:.2f}m | "
              f"Location: ({conflict.location.x:.1f}, "
              f"{conflict.location.y:.1f}, {conflict.location.z:.1f})")

    # Generate visualizations
    print("\nGenerating visualizations...")

    static_fig = plot_static_missions_interactive(primary, simulated, conflict_result)
    static_fig.show()


    if conflict_result.status.lower() == "conflict":
        heatmap_fig = plot_conflict_heatmap_3d(conflict_result)
        heatmap_fig.show()

    # Visualize results
    visualizer = DroneVisualizer()

    print("Creating animated 3D plot...")
    fig2, anim = visualizer.animate_missions(primary, simulated, conflict_result, interval=200, show_trails=True)
    
    # Prevent animation garbage collection
    global _anim
    _anim = anim

    print("Showing animated plot...")
    plt.figure(fig2.number)
    plt.show()

    print("Visualization complete.")

if __name__ == "__main__":
    main()
