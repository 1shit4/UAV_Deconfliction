"""
Advanced 3D Drone Deconfliction Visualization Module using Plotly
Provides interactive visualizations for drone mission analysis and conflict detection.
Compatible with DroneMission, ConflictResult, and Waypoint class structures.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import colorsys


# Type definitions (matching your actual structures)
class Waypoint(NamedTuple):
    x: float
    y: float
    z: float
    time: int


class DroneMission:
    def __init__(self, drone_id: str, start_time: int, end_time: int, waypoints: List[Waypoint]):
        self.drone_id = drone_id
        self.start_time = start_time
        self.end_time = end_time
        self.waypoints = waypoints


@dataclass
class ConflictDetail:
    time: int
    location: Waypoint
    conflicting_drone_id: str
    distance: float
    primary_position: Waypoint
    conflicting_position: Waypoint


class ConflictResult:
    def __init__(self, status: str, conflicts: List[ConflictDetail]):
        self.status = status
        self.conflicts = conflicts


def generate_distinct_colors(n: int) -> List[str]:
    """Generate n visually distinct colors for drone visualization."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
        value = 0.8 + (i % 2) * 0.2       # Vary brightness slightly
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})')
    return colors


def extract_mission_data(missions: List[DroneMission]) -> pd.DataFrame:
    """Extract and structure mission data from DroneMission objects for visualization."""
    data = []
    for mission in missions:
        for i, waypoint in enumerate(mission.waypoints):
            data.append({
                'drone_id': mission.drone_id,
                'x': waypoint.x,
                'y': waypoint.y,
                'z': waypoint.z,
                'time': waypoint.time,
                'waypoint_index': i,
                'start_time': mission.start_time,
                'end_time': mission.end_time
            })
    
    return pd.DataFrame(data)


def extract_conflict_data(primary: DroneMission, conflict_result: Optional[ConflictResult]) -> pd.DataFrame:
    """Extract conflict data from ConflictResult object for visualization."""
    if not conflict_result or not conflict_result.conflicts:
        return pd.DataFrame()
    
    conflicts = []
    for conflict in conflict_result.conflicts:
        conflicts.append({
            'drone1_id': primary.drone_id,  # Primary is always drone1
            'drone2_id': conflict.conflicting_drone_id,
            'x': conflict.location.x,
            'y': conflict.location.y,
            'z': conflict.location.z,
            'time': conflict.time,
            'distance': conflict.distance,
            'primary_x': conflict.primary_position.x,
            'primary_y': conflict.primary_position.y,
            'primary_z': conflict.primary_position.z,
            'conflicting_x': conflict.conflicting_position.x,
            'conflicting_y': conflict.conflicting_position.y,
            'conflicting_z': conflict.conflicting_position.z,
            # Determine severity based on distance
            'severity': 'high' if conflict.distance < 5 else 'medium' if conflict.distance < 10 else 'low'
        })
    
    return pd.DataFrame(conflicts)


def plot_static_missions_interactive(primary: DroneMission, 
                                   simulated: List[DroneMission], 
                                   conflict_result: Optional[ConflictResult] = None) -> go.Figure:
    """
    Create an interactive 3D plot of all drone flight paths with conflict markers.
    
    Args:
        primary: Primary DroneMission object
        simulated: List of simulated DroneMission objects
        conflict_result: Optional ConflictResult object
        
    Returns:
        plotly.graph_objects.Figure: Interactive 3D plot
    """
    fig = go.Figure()
    
    # Combine all missions
    all_missions = [primary] + simulated
    mission_df = extract_mission_data(all_missions)
    
    if mission_df.empty:
        fig.add_annotation(text="No mission data available", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Generate distinct colors for each drone
    unique_drones = mission_df['drone_id'].unique()
    colors = generate_distinct_colors(len(unique_drones))
    color_map = dict(zip(unique_drones, colors))
    
    # Plot each drone's path
    for drone_id in unique_drones:
        drone_data = mission_df[mission_df['drone_id'] == drone_id].sort_values('time')
        
        # Determine if this is primary or simulated
        is_primary = drone_id == primary.drone_id
        line_style = 'solid' if is_primary else 'dash'
        line_width = 5 if is_primary else 3
        
        # Main flight path
        fig.add_trace(go.Scatter3d(
            x=drone_data['x'],
            y=drone_data['y'],
            z=drone_data['z'],
            mode='lines+markers',
            name=f'{"Primary" if is_primary else "Simulated"} Drone {drone_id}',
            line=dict(color=color_map[drone_id], width=line_width, dash=line_style),
            marker=dict(size=4 if is_primary else 3, color=color_map[drone_id]),
            hovertemplate=(
                f'<b>{"Primary" if is_primary else "Simulated"} Drone {drone_id}</b><br>' +
                'X: %{x:.1f}<br>' +
                'Y: %{y:.1f}<br>' +
                'Z: %{z:.1f}<br>' +
                'Time: %{customdata}<br>' +
                '<extra></extra>'
            ),
            customdata=drone_data['time'],
            legendgroup=f'drone_{drone_id}'
        ))
        
        # Mark start point
        start_point = drone_data.iloc[0]
        fig.add_trace(go.Scatter3d(
            x=[start_point['x']],
            y=[start_point['y']],
            z=[start_point['z']],
            mode='markers',
            name=f'Start {drone_id}',
            marker=dict(size=10, color='green', symbol='diamond'),
            hovertemplate=(
                f'<b>START - {"Primary" if is_primary else "Simulated"} Drone {drone_id}</b><br>' +
                'X: %{x:.1f}<br>' +
                'Y: %{y:.1f}<br>' +
                'Z: %{z:.1f}<br>' +
                f'Time: {start_point["time"]}<br>' +
                '<extra></extra>'
            ),
            legendgroup=f'drone_{drone_id}',
            showlegend=False
        ))
        
        # Mark end point  
        end_point = drone_data.iloc[-1]
        fig.add_trace(go.Scatter3d(
            x=[end_point['x']],
            y=[end_point['y']],
            z=[end_point['z']],
            mode='markers',
            name=f'End {drone_id}',
            marker=dict(size=10, color='red', symbol='square'),
            hovertemplate=(
                f'<b>END - {"Primary" if is_primary else "Simulated"} Drone {drone_id}</b><br>' +
                'X: %{x:.1f}<br>' +
                'Y: %{y:.1f}<br>' +
                'Z: %{z:.1f}<br>' +
                f'Time: {end_point["time"]}<br>' +
                '<extra></extra>'
            ),
            legendgroup=f'drone_{drone_id}',
            showlegend=False
        ))
    
    # Add conflict markers
    conflicts_df = extract_conflict_data(primary, conflict_result)
    if not conflicts_df.empty:
        severity_colors = {'low': 'yellow', 'medium': 'orange', 'high': 'red'}
        severity_sizes = {'low': 12, 'medium': 15, 'high': 20}
        
        for idx, conflict in conflicts_df.iterrows():
            color = severity_colors.get(conflict['severity'], 'red')
            size = severity_sizes.get(conflict['severity'], 15)
            
            fig.add_trace(go.Scatter3d(
                x=[conflict['x']],
                y=[conflict['y']],
                z=[conflict['z']],
                mode='markers',
                name='Conflicts',
                marker=dict(
                    size=size,
                    color=color,
                    symbol='x',
                    line=dict(width=3, color='darkred')
                ),
                hovertemplate=(
                    '<b>CONFLICT DETECTED</b><br>' +
                    f'Primary Drone: {conflict["drone1_id"]}<br>' +
                    f'Conflicting Drone: {conflict["drone2_id"]}<br>' +
                    'Conflict Location:<br>' +
                    'X: %{x:.1f}<br>' +
                    'Y: %{y:.1f}<br>' +
                    'Z: %{z:.1f}<br>' +
                    f'Time: {conflict["time"]}<br>' +
                    f'Distance: {conflict["distance"]:.2f}m<br>' +
                    f'Severity: {conflict["severity"].upper()}<br>' +
                    '<extra></extra>'
                ),
                legendgroup='conflicts',
                showlegend=True if idx == 0 else False
            ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Interactive 3D Drone Mission Visualization',
            'x': 0.5,
            'xanchor': 'center'
        },
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)', 
            zaxis_title='Z Position - Altitude (m)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode='cube'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left", 
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        width=1000,
        height=700
    )
    
    return fig


def animate_mission_interactive(primary: DroneMission, 
                              simulated: List[DroneMission], 
                              conflict_result: Optional[ConflictResult] = None) -> go.Figure:
    """
    Create an enhanced animated 3D visualization of drone missions over time with:
    - Animated trails behind all drones (primary + simulated)
    - Legend interactivity for toggling all drones
    - Clear start point markers
    - Time-synced conflict markers
    
    Args:
        primary: Primary DroneMission object
        simulated: List of simulated DroneMission objects
        conflict_result: Optional ConflictResult object
        
    Returns:
        plotly.graph_objects.Figure: Enhanced animated 3D plot with time slider
    """
    # Combine all missions
    all_missions = [primary] + simulated
    mission_df = extract_mission_data(all_missions)
    
    if mission_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No mission data available", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Get time range - use all unique times and sample every few steps for performance
    all_times = sorted(mission_df['time'].unique())
    min_time = int(min(all_times))
    max_time = int(max(all_times))
    
    # Sample time steps for animation (every 2-3 steps for performance)
    step_size = max(1, len(all_times) // 50)  # Limit to ~50 frames max
    time_steps = all_times[::step_size]
    
    # Generate colors for all drones
    unique_drones = mission_df['drone_id'].unique()
    colors = generate_distinct_colors(len(unique_drones))
    color_map = dict(zip(unique_drones, colors))
    
    # Extract conflicts
    conflicts_df = extract_conflict_data(primary, conflict_result)
    
    # Create frames for animation
    frames = []
    
    for time_step in time_steps:
        frame_data = []
        
        # Process each drone (primary + all simulated)
        for drone_id in unique_drones:
            # Get all waypoints for this drone up to current time
            drone_history = mission_df[
                (mission_df['drone_id'] == drone_id) & 
                (mission_df['time'] <= time_step)
            ].sort_values('time')
            
            if drone_history.empty:
                continue
                
            is_primary = drone_id == primary.drone_id
            
            # Get all waypoints for this drone to find start point
            all_drone_data = mission_df[mission_df['drone_id'] == drone_id].sort_values('time')
            start_point = all_drone_data.iloc[0]
            
            # Add start point marker (visible throughout animation)
            frame_data.append(go.Scatter3d(
                x=[start_point['x']],
                y=[start_point['y']],
                z=[start_point['z']],
                mode='markers',
                name=f'Start-{drone_id}',
                marker=dict(
                    size=12,
                    color='green',
                    symbol='diamond',
                    line=dict(width=2, color='darkgreen'),
                    opacity=0.9
                ),
                hovertemplate=(
                    f'<b>🛫 START - {"Primary" if is_primary else "Simulated"} Drone {drone_id}</b><br>' +
                    'X: %{x:.1f}<br>' +
                    'Y: %{y:.1f}<br>' +
                    'Z: %{z:.1f}<br>' +
                    f'Start Time: {start_point["time"]}<br>' +
                    '<extra></extra>'
                ),
                legendgroup=f'drone_{drone_id}',
                showlegend=False
            ))
            
            # Add animated trail - This is the MAIN trace that shows in legend
            if len(drone_history) > 1:
                frame_data.append(go.Scatter3d(
                    x=drone_history['x'].values,
                    y=drone_history['y'].values,
                    z=drone_history['z'].values,
                    mode='lines',
                    name=f'{"🎯 Primary" if is_primary else "🔄 Simulated"} Drone {drone_id}',
                    line=dict(
                        color=color_map[drone_id], 
                        width=6 if is_primary else 4,
                        dash='solid' if is_primary else 'dash'
                    ),
                    opacity=0.8,
                    hoverinfo='skip',
                    legendgroup=f'drone_{drone_id}',
                    showlegend=True,  # This is the main legend entry for each drone
                    visible=True  # Ensure it's visible by default
                ))
            
            # Add current position marker (prominent)
            current = drone_history.iloc[-1]
            frame_data.append(go.Scatter3d(
                x=[current['x']],
                y=[current['y']], 
                z=[current['z']],
                mode='markers',
                name=f'Current-{drone_id}',
                marker=dict(
                    size=14 if is_primary else 11,
                    color=color_map[drone_id],
                    symbol='circle',
                    line=dict(width=3, color='black' if is_primary else 'gray'),
                    opacity=1.0
                ),
                hovertemplate=(
                    f'<b>🚁 {"Primary" if is_primary else "Simulated"} Drone {drone_id}</b><br>' +
                    'Current Position:<br>' +
                    'X: %{x:.1f}m<br>' +
                    'Y: %{y:.1f}m<br>' +
                    'Z: %{z:.1f}m<br>' +
                    f'Time: {time_step}s<br>' +
                    '<extra></extra>'
                ),
                legendgroup=f'drone_{drone_id}',
                showlegend=False  # Don't show separate legend entry
            ))
            
            # Add trail points (smaller markers along the path) - only if there are multiple points
            if len(drone_history) > 1:
                trail_points = drone_history.iloc[:-1]  # All except current position
                if len(trail_points) > 0:
                    frame_data.append(go.Scatter3d(
                        x=trail_points['x'].values,
                        y=trail_points['y'].values,
                        z=trail_points['z'].values,
                        mode='markers',
                        name=f'Trail-{drone_id}',
                        marker=dict(
                            size=3,
                            color=color_map[drone_id],
                            opacity=0.4
                        ),
                        hoverinfo='skip',
                        legendgroup=f'drone_{drone_id}',
                        showlegend=False
                    ))
        
        # Add conflicts that occur at this specific time
        current_conflicts = pd.DataFrame()
        if not conflicts_df.empty:
            current_conflicts = conflicts_df[conflicts_df['time'] == time_step]
            if not current_conflicts.empty:
                severity_colors = {'low': 'yellow', 'medium': 'orange', 'high': 'red'}
                severity_sizes = {'low': 20, 'medium': 25, 'high': 30}
                
                conflict_legend_shown = False
                for idx, conflict in current_conflicts.iterrows():
                    # Main conflict marker
                    frame_data.append(go.Scatter3d(
                        x=[conflict['x']],
                        y=[conflict['y']],
                        z=[conflict['z']],
                        mode='markers',
                        name='⚠️ Active Conflicts',
                        marker=dict(
                            size=severity_sizes.get(conflict['severity'], 25),
                            color=severity_colors.get(conflict['severity'], 'red'),
                            symbol='x',
                            line=dict(width=4, color='darkred'),
                            opacity=1.0
                        ),
                        hovertemplate=(
                            '<b>⚠️ ACTIVE CONFLICT ⚠️</b><br>' +
                            f'Primary: {conflict["drone1_id"]}<br>' +
                            f'Conflicting: {conflict["drone2_id"]}<br>' +
                            'Conflict Location:<br>' +
                            'X: %{x:.1f}m<br>' +
                            'Y: %{y:.1f}m<br>' +
                            'Z: %{z:.1f}m<br>' +
                            f'Time: {conflict["time"]}s<br>' +
                            f'Distance: {conflict["distance"]:.2f}m<br>' +
                            f'Severity: {conflict["severity"].upper()}<br>' +
                            '<extra></extra>'
                        ),
                        legendgroup='conflicts',
                        showlegend=not conflict_legend_shown
                    ))
                    conflict_legend_shown = True
                    
                    # Add warning lines between conflicting drone positions
                    frame_data.append(go.Scatter3d(
                        x=[conflict['primary_x'], conflict['conflicting_x']],
                        y=[conflict['primary_y'], conflict['conflicting_y']],
                        z=[conflict['primary_z'], conflict['conflicting_z']],
                        mode='lines',
                        name='Conflict-Vector',
                        line=dict(
                            color='red',
                            width=6,
                            dash='dot'
                        ),
                        opacity=0.8,
                        hoverinfo='skip',
                        legendgroup='conflicts',
                        showlegend=False
                    ))
        
        # Count active elements for frame info
        active_drones = len(mission_df[mission_df['time'] <= time_step]['drone_id'].unique())
        active_conflicts = len(current_conflicts) if not current_conflicts.empty else 0
        
        frames.append(go.Frame(
            data=frame_data,
            name=str(time_step),
            layout=go.Layout(
                title=f'Enhanced Drone Animation - Time: {time_step}s',
                annotations=[
                    dict(
                        text=(
                            f'<b>⏱️ Time: {time_step}s</b><br>' +
                            f'🚁 Total Drones: {len(unique_drones)}<br>' +
                            f'📈 Active Drones: {active_drones}<br>' +
                            f'⚠️ Active Conflicts: {active_conflicts}<br>' +
                            f'📊 Time Range: {min_time}-{max_time}s'
                        ),
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        bgcolor="rgba(255,255,255,0.95)",
                        bordercolor="red" if active_conflicts > 0 else "green",
                        borderwidth=2,
                        font=dict(size=12, color="black")
                    )
                ]
            )
        ))
    
    # Create initial figure with first frame data
    initial_data = frames[0].data if frames else []
    fig = go.Figure(data=initial_data, frames=frames)
    
    # Enhanced layout with proper legend interactivity
    fig.update_layout(
        title={
            'text': '🚁 Enhanced Animated 3D Drone Mission Visualization',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Z Position - Altitude (m)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            bgcolor='rgba(240,240,240,0.1)',
            aspectmode='cube',
            xaxis=dict(gridcolor='rgba(200,200,200,0.5)'),
            yaxis=dict(gridcolor='rgba(200,200,200,0.5)'),
            zaxis=dict(gridcolor='rgba(200,200,200,0.5)')
        ),
        # Enhanced legend with full interactivity for all drones
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=11),
            itemclick="toggle",  # Enable click to toggle individual traces
            itemdoubleclick="toggleothers",  # Enable double-click to isolate
            groupclick="togglegroup",  # Enable group toggling
            tracegroupgap=5
        ),
        # Enhanced animation controls
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': '▶️ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 1000, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 500, 'easing': 'cubic-in-out'}
                    }]
                },
                {
                    'label': '⏸️ Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': '⏮️ Reset',
                    'method': 'animate',
                    'args': [[str(time_steps[0])], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': '⏩ Fast Forward',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 300, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 200}
                    }]
                }
            ],
            'x': 0.1,
            'y': 0.02,
            'bgcolor': 'rgba(255,255,255,0.8)',
            'bordercolor': 'black',
            'borderwidth': 1
        }],
        # Enhanced time slider
        sliders=[{
            'steps': [
                {
                    'args': [[str(t)], {
                        'frame': {'duration': 500, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 300}
                    }],
                    'label': f'T={t}s',
                    'method': 'animate'
                } for t in time_steps
            ],
            'active': 0,
            'currentvalue': {
                'prefix': 'Time: ',
                'suffix': 's',
                'font': {'size': 14, 'color': 'black'}
            },
            'len': 0.8,
            'x': 0.1,
            'y': 0,
            'bgcolor': 'rgba(255,255,255,0.8)',
            'bordercolor': 'black',
            'borderwidth': 1,
            'tickcolor': 'black'
        }],
        width=1200,
        height=800,
        margin=dict(l=0, r=0, t=50, b=80)
    )
    
    return fig


# Test function to verify the fix works
def test_animation_function():
    """Test function to verify all drones are properly animated and controllable."""
    
    # Sample data using your actual class structures
    primary_waypoints = [
        Waypoint(0.0, 0.0, 10.0, 0),
        Waypoint(10.0, 10.0, 15.0, 5),
        Waypoint(20.0, 20.0, 10.0, 10)
    ]
    primary_mission = DroneMission('P001', 0, 10, primary_waypoints)
    
    # Create multiple simulated missions
    simulated_waypoints_1 = [
        Waypoint(20.0, 0.0, 12.0, 0),
        Waypoint(10.0, 10.0, 18.0, 5),
        Waypoint(0.0, 20.0, 12.0, 10)
    ]
    simulated_mission_1 = DroneMission('S001', 0, 10, simulated_waypoints_1)
    
    simulated_waypoints_2 = [
        Waypoint(0.0, 20.0, 8.0, 0),
        Waypoint(5.0, 15.0, 12.0, 3),
        Waypoint(15.0, 5.0, 16.0, 8)
    ]
    simulated_mission_2 = DroneMission('S002', 0, 8, simulated_waypoints_2)
    
    simulated_missions = [simulated_mission_1, simulated_mission_2]
    
    # Sample conflict
    conflict_detail = ConflictDetail(
        time=5,
        location=Waypoint(10.0, 10.0, 16.0, 5),
        conflicting_drone_id='S001',
        distance=3.0,
        primary_position=Waypoint(10.0, 10.0, 15.0, 5),
        conflicting_position=Waypoint(10.0, 10.0, 18.0, 5)
    )
    conflict_result = ConflictResult('CONFLICT_DETECTED', [conflict_detail])
    
    # Generate animation
    animated_fig = animate_mission_interactive(primary_mission, simulated_missions, conflict_result)
    
    print("Animation function test completed successfully!")
    print(f"Primary drone: {primary_mission.drone_id}")
    print(f"Simulated drones: {[m.drone_id for m in simulated_missions]}")
    
    return animated_fig


def plot_conflict_heatmap_3d(conflict_result: Optional[ConflictResult]) -> go.Figure:
    """
    Create a 3D heatmap/scatter plot of conflict locations.
    
    Args:
        conflict_result: ConflictResult object containing conflict data
        
    Returns:
        plotly.graph_objects.Figure: Interactive 3D conflict heatmap
    """
    # Create a dummy primary mission for the extract function
    dummy_primary = DroneMission("PRIMARY", 0, 100, [])
    conflicts_df = extract_conflict_data(dummy_primary, conflict_result)
    
    if conflicts_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No conflict data available",
            xref="paper", yref="paper", 
            x=0.5, y=0.5,
            font=dict(size=16)
        )
        fig.update_layout(
            title="3D Conflict Heatmap - No Conflicts Detected",
            width=1000,
            height=700
        )
        return fig
    
    # Color mapping for severity
    severity_colors = {'low': 'yellow', 'medium': 'orange', 'high': 'red'}
    severity_sizes = {'low': 10, 'medium': 14, 'high': 18}
    severity_symbols = {'low': 'circle', 'medium': 'diamond', 'high': 'x'}
    
    fig = go.Figure()
    
    # Group conflicts by severity for better visualization
    for severity in ['low', 'medium', 'high']:
        severity_conflicts = conflicts_df[conflicts_df['severity'] == severity]
        
        if not severity_conflicts.empty:
            fig.add_trace(go.Scatter3d(
                x=severity_conflicts['x'],
                y=severity_conflicts['y'],
                z=severity_conflicts['z'],
                mode='markers',
                name=f'{severity.title()} Risk ({len(severity_conflicts)})',
                marker=dict(
                    size=severity_sizes[severity],
                    color=severity_colors[severity],
                    symbol=severity_symbols[severity],
                    opacity=0.8,
                    line=dict(width=2, color='black')
                ),
                hovertemplate=(
                    f'<b>{severity.upper()} RISK CONFLICT</b><br>' +
                    'Primary Drone: %{customdata[0]}<br>' +
                    'Conflicting Drone: %{customdata[1]}<br>' +
                    'Location: (%{x:.1f}, %{y:.1f}, %{z:.1f})<br>' +
                    'Time: %{customdata[2]}<br>' +
                    'Distance: %{customdata[3]:.2f}m<br>' +
                    f'Primary Position: (%{{customdata[4]:.1f}}, %{{customdata[5]:.1f}}, %{{customdata[6]:.1f}})<br>' +
                    f'Conflicting Position: (%{{customdata[7]:.1f}}, %{{customdata[8]:.1f}}, %{{customdata[9]:.1f}})<br>' +
                    '<extra></extra>'
                ),
                customdata=severity_conflicts[[
                    'drone1_id', 'drone2_id', 'time', 'distance',
                    'primary_x', 'primary_y', 'primary_z',
                    'conflicting_x', 'conflicting_y', 'conflicting_z'
                ]].values
            ))
    
    # Add connecting lines between primary and conflicting positions for high-risk conflicts
    high_risk = conflicts_df[conflicts_df['severity'] == 'high']
    if not high_risk.empty:
        for _, conflict in high_risk.iterrows():
            fig.add_trace(go.Scatter3d(
                x=[conflict['primary_x'], conflict['conflicting_x']],
                y=[conflict['primary_y'], conflict['conflicting_y']],
                z=[conflict['primary_z'], conflict['conflicting_z']],
                mode='lines',
                name='High Risk Vectors',
                line=dict(color='red', width=3, dash='dash'),
                hoverinfo='skip',
                showlegend=False,
                opacity=0.6
            ))
    
    # Calculate statistics
    total_conflicts = len(conflicts_df)
    severity_counts = conflicts_df['severity'].value_counts()
    avg_distance = conflicts_df['distance'].mean()
    min_distance = conflicts_df['distance'].min()
    
    # Update layout
    fig.update_layout(
        title={
            'text': '3D Conflict Risk Heatmap - Drone Collision Analysis',
            'x': 0.5,
            'xanchor': 'center'
        },
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Z Position - Altitude (m)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            bgcolor='rgba(240,240,240,0.1)',
            aspectmode='cube'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.9)"
        ),
        width=1000,
        height=700
    )
    
    # Add comprehensive summary statistics
    fig.add_annotation(
        text=(
            f'<b>🚨 Conflict Analysis Summary</b><br><br>' +
            f'<b>Total Conflicts:</b> {total_conflicts}<br>' +
            f'<b>High Risk:</b> {severity_counts.get("high", 0)} conflicts<br>' +
            f'<b>Medium Risk:</b> {severity_counts.get("medium", 0)} conflicts<br>' +
            f'<b>Low Risk:</b> {severity_counts.get("low", 0)} conflicts<br><br>' +
            f'<b>Closest Approach:</b> {min_distance:.2f}m<br>' +
            f'<b>Average Distance:</b> {avg_distance:.2f}m<br>' +
            f'<b>Status:</b> {conflict_result.status if conflict_result else "Unknown"}'
        ),
        xref="paper", yref="paper",
        x=0.02, y=0.25,
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="red" if severity_counts.get("high", 0) > 0 else "orange",
        borderwidth=2,
        font=dict(size=11)
    )
    
    return fig


# Example usage and testing functions
def demo_visualization():
    """Demo function showing how to use the visualization functions with actual class structures."""
    
    # Sample data using actual class structures
    primary_waypoints = [
        Waypoint(0.0, 0.0, 10.0, 0),
        Waypoint(10.0, 10.0, 15.0, 5),
        Waypoint(20.0, 20.0, 10.0, 10)
    ]
    primary_mission = DroneMission('P001', 0, 10, primary_waypoints)
    
    simulated_waypoints = [
        Waypoint(20.0, 0.0, 12.0, 0),
        Waypoint(10.0, 10.0, 18.0, 5),
        Waypoint(0.0, 20.0, 12.0, 10)
    ]
    simulated_missions = [DroneMission('S001', 0, 10, simulated_waypoints)]
    
    # Sample conflict
    conflict_detail = ConflictDetail(
        time=5,
        location=Waypoint(10.0, 10.0, 16.0, 5),
        conflicting_drone_id='S001',
        distance=3.0,
        primary_position=Waypoint(10.0, 10.0, 15.0, 5),
        conflicting_position=Waypoint(10.0, 10.0, 18.0, 5)
    )
    conflict_result = ConflictResult('CONFLICT_DETECTED', [conflict_detail])
    
    # Generate visualizations
    static_fig = plot_static_missions_interactive(primary_mission, simulated_missions, conflict_result)
    animated_fig = animate_mission_interactive(primary_mission, simulated_missions, conflict_result)
    heatmap_fig = plot_conflict_heatmap_3d(conflict_result)
    
    return static_fig, animated_fig, heatmap_fig


if __name__ == "__main__":
    # Run demo
    static, animated, heatmap = demo_visualization()
    
    # Show plots (uncomment to display)
    # static.show()
    # animated.show() 
    # heatmap.show()
    
    print("Plotly visualization module loaded successfully!")
    print("Available functions:")
    print("- plot_static_missions_interactive(primary: DroneMission, simulated: List[DroneMission], conflict_result: ConflictResult)")
    print("- animate_mission_interactive(primary: DroneMission, simulated: List[DroneMission], conflict_result: ConflictResult)")
    print("- plot_conflict_heatmap_3d(conflict_result: ConflictResult)")
