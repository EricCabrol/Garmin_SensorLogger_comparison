#!/usr/bin/env python3
"""
Comparison of Garmin Forerunner 735 XT and SensorLogger (Pixel 4 XL) recordings
from January 21st, 2024 running activity.

This script:
1. Parses GPS data from both devices
2. Converts coordinates to decimal degrees
3. Aligns timestamps between devices
4. Creates interactive map with both paths
5. Creates 2D plots of raw and filtered velocities
6. Calculates comparative metrics

Usage:
    python analyze_recordings.py
    # or with uv:
    uv run python analyze_recordings.py
"""

import pandas as pd
import numpy as np
import fitdecode
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import folium
from scipy.signal import savgol_filter
import os
import sys


# ============================================================================
# CONFIGURATION
# ============================================================================

FIT_FILE = 'data/13603885354_ACTIVITY.fit'
LOCATION_CSV = 'data/Location.csv'
OUTPUT_DIR = 'output'

# Filter parameters for velocity smoothing
WINDOW_LENGTH = 15  # Must be odd
POLYORDER = 2

# Map settings
MAP_ZOOM = 15

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def semicircles_to_degrees(semicircles):
    """Convert Garmin semicircles to decimal degrees."""
    return semicircles * 180 / (2 ** 31)


def nanoseconds_to_datetime(ns):
    """Convert nanoseconds timestamp to datetime."""
    # SensorLogger uses nanoseconds since epoch
    return datetime.fromtimestamp(ns / 1e9)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS points in meters."""
    R = 6371000  # Earth radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = (np.sin(delta_phi / 2) ** 2 + 
         np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def calculate_velocity_from_positions(df, time_col='time', lat_col='latitude', lon_col='longitude'):
    """Calculate velocity from consecutive GPS positions."""
    velocities = []
    times = pd.to_datetime(df[time_col])
    
    for i in range(1, len(df)):
        dt = (times[i] - times[i-1]).total_seconds()
        if dt > 0:
            dist = haversine_distance(
                df[lat_col].iloc[i-1], df[lon_col].iloc[i-1],
                df[lat_col].iloc[i], df[lon_col].iloc[i]
            )
            velocity = dist / dt  # m/s
            velocities.append(velocity)
        else:
            velocities.append(np.nan)
    
    velocities.insert(0, np.nan)  # First point has no previous
    return velocities


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_analysis(fit_file=FIT_FILE, location_csv=LOCATION_CSV, output_dir=OUTPUT_DIR):
    """Run the full analysis pipeline."""
    
    # ============================================================================
    # DATA LOADING
    # ============================================================================

    print("Loading data...")

    # Load SensorLogger data
    print("  Loading SensorLogger CSV...")
    df_sensor = pd.read_csv(location_csv)
    print(f"  SensorLogger: {len(df_sensor)} records")

    # Load Garmin FIT data
    print("  Loading Garmin FIT file...")
    garmin_records = []
    with fitdecode.FitReader(fit_file) as fit:
        for frame in fit:
            if hasattr(frame, 'fields'):
                record = {}
                for field in frame.fields:
                    record[field.name] = field.value
                # Only keep records with position data
                if 'position_lat' in record and 'position_long' in record:
                    garmin_records.append(record)

    df_garmin = pd.DataFrame(garmin_records)
    print(f"  Garmin: {len(df_garmin)} GPS records")

    # ============================================================================
    # DATA PROCESSING
    # ============================================================================

    print("\nProcessing SensorLogger data...")

    # Convert nanoseconds to datetime
    df_sensor['time_dt'] = df_sensor['time'].apply(nanoseconds_to_datetime)

    # Rename columns for consistency
    df_sensor = df_sensor.rename(columns={
        'latitude': 'lat',
        'longitude': 'lon'
    })

    # Add device identifier
    df_sensor['device'] = 'SensorLogger (Pixel 4 XL)'

    # Calculate velocity from positions if speed is not reliable
    # We'll use both the reported speed and calculated speed
    df_sensor['speed_calculated'] = calculate_velocity_from_positions(
        df_sensor, time_col='time_dt', lat_col='lat', lon_col='lon'
    )

    # Also calculate distance from positions for SensorLogger
    distance_array = np.zeros(len(df_sensor))
    for i in range(1, len(df_sensor)):
        if not pd.isna(df_sensor['lat'].iloc[i]) and not pd.isna(df_sensor['lat'].iloc[i-1]):
            dist = haversine_distance(
                df_sensor['lat'].iloc[i-1], df_sensor['lon'].iloc[i-1],
                df_sensor['lat'].iloc[i], df_sensor['lon'].iloc[i]
            )
            distance_array[i] = distance_array[i-1] + dist
        else:
            distance_array[i] = distance_array[i-1]
    df_sensor['distance_from_start'] = distance_array

    print(f"  Time range: {df_sensor['time_dt'].min()} to {df_sensor['time_dt'].max()}")

    print("\nProcessing Garmin data...")

    # Convert semicircles to degrees
    df_garmin['lat'] = semicircles_to_degrees(df_garmin['position_lat'])
    df_garmin['lon'] = semicircles_to_degrees(df_garmin['position_long'])

    # Convert timestamp to datetime (it's already a datetime object from fitdecode)
    # Make timezone naive for comparison with SensorLogger
    df_garmin['time_dt'] = pd.to_datetime(df_garmin['timestamp']).dt.tz_localize(None)

    # Add device identifier
    df_garmin['device'] = 'Garmin Forerunner 735 XT'

    # Use enhanced_speed if available, otherwise speed
    df_garmin['speed'] = df_garmin.get('enhanced_speed', df_garmin.get('speed', np.nan))

    # Calculate velocity from positions for comparison
    df_garmin['speed_calculated'] = calculate_velocity_from_positions(
        df_garmin, time_col='time_dt', lat_col='lat', lon_col='lon'
    )

    print(f"  Time range: {df_garmin['time_dt'].min()} to {df_garmin['time_dt'].max()}")

    # Also calculate distance from positions for Garmin
    distance_array = np.zeros(len(df_garmin))
    for i in range(1, len(df_garmin)):
        if not pd.isna(df_garmin['lat'].iloc[i]) and not pd.isna(df_garmin['lat'].iloc[i-1]):
            dist = haversine_distance(
                df_garmin['lat'].iloc[i-1], df_garmin['lon'].iloc[i-1],
                df_garmin['lat'].iloc[i], df_garmin['lon'].iloc[i]
            )
            distance_array[i] = distance_array[i-1] + dist
        else:
            distance_array[i] = distance_array[i-1]
    df_garmin['distance_from_start'] = distance_array

    # ============================================================================
    # TIME ALIGNMENT
    # ============================================================================

    print("\nAligning timestamps...")

    # Find time offset between devices
    # Use the first common time period
    sensor_start = df_sensor['time_dt'].min()
    garmin_start = df_garmin['time_dt'].min()

    print(f"  SensorLogger start: {sensor_start}")
    print(f"  Garmin start: {garmin_start}")

    # The Garmin starts later, so we need to find overlapping period
    max_start = max(sensor_start, garmin_start)
    min_end = min(df_sensor['time_dt'].max(), df_garmin['time_dt'].max())

    print(f"  Overlapping period: {max_start} to {min_end}")

    # Filter both datasets to overlapping period
    df_sensor_aligned = df_sensor[(df_sensor['time_dt'] >= max_start) & (df_sensor['time_dt'] <= min_end)].copy()
    df_garmin_aligned = df_garmin[(df_garmin['time_dt'] >= max_start) & (df_garmin['time_dt'] <= min_end)].copy()

    print(f"  Aligned SensorLogger: {len(df_sensor_aligned)} records")
    print(f"  Aligned Garmin: {len(df_garmin_aligned)} records")

    # Interpolate SensorLogger data to match Garmin timestamps for better comparison
    # This is important because SensorLogger has more data points
    print("\n  Interpolating SensorLogger data to Garmin timestamps...")
    df_sensor_aligned = df_sensor_aligned.set_index('time_dt').sort_index()
    df_garmin_aligned = df_garmin_aligned.set_index('time_dt').sort_index()

    # Reindex SensorLogger to Garmin timestamps and interpolate
    df_sensor_interp = df_sensor_aligned.reindex(df_garmin_aligned.index, method='nearest')

    # Now both datasets have the same timestamps (Garmin's timestamps)
    df_sensor_aligned = df_sensor_interp.reset_index().copy()
    df_garmin_aligned = df_garmin_aligned.reset_index().copy()

    print(f"  After interpolation: Both datasets have {len(df_garmin_aligned)} records")

    # ============================================================================
    # APPLY VELOCITY FILTERING
    # ============================================================================

    print("\nApplying velocity filters...")

    # Function to apply Savitzky-Golay filter
    def apply_filter(series, window_length=WINDOW_LENGTH, polyorder=POLYORDER):
        """Apply Savitzky-Golay filter to smooth velocity data."""
        try:
            return savgol_filter(series, window_length=window_length, polyorder=polyorder)
        except:
            return series

    # Filter reported speeds
    df_sensor_aligned['speed_filtered'] = apply_filter(df_sensor_aligned['speed'])
    df_garmin_aligned['speed_filtered'] = apply_filter(df_garmin_aligned['speed'])

    # Also filter calculated speeds
    df_sensor_aligned['speed_calculated_filtered'] = apply_filter(df_sensor_aligned['speed_calculated'])
    df_garmin_aligned['speed_calculated_filtered'] = apply_filter(df_garmin_aligned['speed_calculated'])

    # ============================================================================
    # CREATE OUTPUT DIRECTORY
    # ============================================================================

    os.makedirs(output_dir, exist_ok=True)

    # ============================================================================
    # VISUALIZATION 1: INTERACTIVE MAP
    # ============================================================================

    print("\nCreating interactive map...")

    # Calculate average position for map center
    avg_lat = (df_sensor_aligned['lat'].mean() + df_garmin_aligned['lat'].mean()) / 2
    avg_lon = (df_sensor_aligned['lon'].mean() + df_garmin_aligned['lon'].mean()) / 2

    # Create map
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=MAP_ZOOM, tiles='OpenStreetMap')

    # Add Garmin path (blue)
    garmin_coords = list(zip(df_garmin_aligned['lat'], df_garmin_aligned['lon']))
    folium.PolyLine(
        garmin_coords,
        color='blue',
        weight=4,
        opacity=0.8,
        popup='Garmin Forerunner 735 XT',
        tooltip='Garmin'
    ).add_to(m)

    # Add SensorLogger path (red)
    sensor_coords = list(zip(df_sensor_aligned['lat'], df_sensor_aligned['lon']))
    folium.PolyLine(
        sensor_coords,
        color='red',
        weight=4,
        opacity=0.8,
        popup='SensorLogger (Pixel 4 XL)',
        tooltip='SensorLogger'
    ).add_to(m)

    # Add start markers
    folium.CircleMarker(
        [df_garmin_aligned['lat'].iloc[0], df_garmin_aligned['lon'].iloc[0]],
        radius=6,
        color='blue',
        fill=True,
        fill_color='blue',
        popup=f'Garmin Start: {df_garmin_aligned["time_dt"].iloc[0]}'
    ).add_to(m)

    folium.CircleMarker(
        [df_sensor_aligned['lat'].iloc[0], df_sensor_aligned['lon'].iloc[0]],
        radius=6,
        color='red',
        fill=True,
        fill_color='red',
        popup=f'SensorLogger Start: {df_sensor_aligned["time_dt"].iloc[0]}'
    ).add_to(m)

    # Save map
    map_file = os.path.join(output_dir, 'path_comparison.html')
    m.save(map_file)
    print(f"  Map saved to: {map_file}")

    # ============================================================================
    # VISUALIZATION 2: VELOCITY PLOTS
    # ============================================================================

    print("\nCreating velocity plots...")

    # Create figure with multiple subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    # Convert timestamps to seconds since start for easier plotting
    base_time = max_start

    df_sensor_aligned['seconds'] = (df_sensor_aligned['time_dt'] - base_time).dt.total_seconds()
    df_garmin_aligned['seconds'] = (df_garmin_aligned['time_dt'] - base_time).dt.total_seconds()

    # Plot 1: Raw velocities
    axes[0].plot(df_garmin_aligned['seconds'], df_garmin_aligned['speed'], 
                 label='Garmin Raw', color='blue', alpha=0.7, linewidth=1)
    axes[0].plot(df_sensor_aligned['seconds'], df_sensor_aligned['speed'], 
                 label='SensorLogger Raw', color='red', alpha=0.7, linewidth=1)
    axes[0].set_xlabel('Time (seconds since start)')
    axes[0].set_ylabel('Velocity (m/s)')
    axes[0].set_title('Raw Velocity Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Filtered velocities
    axes[1].plot(df_garmin_aligned['seconds'], df_garmin_aligned['speed_filtered'], 
                 label='Garmin Filtered', color='blue', linewidth=2)
    axes[1].plot(df_sensor_aligned['seconds'], df_sensor_aligned['speed_filtered'], 
                 label='SensorLogger Filtered', color='red', linewidth=2)
    axes[1].set_xlabel('Time (seconds since start)')
    axes[1].set_ylabel('Velocity (m/s)')
    axes[1].set_title('Filtered Velocity Comparison (Savitzky-Golay)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Calculated velocities (from GPS positions)
    axes[2].plot(df_garmin_aligned['seconds'], df_garmin_aligned['speed_calculated'], 
                 label='Garmin Calculated', color='blue', alpha=0.7, linewidth=1)
    axes[2].plot(df_sensor_aligned['seconds'], df_sensor_aligned['speed_calculated'], 
                 label='SensorLogger Calculated', color='red', alpha=0.7, linewidth=1)
    axes[2].set_xlabel('Time (seconds since start)')
    axes[2].set_ylabel('Velocity (m/s)')
    axes[2].set_title('Calculated Velocity from GPS Positions')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(1, 3)

    # Plot 4: Filtered calculated velocities
    axes[3].plot(df_garmin_aligned['seconds'], df_garmin_aligned['speed_calculated_filtered'], 
                 label='Garmin Calculated Filtered', color='blue', linewidth=2)
    axes[3].plot(df_sensor_aligned['seconds'], df_sensor_aligned['speed_calculated_filtered'], 
                 label='SensorLogger Calculated Filtered', color='red', linewidth=2)
    axes[3].set_xlabel('Time (seconds since start)')
    axes[3].set_ylabel('Velocity (m/s)')
    axes[3].set_title('Filtered Calculated Velocity')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim(1, 3)

    plt.tight_layout()
    velocity_plot_file = os.path.join(output_dir, 'velocity_comparison.png')
    plt.savefig(velocity_plot_file, dpi=150, bbox_inches='tight')
    print(f"  Velocity plots saved to: {velocity_plot_file}")
    plt.close()

    # ============================================================================
    # VISUALIZATION 3: OVERLAY COMPARISON
    # ============================================================================

    print("\nCreating overlay comparison plot...")

    plt.figure(figsize=(14, 8))

    # Plot both filtered velocities
    plt.plot(df_garmin_aligned['seconds'], df_garmin_aligned['speed_filtered'], 
             label='Garmin Forerunner 735 XT', color='blue', linewidth=2)
    plt.plot(df_sensor_aligned['seconds'], df_sensor_aligned['speed_filtered'], 
             label='SensorLogger (Pixel 4 XL)', color='red', linewidth=2)

    plt.xlabel('Time (seconds since start)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity Comparison: Garmin vs SensorLogger\n(Filtered with Savitzky-Golay, window=15, polyorder=2)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    overlay_file = os.path.join(output_dir, 'velocity_overlay.png')
    plt.savefig(overlay_file, dpi=150, bbox_inches='tight')
    print(f"  Overlay plot saved to: {overlay_file}")
    plt.close()

    # ============================================================================
    # CALCULATE COMPARATIVE METRICS
    # ============================================================================

    print("\nCalculating comparative metrics...")

    # We need to resample both datasets to a common time grid for fair comparison
    # Create a common time index
    min_time = max(df_garmin_aligned['time_dt'].min(), df_sensor_aligned['time_dt'].min())
    max_time = min(df_garmin_aligned['time_dt'].max(), df_sensor_aligned['time_dt'].max())

    # Resample to 1Hz (1 second intervals)
    time_range = pd.date_range(start=min_time, end=max_time, freq='s')

    # Resample Garmin data
    garmin_resampled = df_garmin_aligned.set_index('time_dt')[['speed', 'lat', 'lon']].resample('s').mean()
    garmin_resampled['device'] = 'Garmin'

    # Resample SensorLogger data  
    sensor_resampled = df_sensor_aligned.set_index('time_dt')[['speed', 'lat', 'lon']].resample('s').mean()
    sensor_resampled['device'] = 'SensorLogger'

    # Align both datasets
    common_index = garmin_resampled.index.intersection(sensor_resampled.index)
    garmin_final = garmin_resampled.loc[common_index].copy()
    sensor_final = sensor_resampled.loc[common_index].copy()

    print(f"  Common time points: {len(common_index)}")

    # Calculate metrics
    metrics = {}

    # 1. Total distance for each device (calculate from aligned datasets, not resampled)
    garmin_distance = 0
    for i in range(1, len(df_garmin_aligned)):
        if not pd.isna(df_garmin_aligned['lat'].iloc[i]) and not pd.isna(df_garmin_aligned['lat'].iloc[i-1]):
            garmin_distance += haversine_distance(
                df_garmin_aligned['lat'].iloc[i-1], df_garmin_aligned['lon'].iloc[i-1],
                df_garmin_aligned['lat'].iloc[i], df_garmin_aligned['lon'].iloc[i]
            )

    sensor_distance = 0
    for i in range(1, len(df_sensor_aligned)):
        if not pd.isna(df_sensor_aligned['lat'].iloc[i]) and not pd.isna(df_sensor_aligned['lat'].iloc[i-1]):
            sensor_distance += haversine_distance(
                df_sensor_aligned['lat'].iloc[i-1], df_sensor_aligned['lon'].iloc[i-1],
                df_sensor_aligned['lat'].iloc[i], df_sensor_aligned['lon'].iloc[i]
            )

    metrics['garmin_total_distance_m'] = garmin_distance
    metrics['sensor_total_distance_m'] = sensor_distance
    metrics['distance_difference_m'] = abs(garmin_distance - sensor_distance)
    metrics['distance_difference_pct'] = (abs(garmin_distance - sensor_distance) / garmin_distance * 100) if garmin_distance > 0 else 0

    # 2. Velocity correlation
    valid_mask = (~garmin_final['speed'].isna()) & (~sensor_final['speed'].isna())
    if valid_mask.sum() > 0:
        correlation = garmin_final['speed'][valid_mask].corr(sensor_final['speed'][valid_mask])
        metrics['velocity_correlation'] = correlation
    else:
        metrics['velocity_correlation'] = np.nan

    # 3. RMSE for velocity
    if valid_mask.sum() > 0:
        rmse = np.sqrt(np.mean((garmin_final['speed'][valid_mask] - sensor_final['speed'][valid_mask])**2))
        metrics['velocity_rmse'] = rmse
    else:
        metrics['velocity_rmse'] = np.nan

    # 4. Mean velocity difference
    if valid_mask.sum() > 0:
        mean_diff = np.mean(garmin_final['speed'][valid_mask] - sensor_final['speed'][valid_mask])
        metrics['velocity_mean_difference'] = mean_diff
    else:
        metrics['velocity_mean_difference'] = np.nan

    # 5. Path deviation (average distance between corresponding points)
    # Use the aligned datasets directly for more accurate comparison
    total_deviation = 0
    count = 0
    min_len = min(len(df_garmin_aligned), len(df_sensor_aligned))
    for i in range(min_len):
        if not pd.isna(df_garmin_aligned['lat'].iloc[i]) and not pd.isna(df_sensor_aligned['lat'].iloc[i]):
            dev = haversine_distance(
                df_garmin_aligned['lat'].iloc[i], df_garmin_aligned['lon'].iloc[i],
                df_sensor_aligned['lat'].iloc[i], df_sensor_aligned['lon'].iloc[i]
            )
            total_deviation += dev
            count += 1

    metrics['avg_path_deviation_m'] = total_deviation / count if count > 0 else 0

    # 6. Max path deviation
    max_deviation = 0
    for i in range(min_len):
        if not pd.isna(df_garmin_aligned['lat'].iloc[i]) and not pd.isna(df_sensor_aligned['lat'].iloc[i]):
            dev = haversine_distance(
                df_garmin_aligned['lat'].iloc[i], df_garmin_aligned['lon'].iloc[i],
                df_sensor_aligned['lat'].iloc[i], df_sensor_aligned['lon'].iloc[i]
            )
            max_deviation = max(max_deviation, dev)

    metrics['max_path_deviation_m'] = max_deviation

    # Print metrics
    print("\n" + "="*60)
    print("COMPARATIVE METRICS")
    print("="*60)
    print(f"Garmin Total Distance:        {metrics['garmin_total_distance_m']:.2f} m")
    print(f"SensorLogger Total Distance:  {metrics['sensor_total_distance_m']:.2f} m")
    print(f"Distance Difference:          {metrics['distance_difference_m']:.2f} m ({metrics['distance_difference_pct']:.2f}%)")
    print(f"Velocity Correlation:         {metrics['velocity_correlation']:.4f}")
    print(f"Velocity RMSE:               {metrics['velocity_rmse']:.4f} m/s")
    print(f"Velocity Mean Difference:    {metrics['velocity_mean_difference']:.4f} m/s")
    print(f"Average Path Deviation:      {metrics['avg_path_deviation_m']:.2f} m")
    print(f"Maximum Path Deviation:      {metrics['max_path_deviation_m']:.2f} m")
    print("="*60)

    # ============================================================================
    # SAVE METRICS TO FILE
    # ============================================================================

    metrics_file = os.path.join(output_dir, 'comparison_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("COMPARATIVE METRICS\n")
        f.write("="*60 + "\n")
        f.write(f"Analysis Date: {datetime.now()}\n")
        f.write(f"Garmin FIT File: {fit_file}\n")
        f.write(f"SensorLogger CSV: {location_csv}\n")
        f.write("\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
            else:
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")

    print(f"\nMetrics saved to: {metrics_file}")

    # ============================================================================
    # SAVE PROCESSED DATA TO CSV
    # ============================================================================

    print("\nSaving processed data...")

    df_garmin_aligned.to_csv(os.path.join(output_dir, 'garmin_processed.csv'), index=False)
    df_sensor_aligned.to_csv(os.path.join(output_dir, 'sensorlogger_processed.csv'), index=False)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"  - {os.path.basename(map_file)}")
    print(f"  - {os.path.basename(velocity_plot_file)}")
    print(f"  - {os.path.basename(overlay_file)}")
    print(f"  - {os.path.basename(metrics_file)}")
    print(f"  - garmin_processed.csv")
    print(f"  - sensorlogger_processed.csv")
    print("="*60)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Entry point for running as a module."""
    run_analysis()


if __name__ == "__main__":
    main()
