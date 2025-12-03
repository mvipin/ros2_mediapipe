#!/usr/bin/env python3
"""
Camera FPS Experiment - Report Generation Script

Analyzes collected CSV data from camera FPS experiments and generates
summary tables and visualization graphs.

Usage:
    python3 camera_fps_report.py
    python3 camera_fps_report.py --input-dir ./results --generate-readme-section

Requirements:
    - pandas, matplotlib, numpy
"""

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("Error: pandas/numpy not installed. Run: pip install pandas numpy")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available. Graphs will be skipped.")
    MATPLOTLIB_AVAILABLE = False


def load_experiment_data(input_dir: Path) -> pd.DataFrame:
    """Load all camera FPS experiment CSV files from directory."""
    csv_files = sorted(input_dir.glob('camera_fps_experiment_*.csv'))
    
    if not csv_files:
        print(f"No camera_fps_experiment_*.csv files found in {input_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(csv_files)} experiment file(s):")
    for f in csv_files:
        print(f"  - {f.name}")
    
    # Load and concatenate all CSVs
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df['source_file'] = csv_file.name
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    
    return combined


def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics for each FPS configuration."""
    if df.empty:
        return pd.DataFrame()
    
    # Group by fps_config
    stats_list = []
    
    for fps_config, group in df.groupby('fps_config'):
        # Extract numeric FPS value for sorting
        fps_match = re.search(r'(\d+)', str(fps_config))
        fps_numeric = int(fps_match.group(1)) if fps_match else 0
        
        stats = {
            'fps_config': fps_config,
            'fps_numeric': fps_numeric,
            'samples': len(group),
            'duration_s': (group['timestamp'].max() - group['timestamp'].min()).total_seconds(),
        }
        
        # CPU statistics
        cpu_data = group['cpu_percent'].dropna()
        if len(cpu_data) > 0:
            stats['cpu_mean'] = cpu_data.mean()
            stats['cpu_std'] = cpu_data.std()
            stats['cpu_p95'] = np.percentile(cpu_data, 95)
            stats['cpu_max'] = cpu_data.max()
        
        # Temperature statistics
        temp_data = group['temperature_c'].dropna()
        if len(temp_data) > 0:
            stats['temp_start'] = temp_data.iloc[0]
            stats['temp_end'] = temp_data.iloc[-1]
            stats['temp_max'] = temp_data.max()
            stats['temp_mean'] = temp_data.mean()
        
        # Camera Hz statistics
        cam_hz = group['camera_hz'].dropna()
        if len(cam_hz) > 0:
            stats['camera_hz_mean'] = cam_hz.mean()
            stats['camera_hz_std'] = cam_hz.std()
        
        # Detection Hz statistics
        det_hz = group['detection_hz'].dropna()
        if len(det_hz) > 0:
            stats['detection_hz_mean'] = det_hz.mean()
            stats['detection_hz_std'] = det_hz.std()
        
        # Calculate frame drop rate (camera_hz - detection_hz) / camera_hz
        if stats.get('camera_hz_mean') and stats.get('detection_hz_mean'):
            if stats['camera_hz_mean'] > 0:
                stats['frame_drop_rate'] = max(0, (
                    (stats['camera_hz_mean'] - stats['detection_hz_mean']) 
                    / stats['camera_hz_mean'] * 100
                ))
        
        stats_list.append(stats)
    
    stats_df = pd.DataFrame(stats_list)
    stats_df = stats_df.sort_values('fps_numeric')
    
    return stats_df


def generate_markdown_summary(stats_df: pd.DataFrame, output_file: Path) -> None:
    """Generate markdown summary table."""
    if stats_df.empty:
        print("No statistics to summarize.")
        return
    
    lines = [
        "# Camera FPS Optimization - Experiment Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary Table",
        "",
        "| FPS Config | Samples | CPU Mean (%) | CPU P95 (%) | Temp Start (°C) | Temp Max (°C) | Camera Hz | Detection Hz | Drop Rate (%) |",
        "|------------|---------|--------------|-------------|-----------------|---------------|-----------|--------------|---------------|",
    ]

    for _, row in stats_df.iterrows():
        cpu_mean = f"{row.get('cpu_mean', 0):.1f}" if pd.notna(row.get('cpu_mean')) else "N/A"
        cpu_p95 = f"{row.get('cpu_p95', 0):.1f}" if pd.notna(row.get('cpu_p95')) else "N/A"
        temp_start = f"{row.get('temp_start', 0):.1f}" if pd.notna(row.get('temp_start')) else "N/A"
        temp_max = f"{row.get('temp_max', 0):.1f}" if pd.notna(row.get('temp_max')) else "N/A"
        cam_hz = f"{row.get('camera_hz_mean', 0):.2f}" if pd.notna(row.get('camera_hz_mean')) else "N/A"
        det_hz = f"{row.get('detection_hz_mean', 0):.2f}" if pd.notna(row.get('detection_hz_mean')) else "N/A"
        drop_rate = f"{row.get('frame_drop_rate', 0):.1f}" if pd.notna(row.get('frame_drop_rate')) else "N/A"

        lines.append(
            f"| {row['fps_config']} | {row['samples']} | {cpu_mean} | {cpu_p95} | "
            f"{temp_start} | {temp_max} | {cam_hz} | {det_hz} | {drop_rate} |"
        )

    lines.extend([
        "",
        "## Key Findings",
        "",
        "- **Recommended FPS**: [To be determined based on results]",
        "- **CPU Threshold**: Target < 80% sustained",
        "- **Temperature Threshold**: Target < 75°C",
        "- **Drop Rate Threshold**: Target < 20%",
        "",
    ])

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Markdown summary saved to: {output_file}")


def generate_readme_section(stats_df: pd.DataFrame) -> str:
    """Generate formatted text for README Performance Tuning section."""
    if stats_df.empty:
        return "No data available for README section."

    lines = [
        "#### Camera Frame Rate (FrameDurationLimits)",
        "",
        "The following table shows measured performance for different camera frame rates on Raspberry Pi 5:",
        "",
        "| Frame Rate | FrameDurationLimits | CPU Mean | CPU P95 | Temp Max | Drop Rate | Recommendation |",
        "|------------|---------------------|----------|---------|----------|-----------|----------------|",
    ]

    # Map fps_config to FrameDurationLimits values
    fps_to_limits = {
        '5fps': '[200000, 200000]',
        '10fps': '[100000, 100000]',
        '15fps': '[66667, 66667]',
        '20fps': '[50000, 50000]',
    }

    for _, row in stats_df.iterrows():
        fps_config = row['fps_config']
        limits = fps_to_limits.get(fps_config, 'N/A')

        cpu_mean = row.get('cpu_mean', 0)
        cpu_p95 = row.get('cpu_p95', 0)
        temp_max = row.get('temp_max', 0)
        drop_rate = row.get('frame_drop_rate', 0)

        # Determine recommendation
        if pd.notna(cpu_p95) and pd.notna(temp_max) and pd.notna(drop_rate):
            if cpu_p95 < 80 and temp_max < 75 and drop_rate < 20:
                recommendation = "✅ Viable"
            elif cpu_p95 < 90 and temp_max < 80 and drop_rate < 30:
                recommendation = "⚠️ Marginal"
            else:
                recommendation = "❌ Too High"
        else:
            recommendation = "N/A"

        cpu_mean_str = f"{cpu_mean:.1f}%" if pd.notna(cpu_mean) else "N/A"
        cpu_p95_str = f"{cpu_p95:.1f}%" if pd.notna(cpu_p95) else "N/A"
        temp_max_str = f"{temp_max:.1f}°C" if pd.notna(temp_max) else "N/A"
        drop_rate_str = f"{drop_rate:.1f}%" if pd.notna(drop_rate) else "N/A"

        lines.append(
            f"| {fps_config} | `{limits}` | {cpu_mean_str} | {cpu_p95_str} | "
            f"{temp_max_str} | {drop_rate_str} | {recommendation} |"
        )

    lines.extend([
        "",
        "**Note:** Camera frame rate is controlled via `FrameDurationLimits` in the launch file (microseconds).",
        "Lower values = higher frame rate. Values are hardcoded for optimal Pi 5 performance.",
        "",
    ])

    return '\n'.join(lines)


def generate_graphs(df: pd.DataFrame, stats_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate visualization graphs."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping graph generation (matplotlib not available)")
        return

    if df.empty:
        print("No data for graph generation")
        return

    plt.style.use('seaborn-v0_8-whitegrid')

    # Get unique FPS configs sorted by numeric value
    fps_configs = stats_df['fps_config'].tolist()
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(fps_configs)))
    color_map = dict(zip(fps_configs, colors))

    # =========================================================================
    # Graph 1: CPU Usage Over Time
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for fps_config in fps_configs:
        group = df[df['fps_config'] == fps_config].copy()
        if len(group) > 0:
            # Convert to elapsed seconds
            group['elapsed_s'] = (group['timestamp'] - group['timestamp'].min()).dt.total_seconds()
            ax.plot(
                group['elapsed_s'], group['cpu_percent'],
                label=fps_config, color=color_map[fps_config], linewidth=1.5
            )

    ax.set_xlabel('Elapsed Time (seconds)')
    ax.set_ylabel('CPU Usage (%)')
    ax.set_title('CPU Usage Over Time by FPS Configuration')
    ax.legend(title='FPS Config')
    ax.set_ylim(0, 100)
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% Threshold')

    plt.tight_layout()
    cpu_time_file = output_dir / 'camera_fps_cpu_over_time.png'
    plt.savefig(cpu_time_file, dpi=150)
    plt.close()
    print(f"Graph saved: {cpu_time_file}")

    # =========================================================================
    # Graph 2: Temperature Over Time
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for fps_config in fps_configs:
        group = df[df['fps_config'] == fps_config].copy()
        if len(group) > 0:
            group['elapsed_s'] = (group['timestamp'] - group['timestamp'].min()).dt.total_seconds()
            ax.plot(
                group['elapsed_s'], group['temperature_c'],
                label=fps_config, color=color_map[fps_config], linewidth=1.5
            )

    ax.set_xlabel('Elapsed Time (seconds)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Over Time by FPS Configuration')
    ax.legend(title='FPS Config')
    ax.axhline(y=75, color='red', linestyle='--', alpha=0.5, label='75°C Threshold')

    plt.tight_layout()
    temp_time_file = output_dir / 'camera_fps_temp_over_time.png'
    plt.savefig(temp_time_file, dpi=150)
    plt.close()
    print(f"Graph saved: {temp_time_file}")

    # =========================================================================
    # Graph 3: Bar Chart - Mean CPU Usage Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))

    x_pos = np.arange(len(fps_configs))
    cpu_means = [stats_df[stats_df['fps_config'] == c]['cpu_mean'].values[0]
                 for c in fps_configs if c in stats_df['fps_config'].values]
    cpu_stds = [stats_df[stats_df['fps_config'] == c]['cpu_std'].values[0]
                for c in fps_configs if c in stats_df['fps_config'].values]

    bars = ax.bar(x_pos, cpu_means, yerr=cpu_stds, capsize=5,
                  color=[color_map[c] for c in fps_configs], edgecolor='black')

    ax.set_xlabel('FPS Configuration')
    ax.set_ylabel('CPU Usage (%)')
    ax.set_title('Mean CPU Usage by FPS Configuration')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fps_configs)
    ax.set_ylim(0, 100)
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% Threshold')

    # Add value labels on bars
    for bar, val in zip(bars, cpu_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    cpu_bar_file = output_dir / 'camera_fps_cpu_comparison.png'
    plt.savefig(cpu_bar_file, dpi=150)
    plt.close()
    print(f"Graph saved: {cpu_bar_file}")

    # =========================================================================
    # Graph 4: Bar Chart - Frame Drop Rate Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))

    drop_rates = []
    for c in fps_configs:
        if c in stats_df['fps_config'].values:
            rate = stats_df[stats_df['fps_config'] == c]['frame_drop_rate'].values[0]
            drop_rates.append(rate if pd.notna(rate) else 0)
        else:
            drop_rates.append(0)

    bars = ax.bar(x_pos, drop_rates, color=[color_map[c] for c in fps_configs], edgecolor='black')

    ax.set_xlabel('FPS Configuration')
    ax.set_ylabel('Frame Drop Rate (%)')
    ax.set_title('Frame Drop Rate by FPS Configuration')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fps_configs)
    ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='20% Threshold')

    # Add value labels on bars
    for bar, val in zip(bars, drop_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    drop_bar_file = output_dir / 'camera_fps_drop_rate.png'
    plt.savefig(drop_bar_file, dpi=150)
    plt.close()
    print(f"Graph saved: {drop_bar_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Camera FPS Experiment - Report Generation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 camera_fps_report.py
  python3 camera_fps_report.py --input-dir ./results
  python3 camera_fps_report.py --generate-readme-section
        """
    )
    parser.add_argument(
        '--input-dir', type=str,
        default='gesturebot_ws/src/ros2_mediapipe/benchmark/results',
        help='Directory containing CSV files'
    )
    parser.add_argument(
        '--generate-readme-section', action='store_true',
        help='Generate formatted text for README Performance Tuning section'
    )

    args = parser.parse_args()

    # Resolve input directory
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = Path.home() / 'GestureBot' / args.input_dir

    print("=" * 60)
    print("Camera FPS Experiment - Report Generation")
    print("=" * 60)
    print(f"Input Directory: {input_dir}")
    print("=" * 60)

    # Load data
    df = load_experiment_data(input_dir)

    if df.empty:
        print("\nNo data to analyze. Exiting.")
        sys.exit(1)

    print(f"\nLoaded {len(df)} total samples")
    print(f"FPS configurations: {df['fps_config'].unique().tolist()}")

    # Calculate statistics
    print("\nCalculating statistics...")
    stats_df = calculate_statistics(df)

    # Print summary to console
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(stats_df.to_string(index=False))

    # Generate markdown summary
    summary_file = input_dir / 'camera_fps_summary.md'
    generate_markdown_summary(stats_df, summary_file)

    # Generate graphs
    print("\nGenerating graphs...")
    generate_graphs(df, stats_df, input_dir)

    # Generate README section if requested
    if args.generate_readme_section:
        print("\n" + "=" * 60)
        print("README SECTION (copy below)")
        print("=" * 60)
        readme_section = generate_readme_section(stats_df)
        print(readme_section)

        # Also save to file
        readme_section_file = input_dir / 'camera_fps_readme_section.md'
        with open(readme_section_file, 'w') as f:
            f.write(readme_section)
        print(f"\nREADME section also saved to: {readme_section_file}")

    print("\n" + "=" * 60)
    print("Report generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

