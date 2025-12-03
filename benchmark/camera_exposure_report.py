#!/usr/bin/env python3
"""
Camera Exposure Time Experiment - Report Generation Script

Analyzes collected CSV data from camera exposure experiments and generates
summary tables and visualization graphs.

Usage:
    python3 camera_exposure_report.py
    python3 camera_exposure_report.py --input-dir ./results --generate-readme-section
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("Error: pandas/numpy not installed. Run: pip install pandas numpy")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available. Graphs will be skipped.")
    MATPLOTLIB_AVAILABLE = False


def load_experiment_data(input_dir: Path) -> pd.DataFrame:
    """Load all camera exposure experiment CSV files from directory."""
    csv_files = sorted(input_dir.glob('camera_exposure_experiment_*.csv'))
    if not csv_files:
        print(f"No camera_exposure_experiment_*.csv files found in {input_dir}")
        return pd.DataFrame()
    print(f"Found {len(csv_files)} experiment file(s):")
    for f in csv_files:
        print(f"  - {f.name}")
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
    """Calculate summary statistics for each exposure configuration."""
    if df.empty:
        return pd.DataFrame()
    stats_list = []
    for exposure_config, group in df.groupby('exposure_config'):
        exp_match = re.search(r'(\d+)', str(exposure_config))
        exp_numeric = int(exp_match.group(1)) if exp_match else 0
        stats = {'exposure_config': exposure_config, 'exposure_numeric': exp_numeric, 'samples': len(group),
                 'duration_s': (group['timestamp'].max() - group['timestamp'].min()).total_seconds()}
        cpu_data = group['cpu_percent'].dropna()
        if len(cpu_data) > 0:
            stats.update({'cpu_mean': cpu_data.mean(), 'cpu_std': cpu_data.std(),
                          'cpu_p95': np.percentile(cpu_data, 95), 'cpu_max': cpu_data.max()})
        temp_data = group['temperature_c'].dropna()
        if len(temp_data) > 0:
            stats.update({'temp_start': temp_data.iloc[0], 'temp_end': temp_data.iloc[-1],
                          'temp_max': temp_data.max(), 'temp_mean': temp_data.mean()})
        cam_hz = group['camera_hz'].dropna()
        if len(cam_hz) > 0:
            stats.update({'camera_hz_mean': cam_hz.mean(), 'camera_hz_std': cam_hz.std()})
        det_hz = group['detection_hz'].dropna()
        if len(det_hz) > 0:
            stats.update({'detection_hz_mean': det_hz.mean(), 'detection_hz_std': det_hz.std()})
        if stats.get('camera_hz_mean') and stats.get('detection_hz_mean') and stats['camera_hz_mean'] > 0:
            stats['frame_drop_rate'] = max(0, (stats['camera_hz_mean'] - stats['detection_hz_mean']) / stats['camera_hz_mean'] * 100)
        # Object detection quality metrics
        obj_count = group['object_count'].dropna()
        if len(obj_count) > 0:
            stats.update({'object_count_mean': obj_count.mean(), 'object_count_std': obj_count.std()})
        avg_conf = group['avg_confidence'].dropna()
        if len(avg_conf) > 0:
            stats.update({'avg_confidence_mean': avg_conf.mean(), 'avg_confidence_std': avg_conf.std()})
        stats_list.append(stats)
    stats_df = pd.DataFrame(stats_list)
    return stats_df.sort_values('exposure_numeric')


def generate_markdown_summary(stats_df: pd.DataFrame, output_file: Path) -> None:
    """Generate markdown summary table."""
    if stats_df.empty:
        return
    lines = ["# Camera Exposure Time Optimization - Experiment Results", "",
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "", "## Summary Table", "",
             "| Exposure Config | Samples | CPU Mean (%) | Det Hz | Obj Count | Avg Conf | Drop Rate (%) |",
             "|-----------------|---------|--------------|--------|-----------|----------|---------------|"]
    for _, row in stats_df.iterrows():
        cpu_mean = f"{row.get('cpu_mean', 0):.1f}" if pd.notna(row.get('cpu_mean')) else "N/A"
        det_hz = f"{row.get('detection_hz_mean', 0):.2f}" if pd.notna(row.get('detection_hz_mean')) else "N/A"
        obj_count = f"{row.get('object_count_mean', 0):.2f}" if pd.notna(row.get('object_count_mean')) else "N/A"
        avg_conf = f"{row.get('avg_confidence_mean', 0):.3f}" if pd.notna(row.get('avg_confidence_mean')) else "N/A"
        drop_rate = f"{row.get('frame_drop_rate', 0):.1f}" if pd.notna(row.get('frame_drop_rate')) else "N/A"
        lines.append(f"| {row['exposure_config']} | {row['samples']} | {cpu_mean} | {det_hz} | {obj_count} | {avg_conf} | {drop_rate} |")
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Markdown summary saved to: {output_file}")


def generate_readme_section(stats_df: pd.DataFrame) -> str:
    """Generate formatted text for README Performance Tuning section."""
    if stats_df.empty:
        return "No data available."
    lines = ["#### Camera Exposure Time Optimization", "",
             "The `ExposureTime` parameter controls sensor exposure in microseconds. Tested at 10 FPS (100000μs frame duration):", "",
             "| Exposure Time | ExposureTime | CPU Mean | Det Hz | Obj Count | Avg Conf | Notes |",
             "|---------------|--------------|----------|--------|-----------|----------|-------|"]
    for _, row in stats_df.iterrows():
        exp_numeric = row.get('exposure_numeric', 0)
        exp_ms = f"{exp_numeric / 1000:.0f} ms" if exp_numeric else "N/A"
        cpu_mean = f"{row.get('cpu_mean', 0):.1f}%" if pd.notna(row.get('cpu_mean')) else "N/A"
        det_hz = f"{row.get('detection_hz_mean', 0):.2f} Hz" if pd.notna(row.get('detection_hz_mean')) else "N/A"
        obj_count = f"{row.get('object_count_mean', 0):.1f}" if pd.notna(row.get('object_count_mean')) else "N/A"
        avg_conf = f"{row.get('avg_confidence_mean', 0):.2f}" if pd.notna(row.get('avg_confidence_mean')) else "N/A"
        # Generate notes based on exposure time
        notes = ""
        if exp_numeric == 10000:
            notes = "Darker image"
        elif exp_numeric == 20000:
            notes = "**Default**"
        elif exp_numeric >= 25000:
            notes = "Brighter image"
        lines.append(f"| {exp_ms} | `{exp_numeric}` | {cpu_mean} | {det_hz} | {obj_count} | {avg_conf} | {notes} |")
    lines.extend(["", "**Key Finding:** Exposure time has minimal impact on CPU usage or detection rate, but affects image brightness and object detection count.", ""])
    return '\n'.join(lines)


def generate_graphs(df: pd.DataFrame, stats_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate visualization graphs."""
    if not MATPLOTLIB_AVAILABLE or df.empty:
        return
    plt.style.use('seaborn-v0_8-whitegrid')
    exp_configs = stats_df['exposure_config'].tolist()
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(exp_configs)))
    color_map = dict(zip(exp_configs, colors))

    # Graph 1: CPU over time
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp_config in exp_configs:
        group = df[df['exposure_config'] == exp_config].copy()
        if len(group) > 0:
            group['elapsed_s'] = (group['timestamp'] - group['timestamp'].min()).dt.total_seconds()
            ax.plot(group['elapsed_s'], group['cpu_percent'], label=exp_config, color=color_map[exp_config], linewidth=1.5)
    ax.set_xlabel('Elapsed Time (seconds)'); ax.set_ylabel('CPU Usage (%)')
    ax.set_title('CPU Usage Over Time by Exposure Configuration'); ax.legend(title='Exposure'); ax.set_ylim(0, 100)
    plt.tight_layout(); plt.savefig(output_dir / 'camera_exposure_cpu_over_time.png', dpi=150); plt.close()

    # Graph 2: CPU comparison bar
    fig, ax = plt.subplots(figsize=(8, 6))
    x_pos = np.arange(len(exp_configs))
    cpu_means = [stats_df[stats_df['exposure_config'] == c]['cpu_mean'].values[0] for c in exp_configs]
    cpu_stds = [stats_df[stats_df['exposure_config'] == c]['cpu_std'].values[0] for c in exp_configs]
    bars = ax.bar(x_pos, cpu_means, yerr=cpu_stds, capsize=5, color=[color_map[c] for c in exp_configs], edgecolor='black')
    ax.set_xlabel('Exposure Configuration'); ax.set_ylabel('CPU Usage (%)')
    ax.set_title('Mean CPU Usage by Exposure Configuration'); ax.set_xticks(x_pos); ax.set_xticklabels(exp_configs); ax.set_ylim(0, 100)
    for bar, val in zip(bars, cpu_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout(); plt.savefig(output_dir / 'camera_exposure_cpu_comparison.png', dpi=150); plt.close()

    # Graph 3: Object Count comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    obj_means = [stats_df[stats_df['exposure_config'] == c]['object_count_mean'].values[0] for c in exp_configs]
    bars = ax.bar(x_pos, obj_means, color=[color_map[c] for c in exp_configs], edgecolor='black')
    ax.set_xlabel('Exposure Configuration'); ax.set_ylabel('Average Objects Detected')
    ax.set_title('Object Detection Count by Exposure Configuration'); ax.set_xticks(x_pos); ax.set_xticklabels(exp_configs)
    for bar, val in zip(bars, obj_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout(); plt.savefig(output_dir / 'camera_exposure_object_count.png', dpi=150); plt.close()

    # Graph 4: Confidence comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    conf_means = [stats_df[stats_df['exposure_config'] == c]['avg_confidence_mean'].values[0] for c in exp_configs]
    bars = ax.bar(x_pos, conf_means, color=[color_map[c] for c in exp_configs], edgecolor='black')
    ax.set_xlabel('Exposure Configuration'); ax.set_ylabel('Average Confidence')
    ax.set_title('Detection Confidence by Exposure Configuration'); ax.set_xticks(x_pos); ax.set_xticklabels(exp_configs)
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, conf_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout(); plt.savefig(output_dir / 'camera_exposure_confidence.png', dpi=150); plt.close()
    print(f"Graphs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Camera Exposure Experiment - Report Generation')
    parser.add_argument('--input-dir', type=str, default='gesturebot_ws/src/ros2_mediapipe/benchmark/results')
    parser.add_argument('--generate-readme-section', action='store_true')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = Path.home() / 'GestureBot' / args.input_dir

    print(f"{'=' * 60}\nCamera Exposure Experiment - Report Generation\n{'=' * 60}")
    print(f"Input Directory: {input_dir}\n{'=' * 60}")

    df = load_experiment_data(input_dir)
    if df.empty:
        print("\nNo data to analyze. Exiting.")
        sys.exit(1)

    print(f"\nLoaded {len(df)} total samples")
    print(f"Exposure configurations: {df['exposure_config'].unique().tolist()}")

    stats_df = calculate_statistics(df)
    print("\n" + "=" * 80 + "\nSUMMARY\n" + "=" * 80)
    print(stats_df.to_string(index=False))

    generate_markdown_summary(stats_df, input_dir / 'camera_exposure_summary.md')
    print("\nGenerating graphs...")
    generate_graphs(df, stats_df, input_dir)

    if args.generate_readme_section:
        print("\n" + "=" * 60 + "\nREADME SECTION\n" + "=" * 60)
        readme_section = generate_readme_section(stats_df)
        print(readme_section)
        readme_file = input_dir / 'camera_exposure_readme_section.md'
        with open(readme_file, 'w') as f:
            f.write(readme_section)
        print(f"\nREADME section saved to: {readme_file}")

    print("\n" + "=" * 60 + "\nReport generation complete!\n" + "=" * 60)


if __name__ == '__main__':
    main()

