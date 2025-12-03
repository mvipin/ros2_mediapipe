#!/usr/bin/env python3
"""
Camera Frame Skip Experiment - Report Generation Script

Analyzes collected CSV data from camera frame skip experiments and generates
summary tables and visualization graphs.

Usage:
    python3 camera_frame_skip_report.py
    python3 camera_frame_skip_report.py --input-dir ./results --generate-readme-section
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
    """Load all camera frame skip experiment CSV files from directory."""
    csv_files = sorted(input_dir.glob('camera_frame_skip_experiment_*.csv'))
    if not csv_files:
        print(f"No camera_frame_skip_experiment_*.csv files found in {input_dir}")
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
    """Calculate summary statistics for each frame skip configuration."""
    if df.empty:
        return pd.DataFrame()
    stats_list = []
    for fs_config, group in df.groupby('frame_skip_config'):
        fs_match = re.search(r'fs(\d+)', str(fs_config))
        fs_numeric = int(fs_match.group(1)) if fs_match else 0
        stats = {'frame_skip_config': fs_config, 'frame_skip_numeric': fs_numeric, 'samples': len(group),
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
        obj_count = group['object_count'].dropna()
        if len(obj_count) > 0:
            stats.update({'object_count_mean': obj_count.mean(), 'object_count_std': obj_count.std()})
        avg_conf = group['avg_confidence'].dropna()
        if len(avg_conf) > 0:
            stats.update({'avg_confidence_mean': avg_conf.mean(), 'avg_confidence_std': avg_conf.std()})
        stats_list.append(stats)
    stats_df = pd.DataFrame(stats_list)
    return stats_df.sort_values('frame_skip_numeric')


def generate_markdown_summary(stats_df: pd.DataFrame, output_file: Path) -> None:
    """Generate markdown summary table."""
    if stats_df.empty:
        return
    lines = ["# Camera Frame Skip Optimization - Experiment Results", "",
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "", "## Summary Table", "",
             "| Frame Skip | Samples | CPU Mean (%) | CPU P95 (%) | Det Hz | Obj Count | Avg Conf |",
             "|------------|---------|--------------|-------------|--------|-----------|----------|"]
    for _, row in stats_df.iterrows():
        cpu_mean = f"{row.get('cpu_mean', 0):.1f}" if pd.notna(row.get('cpu_mean')) else "N/A"
        cpu_p95 = f"{row.get('cpu_p95', 0):.1f}" if pd.notna(row.get('cpu_p95')) else "N/A"
        det_hz = f"{row.get('detection_hz_mean', 0):.2f}" if pd.notna(row.get('detection_hz_mean')) else "N/A"
        obj_count = f"{row.get('object_count_mean', 0):.2f}" if pd.notna(row.get('object_count_mean')) else "N/A"
        avg_conf = f"{row.get('avg_confidence_mean', 0):.3f}" if pd.notna(row.get('avg_confidence_mean')) else "N/A"
        lines.append(f"| {row['frame_skip_config']} | {row['samples']} | {cpu_mean} | {cpu_p95} | {det_hz} | {obj_count} | {avg_conf} |")
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Markdown summary saved to: {output_file}")


def generate_readme_section(stats_df: pd.DataFrame) -> str:
    """Generate formatted text for README Performance Tuning section."""
    if stats_df.empty:
        return "No data available."
    lines = ["#### Frame Skip Optimization", "",
             "The `frame_skip` parameter controls how many camera frames are skipped between processing cycles. "
             "Higher values reduce CPU load but decrease detection responsiveness.", "",
             "| frame_skip | Processing | CPU Mean | CPU P95 | Det Hz | Obj Count | Avg Conf | Notes |",
             "|------------|------------|----------|---------|--------|-----------|----------|-------|"]
    for _, row in stats_df.iterrows():
        fs_numeric = row.get('frame_skip_numeric', 0)
        processing = f"Every {fs_numeric + 1} frame" if fs_numeric > 0 else "All frames"
        cpu_mean = f"{row.get('cpu_mean', 0):.1f}%" if pd.notna(row.get('cpu_mean')) else "N/A"
        cpu_p95 = f"{row.get('cpu_p95', 0):.1f}%" if pd.notna(row.get('cpu_p95')) else "N/A"
        det_hz = f"{row.get('detection_hz_mean', 0):.2f} Hz" if pd.notna(row.get('detection_hz_mean')) else "N/A"
        obj_count = f"{row.get('object_count_mean', 0):.1f}" if pd.notna(row.get('object_count_mean')) else "N/A"
        avg_conf = f"{row.get('avg_confidence_mean', 0):.2f}" if pd.notna(row.get('avg_confidence_mean')) else "N/A"
        notes = ""
        if fs_numeric == 0:
            notes = "Max detection rate"
        elif fs_numeric == 1:
            notes = "**Default**"
        elif fs_numeric >= 3:
            notes = "Max CPU savings"
        lines.append(f"| {fs_numeric} | {processing} | {cpu_mean} | {cpu_p95} | {det_hz} | {obj_count} | {avg_conf} | {notes} |")
    lines.extend(["", "**Key Finding:** Frame skip trades detection rate for CPU savings while maintaining detection quality.", ""])
    return '\n'.join(lines)


def generate_graphs(df: pd.DataFrame, stats_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate visualization graphs."""
    if not MATPLOTLIB_AVAILABLE or df.empty:
        return
    plt.style.use('seaborn-v0_8-whitegrid')
    fs_configs = stats_df['frame_skip_config'].tolist()
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(fs_configs)))
    color_map = dict(zip(fs_configs, colors))

    # Graph 1: CPU over time
    fig, ax = plt.subplots(figsize=(10, 6))
    for fs_config in fs_configs:
        group = df[df['frame_skip_config'] == fs_config].copy()
        if len(group) > 0:
            group['elapsed_s'] = (group['timestamp'] - group['timestamp'].min()).dt.total_seconds()
            ax.plot(group['elapsed_s'], group['cpu_percent'], label=fs_config, color=color_map[fs_config], linewidth=1.5)
    ax.set_xlabel('Elapsed Time (seconds)'); ax.set_ylabel('CPU Usage (%)')
    ax.set_title('CPU Usage Over Time by Frame Skip Configuration'); ax.legend(title='Frame Skip'); ax.set_ylim(0, 100)
    plt.tight_layout(); plt.savefig(output_dir / 'camera_frame_skip_cpu_over_time.png', dpi=150); plt.close()

    # Graph 2: CPU comparison bar
    fig, ax = plt.subplots(figsize=(8, 6))
    x_pos = np.arange(len(fs_configs))
    cpu_means = [stats_df[stats_df['frame_skip_config'] == c]['cpu_mean'].values[0] for c in fs_configs]
    cpu_stds = [stats_df[stats_df['frame_skip_config'] == c]['cpu_std'].values[0] for c in fs_configs]
    bars = ax.bar(x_pos, cpu_means, yerr=cpu_stds, capsize=5, color=[color_map[c] for c in fs_configs], edgecolor='black')
    ax.set_xlabel('Frame Skip Configuration'); ax.set_ylabel('CPU Usage (%)')
    ax.set_title('Mean CPU Usage by Frame Skip Configuration'); ax.set_xticks(x_pos); ax.set_xticklabels(fs_configs); ax.set_ylim(0, 100)
    for bar, val in zip(bars, cpu_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout(); plt.savefig(output_dir / 'camera_frame_skip_cpu_comparison.png', dpi=150); plt.close()

    # Graph 3: Detection Hz comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    det_means = [stats_df[stats_df['frame_skip_config'] == c]['detection_hz_mean'].values[0] for c in fs_configs]
    bars = ax.bar(x_pos, det_means, color=[color_map[c] for c in fs_configs], edgecolor='black')
    ax.set_xlabel('Frame Skip Configuration'); ax.set_ylabel('Detection Rate (Hz)')
    ax.set_title('Detection Rate by Frame Skip Configuration'); ax.set_xticks(x_pos); ax.set_xticklabels(fs_configs)
    for bar, val in zip(bars, det_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout(); plt.savefig(output_dir / 'camera_frame_skip_detection_rate.png', dpi=150); plt.close()

    # Graph 4: Object Count comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    obj_means = [stats_df[stats_df['frame_skip_config'] == c]['object_count_mean'].values[0] for c in fs_configs]
    bars = ax.bar(x_pos, obj_means, color=[color_map[c] for c in fs_configs], edgecolor='black')
    ax.set_xlabel('Frame Skip Configuration'); ax.set_ylabel('Average Objects Detected')
    ax.set_title('Object Detection Count by Frame Skip Configuration'); ax.set_xticks(x_pos); ax.set_xticklabels(fs_configs)
    for bar, val in zip(bars, obj_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout(); plt.savefig(output_dir / 'camera_frame_skip_object_count.png', dpi=150); plt.close()
    print(f"Graphs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Camera Frame Skip Experiment - Report Generation')
    parser.add_argument('--input-dir', type=str, default='gesturebot_ws/src/ros2_mediapipe/benchmark/results')
    parser.add_argument('--generate-readme-section', action='store_true')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = Path.home() / 'GestureBot' / args.input_dir

    print(f"{'=' * 60}\nCamera Frame Skip Experiment - Report Generation\n{'=' * 60}")
    print(f"Input Directory: {input_dir}\n{'=' * 60}")

    df = load_experiment_data(input_dir)
    if df.empty:
        print("\nNo data to analyze. Exiting.")
        sys.exit(1)

    print(f"\nLoaded {len(df)} total samples")
    print(f"Frame skip configurations: {df['frame_skip_config'].unique().tolist()}")

    stats_df = calculate_statistics(df)
    print("\n" + "=" * 80 + "\nSUMMARY\n" + "=" * 80)
    print(stats_df.to_string(index=False))

    generate_markdown_summary(stats_df, input_dir / 'camera_frame_skip_summary.md')
    print("\nGenerating graphs...")
    generate_graphs(df, stats_df, input_dir)

    if args.generate_readme_section:
        print("\n" + "=" * 60 + "\nREADME SECTION\n" + "=" * 60)
        readme_section = generate_readme_section(stats_df)
        print(readme_section)
        readme_file = input_dir / 'camera_frame_skip_readme_section.md'
        with open(readme_file, 'w') as f:
            f.write(readme_section)
        print(f"\nREADME section saved to: {readme_file}")

    print("\n" + "=" * 60 + "\nReport generation complete!\n" + "=" * 60)


if __name__ == '__main__':
    main()

