#!/usr/bin/env python3
"""
Confidence Threshold Experiment - Report Generation Script

Analyzes collected CSV data from confidence threshold experiments and generates
summary tables and visualization graphs.

Usage:
    python3 confidence_threshold_report.py
    python3 confidence_threshold_report.py --input-dir ./results --generate-readme-section
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
    """Load all confidence threshold experiment CSV files from directory."""
    csv_files = sorted(input_dir.glob('confidence_threshold_experiment_*.csv'))
    if not csv_files:
        print(f"No confidence_threshold_experiment_*.csv files found in {input_dir}")
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
    """Calculate summary statistics for each threshold configuration."""
    if df.empty:
        return pd.DataFrame()
    stats_list = []
    for threshold_config, group in df.groupby('threshold_config'):
        # Extract numeric threshold value
        threshold_match = re.search(r'([\d.]+)', str(threshold_config))
        threshold_numeric = float(threshold_match.group(1)) if threshold_match else 0.0
        stats = {'threshold_config': threshold_config, 'threshold_numeric': threshold_numeric,
                 'samples': len(group),
                 'duration_s': (group['timestamp'].max() - group['timestamp'].min()).total_seconds()}
        # CPU metrics
        cpu_data = group['cpu_percent'].dropna()
        if len(cpu_data) > 0:
            stats.update({'cpu_mean': cpu_data.mean(), 'cpu_std': cpu_data.std(),
                          'cpu_p95': np.percentile(cpu_data, 95), 'cpu_max': cpu_data.max()})
        # Temperature metrics
        temp_data = group['temperature_c'].dropna()
        if len(temp_data) > 0:
            stats.update({'temp_start': temp_data.iloc[0], 'temp_end': temp_data.iloc[-1],
                          'temp_max': temp_data.max(), 'temp_mean': temp_data.mean()})
        # Detection rate metrics
        det_hz = group['detection_hz'].dropna()
        if len(det_hz) > 0:
            stats.update({'detection_hz_mean': det_hz.mean(), 'detection_hz_std': det_hz.std()})
        # Object detection quality metrics
        obj_count = group['object_count'].dropna()
        if len(obj_count) > 0:
            stats.update({'object_count_mean': obj_count.mean(), 'object_count_std': obj_count.std()})
        avg_conf = group['avg_confidence'].dropna()
        if len(avg_conf) > 0:
            stats.update({'avg_confidence_mean': avg_conf.mean(), 'avg_confidence_std': avg_conf.std()})
        # Detection rate (% of frames with detections)
        det_rate = group['detection_rate'].dropna()
        if len(det_rate) > 0:
            stats.update({'detection_rate_mean': det_rate.mean(), 'detection_rate_std': det_rate.std()})
        # Min/max confidence stats
        min_conf = group['min_confidence'].dropna()
        if len(min_conf) > 0:
            stats.update({'min_confidence_mean': min_conf.mean()})
        max_conf = group['max_confidence'].dropna()
        if len(max_conf) > 0:
            stats.update({'max_confidence_mean': max_conf.mean()})
        stats_list.append(stats)
    stats_df = pd.DataFrame(stats_list)
    return stats_df.sort_values('threshold_numeric')


def generate_markdown_summary(stats_df: pd.DataFrame, output_file: Path) -> None:
    """Generate markdown summary table."""
    if stats_df.empty:
        return
    lines = ["# Confidence Threshold Optimization - Experiment Results", "",
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "", "## Summary Table", "",
             "| Threshold | Samples | CPU Mean (%) | Det Hz | Obj Count | Avg Conf | Det Rate (%) |",
             "|-----------|---------|--------------|--------|-----------|----------|--------------|"]
    for _, row in stats_df.iterrows():
        cpu_mean = f"{row.get('cpu_mean', 0):.1f}" if pd.notna(row.get('cpu_mean')) else "N/A"
        det_hz = f"{row.get('detection_hz_mean', 0):.2f}" if pd.notna(row.get('detection_hz_mean')) else "N/A"
        obj_count = f"{row.get('object_count_mean', 0):.2f}" if pd.notna(row.get('object_count_mean')) else "N/A"
        avg_conf = f"{row.get('avg_confidence_mean', 0):.3f}" if pd.notna(row.get('avg_confidence_mean')) else "N/A"
        det_rate = f"{row.get('detection_rate_mean', 0):.1f}" if pd.notna(row.get('detection_rate_mean')) else "N/A"
        lines.append(f"| {row['threshold_config']} | {row['samples']} | {cpu_mean} | {det_hz} | {obj_count} | {avg_conf} | {det_rate} |")
    lines.extend(["", "## Key Observations", "",
                  "- **Object Count**: Expected inverse relationship with threshold (lower threshold = more detections)",
                  "- **Average Confidence**: Expected direct relationship with threshold (higher threshold = higher avg confidence)",
                  "- **CPU Usage**: Expected minimal impact (threshold applied post-inference)",
                  "- **Detection Rate**: Percentage of frames containing at least one detection", ""])
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Markdown summary saved to: {output_file}")


def generate_readme_section(stats_df: pd.DataFrame) -> str:
    """Generate formatted text for README Performance Tuning section."""
    if stats_df.empty:
        return "No data available."
    lines = ["#### Confidence Threshold Optimization", "",
             "The `confidence_threshold` parameter filters detections post-inference. Tested at 10 FPS, 20ms exposure, frame_skip=0:", "",
             "| Threshold | CPU Mean | Det Hz | Obj Count | Avg Conf | Det Rate | Notes |",
             "|-----------|----------|--------|-----------|----------|----------|-------|"]
    for _, row in stats_df.iterrows():
        threshold = row.get('threshold_numeric', 0)
        cpu_mean = f"{row.get('cpu_mean', 0):.1f}%" if pd.notna(row.get('cpu_mean')) else "N/A"
        det_hz = f"{row.get('detection_hz_mean', 0):.2f} Hz" if pd.notna(row.get('detection_hz_mean')) else "N/A"
        obj_count = f"{row.get('object_count_mean', 0):.1f}" if pd.notna(row.get('object_count_mean')) else "N/A"
        avg_conf = f"{row.get('avg_confidence_mean', 0):.2f}" if pd.notna(row.get('avg_confidence_mean')) else "N/A"
        det_rate = f"{row.get('detection_rate_mean', 0):.0f}%" if pd.notna(row.get('detection_rate_mean')) else "N/A"
        notes = ""
        if threshold == 0.5:
            notes = "**Default**"
        elif threshold <= 0.3:
            notes = "More false positives"
        elif threshold >= 0.7:
            notes = "May miss valid detections"
        lines.append(f"| {threshold:.1f} | {cpu_mean} | {det_hz} | {obj_count} | {avg_conf} | {det_rate} | {notes} |")
    lines.extend(["", "**Key Finding:** Confidence threshold filters post-inference, so CPU impact is minimal. Trade-off is between detection sensitivity (lower threshold) and precision (higher threshold).", ""])
    return '\n'.join(lines)


def generate_graphs(df: pd.DataFrame, stats_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate visualization graphs."""
    if not MATPLOTLIB_AVAILABLE or df.empty:
        return
    plt.style.use('seaborn-v0_8-whitegrid')
    threshold_configs = stats_df['threshold_config'].tolist()
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(threshold_configs)))
    color_map = dict(zip(threshold_configs, colors))
    x_pos = np.arange(len(threshold_configs))

    # Graph 1: Object Count by Threshold
    fig, ax = plt.subplots(figsize=(8, 6))
    obj_means = [stats_df[stats_df['threshold_config'] == c]['object_count_mean'].values[0] for c in threshold_configs]
    obj_stds = [stats_df[stats_df['threshold_config'] == c].get('object_count_std', pd.Series([0])).values[0] for c in threshold_configs]
    bars = ax.bar(x_pos, obj_means, yerr=obj_stds, capsize=5, color=[color_map[c] for c in threshold_configs], edgecolor='black')
    ax.set_xlabel('Confidence Threshold'); ax.set_ylabel('Average Objects Detected')
    ax.set_title('Object Detection Count vs Confidence Threshold'); ax.set_xticks(x_pos); ax.set_xticklabels(threshold_configs)
    for bar, val in zip(bars, obj_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout(); plt.savefig(output_dir / 'confidence_threshold_object_count.png', dpi=150); plt.close()

    # Graph 2: Average Confidence by Threshold
    fig, ax = plt.subplots(figsize=(8, 6))
    conf_means = [stats_df[stats_df['threshold_config'] == c]['avg_confidence_mean'].values[0] for c in threshold_configs]
    bars = ax.bar(x_pos, conf_means, color=[color_map[c] for c in threshold_configs], edgecolor='black')
    ax.set_xlabel('Confidence Threshold'); ax.set_ylabel('Average Confidence')
    ax.set_title('Detection Confidence vs Threshold'); ax.set_xticks(x_pos); ax.set_xticklabels(threshold_configs)
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, conf_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout(); plt.savefig(output_dir / 'confidence_threshold_avg_confidence.png', dpi=150); plt.close()

    # Graph 3: Detection Hz by Threshold
    fig, ax = plt.subplots(figsize=(8, 6))
    det_hz_means = [stats_df[stats_df['threshold_config'] == c]['detection_hz_mean'].values[0] for c in threshold_configs]
    det_hz_stds = [stats_df[stats_df['threshold_config'] == c].get('detection_hz_std', pd.Series([0])).values[0] for c in threshold_configs]
    # Replace NaN with 0 for plotting
    det_hz_means = [0 if pd.isna(v) else v for v in det_hz_means]
    det_hz_stds = [0 if pd.isna(v) else v for v in det_hz_stds]
    bars = ax.bar(x_pos, det_hz_means, yerr=det_hz_stds, capsize=5, color=[color_map[c] for c in threshold_configs], edgecolor='black')
    ax.set_xlabel('Confidence Threshold'); ax.set_ylabel('Detection Hz')
    ax.set_title('Detection Frequency vs Confidence Threshold'); ax.set_xticks(x_pos); ax.set_xticklabels(threshold_configs)
    for bar, val in zip(bars, det_hz_means):
        label = f'{val:.2f}' if val > 0 else 'N/A'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, label, ha='center', va='bottom', fontsize=10)
    plt.tight_layout(); plt.savefig(output_dir / 'confidence_threshold_detection_hz.png', dpi=150); plt.close()

    # Graph 4: CPU comparison (to verify minimal impact)
    fig, ax = plt.subplots(figsize=(8, 6))
    cpu_means = [stats_df[stats_df['threshold_config'] == c]['cpu_mean'].values[0] for c in threshold_configs]
    cpu_stds = [stats_df[stats_df['threshold_config'] == c]['cpu_std'].values[0] for c in threshold_configs]
    bars = ax.bar(x_pos, cpu_means, yerr=cpu_stds, capsize=5, color=[color_map[c] for c in threshold_configs], edgecolor='black')
    ax.set_xlabel('Confidence Threshold'); ax.set_ylabel('CPU Usage (%)')
    ax.set_title('CPU Usage vs Confidence Threshold'); ax.set_xticks(x_pos); ax.set_xticklabels(threshold_configs)
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, cpu_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout(); plt.savefig(output_dir / 'confidence_threshold_cpu.png', dpi=150); plt.close()
    print(f"Graphs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Confidence Threshold Experiment - Report Generation')
    parser.add_argument('--input-dir', type=str, default='gesturebot_ws/src/ros2_mediapipe/benchmark/results')
    parser.add_argument('--generate-readme-section', action='store_true')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = Path.home() / 'GestureBot' / args.input_dir

    print(f"{'=' * 60}\nConfidence Threshold Experiment - Report Generation\n{'=' * 60}")
    print(f"Input Directory: {input_dir}\n{'=' * 60}")

    df = load_experiment_data(input_dir)
    if df.empty:
        print("\nNo data to analyze. Exiting.")
        sys.exit(1)

    print(f"\nLoaded {len(df)} total samples")
    print(f"Threshold configurations: {df['threshold_config'].unique().tolist()}")

    stats_df = calculate_statistics(df)
    print("\n" + "=" * 80 + "\nSUMMARY\n" + "=" * 80)
    print(stats_df.to_string(index=False))

    generate_markdown_summary(stats_df, input_dir / 'confidence_threshold_summary.md')
    print("\nGenerating graphs...")
    generate_graphs(df, stats_df, input_dir)

    if args.generate_readme_section:
        print("\n" + "=" * 60 + "\nREADME SECTION\n" + "=" * 60)
        readme_section = generate_readme_section(stats_df)
        print(readme_section)
        readme_file = input_dir / 'confidence_threshold_readme_section.md'
        with open(readme_file, 'w') as f:
            f.write(readme_section)
        print(f"\nREADME section saved to: {readme_file}")

    print("\n" + "=" * 60 + "\nReport generation complete!\n" + "=" * 60)


if __name__ == '__main__':
    main()

