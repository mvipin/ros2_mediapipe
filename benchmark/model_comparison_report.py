#!/usr/bin/env python3
"""
Model Comparison Report Generator

Generates:
1. Summary comparison table (markdown)
2. 2×2 grid visualization (PNG)
3. Recommendation statement with trade-off analysis

Usage:
    python3 model_comparison_report.py --results results/model_comparison_YYYYMMDD.json
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Sort order for consistent display
QUANT_ORDER = {'int8': 0, 'float16': 1, 'float32': 2}


def load_results(json_path: Path) -> List[dict]:
    """Load benchmark results from JSON."""
    with open(json_path) as f:
        data = json.load(f)
    return data['results']


def sort_results(results: List[dict]) -> List[dict]:
    """Sort results by model family, then quantization."""
    def sort_key(r):
        name = r.get('model_name', '')
        if 'lite0' in name:
            family = 0
        elif 'lite2' in name:
            family = 1
        else:
            family = 2
        quant = QUANT_ORDER.get(r.get('quantization', ''), 3)
        return (family, quant)
    return sorted(results, key=sort_key)


def get_display_name(model_name: str) -> str:
    """Convert model_name to display name."""
    if 'efficientdet_lite0' in model_name:
        return 'EfficientDet-Lite0'
    elif 'efficientdet_lite2' in model_name:
        return 'EfficientDet-Lite2'
    elif 'ssd_mobilenetv2' in model_name:
        return 'SSD MobileNetV2'
    return model_name


def get_short_name(model_name: str, quantization: str) -> str:
    """Get short label for charts."""
    if 'efficientdet_lite0' in model_name:
        prefix = 'ED-L0'
    elif 'efficientdet_lite2' in model_name:
        prefix = 'ED-L2'
    elif 'ssd_mobilenetv2' in model_name:
        prefix = 'MNv2'
    else:
        prefix = model_name[:6]
    return f"{prefix}\n{quantization}"


def generate_comparison_table(results: List[dict]) -> str:
    """Generate markdown comparison table."""
    sorted_results = sort_results(results)
    baseline_name = 'efficientdet_lite0_float16'

    lines = [
        "### Model Comparison Summary",
        "",
        "| Model | Quant | Size (MB) | Input | mAP@0.50 | mAP@0.50:0.95 | Latency (ms) | Det Hz | CPU % | Memory (MB) |",
        "|-------|-------|-----------|-------|----------|---------------|--------------|--------|-------|-------------|",
    ]

    for r in sorted_results:
        is_baseline = r.get('model_name') == baseline_name
        display_name = get_display_name(r.get('model_name', ''))
        input_size = r.get('input_size', [0, 0])
        input_str = f"{input_size[0]}×{input_size[1]}"

        lat_mean = r.get('inference_avg_ms', 0)
        lat_p95 = r.get('inference_p95_ms', 0)
        latency_str = f"{lat_mean:.1f} (p95: {lat_p95:.1f})"

        row_data = [
            display_name,
            r.get('quantization', ''),
            f"{r.get('model_size_mb', 0):.1f}",
            input_str,
            f"{r.get('mAP_50', 0):.3f}",
            f"{r.get('mAP_50_95', 0):.3f}",
            latency_str,
            f"{r.get('effective_fps', 0):.1f}",
            f"{r.get('cpu_mean', 0):.1f}",
            f"{r.get('memory_mb_mean', 0):.0f}",
        ]

        if is_baseline:
            row = "| **" + "** | **".join(row_data) + "** |"
        else:
            row = "| " + " | ".join(row_data) + " |"
        lines.append(row)

    lines.extend([
        "",
        "*Baseline model (EfficientDet-Lite0 float16) in bold. Latency shows mean with p95 in parentheses.*",
    ])

    return '\n'.join(lines)


def generate_grid_chart(results: List[dict], output_path: Path) -> None:
    """Generate 3×2 grid visualization with 6 metrics."""
    sorted_results = sort_results(results)
    baseline_name = 'efficientdet_lite0_float16'

    # Prepare data
    model_labels = [
        get_short_name(r.get('model_name', ''), r.get('quantization', ''))
        for r in sorted_results
    ]
    map_50 = [r.get('mAP_50', 0) for r in sorted_results]
    det_hz = [r.get('effective_fps', 0) for r in sorted_results]
    latency_ms = [r.get('inference_avg_ms', 0) for r in sorted_results]
    recall = [r.get('recall', 0) for r in sorted_results]
    cpu_mean = [r.get('cpu_mean', 0) for r in sorted_results]
    memory_mb = [r.get('memory_mb_mean', 0) for r in sorted_results]

    # Colors: highlight baseline
    colors = ['#4ECDC4'] * len(sorted_results)
    baseline_idx = next(
        (i for i, r in enumerate(sorted_results)
         if r.get('model_name') == baseline_name),
        None
    )
    if baseline_idx is not None:
        colors[baseline_idx] = '#FF6B6B'

    y_pos = np.arange(len(model_labels))

    # Create 3×2 grid
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        'Model Comparison: Accuracy vs Performance Trade-offs',
        fontsize=14, fontweight='bold'
    )

    # Helper function for bar chart
    def plot_metric(ax, values, xlabel, title, fmt, offset_factor=0.02):
        bars = ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_labels)
        max_val = max(values) if values and max(values) > 0 else 1
        ax.set_xlim(0, max_val * 1.25)
        for bar, val in zip(bars, values):
            ax.text(val + max_val * offset_factor, bar.get_y() + bar.get_height()/2,
                    fmt.format(val), va='center', fontsize=9)

    # Row 1: Accuracy and Speed
    plot_metric(axes[0, 0], map_50, 'mAP@0.50', 'Accuracy (Higher is Better)', '{:.3f}')
    plot_metric(axes[0, 1], det_hz, 'FPS', 'Speed (Higher is Better)', '{:.1f}')

    # Row 2: Latency and Recall
    plot_metric(axes[1, 0], latency_ms, 'Latency (ms)', 'Inference Latency (Lower is Better)', '{:.0f}')
    plot_metric(axes[1, 1], recall, 'Recall', 'Recall (Higher is Better)', '{:.3f}')

    # Row 3: CPU and Memory
    plot_metric(axes[2, 0], cpu_mean, 'CPU Usage (%)', 'CPU Load (Lower is Better)', '{:.0f}%')
    plot_metric(axes[2, 1], memory_mb, 'Memory RSS (MB)', 'Memory Usage (Lower is Better)', '{:.0f}')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', edgecolor='black',
              label='Baseline (EfficientDet-Lite0 float16)'),
        Patch(facecolor='#4ECDC4', edgecolor='black', label='Other models'),
    ]
    fig.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: {output_path}")


def find_model(results: List[dict], name: str) -> Optional[dict]:
    """Find model by name substring."""
    return next((r for r in results if name in r.get('model_name', '')), None)


def generate_recommendation(results: List[dict]) -> str:
    """Generate data-driven recommendation statement."""
    baseline = find_model(results, 'efficientdet_lite0_float16')
    lite0_int8 = find_model(results, 'efficientdet_lite0_int8')
    lite2_int8 = find_model(results, 'efficientdet_lite2_int8')
    ssd_int8 = find_model(results, 'ssd_mobilenetv2_int8')

    lines = [
        "### Raspberry Pi 5 Recommendation",
        "",
        "**Recommended: EfficientDet-Lite0 float16** for GestureBot deployment.",
        "",
        "| Model | Accuracy | Speed | CPU | Why/Why Not |",
        "|-------|----------|-------|-----|-------------|",
    ]

    if baseline:
        lines.append(
            f"| **EfficientDet-Lite0 float16** | mAP={baseline.get('mAP_50', 0):.3f} | "
            f"{baseline.get('effective_fps', 0):.1f} Hz | {baseline.get('cpu_mean', 0):.0f}% | "
            f"✅ Best balance for person following |"
        )

    if lite0_int8:
        lines.append(
            f"| EfficientDet-Lite0 int8 | mAP={lite0_int8.get('mAP_50', 0):.3f} | "
            f"{lite0_int8.get('effective_fps', 0):.1f} Hz | {lite0_int8.get('cpu_mean', 0):.0f}% | "
            f"Alternative if CPU headroom needed |"
        )

    if lite2_int8:
        lines.append(
            f"| EfficientDet-Lite2 int8 | mAP={lite2_int8.get('mAP_50', 0):.3f} | "
            f"{lite2_int8.get('effective_fps', 0):.1f} Hz | {lite2_int8.get('cpu_mean', 0):.0f}% | "
            f"Consider if accuracy is critical |"
        )

    if ssd_int8:
        lines.append(
            f"| SSD MobileNetV2 int8 | mAP={ssd_int8.get('mAP_50', 0):.3f} | "
            f"{ssd_int8.get('effective_fps', 0):.1f} Hz | {ssd_int8.get('cpu_mean', 0):.0f}% | "
            f"Fastest, but lower accuracy |"
        )

    lines.extend(["", "**Trade-off Analysis:**", ""])

    # Generate trade-off comparisons
    if baseline and lite0_int8:
        base_fps = baseline.get('effective_fps', 1)
        int8_fps = lite0_int8.get('effective_fps', 0)
        speed_diff = ((int8_fps / base_fps) - 1) * 100 if base_fps > 0 else 0

        base_map = baseline.get('mAP_50', 1)
        int8_map = lite0_int8.get('mAP_50', 0)
        acc_diff = ((int8_map / base_map) - 1) * 100 if base_map > 0 else 0

        lines.append(
            f"- **EfficientDet-Lite0 int8 vs float16**: int8 provides {speed_diff:+.0f}% "
            f"detection rate with {acc_diff:+.1f}% accuracy change. Choose int8 if running "
            f"additional ROS nodes that need CPU headroom."
        )
        lines.append("")

    if baseline and lite2_int8:
        base_fps = baseline.get('effective_fps', 1)
        lite2_fps = lite2_int8.get('effective_fps', 0)
        speed_diff = ((lite2_fps / base_fps) - 1) * 100 if base_fps > 0 else 0

        base_map = baseline.get('mAP_50', 1)
        lite2_map = lite2_int8.get('mAP_50', 0)
        acc_diff = ((lite2_map / base_map) - 1) * 100 if base_map > 0 else 0

        cpu_diff = lite2_int8.get('cpu_mean', 0) - baseline.get('cpu_mean', 0)

        lines.append(
            f"- **EfficientDet-Lite2 vs Lite0**: Lite2 int8 offers {acc_diff:+.1f}% mAP change "
            f"at cost of {speed_diff:+.0f}% detection rate and {cpu_diff:+.0f}% CPU increase. "
            f"Consider if detection accuracy is more important than frame rate."
        )
        lines.append("")

    if baseline and ssd_int8:
        base_fps = baseline.get('effective_fps', 1)
        ssd_fps = ssd_int8.get('effective_fps', 0)
        speed_diff = ((ssd_fps / base_fps) - 1) * 100 if base_fps > 0 else 0

        lines.append(
            f"- **SSD MobileNetV2**: Fastest option ({ssd_fps:.1f} Hz, {speed_diff:+.0f}% faster) "
            f"but significantly lower accuracy (mAP={ssd_int8.get('mAP_50', 0):.3f}). Only suitable "
            f"if speed is critical and detection quality can be compromised."
        )

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Model Comparison Report Generator'
    )
    parser.add_argument(
        '--results', type=Path, required=True,
        help='Path to model_comparison_YYYYMMDD.json'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=None,
        help='Output directory (default: same as results file)'
    )
    args = parser.parse_args()

    if not args.results.exists():
        print(f"ERROR: Results file not found: {args.results}")
        return 1

    output_dir = args.output_dir or args.results.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MODEL COMPARISON REPORT GENERATOR")
    print("=" * 60)
    print(f"Results: {args.results}")
    print(f"Output: {output_dir}")

    results = load_results(args.results)
    print(f"Loaded {len(results)} model results")

    # Generate table
    table_md = generate_comparison_table(results)

    # Determine output filename prefix based on input filename
    # e.g., model_comparison_image_mode.json -> model_comparison_image_mode_grid.png
    results_stem = args.results.stem  # e.g., "model_comparison_image_mode"
    if results_stem.startswith('model_comparison_'):
        prefix = results_stem
    else:
        prefix = 'model_comparison'

    # Generate 3×2 grid chart (accuracy, speed, latency, recall, CPU, memory)
    grid_path = output_dir / f'{prefix}_grid.png'
    generate_grid_chart(results, grid_path)

    # Generate recommendation
    recommendation_md = generate_recommendation(results)

    # Combine into full report
    grid_filename = grid_path.name
    report_lines = [
        "## Object Detection Model Comparison",
        "",
        f"*Benchmarked on Raspberry Pi 5, {datetime.now().strftime('%Y-%m-%d')}*",
        "",
        table_md,
        "",
        f"![Model Comparison]({grid_filename})",
        "",
        recommendation_md,
    ]

    report_path = output_dir / f'{prefix}_readme_section.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\nReport generated: {report_path}")
    print(f"Chart generated: {grid_path}")
    print("\nTo use in README, copy content from the readme_section.md file")


if __name__ == '__main__':
    main()

