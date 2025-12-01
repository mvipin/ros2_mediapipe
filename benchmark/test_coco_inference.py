#!/usr/bin/env python3
"""
COCO Image Inference Test Script

This script verifies that the ros2_mediapipe framework can perform object detection
on static COCO images using MediaPipe's IMAGE running mode (synchronous).

This is a pre-benchmark verification to ensure the core inference pipeline works
independently of ROS and camera dependencies.

Usage:
    python3 test_coco_inference.py [--num-images N] [--model PATH]

Example:
    python3 test_coco_inference.py --num-images 100
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import urllib.request
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless systems
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision as mp_vis

from coco_evaluator import COCOEvaluator


def load_image(image_path: str) -> mp.Image:
    """
    Load an image from disk and convert to MediaPipe format.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        MediaPipe Image object in SRGB format
    """
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)


def create_detector(
    model_path: str,
    max_results: int = 5,
    score_threshold: float = 0.5
) -> mp_vis.ObjectDetector:
    """
    Create a MediaPipe ObjectDetector using IMAGE mode (synchronous).
    
    Args:
        model_path: Path to the EfficientDet model file
        max_results: Maximum number of detection results
        score_threshold: Minimum confidence threshold for detections
        
    Returns:
        MediaPipe ObjectDetector instance
    """
    base_options = mp_py.BaseOptions(model_asset_path=model_path)
    options = mp_vis.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=mp_vis.RunningMode.IMAGE,
        max_results=max_results,
        score_threshold=score_threshold,
    )
    return mp_vis.ObjectDetector.create_from_options(options)


def run_inference(
    image_path: str,
    model_path: str,
    max_results: int = 5,
    score_threshold: float = 0.5
) -> dict:
    """
    Run object detection inference on a single image.
    
    Args:
        image_path: Path to the input image
        model_path: Path to the EfficientDet model
        max_results: Maximum number of detections
        score_threshold: Minimum confidence threshold
        
    Returns:
        Dictionary containing inference results and timing
    """
    # Load image
    mp_image = load_image(image_path)
    
    # Create detector
    detector = create_detector(model_path, max_results, score_threshold)
    
    # Run inference with timing
    start_time = time.perf_counter()
    result = detector.detect(mp_image)
    inference_time_ms = (time.perf_counter() - start_time) * 1000
    
    # Extract detections
    detections = []
    for detection in result.detections:
        category = detection.categories[0]
        bbox = detection.bounding_box
        detections.append({
            'class_name': category.category_name,
            'score': category.score,
            'bbox': {
                'x': bbox.origin_x,
                'y': bbox.origin_y,
                'width': bbox.width,
                'height': bbox.height
            }
        })
    
    # Cleanup
    detector.close()
    
    return {
        'image': image_path,
        'inference_time_ms': inference_time_ms,
        'num_detections': len(detections),
        'detections': detections
    }


def download_coco_image(image_id: int, output_dir: Path) -> Path:
    """Download a COCO val2017 image by ID."""
    filename = f"{image_id:012d}.jpg"
    output_path = output_dir / filename

    if output_path.exists():
        return output_path

    url = f"http://images.cocodataset.org/val2017/{filename}"
    urllib.request.urlretrieve(url, output_path)
    return output_path


def run_evaluation(
    num_images: int,
    model_path: str,
    annotation_file: str,
    max_results: int = 10,
    score_threshold: float = 0.3,
    warmup: int = 5
) -> dict:
    """
    Run object detection on COCO images and evaluate mAP.

    Args:
        num_images: Number of images to evaluate
        model_path: Path to the EfficientDet model
        annotation_file: Path to COCO annotations JSON
        max_results: Maximum detections per image
        score_threshold: Minimum confidence threshold
        warmup: Number of warmup images (not included in timing)

    Returns:
        Dictionary with timing and accuracy metrics
    """
    from pycocotools.coco import COCO

    # Initialize COCO and evaluator
    print("Loading COCO annotations...")
    coco = COCO(annotation_file)
    evaluator = COCOEvaluator(annotation_file)

    # Get image IDs
    image_ids = coco.getImgIds()[:num_images + warmup]

    # Create detector
    print("Initializing detector...")
    detector = create_detector(model_path, max_results, score_threshold)

    # Create temp directory for images
    script_dir = Path(__file__).parent
    image_dir = script_dir / "data" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    inference_times = []
    total_detections = 0

    print(f"Running inference on {num_images} images (+ {warmup} warmup)...")
    print("-" * 60)

    for i, img_id in enumerate(image_ids):
        # Download image
        img_path = download_coco_image(img_id, image_dir)

        # Load image
        mp_image = load_image(str(img_path))

        # Run inference
        start_time = time.perf_counter()
        result = detector.detect(mp_image)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Skip warmup for timing
        if i >= warmup:
            inference_times.append(elapsed_ms)
            # Add results to evaluator
            num_det = evaluator.add_mediapipe_result(img_id, result)
            total_detections += num_det

            if (i - warmup + 1) % 10 == 0:
                print(f"  Processed {i - warmup + 1}/{num_images} images...")

    detector.close()

    # Calculate timing statistics
    inference_times = np.array(inference_times)
    timing_stats = {
        'inference_avg_ms': float(np.mean(inference_times)),
        'inference_std_ms': float(np.std(inference_times)),
        'inference_min_ms': float(np.min(inference_times)),
        'inference_max_ms': float(np.max(inference_times)),
        'inference_p95_ms': float(np.percentile(inference_times, 95)),
        'inference_p99_ms': float(np.percentile(inference_times, 99)),
        'theoretical_fps': float(1000 / np.mean(inference_times))
    }

    # Run COCO evaluation
    print("-" * 60)
    print("Running COCO evaluation...")
    accuracy_metrics = evaluator.evaluate()

    return {
        **timing_stats,
        **accuracy_metrics,
        'num_images': num_images,
        'total_detections': total_detections,
        'evaluator': evaluator  # Return evaluator for detailed metrics
    }


def plot_pr_curves(evaluator: COCOEvaluator, output_dir: Path) -> Path:
    """
    Generate and save combined precision-recall curves for the 'person' class.

    Args:
        evaluator: COCOEvaluator with completed evaluation
        output_dir: Directory to save plots

    Returns:
        Path to the combined plot
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pr_data = evaluator.get_pr_curve('person')
    recall = pr_data['recall']
    prec_iou50 = pr_data['precision_iou50']
    prec_iou75 = pr_data['precision_iou75']
    prec_avg = pr_data['precision_avg']

    # Combined P-R curve with all three IoU thresholds
    plt.figure(figsize=(8, 6))
    plt.plot(recall, prec_iou50, 'b-', linewidth=2, label='mAP@0.50')
    plt.plot(recall, prec_iou75, 'g-', linewidth=2, label='mAP@0.75')
    plt.plot(recall, prec_avg, 'r-', linewidth=2, label='mAP@0.50:0.95')
    plt.fill_between(recall, prec_iou50, alpha=0.15, color='blue')
    plt.fill_between(recall, prec_iou75, alpha=0.15, color='green')
    plt.fill_between(recall, prec_avg, alpha=0.15, color='red')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve: Person Class', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 0.4])
    plt.ylim([0, 1])
    plt.tight_layout()

    combined_path = output_dir / 'pr_curve_person_combined.png'
    plt.savefig(combined_path, dpi=150)
    plt.close()
    print(f"  Saved: {combined_path}")

    return combined_path


# Top 10 categories by ground truth instance count in the 50-image test set
# Based on coco.getImgIds()[:50] from instances_val2017.json
# Format: {category_name: instance_count}
TOP_10_CATEGORIES = {
    'person': 127,
    'car': 34,
    'book': 24,
    'boat': 15,
    'sheep': 13,
    'toilet': 13,
    'bird': 11,
    'elephant': 11,
    'motorcycle': 10,
    'orange': 10,
}


def generate_per_class_table(
    evaluator: COCOEvaluator,
    iou_type: str = 'iou50',
    filter_categories: dict = None
) -> str:
    """
    Generate a markdown table of per-class precision at recall thresholds.

    Args:
        evaluator: COCOEvaluator with completed evaluation
        iou_type: 'iou50' for mAP@0.50 or 'avg' for mAP@0.50:0.95
        filter_categories: Dict of {category_name: instance_count} to include

    Returns:
        Markdown table string
    """
    metrics = evaluator.get_precision_at_recall()
    categories = metrics['categories']
    recall_thresholds = metrics['recall_thresholds']

    if iou_type == 'iou50':
        precision_data = metrics['precision_iou50']
        ap_data = metrics['ap_iou50']
        title = "Per-Class Precision at Recall Thresholds (mAP@0.50)"
    else:
        precision_data = metrics['precision_avg']
        ap_data = metrics['ap_avg']
        title = "Per-Class Precision at Recall Thresholds (mAP@0.50:0.95)"

    # Build header
    header = "| Class |"
    for r in recall_thresholds:
        header += f" R={r:.1f} |"
    header += " AP |"

    separator = "|" + "------|" * (len(recall_thresholds) + 2)

    # Build rows sorted by instance count (descending)
    rows = []
    for cat in categories:
        # Filter to specified categories if provided
        if filter_categories is not None and cat not in filter_categories:
            continue

        prec_values = precision_data[cat]
        ap = ap_data[cat]

        # Get instance count for sorting
        instance_count = filter_categories.get(cat, 0) if filter_categories else 0

        # Add instance count if available
        if filter_categories and cat in filter_categories:
            cat_label = f"{cat} ({filter_categories[cat]})"
        else:
            cat_label = cat

        row = f"| {cat_label} |"
        for p in prec_values:
            row += f" {p:.3f} |" if p > 0 else " 0.000 |"
        row += f" {ap:.3f} |"
        rows.append((instance_count, row))

    rows.sort(key=lambda x: x[0], reverse=True)

    # Combine into table
    num_classes = len(rows)
    table = f"**{title}** (Top {num_classes} categories)\n\n{header}\n{separator}\n"
    for _, row in rows:
        table += row + "\n"

    return table


def generate_evaluation_outputs(
    evaluator: COCOEvaluator,
    output_dir: Path
) -> Dict:
    """
    Generate all evaluation outputs: tables and plots.

    Args:
        evaluator: COCOEvaluator with completed evaluation
        output_dir: Directory to save outputs

    Returns:
        Dict with paths to generated files and table content
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating evaluation outputs...")

    # Generate combined P-R curve
    print("  Generating P-R curves...")
    combined_plot = plot_pr_curves(evaluator, output_dir)

    # Generate per-class tables (top 10 categories by GT instance count)
    print("  Generating per-class tables...")
    table_iou50 = generate_per_class_table(evaluator, 'iou50', TOP_10_CATEGORIES)
    table_avg = generate_per_class_table(evaluator, 'avg', TOP_10_CATEGORIES)

    # Save tables to files
    table_iou50_path = output_dir / 'per_class_precision_iou50.md'
    table_avg_path = output_dir / 'per_class_precision_avg.md'

    with open(table_iou50_path, 'w') as f:
        f.write(table_iou50)
    print(f"  Saved: {table_iou50_path}")

    with open(table_avg_path, 'w') as f:
        f.write(table_avg)
    print(f"  Saved: {table_avg_path}")

    return {
        'combined_plot': combined_plot,
        'table_iou50': table_iou50,
        'table_avg': table_avg,
        'table_iou50_path': table_iou50_path,
        'table_avg_path': table_avg_path
    }


def update_readme(readme_path: Path, outputs: Dict, media_rel_path: str) -> bool:
    """
    Update README.md with evaluation outputs.

    Args:
        readme_path: Path to README.md
        outputs: Dict from generate_evaluation_outputs
        media_rel_path: Relative path from README to media folder

    Returns:
        True if update successful
    """
    if not readme_path.exists():
        print(f"  README not found: {readme_path}")
        return False

    with open(readme_path, 'r') as f:
        content = f.read()

    # Find the insertion point: after "Benchmark Results (EfficientDet Lite on Pi 5)" table
    marker = "| Test Images | 50 (COCO val2017 subset) |"

    if marker not in content:
        print(f"  Marker not found in README")
        return False

    # Check if already inserted
    if "Per-Class Precision at Recall Thresholds" in content:
        print("  Evaluation outputs already in README (skipping)")
        return True

    # Build insertion content
    insertion = f"""

**Precision-Recall Curves (Person Class):**

| mAP@0.50 | mAP@0.50:0.95 |
|:--------:|:-------------:|
| ![P-R Curve IoU=0.50]({media_rel_path}/pr_curve_person_iou50.png) | ![P-R Curve Averaged]({media_rel_path}/pr_curve_person_avg.png) |

<details>
<summary><strong>Per-Class Precision Tables (click to expand)</strong></summary>

{outputs['table_iou50']}

{outputs['table_avg']}

</details>
"""

    # Insert after the marker
    new_content = content.replace(marker, marker + insertion)

    with open(readme_path, 'w') as f:
        f.write(new_content)

    print(f"  Updated: {readme_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Test COCO image inference with mAP evaluation'
    )
    parser.add_argument(
        '--num-images', type=int, default=50,
        help='Number of images to evaluate (default: 50)'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Path to model file (default: uses efficientdet.tflite)'
    )
    parser.add_argument(
        '--annotations', type=str, default=None,
        help='Path to COCO annotations JSON'
    )
    parser.add_argument(
        '--max-results', type=int, default=10,
        help='Max detections per image (default: 10)'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.3,
        help='Confidence threshold (default: 0.3)'
    )
    parser.add_argument(
        '--warmup', type=int, default=5,
        help='Warmup images (default: 5)'
    )
    parser.add_argument(
        '--generate-outputs', action='store_true',
        help='Generate evaluation outputs (tables and plots)'
    )
    parser.add_argument(
        '--update-readme', action='store_true',
        help='Update README.md with evaluation outputs'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory for evaluation outputs (default: gesturebot/media/evaluation)'
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent

    model_path = args.model or str(script_dir.parent / 'models' / 'efficientdet.tflite')
    annotation_file = args.annotations or str(script_dir / 'data' / 'instances_val2017.json')

    # Validate paths
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    if not Path(annotation_file).exists():
        print(f"Error: Annotations not found at {annotation_file}")
        print("Run: wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
        sys.exit(1)

    print("=" * 60)
    print("COCO Inference Test with mAP Evaluation")
    print("=" * 60)
    print(f"Model: {Path(model_path).name}")
    print(f"Images: {args.num_images} (+ {args.warmup} warmup)")
    print(f"Config: max_results={args.max_results}, threshold={args.threshold}")
    print("=" * 60)

    # Run evaluation
    results = run_evaluation(
        num_images=args.num_images,
        model_path=model_path,
        annotation_file=annotation_file,
        max_results=args.max_results,
        score_threshold=args.threshold,
        warmup=args.warmup
    )

    # Print results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("\nTiming Metrics:")
    print(f"  Avg Inference:   {results['inference_avg_ms']:.2f} ms")
    print(f"  Std Dev:         {results['inference_std_ms']:.2f} ms")
    print(f"  Min/Max:         {results['inference_min_ms']:.2f} / {results['inference_max_ms']:.2f} ms")
    print(f"  P95/P99:         {results['inference_p95_ms']:.2f} / {results['inference_p99_ms']:.2f} ms")
    print(f"  Theoretical FPS: {results['theoretical_fps']:.1f}")

    print("\nAccuracy Metrics:")
    print(f"  mAP@0.50:        {results['mAP_50']:.4f}")
    print(f"  mAP@0.50:0.95:   {results['mAP_50_95']:.4f}")
    print(f"  mAP@0.75:        {results['mAP_75']:.4f}")
    print(f"  Recall:          {results['recall']:.4f}")

    print(f"\nTotal detections: {results['total_detections']} across {results['num_images']} images")
    print("=" * 60)
    print("COCO inference test: PASSED")

    # Generate evaluation outputs if requested
    if args.generate_outputs or args.update_readme:
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            # Default: gesturebot/media/evaluation
            output_dir = script_dir.parent.parent / 'gesturebot' / 'media' / 'evaluation'

        evaluator = results['evaluator']
        outputs = generate_evaluation_outputs(evaluator, output_dir)

        # Update README if requested
        if args.update_readme:
            readme_path = script_dir.parent.parent / 'gesturebot' / 'README.md'
            media_rel_path = 'media/evaluation'
            print("\nUpdating README...")
            update_readme(readme_path, outputs, media_rel_path)

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == '__main__':
    main()

