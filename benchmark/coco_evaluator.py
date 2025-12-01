#!/usr/bin/env python3
"""
COCO Evaluation Utility for MediaPipe Object Detection

Converts MediaPipe detection results to COCO format and calculates
mAP, precision, and recall using pycocotools.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# MediaPipe EfficientDet uses COCO class names
# This maps class names to COCO category IDs
COCO_CATEGORIES = {
    'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5,
    'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10,
    'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14, 'bench': 15,
    'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21,
    'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27,
    'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34,
    'skis': 35, 'snowboard': 36, 'sports ball': 37, 'kite': 38,
    'baseball bat': 39, 'baseball glove': 40, 'skateboard': 41,
    'surfboard': 42, 'tennis racket': 43, 'bottle': 44, 'wine glass': 46,
    'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52,
    'apple': 53, 'sandwich': 54, 'orange': 55, 'broccoli': 56, 'carrot': 57,
    'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62,
    'couch': 63, 'potted plant': 64, 'bed': 65, 'dining table': 67,
    'toilet': 70, 'tv': 72, 'laptop': 73, 'mouse': 74, 'remote': 75,
    'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79,
    'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84, 'clock': 85,
    'vase': 86, 'scissors': 87, 'teddy bear': 88, 'hair drier': 89,
    'toothbrush': 90
}


class COCOEvaluator:
    """Evaluates MediaPipe detections against COCO ground truth."""

    def __init__(self, annotation_file: str):
        """
        Initialize with COCO annotation file.

        Args:
            annotation_file: Path to instances_val2017.json
        """
        self.coco_gt = COCO(annotation_file)
        self.results: List[Dict] = []
        self.image_ids: List[int] = []
        self.coco_eval: Optional[COCOeval] = None  # Store for later access
    
    def add_detection(
        self,
        image_id: int,
        class_name: str,
        score: float,
        bbox: Tuple[float, float, float, float]
    ) -> bool:
        """
        Add a detection result in COCO format.
        
        Args:
            image_id: COCO image ID
            class_name: Detected class name (e.g., 'person', 'car')
            score: Detection confidence score
            bbox: Bounding box as (x, y, width, height)
            
        Returns:
            True if detection was added, False if class not in COCO
        """
        category_id = COCO_CATEGORIES.get(class_name.lower())
        if category_id is None:
            return False
        
        self.results.append({
            'image_id': image_id,
            'category_id': category_id,
            'bbox': list(bbox),
            'score': score
        })
        
        if image_id not in self.image_ids:
            self.image_ids.append(image_id)
        
        return True
    
    def add_mediapipe_result(self, image_id: int, mp_result) -> int:
        """
        Add all detections from a MediaPipe result object.
        
        Args:
            image_id: COCO image ID
            mp_result: MediaPipe ObjectDetectorResult
            
        Returns:
            Number of detections added
        """
        count = 0
        for detection in mp_result.detections:
            category = detection.categories[0]
            bbox = detection.bounding_box
            if self.add_detection(
                image_id,
                category.category_name,
                category.score,
                (bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)
            ):
                count += 1
        return count
    
    def evaluate(self) -> Dict:
        """
        Run COCO evaluation and return metrics.

        Returns:
            Dictionary with mAP, precision, recall metrics
        """
        if not self.results:
            return {
                'mAP_50': 0.0,
                'mAP_50_95': 0.0,
                'mAP_75': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'num_detections': 0,
                'num_images': 0
            }

        # Load results into COCO format
        coco_dt = self.coco_gt.loadRes(self.results)

        # Run evaluation
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = self.image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Store for later access to precision/recall arrays
        self.coco_eval = coco_eval

        # Extract metrics from stats array
        # stats[0] = AP @ IoU=0.50:0.95
        # stats[1] = AP @ IoU=0.50
        # stats[2] = AP @ IoU=0.75
        # stats[8] = AR @ IoU=0.50:0.95 (max=100 detections)
        stats = coco_eval.stats

        return {
            'mAP_50_95': float(stats[0]),
            'mAP_50': float(stats[1]),
            'mAP_75': float(stats[2]),
            'recall': float(stats[8]),
            'precision': float(stats[0]),  # AP is precision-recall curve AUC
            'num_detections': len(self.results),
            'num_images': len(self.image_ids)
        }
    
    def reset(self):
        """Clear accumulated results."""
        self.results = []
        self.image_ids = []
        self.coco_eval = None

    def get_precision_at_recall(
        self,
        recall_thresholds: List[float] = None,
        iou_threshold: float = None
    ) -> Dict:
        """
        Extract interpolated precision at specified recall thresholds for each class.

        Args:
            recall_thresholds: List of recall values (default: [0.0, 0.1, ..., 1.0])
            iou_threshold: Specific IoU threshold (0.50-0.95) or None for average

        Returns:
            Dict with category names, precision arrays, and AP values
        """
        if self.coco_eval is None:
            raise ValueError("Must call evaluate() before get_precision_at_recall()")

        if recall_thresholds is None:
            recall_thresholds = [i / 10.0 for i in range(11)]  # [0.0, 0.1, ..., 1.0]

        # Precision array shape: (T, R, K, A, M)
        # T=10 IoU thresholds, R=101 recall thresholds, K=categories, A=4 areas, M=3 maxDets
        precision = self.coco_eval.eval['precision']

        # Map recall thresholds to indices (0.00 -> 0, 0.01 -> 1, ..., 1.00 -> 100)
        recall_indices = [int(r * 100) for r in recall_thresholds]

        # Get category IDs and names
        cat_ids = self.coco_eval.params.catIds

        # Build reverse mapping from category ID to name
        id_to_name = {v: k for k, v in COCO_CATEGORIES.items()}

        results = {
            'recall_thresholds': recall_thresholds,
            'categories': [],
            'precision_iou50': {},
            'precision_avg': {},
            'ap_iou50': {},
            'ap_avg': {}
        }

        for k_idx, cat_id in enumerate(cat_ids):
            cat_name = id_to_name.get(cat_id, f'class_{cat_id}')
            results['categories'].append(cat_name)

            # IoU=0.50 (index 0), area='all' (index 0), maxDets=100 (index 2)
            prec_iou50_full = precision[0, :, k_idx, 0, 2]

            # Average across all IoU thresholds
            prec_avg_full = precision[:, :, k_idx, 0, 2].mean(axis=0)

            # Sample at specified recall thresholds
            prec_iou50_sampled = [
                float(prec_iou50_full[idx]) if prec_iou50_full[idx] >= 0 else 0.0
                for idx in recall_indices
            ]
            prec_avg_sampled = [
                float(prec_avg_full[idx]) if prec_avg_full[idx] >= 0 else 0.0
                for idx in recall_indices
            ]

            results['precision_iou50'][cat_name] = prec_iou50_sampled
            results['precision_avg'][cat_name] = prec_avg_sampled

            # Compute AP (mean of valid precision values)
            valid_prec_iou50 = prec_iou50_full[prec_iou50_full >= 0]
            valid_prec_avg = prec_avg_full[prec_avg_full >= 0]

            results['ap_iou50'][cat_name] = float(valid_prec_iou50.mean()) if len(valid_prec_iou50) > 0 else 0.0
            results['ap_avg'][cat_name] = float(valid_prec_avg.mean()) if len(valid_prec_avg) > 0 else 0.0

        return results

    def get_pr_curve(self, category_name: str) -> Dict:
        """
        Get full precision-recall curve for a specific category.

        Args:
            category_name: COCO category name (e.g., 'person')

        Returns:
            Dict with recall values and precision arrays for IoU=0.50, IoU=0.75, and average
        """
        if self.coco_eval is None:
            raise ValueError("Must call evaluate() before get_pr_curve()")

        cat_id = COCO_CATEGORIES.get(category_name.lower())
        if cat_id is None:
            raise ValueError(f"Unknown category: {category_name}")

        cat_ids = self.coco_eval.params.catIds
        if cat_id not in cat_ids:
            raise ValueError(f"Category '{category_name}' not in evaluated categories")

        k_idx = list(cat_ids).index(cat_id)
        precision = self.coco_eval.eval['precision']

        # 101 recall thresholds from 0.00 to 1.00
        recall_values = [i / 100.0 for i in range(101)]

        # IoU=0.50 (index 0)
        prec_iou50 = precision[0, :, k_idx, 0, 2]

        # IoU=0.75 (index 5, since IoU thresholds are 0.50, 0.55, ..., 0.95)
        prec_iou75 = precision[5, :, k_idx, 0, 2]

        # Average across all IoU thresholds
        prec_avg = precision[:, :, k_idx, 0, 2].mean(axis=0)

        # Replace -1 (no data) with 0
        prec_iou50 = [float(p) if p >= 0 else 0.0 for p in prec_iou50]
        prec_iou75 = [float(p) if p >= 0 else 0.0 for p in prec_iou75]
        prec_avg = [float(p) if p >= 0 else 0.0 for p in prec_avg]

        return {
            'recall': recall_values,
            'precision_iou50': prec_iou50,
            'precision_iou75': prec_iou75,
            'precision_avg': prec_avg
        }

