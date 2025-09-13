#!/usr/bin/env python3
"""
Object Detection Controller for MediaPipe Integration

Provides MediaPipe ObjectDetector (EfficientDet) integration using Tasks API.
"""

from typing import Callable
import mediapipe as mp
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision as mp_vis

from .base import MediaPipeController


class ObjectDetectionController(MediaPipeController):
    """Controller for MediaPipe ObjectDetector (EfficientDet) using Tasks API."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        max_results: int,
        result_callback: Callable,
    ) -> None:
        """
        Initialize object detection controller.
        
        Args:
            model_path: Path to the EfficientDet model file
            confidence_threshold: Minimum confidence threshold for detections
            max_results: Maximum number of detection results
            result_callback: Callback function for processing results
        """
        base_options = mp_py.BaseOptions(model_asset_path=model_path)
        options = mp_vis.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=mp_vis.RunningMode.LIVE_STREAM,
            max_results=max_results,
            score_threshold=confidence_threshold,
            result_callback=result_callback,
        )
        # Initialize detector upon construction
        self._detector = mp_vis.ObjectDetector.create_from_options(options)

    def is_ready(self) -> bool:
        """Check if the detector is ready for processing."""
        return self._detector is not None

    def detect_async(self, mp_image: mp.Image, timestamp_ms: int) -> None:
        """Submit an image for asynchronous object detection."""
        if self._detector is None:
            return
        self._detector.detect_async(mp_image, timestamp_ms)

    def close(self) -> None:
        """Release detector resources."""
        if self._detector is not None:
            try:
                self._detector.close()
            finally:
                self._detector = None
