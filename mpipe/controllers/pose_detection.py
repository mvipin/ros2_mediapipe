#!/usr/bin/env python3
"""
Pose Detection Controller for MediaPipe Integration

Provides MediaPipe PoseLandmarker integration using Tasks API.
"""

from typing import Callable
import mediapipe as mp
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision as mp_vis

from .base import MediaPipeController


class PoseDetectionController(MediaPipeController):
    """Controller for MediaPipe PoseLandmarker using Tasks API."""

    def __init__(
        self,
        model_path: str,
        num_poses: int,
        min_pose_detection_confidence: float,
        min_pose_presence_confidence: float,
        min_tracking_confidence: float,
        output_segmentation_masks: bool,
        result_callback: Callable,
    ) -> None:
        """
        Initialize pose detection controller.
        
        Args:
            model_path: Path to the pose detection model file
            num_poses: Maximum number of poses to detect
            min_pose_detection_confidence: Minimum confidence for pose detection
            min_pose_presence_confidence: Minimum confidence for pose presence
            min_tracking_confidence: Minimum confidence for pose tracking
            output_segmentation_masks: Whether to output segmentation masks
            result_callback: Callback function for processing results
        """
        base_options = mp_py.BaseOptions(model_asset_path=model_path)
        options = mp_vis.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vis.RunningMode.LIVE_STREAM,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=output_segmentation_masks,
            result_callback=result_callback,
        )
        # Initialize landmarker upon construction
        self._landmarker = mp_vis.PoseLandmarker.create_from_options(options)

    def is_ready(self) -> bool:
        """Check if the landmarker is ready for processing."""
        return self._landmarker is not None

    def detect_async(self, mp_image: mp.Image, timestamp_ms: int) -> None:
        """Submit an image for asynchronous pose detection."""
        if self._landmarker is None:
            return
        self._landmarker.detect_async(mp_image, timestamp_ms)

    def close(self) -> None:
        """Release landmarker resources."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            finally:
                self._landmarker = None
