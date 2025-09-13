#!/usr/bin/env python3
"""
Gesture Recognition Controller for MediaPipe Integration

Provides MediaPipe GestureRecognizer integration using Tasks API.
"""

from typing import Callable
import mediapipe as mp
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision as mp_vis

from .base import MediaPipeController


class GestureRecognitionController(MediaPipeController):
    """Controller for MediaPipe GestureRecognizer using Tasks API."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        max_hands: int,
        result_callback: Callable,
    ) -> None:
        """
        Initialize gesture recognition controller.
        
        Args:
            model_path: Path to the gesture recognition model file
            confidence_threshold: Minimum confidence threshold for hand detection
            max_hands: Maximum number of hands to detect
            result_callback: Callback function for processing results
        """
        base_options = mp_py.BaseOptions(model_asset_path=model_path)
        options = mp_vis.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=mp_vis.RunningMode.LIVE_STREAM,
            num_hands=max_hands,
            min_hand_detection_confidence=confidence_threshold,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=result_callback,
        )
        # Initialize recognizer upon construction
        self._recognizer = mp_vis.GestureRecognizer.create_from_options(options)

    def is_ready(self) -> bool:
        """Check if the recognizer is ready for processing."""
        return self._recognizer is not None

    def detect_async(self, mp_image: mp.Image, timestamp_ms: int) -> None:
        """Submit an image for asynchronous gesture recognition."""
        if self._recognizer is None:
            return
        self._recognizer.recognize_async(mp_image, timestamp_ms)

    def close(self) -> None:
        """Release recognizer resources."""
        if self._recognizer is not None:
            try:
                self._recognizer.close()
            finally:
                self._recognizer = None
