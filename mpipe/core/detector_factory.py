#!/usr/bin/env python3
"""
MediaPipe Detector Factory

Factory functions for creating MediaPipe detectors with LIVE_STREAM mode.
These are shared between ROS 2 nodes and standalone benchmark tools.

All detectors use async mode with result callbacks for non-blocking operation.
"""

from typing import Callable

import mediapipe as mp
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision as mp_vis


def create_object_detector(
    model_path: str,
    max_results: int,
    score_threshold: float,
    result_callback: Callable
) -> mp_vis.ObjectDetector:
    """
    Create MediaPipe ObjectDetector with LIVE_STREAM mode.

    Args:
        model_path: Path to EfficientDet TFLite model file
        max_results: Maximum number of detection results per frame
        score_threshold: Minimum confidence threshold for detections
        result_callback: Callback function invoked when detection completes.
                        Signature: callback(result, output_image, timestamp_ms)

    Returns:
        Configured ObjectDetector ready for async detection
    """
    base_options = mp_py.BaseOptions(model_asset_path=model_path)
    options = mp_vis.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=mp_vis.RunningMode.LIVE_STREAM,
        max_results=max_results,
        score_threshold=score_threshold,
        result_callback=result_callback,
    )
    return mp_vis.ObjectDetector.create_from_options(options)


def create_gesture_recognizer(
    model_path: str,
    num_hands: int,
    min_hand_detection_confidence: float,
    min_hand_presence_confidence: float,
    min_tracking_confidence: float,
    result_callback: Callable
) -> mp_vis.GestureRecognizer:
    """
    Create MediaPipe GestureRecognizer with LIVE_STREAM mode.

    Args:
        model_path: Path to gesture recognizer task file
        num_hands: Maximum number of hands to detect
        min_hand_detection_confidence: Minimum confidence for hand detection
        min_hand_presence_confidence: Minimum confidence for hand presence
        min_tracking_confidence: Minimum confidence for hand tracking
        result_callback: Callback function invoked when recognition completes.
                        Signature: callback(result, output_image, timestamp_ms)

    Returns:
        Configured GestureRecognizer ready for async recognition
    """
    base_options = mp_py.BaseOptions(model_asset_path=model_path)
    options = mp_vis.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=mp_vis.RunningMode.LIVE_STREAM,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        result_callback=result_callback,
    )
    return mp_vis.GestureRecognizer.create_from_options(options)


def create_pose_landmarker(
    model_path: str,
    num_poses: int,
    min_pose_detection_confidence: float,
    min_pose_presence_confidence: float,
    min_tracking_confidence: float,
    result_callback: Callable,
    output_segmentation_masks: bool = False
) -> mp_vis.PoseLandmarker:
    """
    Create MediaPipe PoseLandmarker with LIVE_STREAM mode.

    Args:
        model_path: Path to pose landmarker task file
        num_poses: Maximum number of poses to detect
        min_pose_detection_confidence: Minimum confidence for pose detection
        min_pose_presence_confidence: Minimum confidence for pose presence
        min_tracking_confidence: Minimum confidence for pose tracking
        result_callback: Callback function invoked when detection completes.
                        Signature: callback(result, output_image, timestamp_ms)
        output_segmentation_masks: Whether to output segmentation masks (default: False)

    Returns:
        Configured PoseLandmarker ready for async detection
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
    return mp_vis.PoseLandmarker.create_from_options(options)

