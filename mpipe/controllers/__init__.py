#!/usr/bin/env python3
"""
MediaPipe Controllers for ROS 2 Integration

Provides composition-friendly MediaPipe controller abstractions:
- MediaPipeController: Abstract base controller interface
- ObjectDetectionController: EfficientDet object detection
- GestureRecognitionController: Hand gesture recognition
- PoseDetectionController: Pose landmark detection

Controllers handle MediaPipe-specific lifecycle, submission, and cleanup,
while nodes handle ROS 2 infrastructure.
"""

from .base import MediaPipeController
from .object_detection import ObjectDetectionController
from .gesture_recognition import GestureRecognitionController
from .pose_detection import PoseDetectionController

__all__ = [
    'MediaPipeController',
    'ObjectDetectionController',
    'GestureRecognitionController', 
    'PoseDetectionController'
]
