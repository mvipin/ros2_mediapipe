#!/usr/bin/env python3
"""
ROS 2 MediaPipe Integration Package

Provides stateless MediaPipe integration for ROS 2 with core vision processing components:
- MediaPipeBaseNode: Abstract base class for MediaPipe nodes
- ProcessingConfig: Configuration for MediaPipe processing
- MediaPipe Controllers: Object detection, gesture recognition, pose detection
- MessageConverter: ROS message conversion utilities
- Core: Non-ROS components shared with benchmark tools

This package focuses on core MediaPipe functionality without application-specific logic.
"""

from .common import (
    ProcessingConfig,
    MediaPipeBaseNode,
    MediaPipeCallbackMixin
)
from .utils import MessageConverter
from .controllers import (
    MediaPipeController,
    ObjectDetectionController,
    GestureRecognitionController,
    PoseDetectionController
)
from .core import (
    LockLifecycleManager,
    create_object_detector,
    create_gesture_recognizer,
    create_pose_landmarker
)

__version__ = "1.0.0"
__author__ = "Vipin Mehta"
__email__ = "91694248+vm-atmosic@users.noreply.github.com"

__all__ = [
    'ProcessingConfig',
    'MediaPipeBaseNode',
    'MediaPipeCallbackMixin',
    'MessageConverter',
    'MediaPipeController',
    'ObjectDetectionController',
    'GestureRecognitionController',
    'PoseDetectionController',
    'LockLifecycleManager',
    'create_object_detector',
    'create_gesture_recognizer',
    'create_pose_landmarker'
]
