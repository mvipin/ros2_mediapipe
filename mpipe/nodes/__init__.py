"""
MediaPipe Detection Nodes for ROS 2

Provides ready-to-use detection nodes:
- ObjectDetectionNode: COCO object detection with EfficientDet
- GestureRecognitionNode: Hand gesture recognition with landmarks
- PoseDetectionNode: 33-point pose landmark detection with classification
"""

from .object_detection_node import ObjectDetectionNode
from .gesture_recognition_node import GestureRecognitionNode
from .pose_detection_node import PoseDetectionNode

__all__ = [
    'ObjectDetectionNode',
    'GestureRecognitionNode',
    'PoseDetectionNode'
]
