#!/usr/bin/env python3
"""
MediaPipe Base Node for ROS 2 Integration

Provides stateless base functionality for all MediaPipe feature nodes.
Simplified version without state management components (PerformanceStats, PipelineTimer, BufferedLogger).
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from .processing_config import ProcessingConfig
from .controllers.base import MediaPipeController


class MediaPipeBaseNode(Node, ABC):
    """
    Abstract base class for all MediaPipe feature nodes.
    Provides common ROS infrastructure (QoS, pubs/subs, timers, logging).
    Stateless implementation focused on core MediaPipe processing.
    """

    def __init__(self, node_name: str, feature_name: str, config: ProcessingConfig,
                 controller: Optional[MediaPipeController] = None):
        super().__init__(node_name)

        self.feature_name = feature_name
        self.config = config

        # Optionally provided controller (composition). Nodes may set/replace later.
        self.controller: Optional[MediaPipeController] = controller

        # Initialize frame processing state
        self.frame_counter = 0
        self.latest_results = None

        # Initialize CV bridge
        self.bridge = CvBridge()

        # QoS profiles
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Image subscription
        self.image_subscription = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback,
            self.best_effort_qos
        )

        # Annotated image publisher (optional)
        self.annotated_image_publisher = self.create_publisher(
            Image,
            f'/vision/{feature_name}/annotated',
            self.best_effort_qos
        )

        self.get_logger().info(f'Initialized {feature_name} node (stateless)')

    @abstractmethod
    def process_frame(self, frame: np.ndarray, timestamp: float) -> Any:
        """Process a single frame. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def publish_results(self, results: Any, timestamp: float) -> None:
        """Publish processing results. Must be implemented by subclasses."""
        pass

    def image_callback(self, msg: Image) -> None:
        """Handle incoming camera frames."""
        if not self.config.enabled:
            return

        # Frame skipping for performance
        self.frame_counter += 1
        if self.frame_counter % (self.config.frame_skip + 1) != 0:
            return

        try:
            # Convert ROS image to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            timestamp = time.time()

            # Call subclass implementation to submit frame to MediaPipe
            # Results will be published via callback when ready
            results = self.process_frame(frame, timestamp)

            # Store results for potential access (should be None for callback nodes)
            if results is not None:
                self.latest_results = results

        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')

    def publish_annotated_image(self, annotated_frame: np.ndarray, timestamp: float) -> None:
        """Publish annotated image for visualization."""
        try:
            # Convert back to ROS image message
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            annotated_msg.header.stamp = self.get_clock().now().to_msg()
            annotated_msg.header.frame_id = 'camera_frame'
            
            self.annotated_image_publisher.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing annotated image: {e}')

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.controller:
            try:
                self.controller.close()
            except Exception as e:
                self.get_logger().error(f'Error closing controller: {e}')

        self.get_logger().info(f'Shutting down {self.feature_name} node')


class MediaPipeCallbackMixin:
    """
    Mixin class for MediaPipe nodes using callback-based processing.
    Provides callback-based processing with direct publishing from MediaPipe callback context.
    Stateless implementation without performance tracking.
    """

    def __init__(self):
        # Thread-safe callback state
        self.callback_lock = threading.Lock()
        self._callback_active = False

        # Store reference to the base node for direct publishing
        self._base_node_ref = None

    def _set_base_node_reference(self, base_node):
        """Set reference to base node for direct publishing from callback."""
        self._base_node_ref = base_node

    def create_callback(self, result_type: str) -> Callable:
        """
        Create a MediaPipe result callback for the specified result type.
        
        Args:
            result_type: Type of MediaPipe result ('object_detection', 'gesture_recognition', 'pose_detection')
            
        Returns:
            Callback function for MediaPipe processing
        """
        def callback(result, output_image, timestamp_ms: int):
            """MediaPipe result callback."""
            with self.callback_lock:
                if not self._callback_active:
                    return

                try:
                    # Convert timestamp
                    timestamp = timestamp_ms / 1000.0

                    # Call the appropriate result handler
                    if result_type == 'object_detection':
                        self.on_object_detection_result(result, output_image, timestamp_ms)
                    elif result_type == 'gesture_recognition':
                        self.on_gesture_result(result, output_image, timestamp_ms)
                    elif result_type == 'pose_detection':
                        self.on_pose_result(result, output_image, timestamp_ms)
                    else:
                        self.get_logger().error(f'Unknown result type: {result_type}')

                except Exception as e:
                    self.get_logger().error(f'Error in {result_type} callback: {e}')

        return callback

    def activate_callback(self):
        """Activate the callback processing."""
        with self.callback_lock:
            self._callback_active = True

    def deactivate_callback(self):
        """Deactivate the callback processing."""
        with self.callback_lock:
            self._callback_active = False

    # Abstract callback methods to be implemented by subclasses
    def on_object_detection_result(self, result, output_image, timestamp_ms: int):
        """Handle object detection results. Override in subclass."""
        pass

    def on_gesture_result(self, result, output_image, timestamp_ms: int):
        """Handle gesture recognition results. Override in subclass."""
        pass

    def on_pose_result(self, result, output_image, timestamp_ms: int):
        """Handle pose detection results. Override in subclass."""
        pass
