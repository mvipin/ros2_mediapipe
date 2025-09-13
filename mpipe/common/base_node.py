#!/usr/bin/env python3
"""
MediaPipe Base Node

Simplified base class for MediaPipe nodes without performance tracking or buffered logging.
"""

import os
import time
import threading
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from .config import ProcessingConfig, LogLevel
from ..controllers.base import MediaPipeController

# Set MediaPipe warning suppression environment variables early
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # Suppress TensorFlow INFO/WARNING
os.environ.setdefault('GLOG_minloglevel', '2')      # Suppress Google logging INFO/WARNING


class MediaPipeBaseNode(Node, ABC):
    """
    Base class for MediaPipe-based ROS 2 computer vision nodes.

    This abstract base class provides a robust foundation for building MediaPipe
    computer vision nodes in ROS 2. It handles the complexities of real-time
    image processing while maintaining system responsiveness through proper
    threading and resource management.

    Key Features:
    - Asynchronous frame processing: image_callback() → threading.Thread() → _process_frame_async()
    - Frame skipping logic to maintain real-time performance
    - Thread-safe processing with non-blocking locks
    - Standardized logging with configurable severity levels
    - Performance monitoring and frame drop detection
    - Clean controller composition pattern for MediaPipe pipelines
    - Automatic resource cleanup and lifecycle management

    Usage:
        Inherit from this class and implement the abstract process_frame method
        to create custom MediaPipe-based computer vision nodes.
    """

    def __init__(self, node_name: str, feature_name: str, config: ProcessingConfig,
                 controller: Optional[MediaPipeController] = None):
        super().__init__(node_name)

        self.feature_name = feature_name
        self.config = config

        # Controller composition (can be set later)
        self.controller: Optional[MediaPipeController] = controller

        # Essential threading and synchronization
        self.processing_lock = threading.Lock()
        self.latest_results = None

        # Simple frame counting for skip logic
        self.frame_count = 0

        # Set up callback mixin reference if available
        if hasattr(self, '_set_base_node_reference'):
            self._set_base_node_reference(self)
            self.enable_callback_processing()

        # ROS 2 components
        self.cv_bridge = CvBridge()

        # QoS profiles
        self.image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.result_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers - use configurable camera topic
        camera_topic = config.topics.camera_topic
        self.image_subscription = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            self.image_qos
        )

        # Configure ROS logger level based on config
        if config.log_level == LogLevel.DEBUG:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        elif config.log_level == LogLevel.INFO:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
        elif config.log_level == LogLevel.WARN:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.WARN)
        elif config.log_level == LogLevel.ERROR:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.ERROR)

        self.get_logger().info(f"[{feature_name}] {feature_name} node initialized")

    def image_callback(self, msg: Image) -> None:
        """
        Process incoming camera images using asynchronous threading architecture.

        This threading pattern prevents ROS callback blocking by processing frames
        in separate threads, ensuring real-time performance and system responsiveness.
        """
        if not self.config.enabled:
            return

        # Frame skipping logic
        self.frame_count += 1
        if self.frame_count % (self.config.frame_skip + 1) != 0:
            return

        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'rgb8')
            timestamp = time.time()

            # Process frame asynchronously to prevent ROS callback blocking
            threading.Thread(
                target=self._process_frame_async,
                args=(cv_image, timestamp),
                daemon=True
            ).start()

        except Exception as e:
            self.get_logger().error(f'[{self.feature_name}] Error in image callback: {e}')

    def _process_frame_async(self, frame: np.ndarray, timestamp: float) -> None:
        """
        Process frame in separate thread.
        
        Uses non-blocking lock to skip frames if still processing previous one.
        This prevents frame queue buildup and maintains real-time performance.
        """
        if not self.processing_lock.acquire(blocking=False):
            # Skip frame if still processing previous one
            return

        try:
            # Call subclass implementation
            results = self.process_frame(frame, timestamp)

            if results is not None:
                self.latest_results = results

        except Exception as e:
            self.get_logger().error(f'[{self.feature_name}] Error processing frame: {e}')
        finally:
            self.processing_lock.release()

    @abstractmethod
    def process_frame(self, frame: np.ndarray, timestamp: float):
        """
        Process a single frame. Must be implemented by subclass.
        
        Args:
            frame: OpenCV image in RGB format
            timestamp: Frame timestamp in seconds
            
        Returns:
            Processing results or None
        """
        pass

    @abstractmethod
    def publish_results(self, results, timestamp: float):
        """
        Publish processing results. Must be implemented by subclass.
        
        Args:
            results: Processed results from callback
            timestamp: Result timestamp in seconds
        """
        pass

    def cleanup(self):
        """Clean up resources."""
        if self.controller:
            self.controller.cleanup()
        
        self.get_logger().info(f'[{self.feature_name}] {self.feature_name} node cleanup complete')
