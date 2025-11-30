#!/usr/bin/env python3
"""
Object Detection Node for ros2_mediapipe Package

Real-time object detection using MediaPipe Object Detector with support for
common objects from the COCO dataset. Provides bounding boxes, confidence
scores, and class labels for detected objects in camera images.

Features:
- Real-time object detection from COCO dataset (80+ classes)
- Configurable confidence thresholds and visualization parameters
- Bounding box detection with confidence scores
- Multi-object detection support
- Annotated image output with configurable styling
- Thread-safe asynchronous processing
- Comprehensive parameter system for runtime configuration
"""

import time
from typing import Optional, Dict

import cv2
import numpy as np
import mediapipe as mp
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ros2_mediapipe.msg import DetectedObjects

from mpipe.common import ProcessingConfig, MediaPipeBaseNode, MediaPipeCallbackMixin
from mpipe.common.config import LogLevel
from mpipe.controllers import ObjectDetectionController
from mpipe.utils import MessageConverter


class ObjectDetectionNode(MediaPipeBaseNode, MediaPipeCallbackMixin):
    """
    ROS 2 node for real-time object detection using MediaPipe.

    This node processes camera images to detect and classify objects from the
    COCO dataset. Provides bounding boxes, confidence scores, and class labels
    for detected objects in real-time applications.

    Publishers:
        /vision/objects (DetectedObjects): Detected objects with bounding boxes
        /vision/objects/annotated (Image): Annotated image with bounding boxes

    Subscribers:
        /camera/image_raw (Image): Input camera images

    Parameters:
        model_path (str): Path to MediaPipe object detection model file
        log_level (str): Logging level (DEBUG, INFO, WARN, ERROR)
        enabled (bool): Enable/disable object detection processing
        frame_skip (int): Number of frames to skip between processing
        confidence_threshold (float): Minimum confidence for object detection
        max_results (int): Maximum number of objects to detect
    """

    def __init__(self):
        """Initialize object detection node."""
        # Create temporary node to read parameters
        temp_node = Node('temp_object_detection_node')

        # Declare and read parameters
        temp_node.declare_parameter('frame_skip', 1)
        temp_node.declare_parameter('confidence_threshold', 0.5)
        temp_node.declare_parameter('max_results', 5)
        temp_node.declare_parameter('log_level', 'INFO')
        temp_node.declare_parameter('debug_mode', False)

        frame_skip = temp_node.get_parameter('frame_skip').value
        confidence_threshold = temp_node.get_parameter('confidence_threshold').value
        max_results = temp_node.get_parameter('max_results').value
        log_level_str = temp_node.get_parameter('log_level').value
        debug_mode = temp_node.get_parameter('debug_mode').value

        # Convert log level string to enum
        try:
            log_level = LogLevel(log_level_str.upper())
        except ValueError:
            log_level = LogLevel.INFO

        temp_node.destroy_node()

        # Create configuration
        config = ProcessingConfig(
            enabled=True,
            frame_skip=frame_skip,
            confidence_threshold=confidence_threshold,
            max_results=max_results,
            log_level=log_level,
            debug_mode=debug_mode
        )

        # Initialize callback mixin first
        MediaPipeCallbackMixin.__init__(self)

        # Initialize base node
        super().__init__(
            'object_detection_node',
            'object_detection',
            config,
            controller=None
        )

        # Essential components only
        self.controller = None
        self.message_converter = MessageConverter()

        # Frame storage for callback access
        self._current_rgb_frame = None
        self._current_frame_timestamp = None

        # Declare parameters
        self.declare_parameter('model_path', 'models/efficientdet.tflite')
        self.declare_parameter('confidence_threshold', config.confidence_threshold)
        self.declare_parameter('max_results', config.max_results)
        self.declare_parameter('frame_skip', config.frame_skip)
        self.declare_parameter('log_level', config.log_level.value)
        self.declare_parameter('debug_mode', config.debug_mode)  # Deprecated, use log_level

        # Topic configuration parameters
        self.declare_parameter('camera_topic', config.topics.camera_topic)
        self.declare_parameter('objects_topic', config.topics.objects_topic)
        self.declare_parameter('objects_annotated_topic', config.topics.objects_annotated_topic)

        # Visualization parameters
        self.declare_parameter('bbox_color_r', config.visualization.landmark_color[2])  # BGR to RGB
        self.declare_parameter('bbox_color_g', config.visualization.landmark_color[1])
        self.declare_parameter('bbox_color_b', config.visualization.landmark_color[0])
        self.declare_parameter('text_color_r', config.visualization.text_color[2])
        self.declare_parameter('text_color_g', config.visualization.text_color[1])
        self.declare_parameter('text_color_b', config.visualization.text_color[0])
        self.declare_parameter('font_scale', config.visualization.font_scale)
        self.declare_parameter('bbox_thickness', config.visualization.skeleton_thickness)  # Reuse skeleton thickness for bbox

        # Publishers - use configurable topic names
        objects_topic = self.get_parameter('objects_topic').get_parameter_value().string_value
        objects_annotated_topic = self.get_parameter('objects_annotated_topic').get_parameter_value().string_value

        self.detections_pub = self.create_publisher(
            DetectedObjects,
            objects_topic,
            self.result_qos
        )
        self.annotated_image_pub = self.create_publisher(
            Image,
            objects_annotated_topic,
            self.result_qos
        )

        # Initialize object detection
        self._initialize_object_detection()

        self.get_logger().info("Object detection node initialized with comprehensive parameter system")

    def _initialize_object_detection(self):
        """Initialize MediaPipe object detection controller."""
        try:
            # Get parameters
            confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
            max_results = self.get_parameter('max_results').get_parameter_value().integer_value
            model_path = self.get_parameter('model_path').get_parameter_value().string_value

            # Resolve model path (relative to ros2_mediapipe package install directory)
            if not model_path.startswith('/'):
                import os
                from ament_index_python.packages import get_package_prefix
                package_prefix = get_package_prefix('ros2_mediapipe')
                model_path = os.path.join(package_prefix, model_path)

            # Create callback
            callback = self.create_callback('object_detection')

            # Initialize controller
            self.controller = ObjectDetectionController(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                max_results=max_results,
                result_callback=callback
            )

            if self.config.debug_mode:
                self.get_logger().info(f"Object detection initialized with model: {model_path}")

        except Exception as e:
            self.get_logger().error(f"Failed to initialize object detection: {e}")
            raise

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Optional[Dict]:
        """
        Process frame for object detection (called from threaded context).
        """
        try:
            # Store frame for callback access
            self._current_rgb_frame = frame
            self._current_frame_timestamp = timestamp

            # Convert to MediaPipe format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # Submit for async processing
            timestamp_ms = int(timestamp * 1000)

            if self.controller and self.controller.is_ready():
                self.controller.detect_async(mp_image, timestamp_ms)
            else:
                if self.config.debug_mode:
                    self.get_logger().warn("Controller not ready for processing")

            # Return None for callback-based processing
            return None

        except Exception as e:
            self.get_logger().error(f"Error in process_frame: {e}")
            return None

    def _process_callback_results(self, result, output_image, timestamp_ms, result_type):
        """
        Process MediaPipe callback results.
        """
        try:
            if result is None:
                return None

            detections = result.detections if result.detections else []

            if self.config.debug_mode and detections:
                self.get_logger().info(f"Detected {len(detections)} objects")

            # Create result dictionary
            result_data = {
                'detections': detections,
                'frame': self._current_rgb_frame,
                'timestamp': timestamp_ms / 1000.0
            }

            return result_data

        except Exception as e:
            self.get_logger().error(f"Error processing callback results: {e}")
            return None

    def publish_results(self, results: Dict, timestamp: float) -> None:
        """
        Publish detection results and annotated images.
        """
        try:
            detections = results.get('detections', [])
            frame = results.get('frame')

            # Publish detection messages
            if detections:
                ros_msg = self.message_converter.mediapipe_detections_to_ros(
                    detections,
                    "ros2_mediapipe_detector",
                    0.1  # processing_time placeholder
                )
                ros_msg.header.stamp = self.get_clock().now().to_msg()
                ros_msg.header.frame_id = "camera_frame"
                self.detections_pub.publish(ros_msg)

            # Publish annotated images
            if frame is not None:
                annotated_frame = self._annotate_image(frame.copy(), detections)
                self._publish_annotated_image(annotated_frame)

        except Exception as e:
            self.get_logger().error(f"Error publishing results: {e}")

    def _annotate_image(self, image, detections):
        """Annotate image with detection results using configurable styling."""
        height, width = image.shape[:2]

        # Get configurable visualization parameters
        bbox_color_r = self.get_parameter('bbox_color_r').get_parameter_value().integer_value
        bbox_color_g = self.get_parameter('bbox_color_g').get_parameter_value().integer_value
        bbox_color_b = self.get_parameter('bbox_color_b').get_parameter_value().integer_value
        bbox_color = (bbox_color_b, bbox_color_g, bbox_color_r)  # BGR format

        text_color_r = self.get_parameter('text_color_r').get_parameter_value().integer_value
        text_color_g = self.get_parameter('text_color_g').get_parameter_value().integer_value
        text_color_b = self.get_parameter('text_color_b').get_parameter_value().integer_value
        text_color = (text_color_b, text_color_g, text_color_r)  # BGR format

        font_scale = self.get_parameter('font_scale').get_parameter_value().double_value
        bbox_thickness = self.get_parameter('bbox_thickness').get_parameter_value().integer_value

        for detection in detections:
            bbox = detection.bounding_box
            if bbox:
                # MediaPipe coordinates are in pixel space
                x1 = max(0, min(int(bbox.origin_x), width - 1))
                y1 = max(0, min(int(bbox.origin_y), height - 1))
                x2 = max(0, min(int(bbox.origin_x + bbox.width), width - 1))
                y2 = max(0, min(int(bbox.origin_y + bbox.height), height - 1))

                # Draw configurable colored bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thickness)

                # Add label if available
                if detection.categories:
                    best_category = max(detection.categories, key=lambda c: c.score if c.score else 0)
                    class_name = best_category.category_name if hasattr(best_category, 'category_name') else 'unknown'
                    confidence = best_category.score if hasattr(best_category, 'score') else 0.0

                    label = f"{class_name}: {int(confidence * 100)}%"

                    # Configurable text placement and styling
                    text_y = y1 - 10 if y1 > 30 else y2 + 25
                    cv2.putText(image, label, (x1, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, text_color, 2)

        # Add object count info overlay with configurable styling
        info_text = f"Objects: {len(detections)}"
        cv2.putText(image, info_text, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (255, 255, 255), 2)

        return image

    def _publish_annotated_image(self, image):
        """Publish annotated image to ROS topic with standard BGR8 encoding."""
        try:
            # Publish directly as BGR8 for maximum ROS2 tool compatibility
            # This ensures compatibility with image_view, rviz2, and rqt_image_view
            ros_image = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = "camera_frame"
            self.annotated_image_pub.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f"Error publishing annotated image: {e}")

    def cleanup(self):
        """Clean up resources."""
        super().cleanup()


def main(args=None):
    """Main entry point."""
    import rclpy
    
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
