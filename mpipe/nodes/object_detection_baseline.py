#!/usr/bin/env python3
"""
Object Detection Node - Complete Baseline Implementation
Copied from working GestureBot architecture to establish reliable foundation.
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import sys
import os

# Add the package to Python path
sys.path.append('/home/pi/GestureBot/gesturebot_ws/install/ros2_mediapipe/lib/python3.12/site-packages')

# Import from mpipe package structure
from mpipe.controllers import ObjectDetectionController
from mpipe.utils import MessageConverter
from ros2_mediapipe.msg import DetectedObjects


@dataclass
class ProcessingConfig:
    """Configuration for MediaPipe processing."""
    enabled: bool = True
    max_fps: float = 15.0
    frame_skip: int = 1
    confidence_threshold: float = 0.5
    max_results: int = 5
    priority: int = 0


class BufferedLogger:
    """Simplified buffered logger from working GestureBot."""
    
    def __init__(self, buffer_size: int = 200, logger=None, unlimited_mode: bool = False, enabled: bool = True):
        self.buffer_size = buffer_size
        self.logger = logger
        self.unlimited_mode = unlimited_mode
        self.enabled = enabled
        self.lock = threading.Lock()
        self.entry_count = 0
        
        if not self.enabled:
            self.buffer = None
        elif self.unlimited_mode:
            self.buffer = deque()
        else:
            self.buffer = deque(maxlen=self.buffer_size)
    
    def log_event(self, event_type: str, message: str, **kwargs):
        """Add an event to the buffer."""
        with self.lock:
            self.entry_count += 1
        
        if not self.enabled:
            if event_type in ['PUBLISH_ERROR', 'CRITICAL_ERROR', 'INITIALIZATION_ERROR']:
                if self.logger:
                    self.logger.error(f"{event_type}: {message}")
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = {
            'timestamp': timestamp,
            'event_type': event_type,
            'message': message,
            'metadata': kwargs
        }
        
        with self.lock:
            self.buffer.append(entry)
    
    def flush_buffer(self):
        """Flush buffer contents to logger."""
        if not self.enabled or not self.buffer:
            return
        
        with self.lock:
            if self.logger and self.buffer:
                mode_str = "UNLIMITED" if self.unlimited_mode else "CIRCULAR"
                self.logger.info(f"=== BUFFERED LOG DUMP [{mode_str}] ({len(self.buffer)} entries) ===")
                for entry in self.buffer:
                    metadata_str = ""
                    if entry['metadata']:
                        metadata_str = " | " + " | ".join([f"{k}={v}" for k, v in entry['metadata'].items()])
                    self.logger.info(f"[{entry['timestamp']}] {entry['event_type']}: {entry['message']}{metadata_str}")
                self.logger.info(f"=== END BUFFER DUMP (Total processed: {self.entry_count}) ===")
                self.buffer.clear()
    
    def get_stats(self):
        """Get buffer statistics."""
        with self.lock:
            return {
                'enabled': self.enabled,
                'mode': 'unlimited' if self.unlimited_mode else 'circular',
                'buffer_size': len(self.buffer) if self.buffer else 0,
                'total_entries': self.entry_count
            }


class MediaPipeCallbackMixin:
    """Callback mixin from working GestureBot."""
    
    def __init__(self):
        self.callback_lock = threading.Lock()
        self._callback_active = True
        self._base_node_ref = None
    
    def _set_base_node_reference(self, node):
        """Set reference to base node for callback publishing."""
        self._base_node_ref = node
    
    def enable_callback_processing(self):
        """Enable callback processing."""
        with self.callback_lock:
            self._callback_active = True
    
    def disable_callback_processing(self):
        """Disable callback processing."""
        with self.callback_lock:
            self._callback_active = False
    
    def create_callback(self, result_type: str) -> Callable:
        """Create callback function for MediaPipe results."""
        def callback(result, output_image, timestamp_ms):
            with self.callback_lock:
                if not self._callback_active:
                    return
                
                try:
                    timestamp_seconds = timestamp_ms / 1000.0
                    processed_results = self._process_callback_results(
                        result, output_image, timestamp_ms, result_type
                    )
                    
                    if processed_results and self._base_node_ref:
                        self._base_node_ref.publish_results(processed_results, timestamp_seconds)
                        
                except Exception as e:
                    if self._base_node_ref:
                        self._base_node_ref.get_logger().error(f"Error in callback: {e}")
        
        return callback
    
    @abstractmethod
    def _process_callback_results(self, result, output_image, timestamp_ms, result_type):
        """Process MediaPipe callback results. Must be implemented by subclass."""
        pass


class MediaPipeBaseNode(Node, ABC):
    """Base node class from working GestureBot."""
    
    def __init__(self, node_name: str, feature_name: str, config: ProcessingConfig, 
                 enable_buffered_logging: bool = True, unlimited_buffer_mode: bool = False,
                 enable_performance_tracking: bool = False, controller=None):
        super().__init__(node_name)
        
        self.feature_name = feature_name
        self.config = config
        self.enable_performance_tracking = enable_performance_tracking
        
        # Initialize buffered logger
        self.buffered_logger = BufferedLogger(
            buffer_size=200,
            logger=self.get_logger(),
            unlimited_mode=unlimited_buffer_mode,
            enabled=enable_buffered_logging
        )
        
        # Threading and synchronization
        self.processing_lock = threading.Lock()
        self.latest_results = None
        
        # Legacy performance tracking
        self.processing_times = []
        self.frame_counter = 0
        self.last_fps_time = time.time()
        
        # Set up callback mixin reference
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
        
        # Subscribers
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            self.image_qos
        )
        
        # Buffer flush timer
        self.buffer_flush_timer = self.create_timer(10.0, self._flush_buffered_logger)
        
        self.get_logger().info(f'{self.feature_name} node initialized')
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray, timestamp: float) -> Any:
        """Process a single frame. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def publish_results(self, results: Any, timestamp: float) -> None:
        """Publish processing results. Must be implemented by subclasses."""
        pass
    
    def image_callback(self, msg: Image) -> None:
        """Handle incoming camera frames with threaded processing."""
        if not self.config.enabled:
            return
        
        # Frame skipping
        self.frame_counter += 1
        if self.frame_counter % (self.config.frame_skip + 1) != 0:
            return
        
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'rgb8')
            timestamp = time.time()
            
            # Process frame asynchronously (KEY: This prevents blocking)
            threading.Thread(
                target=self._process_frame_async,
                args=(cv_image, timestamp),
                daemon=True
            ).start()
            
        except Exception as e:
            self.get_logger().error(f'[{self.feature_name}] Error in image callback: {e}')
    
    def _process_frame_async(self, frame: np.ndarray, timestamp: float) -> None:
        """Process frame in separate thread."""
        if not self.processing_lock.acquire(blocking=False):
            # Skip frame if still processing previous one
            return
        
        try:
            start_time = time.perf_counter()
            
            # Call subclass implementation
            results = self.process_frame(frame, timestamp)
            
            if results is not None:
                self.latest_results = results
            
            # Track performance
            processing_time = (time.perf_counter() - start_time) * 1000
            self._update_performance_stats(processing_time)
            
        except Exception as e:
            self.get_logger().error(f'[{self.feature_name}] Error processing frame: {e}')
        finally:
            self.processing_lock.release()
    
    def _update_performance_stats(self, processing_time: float) -> None:
        """Update performance statistics."""
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
    
    def _flush_buffered_logger(self):
        """Flush buffered logger."""
        if self.buffered_logger:
            self.buffered_logger.flush_buffer()
    
    def log_buffered_event(self, event_type: str, message: str, **kwargs):
        """Log event to buffered logger."""
        if self.buffered_logger:
            self.buffered_logger.log_event(event_type, message, **kwargs)


class ObjectDetectionNode(MediaPipeBaseNode, MediaPipeCallbackMixin):
    """Complete baseline object detection node from working GestureBot."""

    def __init__(self):
        # Configuration
        config = ProcessingConfig(
            enabled=True,
            max_fps=15.0,
            frame_skip=1,
            confidence_threshold=0.5,
            max_results=5,
            priority=0
        )

        # Initialize MediaPipe controller
        self.controller = None
        self.model_path = None

        # Initialize callback mixin first
        MediaPipeCallbackMixin.__init__(self)

        # Initialize base node
        super().__init__(
            'object_detection_node',
            'object_detection',
            config,
            enable_buffered_logging=True,
            unlimited_buffer_mode=False,
            enable_performance_tracking=False,
            controller=None
        )

        # Detection counter
        self._detection_count = 0

        # Frame storage for callback access
        self._current_rgb_frame = None
        self._current_frame_timestamp = None

        # Message converter
        self.message_converter = MessageConverter()

        # Declare parameters
        self.declare_parameter('confidence_threshold', config.confidence_threshold)
        self.declare_parameter('max_results', config.max_results)
        self.declare_parameter('publish_annotated_images', True)
        self.declare_parameter('model_path', 'models/efficientdet.tflite')
        self.declare_parameter('debug_mode', False)

        # Publishers
        self.detections_pub = self.create_publisher(
            DetectedObjects,
            '/vision/objects',
            self.result_qos
        )
        self.annotated_image_pub = self.create_publisher(
            Image,
            '/vision/objects/annotated',
            self.result_qos
        )

        # Initialize object detection
        self._initialize_object_detection()

        self.get_logger().info("Object Detection Node (Baseline) initialized")

    def _initialize_object_detection(self):
        """Initialize MediaPipe object detection controller."""
        try:
            # Get parameters
            confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
            max_results = self.get_parameter('max_results').get_parameter_value().integer_value
            model_path = self.get_parameter('model_path').get_parameter_value().string_value

            # Resolve model path
            if not model_path.startswith('/'):
                model_path = f"/home/pi/GestureBot/gesturebot_ws/src/gesturebot/{model_path}"

            self.model_path = model_path

            # Create callback
            callback = self.create_callback('object_detection')

            # Initialize controller
            self.controller = ObjectDetectionController(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                max_results=max_results,
                result_callback=callback
            )

            self.get_logger().info(f"Object detection initialized with model: {model_path}")

        except Exception as e:
            self.get_logger().error(f"Failed to initialize object detection: {e}")
            raise

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Optional[Dict]:
        """Process frame for object detection (called from threaded context)."""
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

                debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value
                if debug_mode and self.frame_counter % 30 == 0:
                    self.get_logger().info(f"Submitted frame {self.frame_counter} for processing")
            else:
                self.get_logger().warn("Controller not ready for processing")

            # Return None for callback-based processing
            return None

        except Exception as e:
            self.get_logger().error(f"Error in process_frame: {e}")
            return None

    def _process_callback_results(self, result, output_image, timestamp_ms, result_type):
        """Process MediaPipe callback results."""
        try:
            if result is None:
                return None

            detections = result.detections if result.detections else []

            debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value
            if debug_mode:
                self.get_logger().info(f"Detected {len(detections)} objects")

                # Log details of each detection
                for i, detection in enumerate(detections):
                    if detection.categories:
                        category = detection.categories[0]
                        class_name = category.category_name if hasattr(category, 'category_name') else f"Class_{category.index}"
                        confidence = category.score if hasattr(category, 'score') else 0.0
                        self.get_logger().info(f"  Object {i+1}: {class_name} (confidence: {confidence:.3f})")

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
        """Publish detection results and annotated images."""
        try:
            detections = results.get('detections', [])
            frame = results.get('frame')

            debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value
            if debug_mode:
                self.get_logger().info(f"Publishing results: {len(detections)} detections, frame shape: {frame.shape if frame is not None else 'None'}")

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

                if debug_mode:
                    self.get_logger().info(f"Published detection message with {len(detections)} objects")

            # Publish annotated images
            publish_annotated = self.get_parameter('publish_annotated_images').get_parameter_value().bool_value
            if publish_annotated and frame is not None:
                if debug_mode:
                    self.get_logger().info(f"Starting annotation of frame with {len(detections)} detections")

                annotated_frame = self._annotate_image(frame.copy(), detections)
                self._publish_annotated_image(annotated_frame)

                if debug_mode:
                    self.get_logger().info("Published annotated image")
            elif debug_mode:
                self.get_logger().info(f"Not publishing annotated image: publish_annotated={publish_annotated}, frame is None={frame is None}")

        except Exception as e:
            self.get_logger().error(f"Error publishing results: {e}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")

    def _annotate_image(self, image, detections):
        """Annotate image with detection results."""
        height, width = image.shape[:2]

        debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value
        if debug_mode:
            self.get_logger().info(f"Annotating image: {width}x{height}, {len(detections)} detections")

        for i, detection in enumerate(detections):
            bbox = detection.bounding_box
            if debug_mode:
                self.get_logger().info(f"Detection {i}: bbox={bbox}")

            if bbox:
                # MediaPipe coordinates are already in pixel space (not normalized)
                x1 = int(bbox.origin_x)
                y1 = int(bbox.origin_y)
                x2 = int(bbox.origin_x + bbox.width)
                y2 = int(bbox.origin_y + bbox.height)

                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                if debug_mode:
                    self.get_logger().info(f"  Pixel coords: ({x1},{y1}) to ({x2},{y2})")

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Get detection info
                if detection.categories:
                    # Get the best category (highest confidence) like working GestureBot
                    best_category = max(detection.categories, key=lambda c: c.score if c.score else 0)
                    class_name = best_category.category_name if hasattr(best_category, 'category_name') else 'unknown'
                    confidence = best_category.score if hasattr(best_category, 'score') else 0.0

                    if debug_mode:
                        self.get_logger().info(f"  Label: {class_name} ({confidence:.3f})")

                    # Draw label with percentage like working GestureBot
                    confidence_percent = int(confidence * 100)
                    label = f"{class_name}: {confidence_percent}%"

                    # Calculate text size for background rectangle
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                    # Position text above bounding box, or below if not enough space
                    text_x = x1
                    text_y = y1 - 10 if y1 - 10 > text_height else y2 + text_height + 10

                    # Draw background rectangle for text (filled)
                    cv2.rectangle(
                        image,
                        (text_x, text_y - text_height - baseline),
                        (text_x + text_width, text_y + baseline),
                        (0, 255, 0),
                        -1  # Filled rectangle
                    )

                    # Draw text label in black for good contrast
                    cv2.putText(
                        image,
                        label,
                        (text_x, text_y),
                        font,
                        font_scale,
                        (0, 0, 0),  # Black text
                        thickness,
                        cv2.LINE_AA
                    )
                else:
                    if debug_mode:
                        self.get_logger().info(f"  No categories for detection {i}")
            else:
                if debug_mode:
                    self.get_logger().info(f"  No bounding box for detection {i}")

        # Add info overlay
        info_text = f"Objects: {len(detections)} | Frame: {self.frame_counter}"
        cv2.putText(image, info_text, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if debug_mode and len(detections) > 0:
            self.get_logger().info(f"Annotation complete for {len(detections)} objects")

        return image

    def _publish_annotated_image(self, image):
        """Publish annotated image to ROS topic."""
        try:
            # Convert RGB to BGR for ROS
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ros_image = self.cv_bridge.cv2_to_imgmsg(bgr_image, "bgr8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = "camera_frame"
            self.annotated_image_pub.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")


def main(args=None):
    """Main function."""
    rclpy.init(args=args)

    try:
        node = ObjectDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
