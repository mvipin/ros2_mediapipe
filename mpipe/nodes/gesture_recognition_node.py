#!/usr/bin/env python3
"""
Hand Gesture Recognition Node for ros2_mediapipe Package

Real-time hand gesture recognition using MediaPipe Hand Landmarker with
21-point hand landmark detection and gesture classification. Detects and
classifies common hand gestures for human-robot interaction applications.

Features:
- 21-point hand landmark detection with confidence scores
- Multi-hand support (up to 2 hands simultaneously)
- Real-time gesture classification
- Configurable confidence thresholds and visualization parameters
- Annotated image output with configurable hand landmark styling
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
from ros2_mediapipe.msg import HandGesture

from mpipe.common import ProcessingConfig, MediaPipeBaseNode, MediaPipeCallbackMixin
from mpipe.common.config import LogLevel
from mpipe.controllers import GestureRecognitionController


class GestureRecognitionNode(MediaPipeBaseNode, MediaPipeCallbackMixin):
    """
    ROS 2 node for real-time hand gesture recognition using MediaPipe.

    This node processes camera images to detect hand landmarks and classify
    gestures for human-robot interaction. Supports multi-hand detection and
    provides both detailed landmark data and gesture classifications.

    Publishers:
        /vision/gestures (HandGesture): Detected hand gestures and landmarks
        /vision/gestures/annotated_image (Image): Annotated image with hand overlay

    Subscribers:
        /camera/image_raw (Image): Input camera images

    Parameters:
        model_path (str): Path to MediaPipe gesture recognition model file
        log_level (str): Logging level (DEBUG, INFO, WARN, ERROR)
        enabled (bool): Enable/disable gesture recognition processing
        frame_skip (int): Number of frames to skip between processing
        confidence_threshold (float): Minimum confidence for hand detection
        max_hands (int): Maximum number of hands to detect (default: 2)
    """

    def __init__(self):
        """Initialize gesture recognition node."""
        # Create temporary node to read parameters
        temp_node = Node('temp_gesture_recognition_node')

        # Declare and read parameters
        temp_node.declare_parameter('frame_skip', 1)
        temp_node.declare_parameter('confidence_threshold', 0.7)
        temp_node.declare_parameter('max_results', 2)  # Max hands
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
            'gesture_recognition_node',
            'gesture_recognition',
            config,
            controller=None
        )

        # Essential components only
        self.controller = None

        # Frame storage for callback access
        self._current_rgb_frame = None
        self._current_frame_timestamp = None
        self._current_hand_landmarks = None
        self.latest_gestures = []

        # Declare parameters
        self.declare_parameter('model_path', 'models/gesture_recognizer.task')
        self.declare_parameter('confidence_threshold', config.confidence_threshold)
        self.declare_parameter('max_hands', config.max_results)
        self.declare_parameter('frame_skip', config.frame_skip)
        self.declare_parameter('log_level', config.log_level.value)
        self.declare_parameter('debug_mode', config.debug_mode)  # Deprecated, use log_level

        # Topic configuration parameters
        self.declare_parameter('camera_topic', config.topics.camera_topic)
        self.declare_parameter('gesture_topic', config.topics.gesture_topic)
        self.declare_parameter('gesture_annotated_topic', config.topics.gesture_annotated_topic)

        # Visualization parameters
        self.declare_parameter('landmark_color_r', config.visualization.landmark_color[2])  # BGR to RGB
        self.declare_parameter('landmark_color_g', config.visualization.landmark_color[1])
        self.declare_parameter('landmark_color_b', config.visualization.landmark_color[0])
        self.declare_parameter('text_color_r', config.visualization.text_color[2])
        self.declare_parameter('text_color_g', config.visualization.text_color[1])
        self.declare_parameter('text_color_b', config.visualization.text_color[0])
        self.declare_parameter('font_scale', config.visualization.font_scale)
        self.declare_parameter('landmark_radius', config.visualization.landmark_radius)
        self.declare_parameter('skeleton_thickness', config.visualization.skeleton_thickness)

        # Publishers - use configurable topic names
        gesture_topic = self.get_parameter('gesture_topic').get_parameter_value().string_value
        gesture_annotated_topic = self.get_parameter('gesture_annotated_topic').get_parameter_value().string_value

        self.gestures_pub = self.create_publisher(
            HandGesture,
            gesture_topic,
            self.result_qos
        )
        self.annotated_image_pub = self.create_publisher(
            Image,
            gesture_annotated_topic,
            self.result_qos
        )

        # Initialize gesture recognition
        self._initialize_gesture_recognition()

        self.get_logger().info("Gesture recognition node initialized with comprehensive parameter system")

    def _initialize_gesture_recognition(self):
        """Initialize MediaPipe gesture recognition controller."""
        try:
            # Get parameters
            confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
            max_hands = self.get_parameter('max_hands').get_parameter_value().integer_value
            model_path = self.get_parameter('model_path').get_parameter_value().string_value

            # Resolve model path (relative to gesturebot package)
            if not model_path.startswith('/'):
                model_path = f"/home/pi/GestureBot/gesturebot_ws/src/gesturebot/{model_path}"

            # Create callback
            callback = self.create_callback('gesture_recognition')

            # Initialize controller
            self.controller = GestureRecognitionController(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                max_hands=max_hands,
                result_callback=callback
            )

            if self.config.debug_mode:
                self.get_logger().info(f"Gesture recognition initialized with model: {model_path}")

        except Exception as e:
            self.get_logger().error(f"Failed to initialize gesture recognition: {e}")
            raise

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Optional[Dict]:
        """
        Process frame for gesture recognition (called from threaded context).
        
        CRITICAL: This maintains the exact threading pattern from working GestureBot.
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
        
        CRITICAL: This is part of the essential callback chain from working GestureBot.
        """
        try:
            if result and (result.gestures or result.hand_landmarks):
                self.latest_gestures = result.gestures
                self._current_hand_landmarks = result.hand_landmarks

                # Analyze gesture results
                gesture_info = self._analyze_gesture_results(
                    result.gestures,
                    result.hand_landmarks,
                    result.handedness,
                    timestamp_ms
                )

                if gesture_info:
                    # Create result dictionary
                    result_data = {
                        'gesture_info': gesture_info,
                        'frame': self._current_rgb_frame,
                        'timestamp': timestamp_ms / 1000.0
                    }
                    return result_data
            else:
                # No gestures detected
                self.latest_gestures = []
                self._current_hand_landmarks = None

            # Return frame for annotation even with no gestures
            return {
                'gesture_info': None,
                'frame': self._current_rgb_frame,
                'timestamp': timestamp_ms / 1000.0
            }

        except Exception as e:
            self.get_logger().error(f"Error processing callback results: {e}")
            return None

    def _analyze_gesture_results(self, gestures, hand_landmarks, handedness, timestamp_ms: int) -> Optional[Dict]:
        """Analyze gesture recognition results and extract best gesture."""
        if not gestures or not gestures[0]:
            return None

        try:
            # Get the first hand's gestures
            hand_gestures = gestures[0]
            if not hand_gestures:
                return None

            # Find the gesture with highest confidence
            best_gesture = max(hand_gestures, key=lambda g: g.score)

            if best_gesture.score < 0.5:  # Minimum confidence threshold
                return None

            gesture_info = {
                'gesture_name': best_gesture.category_name,
                'confidence': best_gesture.score,
                'timestamp_ms': timestamp_ms,
                'hand_landmarks': hand_landmarks[0] if hand_landmarks else None,
                'handedness': handedness[0] if handedness else None
            }

            return gesture_info

        except Exception as e:
            self.get_logger().error(f'Error analyzing gesture results: {e}')
            return None

    def publish_results(self, results: Dict, timestamp: float) -> None:
        """
        Publish gesture results and annotated images.
        
        CRITICAL: This completes the essential callback chain from working GestureBot.
        """
        try:
            gesture_info = results.get('gesture_info')
            frame = results.get('frame')

            # Publish gesture message if detected
            if gesture_info:
                gesture_msg = self._create_gesture_message(gesture_info, timestamp)
                self.gestures_pub.publish(gesture_msg)

                if self.config.debug_mode:
                    self.get_logger().info(f"Published gesture: {gesture_info['gesture_name']} "
                                         f"(confidence: {gesture_info['confidence']:.2f})")

            # Publish annotated image
            if frame is not None:
                annotated_frame = self._annotate_image(frame.copy())
                self._publish_annotated_image(annotated_frame)

        except Exception as e:
            self.get_logger().error(f"Error publishing results: {e}")

    def _create_gesture_message(self, gesture_info: Dict, timestamp: float) -> HandGesture:
        """Create ROS HandGesture message from gesture info."""
        msg = HandGesture()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame"

        msg.gesture_name = gesture_info['gesture_name']
        msg.confidence = float(gesture_info['confidence'])
        msg.is_present = True

        # Set handedness if available
        if gesture_info.get('handedness'):
            handedness = gesture_info['handedness']
            if hasattr(handedness, 'category_name'):
                msg.handedness = handedness.category_name
            else:
                msg.handedness = str(handedness)
        else:
            msg.handedness = "Unknown"

        return msg

    def _annotate_image(self, image: np.ndarray) -> np.ndarray:
        """Annotate image with gesture recognition results using configurable styling."""
        height, width = image.shape[:2]

        # Draw hand landmarks if available
        if self._current_hand_landmarks:
            for hand_landmarks in self._current_hand_landmarks:
                self._draw_hand_landmarks(image, hand_landmarks)

        # Get configurable text styling
        text_color_r = self.get_parameter('text_color_r').get_parameter_value().integer_value
        text_color_g = self.get_parameter('text_color_g').get_parameter_value().integer_value
        text_color_b = self.get_parameter('text_color_b').get_parameter_value().integer_value
        text_color = (text_color_b, text_color_g, text_color_r)  # BGR format

        font_scale = self.get_parameter('font_scale').get_parameter_value().double_value

        # Add gesture info overlay with configurable styling (consistent with pose detection)
        if self.latest_gestures and self.latest_gestures[0]:
            best_gesture = max(self.latest_gestures[0], key=lambda g: g.score)
            if best_gesture.score >= 0.5:
                # Draw detected gesture with consistent font
                gesture_text = f"Gesture: {best_gesture.category_name}: {int(best_gesture.score * 100)}%"
                cv2.putText(image, gesture_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, text_color, 2)

        # Add gesture count info overlay (consistent with pose count display)
        gesture_count = len(self._current_hand_landmarks) if self._current_hand_landmarks else 0
        info_text = f"Hands: {gesture_count}"
        cv2.putText(image, info_text, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (255, 255, 255), 2)

        return image

    def _draw_hand_landmarks(self, image: np.ndarray, hand_landmarks):
        """Draw hand landmarks and skeleton connections on image using configurable styling."""
        try:
            # Get configurable visualization parameters
            landmark_color_r = self.get_parameter('landmark_color_r').get_parameter_value().integer_value
            landmark_color_g = self.get_parameter('landmark_color_g').get_parameter_value().integer_value
            landmark_color_b = self.get_parameter('landmark_color_b').get_parameter_value().integer_value
            landmark_color = (landmark_color_b, landmark_color_g, landmark_color_r)  # BGR format

            landmark_radius = self.get_parameter('landmark_radius').get_parameter_value().integer_value

            # Handle MediaPipe landmark format
            if hasattr(hand_landmarks, 'landmark'):
                landmark_list = hand_landmarks.landmark
            else:
                landmark_list = hand_landmarks

            # Draw hand skeleton connections first (so they appear behind landmarks)
            self._draw_hand_connections(image, landmark_list)

            # Draw landmarks as configurable colored circles on top
            for landmark in landmark_list:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), landmark_radius, landmark_color, -1)

        except Exception as e:
            if self.config.debug_mode:
                self.get_logger().error(f"Error drawing hand landmarks: {e}")

    def _draw_hand_connections(self, image: np.ndarray, landmark_list):
        """Draw hand skeleton connections between landmarks using configurable styling."""
        # MediaPipe hand connections - same as baseline implementation
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm connections
            (5, 9), (9, 13), (13, 17)
        ]

        try:
            # Get configurable skeleton parameters
            landmark_color_r = self.get_parameter('landmark_color_r').get_parameter_value().integer_value
            landmark_color_g = self.get_parameter('landmark_color_g').get_parameter_value().integer_value
            landmark_color_b = self.get_parameter('landmark_color_b').get_parameter_value().integer_value
            skeleton_color = (landmark_color_b, landmark_color_g, landmark_color_r)  # BGR format, same as landmarks

            skeleton_thickness = self.get_parameter('skeleton_thickness').get_parameter_value().integer_value

            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(landmark_list) and end_idx < len(landmark_list):
                    start_point = landmark_list[start_idx]
                    end_point = landmark_list[end_idx]

                    start_x = int(start_point.x * image.shape[1])
                    start_y = int(start_point.y * image.shape[0])
                    end_x = int(end_point.x * image.shape[1])
                    end_y = int(end_point.y * image.shape[0])

                    # Draw configurable colored lines to match the landmark circles
                    cv2.line(image, (start_x, start_y), (end_x, end_y), skeleton_color, skeleton_thickness)

        except Exception as e:
            if self.config.debug_mode:
                self.get_logger().error(f"Error drawing hand connections: {e}")

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
    node = GestureRecognitionNode()
    
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
