#!/usr/bin/env python3
"""
Pose Detection Node for ros2_mediapipe Package

Real-time human pose detection using MediaPipe Pose Landmarker with 33-point
landmark detection and pose classification. Provides both landmark coordinates
and high-level pose actions (arms_raised, pointing_left, etc.) for robotics
and computer vision applications.

Features:
- 33-point pose landmark detection with confidence scores
- Real-time pose classification (4 predefined poses)
- Configurable processing parameters via ROS parameters
- Annotated image output for visualization
- Thread-safe asynchronous processing
"""

import cv2
import numpy as np
import threading
import mediapipe as mp
from typing import Optional, List, Dict

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Import from mpipe package structure
from mpipe.common import MediaPipeBaseNode, MediaPipeCallbackMixin, ProcessingConfig
from mpipe.common.config import LogLevel
from mpipe.controllers import PoseDetectionController
from mpipe.utils import MessageConverter
from ros2_mediapipe.msg import PoseLandmarks


class PoseDetectionNode(MediaPipeBaseNode, MediaPipeCallbackMixin):
    """
    ROS 2 node for real-time human pose detection using MediaPipe.

    This node processes camera images to detect human poses with 33 landmark points
    and classifies poses into predefined actions. It publishes both detailed
    landmark data and high-level pose classifications for downstream applications.

    Publishers:
        /vision/poses (PoseLandmarks): Detected pose landmarks and classification
        /vision/poses/annotated (Image): Annotated image with pose overlay

    Subscribers:
        /camera/image_raw (Image): Input camera images

    Parameters:
        model_path (str): Path to MediaPipe pose landmarker model file
        log_level (str): Logging level (DEBUG, INFO, WARN, ERROR)
        enabled (bool): Enable/disable pose detection processing
        frame_skip (int): Number of frames to skip between processing
    """

    def __init__(self):
        """Initialize pose detection node."""
        # Create temporary node to read parameters
        temp_node = Node('temp_pose_detection_node')

        # Declare and read parameters
        temp_node.declare_parameter('frame_skip', 1)
        temp_node.declare_parameter('confidence_threshold', 0.5)
        temp_node.declare_parameter('max_results', 1)
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
            'pose_detection_node',
            'pose_detection',
            config,
            controller=None
        )

        # Essential components only
        self.controller = None

        # Frame storage for callback access
        self._current_rgb_frame = None
        self._current_frame_timestamp = None
        self._current_pose_landmarks = []
        self.latest_poses = []

        # Pose action state variables
        self.current_pose_action = 'no_pose'
        self.last_pose_action = 'no_pose'
        self.pose_change_count = 0

        # Declare parameters
        self.declare_parameter('model_path', 'models/pose_landmarker.task')
        self.declare_parameter('num_poses', config.max_results)
        self.declare_parameter('min_pose_detection_confidence', config.confidence_threshold)
        self.declare_parameter('min_pose_presence_confidence', 0.5)
        self.declare_parameter('min_tracking_confidence', 0.5)
        self.declare_parameter('frame_skip', config.frame_skip)
        self.declare_parameter('log_level', config.log_level.value)
        self.declare_parameter('debug_mode', config.debug_mode)  # Deprecated, use log_level

        # Topic configuration parameters
        self.declare_parameter('camera_topic', config.topics.camera_topic)
        self.declare_parameter('pose_topic', config.topics.pose_topic)
        self.declare_parameter('pose_annotated_topic', config.topics.pose_annotated_topic)

        # Pose classification parameters
        self.declare_parameter('enable_pose_classification', config.pose_classification.enable_pose_classification)
        self.declare_parameter('horizontal_tolerance', config.pose_classification.horizontal_tolerance)
        self.declare_parameter('pose_stability_frames', config.pose_classification.pose_stability_frames)

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
        pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        pose_annotated_topic = self.get_parameter('pose_annotated_topic').get_parameter_value().string_value

        self.pose_data_pub = self.create_publisher(
            PoseLandmarks,
            pose_topic,
            self.result_qos
        )
        self.annotated_image_pub = self.create_publisher(
            Image,
            pose_annotated_topic,
            self.result_qos
        )

        # Initialize pose detection
        self._initialize_pose_detection()

        self.get_logger().info("Simplified pose detection node initialized")



    def _initialize_pose_detection(self):
        """Initialize MediaPipe pose detection controller."""
        try:
            # Get parameters
            model_path = self.get_parameter('model_path').get_parameter_value().string_value
            num_poses = self.get_parameter('num_poses').get_parameter_value().integer_value
            min_pose_detection_confidence = self.get_parameter('min_pose_detection_confidence').get_parameter_value().double_value
            min_pose_presence_confidence = self.get_parameter('min_pose_presence_confidence').get_parameter_value().double_value
            min_tracking_confidence = self.get_parameter('min_tracking_confidence').get_parameter_value().double_value

            # Resolve model path (relative to ros2_mediapipe package install directory)
            if not model_path.startswith('/'):
                import os
                from ament_index_python.packages import get_package_prefix
                package_prefix = get_package_prefix('ros2_mediapipe')
                model_path = os.path.join(package_prefix, model_path)

            # Create callback
            callback = self.create_callback('pose_detection')

            # Initialize controller
            self.controller = PoseDetectionController(
                model_path=model_path,
                num_poses=num_poses,
                min_pose_detection_confidence=min_pose_detection_confidence,
                min_pose_presence_confidence=min_pose_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_segmentation_masks=False,
                result_callback=callback
            )

            if self.config.debug_mode:
                self.get_logger().info(f"Pose detection initialized with model: {model_path}")

        except Exception as e:
            self.get_logger().error(f"Failed to initialize pose detection: {e}")
            raise



    def _process_callback_results(self, result, output_image, timestamp_ms: int, result_type: str):
        """Process MediaPipe pose detection callback results."""
        try:
            if result_type == "pose_detection":
                self._analyze_pose_results(result, output_image, timestamp_ms)
        except Exception as e:
            if self.config.debug_mode:
                self.get_logger().error(f"Error processing pose callback: {e}")

    def _analyze_pose_results(self, result, output_image, timestamp_ms: int):
        """Analyze pose detection results and publish data."""
        try:
            # Store pose landmarks
            self._current_pose_landmarks = []
            self.latest_poses = []
            
            if result.pose_landmarks:
                # Classify pose action from landmarks
                pose_action = self._classify_pose_action(result.pose_landmarks)

                # Track pose changes for statistics
                if pose_action != self.current_pose_action:
                    self.last_pose_action = self.current_pose_action
                    self.current_pose_action = pose_action
                    self.pose_change_count += 1

                for pose_landmarks in result.pose_landmarks:
                    self._current_pose_landmarks.append(pose_landmarks)

                    # Create pose message with classification
                    pose_msg = self._create_pose_message(pose_landmarks, timestamp_ms, pose_action)
                    self.latest_poses.append(pose_msg)

                    # Publish pose data
                    self.pose_data_pub.publish(pose_msg)

                if self.config.debug_mode:
                    # Handle both MediaPipe pose landmark formats for debug logging
                    pose_first = result.pose_landmarks[0] if result.pose_landmarks else None
                    if pose_first:
                        if hasattr(pose_first, 'landmark'):
                            landmark_count = len(pose_first.landmark)
                        elif hasattr(pose_first, '__len__'):
                            landmark_count = len(pose_first)
                        else:
                            landmark_count = 0
                    else:
                        landmark_count = 0
                    self.get_logger().info(f"Published pose: {pose_action} (landmarks: {landmark_count})")
            else:
                # No poses detected
                self.current_pose_action = 'no_pose'
            
            # Always publish annotated image (even with 0 poses) using stored RGB frame
            if self._current_rgb_frame is not None:
                annotated_image = self._create_annotated_image(self._current_rgb_frame.copy())
                self._publish_annotated_image(annotated_image)
                if self.config.debug_mode:
                    self.get_logger().info("Published annotated pose image")
                
        except Exception as e:
            if self.config.debug_mode:
                self.get_logger().error(f"Error analyzing pose results: {e}")

    def _create_pose_message(self, pose_landmarks, timestamp_ms: int, pose_action: str = 'no_pose') -> PoseLandmarks:
        """Create ROS pose landmarks message with pose action."""
        try:
            msg = MessageConverter.pose_landmarks_to_ros(pose_landmarks, timestamp_ms)
            msg.pose_action = pose_action
            return msg
        except Exception as e:
            if self.config.debug_mode:
                self.get_logger().error(f"Error creating pose message: {e}")
            return PoseLandmarks()

    def _classify_pose_action(self, pose_landmarks) -> str:
        """Classify pose action from landmarks using configurable thresholds."""
        # Check if pose classification is enabled
        enable_classification = self.get_parameter('enable_pose_classification').get_parameter_value().bool_value
        if not enable_classification:
            return 'no_pose'

        if not pose_landmarks or len(pose_landmarks) == 0:
            return 'no_pose'

        try:
            # Get configurable threshold
            horizontal_tolerance = self.get_parameter('horizontal_tolerance').get_parameter_value().double_value

            # Use the first detected pose for action classification
            pose_landmarks_first = pose_landmarks[0]

            # Handle both possible MediaPipe pose landmark structures
            landmarks_list = None
            try:
                if hasattr(pose_landmarks_first, '__iter__') and not hasattr(pose_landmarks_first, 'landmark'):
                    landmarks_list = pose_landmarks_first
                else:
                    landmarks_list = pose_landmarks_first.landmark
            except Exception:
                return 'no_pose'

            if not landmarks_list or len(landmarks_list) < 33:
                return 'no_pose'

            # Get key landmarks (MediaPipe pose landmarks)
            left_shoulder = landmarks_list[11]   # Left shoulder
            right_shoulder = landmarks_list[12]  # Right shoulder
            left_elbow = landmarks_list[13]      # Left elbow
            right_elbow = landmarks_list[14]     # Right elbow
            left_wrist = landmarks_list[15]      # Left wrist
            right_wrist = landmarks_list[16]     # Right wrist

            # Configurable pose classification logic
            # Arms raised: both wrists above shoulders
            if (left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y):
                return 'arms_raised'

            # Pointing left: left arm extended horizontally
            elif (left_wrist.x < left_elbow.x < left_shoulder.x and
                  abs(left_wrist.y - left_shoulder.y) < horizontal_tolerance):
                return 'pointing_left'

            # Pointing right: right arm extended horizontally
            elif (right_wrist.x > right_elbow.x > right_shoulder.x and
                  abs(right_wrist.y - right_shoulder.y) < horizontal_tolerance):
                return 'pointing_right'

            # T-pose: both arms extended horizontally
            elif (abs(left_wrist.y - left_shoulder.y) < horizontal_tolerance and
                  abs(right_wrist.y - right_shoulder.y) < horizontal_tolerance and
                  left_wrist.x < left_shoulder.x and right_wrist.x > right_shoulder.x):
                return 't_pose'

            else:
                return 'no_pose'

        except Exception as e:
            if self.config.debug_mode:
                self.get_logger().error(f'Error classifying pose: {e}')
            return 'no_pose'

    def _create_annotated_image(self, image: np.ndarray) -> np.ndarray:
        """Create annotated image with pose landmarks and skeleton."""
        # Convert RGB to BGR for OpenCV (input is RGB from stored frame)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        height, width = image.shape[:2]
        
        # Draw pose landmarks if available
        if self._current_pose_landmarks:
            for pose_landmarks in self._current_pose_landmarks:
                self._draw_pose_landmarks(image, pose_landmarks)
        
        # Get configurable visualization parameters
        text_color_r = self.get_parameter('text_color_r').get_parameter_value().integer_value
        text_color_g = self.get_parameter('text_color_g').get_parameter_value().integer_value
        text_color_b = self.get_parameter('text_color_b').get_parameter_value().integer_value
        text_color = (text_color_b, text_color_g, text_color_r)  # BGR format

        font_scale = self.get_parameter('font_scale').get_parameter_value().double_value

        # Draw pose action information (generic pose classification only)
        if self.current_pose_action != 'no_pose':
            # Draw detected pose action with consistent font
            text = f"Pose: {self.current_pose_action}"
            cv2.putText(image, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, text_color, 2)

        # Add pose info overlay
        pose_text = f"Poses: {len(self._current_pose_landmarks)}"
        cv2.putText(image, pose_text, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (255, 255, 255), 2)

        return image

    def _draw_pose_landmarks(self, image: np.ndarray, pose_landmarks):
        """Draw pose landmarks and skeleton connections on image."""
        try:
            # Get configurable visualization parameters
            landmark_color_r = self.get_parameter('landmark_color_r').get_parameter_value().integer_value
            landmark_color_g = self.get_parameter('landmark_color_g').get_parameter_value().integer_value
            landmark_color_b = self.get_parameter('landmark_color_b').get_parameter_value().integer_value
            landmark_color = (landmark_color_b, landmark_color_g, landmark_color_r)  # BGR format

            landmark_radius = self.get_parameter('landmark_radius').get_parameter_value().integer_value

            # Handle MediaPipe landmark format
            if hasattr(pose_landmarks, 'landmark'):
                landmark_list = pose_landmarks.landmark
            else:
                landmark_list = pose_landmarks

            # Draw pose skeleton connections first (so they appear behind landmarks)
            self._draw_pose_connections(image, landmark_list)

            # Draw landmarks as configurable colored circles on top
            for landmark in landmark_list:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), landmark_radius, landmark_color, -1)

        except Exception as e:
            if self.config.debug_mode:
                self.get_logger().error(f"Error drawing pose landmarks: {e}")

    def _draw_pose_connections(self, image: np.ndarray, landmark_list):
        """Draw pose skeleton connections between landmarks."""
        # MediaPipe pose connections - same as baseline implementation
        connections = [
            # Torso
            (11, 12),  # Shoulders
            (11, 23), (12, 24),  # Shoulder to hip
            (23, 24),  # Hips
            # Arms
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            # Legs (simplified)
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28),  # Right leg
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

                    # Draw configurable colored lines
                    cv2.line(image, (start_x, start_y), (end_x, end_y), skeleton_color, skeleton_thickness)

        except Exception as e:
            if self.config.debug_mode:
                self.get_logger().error(f"Error drawing pose connections: {e}")

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
            if self.config.debug_mode:
                self.get_logger().error(f"Error publishing annotated image: {e}")

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Optional[Dict]:
        """
        Process frame for pose detection (called from threaded context).
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

    def publish_results(self, results, timestamp: float):
        """Publish results - required abstract method implementation."""
        # This method is required by MediaPipeBaseNode but we handle publishing
        # in the callback methods (_analyze_pose_results)
        pass


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    try:
        node = PoseDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in pose detection node: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
