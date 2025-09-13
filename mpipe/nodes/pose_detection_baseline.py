#!/usr/bin/env python3
"""
Pose Detection Baseline for ros2_mediapipe Package
Complete baseline implementation copying working GestureBot architecture.
"""

import cv2
import numpy as np
import time
import mediapipe as mp
from typing import Optional, Dict, List, Any
import threading
from collections import deque
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Point

import sys
import os

# Add the package to Python path
sys.path.append('/home/pi/GestureBot/gesturebot_ws/install/ros2_mediapipe/lib/python3.12/site-packages')

# Import from mpipe package structure
from mpipe.controllers import PoseDetectionController
from mpipe.utils import MessageConverter
from ros2_mediapipe.msg import PoseLandmarks


class BufferedLogger:
    """Buffered logging system for high-frequency events."""
    
    def __init__(self, buffer_size: int = 200, logger=None, unlimited_mode: bool = False, enabled: bool = True):
        self.buffer_size = buffer_size
        self.logger = logger
        self.unlimited_mode = unlimited_mode
        self.enabled = enabled
        self.buffer = deque(maxlen=None if unlimited_mode else buffer_size)
        self.lock = threading.Lock()
        
    def log_event(self, event_type: str, message: str, **kwargs):
        if not self.enabled:
            return
            
        with self.lock:
            timestamp = datetime.now().isoformat()
            event = {
                'timestamp': timestamp,
                'event_type': event_type,
                'message': message,
                **kwargs
            }
            self.buffer.append(event)
            
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'buffer_size': len(self.buffer),
                'max_size': self.buffer_size if not self.unlimited_mode else 'unlimited',
                'enabled': self.enabled
            }


class MediaPipeBaseNode(Node):
    """Base node with complete MediaPipe architecture from working GestureBot."""
    
    def __init__(self, node_name: str):
        super().__init__(node_name)
        
        # Initialize buffered logging
        self.buffered_logger = BufferedLogger(
            buffer_size=200,
            logger=self.get_logger(),
            unlimited_mode=False,
            enabled=True
        )
        
        # QoS profiles
        self.image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self.result_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Threading and synchronization
        self.processing_lock = threading.Lock()
        self.callback_lock = threading.Lock()
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = deque(maxlen=100)
        
        # CV Bridge
        self.cv_bridge = CvBridge()
        
        # State variables
        self._current_rgb_frame = None
        self._current_frame_timestamp = None
        self._callback_active = False
        
    def log_buffered_event(self, event_type: str, message: str, **kwargs):
        """Log event to buffered logger."""
        self.buffered_logger.log_event(event_type, message, **kwargs)
        
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffered logger statistics."""
        return self.buffered_logger.get_stats()


class MediaPipeCallbackMixin:
    """Callback mixin for MediaPipe processing."""
    
    def __init__(self):
        self.callback_lock = threading.Lock()
        self._callback_active = False
        
    def create_callback(self, result_type: str):
        """Create MediaPipe result callback."""
        def callback(result, output_image, timestamp_ms):
            with self.callback_lock:
                if self._callback_active:
                    return
                self._callback_active = True
                
            try:
                self._process_callback_results(result, output_image, timestamp_ms, result_type)
            finally:
                with self.callback_lock:
                    self._callback_active = False
                    
        return callback
        
    def _process_callback_results(self, result, output_image, timestamp_ms: int, result_type: str):
        """Process MediaPipe callback results."""
        # To be implemented by subclasses
        pass


class PoseDetectionBaselineNode(MediaPipeBaseNode, MediaPipeCallbackMixin):
    """Complete pose detection baseline copying working GestureBot architecture."""
    
    # Simplified 4-pose navigation control system
    POSE_ACTION_MAP = {
        'arms_raised': 'forward',      # Both arms raised → move forward
        'pointing_left': 'left',       # Left arm pointing → turn left
        'pointing_right': 'right',     # Right arm pointing → turn right
        't_pose': 'stop',              # T-pose → stop
        'no_pose': 'stop'              # No clear pose → stop (default)
    }
    
    def __init__(self):
        MediaPipeBaseNode.__init__(self, 'pose_detection_baseline')
        MediaPipeCallbackMixin.__init__(self)
        
        # Declare parameters
        self.declare_node_parameters()
        
        # Initialize components
        self.message_converter = MessageConverter()
        
        # State variables
        self.latest_poses = []
        self.latest_image = None
        self._current_pose_landmarks = None
        self.current_pose_action = 'no_pose'
        self.last_pose_action = 'no_pose'
        self.pose_change_count = 0
        
        # Publishers
        self.annotated_image_pub = self.create_publisher(
            Image, 
            '/vision/pose/annotated', 
            self.image_qos
        )
        self.poses_pub = self.create_publisher(
            PoseLandmarks,
            '/vision/pose',
            self.result_qos
        )
        
        # Initialize pose detection
        self.init_pose_detection()
        
        # Subscriber
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            self.image_qos
        )
        
        self.get_logger().info('Pose Detection Baseline Node initialized')
        
    def declare_node_parameters(self):
        """Declare ROS parameters."""
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('model_path', 'models/pose_landmarker.task')
        self.declare_parameter('num_poses', 1)
        self.declare_parameter('min_pose_detection_confidence', 0.5)
        self.declare_parameter('min_pose_presence_confidence', 0.5)
        self.declare_parameter('min_tracking_confidence', 0.5)
        self.declare_parameter('output_segmentation_masks', False)
        self.declare_parameter('debug_mode', False)
        self.declare_parameter('publish_annotated_images', True)
        
    def init_pose_detection(self):
        """Initialize the pose detection controller."""
        try:
            # Get parameters
            model_path = self.get_parameter('model_path').get_parameter_value().string_value
            num_poses = self.get_parameter('num_poses').get_parameter_value().integer_value
            min_pose_detection_confidence = self.get_parameter('min_pose_detection_confidence').get_parameter_value().double_value
            min_pose_presence_confidence = self.get_parameter('min_pose_presence_confidence').get_parameter_value().double_value
            min_tracking_confidence = self.get_parameter('min_tracking_confidence').get_parameter_value().double_value
            output_segmentation_masks = self.get_parameter('output_segmentation_masks').get_parameter_value().bool_value
            
            # Resolve model path (relative to gesturebot package)
            if not model_path.startswith('/'):
                model_path = f"/home/pi/GestureBot/gesturebot_ws/src/gesturebot/{model_path}"
            
            self.controller = PoseDetectionController(
                model_path=model_path,
                num_poses=num_poses,
                min_pose_detection_confidence=min_pose_detection_confidence,
                min_pose_presence_confidence=min_pose_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_segmentation_masks=output_segmentation_masks,
                result_callback=self.create_callback('pose')
            )
            
            self.get_logger().info(f'Pose detection controller initialized with model: {model_path}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize pose detection: {e}')
            self.controller = None

    def image_callback(self, msg: Image) -> None:
        """Process incoming camera images using threading architecture."""
        # Convert ROS image to OpenCV format
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # Use threading to prevent blocking ROS callback (like working GestureBot)
        timestamp = time.time()
        threading.Thread(
            target=self._process_frame_async,
            args=(cv_image, timestamp),
            daemon=True
        ).start()

    def _process_frame_async(self, frame: np.ndarray, timestamp: float) -> None:
        """Process frame asynchronously to prevent ROS callback blocking."""
        # Non-blocking lock acquisition for performance
        if not self.processing_lock.acquire(blocking=False):
            return  # Skip frame if still processing previous one

        try:
            self.frame_count += 1

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Store latest image for annotation
            self.latest_image = frame

            # Convert to MediaPipe format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=rgb_frame)

            # Submit for async processing
            timestamp_ms = int(timestamp * 1000)

            # Check if controller is ready before processing
            if self.controller and self.controller.is_ready():
                # Store RGB frame for callback access
                self._current_rgb_frame = rgb_frame
                self._current_frame_timestamp = timestamp

                self.controller.detect_async(mp_image, timestamp_ms)

                debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value
                if debug_mode and self.frame_count % 30 == 0:  # Log every 30 frames
                    self.get_logger().info(f"Submitted frame {self.frame_count} for pose processing")
            else:
                self.get_logger().warn("Pose controller not ready for processing")

        except Exception as e:
            self.get_logger().error(f'Error in async frame processing: {e}')
        finally:
            self.processing_lock.release()

    def _process_callback_results(self, result, output_image, timestamp_ms: int, result_type: str):
        """Process MediaPipe pose detection callback results."""
        debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value

        try:
            if debug_mode:
                self.get_logger().info(f"Processing pose callback results at {timestamp_ms}")

            # Store current pose landmarks for visualization
            self._current_pose_landmarks = result.pose_landmarks if result.pose_landmarks else None

            # Process pose classification if poses detected
            if result.pose_landmarks:
                # Classify pose action from landmarks
                pose_action = self._classify_pose_action(result.pose_landmarks)

                # Track pose changes for statistics
                if pose_action != self.current_pose_action:
                    self.last_pose_action = self.current_pose_action
                    self.current_pose_action = pose_action
                    self.pose_change_count += 1

                # Create and publish pose landmarks message with classification
                pose_msg = self._create_pose_landmarks_message(result, timestamp_ms, pose_action)
                self.poses_pub.publish(pose_msg)

                if debug_mode:
                    # Calculate landmark count safely using same pattern as working GestureBot
                    landmark_count = 0
                    if result.pose_landmarks and result.pose_landmarks[0]:
                        try:
                            pose_landmarks_first = result.pose_landmarks[0]
                            if hasattr(pose_landmarks_first, '__iter__') and not hasattr(pose_landmarks_first, 'landmark'):
                                landmark_count = len(pose_landmarks_first)
                            else:
                                landmark_count = len(pose_landmarks_first.landmark)
                        except Exception:
                            landmark_count = 0

                    self.get_logger().info(f"Published pose: {pose_action} (landmarks: {landmark_count})")
            else:
                # No poses detected
                self.current_pose_action = 'no_pose'
                self._current_pose_landmarks = None

            # Always publish annotated image (even with 0 poses)
            if self.latest_image is not None:
                annotated_img = self.annotate_image(self.latest_image.copy())
                self.publish_annotated_image(annotated_img)
                if debug_mode:
                    self.get_logger().info("Published annotated pose image")

        except Exception as e:
            self.get_logger().error(f'Error processing pose callback results: {e}')

    def _classify_pose_action(self, pose_landmarks) -> str:
        """Classify pose action from landmarks (simplified 4-pose system)."""
        if not pose_landmarks or len(pose_landmarks) == 0:
            return 'no_pose'

        try:
            # Use the first detected pose for action classification
            pose_landmarks_first = pose_landmarks[0]

            # Handle both possible MediaPipe pose landmark structures (like working GestureBot)
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

            # Simple pose classification logic
            # Arms raised: both wrists above shoulders
            if (left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y):
                return 'arms_raised'

            # Pointing left: left arm extended horizontally
            elif (left_wrist.x < left_elbow.x < left_shoulder.x and
                  abs(left_wrist.y - left_shoulder.y) < 0.1):
                return 'pointing_left'

            # Pointing right: right arm extended horizontally
            elif (right_wrist.x > right_elbow.x > right_shoulder.x and
                  abs(right_wrist.y - right_shoulder.y) < 0.1):
                return 'pointing_right'

            # T-pose: both arms extended horizontally
            elif (abs(left_wrist.y - left_shoulder.y) < 0.1 and
                  abs(right_wrist.y - right_shoulder.y) < 0.1 and
                  left_wrist.x < left_shoulder.x and right_wrist.x > right_shoulder.x):
                return 't_pose'

            else:
                return 'no_pose'

        except Exception as e:
            self.get_logger().error(f'Error classifying pose: {e}')
            return 'no_pose'

    def _create_pose_landmarks_message(self, result, timestamp_ms: int, pose_action: str) -> PoseLandmarks:
        """Create ROS PoseLandmarks message from MediaPipe result."""
        msg = PoseLandmarks()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame"

        msg.pose_action = pose_action
        msg.num_poses = len(result.pose_landmarks) if result.pose_landmarks else 0
        msg.timestamp_ms = timestamp_ms

        # Add pose landmarks if available (flatten all poses like working GestureBot)
        if result.pose_landmarks:
            # Flatten all pose landmarks into a single array
            for pose_landmarks in result.pose_landmarks:
                # Handle both possible MediaPipe pose landmark structures (like working GestureBot)
                try:
                    # Try accessing as list directly (newer format)
                    if hasattr(pose_landmarks, '__iter__') and not hasattr(pose_landmarks, 'landmark'):
                        # pose_landmarks is already a list of landmarks
                        for landmark in pose_landmarks:
                            point = Point()
                            point.x = float(landmark.x)
                            point.y = float(landmark.y)
                            point.z = float(landmark.z)
                            msg.landmarks.append(point)
                    else:
                        # Try accessing via .landmark attribute (older format)
                        for landmark in pose_landmarks.landmark:
                            point = Point()
                            point.x = float(landmark.x)
                            point.y = float(landmark.y)
                            point.z = float(landmark.z)
                            msg.landmarks.append(point)
                except Exception as e:
                    self.get_logger().error(f'Error processing pose landmarks: {e}')
                    continue

        return msg

    def annotate_image(self, image: np.ndarray) -> np.ndarray:
        """Annotate image with pose detection results."""
        annotated = image.copy()

        # Draw pose landmarks if available
        if self._current_pose_landmarks:
            for pose_landmarks in self._current_pose_landmarks:
                # Handle both possible MediaPipe pose landmark structures (like working GestureBot)
                landmarks_list = None
                try:
                    # Try accessing as list directly (newer format)
                    if hasattr(pose_landmarks, '__iter__') and not hasattr(pose_landmarks, 'landmark'):
                        landmarks_list = pose_landmarks
                    else:
                        # Try accessing via .landmark attribute (older format)
                        landmarks_list = pose_landmarks.landmark
                except Exception as e:
                    self.get_logger().error(f"Error accessing pose landmarks structure: {e}")
                    continue

                if landmarks_list is None:
                    continue

                # Draw pose landmarks
                for i, landmark in enumerate(landmarks_list):
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])

                    # Draw landmark point
                    cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)

                    # Draw landmark index for key points
                    if i in [11, 12, 13, 14, 15, 16]:  # Key arm landmarks
                        cv2.putText(annotated, str(i), (x + 5, y - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                # Draw pose connections
                try:
                    self.draw_pose_connections(annotated, pose_landmarks, image.shape)
                except Exception as e:
                    self.get_logger().error(f'Error drawing pose connections: {e}')

        # Draw pose action information
        if self.current_pose_action != 'no_pose':
            # Draw pose action
            text = f"Pose: {self.current_pose_action}"
            cv2.putText(annotated, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Draw navigation command if available
            if self.current_pose_action in self.POSE_ACTION_MAP:
                command = self.POSE_ACTION_MAP[self.current_pose_action]
                cv2.putText(annotated, f"Command: {command}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Draw frame counter and info
        info_text = f"Frame: {self.frame_count} | Poses: {len(self._current_pose_landmarks) if self._current_pose_landmarks else 0}"
        cv2.putText(annotated, info_text, (10, annotated.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return annotated

    def draw_pose_connections(self, image: np.ndarray, landmarks, image_shape):
        """Draw pose landmark connections (simplified skeleton)."""
        # MediaPipe pose connections (key connections only)
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
            # Handle both possible MediaPipe pose landmark structures (like working GestureBot)
            landmarks_list = None
            try:
                # Try accessing as list directly (newer format)
                if hasattr(landmarks, '__iter__') and not hasattr(landmarks, 'landmark'):
                    landmarks_list = landmarks
                else:
                    # Try accessing via .landmark attribute (older format)
                    landmarks_list = landmarks.landmark
            except Exception as e:
                self.get_logger().error(f"Error accessing pose landmarks structure in connections: {e}")
                return

            if landmarks_list is None:
                return

            for connection in connections:
                start_idx, end_idx = connection
                if (start_idx < len(landmarks_list) and
                    end_idx < len(landmarks_list)):

                    start_point = landmarks_list[start_idx]
                    end_point = landmarks_list[end_idx]

                    start_x = int(start_point.x * image_shape[1])
                    start_y = int(start_point.y * image_shape[0])
                    end_x = int(end_point.x * image_shape[1])
                    end_y = int(end_point.y * image_shape[0])

                    cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
        except Exception as e:
            self.get_logger().error(f'Error drawing pose connections: {e}')

    def publish_annotated_image(self, annotated_image: np.ndarray):
        """Publish annotated image."""
        try:
            # Convert back to ROS Image message
            annotated_msg = self.cv_bridge.cv2_to_imgmsg(annotated_image, 'bgr8')
            annotated_msg.header.stamp = self.get_clock().now().to_msg()
            annotated_msg.header.frame_id = "camera_frame"

            self.annotated_image_pub.publish(annotated_msg)

        except Exception as e:
            self.get_logger().error(f'Failed to publish annotated image: {e}')


def main(args=None):
    """Main function."""
    rclpy.init(args=args)

    try:
        node = PoseDetectionBaselineNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in pose detection baseline: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
