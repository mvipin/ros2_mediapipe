#!/usr/bin/env python3
"""
Gesture Recognition Baseline for ros2_mediapipe Package
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
from mpipe.controllers import GestureRecognitionController
from mpipe.utils import MessageConverter
from ros2_mediapipe.msg import HandGesture


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


class GestureRecognitionBaselineNode(MediaPipeBaseNode, MediaPipeCallbackMixin):
    """Complete gesture recognition baseline copying working GestureBot architecture."""
    
    # Gesture to navigation command mapping
    GESTURE_COMMANDS = {
        'thumbs_up': 'start_navigation',
        'thumbs_down': 'stop_navigation',
        'open_palm': 'pause_navigation',
        'pointing_up': 'move_forward',
        'pointing_left': 'turn_left',
        'pointing_right': 'turn_right',
        'peace': 'follow_person',
        'fist': 'emergency_stop',
        'wave': 'return_home'
    }
    
    def __init__(self):
        MediaPipeBaseNode.__init__(self, 'gesture_recognition_baseline')
        MediaPipeCallbackMixin.__init__(self)
        
        # Declare parameters
        self.declare_node_parameters()
        
        # Initialize components
        self.message_converter = MessageConverter()
        
        # State variables
        self.latest_gestures = []
        self.latest_image = None
        self._current_hand_landmarks = None
        
        # Publishers
        self.annotated_image_pub = self.create_publisher(
            Image, 
            '/vision/gestures/annotated', 
            self.image_qos
        )
        self.gestures_pub = self.create_publisher(
            HandGesture,
            '/vision/gestures',
            self.result_qos
        )
        
        # Initialize gesture recognition
        self.init_gesture_recognition()
        
        # Subscriber
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            self.image_qos
        )
        
        self.get_logger().info('Gesture Recognition Baseline Node initialized')
        
    def declare_node_parameters(self):
        """Declare ROS parameters."""
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('model_path', 'models/gesture_recognizer.task')
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('max_hands', 2)
        self.declare_parameter('debug_mode', False)
        self.declare_parameter('publish_annotated_images', True)
        
    def init_gesture_recognition(self):
        """Initialize the gesture recognition controller."""
        try:
            # Get parameters
            model_path = self.get_parameter('model_path').get_parameter_value().string_value
            confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
            max_hands = self.get_parameter('max_hands').get_parameter_value().integer_value
            
            # Resolve model path (relative to gesturebot package)
            if not model_path.startswith('/'):
                model_path = f"/home/pi/GestureBot/gesturebot_ws/src/gesturebot/{model_path}"
            
            self.controller = GestureRecognitionController(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                max_hands=max_hands,
                result_callback=self.create_callback('gesture')
            )
            
            self.get_logger().info(f'Gesture recognition controller initialized with model: {model_path}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize gesture recognition: {e}')
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

                # Temporarily commented out for reduced logging
                # debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value
                # if debug_mode and self.frame_count % 30 == 0:  # Log every 30 frames
                #     self.get_logger().info(f"Submitted frame {self.frame_count} for gesture processing")
            else:
                self.get_logger().warn("Gesture controller not ready for processing")

        except Exception as e:
            self.get_logger().error(f'Error in async frame processing: {e}')
        finally:
            self.processing_lock.release()

    def _process_callback_results(self, result, output_image, timestamp_ms: int, result_type: str):
        """Process MediaPipe gesture recognition callback results."""
        debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value

        try:
            # Temporarily commented out for reduced logging
            # if debug_mode:
            #     self.get_logger().info(f"Processing gesture callback results at {timestamp_ms}")

            # Store latest results
            if result and (result.gestures or result.hand_landmarks):
                self.latest_gestures = result.gestures
                self._current_hand_landmarks = result.hand_landmarks

                # Process gesture recognition results
                gesture_info = self.analyze_gesture_results(
                    result.gestures,
                    result.hand_landmarks,
                    result.handedness,
                    timestamp_ms
                )

                if gesture_info:
                    # Publish gesture message
                    gesture_msg = self.create_gesture_message(gesture_info, timestamp_ms)
                    self.gestures_pub.publish(gesture_msg)

                    if debug_mode:
                        self.get_logger().info(f"Published gesture: {gesture_info['gesture_name']} "
                                             f"(confidence: {gesture_info['confidence']:.2f})")
            else:
                # No gestures detected
                self.latest_gestures = []
                self._current_hand_landmarks = None

            # Always publish annotated image (even with 0 gestures)
            if self.latest_image is not None:
                annotated_img = self.annotate_image(self.latest_image.copy())
                self.publish_annotated_image(annotated_img)
                if debug_mode:
                    self.get_logger().info("Published annotated gesture image")

        except Exception as e:
            self.get_logger().error(f'Error processing gesture callback results: {e}')

    def analyze_gesture_results(self, gestures, hand_landmarks, handedness, timestamp_ms: int) -> Optional[Dict]:
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

    def create_gesture_message(self, gesture_info: Dict, timestamp_ms: int) -> HandGesture:
        """Create ROS HandGesture message from gesture info."""
        msg = HandGesture()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame"

        msg.gesture_name = gesture_info['gesture_name']
        msg.confidence = float(gesture_info['confidence'])

        # Set handedness if available
        if gesture_info.get('handedness'):
            handedness = gesture_info['handedness']
            if hasattr(handedness, 'category_name'):
                msg.handedness = handedness.category_name
            else:
                msg.handedness = str(handedness)
        else:
            msg.handedness = "Unknown"

        msg.is_present = True

        # Calculate hand center from landmarks if available (but don't store landmarks)
        if gesture_info.get('hand_landmarks'):
            try:
                landmarks = gesture_info['hand_landmarks']
                # Handle both MediaPipe landmark formats
                if hasattr(landmarks, 'landmark') and landmarks.landmark:
                    landmark_list = landmarks.landmark
                elif hasattr(landmarks, '__iter__') and not hasattr(landmarks, 'landmark'):
                    landmark_list = landmarks
                else:
                    landmark_list = landmarks.landmark if hasattr(landmarks, 'landmark') else landmarks

                # Calculate hand center (average of all landmarks)
                if landmark_list and len(landmark_list) > 0:
                    center_x = sum(lm.x for lm in landmark_list) / len(landmark_list)
                    center_y = sum(lm.y for lm in landmark_list) / len(landmark_list)
                    center_z = sum(lm.z for lm in landmark_list) / len(landmark_list)

                    msg.hand_center.x = float(center_x)
                    msg.hand_center.y = float(center_y)
                    msg.hand_center.z = float(center_z)

                    # Estimate hand size (distance from wrist to middle finger tip)
                    if len(landmark_list) >= 21:  # Standard hand has 21 landmarks
                        wrist = landmark_list[0]  # Wrist
                        middle_tip = landmark_list[12]  # Middle finger tip
                        hand_size = ((middle_tip.x - wrist.x)**2 +
                                   (middle_tip.y - wrist.y)**2 +
                                   (middle_tip.z - wrist.z)**2)**0.5
                        msg.hand_size = float(hand_size)

            except Exception as e:
                self.get_logger().error(f'Error calculating hand center: {e}')
                # Set default values
                msg.hand_center.x = 0.0
                msg.hand_center.y = 0.0
                msg.hand_center.z = 0.0
                msg.hand_size = 0.0

        return msg

    def annotate_image(self, image: np.ndarray) -> np.ndarray:
        """Annotate image with gesture recognition results."""
        annotated = image.copy()

        # Draw hand landmarks if available
        if self._current_hand_landmarks:
            for hand_landmarks in self._current_hand_landmarks:
                # Handle both MediaPipe landmark formats like working GestureBot
                try:
                    if hasattr(hand_landmarks, 'landmark') and hand_landmarks.landmark:
                        # MediaPipe NormalizedLandmarkList format
                        landmark_list = hand_landmarks.landmark
                    elif hasattr(hand_landmarks, '__iter__') and not hasattr(hand_landmarks, 'landmark'):
                        # Already a list of landmarks
                        landmark_list = hand_landmarks
                    else:
                        landmark_list = hand_landmarks.landmark if hasattr(hand_landmarks, 'landmark') else hand_landmarks

                    # Draw hand landmarks
                    for i, landmark in enumerate(landmark_list):
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])

                        # Draw landmark point
                        cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)

                        # Draw landmark index (for debugging)
                        if i % 4 == 0:  # Show every 4th landmark to avoid clutter
                            cv2.putText(annotated, str(i), (x + 5, y - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                    # Draw hand connections
                    self.draw_hand_connections(annotated, hand_landmarks, image.shape)
                except Exception as e:
                    self.get_logger().error(f'Error drawing hand landmarks: {e}')

        # Draw gesture information
        if self.latest_gestures and self.latest_gestures[0]:
            hand_gestures = self.latest_gestures[0]
            if hand_gestures:
                best_gesture = max(hand_gestures, key=lambda g: g.score)

                # Draw gesture name and confidence
                text = f"{best_gesture.category_name}: {best_gesture.score:.2f}"
                cv2.putText(annotated, text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # Draw gesture command mapping if available
                if best_gesture.category_name in self.GESTURE_COMMANDS:
                    command = self.GESTURE_COMMANDS[best_gesture.category_name]
                    cv2.putText(annotated, f"Command: {command}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Draw frame counter and info
        info_text = f"Frame: {self.frame_count} | Hands: {len(self._current_hand_landmarks) if self._current_hand_landmarks else 0}"
        cv2.putText(annotated, info_text, (10, annotated.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return annotated

    def draw_hand_connections(self, image: np.ndarray, landmarks, image_shape):
        """Draw hand landmark connections."""
        # MediaPipe hand connections (simplified)
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
            # Handle both MediaPipe landmark formats like working GestureBot
            if hasattr(landmarks, 'landmark') and landmarks.landmark:
                # MediaPipe NormalizedLandmarkList format
                landmark_list = landmarks.landmark
            elif hasattr(landmarks, '__iter__') and not hasattr(landmarks, 'landmark'):
                # Already a list of landmarks
                landmark_list = landmarks
            else:
                landmark_list = landmarks.landmark if hasattr(landmarks, 'landmark') else landmarks

            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(landmark_list) and end_idx < len(landmark_list):
                    start_point = landmark_list[start_idx]
                    end_point = landmark_list[end_idx]

                    start_x = int(start_point.x * image_shape[1])
                    start_y = int(start_point.y * image_shape[0])
                    end_x = int(end_point.x * image_shape[1])
                    end_y = int(end_point.y * image_shape[0])

                    cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
        except Exception as e:
            self.get_logger().error(f'Error drawing hand connections: {e}')

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
        node = GestureRecognitionBaselineNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in gesture recognition baseline: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
