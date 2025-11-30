# ros2_mediapipe

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ROS 2](https://img.shields.io/badge/ROS-2-blue.svg)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

A ROS 2 package providing stateless MediaPipe integration for computer vision in robotics applications. This package offers real-time pose detection, gesture recognition, and object detection with unified visual consistency and configurable parameters.

## 🚀 Features

- **Object Detection**: Real-time detection of 80 COCO dataset object categories with configurable confidence thresholds, bounding box coordinates, and multi-object tracking capabilities optimized for mobile and edge devices
- **Gesture Recognition**: 21-point hand landmark detection with recognition of 8 predefined gestures (Closed Fist, Open Palm, Pointing Up, Thumbs Up/Down, Victory, ILoveYou) plus support for custom gesture training and multi-hand tracking
- **Pose Detection**: 33-point body landmark detection with pose classification featuring full-body tracking from head to toe, 3D world coordinates, optional segmentation masks, and real-time performance optimized for fitness and movement analysis applications
- **Unified Architecture**: Consistent threading model and callback patterns across all nodes
- **Professional Visualization**: Configurable text annotations with consistent styling
- **Modular Design**: Stateless components for easy integration and testing

## 📋 Requirements

- **ROS 2**: Humble Hawksbill or later
- **Python**: 3.8+
- **OpenCV**: 4.5+
- **MediaPipe**: 0.10+

## 🛠️ Installation

### 1. Clone the Repository

```bash
cd ~/your_ros2_ws/src
git clone https://github.com/your-username/ros2_mediapipe.git
```

### 2. Install Dependencies

```bash
cd ~/your_ros2_ws
rosdep install --from-paths src --ignore-src -r -y --skip-keys=libcamera
```

### 3. Install Python Dependencies

```bash
pip install mediapipe opencv-python numpy
```

### 4. Build the Package

```bash
colcon build --packages-select ros2_mediapipe --event-handlers=console_direct+
source install/setup.bash
```

## 🎯 Quick Start

**Note**: ros2_mediapipe nodes require a camera source publishing to `/camera/image_raw`. Start a camera node first:

```bash
# Start camera (in a separate terminal)
ros2 run camera_ros camera_node --ros-args -p camera:=0 -p width:=640 -p height:=480
```

### Pose Detection

```bash
# Launch pose detection node
ros2 launch ros2_mediapipe pose_detection.launch.py

# View detected poses (with pose_action classification)
ros2 topic echo /vision/pose

# Visualize annotated image
ros2 run image_tools showimage --ros-args -r image:=/vision/pose/annotated
```

### Gesture Recognition

```bash
# Launch gesture recognition node
ros2 launch ros2_mediapipe gesture_recognition.launch.py

# View recognized gestures
ros2 topic echo /vision/gestures

# Visualize annotated image
ros2 run image_tools showimage --ros-args -r image:=/vision/gestures/annotated
```

### Object Detection

```bash
# Launch object detection node
ros2 launch ros2_mediapipe object_detection.launch.py

# View detected objects
ros2 topic echo /vision/objects

# Visualize annotated image
ros2 run image_tools showimage --ros-args -r image:=/vision/objects/annotated
```

## 📡 Topics

### Input Topics (Subscribed)

| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Input camera feed (any encoding, converted internally) |

### Output Topics (Published)

#### Object Detection Node
| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/vision/objects` | `ros2_mediapipe/DetectedObjects` | Detected objects with bounding boxes, labels, confidence |
| `/vision/objects/annotated` | `sensor_msgs/Image` | BGR8 annotated image with bounding boxes |

#### Gesture Recognition Node
| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/vision/gestures` | `ros2_mediapipe/HandGesture` | Hand gesture and 21-point landmarks with confidence |
| `/vision/gestures/annotated` | `sensor_msgs/Image` | BGR8 annotated image with hand landmarks |

#### Pose Detection Node
| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/vision/pose` | `ros2_mediapipe/PoseLandmarks` | 33-point pose landmarks with pose_action classification |
| `/vision/pose/annotated` | `sensor_msgs/Image` | BGR8 annotated image with skeleton overlay |

### Pose Classification Values
The `pose_action` field in PoseLandmarks can be:
- `arms_raised` - Both arms raised above shoulders
- `pointing_left` - Left arm extended horizontally
- `pointing_right` - Right arm extended horizontally
- `t_pose` - Both arms extended horizontally (T-shape)
- `no_pose` - No recognized pose pattern

## ⚙️ Configuration

### Launch Parameters

Each node supports comprehensive parameter configuration:

```bash
# Pose detection with custom parameters
ros2 launch ros2_mediapipe pose_detection.launch.py \
    model_path:=models/pose_landmarker.task \
    num_poses:=2 \
    min_pose_detection_confidence:=0.7 \
    min_pose_presence_confidence:=0.5 \
    min_tracking_confidence:=0.5

# Gesture recognition with custom settings
ros2 launch ros2_mediapipe gesture_recognition.launch.py \
    model_path:=models/gesture_recognizer.task \
    num_hands:=2 \
    min_hand_detection_confidence:=0.8 \
    min_hand_presence_confidence:=0.6

# Object detection with custom model
ros2 launch ros2_mediapipe object_detection.launch.py \
    model_path:=models/efficientdet_lite0.tflite \
    max_results:=10 \
    score_threshold:=0.5
```

### Visualization Configuration

Customize visual annotations through ROS parameters:

```yaml
# Visualization settings
landmark_color: [0, 255, 0]      # Green landmarks (BGR)
skeleton_color: [0, 255, 0]      # Green skeleton lines
text_color: [0, 255, 0]          # Green text
font_scale: 1.0                  # Text size scaling
landmark_radius: 3               # Landmark point size
skeleton_thickness: 2            # Skeleton line thickness
```

## 🏗️ Architecture

### Core Components

- **Base Node**: Unified threading architecture with async frame processing
- **Callback Mixin**: Standardized callback chain for result processing
- **Configuration System**: Centralized parameter management and validation
- **Message Types**: Custom ROS 2 messages for pose, gesture, and object data

### Threading Model

All nodes implement a consistent threading pattern:
1. **Image Callback**: Receives camera frames on ROS callback thread
2. **Async Processing**: Spawns worker thread for MediaPipe processing
3. **Result Publishing**: Thread-safe result publication with frame drop monitoring

## 🔧 Integration Examples

### Robot Navigation Control

```python
import rclpy
from rclpy.node import Node
from ros2_mediapipe.msg import HandGesture

class GestureNavigationNode(Node):
    def __init__(self):
        super().__init__('gesture_navigation')
        self.subscription = self.create_subscription(
            HandGesture,
            '/vision/gestures',
            self.gesture_callback,
            10
        )

    def gesture_callback(self, msg):
        if msg.gesture_name == 'Pointing_Up':
            # Move robot forward
            self.publish_cmd_vel(linear_x=0.5)
        elif msg.gesture_name == 'Open_Palm':
            # Stop robot
            self.publish_cmd_vel(linear_x=0.0)
```

### Pose-Based Robot Control

```python
from ros2_mediapipe.msg import PoseLandmarks
from geometry_msgs.msg import Twist

class PoseControlNode(Node):
    def __init__(self):
        super().__init__('pose_control')
        self.pose_sub = self.create_subscription(
            PoseLandmarks,
            '/vision/pose',
            self.pose_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def pose_callback(self, msg):
        if msg.pose_action == 'pointing_left':
            twist = Twist()
            twist.angular.z = 0.5  # Turn left
            self.cmd_pub.publish(twist)
```

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
colcon test --packages-select ros2_mediapipe

# Manual integration test with camera
# Terminal 1: Start camera
ros2 run camera_ros camera_node --ros-args -p camera:=0 -p width:=640 -p height:=480

# Terminal 2: Start detection node
ros2 launch ros2_mediapipe pose_detection.launch.py debug_mode:=true

# Terminal 3: Verify topic output
ros2 topic echo /vision/pose
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - Google's framework for building perception pipelines
- [ROS 2](https://docs.ros.org/en/humble/) - Robot Operating System
- [OpenCV](https://opencv.org/) - Computer Vision Library

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/ros2_mediapipe/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/ros2_mediapipe/discussions)
- **Documentation**: [Wiki](https://github.com/your-username/ros2_mediapipe/wiki)

---

**Built with ❤️ for the robotics community**
