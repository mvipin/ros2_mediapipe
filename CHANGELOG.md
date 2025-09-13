# Changelog

All notable changes to the ros2_mediapipe package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-09-13

### Added
- Initial release of ros2_mediapipe package
- Pose detection node with 33-point landmark detection
- Gesture recognition node with hand gesture classification
- Object detection node with multi-object detection capabilities
- Unified threading architecture across all nodes
- Professional visualization system with configurable parameters
- Custom ROS 2 message types:
  - `PoseLandmarks` for pose detection results
  - `HandGesture` for gesture recognition results
  - `DetectedObjects` for object detection results
- Comprehensive launch files with parameter configuration
- Baseline nodes for simplified usage
- Complete documentation and examples

### Features
- Real-time processing with async threading model
- Configurable visualization parameters (colors, fonts, sizes)
- Pose classification (pointing directions, neutral poses)
- Hand gesture recognition with confidence scoring
- Multi-object detection with bounding boxes
- Professional-grade text annotations
- Memory-efficient stateless design
- Thread-safe result publishing

### Technical Details
- Built on MediaPipe framework
- ROS 2 Jazzy compatibility
- Python 3.8+ support
- OpenCV integration for visualization
- Modular architecture with reusable components
