#!/usr/bin/env python3
"""
Object Detection Launch File for ros2_mediapipe Package
Launches object detection node that processes camera feed and publishes annotated results.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for ros2_mediapipe object detection."""

    # ========================================
    # LAUNCH ARGUMENTS
    # ========================================

    declare_camera_topic = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Input camera topic'
    )

    declare_model_path = DeclareLaunchArgument(
        'model_path',
        default_value='models/efficientdet.tflite',
        description='Path to object detection model (relative to ros2_mediapipe package)'
    )

    declare_confidence_threshold = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='Confidence threshold for object detection'
    )

    declare_max_results = DeclareLaunchArgument(
        'max_results',
        default_value='5',
        description='Maximum number of detection results'
    )

    declare_frame_skip = DeclareLaunchArgument(
        'frame_skip',
        default_value='1',
        description='Process every Nth frame (1 = process all frames)'
    )

    declare_debug_mode = DeclareLaunchArgument(
        'debug_mode',
        default_value='false',
        description='Enable debug mode with additional logging'
    )

    # ========================================
    # OBJECT DETECTION NODE
    # ========================================

    object_detection_node = Node(
        package='ros2_mediapipe',
        executable='object_detection_node.py',
        name='object_detection_node',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'model_path': LaunchConfiguration('model_path'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'max_results': LaunchConfiguration('max_results'),
            'frame_skip': LaunchConfiguration('frame_skip'),
            'debug_mode': LaunchConfiguration('debug_mode'),
        }],
        output='screen'
    )

    # ========================================
    # LAUNCH DESCRIPTION
    # ========================================

    return LaunchDescription([
        # Launch arguments
        declare_camera_topic,
        declare_model_path,
        declare_confidence_threshold,
        declare_max_results,
        declare_frame_skip,
        declare_debug_mode,

        # Object detection node
        object_detection_node
    ])
