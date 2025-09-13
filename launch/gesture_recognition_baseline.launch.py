#!/usr/bin/env python3
"""
Gesture Recognition Baseline Launch File for ros2_mediapipe Package
Launches gesture recognition baseline node that processes camera feed and publishes gesture results.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for gesture recognition baseline."""
    
    # Declare launch arguments
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Input camera topic'
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='models/gesture_recognizer.task',
        description='Path to MediaPipe gesture recognition model file'
    )
    
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.7',
        description='Confidence threshold for gesture recognition'
    )
    
    max_hands_arg = DeclareLaunchArgument(
        'max_hands',
        default_value='2',
        description='Maximum number of hands to detect'
    )
    
    debug_mode_arg = DeclareLaunchArgument(
        'debug_mode',
        default_value='true',
        description='Enable debug mode'
    )
    
    # Gesture recognition node
    gesture_recognition_node = Node(
        package='ros2_mediapipe',
        executable='gesture_recognition_baseline.py',
        name='gesture_recognition_baseline',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'model_path': LaunchConfiguration('model_path'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'max_hands': LaunchConfiguration('max_hands'),
            'debug_mode': LaunchConfiguration('debug_mode'),
            'publish_annotated_images': True,
        }],
        output='screen'
    )
    
    return LaunchDescription([
        camera_topic_arg,
        model_path_arg,
        confidence_threshold_arg,
        max_hands_arg,
        debug_mode_arg,
        gesture_recognition_node,
    ])
