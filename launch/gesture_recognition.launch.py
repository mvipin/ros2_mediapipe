#!/usr/bin/env python3
"""
Launch file for simplified gesture recognition node.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate launch description for simplified gesture recognition."""
    
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='models/gesture_recognizer.task',
        description='Path to gesture recognition model'
    )
    
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.7',
        description='Gesture recognition confidence threshold'
    )
    
    max_hands_arg = DeclareLaunchArgument(
        'max_hands',
        default_value='2',
        description='Maximum number of hands to detect'
    )
    
    debug_mode_arg = DeclareLaunchArgument(
        'debug_mode',
        default_value='false',
        description='Enable debug mode'
    )

    # Gesture recognition node
    gesture_recognition_node = Node(
        package='ros2_mediapipe',
        executable='gesture_recognition_node.py',
        name='gesture_recognition_node',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'max_hands': LaunchConfiguration('max_hands'),
            'debug_mode': LaunchConfiguration('debug_mode')
        }],
        output='screen'
    )

    return LaunchDescription([
        model_path_arg,
        confidence_threshold_arg,
        max_hands_arg,
        debug_mode_arg,
        gesture_recognition_node
    ])
