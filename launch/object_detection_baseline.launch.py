#!/usr/bin/env python3
"""
Launch file for ros2_mediapipe baseline object detection node.
Complete baseline implementation from working GestureBot.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for baseline object detection."""
    
    # Declare launch arguments
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Input camera topic'
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='models/efficientdet.tflite',
        description='Path to MediaPipe model file'
    )
    
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='Confidence threshold for detection'
    )
    
    max_results_arg = DeclareLaunchArgument(
        'max_results',
        default_value='5',
        description='Maximum number of detection results'
    )
    
    debug_mode_arg = DeclareLaunchArgument(
        'debug_mode',
        default_value='true',
        description='Enable debug mode'
    )
    
    # Object detection node
    object_detection_node = Node(
        package='ros2_mediapipe',
        executable='object_detection_baseline.py',
        name='object_detection_baseline',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'model_path': LaunchConfiguration('model_path'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'max_results': LaunchConfiguration('max_results'),
            'debug_mode': LaunchConfiguration('debug_mode'),
            'publish_annotated_images': True,
        }],
        output='screen'
    )
    
    return LaunchDescription([
        camera_topic_arg,
        model_path_arg,
        confidence_threshold_arg,
        max_results_arg,
        debug_mode_arg,
        object_detection_node,
    ])
