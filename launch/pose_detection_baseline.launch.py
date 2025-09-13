#!/usr/bin/env python3
"""
Pose Detection Baseline Launch File for ros2_mediapipe Package
Launches pose detection baseline node that processes camera feed and publishes pose results.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for pose detection baseline."""
    
    # Declare launch arguments
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Input camera topic'
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='models/pose_landmarker.task',
        description='Path to MediaPipe pose landmarker model file'
    )
    
    num_poses_arg = DeclareLaunchArgument(
        'num_poses',
        default_value='1',
        description='Maximum number of poses to detect'
    )
    
    min_pose_detection_confidence_arg = DeclareLaunchArgument(
        'min_pose_detection_confidence',
        default_value='0.5',
        description='Minimum confidence for pose detection'
    )
    
    min_pose_presence_confidence_arg = DeclareLaunchArgument(
        'min_pose_presence_confidence',
        default_value='0.5',
        description='Minimum confidence for pose presence'
    )
    
    min_tracking_confidence_arg = DeclareLaunchArgument(
        'min_tracking_confidence',
        default_value='0.5',
        description='Minimum confidence for pose tracking'
    )
    
    output_segmentation_masks_arg = DeclareLaunchArgument(
        'output_segmentation_masks',
        default_value='false',
        description='Enable pose segmentation mask output'
    )
    
    debug_mode_arg = DeclareLaunchArgument(
        'debug_mode',
        default_value='true',
        description='Enable debug mode'
    )
    
    # Pose detection node
    pose_detection_node = Node(
        package='ros2_mediapipe',
        executable='pose_detection_baseline.py',
        name='pose_detection_baseline',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'model_path': LaunchConfiguration('model_path'),
            'num_poses': LaunchConfiguration('num_poses'),
            'min_pose_detection_confidence': LaunchConfiguration('min_pose_detection_confidence'),
            'min_pose_presence_confidence': LaunchConfiguration('min_pose_presence_confidence'),
            'min_tracking_confidence': LaunchConfiguration('min_tracking_confidence'),
            'output_segmentation_masks': LaunchConfiguration('output_segmentation_masks'),
            'debug_mode': LaunchConfiguration('debug_mode'),
            'publish_annotated_images': True,
        }],
        output='screen'
    )
    
    return LaunchDescription([
        camera_topic_arg,
        model_path_arg,
        num_poses_arg,
        min_pose_detection_confidence_arg,
        min_pose_presence_confidence_arg,
        min_tracking_confidence_arg,
        output_segmentation_masks_arg,
        debug_mode_arg,
        pose_detection_node,
    ])
