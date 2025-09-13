#!/usr/bin/env python3
"""
Launch file for simplified pose detection node.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate launch description for simplified pose detection."""
    
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='models/pose_landmarker.task',
        description='Path to pose detection model'
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
    
    frame_skip_arg = DeclareLaunchArgument(
        'frame_skip',
        default_value='1',
        description='Process every Nth frame (1 = process all frames)'
    )
    
    debug_mode_arg = DeclareLaunchArgument(
        'debug_mode',
        default_value='false',
        description='Enable debug mode'
    )

    # Pose detection node
    pose_detection_node = Node(
        package='ros2_mediapipe',
        executable='pose_detection_node.py',
        name='pose_detection_node',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'num_poses': LaunchConfiguration('num_poses'),
            'min_pose_detection_confidence': LaunchConfiguration('min_pose_detection_confidence'),
            'min_pose_presence_confidence': LaunchConfiguration('min_pose_presence_confidence'),
            'min_tracking_confidence': LaunchConfiguration('min_tracking_confidence'),
            'frame_skip': LaunchConfiguration('frame_skip'),
            'debug_mode': LaunchConfiguration('debug_mode')
        }],
        output='screen'
    )

    return LaunchDescription([
        model_path_arg,
        num_poses_arg,
        min_pose_detection_confidence_arg,
        min_pose_presence_confidence_arg,
        min_tracking_confidence_arg,
        frame_skip_arg,
        debug_mode_arg,
        pose_detection_node
    ])
