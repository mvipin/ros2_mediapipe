#!/usr/bin/env python3
"""
Simple example demonstrating how to subscribe to pose detection results.

This example shows basic usage of the ros2_mediapipe package for pose-based
robot control applications.
"""

import rclpy
from rclpy.node import Node
from ros2_mediapipe.msg import PoseLandmarks


class SimplePoseSubscriber(Node):
    """Example node that subscribes to pose landmark messages."""
    
    def __init__(self):
        super().__init__('simple_pose_subscriber')
        
        # Subscribe to pose landmarks topic
        self.subscription = self.create_subscription(
            PoseLandmarks,
            '/pose_landmarks',
            self.pose_callback,
            10
        )
        
        self.get_logger().info('Simple pose subscriber started')
        self.get_logger().info('Listening for pose landmarks on /pose_landmarks topic')
    
    def pose_callback(self, msg):
        """Process received pose messages."""
        if msg.is_present:
            self.get_logger().info(
                f'Detected pose: {msg.pose_class} '
                f'(confidence: {msg.pose_confidence:.2f})'
            )
            
            # Example pose-based actions
            if msg.pose_class == 'pointing_left':
                self.get_logger().info('Action: Navigate left')
            elif msg.pose_class == 'pointing_right':
                self.get_logger().info('Action: Navigate right')
            elif msg.pose_class == 'pointing_up':
                self.get_logger().info('Action: Move forward')
            elif msg.pose_class == 'pointing_down':
                self.get_logger().info('Action: Move backward')
            elif msg.pose_class == 'neutral':
                self.get_logger().info('Action: Stay in place')
            else:
                self.get_logger().info(f'Action: Unknown pose - {msg.pose_class}')
            
            # Access individual landmarks if needed
            if len(msg.landmarks) >= 33:
                # Example: Get nose position (landmark 0)
                nose = msg.landmarks[0]
                self.get_logger().debug(
                    f'Nose position: x={nose.x:.3f}, y={nose.y:.3f}, z={nose.z:.3f}'
                )
        else:
            self.get_logger().debug('No pose detected')


def main(args=None):
    """Main function to run the example node."""
    rclpy.init(args=args)
    
    node = SimplePoseSubscriber()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
