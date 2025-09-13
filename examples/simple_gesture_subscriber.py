#!/usr/bin/env python3
"""
Simple example demonstrating how to subscribe to gesture recognition results.

This example shows basic usage of the ros2_mediapipe package for gesture-based
robot control applications.
"""

import rclpy
from rclpy.node import Node
from ros2_mediapipe.msg import HandGesture


class SimpleGestureSubscriber(Node):
    """Example node that subscribes to hand gesture messages."""
    
    def __init__(self):
        super().__init__('simple_gesture_subscriber')
        
        # Subscribe to hand gesture topic
        self.subscription = self.create_subscription(
            HandGesture,
            '/hand_gestures',
            self.gesture_callback,
            10
        )
        
        self.get_logger().info('Simple gesture subscriber started')
        self.get_logger().info('Listening for hand gestures on /hand_gestures topic')
    
    def gesture_callback(self, msg):
        """Process received gesture messages."""
        if msg.is_present:
            self.get_logger().info(
                f'Detected gesture: {msg.gesture_name} '
                f'(confidence: {msg.confidence:.2f}, '
                f'hand: {msg.handedness})'
            )
            
            # Example gesture-based actions
            if msg.gesture_name == 'pointing_up':
                self.get_logger().info('Action: Move forward')
            elif msg.gesture_name == 'pointing_down':
                self.get_logger().info('Action: Move backward')
            elif msg.gesture_name == 'pointing_left':
                self.get_logger().info('Action: Turn left')
            elif msg.gesture_name == 'pointing_right':
                self.get_logger().info('Action: Turn right')
            elif msg.gesture_name == 'open_palm':
                self.get_logger().info('Action: Stop')
            elif msg.gesture_name == 'closed_fist':
                self.get_logger().info('Action: Emergency stop')
            else:
                self.get_logger().info(f'Action: Unknown gesture - {msg.gesture_name}')
        else:
            self.get_logger().debug('No hand detected')


def main(args=None):
    """Main function to run the example node."""
    rclpy.init(args=args)
    
    node = SimpleGestureSubscriber()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
