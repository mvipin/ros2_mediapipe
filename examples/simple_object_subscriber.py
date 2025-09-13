#!/usr/bin/env python3
"""
Simple example demonstrating how to subscribe to object detection results.

This example shows basic usage of the ros2_mediapipe package for object-based
robot control and navigation applications.
"""

import rclpy
from rclpy.node import Node
from ros2_mediapipe.msg import DetectedObjects


class SimpleObjectSubscriber(Node):
    """Example node that subscribes to detected object messages."""
    
    def __init__(self):
        super().__init__('simple_object_subscriber')
        
        # Subscribe to detected objects topic
        self.subscription = self.create_subscription(
            DetectedObjects,
            '/detected_objects',
            self.objects_callback,
            10
        )
        
        self.get_logger().info('Simple object subscriber started')
        self.get_logger().info('Listening for detected objects on /detected_objects topic')
    
    def objects_callback(self, msg):
        """Process received object detection messages."""
        if len(msg.objects) > 0:
            self.get_logger().info(f'Detected {len(msg.objects)} objects:')
            
            for obj in msg.objects:
                self.get_logger().info(
                    f'  - {obj.class_name}: {obj.confidence:.2f} confidence '
                    f'at ({obj.bbox.center.x:.2f}, {obj.bbox.center.y:.2f})'
                )
                
                # Example object-based actions
                self.handle_object_action(obj)
        else:
            self.get_logger().debug('No objects detected')
    
    def handle_object_action(self, detected_object):
        """Handle actions based on detected objects."""
        class_name = detected_object.class_name.lower()
        confidence = detected_object.confidence
        
        # Only act on high-confidence detections
        if confidence < 0.7:
            return
        
        # Example actions based on object type
        if class_name == 'person':
            self.get_logger().info('Action: Person following mode activated')
            self.follow_person(detected_object)
        elif class_name == 'bottle':
            self.get_logger().info('Action: Navigate to bottle for pickup')
            self.navigate_to_object(detected_object)
        elif class_name == 'chair':
            self.get_logger().info('Action: Avoid chair obstacle')
            self.avoid_obstacle(detected_object)
        elif class_name == 'cup':
            self.get_logger().info('Action: Cup detected - potential pickup target')
            self.mark_pickup_target(detected_object)
        elif class_name in ['car', 'truck', 'bus']:
            self.get_logger().info('Action: Large vehicle detected - maintain safe distance')
            self.maintain_safe_distance(detected_object)
        elif class_name == 'stop sign':
            self.get_logger().info('Action: Stop sign detected - halt movement')
            self.emergency_stop()
        else:
            self.get_logger().info(f'Action: Unknown object "{class_name}" - monitoring')
    
    def follow_person(self, person_obj):
        """Example person following logic."""
        center_x = person_obj.bbox.center.x
        
        # Simple following logic based on person position
        if center_x < 0.4:
            self.get_logger().info('  -> Turn left to center person')
        elif center_x > 0.6:
            self.get_logger().info('  -> Turn right to center person')
        else:
            self.get_logger().info('  -> Move forward to follow person')
    
    def navigate_to_object(self, obj):
        """Example navigation to object logic."""
        center_x = obj.bbox.center.x
        center_y = obj.bbox.center.y
        
        self.get_logger().info(
            f'  -> Navigate to object at screen position ({center_x:.2f}, {center_y:.2f})'
        )
    
    def avoid_obstacle(self, obstacle):
        """Example obstacle avoidance logic."""
        center_x = obstacle.bbox.center.x
        
        if center_x < 0.5:
            self.get_logger().info('  -> Obstacle on left - turn right')
        else:
            self.get_logger().info('  -> Obstacle on right - turn left')
    
    def mark_pickup_target(self, target):
        """Example pickup target marking logic."""
        self.get_logger().info(
            f'  -> Marked {target.class_name} as pickup target '
            f'(confidence: {target.confidence:.2f})'
        )
    
    def maintain_safe_distance(self, vehicle):
        """Example safe distance logic for vehicles."""
        bbox_area = vehicle.bbox.size_x * vehicle.bbox.size_y
        
        if bbox_area > 0.3:  # Large object in view
            self.get_logger().info('  -> Vehicle too close - back away')
        else:
            self.get_logger().info('  -> Vehicle at safe distance - continue')
    
    def emergency_stop(self):
        """Example emergency stop logic."""
        self.get_logger().warn('  -> EMERGENCY STOP - Stop sign detected!')


def main(args=None):
    """Main function to run the example node."""
    rclpy.init(args=args)
    
    node = SimpleObjectSubscriber()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
