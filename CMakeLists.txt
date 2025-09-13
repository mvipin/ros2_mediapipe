cmake_minimum_required(VERSION 3.8)
project(ros2_mediapipe)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DetectedObject.msg"
  "msg/DetectedObjects.msg"
  "msg/HandGesture.msg"
  "msg/PoseLandmarks.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Install Python modules manually to avoid target conflicts
install(DIRECTORY mpipe/
  DESTINATION lib/python3.12/site-packages/mpipe
  FILES_MATCHING PATTERN "*.py"
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.py"
)

# Install executable scripts
install(PROGRAMS
  mpipe/nodes/object_detection_node.py
  mpipe/nodes/gesture_recognition_node.py
  mpipe/nodes/pose_detection_node.py
  mpipe/nodes/object_detection_baseline.py
  mpipe/nodes/gesture_recognition_baseline.py
  mpipe/nodes/pose_detection_baseline.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Export dependencies
ament_export_dependencies(rosidl_default_runtime)
ament_package()
