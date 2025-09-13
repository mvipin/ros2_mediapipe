# Single object detection result

# Object identification
string class_name
int32 class_id
float64 confidence

# Bounding box (pixel coordinates)
int32 bbox_x
int32 bbox_y
int32 bbox_width
int32 bbox_height

# 3D information (if available)
geometry_msgs/Point position_3d
bool has_3d_position

# Tracking information
int32 track_id
bool is_tracked
