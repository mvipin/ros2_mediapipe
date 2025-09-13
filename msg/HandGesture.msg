# Hand gesture recognition result

std_msgs/Header header

# Gesture identification
string gesture_name
int32 gesture_id
float64 confidence

# Hand information
string handedness              # "Left" or "Right"
bool is_present

# Gesture properties
geometry_msgs/Point hand_center
float64 hand_size             # normalized size
float64 gesture_duration      # seconds
