# Multiple object detection results

std_msgs/Header header

# Detection metadata
string detector_name
float64 processing_time       # milliseconds
int32 total_detections

# Individual object detections
DetectedObject[] objects
