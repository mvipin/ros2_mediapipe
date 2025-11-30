# MediaPipe Models for ros2_mediapipe

This directory contains the MediaPipe model files required for the ros2_mediapipe vision nodes.

## Required Models

### 1. Pose Landmarker Model
- **File**: `pose_landmarker.task`
- **Purpose**: 33-point pose landmark detection and tracking
- **Size**: ~10MB

### 2. Gesture Recognizer Model
- **File**: `gesture_recognizer.task`
- **Purpose**: Hand gesture recognition with 21-point landmarks
- **Size**: ~10MB

### 3. Object Detection Model
- **File**: `efficientdet.tflite`
- **Purpose**: Object detection with bounding boxes
- **Size**: ~6MB

## Manual Download

If models are missing, download from MediaPipe:

```bash
# Pose landmarker
wget -O pose_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

# Gesture recognizer
wget -O gesture_recognizer.task \
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"

# Object detection
wget -O efficientdet.tflite \
  "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
```

## Model Verification

```bash
ls -lh *.task *.tflite
```

Expected files:
- `pose_landmarker.task`: ~10MB
- `gesture_recognizer.task`: ~10MB
- `efficientdet.tflite`: ~6MB

## Troubleshooting

### Model Not Found Error
```
ERROR: Could not resolve model path: models/pose_landmarker.task
```

**Solution**: Download the model using the script or manually place it in this directory.

### Model Loading Error
```
ERROR: Failed to initialize MediaPipe: Invalid model file
```

**Solution**: Re-download the model file, it may be corrupted.

### Permission Issues
```
ERROR: Permission denied accessing model file
```

**Solution**: Ensure proper file permissions:
```bash
chmod 644 *.task *.tflite
```
