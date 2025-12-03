#!/usr/bin/env python3
"""
Pose Detection Performance Test Script

Interactive script that prompts the user to show specific poses in sequence,
measures recognition accuracy, and collects performance metrics (FPS, latency,
CPU usage, memory usage, temperature).

Supports both Picamera2 (Raspberry Pi) and OpenCV VideoCapture (USB cameras).

Usage:
    python3 test_pose_performance.py [--duration SECONDS] [--model PATH]

Example:
    python3 test_pose_performance.py --duration 10 --headless
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision as mp_vis

from resource_monitor import ResourceMonitor

# Try to import Picamera2 for Raspberry Pi camera support
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


# 4-Pose Navigation System poses
POSES = [
    "arms_raised",
    "pointing_left",
    "pointing_right",
    "t_pose",
]

POSE_EMOJI = {
    "arms_raised": "🙌",
    "pointing_left": "👈",
    "pointing_right": "👉",
    "t_pose": "🤸",
    "no_pose": "❌",
}

POSE_INSTRUCTIONS = {
    "arms_raised": "Raise both arms above your head",
    "pointing_left": "Extend your left arm horizontally to the left",
    "pointing_right": "Extend your right arm horizontally to the right",
    "t_pose": "Extend both arms horizontally (T-pose)",
}


class CameraInterface:
    """Camera interface supporting rpicam-vid, Picamera2, and OpenCV backends."""

    def __init__(self, use_picamera: bool = True, camera_id: int = 0,
                 width: int = 640, height: int = 480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self._camera = None
        self._process = None
        self._backend = None
        self.use_picamera = use_picamera

    def _try_rpicam(self) -> bool:
        """Try to initialize rpicam-vid for streaming."""
        import subprocess
        import shutil

        rpicam_path = shutil.which('rpicam-vid')
        if not rpicam_path:
            return False

        try:
            cmd = [
                rpicam_path,
                '--width', str(self.width),
                '--height', str(self.height),
                '--framerate', '15',
                '--timeout', '0',
                '--nopreview',
                '-o', '-',
                '--codec', 'yuv420'
            ]
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.width * self.height * 3 // 2
            )
            time.sleep(1.0)
            if self._process.poll() is not None:
                return False
            self._backend = 'rpicam'
            print(f"rpicam-vid initialized: {self.width}x{self.height} @ 15fps")
            return True
        except Exception as e:
            print(f"rpicam-vid failed: {e}")
            if self._process:
                self._process.terminate()
                self._process = None
            return False

    def _try_picamera2(self) -> bool:
        """Try to initialize Picamera2."""
        if not PICAMERA2_AVAILABLE:
            return False
        try:
            self._camera = Picamera2()
            config = self._camera.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "BGR888"}
            )
            self._camera.configure(config)
            self._camera.start()
            time.sleep(1.0)
            self._backend = 'picamera2'
            print(f"Picamera2 initialized: {self.width}x{self.height}")
            return True
        except Exception as e:
            print(f"Picamera2 failed: {e}")
            self._camera = None
            return False

    def _try_opencv(self) -> bool:
        """Try to initialize OpenCV VideoCapture."""
        self._camera = cv2.VideoCapture(self.camera_id)
        if not self._camera.isOpened():
            print(f"ERROR: Could not open camera {self.camera_id}")
            return False
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._backend = 'opencv'
        print(f"OpenCV VideoCapture initialized: camera {self.camera_id}")
        return True

    def start(self) -> bool:
        """Initialize and start the camera. Returns True on success."""
        if self.use_picamera:
            if self._try_rpicam():
                return True
            if self._try_picamera2():
                return True
        return self._try_opencv()

    def read(self) -> Optional[np.ndarray]:
        """Capture a frame. Returns BGR numpy array or None on failure."""
        if self._backend == 'rpicam':
            try:
                yuv_size = self.width * self.height * 3 // 2
                raw_data = self._process.stdout.read(yuv_size)
                if len(raw_data) != yuv_size:
                    return None
                yuv = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                    (self.height * 3 // 2, self.width)
                )
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                return bgr
            except Exception:
                return None
        elif self._backend == 'picamera2':
            try:
                frame = self._camera.capture_array()
                return frame
            except Exception:
                return None
        else:
            ret, frame = self._camera.read()
            return frame if ret else None

    def stop(self) -> None:
        """Release camera resources."""
        if self._backend == 'rpicam' and self._process:
            self._process.terminate()
            self._process.wait()
            self._process = None
        elif self._backend == 'picamera2' and self._camera:
            try:
                self._camera.stop()
            except Exception:
                pass
            self._camera = None
        elif self._backend == 'opencv' and self._camera:
            self._camera.release()
            self._camera = None
        self._backend = None


@dataclass
class PoseTestResult:
    """Results from a single pose test frame."""
    expected_pose: str
    detected_pose: str
    confidence: float
    latency_ms: float
    correct: bool
    num_landmarks: int = 0


@dataclass
class PerformanceMetrics:
    """Aggregate performance metrics."""
    total_frames: int = 0
    total_detections: int = 0
    correct_detections: int = 0
    inference_times_ms: List[float] = field(default_factory=list)
    pose_results: List[PoseTestResult] = field(default_factory=list)

    def add_result(self, result: PoseTestResult):
        self.total_frames += 1
        if result.detected_pose != "no_pose":
            self.total_detections += 1
        if result.correct:
            self.correct_detections += 1
        self.inference_times_ms.append(result.latency_ms)
        self.pose_results.append(result)

    def get_summary(self) -> dict:
        if not self.inference_times_ms:
            return {}

        times = np.array(self.inference_times_ms)
        return {
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "correct_detections": self.correct_detections,
            "accuracy": self.correct_detections / max(self.total_frames, 1),
            "detection_rate": self.total_detections / max(self.total_frames, 1),
            "fps_mean": 1000.0 / np.mean(times) if np.mean(times) > 0 else 0,
            "latency_mean_ms": float(np.mean(times)),
            "latency_std_ms": float(np.std(times)),
            "latency_p95_ms": float(np.percentile(times, 95)),
            "latency_p99_ms": float(np.percentile(times, 99)),
        }


def create_pose_landmarker(
    model_path: str,
    confidence_threshold: float = 0.5
) -> mp_vis.PoseLandmarker:
    """Create MediaPipe PoseLandmarker in IMAGE mode (synchronous)."""
    base_options = mp_py.BaseOptions(model_asset_path=model_path)
    options = mp_vis.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vis.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=confidence_threshold,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return mp_vis.PoseLandmarker.create_from_options(options)


def classify_pose(landmarks) -> tuple:
    """
    Classify pose action from landmarks.

    Note: MediaPipe labels landmarks from the PERSON's perspective, but X-coordinates
    are in IMAGE space. When facing the camera, the person's left arm appears on the
    right side of the image (higher X values).

    Priority order: arms_raised -> t_pose -> pointing_left -> pointing_right
    T-pose is checked before pointing because it's more specific (both arms extended).

    Returns:
        Tuple of (pose_action, confidence)
    """
    if not landmarks or len(landmarks) < 33:
        return "no_pose", 0.0

    # Get key landmarks
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_elbow = landmarks[13]
    right_elbow = landmarks[14]
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]

    horizontal_tolerance = 0.15

    # Arms raised: both wrists above shoulders
    if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
        # Calculate confidence based on how high arms are raised
        left_diff = left_shoulder.y - left_wrist.y
        right_diff = right_shoulder.y - right_wrist.y
        confidence = min(1.0, (left_diff + right_diff) / 0.4)
        return "arms_raised", confidence

    # T-pose: both arms extended horizontally outward (checked before pointing poses)
    # Person's left arm extends to RIGHT of image (higher X), right arm to LEFT (lower X)
    left_horizontal = abs(left_wrist.y - left_shoulder.y) < horizontal_tolerance
    right_horizontal = abs(right_wrist.y - right_shoulder.y) < horizontal_tolerance
    left_extended = left_wrist.x > left_shoulder.x  # Left arm extends to higher X
    right_extended = right_wrist.x < right_shoulder.x  # Right arm extends to lower X

    if left_horizontal and right_horizontal and left_extended and right_extended:
        confidence = 1.0 - (abs(left_wrist.y - left_shoulder.y) + abs(right_wrist.y - right_shoulder.y)) / (2 * horizontal_tolerance)
        return "t_pose", confidence

    # Pointing left: left arm extended horizontally (person's left extends to higher X in image)
    if (left_wrist.x > left_elbow.x > left_shoulder.x and
        abs(left_wrist.y - left_shoulder.y) < horizontal_tolerance):
        confidence = 1.0 - abs(left_wrist.y - left_shoulder.y) / horizontal_tolerance
        return "pointing_left", confidence

    # Pointing right: right arm extended horizontally (person's right extends to lower X in image)
    if (right_wrist.x < right_elbow.x < right_shoulder.x and
        abs(right_wrist.y - right_shoulder.y) < horizontal_tolerance):
        confidence = 1.0 - abs(right_wrist.y - right_shoulder.y) / horizontal_tolerance
        return "pointing_right", confidence

    return "no_pose", 0.0


def process_frame(
    landmarker: mp_vis.PoseLandmarker,
    frame: np.ndarray,
    expected_pose: str
) -> PoseTestResult:
    """Process a single frame and return test result."""
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run inference with timing
    start_time = time.perf_counter()
    result = landmarker.detect(mp_image)
    latency_ms = (time.perf_counter() - start_time) * 1000

    # Extract pose
    detected_pose = "no_pose"
    confidence = 0.0
    num_landmarks = 0

    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        landmarks = result.pose_landmarks[0]
        num_landmarks = len(landmarks)
        detected_pose, confidence = classify_pose(landmarks)

    correct = (detected_pose == expected_pose)

    return PoseTestResult(
        expected_pose=expected_pose,
        detected_pose=detected_pose,
        confidence=confidence,
        latency_ms=latency_ms,
        correct=correct,
        num_landmarks=num_landmarks
    )



def draw_overlay(
    frame: np.ndarray,
    expected: str,
    detected: str,
    confidence: float,
    fps: float,
    time_remaining: float
) -> np.ndarray:
    """Draw status overlay on frame."""
    h, w = frame.shape[:2]

    # Expected pose with instruction
    expected_text = f"Show: {POSE_EMOJI.get(expected, '')} {expected}"
    cv2.putText(frame, expected_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    instruction = POSE_INSTRUCTIONS.get(expected, "")
    cv2.putText(frame, instruction, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Detected pose
    color = (0, 255, 0) if detected == expected else (0, 0, 255)
    detected_text = f"Detected: {POSE_EMOJI.get(detected, '')} {detected} ({confidence:.0%})"
    cv2.putText(frame, detected_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # FPS and time remaining
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Time: {time_remaining:.1f}s", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def run_pose_test(
    model_path: str,
    duration_per_pose: float = 10.0,
    confidence_threshold: float = 0.5,
    camera_id: int = 0,
    headless: bool = False,
    use_picamera: bool = True
) -> dict:
    """
    Run interactive pose detection test.

    Args:
        model_path: Path to pose_landmarker.task model
        duration_per_pose: Seconds to test each pose
        confidence_threshold: Minimum confidence for detection
        camera_id: Camera device ID (for OpenCV fallback)
        headless: Run without display (for SSH sessions)
        use_picamera: Use Picamera2 if available (default: True)

    Returns:
        Dictionary with performance metrics and resource usage
    """
    print("=" * 60)
    print("Pose Detection Performance Test")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Duration per pose: {duration_per_pose}s")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Headless mode: {headless}")
    print(f"Picamera2 available: {PICAMERA2_AVAILABLE}")
    print("-" * 60)

    # Initialize camera
    camera = CameraInterface(use_picamera=use_picamera, camera_id=camera_id)
    if not camera.start():
        print("ERROR: Could not initialize camera")
        sys.exit(1)

    # Initialize pose landmarker
    print("Initializing pose landmarker...")
    landmarker = create_pose_landmarker(model_path, confidence_threshold)

    # Initialize resource monitor
    resource_monitor = ResourceMonitor(interval_ms=100)

    # Metrics collection
    metrics = PerformanceMetrics()
    per_pose_metrics: Dict[str, PerformanceMetrics] = {p: PerformanceMetrics() for p in POSES}

    print("\nStarting test sequence...")
    if not headless:
        print("Press 'q' to quit, 's' to skip current pose\n")
    else:
        print("Running in headless mode (no display)\n")

    resource_monitor.start()

    for pose in POSES:
        print(f"\n>>> Show pose: {POSE_EMOJI.get(pose, '')} {pose}")
        print(f"    {POSE_INSTRUCTIONS.get(pose, '')}")
        print(f"    Hold for {duration_per_pose} seconds...")

        start_time = time.time()
        frame_count = 0
        skip_pose = False

        while time.time() - start_time < duration_per_pose:
            frame = camera.read()
            if frame is None:
                continue

            # Process frame
            result = process_frame(landmarker, frame, pose)
            metrics.add_result(result)
            per_pose_metrics[pose].add_result(result)
            frame_count += 1

            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            time_remaining = duration_per_pose - elapsed

            # Draw overlay and display
            if not headless:
                display_frame = draw_overlay(
                    frame.copy(), pose, result.detected_pose,
                    result.confidence, fps, time_remaining
                )
                cv2.imshow("Pose Test", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nTest aborted by user.")
                    break
                elif key == ord('s'):
                    skip_pose = True
                    break

            # Print progress
            if frame_count % 30 == 0:
                status = "✓" if result.correct else "✗"
                print(f"    [{status}] {result.detected_pose} ({result.confidence:.0%}) - {fps:.1f} FPS")

        if skip_pose:
            print(f"    Skipped {pose}")
            continue

        # Per-pose summary
        pose_summary = per_pose_metrics[pose].get_summary()
        print(f"    Accuracy: {pose_summary.get('accuracy', 0):.1%}, "
              f"FPS: {pose_summary.get('fps_mean', 0):.1f}")

    resource_monitor.stop()

    # Capture backend before stopping camera
    camera_backend = camera._backend or "unknown"

    camera.stop()
    if not headless:
        cv2.destroyAllWindows()
    landmarker.close()

    # Compile results
    resource_stats = resource_monitor.get_statistics()
    overall_summary = metrics.get_summary()

    results = {
        "overall": overall_summary,
        "resource_usage": resource_stats,
        "per_pose": {p: per_pose_metrics[p].get_summary() for p in POSES},
        "config": {
            "model_path": model_path,
            "duration_per_pose": duration_per_pose,
            "confidence_threshold": confidence_threshold,
            "camera_backend": camera_backend,
        }
    }

    return results


def print_summary(results: dict) -> None:
    """Print formatted summary of test results."""
    print("\n" + "=" * 60)
    print("POSE DETECTION PERFORMANCE SUMMARY")
    print("=" * 60)

    overall = results.get("overall", {})
    resource = results.get("resource_usage", {})

    print(f"\nOverall Accuracy: {overall.get('accuracy', 0):.1%}")
    print(f"Detection Rate: {overall.get('detection_rate', 0):.1%}")
    print(f"Total Frames: {overall.get('total_frames', 0)}")

    print(f"\nPerformance:")
    print(f"  FPS (mean): {overall.get('fps_mean', 0):.1f}")
    print(f"  Latency (mean): {overall.get('latency_mean_ms', 0):.1f} ms")
    print(f"  Latency (p95): {overall.get('latency_p95_ms', 0):.1f} ms")

    print(f"\nResource Usage:")
    print(f"  CPU (mean): {resource.get('cpu_mean', 0):.1f}%")
    print(f"  CPU (p95): {resource.get('cpu_p95', 0):.1f}%")
    print(f"  Memory (mean): {resource.get('memory_mb_mean', 0):.1f} MB")
    print(f"  Temperature (max): {resource.get('temp_c_max', 0):.1f}°C")

    print(f"\nPer-Pose Accuracy:")
    for pose, data in results.get("per_pose", {}).items():
        acc = data.get("accuracy", 0)
        emoji = POSE_EMOJI.get(pose, "")
        print(f"  {emoji} {pose}: {acc:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Pose Detection Performance Test")
    parser.add_argument("--model", type=str,
                       default="/home/pi/GestureBot/gesturebot_ws/install/ros2_mediapipe/models/pose_landmarker.task",
                       help="Path to pose_landmarker.task model")
    parser.add_argument("--duration", type=float, default=10.0,
                       help="Duration per pose in seconds")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device ID (for OpenCV fallback)")
    parser.add_argument("--headless", action="store_true",
                       help="Run without display (for SSH)")
    parser.add_argument("--no-picamera", action="store_true",
                       help="Force OpenCV VideoCapture instead of Picamera2")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")

    args = parser.parse_args()

    # Run test
    results = run_pose_test(
        model_path=args.model,
        duration_per_pose=args.duration,
        confidence_threshold=args.confidence,
        camera_id=args.camera,
        headless=args.headless,
        use_picamera=not args.no_picamera
    )

    # Print summary
    print_summary(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / "results" / "pose_performance.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
