#!/usr/bin/env python3
"""
Gesture Recognition Performance Test Script

Interactive script that prompts the user to show specific gestures in sequence,
measures recognition accuracy, and collects performance metrics (FPS, latency,
CPU usage, memory usage, temperature).

Supports both Picamera2 (Raspberry Pi) and OpenCV VideoCapture (USB cameras).

Usage:
    python3 test_gesture_performance.py [--duration SECONDS] [--model PATH]

Example:
    python3 test_gesture_performance.py --duration 10 --headless
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

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


# Built-in gestures supported by MediaPipe Gesture Recognizer
GESTURES = [
    "Closed_Fist",
    "Open_Palm",
    "Pointing_Up",
    "Thumb_Down",
    "Thumb_Up",
    "Victory",
    "ILoveYou",
]

GESTURE_EMOJI = {
    "Closed_Fist": "👊",
    "Open_Palm": "🖐️",
    "Pointing_Up": "☝️",
    "Thumb_Down": "👎",
    "Thumb_Up": "👍",
    "Victory": "✌️",
    "ILoveYou": "🤟",
    "None": "❌",
}


class CameraInterface:
    """Camera interface supporting rpicam-vid, Picamera2, and OpenCV backends."""

    def __init__(self, use_picamera: bool = True, camera_id: int = 0,
                 width: int = 640, height: int = 480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.frame_size = width * height * 3  # BGR/RGB = 3 bytes per pixel
        self._camera = None
        self._process = None
        self._backend = None  # 'rpicam', 'picamera2', or 'opencv'
        self.use_picamera = use_picamera

    def _try_rpicam(self) -> bool:
        """Try to initialize rpicam-vid for streaming."""
        import subprocess
        import shutil

        rpicam_path = shutil.which('rpicam-vid')
        if not rpicam_path:
            return False

        try:
            # rpicam-vid streams raw BGR frames to stdout
            cmd = [
                rpicam_path,
                '--width', str(self.width),
                '--height', str(self.height),
                '--framerate', '15',
                '--timeout', '0',  # Run indefinitely
                '--nopreview',
                '-o', '-',  # Output to stdout
                '--codec', 'yuv420'  # Raw YUV format
            ]
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.width * self.height * 3 // 2  # YUV420 size
            )
            time.sleep(1.0)  # Camera warmup
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
            # Try rpicam-vid first (works on Pi without extra Python packages)
            if self._try_rpicam():
                return True
            # Then try Picamera2
            if self._try_picamera2():
                return True
        # Fall back to OpenCV
        return self._try_opencv()

    def read(self) -> Optional[np.ndarray]:
        """Capture a frame. Returns BGR numpy array or None on failure."""
        if self._backend == 'rpicam':
            try:
                # Read YUV420 frame (1.5 bytes per pixel)
                yuv_size = self.width * self.height * 3 // 2
                raw_data = self._process.stdout.read(yuv_size)
                if len(raw_data) != yuv_size:
                    return None
                # Convert YUV420 to BGR
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
        else:  # opencv
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
class GestureTestResult:
    """Results from a single gesture test."""
    expected_gesture: str
    detected_gesture: str
    confidence: float
    latency_ms: float
    correct: bool


@dataclass
class PerformanceMetrics:
    """Aggregate performance metrics."""
    total_frames: int = 0
    total_detections: int = 0
    correct_detections: int = 0
    inference_times_ms: List[float] = field(default_factory=list)
    gesture_results: List[GestureTestResult] = field(default_factory=list)
    
    def add_result(self, result: GestureTestResult):
        self.total_frames += 1
        if result.detected_gesture != "None":
            self.total_detections += 1
        if result.correct:
            self.correct_detections += 1
        self.inference_times_ms.append(result.latency_ms)
        self.gesture_results.append(result)
    
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


def create_gesture_recognizer(
    model_path: str,
    confidence_threshold: float = 0.5
) -> mp_vis.GestureRecognizer:
    """Create MediaPipe GestureRecognizer in IMAGE mode (synchronous)."""
    base_options = mp_py.BaseOptions(model_asset_path=model_path)
    options = mp_vis.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=mp_vis.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=confidence_threshold,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vis.GestureRecognizer.create_from_options(options)


def process_frame(
    recognizer: mp_vis.GestureRecognizer,
    frame: np.ndarray,
    expected_gesture: str
) -> GestureTestResult:
    """Process a single frame and return test result."""
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Run inference with timing
    start_time = time.perf_counter()
    result = recognizer.recognize(mp_image)
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # Extract gesture
    detected_gesture = "None"
    confidence = 0.0
    
    if result.gestures and result.gestures[0]:
        top_gesture = result.gestures[0][0]
        detected_gesture = top_gesture.category_name
        confidence = top_gesture.score
    
    correct = (detected_gesture == expected_gesture)
    
    return GestureTestResult(
        expected_gesture=expected_gesture,
        detected_gesture=detected_gesture,
        confidence=confidence,
        latency_ms=latency_ms,
        correct=correct
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

    # Expected gesture
    expected_text = f"Show: {GESTURE_EMOJI.get(expected, '')} {expected}"
    cv2.putText(frame, expected_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Detected gesture
    color = (0, 255, 0) if detected == expected else (0, 0, 255)
    detected_text = f"Detected: {GESTURE_EMOJI.get(detected, '')} {detected} ({confidence:.0%})"
    cv2.putText(frame, detected_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # FPS and time remaining
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Time: {time_remaining:.1f}s", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def run_gesture_test(
    model_path: str,
    duration_per_gesture: float = 10.0,
    confidence_threshold: float = 0.5,
    camera_id: int = 0,
    headless: bool = False,
    use_picamera: bool = True
) -> dict:
    """
    Run interactive gesture recognition test.

    Args:
        model_path: Path to gesture_recognizer.task model
        duration_per_gesture: Seconds to test each gesture
        confidence_threshold: Minimum confidence for detection
        camera_id: Camera device ID (for OpenCV fallback)
        headless: Run without display (for SSH sessions)
        use_picamera: Use Picamera2 if available (default: True)

    Returns:
        Dictionary with performance metrics and resource usage
    """
    print("=" * 60)
    print("Gesture Recognition Performance Test")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Duration per gesture: {duration_per_gesture}s")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Headless mode: {headless}")
    print(f"Picamera2 available: {PICAMERA2_AVAILABLE}")
    print("-" * 60)

    # Initialize camera using CameraInterface
    camera = CameraInterface(use_picamera=use_picamera, camera_id=camera_id)
    if not camera.start():
        print("ERROR: Could not initialize camera")
        sys.exit(1)

    # Initialize recognizer
    print("Initializing gesture recognizer...")
    recognizer = create_gesture_recognizer(model_path, confidence_threshold)

    # Initialize resource monitor
    resource_monitor = ResourceMonitor(interval_ms=100)

    # Metrics collection
    metrics = PerformanceMetrics()
    per_gesture_metrics: Dict[str, PerformanceMetrics] = {g: PerformanceMetrics() for g in GESTURES}

    print("\nStarting test sequence...")
    if not headless:
        print("Press 'q' to quit, 's' to skip current gesture\n")
    else:
        print("Running in headless mode (no display)\n")

    resource_monitor.start()

    for gesture in GESTURES:
        print(f"\n>>> Show gesture: {GESTURE_EMOJI.get(gesture, '')} {gesture}")
        print(f"    Hold for {duration_per_gesture} seconds...")

        start_time = time.time()
        frame_count = 0
        skip_gesture = False

        while time.time() - start_time < duration_per_gesture:
            frame = camera.read()
            if frame is None:
                continue

            # Process frame
            result = process_frame(recognizer, frame, gesture)
            metrics.add_result(result)
            per_gesture_metrics[gesture].add_result(result)
            frame_count += 1

            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            time_remaining = duration_per_gesture - elapsed

            # Draw overlay and display
            if not headless:
                display_frame = draw_overlay(
                    frame.copy(), gesture, result.detected_gesture,
                    result.confidence, fps, time_remaining
                )
                cv2.imshow("Gesture Test", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nTest aborted by user.")
                    break
                elif key == ord('s'):
                    skip_gesture = True
                    break

            # Print progress
            if frame_count % 30 == 0:
                status = "✓" if result.correct else "✗"
                print(f"    [{status}] {result.detected_gesture} ({result.confidence:.0%}) - {fps:.1f} FPS")

        if skip_gesture:
            print(f"    Skipped {gesture}")
            continue

        # Per-gesture summary
        gesture_summary = per_gesture_metrics[gesture].get_summary()
        print(f"    Accuracy: {gesture_summary.get('accuracy', 0):.1%}, "
              f"FPS: {gesture_summary.get('fps_mean', 0):.1f}")

    resource_monitor.stop()
    camera.stop()
    if not headless:
        cv2.destroyAllWindows()
    recognizer.close()

    # Compile results
    resource_stats = resource_monitor.get_statistics()
    overall_summary = metrics.get_summary()

    results = {
        "overall": overall_summary,
        "resource_usage": resource_stats,
        "per_gesture": {g: per_gesture_metrics[g].get_summary() for g in GESTURES},
        "config": {
            "model_path": model_path,
            "duration_per_gesture": duration_per_gesture,
            "confidence_threshold": confidence_threshold,
            "camera_backend": camera._backend or "unknown",
        }
    }

    return results


def print_summary(results: dict) -> None:
    """Print formatted summary of test results."""
    print("\n" + "=" * 60)
    print("GESTURE RECOGNITION PERFORMANCE SUMMARY")
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

    print(f"\nPer-Gesture Accuracy:")
    for gesture, data in results.get("per_gesture", {}).items():
        acc = data.get("accuracy", 0)
        emoji = GESTURE_EMOJI.get(gesture, "")
        print(f"  {emoji} {gesture}: {acc:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Gesture Recognition Performance Test")
    parser.add_argument("--model", type=str,
                       default="/home/pi/GestureBot/gesturebot_ws/install/ros2_mediapipe/models/gesture_recognizer.task",
                       help="Path to gesture_recognizer.task model")
    parser.add_argument("--duration", type=float, default=10.0,
                       help="Duration per gesture in seconds")
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
    results = run_gesture_test(
        model_path=args.model,
        duration_per_gesture=args.duration,
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
        output_path = Path(__file__).parent / "results" / "gesture_performance.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

