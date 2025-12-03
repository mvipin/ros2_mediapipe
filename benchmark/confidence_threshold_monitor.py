#!/usr/bin/env python3
"""
Confidence Threshold Experiment - Data Collection Script

Monitors and logs system metrics during confidence threshold optimization experiments.
Collects CPU usage, temperature, camera frame rate, detection rate, object count, and confidence.

This experiment tests the impact of different confidence_threshold values on detection quality.
The confidence_threshold is applied post-inference as a filter, so CPU impact should be minimal.

Usage:
    python3 confidence_threshold_monitor.py --threshold-config 0.5 --duration 60
    python3 confidence_threshold_monitor.py --threshold-config 0.3 --duration 60 --output-dir ./results

Requirements:
    - ROS 2 environment sourced
    - object_detection.launch.py running with desired confidence_threshold
    - psutil module
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import psutil
except ImportError:
    print("Error: psutil not installed. Run: pip install psutil")
    sys.exit(1)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ros2_mediapipe.msg import DetectedObjects


class DetectionMonitor(Node):
    """ROS 2 node to monitor detection messages for object count, confidence, and detection events."""

    def __init__(self):
        super().__init__('detection_monitor')
        self._lock = threading.Lock()
        self._recent_detections: List[Dict] = []
        self._max_buffer = 50

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.subscription = self.create_subscription(
            DetectedObjects, '/vision/objects', self._detection_callback, qos
        )

    def _detection_callback(self, msg: DetectedObjects):
        confidences = [obj.confidence for obj in msg.objects] if msg.objects else []
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        min_conf = min(confidences) if confidences else 0.0
        max_conf = max(confidences) if confidences else 0.0

        with self._lock:
            self._recent_detections.append({
                'count': msg.total_detections,
                'avg_confidence': avg_conf,
                'min_confidence': min_conf,
                'max_confidence': max_conf,
                'has_detection': len(confidences) > 0,
                'timestamp': time.time()
            })
            if len(self._recent_detections) > self._max_buffer:
                self._recent_detections.pop(0)

    def get_stats(self, window_seconds: float = 5.0) -> Dict:
        now = time.time()
        with self._lock:
            recent = [d for d in self._recent_detections if now - d['timestamp'] < window_seconds]
        if not recent:
            return {'object_count': None, 'avg_confidence': None, 'min_confidence': None,
                    'max_confidence': None, 'detection_rate': None, 'message_count': 0}
        avg_count = sum(d['count'] for d in recent) / len(recent)
        avg_conf = sum(d['avg_confidence'] for d in recent) / len(recent)
        min_conf = min(d['min_confidence'] for d in recent)
        max_conf = max(d['max_confidence'] for d in recent)
        detection_rate = sum(1 for d in recent if d['has_detection']) / len(recent) * 100
        return {'object_count': avg_count, 'avg_confidence': avg_conf, 'min_confidence': min_conf,
                'max_confidence': max_conf, 'detection_rate': detection_rate, 'message_count': len(recent)}


class TopicHzMonitor:
    """Monitor ROS 2 topic frequency using ros2 topic hz."""

    def __init__(self, topic: str, window_size: int = 10):
        self.topic = topic
        self.window_size = window_size
        self.hz_value: Optional[float] = None
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._thread:
            self._thread.join(timeout=2)

    def get_hz(self) -> Optional[float]:
        with self._lock:
            return self.hz_value

    def _monitor_loop(self) -> None:
        ws_setup = os.path.expanduser("~/GestureBot/gesturebot_ws/install/setup.bash")
        cmd = f"source /opt/ros/jazzy/setup.bash && source {ws_setup} && ros2 topic hz {self.topic} -w {self.window_size}"
        try:
            self._process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, shell=True, executable='/bin/bash'
            )
            hz_pattern = re.compile(r'average rate:\s*([\d.]+)')
            while self._running and self._process.poll() is None:
                line = self._process.stdout.readline()
                if line:
                    match = hz_pattern.search(line)
                    if match:
                        with self._lock:
                            self.hz_value = float(match.group(1))
        except Exception as e:
            print(f"Warning: Topic hz monitor error for {self.topic}: {e}")


def get_temperature() -> Optional[float]:
    try:
        result = subprocess.run(['vcgencmd', 'measure_temp'], capture_output=True, text=True, timeout=2)
        match = re.search(r"temp=([\d.]+)'C", result.stdout)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def find_node_process(node_name: str = 'object_detection') -> Optional[psutil.Process]:
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            cmdline_str = ' '.join(cmdline)
            if (node_name in cmdline_str and 'python' in cmdline_str.lower() and
                'ros2 launch' not in cmdline_str and '_node.py' in cmdline_str):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def get_cpu_usage(process: Optional[psutil.Process]) -> Optional[float]:
    if process is None:
        return None
    try:
        return process.cpu_percent()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


class ConfidenceThresholdMonitor:
    """Main monitoring class for confidence threshold experiments."""

    def __init__(self, threshold_config: str, duration: int, output_dir: Path, sample_interval: int = 5, warmup: int = 15):
        self.threshold_config = threshold_config
        self.duration = duration
        self.output_dir = output_dir
        self.sample_interval = sample_interval
        self.warmup = warmup
        self.samples: List[Dict] = []
        self.camera_hz_monitor = TopicHzMonitor('/camera/image_raw')
        self.detection_hz_monitor = TopicHzMonitor('/vision/objects')
        self.detection_node: Optional[DetectionMonitor] = None
        self.node_process: Optional[psutil.Process] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._spinning = False

    def _spin_node(self):
        while self._spinning and rclpy.ok():
            rclpy.spin_once(self.detection_node, timeout_sec=0.1)

    def run(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f'confidence_threshold_experiment_{timestamp}.csv'

        print(f"{'=' * 70}")
        print("Confidence Threshold Experiment - Data Collection")
        print(f"{'=' * 70}")
        print(f"Threshold Config: {self.threshold_config}")
        print(f"Duration: {self.duration}s")
        print(f"Output: {output_file}")
        print(f"{'=' * 70}")

        print("\nInitializing ROS 2 detection monitor...")
        rclpy.init()
        self.detection_node = DetectionMonitor()
        self._spinning = True
        self._spin_thread = threading.Thread(target=self._spin_node, daemon=True)
        self._spin_thread.start()

        print("Searching for object_detection_node process...")
        self.node_process = find_node_process('object_detection')
        if self.node_process:
            print(f"  Found: PID {self.node_process.pid}")
            self.node_process.cpu_percent()
        else:
            print("  WARNING: object_detection_node not found!")

        print("\nStarting topic frequency monitors...")
        self.camera_hz_monitor.start()
        self.detection_hz_monitor.start()

        print(f"\nWarming up for {self.warmup}s to stabilize measurements...")
        time.sleep(self.warmup)

        header = f"{'Time':<8} {'CPU %':<8} {'Temp':<8} {'DetHz':<8} {'ObjCnt':<8} {'AvgConf':<8} {'DetRate':<8}"
        print(f"\nCollecting samples (every {self.sample_interval}s for {self.duration}s)...")
        print("-" * 70)
        print(header)
        print("-" * 70)

        start_time = time.time()
        sample_count = 0
        try:
            while (time.time() - start_time) < self.duration:
                sample = self._collect_sample(sample_count)
                self.samples.append(sample)
                elapsed = int(time.time() - start_time)
                cpu_str = f"{sample['cpu_percent']:.1f}" if sample['cpu_percent'] else "N/A"
                temp_str = f"{sample['temperature_c']:.1f}" if sample['temperature_c'] else "N/A"
                det_str = f"{sample['detection_hz']:.2f}" if sample['detection_hz'] else "N/A"
                obj_str = f"{sample['object_count']:.1f}" if sample['object_count'] is not None else "N/A"
                conf_str = f"{sample['avg_confidence']:.2f}" if sample['avg_confidence'] is not None else "N/A"
                rate_str = f"{sample['detection_rate']:.0f}%" if sample['detection_rate'] is not None else "N/A"
                print(f"{elapsed:<8} {cpu_str:<8} {temp_str:<8} {det_str:<8} {obj_str:<8} {conf_str:<8} {rate_str:<8}")
                sample_count += 1
                time.sleep(self.sample_interval)
        except KeyboardInterrupt:
            print("\n\nInterrupted. Saving collected data...")

        self._spinning = False
        self.camera_hz_monitor.stop()
        self.detection_hz_monitor.stop()
        if self._spin_thread:
            self._spin_thread.join(timeout=2)
        self.detection_node.destroy_node()
        rclpy.shutdown()

        self._save_csv(output_file)
        print(f"\n{'=' * 70}")
        print(f"Data collection complete! Samples: {len(self.samples)}")
        print(f"Output: {output_file}")
        print(f"{'=' * 70}")
        return output_file

    def _collect_sample(self, sample_num: int) -> Dict:
        det_stats = self.detection_node.get_stats() if self.detection_node else {}
        return {
            'timestamp': datetime.now().isoformat(),
            'sample_num': sample_num,
            'threshold_config': self.threshold_config,
            'cpu_percent': get_cpu_usage(self.node_process),
            'temperature_c': get_temperature(),
            'camera_hz': self.camera_hz_monitor.get_hz(),
            'detection_hz': self.detection_hz_monitor.get_hz(),
            'object_count': det_stats.get('object_count'),
            'avg_confidence': det_stats.get('avg_confidence'),
            'min_confidence': det_stats.get('min_confidence'),
            'max_confidence': det_stats.get('max_confidence'),
            'detection_rate': det_stats.get('detection_rate'),
            'notes': ''
        }

    def _save_csv(self, output_file: Path) -> None:
        if not self.samples:
            return
        fieldnames = [
            'timestamp', 'sample_num', 'threshold_config', 'cpu_percent', 'temperature_c',
            'camera_hz', 'detection_hz', 'object_count', 'avg_confidence',
            'min_confidence', 'max_confidence', 'detection_rate', 'notes'
        ]
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.samples)


def main():
    parser = argparse.ArgumentParser(description='Confidence Threshold Experiment - Data Collection')
    parser.add_argument('--threshold-config', required=True, help='Label for threshold config (e.g., "0.5", "0.3")')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds (default: 60)')
    parser.add_argument('--output-dir', type=str, default='gesturebot_ws/src/ros2_mediapipe/benchmark/results')
    parser.add_argument('--sample-interval', type=int, default=5, help='Sample interval in seconds (default: 5)')
    parser.add_argument('--warmup', type=int, default=15, help='Warmup time in seconds before data collection (default: 15)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.home() / 'GestureBot' / args.output_dir

    monitor = ConfidenceThresholdMonitor(args.threshold_config, args.duration, output_dir, args.sample_interval, args.warmup)
    monitor.run()


if __name__ == '__main__':
    main()

