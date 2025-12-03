#!/usr/bin/env python3
"""
Camera FPS Experiment - Data Collection Script

Monitors and logs system metrics during camera FPS optimization experiments.
Collects CPU usage, temperature, camera frame rate, and detection rate.

Usage:
    python3 camera_fps_monitor.py --fps-config 5fps --duration 300
    python3 camera_fps_monitor.py --fps-config 10fps --duration 300 --output-dir ./results

Requirements:
    - ROS 2 environment sourced
    - object_detection.launch.py running
    - psutil, subprocess modules
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
from typing import Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:
    print("Error: psutil not installed. Run: pip install psutil")
    sys.exit(1)


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
        """Start monitoring topic frequency."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop monitoring."""
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
        """Get current Hz value."""
        with self._lock:
            return self.hz_value
    
    def _monitor_loop(self) -> None:
        """Background loop to parse ros2 topic hz output."""
        # Need to source ROS 2 environment and workspace for custom message types
        ws_setup = os.path.expanduser("~/GestureBot/gesturebot_ws/install/setup.bash")
        cmd = f"source /opt/ros/jazzy/setup.bash && source {ws_setup} && ros2 topic hz {self.topic} -w {self.window_size}"
        try:
            self._process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, shell=True, executable='/bin/bash'
            )
            # Pattern: "average rate: 5.00"
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
    """Get Raspberry Pi temperature using vcgencmd."""
    try:
        result = subprocess.run(
            ['vcgencmd', 'measure_temp'],
            capture_output=True, text=True, timeout=2
        )
        # Parse "temp=45.0'C"
        match = re.search(r"temp=([\d.]+)'C", result.stdout)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def find_node_process(node_name: str = 'object_detection') -> Optional[psutil.Process]:
    """Find the ROS 2 node process by name pattern.

    Looks for the actual Python node process, not the ros2 launch wrapper.
    The node process runs via the executable in the install directory.
    """
    candidates = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            cmdline_str = ' '.join(cmdline)
            # Look for the actual node executable, not ros2 launch
            if (node_name in cmdline_str and
                'python' in cmdline_str.lower() and
                'ros2 launch' not in cmdline_str and
                '_node.py' in cmdline_str):
                candidates.append((proc, cmdline_str))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Return the first matching process
    if candidates:
        return candidates[0][0]
    return None


def get_cpu_usage(process: Optional[psutil.Process]) -> Optional[float]:
    """Get CPU usage for a process."""
    if process is None:
        return None
    try:
        return process.cpu_percent()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


class CameraFPSMonitor:
    """Main monitoring class for camera FPS experiments."""

    def __init__(
        self,
        fps_config: str,
        duration: int,
        output_dir: Path,
        sample_interval: int = 5
    ):
        self.fps_config = fps_config
        self.duration = duration
        self.output_dir = output_dir
        self.sample_interval = sample_interval
        self.samples: List[Dict] = []

        # Initialize monitors with correct GestureBot topic names
        self.camera_hz_monitor = TopicHzMonitor('/camera/image_raw')
        self.detection_hz_monitor = TopicHzMonitor('/vision/objects')
        self.node_process: Optional[psutil.Process] = None

    def run(self) -> Path:
        """Run the monitoring session and return output file path."""
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f'camera_fps_experiment_{timestamp}.csv'

        print(f"=" * 60)
        print(f"Camera FPS Experiment - Data Collection")
        print(f"=" * 60)
        print(f"FPS Config:      {self.fps_config}")
        print(f"Duration:        {self.duration} seconds")
        print(f"Sample Interval: {self.sample_interval} seconds")
        print(f"Output File:     {output_file}")
        print(f"=" * 60)

        # Find the object detection node process
        print("\nSearching for object_detection_node process...")
        self.node_process = find_node_process('object_detection')
        if self.node_process:
            print(f"  Found: PID {self.node_process.pid}")
            # Prime CPU measurement
            self.node_process.cpu_percent()
        else:
            print("  WARNING: object_detection_node not found!")
            print("  CPU monitoring will be unavailable.")

        # Start topic Hz monitors
        print("\nStarting topic frequency monitors...")
        self.camera_hz_monitor.start()
        self.detection_hz_monitor.start()
        print("  /camera/image_raw - monitoring")
        print("  /vision/objects - monitoring")

        # Wait a moment for Hz monitors to get initial readings
        time.sleep(3)

        # Collect samples
        print(f"\nCollecting samples (every {self.sample_interval}s for {self.duration}s)...")
        print("-" * 60)
        print(f"{'Time':<10} {'CPU %':<10} {'Temp °C':<10} {'Cam Hz':<10} {'Det Hz':<10}")
        print("-" * 60)

        start_time = time.time()
        sample_count = 0

        try:
            while (time.time() - start_time) < self.duration:
                sample = self._collect_sample(sample_count)
                self.samples.append(sample)

                # Print current sample
                elapsed = int(time.time() - start_time)
                cpu_str = f"{sample['cpu_percent']:.1f}" if sample['cpu_percent'] else "N/A"
                temp_str = f"{sample['temperature_c']:.1f}" if sample['temperature_c'] else "N/A"
                cam_str = f"{sample['camera_hz']:.2f}" if sample['camera_hz'] else "N/A"
                det_str = f"{sample['detection_hz']:.2f}" if sample['detection_hz'] else "N/A"
                print(f"{elapsed:<10} {cpu_str:<10} {temp_str:<10} {cam_str:<10} {det_str:<10}")

                sample_count += 1
                time.sleep(self.sample_interval)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Saving collected data...")

        # Stop monitors
        self.camera_hz_monitor.stop()
        self.detection_hz_monitor.stop()

        # Save to CSV
        self._save_csv(output_file)

        print(f"\n{'=' * 60}")
        print(f"Data collection complete!")
        print(f"Samples collected: {len(self.samples)}")
        print(f"Output saved to: {output_file}")
        print(f"{'=' * 60}")

        return output_file

    def _collect_sample(self, sample_num: int) -> Dict:
        """Collect a single sample of all metrics."""
        return {
            'timestamp': datetime.now().isoformat(),
            'sample_num': sample_num,
            'fps_config': self.fps_config,
            'cpu_percent': get_cpu_usage(self.node_process),
            'temperature_c': get_temperature(),
            'camera_hz': self.camera_hz_monitor.get_hz(),
            'detection_hz': self.detection_hz_monitor.get_hz(),
            'notes': ''
        }

    def _save_csv(self, output_file: Path) -> None:
        """Save collected samples to CSV."""
        if not self.samples:
            print("Warning: No samples to save!")
            return

        fieldnames = [
            'timestamp', 'sample_num', 'fps_config', 'cpu_percent',
            'temperature_c', 'camera_hz', 'detection_hz', 'notes'
        ]

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.samples)


def main():
    parser = argparse.ArgumentParser(
        description='Camera FPS Experiment - Data Collection Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 camera_fps_monitor.py --fps-config 5fps
  python3 camera_fps_monitor.py --fps-config 10fps --duration 300
  python3 camera_fps_monitor.py --fps-config 15fps --output-dir ./my_results
        """
    )
    parser.add_argument(
        '--fps-config', required=True,
        help='Label for current FPS configuration (e.g., "5fps", "10fps")'
    )
    parser.add_argument(
        '--duration', type=int, default=300,
        help='Test duration in seconds (default: 300)'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='gesturebot_ws/src/ros2_mediapipe/benchmark/results',
        help='Output directory for CSV files'
    )
    parser.add_argument(
        '--sample-interval', type=int, default=5,
        help='Sample interval in seconds (default: 5)'
    )

    args = parser.parse_args()

    # Resolve output directory relative to home if needed
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.home() / 'GestureBot' / args.output_dir

    monitor = CameraFPSMonitor(
        fps_config=args.fps_config,
        duration=args.duration,
        output_dir=output_dir,
        sample_interval=args.sample_interval
    )

    monitor.run()


if __name__ == '__main__':
    main()

