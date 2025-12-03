#!/usr/bin/env python3
"""
Resource Monitor for Benchmark Profiling

Background thread that samples CPU, memory, and temperature at configurable intervals.
Designed for minimal overhead during benchmark runs.
"""

import threading
import time
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import List, Optional

import psutil


@dataclass
class ResourceSamples:
    """Container for resource usage samples."""
    cpu_percent: List[float] = field(default_factory=list)
    memory_mb: List[float] = field(default_factory=list)
    temperature_c: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)


class ResourceMonitor:
    """
    Background resource monitor that samples CPU, memory, and temperature.
    
    Usage:
        monitor = ResourceMonitor(interval_ms=100)
        monitor.start()
        # ... run benchmark ...
        monitor.stop()
        stats = monitor.get_statistics()
    """
    
    def __init__(self, interval_ms: int = 100):
        """
        Initialize resource monitor.
        
        Args:
            interval_ms: Sampling interval in milliseconds (default: 100ms)
        """
        self.interval = interval_ms / 1000.0
        self.samples = ResourceSamples()
        self.process = psutil.Process()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0
    
    def start(self) -> None:
        """Start background sampling thread."""
        if self._running:
            return
        
        self._running = True
        self._start_time = time.time()
        self.samples = ResourceSamples()  # Reset samples
        
        # Prime CPU measurement (first call always returns 0)
        self.process.cpu_percent()
        
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop background sampling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
    
    def _sample_loop(self) -> None:
        """Background loop that collects resource samples."""
        while self._running:
            try:
                self.samples.cpu_percent.append(self.process.cpu_percent())
                self.samples.memory_mb.append(self.process.memory_info().rss / 1024 / 1024)
                self.samples.temperature_c.append(self._read_temperature())
                self.samples.timestamps.append(time.time() - self._start_time)
            except Exception:
                pass  # Ignore sampling errors
            time.sleep(self.interval)
    
    def _read_temperature(self) -> float:
        """Read CPU temperature from thermal zone (Raspberry Pi specific)."""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return float(f.read().strip()) / 1000.0
        except (FileNotFoundError, ValueError, PermissionError):
            return 0.0
    
    def get_statistics(self) -> dict:
        """
        Calculate aggregate statistics from collected samples.
        
        Returns:
            Dictionary with mean, max, p95, p99 for each resource metric
        """
        def calc_stats(samples: List[float], prefix: str) -> dict:
            if not samples:
                return {
                    f'{prefix}_mean': 0.0,
                    f'{prefix}_max': 0.0,
                    f'{prefix}_p95': 0.0,
                    f'{prefix}_p99': 0.0,
                }
            
            sorted_samples = sorted(samples)
            n = len(sorted_samples)
            p95_idx = int(n * 0.95)
            p99_idx = int(n * 0.99)
            
            return {
                f'{prefix}_mean': mean(samples),
                f'{prefix}_max': max(samples),
                f'{prefix}_p95': sorted_samples[min(p95_idx, n - 1)],
                f'{prefix}_p99': sorted_samples[min(p99_idx, n - 1)],
            }
        
        stats = {
            'num_samples': len(self.samples.cpu_percent),
            'duration_s': self.samples.timestamps[-1] if self.samples.timestamps else 0.0,
        }
        
        stats.update(calc_stats(self.samples.cpu_percent, 'cpu'))
        stats.update(calc_stats(self.samples.memory_mb, 'memory_mb'))
        stats.update(calc_stats(self.samples.temperature_c, 'temp_c'))
        
        return stats
    
    def reset(self) -> None:
        """Reset collected samples."""
        self.samples = ResourceSamples()
        self._start_time = time.time()

