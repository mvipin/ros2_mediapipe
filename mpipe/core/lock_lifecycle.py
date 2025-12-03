#!/usr/bin/env python3
"""
Lock Lifecycle Manager

Thread-safe lock holder tracking for MediaPipe async callbacks.
Used to ensure the processing lock covers the full inference cycle,
not just the detect_async() call.

This pattern is shared between:
- ROS 2 nodes (base_node.py + callback_mixin.py)
- Benchmark runner (benchmark_runner.py)
"""

import threading
from typing import Set


class LockLifecycleManager:
    """
    Manages lock acquisition and release across async boundaries.

    In MediaPipe LIVE_STREAM mode, detect_async() returns immediately while
    inference runs asynchronously. The lock must be held until the result
    callback fires to properly implement frame dropping behavior.

    Usage:
        lock_manager = LockLifecycleManager(threading.Lock())

        # In frame processing thread:
        if lock_manager.acquire_for_timestamp(timestamp_ms):
            detector.detect_async(image, timestamp_ms)
            # Lock is NOT released here - held until callback

        # In result callback:
        lock_manager.release_for_timestamp(timestamp_ms)

    Attributes:
        lock: The underlying threading.Lock
    """

    def __init__(self, lock: threading.Lock):
        """
        Initialize with an existing lock.

        Args:
            lock: Threading lock to manage. The same lock instance should be
                  used across all threads that need synchronization.
        """
        self.lock = lock
        self._lock_holders: Set[int] = set()
        self._holders_lock = threading.Lock()  # Protects _lock_holders set

    def acquire_for_timestamp(self, timestamp_ms: int) -> bool:
        """
        Attempt non-blocking lock acquisition for a timestamp.

        If successful, the timestamp is tracked as a lock holder and must be
        released via release_for_timestamp() when processing completes.

        Args:
            timestamp_ms: Unique timestamp identifying this frame/request

        Returns:
            True if lock was acquired, False if busy (frame should be dropped)
        """
        if self.lock.acquire(blocking=False):
            with self._holders_lock:
                self._lock_holders.add(timestamp_ms)
            return True
        return False

    def release_for_timestamp(self, timestamp_ms: int) -> bool:
        """
        Release the lock for a given timestamp.

        Should be called from the async callback when processing completes.
        Only releases if the timestamp is actually a tracked holder.

        Args:
            timestamp_ms: Timestamp that was passed to acquire_for_timestamp()

        Returns:
            True if lock was released, False if timestamp wasn't a holder
        """
        with self._holders_lock:
            if timestamp_ms in self._lock_holders:
                self._lock_holders.discard(timestamp_ms)
                self.lock.release()
                return True
        return False

    def release_on_error(self, timestamp_ms: int) -> bool:
        """
        Release lock on error before callback fires.

        Use this when an exception occurs after acquiring the lock but before
        detect_async() is called, since the callback won't fire to release it.

        Args:
            timestamp_ms: Timestamp that was passed to acquire_for_timestamp()

        Returns:
            True if lock was released, False if timestamp wasn't a holder
        """
        return self.release_for_timestamp(timestamp_ms)

    def is_holding(self, timestamp_ms: int) -> bool:
        """
        Check if a timestamp is currently holding the lock.

        Args:
            timestamp_ms: Timestamp to check

        Returns:
            True if timestamp is a current lock holder
        """
        with self._holders_lock:
            return timestamp_ms in self._lock_holders

    @property
    def pending_count(self) -> int:
        """Return the number of timestamps currently holding the lock."""
        with self._holders_lock:
            return len(self._lock_holders)

