#!/usr/bin/env python3
"""
MediaPipe Callback Mixin

Simplified callback handling for MediaPipe results without performance tracking.
Uses LockLifecycleManager from core module for thread-safe lock management.
"""

from abc import abstractmethod
from typing import Callable, Optional

from ..core import LockLifecycleManager


class MediaPipeCallbackMixin:
    """
    Simplified callback mixin for MediaPipe result processing.

    Provides essential callback chain: create_callback() → _process_callback_results() → publish_results()

    Note: MediaPipe LIVE_STREAM mode uses a single callback thread internally,
    so no additional synchronization is needed for callback serialization.
    The LockLifecycleManager handles frame dropping during inference.
    """

    def __init__(self):
        self._base_node_ref = None
        self._lock_manager: Optional[LockLifecycleManager] = None

    def _set_base_node_reference(self, node):
        """Set reference to base node for callback publishing and lock management."""
        self._base_node_ref = node
        # Initialize lock manager with node's processing lock
        if hasattr(node, 'processing_lock'):
            self._lock_manager = LockLifecycleManager(node.processing_lock)

    def create_callback(self, result_type: str) -> Callable:
        """
        Create callback function for MediaPipe results.

        This is the entry point for MediaPipe controller callbacks.
        The callback releases the processing_lock that was acquired in _process_frame_async(),
        ensuring the lock covers the full inference cycle.
        """
        def callback(result, output_image, timestamp_ms):
            try:
                timestamp_seconds = timestamp_ms / 1000.0
                processed_results = self._process_callback_results(
                    result, output_image, timestamp_ms, result_type
                )

                if processed_results and self._base_node_ref:
                    self._base_node_ref.publish_results(processed_results, timestamp_seconds)

            except Exception as e:
                if self._base_node_ref:
                    self._base_node_ref.get_logger().error(f"Error in callback: {e}")
            finally:
                # Release the processing lock (held since _process_frame_async acquired it)
                self._release_processing_lock(timestamp_ms)

        return callback

    def _release_processing_lock(self, timestamp_ms: int) -> None:
        """
        Release the processing lock for a given timestamp.

        Called from callback after inference completes to allow next frame to be processed.
        """
        if self._lock_manager:
            self._lock_manager.release_for_timestamp(timestamp_ms)

    def _acquire_processing_lock(self, timestamp_ms: int) -> bool:
        """
        Acquire the processing lock for a given timestamp.

        Called from _process_frame_async to acquire lock with timestamp tracking.

        Returns:
            True if lock was acquired, False if busy (frame should be dropped)
        """
        if self._lock_manager:
            return self._lock_manager.acquire_for_timestamp(timestamp_ms)
        return False

    @abstractmethod
    def _process_callback_results(self, result, output_image, timestamp_ms, result_type):
        """
        Process MediaPipe callback results. Must be implemented by subclass.

        Args:
            result: MediaPipe detection result
            output_image: Annotated output image from MediaPipe
            timestamp_ms: Timestamp in milliseconds
            result_type: Type of result ('object', 'gesture', 'pose')

        Returns:
            Processed results ready for publishing
        """
        pass
