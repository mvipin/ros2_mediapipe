#!/usr/bin/env python3
"""
MediaPipe Callback Mixin

Simplified callback handling for MediaPipe results without performance tracking.
"""

from abc import abstractmethod
from typing import Callable


class MediaPipeCallbackMixin:
    """
    Simplified callback mixin for MediaPipe result processing.

    Provides essential callback chain: create_callback() → _process_callback_results() → publish_results()

    Note: MediaPipe LIVE_STREAM mode uses a single callback thread internally,
    so no additional synchronization is needed for callback serialization.
    The processing_lock in base_node.py handles frame dropping during inference.
    """

    def __init__(self):
        self._base_node_ref = None
        # Track which timestamps hold the processing lock (released in callback)
        self._lock_holders: set = set()

    def _set_base_node_reference(self, node):
        """Set reference to base node for callback publishing."""
        self._base_node_ref = node

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
        if self._base_node_ref and timestamp_ms in self._lock_holders:
            self._lock_holders.discard(timestamp_ms)
            self._base_node_ref.processing_lock.release()
    
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
