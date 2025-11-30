#!/usr/bin/env python3
"""
MediaPipe Callback Mixin

Simplified callback handling for MediaPipe results without performance tracking.
"""

import threading
from abc import abstractmethod
from typing import Callable


class MediaPipeCallbackMixin:
    """
    Simplified callback mixin for MediaPipe result processing.
    
    Provides essential callback chain: create_callback() → _process_callback_results() → publish_results()
    Removes all performance tracking and buffered logging complexity.
    """
    
    def __init__(self):
        self.callback_lock = threading.Lock()
        self._callback_active = True
        self._base_node_ref = None
    
    def _set_base_node_reference(self, node):
        """Set reference to base node for callback publishing."""
        self._base_node_ref = node
    
    def enable_callback_processing(self):
        """Enable callback processing."""
        with self.callback_lock:
            self._callback_active = True
    
    def disable_callback_processing(self):
        """Disable callback processing."""
        with self.callback_lock:
            self._callback_active = False
    
    def create_callback(self, result_type: str) -> Callable:
        """
        Create callback function for MediaPipe results.

        This is the entry point for MediaPipe controller callbacks.
        """
        def callback(result, output_image, timestamp_ms):
            with self.callback_lock:
                if not self._callback_active:
                    return
                
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
        
        return callback
    
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
