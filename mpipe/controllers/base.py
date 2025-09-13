#!/usr/bin/env python3
"""
Abstract MediaPipe Controller Interface

Provides the base controller interface for all MediaPipe integrations.
Controllers handle MediaPipe-specific lifecycle, submission, and cleanup.
"""

from abc import ABC, abstractmethod
import mediapipe as mp


class MediaPipeController(ABC):
    """Abstract controller for MediaPipe pipelines."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True when the underlying pipeline is initialized and ready."""
        raise NotImplementedError

    @abstractmethod
    def detect_async(self, mp_image: mp.Image, timestamp_ms: int) -> None:
        """Submit an image to the pipeline asynchronously."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release underlying resources."""
        raise NotImplementedError
