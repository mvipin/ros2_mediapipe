#!/usr/bin/env python3
"""
Processing Configuration for MediaPipe Integration

Simplified configuration class for MediaPipe processing.
"""

from dataclasses import dataclass
from enum import Enum


class LogLevel(Enum):
    """Logging levels for MediaPipe nodes."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass
class VisualizationConfig:
    """Configuration for visual annotations and overlays."""
    # Colors (BGR format for OpenCV)
    landmark_color: tuple = (0, 255, 0)      # Green landmarks
    skeleton_color: tuple = (0, 255, 0)      # Green skeleton lines
    text_color: tuple = (0, 255, 0)          # Green text
    info_text_color: tuple = (255, 255, 255) # White info text

    # Font settings
    font_face: int = 0  # cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 1.0
    info_font_scale: float = 0.6
    font_thickness: int = 2

    # Drawing parameters
    landmark_radius: int = 3
    skeleton_thickness: int = 2

    # Text positioning
    main_text_x: int = 10
    main_text_y: int = 30
    info_text_x: int = 10
    info_text_y_offset: int = 20  # Offset from bottom


@dataclass
class PoseClassificationConfig:
    """Configuration for pose classification thresholds."""
    horizontal_tolerance: float = 0.1  # Tolerance for horizontal alignment in pointing poses
    enable_pose_classification: bool = True
    pose_stability_frames: int = 3  # Frames to confirm pose change


@dataclass
class TopicConfig:
    """Configuration for ROS topic names."""
    # Input topics
    camera_topic: str = "/camera/image_raw"

    # Output topics (will be prefixed with node-specific namespace)
    pose_topic: str = "/vision/pose"
    gesture_topic: str = "/vision/gestures"
    objects_topic: str = "/vision/objects"

    # Annotated image topics
    pose_annotated_topic: str = "/vision/pose/annotated"
    gesture_annotated_topic: str = "/vision/gestures/annotated"
    objects_annotated_topic: str = "/vision/objects/annotated"


@dataclass
class ProcessingConfig:
    """
    Configuration for MediaPipe processing nodes.

    Attributes:
        enabled: Whether processing is enabled
        frame_skip: Number of frames to skip between processing (0 = process every frame)
        confidence_threshold: Minimum confidence threshold for detections (0.0-1.0)
        max_results: Maximum number of results to return per frame
        log_level: Logging level for the node
        debug_mode: Enable debug mode (deprecated, use log_level=DEBUG instead)
        visualization: Visual annotation configuration
        pose_classification: Pose classification configuration
        topics: Topic name configuration
    """
    enabled: bool = True
    frame_skip: int = 1
    confidence_threshold: float = 0.5
    max_results: int = 5
    log_level: LogLevel = LogLevel.INFO
    debug_mode: bool = False  # Deprecated, kept for backward compatibility

    # Sub-configurations
    visualization: VisualizationConfig = None
    pose_classification: PoseClassificationConfig = None
    topics: TopicConfig = None

    def __post_init__(self):
        """Handle backward compatibility and validation."""
        # Initialize sub-configurations if not provided
        if self.visualization is None:
            self.visualization = VisualizationConfig()
        if self.pose_classification is None:
            self.pose_classification = PoseClassificationConfig()
        if self.topics is None:
            self.topics = TopicConfig()

        # Validate main parameters
        if self.frame_skip < 0:
            raise ValueError("frame_skip must be non-negative")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if self.max_results <= 0:
            raise ValueError("max_results must be positive")

        # Handle backward compatibility for debug_mode
        if self.debug_mode and self.log_level == LogLevel.INFO:
            self.log_level = LogLevel.DEBUG

        # Validate pose classification config
        if not 0.0 <= self.pose_classification.horizontal_tolerance <= 1.0:
            raise ValueError("pose_classification.horizontal_tolerance must be between 0.0 and 1.0")

        # Validate visualization config
        if self.visualization.landmark_radius < 1:
            raise ValueError("visualization.landmark_radius must be positive")
        if self.visualization.font_scale <= 0:
            raise ValueError("visualization.font_scale must be positive")

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ProcessingConfig':
        """Create ProcessingConfig from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        """Convert ProcessingConfig to dictionary."""
        return {
            'enabled': self.enabled,
            'frame_skip': self.frame_skip,
            'confidence_threshold': self.confidence_threshold,
            'max_results': self.max_results,
            'log_level': self.log_level.value,
            'debug_mode': self.debug_mode
        }
