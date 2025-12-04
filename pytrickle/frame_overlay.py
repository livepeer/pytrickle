"""
Frame Overlay for PyTrickle.

Manages loading overlay state, timing, frame rendering, and configuration.

Inspired by the ai-runner feature (Credit to @victorges):
https://github.com/livepeer/ai-runner/blob/main/runner/app/live/process/loading_overlay.py
"""

import math
import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import av
import numpy as np

from .frames import VideoFrame

__all__ = [
    "OverlayMode",
    "OverlayConfig",
    "OverlayController",
    "build_frame_overlay",
]

logger = logging.getLogger(__name__)


_cv2_module = None


def _get_cv2():
    """Retrieve OpenCV and verify it's properly installed."""
    global _cv2_module
    if _cv2_module is None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised in environments without cv2
            raise RuntimeError(
                "Frame overlay support requires the 'opencv-python' package. "
                "Install it to enable frame overlay rendering."
            ) from exc
        
        # Verify OpenCV is properly built and functional
        try:
            build_info = cv2.getBuildInformation()
            if not build_info or not isinstance(build_info, str):
                raise RuntimeError("OpenCV build information is invalid or empty.")
        except Exception as exc:
            raise RuntimeError(
                "OpenCV is installed but not functioning correctly. "
                "Please reinstall opencv-python."
            ) from exc
        
        _cv2_module = cv2
    return _cv2_module


def _create_overlay_frame_numpy(
    width: int,
    height: int,
    message: str,
    frame_counter: int,
    progress: Optional[float],
) -> np.ndarray:
    """
    Internal helper to create a frame overlay as numpy array.

    Creates an RGB overlay with animated progress bar using OpenCV.
    This is an implementation detail and should not be used directly.
    """
    cv2 = _get_cv2()
    if width <= 0 or height <= 0:
        raise ValueError(f"Width and height must be positive, got {width}x{height}")
    if progress is not None and not (0.0 <= progress <= 1.0):
        raise ValueError(f"Progress must be between 0.0 and 1.0, got {progress}")

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    dark_bg = (30, 30, 30)
    overlay_bg = (20, 20, 20)
    border_color = (60, 60, 60)
    text_color = (200, 200, 200)
    progress_color = (100, 150, 255)

    frame[:] = dark_bg

    center_x = width // 2
    center_y = height // 2

    overlay_width = min(400, width - 40)
    overlay_height = min(200, height - 40)
    overlay_x = center_x - overlay_width // 2
    overlay_y = center_y - overlay_height // 2

    frame[overlay_y:overlay_y + overlay_height, overlay_x:overlay_x + overlay_width] = overlay_bg

    cv2.rectangle(
        frame,
        (overlay_x, overlay_y),
        (overlay_x + overlay_width, overlay_y + overlay_height),
        border_color,
        2,
    )

    if message:
        message_size = 1.2
        message_thickness = 2
        message_y = center_y

        (msg_width, _), _ = cv2.getTextSize(
            message,
            cv2.FONT_HERSHEY_SIMPLEX,
            message_size,
            message_thickness,
        )
        message_x = center_x - msg_width // 2

        cv2.putText(
            frame,
            message,
            (message_x, message_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            message_size,
            text_color,
            message_thickness,
        )

    bar_width = overlay_width - 60
    bar_height = 6
    bar_x = overlay_x + 30
    bar_y = center_y + 40

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (40, 40, 40), -1)

    if progress is not None and progress > 0:
        current_progress = min(1.0, max(0.0, progress))
    else:
        current_progress = (math.sin(frame_counter * 0.1) + 1) * 0.5

    progress_width = int(bar_width * current_progress)

    if progress_width > 0:
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + progress_width, bar_y + bar_height),
            progress_color,
            -1,
        )

    return frame


def build_frame_overlay(
    original_frame: VideoFrame,
    message: str = "Loading...",
    frame_counter: int = 0,
    progress: Optional[float] = None,
) -> VideoFrame:
    """
    Create a frame overlay VideoFrame with timing preserved from original frame.

    Replaces the video content with an animated overlay while preserving
    timing information (timestamp, time_base) from the original frame. This is
    useful for frame processors that need to show loading state during warmup or
    processing delays.
    """
    tensor = original_frame.tensor
    if tensor.dim() == 4:
        if tensor.shape[1] == 3:
            height, width = tensor.shape[2], tensor.shape[3]
        else:
            height, width = tensor.shape[1], tensor.shape[2]
    elif tensor.dim() == 3:
        if tensor.shape[0] == 3:
            height, width = tensor.shape[1], tensor.shape[2]
        else:
            height, width = tensor.shape[0], tensor.shape[1]
    else:
        raise ValueError(f"Unexpected tensor dimensions: {tensor.shape}")

    overlay_np = _create_overlay_frame_numpy(
        width=width,
        height=height,
        message=message,
        frame_counter=frame_counter,
        progress=progress,
    )

    overlay_av = av.VideoFrame.from_ndarray(overlay_np, format="rgb24")
    overlay_av.pts = original_frame.timestamp
    overlay_av.time_base = original_frame.time_base

    overlay_frame = VideoFrame.from_av_frame_with_timing(overlay_av, original_frame)

    return overlay_frame


class OverlayMode(Enum):
    """Mode for handling video frames during loading."""
    PROGRESSBAR = "progressbar"  # Show loading animation overlay
    PASSTHROUGH = "passthrough"  # Pass through original frames


@dataclass
class OverlayConfig:
    """Configuration for loading behavior."""
    mode: OverlayMode = OverlayMode.PROGRESSBAR  # How to handle video frames during loading
    message: str = "Loading..."              # Loading message to display in overlay
    progress: Optional[float] = None         # Progress value 0.0-1.0 (None = animated)
    enabled: bool = True                     # Whether loading gating is active
    auto_timeout_seconds: Optional[float] = 1.5  # Seconds without output before overlay auto-enables
    frame_count_to_disable: int = 1  # Consecutive valid frames required before overlay auto-disables
    
    def is_overlay_mode(self) -> bool:
        """Return True if this config is set to overlay mode."""
        return self.mode == OverlayMode.PROGRESSBAR

    def is_passthrough_mode(self) -> bool:
        """Return True if this config is set to passthrough mode."""
        return self.mode == OverlayMode.PASSTHROUGH


class OverlayController:
    """Helper to coordinate loading overlay state and rendering within the client."""

    def __init__(self, overlay_config: Optional[OverlayConfig]):
        self.overlay_config = overlay_config if overlay_config is not None else OverlayConfig()
        self._last_video_frame_time = time.time()
        self._loading_active = False
        self._is_manual_loading = False  # Track if loading was set manually
        self._frame_counter = 0
        self._consecutive_received_frames = 0

    def reset(self) -> None:
        """Reset timing/state for a new stream."""
        self._last_video_frame_time = time.time()
        self._loading_active = False
        self._is_manual_loading = False
        self._frame_counter = 0
        self._consecutive_received_frames = 0

    def update_and_apply(
        self,
        original_frame: VideoFrame,
        processed_frame: Optional[VideoFrame],
    ) -> Optional[VideoFrame]:
        """
        Update loading state and apply overlay if needed.
        
        Args:
            original_frame: The original frame from ingress
            processed_frame: The frame returned by the processor (may be None)
            
        Returns:
            The final frame to send to egress (with overlay if applicable)
        """
        received_frame = processed_frame is not None
        self._update_loading_state(received_frame)
        return self._apply_loading_overlay(original_frame, processed_frame)

    def apply_overlay_config(self, config: Optional[OverlayConfig]) -> None:
        """Push a new loading configuration into the controller."""
        self.overlay_config = config if config is not None else OverlayConfig()
        self._frame_counter = 0
        if (
            not self.overlay_config.enabled
            or self.overlay_config.mode == OverlayMode.PASSTHROUGH
        ):
            self._disable_overlay()

    def set_manual_loading(self, active: bool) -> None:
        """Toggle manual loading overlay state."""
        self._loading_active = bool(active)
        self._is_manual_loading = bool(active)
        self._last_video_frame_time = time.time()
        if not active:
            self._frame_counter = 0

    def should_skip_processing(self) -> bool:
        """
        Return True if processing should be skipped entirely.
        
        This happens when manual loading is active. In this case, the client
        should generate an overlay frame immediately without calling the processor.
        """
        if self.overlay_config and self.overlay_config.is_passthrough_mode():
            return False
        return self._is_manual_loading

    def _update_loading_state(self, received_frame_from_processor: bool) -> None:
        """Update loading state based on whether the processor returned a frame."""
        if not self.overlay_config or not self.overlay_config.enabled:
            self._disable_overlay()
            return

        if self.overlay_config.mode != OverlayMode.PROGRESSBAR:
            self._disable_overlay()
            return

        now = time.time()
        if received_frame_from_processor:
            self._last_video_frame_time = now
            # auto-disable if loading was set automatically (not manually enabled) and `frame_count_to_disable` threshold is reached
            self._consecutive_received_frames += 1
            if self._loading_active and not self._is_manual_loading:
                threshold = max(1, self.overlay_config.frame_count_to_disable)
                if self._consecutive_received_frames >= threshold:
                    self._loading_active = False
        else:
            self._consecutive_received_frames = 0
            # Only auto-enable if not already manually enabled
            if not self._is_manual_loading:
                timeout = self.overlay_config.auto_timeout_seconds
                if (
                    timeout is not None
                    and timeout >= 0.0
                    and (now - self._last_video_frame_time) >= timeout
                ):
                    if not self._loading_active:
                        self._loading_active = True

    def _disable_overlay(self) -> None:
        """Disable the overlay if it's currently active."""
        self._loading_active = False
        self._is_manual_loading = False
        self._frame_counter = 0
        self._consecutive_received_frames = 0

    def _apply_loading_overlay(
        self,
        original_frame: VideoFrame,
        processed_frame: Optional[VideoFrame],
    ) -> Optional[VideoFrame]:
        """
        Apply loading overlay to the frame if needed.
        
        Args:
            original_frame: Frame received from ingress (used for overlay timing).
            processed_frame: Frame returned by the user handler (may be None).

        Returns:
            The frame that should be forwarded downstream (overlay, passthrough, or None to skip).
        """
        # If overlay is disabled or in passthrough mode, return processed_frame as-is (may be None)
        if (
            not self.overlay_config
            or not self.overlay_config.enabled
            or self.overlay_config.is_passthrough_mode()
        ):
            return processed_frame
        
        # For overlay modes: use original_frame as fallback only when overlay logic applies
        fallback_frame = processed_frame if processed_frame is not None else original_frame

        if not self._loading_active:
            return fallback_frame

        self._frame_counter += 1
        return build_frame_overlay(
            original_frame=original_frame,
            message=self.overlay_config.message,
            frame_counter=self._frame_counter,
            progress=self.overlay_config.progress,
        )

