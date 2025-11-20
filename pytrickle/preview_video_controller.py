"""
Preview Video Controller for TrickleClient.

Manages preview video state, timing, and frame rendering.
"""

import time
import logging
from typing import Optional

from .frame_processor import FrameProcessor
from .frames import VideoFrame
from .preview_video_config import PreviewVideoConfig, PreviewVideoMode
from .utils.preview_video import build_preview_video_frame

logger = logging.getLogger(__name__)


class PreviewVideoController:
    """Helper to coordinate preview video state and rendering within the client."""

    def __init__(self, frame_processor: FrameProcessor, preview_video_config: Optional[PreviewVideoConfig]):
        self.frame_processor = frame_processor
        self.preview_video_config = preview_video_config if preview_video_config is not None else PreviewVideoConfig()
        self._last_video_frame_time = time.time()
        self._preview_active = False
        self._is_manual_preview = False  # Track if preview was set manually
        self._frame_counter = 0

    def reset(self) -> None:
        """Reset timing/state for a new stream."""
        self._last_video_frame_time = time.time()
        self._preview_active = False
        self._is_manual_preview = False
        self._frame_counter = 0

    def update_and_apply(
        self,
        original_frame: VideoFrame,
        processed_frame: Optional[VideoFrame],
    ) -> Optional[VideoFrame]:
        """
        Update preview state and apply view if needed.
        
        Args:
            original_frame: The original frame from ingress
            processed_frame: The frame returned by the processor (may be None)
            
        Returns:
            The final frame to send to egress (with preview view if applicable)
        """
        received_frame = processed_frame is not None
        self._update_preview_state(received_frame)
        return self._apply_preview_video(original_frame, processed_frame)

    def apply_preview_config(self, config: Optional[PreviewVideoConfig]) -> None:
        """Push a new preview configuration into the controller."""
        self.preview_video_config = config if config is not None else PreviewVideoConfig()
        self._frame_counter = 0
        if (
            not self.preview_video_config.enabled
            or self.preview_video_config.mode == PreviewVideoMode.Passthrough
        ):
            self._disable_preview()

    def set_manual_preview(self, active: bool) -> None:
        """Toggle manual preview video state."""
        self._preview_active = bool(active)
        self._is_manual_preview = bool(active)
        self._last_video_frame_time = time.time()
        if not active:
            self._frame_counter = 0

    def _update_preview_state(self, received_frame_from_processor: bool) -> None:
        """Update preview state based on whether the processor returned a frame."""
        if not self.preview_video_config or not self.preview_video_config.enabled:
            self._disable_preview()
            return

        if self.preview_video_config.mode != PreviewVideoMode.ProgressBar:
            self._disable_preview()
            return

        now = time.time()
        if received_frame_from_processor:
            self._last_video_frame_time = now
            # Only auto-disable if preview was set automatically (not manually)
            if self._preview_active and not self._is_manual_preview:
                self._preview_active = False
        else:
            # Only auto-enable if not already manually enabled
            if not self._is_manual_preview:
                timeout = self.preview_video_config.auto_timeout_seconds
                if (
                    timeout is not None
                    and timeout >= 0.0
                    and (now - self._last_video_frame_time) >= timeout
                ):
                    if not self._preview_active:
                        self._preview_active = True

    def _disable_preview(self) -> None:
        """Disable the preview video if it's currently active."""
        self._preview_active = False
        self._is_manual_preview = False
        self._frame_counter = 0

    def _apply_preview_video(
        self,
        original_frame: VideoFrame,
        processed_frame: Optional[VideoFrame],
    ) -> Optional[VideoFrame]:
        """
        Apply preview video to the frame if needed.
        
        Args:
            original_frame: Frame received from ingress (used for timing).
            processed_frame: Frame returned by the user handler (may be None).

        Returns:
            The frame that should be forwarded downstream (preview or passthrough).
        """
        fallback_frame = processed_frame if processed_frame is not None else original_frame

        if (
            not self.preview_video_config
            or not self.preview_video_config.enabled
            or self.preview_video_config.is_passthrough_mode()
        ):
            return fallback_frame

        if not self._preview_active:
            return fallback_frame

        self._frame_counter += 1
        return build_preview_video_frame(
            original_frame=original_frame,
            message=self.preview_video_config.message,
            frame_counter=self._frame_counter,
            progress=self.preview_video_config.progress,
        )
