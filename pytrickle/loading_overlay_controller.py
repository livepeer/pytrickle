"""
Loading Overlay Controller for TrickleClient.

Manages loading overlay state, timing, and frame rendering.
"""

import time
import logging
from typing import Optional

from .frame_processor import FrameProcessor
from .frames import VideoFrame
from .loading_config import LoadingConfig, LoadingMode
from .utils.loading_overlay import build_loading_overlay_frame

logger = logging.getLogger(__name__)


class LoadingOverlayController:
    """Helper to coordinate loading overlay state and rendering within the client."""

    def __init__(self, frame_processor: FrameProcessor, loading_config: Optional[LoadingConfig]):
        self.frame_processor = frame_processor
        self.loading_config = loading_config if loading_config is not None else LoadingConfig()
        self._last_video_frame_time = time.time()
        self._loading_active = False
        self._is_manual_loading = False  # Track if loading was set manually
        self._frame_counter = 0

    def reset(self) -> None:
        """Reset timing/state for a new stream."""
        self._last_video_frame_time = time.time()
        self._loading_active = False
        self._is_manual_loading = False
        self._frame_counter = 0

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

    def apply_loading_config(self, config: Optional[LoadingConfig]) -> None:
        """Push a new loading configuration into the controller."""
        self.loading_config = config if config is not None else LoadingConfig()
        self._frame_counter = 0
        if (
            not self.loading_config.enabled
            or self.loading_config.mode == LoadingMode.PASSTHROUGH
        ):
            self._disable_overlay()

    def set_manual_loading(self, active: bool) -> None:
        """Toggle manual loading overlay state."""
        self._loading_active = bool(active)
        self._is_manual_loading = bool(active)
        if active:
            # When manually enabling, update timestamp
            self._last_video_frame_time = time.time()
        else:
            # When manually disabling, reset timestamp to give grace period before auto can re-enable
            self._last_video_frame_time = time.time()
            self._frame_counter = 0

    def _update_loading_state(self, received_frame_from_processor: bool) -> None:
        """Update loading state based on whether the processor returned a frame."""
        if not self.loading_config or not self.loading_config.enabled:
            self._disable_overlay()
            return

        if self.loading_config.mode != LoadingMode.OVERLAY:
            self._disable_overlay()
            return

        now = time.time()
        if received_frame_from_processor:
            self._last_video_frame_time = now
            # Only auto-disable if loading was set automatically (not manually)
            if self._loading_active and not self._is_manual_loading:
                self._loading_active = False
        else:
            # Only auto-enable if not already manually enabled
            if not self._is_manual_loading:
                timeout = self.loading_config.auto_timeout_seconds
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
            The frame that should be forwarded downstream (overlay or passthrough).
        """
        fallback_frame = processed_frame if processed_frame is not None else original_frame

        if (
            not self.loading_config
            or not self.loading_config.enabled
            or self.loading_config.is_passthrough_mode()
        ):
            return fallback_frame

        if not self._loading_active:
            return fallback_frame

        self._frame_counter += 1
        return build_loading_overlay_frame(
            original_frame=original_frame,
            message=self.loading_config.message,
            frame_counter=self._frame_counter,
            progress=self.loading_config.progress,
        )
