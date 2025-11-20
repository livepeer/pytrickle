"""
Preview video utilities for PyTrickle video frames.

Inspired by the ai-runner feature (Credit to @victorges):
https://github.com/livepeer/ai-runner/blob/main/runner/app/live/process/loading_overlay.py
"""

import math
from typing import Optional

import av
import numpy as np
from pytrickle.frames import VideoFrame

__all__ = [
    "build_preview_video_frame",
]


_cv2_module = None


def _get_cv2():
    """Retrieve OpenCV and verify it's properly installed."""
    global _cv2_module
    if _cv2_module is None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised in environments without cv2
            raise RuntimeError(
                "Preview video support requires the 'opencv-python' package. "
                "Install it to enable preview video rendering."
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


def _create_preview_frame_numpy(
    width: int,
    height: int,
    message: str,
    frame_counter: int,
    progress: Optional[float],
) -> np.ndarray:
    """
    Internal helper to create a preview video frame as numpy array.

    Creates an RGB preview video with animated progress bar using OpenCV.
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


def build_preview_video_frame(
    original_frame: VideoFrame,
    message: str = "Loading...",
    frame_counter: int = 0,
    progress: Optional[float] = None,
) -> VideoFrame:
    """
    Create a preview video VideoFrame with timing preserved from original frame.

    Replaces the video content with an animated preview video while preserving
    timing information (timestamp, time_base) from the original frame. This is
    useful for frame processors that need to show preview/loading state during warmup or
    processing delays.
    """
    from ..frames import VideoFrame

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

    overlay_np = _create_preview_frame_numpy(
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
