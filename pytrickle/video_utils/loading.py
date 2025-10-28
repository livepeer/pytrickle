"""
Loading utilities for PyTrickle - Video loading overlay generation.

This module provides utilities for creating loading overlay frames for video processing.

Loading overlay inspired by ai-runner feature (Credit to @victorges):
https://github.com/livepeer/ai-runner/blob/main/runner/app/live/process/loading_overlay.py
"""

import math
import cv2
import numpy as np
from typing import Optional


def create_loading_frame(
    width: int, 
    height: int, 
    message: str = "Loading...",
    frame_counter: int = 0,
    progress: Optional[float] = None,
    color_format: str = "RGB"
) -> np.ndarray:
    """
    Create a loading overlay frame with animated progress bar.
    
    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        message: Loading message to display
        frame_counter: Current frame number for animations
        progress: Progress value 0.0-1.0, or None for animated progress
        color_format: Color format - "BGR" (OpenCV default) or "RGB"
    
    Returns:
        np.ndarray: Loading overlay frame in specified color format
    
    Raises:
        ValueError: If width/height are non-positive or progress is out of range
    """
    # Input validation
    if width <= 0 or height <= 0:
        raise ValueError(f"Width and height must be positive, got {width}x{height}")
    if progress is not None and not (0.0 <= progress <= 1.0):
        raise ValueError(f"Progress must be between 0.0 and 1.0, got {progress}")
    if color_format not in ["BGR", "RGB"]:
        raise ValueError(f"Color format must be 'BGR' or 'RGB', got '{color_format}'")
    # Create dark background
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Set colors based on format (BGR vs RGB)
    if color_format == "BGR":
        dark_bg = (30, 30, 30)      # BGR: Dark gray
        overlay_bg = (20, 20, 20)   # BGR: Darker gray
        border_color = (60, 60, 60) # BGR: Medium gray
        text_color = (200, 200, 200) # BGR: Light gray
        progress_color = (255, 150, 100) # BGR: Blue progress
    else:  # RGB
        dark_bg = (30, 30, 30)      # RGB: Dark gray
        overlay_bg = (20, 20, 20)   # RGB: Darker gray
        border_color = (60, 60, 60) # RGB: Medium gray
        text_color = (200, 200, 200) # RGB: Light gray
        progress_color = (100, 150, 255) # RGB: Blue progress
    
    frame[:] = dark_bg
    
    # Calculate center position
    center_x = width // 2
    center_y = height // 2
    
    # Add overlay panel
    overlay_width = min(400, width - 40)
    overlay_height = min(200, height - 40)
    overlay_x = center_x - overlay_width // 2
    overlay_y = center_y - overlay_height // 2
    
    # Create overlay background
    frame[overlay_y:overlay_y + overlay_height, overlay_x:overlay_x + overlay_width] = overlay_bg
    
    # Add border
    cv2.rectangle(frame, (overlay_x, overlay_y), 
                 (overlay_x + overlay_width, overlay_y + overlay_height), 
                 border_color, 2)
    
    # Add loading message in center
    if message:
        message_size = 1.2
        message_thickness = 2
        message_y = center_y
        
        (msg_width, _), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 
                                          message_size, message_thickness)
        message_x = center_x - msg_width // 2
        
        cv2.putText(frame, message, (message_x, message_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, message_size, text_color, message_thickness)
    
    # Add progress bar
    bar_width = overlay_width - 60
    bar_height = 6
    bar_x = overlay_x + 30
    bar_y = center_y + 40
    
    # Progress bar background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                 (40, 40, 40), -1)
    
    # Use controllable progress or animated progress
    if progress is not None and progress > 0:
        current_progress = min(1.0, max(0.0, progress))
    else:
        # Animated progress (oscillating)
        current_progress = (math.sin(frame_counter * 0.1) + 1) * 0.5
    
    progress_width = int(bar_width * current_progress)
    
    if progress_width > 0:
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + progress_width, bar_y + bar_height), 
                     progress_color, -1)  # Blue progress
    
    return frame