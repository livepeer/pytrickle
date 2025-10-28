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
    progress: Optional[float] = None
) -> np.ndarray:
    """
    Create a loading overlay frame with animated progress bar.
    
    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        message: Loading message to display
        frame_counter: Current frame number for animations
        progress: Progress value 0.0-1.0, or None for animated progress
    
    Returns:
        np.ndarray: Loading overlay frame as BGR image
    """
    # Create dark background
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)  # Dark gray background
    
    # Calculate center position
    center_x = width // 2
    center_y = height // 2
    
    # Add overlay panel
    overlay_width = min(400, width - 40)
    overlay_height = min(200, height - 40)
    overlay_x = center_x - overlay_width // 2
    overlay_y = center_y - overlay_height // 2
    
    # Create overlay background
    frame[overlay_y:overlay_y + overlay_height, overlay_x:overlay_x + overlay_width] = (20, 20, 20)
    
    # Add border
    cv2.rectangle(frame, (overlay_x, overlay_y), 
                 (overlay_x + overlay_width, overlay_y + overlay_height), 
                 (60, 60, 60), 2)
    
    # Add loading message in center
    if message:
        message_size = 1.2
        message_thickness = 2
        message_y = center_y
        
        (msg_width, _), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 
                                          message_size, message_thickness)
        message_x = center_x - msg_width // 2
        
        cv2.putText(frame, message, (message_x, message_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, message_size, (200, 200, 200), message_thickness)
    
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
                     (100, 150, 255), -1)  # Blue progress
    
    return frame