"""
Frame Processing Types and Enums

Defines type-safe enums and constants for frame processing operations.
"""

from enum import Enum
from typing import Union, Optional
from .frames import VideoFrame, AudioFrame

class FrameProcessingResult(Enum):
    """
    Enum for frame processing results to replace string sentinels.
    
    This provides type safety and prevents string-based sentinel confusion.
    """
    SHUTDOWN = "shutdown"           # Queue received shutdown sentinel
    FRAME_SKIPPED = "frame_skipped" # Frame was intentionally skipped
    TIMEOUT = "timeout"             # Operation timed out
    ERROR = "error"                 # Processing error occurred


# Type alias for frame processing return values
FrameOrResult = Union[VideoFrame, AudioFrame, FrameProcessingResult]

def is_processing_result(value: any) -> bool:
    """
    Check if a value is a FrameProcessingResult enum.
    
    Args:
        value: Value to check
        
    Returns:
        True if value is a FrameProcessingResult, False otherwise
    """
    return isinstance(value, FrameProcessingResult)


def is_frame(value: any) -> bool:
    """
    Check if a value is a frame (VideoFrame or AudioFrame).
    
    Args:
        value: Value to check
        
    Returns:
        True if value is a frame, False otherwise
    """
    return isinstance(value, (VideoFrame, AudioFrame))
