"""
Frame data structures for trickle streaming.

Defines VideoFrame, AudioFrame, and their output counterparts for handling
media data in the trickle streaming pipeline.
"""

import time
import torch
import numpy as np
import av
from typing import Optional, Dict, Any, Union, List
from fractions import Fraction
from abc import ABC

# Default dimensions for video frames  
DEFAULT_WIDTH = 704
DEFAULT_HEIGHT = 384

class SideData:
    """Base class for side data, needed to keep it consistent with av frame side_data"""
    skipped: bool = True
    input: Optional[Union[torch.Tensor, np.ndarray]] = None

class InputFrame(ABC):
    """Base class for input frames."""
    
    timestamp: int
    time_base: Fraction
    log_timestamps: Dict[str, float]
    side_data: SideData

    @classmethod
    def from_av_video(cls, tensor: torch.Tensor, timestamp: int, time_base: Fraction):
        return VideoFrame(tensor, timestamp, time_base)

    @classmethod
    def from_av_audio(cls, frame: av.AudioFrame):
        return AudioFrame(frame)

class VideoFrame(InputFrame):
    """Represents a video frame with tensor data and timing information."""
    
    tensor: torch.Tensor

    def __init__(self, tensor: torch.Tensor, timestamp: int, time_base: Fraction, log_timestamps: Dict[str, float] = None):
        self.tensor = tensor
        self.timestamp = timestamp
        self.time_base = time_base
        self.log_timestamps = log_timestamps or {}
        self.side_data = SideData()
    
    @classmethod
    def from_av_video(cls, tensor: torch.Tensor, timestamp: int, time_base: Fraction) -> 'VideoFrame':
        """Create VideoFrame from av video data."""
        return cls(tensor, timestamp, time_base)
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, timestamp: int = 0) -> 'VideoFrame':
        """Create VideoFrame from tensor with timestamp."""
        return cls(tensor, timestamp, Fraction(1, 1))
    
    def replace_tensor(self, new_tensor: torch.Tensor) -> 'VideoFrame':
        """Create a new VideoFrame with a different tensor."""
        new_frame = VideoFrame(new_tensor, self.timestamp, self.time_base, self.log_timestamps.copy())
        new_frame.side_data = self.side_data
        return new_frame

class AudioFrame(InputFrame):
    """Represents an audio frame with sample data and timing information."""
    
    samples: np.ndarray
    format: str  # av.audio.format.AudioFormat
    layout: str  # av.audio.layout.AudioLayout
    rate: int
    nb_samples: int

    def __init__(self, frame):
        if frame.pts is None:
            raise ValueError("Audio frame has no timestamp")
        self.samples = frame.to_ndarray()
        self.nb_samples = frame.samples
        self.format = frame.format.name
        self.rate = frame.sample_rate
        self.layout = frame.layout.name
        self.timestamp = frame.pts
        self.time_base = frame.time_base
        self.log_timestamps = {}
        self.side_data = SideData()
    
    @classmethod
    def from_av_audio(cls, av_frame) -> 'AudioFrame':
        """Create AudioFrame from av audio frame."""
        return cls(av_frame)

class OutputFrame(ABC):
    """Base class for output frames."""
    pass

class VideoOutput(OutputFrame):
    """Represents processed video output."""
    
    frame: VideoFrame
    request_id: str

    def __init__(self, frame: VideoFrame, request_id: str = ''):
        self.frame = frame
        self.request_id = request_id

    def replace_tensor(self, tensor: torch.Tensor):
        new_frame = self.frame.replace_tensor(tensor)
        return VideoOutput(new_frame, self.request_id)

    @property
    def tensor(self):
        return self.frame.tensor

    @property
    def timestamp(self):
        return self.frame.timestamp

    @property
    def time_base(self):
        return self.frame.time_base

    @property
    def log_timestamps(self):
        return self.frame.log_timestamps

class AudioOutput(OutputFrame):
    """Represents processed audio output."""
    
    frames: List[AudioFrame]
    request_id: str
    
    def __init__(self, frames: List[AudioFrame], request_id: str = ''):
        self.frames = frames
        self.request_id = request_id 