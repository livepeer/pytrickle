"""
Frame data structures and processing utilities for trickle streaming.

Defines VideoFrame, AudioFrame, their output counterparts, frame processing
utilities, and streaming utilities for handling media data in the trickle 
streaming pipeline.
"""

import logging
import torch
import numpy as np
import av
from typing import Optional, Dict, Union, List, Deque
from fractions import Fraction
from collections import deque
from abc import ABC

logger = logging.getLogger(__name__)

# Tensor Conversion Utilities
# ============================

DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512

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

    @classmethod
    def from_av_frame_with_timing(cls, av_frame: 'av.VideoFrame', original_frame: 'VideoFrame') -> 'VideoFrame':
        """Create a VideoFrame from an av.VideoFrame while preserving original timing."""
        frame_np = av_frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
        tensor = torch.from_numpy(frame_np)
        return cls(
            tensor=tensor,
            timestamp=original_frame.timestamp,
            time_base=original_frame.time_base
        )
    
    def to_av_frame(self, tensor: torch.Tensor) -> av.VideoFrame:
        """
        Convert a tensor to av.VideoFrame for use in video pipelines.
        Handles [B, H, W, C] or [H, W, C] formats, float or uint8, and grayscale/RGB.
        """
        # Normalize tensor dimensions to [H, W, C]
        if tensor.dim() == 4:
            if tensor.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got {tensor.shape[0]}")
            tensor = tensor.squeeze(0)
        elif tensor.dim() != 3:
            raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D tensor with shape {tensor.shape}")

        # Validate channel count
        if tensor.shape[2] not in [1, 3, 4]:
            raise ValueError(f"Expected 1, 3, or 4 channels, got {tensor.shape[2]}")

        # Convert to uint8 numpy array in [0, 255] range
        if tensor.dtype in [torch.float32, torch.float64]:
            # Handle normalized [0, 1] range
            tensor_np = (tensor * 255.0 if tensor.max() <= 1.0 else tensor).clamp(0, 255).to(torch.uint8).cpu().numpy()
        else:
            tensor_np = tensor.clamp(0, 255).to(torch.uint8).cpu().numpy()

        # Ensure contiguous memory layout
        if not tensor_np.flags.c_contiguous:
            tensor_np = np.ascontiguousarray(tensor_np)

        # Convert grayscale to RGB if needed
        if tensor_np.shape[2] == 1:
            tensor_np = np.repeat(tensor_np, 3, axis=2)

        return av.VideoFrame.from_ndarray(tensor_np, format="rgb24")

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
    
    @classmethod
    def _from_existing_with_timestamp(cls, existing_frame: 'AudioFrame', new_timestamp: int) -> 'AudioFrame':
        """Create a new AudioFrame with the same properties but a different timestamp."""
        # Create a new frame object with corrected timestamp
        new_frame = cls.__new__(cls)
        new_frame.samples = existing_frame.samples
        new_frame.nb_samples = existing_frame.nb_samples
        new_frame.format = existing_frame.format
        new_frame.rate = existing_frame.rate
        new_frame.layout = existing_frame.layout
        new_frame.timestamp = new_timestamp  # Use the corrected timestamp
        new_frame.time_base = existing_frame.time_base
        new_frame.log_timestamps = existing_frame.log_timestamps.copy()
        new_frame.side_data = existing_frame.side_data
        return new_frame

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, format: str = 's16', layout: str = 'mono', 
                   sample_rate: int = 48000, timestamp: int = 0, time_base = None) -> 'AudioFrame':
        """Create AudioFrame from torch tensor."""
        from fractions import Fraction
        if time_base is None:
            time_base = Fraction(1, sample_rate)
            
        # Convert tensor to numpy samples
        samples = tensor.detach().cpu().numpy()
        
        # Convert to target format
        if format == 's16':
            samples = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)
        elif format == 's32':
            samples = np.clip(samples * 2147483648.0, -2147483648, 2147483647).astype(np.int32)
        else:
            samples = samples.astype(np.float32)
        
        # Handle format layout
        if format.endswith('p'):
            # Planar format - keep as [channels, samples]
            pass
        else:
            # Packed format - convert to interleaved if multi-channel
            if samples.shape[0] > 1:
                samples = samples.T
            else:
                samples = samples.squeeze(0)
        
        # Create new frame manually
        new_frame = cls.__new__(cls)
        new_frame.samples = samples
        new_frame.nb_samples = samples.shape[-1] if samples.ndim > 1 else len(samples)
        new_frame.format = format
        new_frame.rate = sample_rate
        new_frame.layout = layout
        new_frame.timestamp = timestamp
        new_frame.time_base = time_base
        new_frame.log_timestamps = {}
        new_frame.side_data = SideData()
        return new_frame

    def to_av_frame(self) -> av.AudioFrame:
        """Convert this AudioFrame to av.AudioFrame."""
        try:
            samples = self.samples
            if self.format.endswith('p'):
                # Planar format - channels are separated (channels, samples)
                if samples.ndim == 1:
                    samples = samples.reshape(1, -1)
            else:
                # Packed format - channels are interleaved
                if samples.ndim == 2 and samples.shape[0] > 1:
                    # Convert (channels, samples) to (samples, channels) for packed format
                    samples = samples.T
                elif samples.ndim == 1:
                    # Keep 1D for mono packed format or reshape for multi-channel
                    if self.layout != 'mono':
                        pass
            av_frame = av.AudioFrame.from_ndarray(samples, format=self.format, layout=self.layout)
            av_frame.sample_rate = self.rate
            av_frame.pts = self.timestamp
            av_frame.time_base = self.time_base
            return av_frame
        except Exception as e:
            logger.warning(f"Audio conversion failed ({e}), creating dummy frame")
            dummy_samples = np.zeros((1, 1024), dtype=np.int16)
            av_frame = av.AudioFrame.from_ndarray(dummy_samples, format='s16', layout='mono')
            av_frame.sample_rate = self.rate
            av_frame.pts = self.timestamp
            av_frame.time_base = self.time_base
            return av_frame

class OutputFrame(ABC):
    """Base class for output frames."""
    
    @property
    def timestamp(self):
        """Get the timestamp of this output frame."""
        raise NotImplementedError("Subclasses must implement timestamp property")

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
    
    @property
    def timestamp(self):
        """Get the timestamp of the first audio frame, or 0 if no frames."""
        if self.frames:
            return self.frames[0].timestamp
        return 0
    
    @classmethod
    def with_monotonic_timestamps(cls, frames: List[AudioFrame], request_id: str, start_timestamp: int, frame_duration: int) -> 'AudioOutput':
        """
        Create AudioOutput with corrected monotonic timestamps in individual frames.
        
        The encoder uses individual AudioFrame.timestamp values, so we need to fix
        the actual frame timestamps, not just add a wrapper property.
        """
        corrected_frames = []
        for i, frame in enumerate(frames):
            # Create new frame with corrected timestamp
            corrected_timestamp = start_timestamp + (i * frame_duration)
            corrected_frame = AudioFrame._from_existing_with_timestamp(frame, corrected_timestamp)
            corrected_frames.append(corrected_frame)
        
        return cls(corrected_frames, request_id)
class FrameBuffer:
    """Rolling frame buffer that keeps a fixed number of frames."""
    
    def __init__(self, max_frames: int = 300):
        self.max_frames = max_frames
        self.frames: Deque[Union[VideoFrame, AudioFrame]] = deque(maxlen=max_frames)
        self.total_frames_received = 0
        self.total_frames_discarded = 0
        
    def add_frame(self, frame: Union[VideoFrame, AudioFrame]):
        if len(self.frames) >= self.max_frames:
            self.total_frames_discarded += 1
        self.frames.append(frame)
        self.total_frames_received += 1
        
    def get_frame(self) -> Optional[Union[VideoFrame, AudioFrame]]:
        return self.frames.popleft() if self.frames else None
        
    def get_all_frames(self) -> List[Union[VideoFrame, AudioFrame]]:
        frames = list(self.frames)
        self.frames.clear()
        return frames
        
    def clear(self):
        self.frames.clear()
        
    def size(self) -> int:
        return len(self.frames)
        
    def get_stats(self) -> Dict[str, int]:
        return {
            "current_frames": len(self.frames),
            "max_frames": self.max_frames,
            "total_received": self.total_frames_received,
            "total_discarded": self.total_frames_discarded
        }
