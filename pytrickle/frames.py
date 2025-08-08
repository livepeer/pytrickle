"""
Frame data structures and processing utilities for trickle streaming.

Defines VideoFrame, AudioFrame, their output counterparts, frame processing
utilities, and streaming utilities for handling media data in the trickle 
streaming pipeline.
"""

import asyncio
import logging
import time
import torch
import numpy as np
import av
from typing import Optional, Dict, Any, Union, List, Deque
from fractions import Fraction
from collections import deque
from abc import ABC

logger = logging.getLogger(__name__)

# Tensor Conversion Utilities
# ============================

def tensor_to_av_frame(tensor: torch.Tensor) -> av.VideoFrame:
    """
    Convert a tensor to av.VideoFrame for use in video pipelines.
    Handles [B, H, W, C] or [H, W, C] formats, float or uint8, and grayscale/RGB.
    """
    try:
        # Handle tensor format conversion - trickle uses [B, H, W, C] or [H, W, C]
        if tensor.dim() == 4:
            # Expected format: [B, H, W, C] where B=1
            if tensor.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got {tensor.shape[0]}")
            tensor = tensor.squeeze(0)  # Remove batch dimension: [H, W, C]
        elif tensor.dim() == 3:
            # Already in [H, W, C] format
            pass
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D tensor with shape {tensor.shape}")

        # Validate tensor format
        if tensor.dim() != 3:
            raise ValueError(f"Expected 3D tensor after conversion, got {tensor.dim()}D")
        if tensor.shape[2] not in [1, 3, 4]:
            raise ValueError(f"Expected 1, 3, or 4 channels, got {tensor.shape[2]}")

        # Convert tensor to numpy array for av.VideoFrame
        # Handle different tensor value ranges
        if tensor.dtype in [torch.float32, torch.float64]:
            if tensor.max() <= 1.0:
                # Tensor is in [0, 1] range, convert to [0, 255]
                tensor_np = (tensor * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
            else:
                # Tensor is already in [0, 255] range
                tensor_np = tensor.clamp(0, 255).to(torch.uint8).cpu().numpy()
        elif tensor.dtype == torch.uint8:
            tensor_np = tensor.cpu().numpy()
        else:
            # Convert other types to uint8
            tensor_np = tensor.clamp(0, 255).to(torch.uint8).cpu().numpy()

        # Ensure numpy array is contiguous
        if not tensor_np.flags.c_contiguous:
            tensor_np = np.ascontiguousarray(tensor_np)

        # Handle grayscale to RGB conversion if needed
        if tensor_np.shape[2] == 1:
            tensor_np = np.repeat(tensor_np, 3, axis=2)

        # Create av.VideoFrame from numpy array
        av_frame = av.VideoFrame.from_ndarray(tensor_np, format="rgb24")

        return av_frame

    except Exception as e:
        # Optionally, you could log here if logger is available
        raise

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

    # Consolidated utilities (previously in FrameProcessor/FrameConversionMixin)
    def to_av_frame(self) -> 'av.VideoFrame':
        """Convert this VideoFrame to av.VideoFrame."""
        return tensor_to_av_frame(self.tensor)

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

    # Consolidated utilities (previously in FrameProcessor/FrameConversionMixin)
    def to_av_frame(self) -> 'av.AudioFrame':
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


# Frame Processing Utilities
# ===========================



# Streaming Utilities
# ====================

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


class StreamState:
    """Unified state management for stream lifecycle."""
    
    def __init__(self):
        self.running = False
        self.pipeline_ready = False
        self.shutting_down = False
        self.error_occurred = False
        self.cleanup_in_progress = False
        
        self.running_event = asyncio.Event()
        self.shutdown_event = asyncio.Event()
        self.error_event = asyncio.Event()
        self.pipeline_ready_event = asyncio.Event()
    
    @property
    def is_active(self) -> bool:
        return self.running and not self.shutting_down and not self.error_occurred
    
    @property
    def shutdown_flags(self) -> Dict[str, bool]:
        return {
            'shutdown_event': self.shutdown_event.is_set(),
            'cleanup_in_progress': self.cleanup_in_progress
        }
    
    def start(self):
        self.running = True
        self.running_event.set()
    
    def mark_pipeline_ready(self):
        self.pipeline_ready = True
        self.pipeline_ready_event.set()
    
    def initiate_shutdown(self, due_to_error: bool = False):
        self.shutting_down = True
        self.shutdown_event.set()
        if due_to_error:
            self.error_occurred = True
            self.error_event.set()
    
    def mark_cleanup_in_progress(self):
        self.cleanup_in_progress = True
    
    def finalize(self):
        self.running = False
        self.running_event.clear()


class StreamErrorHandler:
    """Centralized error handling for streaming applications."""
    
    @staticmethod
    def log_error(error_type: str, exception: Optional[Exception], request_id: str, critical: bool = False):
        level = logger.error if critical else logger.warning
        msg = f"{error_type} for stream {request_id}"
        if exception:
            msg += f": {exception}"
        level(msg)

    @staticmethod
    def is_shutdown_error(shutdown_flags: Dict) -> bool:
        return shutdown_flags.get('shutdown_event', False) or shutdown_flags.get('cleanup_in_progress', False)


class StreamingUtils:
    """Generic utilities for streaming applications."""
    
    @staticmethod
    async def cancel_task_with_timeout(task: Optional[asyncio.Task], task_name: str, timeout: float = 3.0) -> bool:
        """Cancel an asyncio task with a timeout."""
        if not task or task.done():
            return True
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=timeout)
            return True
        except (asyncio.CancelledError, asyncio.TimeoutError):
            return True
        except Exception:
            return False


# Legacy alias for backward compatibility
def tensor_to_av_frame_legacy(tensor: torch.Tensor) -> av.VideoFrame:
    """Legacy alias for tensor_to_av_frame for backward compatibility."""
    return tensor_to_av_frame(tensor)