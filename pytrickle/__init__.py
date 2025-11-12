"""
Trickle App - A Python package for high-performance video streaming over trickle protocol.

Provides functionality to subscribe to and publish video streams with real-time processing.
"""

from .client import TrickleClient
from .server import StreamServer
from .protocol import TrickleProtocol
from .frames import (
    VideoFrame, AudioFrame, VideoOutput, AudioOutput,
    FrameBuffer,
)
from .state import StreamState
from .base import TrickleComponent, ComponentState, ErrorCallback
from .publisher import TricklePublisher
from .subscriber import TrickleSubscriber
from .manager import BaseStreamManager, TrickleStreamManager, StreamHandler
from .stream_handler import TrickleStreamHandler
from .utils.register import RegisterCapability
from .utils.loading_overlay import build_loading_overlay_frame

# Processing utilities
from .frame_processor import FrameProcessor
from .stream_processor import StreamProcessor, VideoProcessingResult
from .fps_meter import FPSMeter
from .frame_skipper import FrameSkipConfig
from .loading_config import LoadingConfig, LoadingMode

from . import api

from .version import __version__

__all__ = [
    "TrickleClient",
    "StreamServer",
    "StreamProcessor",
    "VideoProcessingResult",
    "TrickleProtocol",
    "VideoFrame",
    "AudioFrame", 
    "VideoOutput",
    "AudioOutput",
    "TricklePublisher",
    "TrickleSubscriber",
    "BaseStreamManager", 
    "TrickleStreamManager",
    "StreamHandler",
    "TrickleStreamHandler",
    "FrameBuffer",
    "StreamState",
    "TrickleComponent",
    "ComponentState",
    "RegisterCapability",
    "api",
    "ErrorCallback",
    "FrameProcessor",
    "FPSMeter",
    "FrameSkipConfig",
    "LoadingConfig",
    "LoadingMode",
    "build_loading_overlay_frame",
    "__version__"
] 