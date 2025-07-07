"""
Trickle App - A Python package for high-performance video streaming over trickle protocol.

Provides functionality to subscribe to and publish video streams with real-time processing.
"""

from .client import TrickleClient, SimpleTrickleClient
from .server import TrickleApp, create_app
from .protocol import TrickleProtocol
from .frames import VideoFrame, AudioFrame, VideoOutput, AudioOutput
from .publisher import TricklePublisher
from .subscriber import TrickleSubscriber

__version__ = "0.1.0"

__all__ = [
    "TrickleClient",
    "SimpleTrickleClient",
    "TrickleApp",
    "create_app",
    "TrickleProtocol",
    "VideoFrame",
    "AudioFrame", 
    "VideoOutput",
    "AudioOutput",
    "TricklePublisher",
    "TrickleSubscriber",
] 