"""
Test utilities for Trickle streaming components.

Provides simple helpers for creating test doubles that work with the simplified architecture.
"""

import asyncio
from typing import Optional, Dict, Any
from unittest.mock import MagicMock, AsyncMock

from .frame_processor import FrameProcessor


class MockFrameProcessor(FrameProcessor):
    """Simple frame processor for testing."""
    
    def __init__(self, **kwargs):
        self.test_params = {}
        super().__init__(**kwargs)
    
    async def load_model(self, **kwargs):
        """Test model loader."""
        pass
    
    async def process_video_async(self, frame):
        """Test video processor."""
        return frame
    
    async def process_audio_async(self, frame):
        """Test audio processor."""
        return [frame]
    
    async def update_params(self, params: Dict[str, Any]):
        """Test parameter updater."""
        if params:
            self.test_params.update(params)


def create_mock_client():
    """Create a simple mock TrickleClient for testing."""
    mock_client = MagicMock()
    mock_client.running = False
    
    # Simple start that returns immediately to avoid hanging
    async def mock_start_immediate(request_id="default"):
        mock_client.running = True
        return
    
    async def mock_stop():
        mock_client.running = False
    
    mock_client.start = AsyncMock(side_effect=mock_start_immediate)
    mock_client.stop = AsyncMock(side_effect=mock_stop)
    mock_client.publish_data = AsyncMock()
    
    # Mock protocol with proper monitoring support
    mock_protocol = MagicMock()
    mock_protocol.fps_meter = MagicMock()
    mock_protocol.fps_meter.get_fps_stats.return_value = {"ingress": 30.0, "egress": 29.5}
    mock_protocol.emit_monitoring_event = AsyncMock()
    mock_client.protocol = mock_protocol
    
    return mock_client


def create_test_server_for_endpoints(
    capability_name: str = "test-model",
    pipeline: str = "test-pipeline", 
    version: str = "1.0.0",
    enable_default_routes: bool = True,
):
    """
    Create a StreamServer for testing HTTP endpoints.
    
    Uses real processor from examples for comprehensive testing.
    """
    from .stream_processor import _InternalFrameProcessor
    from .server import StreamServer
    
    # Import example functions for a real processor
    try:
        from examples.process_video_example import load_model, process_video, update_params
        
        processor = _InternalFrameProcessor(
            video_processor=process_video,
            audio_processor=None,
            model_loader=load_model,
            param_updater=update_params,
            name=capability_name,
        )
    except ImportError:
        # Fallback to mock processor if examples aren't available
        processor = MockFrameProcessor()
    
    server = StreamServer(
        frame_processor=processor,
        port=0,  # ephemeral port for safety
        capability_name=capability_name,
        pipeline=pipeline,
        version=version,
        enable_default_routes=enable_default_routes,
    )
    
    # Attach state and mark ready
    processor.attach_state(server.state)
    server.state.set_startup_complete()
    
    return server


def create_simple_async_mock():
    """Create a simple AsyncMock that returns immediately."""
    return AsyncMock(return_value=None)
