#!/usr/bin/env python3
"""
Test sentinel message model loading approach.

This test verifies that model loading can be triggered via parameter updates
using a sentinel message, leveraging the existing parameter update infrastructure.
"""

import asyncio
import pytest
import time
from aiohttp.test_utils import TestServer, TestClient

from pytrickle.stream_processor import StreamProcessor
from pytrickle.server import StreamServer
from pytrickle.frame_processor import FrameProcessor
from pytrickle.state import PipelineState


class SentinelTestProcessor(FrameProcessor):
    """Test processor for sentinel message testing."""
    
    def __init__(self, load_delay: float = 0.2):
        super().__init__()
        self.load_delay = load_delay
        self.load_called = False
        self.load_start_time = None
        self.load_end_time = None
    
    async def load_model(self, **kwargs):
        """Simulate model loading with delay."""
        self.load_called = True
        self.load_start_time = time.time()
        await asyncio.sleep(self.load_delay)
        self.load_end_time = time.time()
    
    async def process_video_async(self, frame):
        return frame
    
    async def process_audio_async(self, frame):
        return [frame]
    
    async def update_params(self, params):
        # This should not be called for sentinel messages
        pass


class TestSentinelModelLoading:
    """Test sentinel message approach for model loading."""

    @pytest.mark.asyncio
    async def test_sentinel_message_triggers_model_loading(self):
        """Test that _load_model sentinel message triggers model loading."""
        processor = SentinelTestProcessor(load_delay=0.1)
        
        # Create server directly
        server = StreamServer(
            frame_processor=processor,
            port=0,  # ephemeral port
            capability_name="test-sentinel",
        )
        
        # Attach state
        processor.attach_state(server.state)
        
        # Initially should be LOADING and model not loaded
        assert server.state.get_pipeline_state()["status"] == "LOADING"
        assert not processor.load_called
        assert not processor._model_loaded
        
        # Send sentinel message via direct call to ensure_model_loaded (simulating the sentinel flow)
        await processor.ensure_model_loaded()
        
        # After sentinel, model should be loaded and state should be IDLE
        assert processor.load_called
        assert processor._model_loaded
        assert server.state.get_pipeline_state()["status"] == "IDLE"

    @pytest.mark.asyncio
    async def test_sentinel_message_does_not_reach_user_param_updater(self):
        """Test that sentinel messages are filtered out and don't reach user code."""
        user_params_received = []
        
        async def user_param_updater(params):
            user_params_received.append(params)
        
        # Create StreamProcessor with user param updater
        processor = StreamProcessor(
            model_loader=lambda **kwargs: asyncio.sleep(0.05),
            param_updater=user_param_updater,
            name="test-sentinel-filter",
            port=8006
        )
        
        # Send sentinel message
        await processor._frame_processor.update_params({"_load_model": True})
        
        # Send normal parameter update
        await processor._frame_processor.update_params({"intensity": 0.8})
        
        # Only normal params should reach user code, not sentinel
        assert len(user_params_received) == 1
        assert user_params_received[0] == {"intensity": 0.8}

    @pytest.mark.asyncio
    async def test_stream_processor_automatic_sentinel_trigger(self):
        """Test that StreamProcessor automatically triggers model loading via sentinel."""
        model_loaded = False
        
        async def test_model_loader(**kwargs):
            nonlocal model_loaded
            await asyncio.sleep(0.2)
            model_loaded = True
        
        # Create StreamProcessor (should automatically trigger model loading)
        processor = StreamProcessor(
            model_loader=test_model_loader,
            name="test-auto-sentinel",
            port=8007
        )
        
        # Start server in background
        server_task = asyncio.create_task(processor.run_forever())
        
        try:
            # Give server time to start and trigger model loading
            await asyncio.sleep(0.1)
            
            # Server should be available immediately
            import aiohttp
            async with aiohttp.ClientSession() as session:
                # Check health - should be LOADING initially
                async with session.get("http://localhost:8007/health") as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert data["status"] == "LOADING"
                
                # Model should not be loaded yet
                assert not model_loaded
                
                # Wait for model loading to complete
                await asyncio.sleep(0.3)
                
                # Now model should be loaded and status should be IDLE
                async with session.get("http://localhost:8007/health") as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert data["status"] == "IDLE"
                
                assert model_loaded
                
        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_sentinel_message_error_handling(self):
        """Test error handling when model loading fails via sentinel message."""
        
        async def failing_model_loader(**kwargs):
            await asyncio.sleep(0.05)
            raise Exception("Model loading failed via sentinel")
        
        processor = StreamProcessor(
            model_loader=failing_model_loader,
            name="test-sentinel-error",
            port=8008
        )
        
        # Manually trigger sentinel (simulating what happens automatically)
        with pytest.raises(Exception, match="Model loading failed via sentinel"):
            await processor._frame_processor.update_params({"_load_model": True})
        
        # State should remain LOADING (not IDLE) due to error
        assert processor._frame_processor.state.get_pipeline_state()["status"] == "LOADING"
        assert not processor._frame_processor._model_loaded

    @pytest.mark.asyncio
    async def test_mixed_sentinel_and_normal_params(self):
        """Test that normal parameters work alongside sentinel messages."""
        normal_params_received = []
        model_load_count = 0
        
        async def test_model_loader(**kwargs):
            nonlocal model_load_count
            model_load_count += 1
            await asyncio.sleep(0.05)
        
        async def param_updater(params):
            normal_params_received.append(params)
        
        processor = StreamProcessor(
            model_loader=test_model_loader,
            param_updater=param_updater,
            name="test-mixed-params",
            port=8009
        )
        
        # Send mixed parameters including sentinel
        await processor._frame_processor.update_params({
            "_load_model": True,
            "intensity": 0.7,
            "effect": "blur"
        })
        
        # Send normal parameters
        await processor._frame_processor.update_params({
            "intensity": 0.9,
            "speed": 1.5
        })
        
        # Model should be loaded once
        assert model_load_count == 1
        assert processor._frame_processor._model_loaded
        
        # Only normal parameters should reach user param_updater
        assert len(normal_params_received) == 1
        assert normal_params_received[0] == {"intensity": 0.9, "speed": 1.5}

    @pytest.mark.asyncio
    async def test_sentinel_message_thread_safety(self):
        """Test that concurrent sentinel messages are handled safely."""
        load_count = 0
        
        async def counting_model_loader(**kwargs):
            nonlocal load_count
            load_count += 1
            await asyncio.sleep(0.1)
        
        processor = StreamProcessor(
            model_loader=counting_model_loader,
            name="test-sentinel-thread-safety",
            port=8010
        )
        
        # Send multiple concurrent sentinel messages
        tasks = [
            processor._frame_processor.update_params({"_load_model": True})
            for _ in range(5)
        ]
        
        await asyncio.gather(*tasks)
        
        # Model should only be loaded once despite multiple sentinel messages
        assert load_count == 1
        assert processor._frame_processor._model_loaded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
