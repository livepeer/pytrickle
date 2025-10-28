#!/usr/bin/env python3
"""
Tests for non-blocking model loading, state transitions, and sentinel messages.

This module consolidates tests for the model loading mechanism, covering:
- Correct state transitions from LOADING to IDLE.
- Thread-safe, idempotent model loading via `ensure_model_loaded`.
- Triggering model loading via `load_model` sentinel messages.
- Integration with StreamProcessor and the `/health` endpoint.
"""

import asyncio
import pytest
import time

from pytrickle.frame_processor import FrameProcessor
from pytrickle.stream_processor import StreamProcessor, _InternalFrameProcessor
from pytrickle.state import StreamState


class ModelLoadingTestProcessor(FrameProcessor):
    """Test frame processor that simulates model loading with delays and failures."""

    def __init__(self, load_delay: float = 0.1, should_fail: bool = False):
        super().__init__()
        self.load_delay = load_delay
        self.should_fail = should_fail
        self.load_call_count = 0
        self.load_start_time = None
        self.load_end_time = None

    async def load_model(self, **kwargs):
        """Simulate model loading with configurable delay and failure."""
        self.load_call_count += 1
        self.load_start_time = time.time()

        if self.should_fail:
            raise Exception("Simulated model loading failure")

        await asyncio.sleep(self.load_delay)
        self.load_end_time = time.time()

    async def process_video_async(self, frame):
        return frame

    async def process_audio_async(self, frame):
        return [frame]

    async def update_params(self, params):
        pass


class TestModelLoading:
    """Consolidated tests for model loading, state, and sentinel messages."""

    @pytest.mark.asyncio
    async def test_ensure_model_loaded_state_transition(self):
        """Test that ensure_model_loaded properly transitions state from LOADING to IDLE."""
        processor = ModelLoadingTestProcessor(load_delay=0.1)
        state = StreamState()
        processor.attach_state(state)

        assert state.get_pipeline_state()["status"] == "LOADING"
        assert not state.startup_complete
        assert not processor._model_loaded

        await processor.ensure_model_loaded()

        assert state.get_pipeline_state()["status"] == "IDLE"
        assert state.startup_complete
        assert processor._model_loaded
        assert processor.load_call_count == 1

    @pytest.mark.asyncio
    async def test_ensure_model_loaded_only_loads_once(self):
        """Test that ensure_model_loaded only loads the model once, even with concurrent calls."""
        processor = ModelLoadingTestProcessor(load_delay=0.2)
        state = StreamState()
        processor.attach_state(state)

        tasks = [
            processor.ensure_model_loaded(),
            processor.ensure_model_loaded(),
            processor.ensure_model_loaded()
        ]
        await asyncio.gather(*tasks)

        assert processor.load_call_count == 1
        assert processor._model_loaded
        assert state.startup_complete

    @pytest.mark.asyncio
    async def test_ensure_model_loaded_thread_safety(self):
        """Test that ensure_model_loaded is thread-safe with proper locking."""
        processor = ModelLoadingTestProcessor(load_delay=0.1)
        state = StreamState()
        processor.attach_state(state)

        start_time = time.time()
        async def load_and_track():
            await processor.ensure_model_loaded()
            return time.time()

        tasks = [load_and_track() for _ in range(3)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        total_time = end_time - start_time

        assert len(results) == 3
        assert processor.load_call_count == 1
        assert processor._model_loaded
        assert total_time < 0.3  # Should be ~0.1s + overhead, not 0.3s

    @pytest.mark.asyncio
    async def test_ensure_model_loaded_handles_failures(self):
        """Test that ensure_model_loaded properly handles model loading failures."""
        processor = ModelLoadingTestProcessor(should_fail=True)
        state = StreamState()
        processor.attach_state(state)

        with pytest.raises(Exception, match="Simulated model loading failure"):
            await processor.ensure_model_loaded()

        assert state.get_pipeline_state()["status"] == "LOADING"
        assert not state.startup_complete
        assert not processor._model_loaded
        assert processor.load_call_count == 1

    @pytest.mark.asyncio
    async def test_ensure_model_loaded_without_state(self):
        """Test that ensure_model_loaded works even without attached state."""
        processor = ModelLoadingTestProcessor(load_delay=0.05)
        await processor.ensure_model_loaded()
        assert processor._model_loaded
        assert processor.load_call_count == 1

    @pytest.mark.asyncio
    async def test_stream_processor_model_preloading_via_sentinel(self):
        """Test StreamProcessor preloads model via sentinel and transitions state."""
        model_load_called = False
        async def mock_load_model(**kwargs):
            nonlocal model_load_called
            model_load_called = True
            await asyncio.sleep(0.1)

        processor = StreamProcessor(
            model_loader=mock_load_model,
            name="test-processor",
            port=8001
        )
        server_state = processor.server.state

        initial_state = server_state.get_pipeline_state()
        assert initial_state["status"] == "LOADING"
        assert not model_load_called

        await processor._frame_processor.update_params({"load_model": True})

        assert model_load_called
        assert processor._frame_processor._model_loaded
        final_state = server_state.get_pipeline_state()
        assert final_state["status"] == "IDLE"
        assert server_state.startup_complete

    @pytest.mark.asyncio
    async def test_internal_frame_processor_ensure_model_loaded(self):
        """Test _InternalFrameProcessor uses ensure_model_loaded correctly."""
        load_called = False
        async def mock_load_model(**kwargs):
            nonlocal load_called
            load_called = True
            await asyncio.sleep(0.05)

        processor = _InternalFrameProcessor(model_loader=mock_load_model, name="test-internal")
        state = StreamState()
        processor.attach_state(state)

        assert state.get_pipeline_state()["status"] == "LOADING"
        await processor.ensure_model_loaded()
        assert load_called
        assert processor._model_loaded
        assert state.get_pipeline_state()["status"] == "IDLE"

    @pytest.mark.asyncio
    async def test_sentinel_message_is_filtered_from_user_param_updater(self):
        """Test that sentinel messages are filtered out and don't reach user code."""
        user_params_received = []
        async def user_param_updater(params):
            user_params_received.append(params)

        processor = StreamProcessor(
            model_loader=lambda **kwargs: asyncio.sleep(0.05),
            param_updater=user_param_updater,
            name="test-sentinel-filter",
            port=8006
        )

        await processor._frame_processor.update_params({"load_model": True})
        await processor._frame_processor.update_params({"intensity": 0.8})

        assert len(user_params_received) == 1
        assert user_params_received[0] == {"intensity": 0.8}

    @pytest.mark.asyncio
    async def test_stream_processor_automatic_trigger_and_health_check(self):
        """Test StreamProcessor auto-triggers model loading and /health reflects state."""
        model_loaded = False
        async def test_model_loader(**kwargs):
            nonlocal model_loaded
            await asyncio.sleep(0.2)
            model_loaded = True

        processor = StreamProcessor(
            model_loader=test_model_loader,
            name="test-auto-sentinel",
            port=8007
        )

        server_task = asyncio.create_task(processor.run_forever())
        try:
            await asyncio.sleep(0.1) # Give server time to start
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8007/health") as resp:
                    assert resp.status == 200
                    assert await resp.json() == {"status": "LOADING"}
                
                assert not model_loaded
                await asyncio.sleep(0.3) # Wait for model loading
                
                async with session.get("http://localhost:8007/health") as resp:
                    assert resp.status == 200
                    assert await resp.json() == {"status": "IDLE"}
                
                assert model_loaded
        finally:
            server_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await server_task

    @pytest.mark.asyncio
    async def test_sentinel_message_error_handling(self):
        """Test error handling when model loading fails via sentinel message."""
        async def failing_model_loader(**kwargs):
            raise Exception("Model loading failed via sentinel")

        processor = StreamProcessor(
            model_loader=failing_model_loader,
            name="test-sentinel-error",
            port=8008
        )

        with pytest.raises(Exception, match="Model loading failed via sentinel"):
            await processor._frame_processor.update_params({"load_model": True})

        assert processor._frame_processor.state.get_pipeline_state()["status"] == "LOADING"
        assert not processor._frame_processor._model_loaded

    @pytest.mark.asyncio
    async def test_mixed_sentinel_and_normal_params(self):
        """Test that normal parameters are ignored when sentinel is present."""
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

        # This call should only trigger model loading; other params are ignored.
        await processor._frame_processor.update_params({
            "load_model": True,
            "intensity": 0.7,
        })
        # This call should go to the user updater.
        await processor._frame_processor.update_params({"intensity": 0.9})

        assert model_load_count == 1
        assert processor._frame_processor._model_loaded
        assert len(normal_params_received) == 1
        assert normal_params_received[0] == {"intensity": 0.9}

    @pytest.mark.asyncio
    async def test_concurrent_sentinel_messages_are_thread_safe(self):
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

        tasks = [
            processor._frame_processor.update_params({"load_model": True})
            for _ in range(5)
        ]
        await asyncio.gather(*tasks)

        assert load_count == 1
        assert processor._frame_processor._model_loaded

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
