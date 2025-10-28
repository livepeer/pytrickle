#!/usr/bin/env python3
"""
Test model loading state transitions.

This test specifically verifies that the ensure_model_loaded functionality
correctly manages state transitions from LOADING to IDLE during model loading.

The model loading is triggered via sentinel messages through the parameter update
mechanism, leveraging existing infrastructure rather than custom background tasks.
"""

import asyncio
import pytest
import time

from pytrickle.frame_processor import FrameProcessor
from pytrickle.stream_processor import StreamProcessor, _InternalFrameProcessor
from pytrickle.state import StreamState


class TestModelLoadingFrameProcessor(FrameProcessor):
    """Test frame processor that simulates model loading with delays."""
    
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
        
        # Simulate model loading time
        await asyncio.sleep(self.load_delay)
        
        self.load_end_time = time.time()
    
    async def process_video_async(self, frame):
        return frame
    
    async def process_audio_async(self, frame):
        return [frame]
    
    async def update_params(self, params):
        pass


class TestModelLoadingStateTransitions:
    """Test state transitions during model loading."""

    @pytest.mark.asyncio
    async def test_ensure_model_loaded_state_transition(self):
        """Test that ensure_model_loaded properly transitions state from LOADING to IDLE."""
        processor = TestModelLoadingFrameProcessor(load_delay=0.1)
        state = StreamState()
        processor.attach_state(state)
        
        # Initial state should be LOADING
        assert state.get_pipeline_state()["status"] == "LOADING"
        assert not state.startup_complete
        assert not processor._model_loaded
        
        # Call ensure_model_loaded
        await processor.ensure_model_loaded()
        
        # After model loading, state should be IDLE
        assert state.get_pipeline_state()["status"] == "IDLE"
        assert state.startup_complete
        assert processor._model_loaded
        assert processor.load_call_count == 1

    @pytest.mark.asyncio
    async def test_ensure_model_loaded_only_loads_once(self):
        """Test that ensure_model_loaded only loads the model once, even with concurrent calls."""
        processor = TestModelLoadingFrameProcessor(load_delay=0.2)
        state = StreamState()
        processor.attach_state(state)
        
        # Make multiple concurrent calls to ensure_model_loaded
        tasks = [
            processor.ensure_model_loaded(),
            processor.ensure_model_loaded(),
            processor.ensure_model_loaded()
        ]
        
        await asyncio.gather(*tasks)
        
        # Model should only be loaded once
        assert processor.load_call_count == 1
        assert processor._model_loaded
        assert state.startup_complete

    @pytest.mark.asyncio
    async def test_ensure_model_loaded_thread_safety(self):
        """Test that ensure_model_loaded is thread-safe with proper locking."""
        processor = TestModelLoadingFrameProcessor(load_delay=0.1)
        state = StreamState()
        processor.attach_state(state)
        
        # Track timing to verify sequential execution due to lock
        start_time = time.time()
        
        # Create multiple tasks that should be serialized by the lock
        async def load_and_track():
            await processor.ensure_model_loaded()
            return time.time()
        
        tasks = [load_and_track() for _ in range(3)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All tasks should complete, but model loaded only once
        assert len(results) == 3
        assert processor.load_call_count == 1
        assert processor._model_loaded
        
        # Time should be close to single load time (not 3x due to concurrency)
        # Allow some overhead for task scheduling
        assert total_time < 0.3  # Should be ~0.1s + overhead, not 0.3s

    @pytest.mark.asyncio
    async def test_ensure_model_loaded_handles_failures(self):
        """Test that ensure_model_loaded properly handles model loading failures."""
        processor = TestModelLoadingFrameProcessor(should_fail=True)
        state = StreamState()
        processor.attach_state(state)
        
        # ensure_model_loaded should propagate the exception
        with pytest.raises(Exception, match="Simulated model loading failure"):
            await processor.ensure_model_loaded()
        
        # State should remain in LOADING, model not marked as loaded
        assert state.get_pipeline_state()["status"] == "LOADING"
        assert not state.startup_complete
        assert not processor._model_loaded
        assert processor.load_call_count == 1

    @pytest.mark.asyncio
    async def test_ensure_model_loaded_without_state(self):
        """Test that ensure_model_loaded works even without attached state."""
        processor = TestModelLoadingFrameProcessor(load_delay=0.05)
        # Don't attach state
        
        # Should still work, just won't update state
        await processor.ensure_model_loaded()
        
        assert processor._model_loaded
        assert processor.load_call_count == 1

    @pytest.mark.asyncio
    async def test_stream_processor_model_preloading_state_transition(self):
        """Test that StreamProcessor properly preloads model via sentinel message."""
        model_load_called = False
        
        async def mock_load_model(**kwargs):
            nonlocal model_load_called
            model_load_called = True
            await asyncio.sleep(0.1)  # Simulate loading time
        
        # Create StreamProcessor with mock loader
        processor = StreamProcessor(
            model_loader=mock_load_model,
            name="test-processor",
            port=8001  # Use different port to avoid conflicts
        )
        
        # Get the server state
        server_state = processor.server.state
        
        # Initially should be LOADING
        initial_state = server_state.get_pipeline_state()
        assert initial_state["status"] == "LOADING"
        assert not model_load_called
        
        # Trigger model loading via sentinel message (simulating automatic trigger)
        await processor._frame_processor.update_params({"_load_model": True})
        
        # After sentinel message, model should be loaded and state should be IDLE
        assert model_load_called
        assert processor._frame_processor._model_loaded
        final_state = server_state.get_pipeline_state()
        assert final_state["status"] == "IDLE"
        assert server_state.startup_complete

    @pytest.mark.asyncio
    async def test_internal_frame_processor_ensure_model_loaded(self):
        """Test that _InternalFrameProcessor uses ensure_model_loaded correctly."""
        load_called = False
        
        async def mock_load_model(**kwargs):
            nonlocal load_called
            load_called = True
            await asyncio.sleep(0.05)
        
        processor = _InternalFrameProcessor(
            model_loader=mock_load_model,
            name="test-internal"
        )
        
        state = StreamState()
        processor.attach_state(state)
        
        # Initially LOADING
        assert state.get_pipeline_state()["status"] == "LOADING"
        
        # Call ensure_model_loaded
        await processor.ensure_model_loaded()
        
        # Should have called load_model and transitioned to IDLE
        assert load_called
        assert processor._model_loaded
        assert state.get_pipeline_state()["status"] == "IDLE"

    @pytest.mark.asyncio
    async def test_concurrent_ensure_model_loaded_calls(self):
        """Test multiple concurrent ensure_model_loaded calls with realistic timing."""
        processor = TestModelLoadingFrameProcessor(load_delay=0.2)
        state = StreamState()
        processor.attach_state(state)
        
        # Record timing for each call
        call_times = []
        
        async def timed_ensure_load():
            start = time.time()
            await processor.ensure_model_loaded()
            end = time.time()
            call_times.append((start, end))
        
        # Start multiple concurrent calls
        tasks = [timed_ensure_load() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        # Verify model loaded only once
        assert processor.load_call_count == 1
        assert processor._model_loaded
        assert state.startup_complete
        
        # All calls should complete successfully
        assert len(call_times) == 5
        
        # The first call should take ~0.2s, others should be much faster
        # (they wait for the lock but don't reload)
        sorted_times = sorted(call_times, key=lambda x: x[1] - x[0])
        first_duration = sorted_times[0][1] - sorted_times[0][0]
        
        # First call should take at least the load delay
        assert first_duration >= 0.15  # Allow some timing variance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
