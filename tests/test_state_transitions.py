#!/usr/bin/env python3
"""
Test script to verify pipeline state transitions work correctly.
"""

import asyncio
import logging
from pytrickle import StreamProcessor
from pytrickle.state import PipelineState
from pytrickle.frames import VideoFrame
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test state
model_loaded = False
processing_count = 0

async def load_model(**kwargs):
    """Test model loader that simulates loading time."""
    global model_loaded
    logger.info("Starting model loading...")
    await asyncio.sleep(0.1)  # Simulate loading time
    model_loaded = True
    logger.info("Model loaded successfully")

async def process_video(frame: VideoFrame) -> VideoFrame:
    """Simple video processor for testing."""
    global processing_count
    processing_count += 1
    logger.info(f"Processing frame #{processing_count}")
    return frame

async def update_params(params: dict):
    """Test parameter updater."""
    logger.info(f"Parameters updated: {params}")

async def test_state_transitions():
    """Test that state transitions work correctly."""
    logger.info("Creating StreamProcessor...")
    
    processor = StreamProcessor(
        video_processor=process_video,
        model_loader=load_model,
        param_updater=update_params,
        name="test-processor",
        port=8001
    )
    
    # Check initial state
    logger.info(f"Initial state: {processor.get_pipeline_state()}")
    assert processor.state.state == PipelineState.LOADING, "Should start in LOADING state"
    
    # Simulate model loading
    logger.info("Loading model...")
    await processor._frame_processor.load_model()
    
    # Check state after model loading
    logger.info(f"State after model loading: {processor.get_pipeline_state()}")
    assert processor.is_ready(), "Should be ready after model loading"
    assert processor.state.state == PipelineState.IDLE, "Should be in IDLE state after loading"
    
    # Test error handling
    logger.info("Testing error state...")
    processor.state.set_error("Test error")
    assert processor.is_error(), "Should be in error state"
    logger.info(f"Error state: {processor.get_pipeline_state()}")
    
    # Test error clearing
    logger.info("Clearing error...")
    processor.clear_error()
    assert not processor.is_error(), "Error should be cleared"
    logger.info(f"State after clearing error: {processor.get_pipeline_state()}")
    
    logger.info("✅ All state transition tests passed!")

if __name__ == "__main__":
    asyncio.run(test_state_transitions())