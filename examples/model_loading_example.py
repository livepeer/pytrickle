#!/usr/bin/env python3
"""
Model Loading Example

Demonstrates non-blocking model loading with server health state transitions.

- Server starts immediately and is available for /health checks.
- Model loading is triggered in the background.
- Health endpoint transitions from LOADING to IDLE.

To test:
1. Run: python examples/model_loading_example.py
2. Check health: curl http://localhost:8001/health
"""

import asyncio
import logging
import time
from pytrickle import StreamProcessor
from pytrickle.frames import VideoFrame, AudioFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model loading configuration
MODEL_LOAD_DELAY_SECONDS = 5.0
model_loaded = False
model_load_start_time = None

async def load_model(**kwargs):
    """Simulate a model loading process with a delay."""
    global model_loaded, model_load_start_time
    
    model_load_start_time = time.time()
    logger.info("Model loading started...")
    
    if MODEL_LOAD_DELAY_SECONDS > 0:
        logger.info(f"Simulating model load for {MODEL_LOAD_DELAY_SECONDS:.1f}s...")
        await asyncio.sleep(MODEL_LOAD_DELAY_SECONDS)
    
    # In a real application, you would load your model here.
    # e.g., model = torch.load('my_model.pth')
    
    model_loaded = True
    load_duration = time.time() - model_load_start_time
    logger.info(f"Model loading complete in {load_duration:.2f}s.")

async def process_video(frame: VideoFrame) -> VideoFrame:
    """Pass-through video processing. This example focuses on model loading."""
    return frame

async def process_audio(frame: AudioFrame) -> list[AudioFrame]:
    """Pass-through audio processing."""
    return [frame]

async def update_params(params: dict):
    """Handle real-time parameter updates from the client."""
    # The 'load_model' sentinel is handled internally and won't appear here.
    logger.info(f"Parameters updated: {params}")
    if "model_param" in params:
        logger.info(f"Model parameter updated: {params['model_param']}")

async def on_stream_start():
    """Called when a stream starts."""
    logger.info("🎬 Stream started")
    if not model_loaded:
        logger.warning(" Model not loaded yet - this shouldn't happen in a real app")

async def on_stream_stop():
    """Called when a stream stops."""
    logger.info("Stream stopped")

if __name__ == "__main__":
    processor = StreamProcessor(
        video_processor=process_video,
        audio_processor=process_audio,
        model_loader=load_model,
        param_updater=update_params,
        on_stream_start=on_stream_start,
        on_stream_stop=on_stream_stop,
        name="model-loading-demo",
        port=8001,
    )
    
    logger.info("Starting Model Loading Example on http://localhost:8001")
    logger.info("Health endpoint will transition from LOADING to IDLE.")
    logger.info("Test with: curl http://localhost:8001/health")
    
    processor.run()