#!/usr/bin/env python3
"""
Model Loading and Overlay Example

Demonstrates:
1. Non-blocking model loading with configurable delay
2. Server health state transitions (LOADING -> IDLE)
3. Automatic loading overlay during processing delays
4. Real-time parameter updates

The server starts immediately and is available for /health checks while
the model loads in the background. The loading overlay automatically appears
when frames are withheld during processing delays.

To test:
1. Run: python examples/overlay_example.py
2. Check health: curl http://localhost:8000/health
3. Simulate a 15s processing stall:
   curl -X POST http://localhost:8000/update_params \
     -H "Content-Type: application/json" \
     -d '{"processing_delay": 15}'
"""

import asyncio
import logging
import time
from typing import Optional

from pytrickle.frames import AudioFrame, VideoFrame
from pytrickle.frame_skipper import FrameSkipConfig
from pytrickle.frame_overlay import OverlayConfig, OverlayMode
from pytrickle.stream_processor import StreamProcessor, VideoProcessingResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model loading configuration
MODEL_LOAD_DELAY_SECONDS = 3.0  # Configurable model load delay
model_loaded = False
model_load_start_time = None

# Reference to the running StreamProcessor (set in __main__)
processor: Optional[StreamProcessor] = None

# Delay coordination
STARTUP_BLOCK_SECONDS = 15.0
blocked_until: float = 0.0
pending_startup_block: float = STARTUP_BLOCK_SECONDS


async def load_model(**kwargs):
    """
    Simulate a model loading process with configurable delay.
    
    This demonstrates non-blocking model loading. The server is available
    for health checks immediately, while model loading happens in background.
    Health endpoint transitions from LOADING to IDLE once complete.
    
    In a real application, you would load your model here, e.g.:
    - model = torch.load('my_model.pth')
    - tokenizer = AutoTokenizer.from_pretrained('model_name')
    """
    global model_loaded, model_load_start_time
    
    model_load_start_time = time.time()
    logger.info("🔄 Model loading started...")
    
    # Get configurable delay from kwargs or use default
    load_delay = kwargs.get('load_delay', MODEL_LOAD_DELAY_SECONDS)
    
    if load_delay > 0:
        logger.info(f"Simulating model load for {load_delay:.1f}s...")
        await asyncio.sleep(load_delay)
    
    # In a real application, load your model here
    # e.g., self.model = torch.load('my_model.pth')
    
    model_loaded = True
    load_duration = time.time() - model_load_start_time
    logger.info(f"✅ Model loading complete in {load_duration:.2f}s")

async def on_stream_start(params: Optional[dict] = None):
    """Called when a stream starts."""
    global pending_startup_block, blocked_until

    logger.info("🎬 Stream started")
    if not model_loaded:
        logger.warning("⚠️  Model not loaded yet - frames will pass through until ready")

    if pending_startup_block > 0:
        block = pending_startup_block
        pending_startup_block = 0.0
        blocked_until = max(blocked_until, time.time() + block)
        logger.info("Startup block scheduled for %.1fs", block)

async def on_stream_stop():
    """Called when stream stops."""
    logger.info("🛑 Stream stopped")

async def process_video(frame: VideoFrame):
    """Return WITHHELD while blocked to trigger the automatic overlay."""
    global blocked_until

    if time.time() < blocked_until:
        return VideoProcessingResult.WITHHELD

    return frame

async def process_audio(frame: AudioFrame) -> list[AudioFrame]:
    """Pass-through audio processing."""
    return [frame]

async def update_params(params: dict):
    """
    Handle parameter updates.

    Parameters:
    - processing_delay: seconds to hold video output for the active stream
    - simulate_startup_block: seconds to hold output for the next stream start
    """
    global pending_startup_block, blocked_until

    logger.info(f"Custom parameters updated: {params}")

    if "simulate_startup_block" in params:
        pending_startup_block = max(0.0, float(params["simulate_startup_block"]))
        if pending_startup_block > 0:
            logger.info("Next stream will withhold frames for %.1fs", pending_startup_block)
        else:
            logger.info("Startup block cleared for next stream")

    processing_delay = float(params.get("processing_delay", 0.0))
    if processing_delay > 0:
        blocked_until = max(blocked_until, time.time() + processing_delay)
        logger.info("Simulating processing delay for %.1fs", processing_delay)
        asyncio.create_task(_log_processing_delay_completion(processing_delay))

# Background helpers
async def _log_processing_delay_completion(duration: float):
    """Log when the simulated processing delay completes."""
    try:
        await asyncio.sleep(duration)
        logger.info("Processing delay complete")
    except asyncio.CancelledError:
        logger.debug("Processing delay logging cancelled")


    # Additional custom parameters can be handled here.

# Create and run StreamProcessor
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Model Loading & Loading Overlay Example")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Server: http://localhost:8000")
    logger.info(f"Model loading delay: {MODEL_LOAD_DELAY_SECONDS}s")
    logger.info("")
    logger.info("Test endpoints:")
    logger.info("  curl http://localhost:8000/health")
    logger.info("  curl -X POST http://localhost:8000/update_params \\")
    logger.info('    -H "Content-Type: application/json" \\')
    logger.info('    -d \'{"processing_delay": 15}\'')
    logger.info("")
    
    processor = StreamProcessor(
        video_processor=process_video,
        audio_processor=process_audio,
        model_loader=load_model,
        param_updater=update_params,
        on_stream_start=on_stream_start,
        on_stream_stop=on_stream_stop,
        name="model-loading-demo",
        port=8000,
        frame_skip_config=FrameSkipConfig(),
        overlay_config=OverlayConfig(
            mode=OverlayMode.PROGRESSBAR,
            message="Loading...",
            enabled=True,
            auto_timeout_seconds=1.0,
        ),
        route_prefix="/"
    )
    processor.run()