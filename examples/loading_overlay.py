#!/usr/bin/env python3
"""
Model Loading and Loading Overlay Example

This comprehensive example demonstrates:
1. Non-blocking model loading with configurable delay
2. Server health state transitions (LOADING -> IDLE)
3. Optional animated loading overlay during processing
4. Real-time parameter updates to control overlay

The server starts immediately and is available for /health checks while
the model loads in the background. The loading overlay can be toggled
independently to show visual feedback during processing.

To test:
1. Run: python examples/loading_overlay.py
2. Check health: curl http://localhost:8001/health
3. Update parameters:
   curl -X POST http://localhost:8001/update_params \
     -H "Content-Type: application/json" \
     -d '{"show_loading": true, "loading_message": "Processing..."}'
"""

import asyncio
import logging
import time
import torch
from pytrickle import StreamProcessor
from pytrickle.frames import VideoFrame, AudioFrame
from pytrickle.frame_skipper import FrameSkipConfig
from pytrickle.utils.register import RegisterCapability
import numpy as np
from pytrickle.video_utils import create_loading_frame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model loading configuration
MODEL_LOAD_DELAY_SECONDS = 3.0  # Configurable model load delay
model_loaded = False
model_load_start_time = None

# Loading overlay state
show_loading = False
loading_message = "Loading..."
frame_counter = 0

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
    global model_loaded, model_load_start_time, show_loading
    
    model_load_start_time = time.time()
    logger.info("🔄 Model loading started...")
    
    # Get configurable delay from kwargs or use default
    load_delay = kwargs.get('load_delay', MODEL_LOAD_DELAY_SECONDS)
    show_loading = kwargs.get('show_loading', False)
    
    if load_delay > 0:
        logger.info(f"Simulating model load for {load_delay:.1f}s...")
        await asyncio.sleep(load_delay)
    
    # In a real application, load your model here
    # e.g., self.model = torch.load('my_model.pth')
    
    model_loaded = True
    load_duration = time.time() - model_load_start_time
    logger.info(f"✅ Model loading complete in {load_duration:.2f}s (show_loading: {show_loading})")

async def on_stream_start():
    """Called when a stream starts."""
    logger.info("🎬 Stream started")
    if not model_loaded:
        logger.warning("⚠️  Model not loaded yet - frames will pass through until ready")

async def on_stream_stop():
    """Called when stream stops - cleanup resources."""
    logger.info("🛑 Stream stopped, cleaning up resources")
    # Reset frame counter for next stream
    global frame_counter
    frame_counter = 0
    logger.info("✅ Resources cleaned up")

async def process_video(frame: VideoFrame) -> VideoFrame:
    """
    Process video frame - show loading overlay if enabled, otherwise passthrough.
    
    This demonstrates how to conditionally replace frames with a loading overlay.
    When show_loading is False, frames pass through unchanged.
    When show_loading is True, frames are replaced with animated loading graphics.
    """
    global show_loading, frame_counter, loading_message
    
    # Increment frame counter for animations
    frame_counter += 1
    
    if not show_loading:
        # Passthrough mode - return original frame
        return frame
    
    # Loading overlay mode - replace frame with loading animation
    frame_tensor = frame.tensor
    
    # Track if we need to add batch dimension back
    had_batch_dim = False
    
    # Handle both 3D and 4D tensors (with batch dimension)
    if len(frame_tensor.shape) == 4:
        if frame_tensor.shape[0] == 1:
            frame_tensor = frame_tensor.squeeze(0)
            had_batch_dim = True
        else:
            logger.error(f"Unexpected batch size: {frame_tensor.shape[0]}")
            return frame
    
    # Get frame dimensions
    if len(frame_tensor.shape) == 3:
        if frame_tensor.shape[0] == 3:  # CHW format
            height, width = frame_tensor.shape[1], frame_tensor.shape[2]
            was_chw = True
        else:  # HWC format
            height, width = frame_tensor.shape[0], frame_tensor.shape[1]
            was_chw = False
    else:
        logger.error(f"Unexpected tensor shape: {frame_tensor.shape}")
        return frame
    
    # Create loading overlay frame using utility (RGB format to match tensor expectations)
    loading_frame = create_loading_frame(width, height, loading_message, frame_counter, color_format="RGB")
    
    # Convert to tensor format matching input
    loading_tensor = torch.from_numpy(loading_frame.astype(np.float32))
    
    # Check if input was normalized (0-1) or (0-255)
    if frame_tensor.max() <= 1.0:
        loading_tensor = loading_tensor / 255.0
    
    # Convert to original tensor format
    if was_chw:
        loading_tensor = loading_tensor.permute(2, 0, 1)  # HWC to CHW
    
    # Add batch dimension back if needed
    if had_batch_dim:
        loading_tensor = loading_tensor.unsqueeze(0)
    
    # Move to same device as original tensor
    loading_tensor = loading_tensor.to(frame.tensor.device)
    
    return frame.replace_tensor(loading_tensor)

async def process_audio(frame: AudioFrame) -> list[AudioFrame]:
    """Pass-through audio processing."""
    return [frame]

async def update_params(params: dict):
    """
    Handle real-time parameter updates from the client.
    
    Note: The 'load_model' sentinel parameter is handled internally by
    StreamProcessor and won't appear here. User-defined parameters like
    'show_loading' and 'loading_message' are passed through.
    """
    global show_loading, loading_message
    
    logger.info(f"Parameters updated: {params}")
    
    if "show_loading" in params:
        old = show_loading
        show_loading = bool(params["show_loading"])
        if old != show_loading:
            status = "enabled" if show_loading else "disabled"
            logger.info(f"🎨 Loading overlay: {status}")
    
    if "loading_message" in params:
        old = loading_message
        loading_message = str(params["loading_message"])
        if old != loading_message:
            logger.info(f"💬 Loading message: '{old}' → '{loading_message}'")

async def register_with_orchestrator(app):
    """Register this capability with the orchestrator on startup."""
    registrar = RegisterCapability(logger)
    result = await registrar.register_capability()
    if result:
        logger.info(f"✅ Successfully registered capability with orchestrator: {result}")
    else:
        logger.info("ℹ️  Orchestrator registration skipped (no ORCH_URL/ORCH_SECRET provided)")

# Create and run StreamProcessor
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Model Loading & Loading Overlay Example")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This example demonstrates:")
    logger.info("  1. Non-blocking model loading with health state transitions")
    logger.info("  2. Animated loading overlay that can be toggled via parameters")
    logger.info("")
    logger.info("Server will start immediately on http://localhost:8001")
    logger.info(f"Model will load in background (~{MODEL_LOAD_DELAY_SECONDS}s delay)")
    logger.info("")
    logger.info("Test endpoints:")
    logger.info("  curl http://localhost:8001/health")
    logger.info("  curl -X POST http://localhost:8001/update_params \\")
    logger.info('    -H "Content-Type: application/json" \\')
    logger.info('    -d \'{"show_loading": true, "loading_message": "Processing..."}\'')
    logger.info("")
    logger.info("Parameters you can update:")
    logger.info("  - show_loading: true/false - Enable/disable loading overlay")
    logger.info("  - loading_message: string - Custom loading message")
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
        # Add orchestrator registration on startup
        on_startup=[register_with_orchestrator],
        route_prefix="/"
    )
    processor.run()