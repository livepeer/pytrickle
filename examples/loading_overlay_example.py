#!/usr/bin/env python3
"""
Model Loading and Loading Overlay Example

This comprehensive example demonstrates:
1. Non-blocking model loading with configurable delay
2. Server health state transitions (LOADING -> IDLE)
3. Automatic animated loading overlay during processing (internalized feature)
4. Real-time parameter updates to control overlay

The server starts immediately and is available for /health checks while
the model loads in the background. The loading overlay is now automatically
handled by the framework - no manual implementation needed!

To test:
1. Run: python examples/loading_overlay_example.py
2. Check health: curl http://localhost:8000/health
3. Update parameters to enable loading overlay:
   curl -X POST http://localhost:8000/update_params \
     -H "Content-Type: application/json" \
     -d '{"show_loading": true, "loading_message": "Processing..."}'
"""

import asyncio
import logging
import time
from pytrickle.stream_processor import StreamProcessor

from pytrickle.frames import VideoFrame, AudioFrame
from pytrickle.frame_skipper import FrameSkipConfig
from pytrickle.utils.register import RegisterCapability

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model loading configuration
MODEL_LOAD_DELAY_SECONDS = 3.0  # Configurable model load delay
model_loaded = False
model_load_start_time = None

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

async def on_stream_start():
    """Called when a stream starts."""
    logger.info("🎬 Stream started")
    if not model_loaded:
        logger.warning("⚠️  Model not loaded yet - frames will pass through until ready")

async def on_stream_stop():
    """Called when stream stops - cleanup resources."""
    logger.info("🛑 Stream stopped, cleaning up resources")
    # Note: Loading state is now automatically reset by the framework
    logger.info("✅ Resources cleaned up")

async def process_video(frame: VideoFrame) -> VideoFrame:
    """
    Process video frame with normal processing logic.
    
    The loading overlay is now automatically handled by the framework when enabled
    via parameter updates (show_loading=true). No manual implementation needed!
    
    You can focus on your actual processing logic here.
    """
    # Just do your normal video processing here
    # The framework automatically shows loading overlay when show_loading=true
    return frame

async def process_audio(frame: AudioFrame) -> list[AudioFrame]:
    """Pass-through audio processing."""
    return [frame]

async def update_params(params: dict):
    """
    Handle real-time parameter updates from the client.
    
    Note: Loading-related parameters (show_loading, loading_message, loading_mode, 
    loading_progress) are now handled automatically by the framework. They won't
    appear in this callback.
    
    This callback is now only for your custom application parameters.
    """
    logger.info(f"Custom parameters updated: {params}")
    
    # Add your custom parameter handling here
    # e.g., if "threshold" in params: ...
    pass

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
    logger.info("Model Loading & Loading Overlay Example (Internalized)")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This example demonstrates:")
    logger.info("  1. Non-blocking model loading with health state transitions")
    logger.info("  2. Automatic loading overlay (internalized in framework)")
    logger.info("  3. Simple parameter updates to control overlay behavior")
    logger.info("")
    logger.info("Server will start immediately on http://localhost:8000")
    logger.info(f"Model will load in background (~{MODEL_LOAD_DELAY_SECONDS}s delay)")
    logger.info("")
    logger.info("✨ NEW: Loading overlay is now built-in! Just enable it with parameters.")
    logger.info("")
    logger.info("Test endpoints:")
    logger.info("  curl http://localhost:8000/health")
    logger.info("  curl -X POST http://localhost:8000/update_params \\")
    logger.info('    -H "Content-Type: application/json" \\')
    logger.info('    -d \'{"show_loading": true, "loading_message": "Processing..."}\'')
    logger.info("")
    logger.info("Loading overlay parameters (handled automatically):")
    logger.info("  - show_loading: true/false - Enable/disable loading overlay")
    logger.info("  - loading_message: string - Custom loading message")
    logger.info("  - loading_mode: 'overlay'/'passthrough' - Display mode")
    logger.info("  - loading_progress: 0.0-1.0 - Progress bar (optional)")
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