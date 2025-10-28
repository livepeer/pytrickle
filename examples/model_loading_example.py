#!/usr/bin/env python3
"""
Model Loading State Transition Example

This example demonstrates the sentinel message model loading pattern:

1. Server starts immediately and is available for /health checks
2. Model loading is triggered automatically via '_load_model' sentinel message
3. Health endpoint shows LOADING -> IDLE transition during model loading
4. Uses simple passthrough processing to focus on the model loading behavior

To test the model loading behavior:
1. Start this example: python examples/model_loading_example.py
2. Check health during loading: curl http://localhost:8000/health
3. Should see: {"status": "LOADING"} -> {"status": "IDLE"}

The model loading leverages the existing parameter update infrastructure
rather than custom background tasks, making it more reliable and testable.
"""

import asyncio
import logging
import time
from pytrickle import StreamProcessor
from pytrickle.frames import VideoFrame, AudioFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model loading configuration
MODEL_LOAD_DELAY_SECONDS = 10.0
model_loaded = False
model_load_start_time = None
model_load_end_time = None

async def load_model(**kwargs):
    """Simulate model loading with realistic delay.
    
    This function is triggered automatically by the StreamProcessor via a 
    '_load_model' sentinel message sent through the parameter update mechanism.
    The server starts immediately and is available for /health checks while 
    this model loading happens in the background.
    """
    global model_loaded, model_load_start_time, model_load_end_time
    
    model_load_start_time = time.time()
    logger.info("🔄 Model loading started - triggered via sentinel message")
    
    # Simulate realistic model loading operations
    if MODEL_LOAD_DELAY_SECONDS > 0:
        logger.info(f"Loading model components for {MODEL_LOAD_DELAY_SECONDS:.1f}s (server available at /health)")
        
        # Simulate different model loading phases
        phases = [
            ("Loading model weights", 0.4),
            ("Initializing GPU memory", 0.3), 
            ("Compiling model", 0.3)
        ]
        
        for phase, duration in phases:
            logger.info(f"  {phase}...")
            await asyncio.sleep(duration * MODEL_LOAD_DELAY_SECONDS)
    
    # In real applications, load your model here:
    # model = torch.load('my_model.pth')
    # model.to('cuda')
    # model.eval()
    # tokenizer = AutoTokenizer.from_pretrained('model_name')
    
    model_loaded = True
    model_load_end_time = time.time()
    load_duration = model_load_end_time - model_load_start_time
    
    logger.info(f"✅ Model loading completed in {load_duration:.2f}s - server will transition to IDLE")

async def process_video(frame: VideoFrame) -> VideoFrame:
    """Simple passthrough video processing - focus is on model loading, not processing."""
    # In a real application, you would use your loaded model here:
    # processed_tensor = model(frame.tensor)
    # return frame.replace_tensor(processed_tensor)
    
    # For this example, just pass through unchanged
    return frame

async def process_audio(frame: AudioFrame) -> list[AudioFrame]:
    """Simple passthrough audio processing."""
    # In a real application, you would use your loaded model here:
    # processed_audio = model(frame.tensor)
    # return [frame.replace_tensor(processed_audio)]
    
    # For this example, just pass through unchanged
    return [frame]

async def update_params(params: dict):
    """Handle parameter updates.
    
    Note: The '_load_model' sentinel message is automatically filtered out
    by the StreamProcessor and won't reach this function.
    """
    logger.info(f"Parameters updated: {params}")
    
    # Handle your application-specific parameters here
    if "model_param" in params:
        logger.info(f"Model parameter updated: {params['model_param']}")

async def on_stream_start():
    """Called when a stream starts."""
    logger.info("🎬 Stream started - model should already be loaded")
    if model_loaded:
        logger.info("✅ Model is ready for processing")
    else:
        logger.warning("⚠️  Model not loaded yet - this shouldn't happen")

async def on_stream_stop():
    """Called when a stream stops."""
    logger.info("🛑 Stream stopped")

# Create and run StreamProcessor
if __name__ == "__main__":
    logger.info("🚀 Starting Model Loading Example")
    logger.info("=" * 50)
    logger.info("This example demonstrates:")
    logger.info("1. Immediate server startup (non-blocking)")
    logger.info("2. Background model loading via sentinel messages")
    logger.info("3. LOADING -> IDLE state transition")
    logger.info("4. Health endpoint availability during loading")
    logger.info("=" * 50)
    
    processor = StreamProcessor(
        video_processor=process_video,
        audio_processor=process_audio,
        model_loader=load_model,  # Automatically triggered via sentinel message
        param_updater=update_params,
        on_stream_start=on_stream_start,
        on_stream_stop=on_stream_stop,
        name="model-loading-demo",
        port=8001,
    )
    
    logger.info("Starting StreamProcessor...")
    logger.info("Server will be available immediately at http://localhost:8001")
    logger.info("Try: curl http://localhost:8001/health")
    
    processor.run()