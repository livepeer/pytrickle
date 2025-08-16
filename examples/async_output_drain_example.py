#!/usr/bin/env python3
"""
Minimal Async StreamProcessor Demo (queue_mode + output drain)

This example shows the StreamProcessor pattern with queue_mode enabled so the 
Trickle client drains processed outputs via its output drain task. Audio is 
passed through unchanged, while video is processed asynchronously with a 
subtle Accent Green color tint.

Usage:
    python async_output_drain_example.py

Then start a stream with something like:
    curl -X POST http://localhost:8000/api/stream/start \
      -H 'Content-Type: application/json' \
      -d '{
            "subscribe_url": "http://127.0.0.1:3389/sample",
            "publish_url":   "http://127.0.0.1:3389/output",
            "gateway_request_id": "demo",
            "params": {"intensity": 0.5}
          }'
"""

import logging
import torch
from pytrickle import StreamProcessor, AudioPassthrough
from pytrickle.frames import VideoFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
intensity = 0.5
ready = False

def load_model(**kwargs):
    """Initialize processor state - called during model loading phase."""
    global intensity, ready
    
    logger.info(f"load_model called with kwargs: {kwargs}")
    
    # Set processor variables from kwargs or use defaults
    intensity = kwargs.get('intensity', 0.5)
    intensity = max(0.0, min(1.0, intensity))
    
    ready = True
    logger.info(f"✅ Minimal Async processor ready (intensity: {intensity:.3f}, ready: {ready})")

async def process_video(frame: VideoFrame) -> VideoFrame:
    """Apply subtle Accent Green color tint (#18794E)."""
    global intensity, ready
    
    if not ready:
        return frame
        
    try:
        tensor = frame.tensor
        if tensor is None:
            return frame
            
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
            
        # Ensure 4D (N H W C)
        if hasattr(tensor, "ndim") and tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        # Apply Accent Green tint (#18794E -> rgb approx [0.094, 0.475, 0.306])
        if intensity > 0.0:
            target = torch.tensor([0.094, 0.475, 0.306], device=tensor.device, dtype=tensor.dtype)
            tinted = torch.clamp(tensor + (target - 0.5) * 0.4, 0.0, 1.0)
            tensor = tensor * (1.0 - intensity) + tinted * intensity
            tensor = torch.clamp(tensor, 0.0, 1.0)

        return frame.replace_tensor(tensor)
    except Exception as e:
        logger.warning(f"Processing error: video_processing_error - {e}")
        return frame

def update_params(params: dict):
    """Update accent green intensity (0.0 to 1.0)."""
    global intensity
    if "intensity" in params:
        old = intensity
        try:
            new_i = float(params["intensity"])
            intensity = max(0.0, min(1.0, new_i))
            if old != intensity:
                logger.info(f"Intensity: {old:.3f} → {intensity:.3f}")
        except Exception:
            pass

# Create and run StreamProcessor
if __name__ == "__main__":
    processor = StreamProcessor(
        video_processor=process_video,  # Simple video processor function
        audio_processor=AudioPassthrough(),  # Explicit audio passthrough
        model_loader=load_model,
        param_updater=update_params,
        name="minimal-async-output-drain",
        port=8000,
        capability_name="minimal-async-output-drain",
        queue_mode=True
    )
    processor.run()


