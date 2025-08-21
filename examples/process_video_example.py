#!/usr/bin/env python3
"""
Efficient OpenCV Green Processor using tensor utilities efficiently.

This version minimizes color space conversions for better performance.
"""
import asyncio
import logging
import torch
import cv2
import numpy as np
from pytrickle import StreamProcessor
from pytrickle.frames import VideoFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
intensity = 0.5
delay = 0.00
ready = False

def load_model(**kwargs):
    """Initialize processor state - called during model loading phase."""
    global intensity, ready
    
    logger.info(f"load_model called with kwargs: {kwargs}")
    
    # Set processor variables from kwargs or use defaults
    intensity = kwargs.get('intensity', 0.5)
    intensity = max(0.0, min(1.0, intensity))
    
    ready = True
    logger.info(f"✅ Efficient OpenCV Green processor ready (intensity: {intensity})")

async def process_video(frame: VideoFrame) -> VideoFrame:
    """Apply horizontal flip and green hue - EFFICIENT VERSION."""
    global intensity, ready, delay
    
    # Simulated processing time
    if delay > 0:
        await asyncio.sleep(delay)

    # Work directly with the internal tensor format to minimize conversions
    tensor = frame.tensor  # [B,H,W,C] RGB [0,1]
    
    # Remove batch dimension for processing
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # [H,W,C]
    
    # Mirror the frame horizontally (flip left to right)
    mirrored = torch.flip(tensor, dims=[1])  # Flip width dimension
    
    # Accent Green target color (#18794E: rgb(24,121,78) -> (0.094,0.475,0.306))
    target = torch.tensor([0.094, 0.475, 0.306], device=mirrored.device, dtype=mirrored.dtype)
    tint_adjustment = (target - 0.5) * intensity * 0.4
    
    # Apply tint adjustment using broadcasting
    tinted = torch.clamp(mirrored + tint_adjustment, 0, 1)
    
    # Add batch dimension back
    result_tensor = tinted.unsqueeze(0)  # [B,H,W,C]
    
    # Create result frame (no format conversion needed!)
    return frame.replace_tensor(result_tensor)

def update_params(params: dict):
    """Update tint intensity (0.0 to 1.0)."""
    global intensity, delay
    if "intensity" in params:
        old = intensity
        intensity = max(0.0, min(1.0, float(params["intensity"])))
        if old != intensity:
            logger.info(f"Intensity: {old:.2f} → {intensity:.2f}")
    if "delay" in params:
        old = delay
        delay = max(0.0, float(params["delay"]))
        if old != delay:
            logger.info(f"Delay: {old:.2f} → {delay:.2f}")

# Create and run StreamProcessor
if __name__ == "__main__":
    processor = StreamProcessor(
        video_processor=process_video,
        model_loader=load_model,
        param_updater=update_params,
        name="green-processor",
        port=8000
    )
    processor.run()
