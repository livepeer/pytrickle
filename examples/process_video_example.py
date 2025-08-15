#!/usr/bin/env python3
"""
OpenCV Green Processor using StreamProcessor
"""
import asyncio
import logging
import torch
import cv2
import numpy as np
from pytrickle import StreamProcessor
from pytrickle.frames import VideoFrame
import time

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
    
    # Load the model here if needed
    # model = torch.load('my_model.pth')
    
    ready = True
    logger.info(f"✅ OpenCV Green processor with horizontal flip ready (intensity: {intensity}, ready: {ready})")

async def process_video(frame: VideoFrame) -> VideoFrame:
    """Apply horizontal flip and green hue using OpenCV."""
    global intensity, ready, delay
    
    # Simulated processing time
    time.sleep(delay)

    frame_tensor = frame.tensor
    
    # Track if we need to add batch dimension back
    had_batch_dim = False
    
    # Handle both 3D and 4D tensors (with batch dimension)
    if len(frame_tensor.shape) == 4:
        # 4D tensor: [batch, height, width, channels] or [batch, channels, height, width]
        if frame_tensor.shape[0] == 1:
            # Remove batch dimension
            frame_tensor = frame_tensor.squeeze(0)
            had_batch_dim = True
        else:
            logger.error(f"Unexpected batch size: {frame_tensor.shape[0]}")
            return frame
    
    # Convert torch tensor to numpy array for OpenCV processing
    # Handle different tensor formats (CHW or HWC)
    if len(frame_tensor.shape) == 3:
        if frame_tensor.shape[0] == 3:  # CHW format (3, height, width)
            # Convert CHW to HWC for OpenCV
            img = frame_tensor.permute(1, 2, 0).cpu().numpy()
            was_chw = True
        else:  # HWC format (height, width, 3)
            img = frame_tensor.cpu().numpy()
            was_chw = False
    else:
        logger.error(f"Unexpected tensor shape after processing: {frame_tensor.shape}")
        return frame
    
    # Mirror the frame horizontally (flip left to right)
    mirrored = torch.flip(frame.tensor, dims=[1])
    
    # Accent Green target color (#18794E: rgb(24,121,78) -> (0.094,0.475,0.306))
    target = torch.tensor([0.094, 0.475, 0.306], device=mirrored.device)
    
    #simulate processing time that is adjustable
    await asyncio.sleep(delay)

    # Create tinted version
    tinted = mirrored.clone()
    for c in range(3):
        tinted[:, :, c] = torch.clamp(
            mirrored[:, :, c] + (target[c] - 0.5) * 0.4, 0, 1
        )
    
    # Add batch dimension back if it was originally present
    if had_batch_dim:
        result_tensor = result_tensor.unsqueeze(0)
    
    # Move to same device as original tensor
    result_tensor = result_tensor.to(frame.tensor.device)
    
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