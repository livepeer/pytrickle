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
    if delay > 0:
        await asyncio.sleep(delay)

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
    mirrored = torch.flip(frame_tensor, dims=[-1])  # Flip width dimension
    
    # Accent Green target color (#18794E: rgb(24,121,78) -> (0.094,0.475,0.306))
    target = torch.tensor([0.094, 0.475, 0.306], device=mirrored.device)
    # Simulate processing time that is adjustable
    if delay > 0:
        await asyncio.sleep(delay)

    # Create tinted version using vectorized operations
    tinted = mirrored.clone()
    tint_adjustment = (target - 0.5) * intensity * 0.4
    
    if len(tinted.shape) == 3 and tinted.shape[-1] == 3:  # HWC format
        tinted = torch.clamp(mirrored + tint_adjustment, 0, 1)
    elif len(tinted.shape) == 3 and tinted.shape[0] == 3:  # CHW format
        tinted = torch.clamp(mirrored + tint_adjustment.view(3, 1, 1), 0, 1)
    
    # Restore original tensor properties
    if had_batch_dim:
        tinted = tinted.unsqueeze(0)
    tinted = tinted.to(frame.tensor.device)
    
    return frame.replace_tensor(tinted)

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