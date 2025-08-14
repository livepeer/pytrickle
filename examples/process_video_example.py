#!/usr/bin/env python3
"""
Accent Green Processor using StreamProcessor
"""

import logging
import torch
from pytrickle import StreamProcessor
from pytrickle.frames import VideoFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
intensity = 0.5
ready = False

def load_model(**kwargs):
    """Initialize processor state - called during model loading phase."""
    global intensity, ready
    
    # Set processor variables from kwargs or use defaults
    intensity = kwargs.get('intensity', 0.5)
    intensity = max(0.0, min(1.0, intensity))
    
    # Load the model here if needed
    # model = torch.load('my_model.pth')
    
    ready = True
    logger.info(f"✅ Accent Green processor ready (intensity: {intensity})")

def process_video(frame: VideoFrame) -> VideoFrame:
    """Apply Accent Green tinting to video frame."""
    global intensity, ready
    
    # Return frames unchanged if not ready
    if not ready:
        return frame
    
    # Accent Green target color (#18794E: rgb(24,121,78) -> (0.094,0.475,0.306))
    target = torch.tensor([0.094, 0.475, 0.306], device=frame.tensor.device)

    # Create tinted version
    tinted = frame.tensor.clone()
    for c in range(3):
        tinted[:, :, c] = torch.clamp(
            frame.tensor[:, :, c] + (target[c] - 0.5) * 0.4, 0, 1
        )
    
    # Blend based on intensity and return new frame
    blended = frame.tensor * (1.0 - intensity) + tinted * intensity
    return frame.replace_tensor(blended)

def update_params(params: dict):
    """Update tint intensity (0.0 to 1.0)."""
    global intensity
    if "intensity" in params:
        old = intensity
        intensity = max(0.0, min(1.0, float(params["intensity"])))
        if old != intensity:
            logger.info(f"Intensity: {old:.2f} → {intensity:.2f}")

# Create and run StreamProcessor
if __name__ == "__main__":
    processor = StreamProcessor(
        video_processor=process_video,
        model_loader=load_model,
        param_updater=update_params,
        name="accent-green-processor",
        port=8000
    )
    processor.run()