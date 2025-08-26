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
intensity = 0.8
delay = 0.0
ready = False
processor = None
background_tasks = []
background_task_started = False

def load_model(processor_ref, **kwargs):
    """Initialize processor state - called during model loading phase."""
    global intensity, ready, processor
    
    logger.info(f"load_model called with kwargs: {kwargs}")
    
    # Store processor reference for later use
    processor = processor_ref
    
    # Set processor variables from kwargs or use defaults
    intensity = kwargs.get('intensity', 0.5)
    intensity = max(0.0, min(1.0, intensity))
    
    # Load the model here if needed
    # model = torch.load('my_model.pth')
    
    # Note: Cannot start background tasks here as event loop isn't running yet
    # Background task will be started when first frame is processed
    
    ready = True
    logger.info(f"✅ OpenCV Green processor with horizontal flip ready (intensity: {intensity}, ready: {ready})")

def start_background_task():
    """Start the background task if not already started."""
    global background_task_started, background_tasks
    
    if not background_task_started and processor:
        task = asyncio.create_task(send_periodic_status())
        background_tasks.append(task)
        background_task_started = True
        logger.info("Started background status task")

async def send_periodic_status():
    """Background task that sends periodic status updates."""
    global processor
    counter = 0
    try:
        while True:
            await asyncio.sleep(5.0)  # Send status every 5 seconds
            counter += 1
            if processor:
                status_data = {
                    "type": "status_update",
                    "counter": counter,
                    "intensity": intensity,
                    "ready": ready,
                    "timestamp": time.time()
                }
                success = await processor.send_data(str(status_data))
                if success:
                    logger.info(f"Sent status update #{counter}")
                else:
                    logger.warning(f"Failed to send status update #{counter}, stopping background task")
                    break  # Exit the loop if sending fails
    except asyncio.CancelledError:
        logger.info("Background status task cancelled")
        raise
    except Exception as e:
        logger.error(f"Error in background status task: {e}")

def on_stream_stop():
    """Called when stream stops - cleanup background tasks."""
    global background_tasks, background_task_started
    logger.info("🛑 Stream stopped, cleaning up background tasks")
    
    for task in background_tasks:
        if not task.done():
            task.cancel()
            logger.info("Cancelled background task")
    
    background_tasks.clear()
    background_task_started = False  # Reset flag for next stream
    logger.info("✅ All background tasks cleaned up")

async def process_video(frame: VideoFrame) -> VideoFrame:
    """Apply horizontal flip and green hue using OpenCV."""
    global intensity, ready, delay
    
    # Start background task on first frame (when event loop is running)
    start_background_task()
    
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
    
    # Ensure the image is in the correct range [0, 255] for OpenCV
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
        was_normalized = True
    else:
        img = img.astype(np.uint8)
        was_normalized = False
    
    # Apply horizontal flip using OpenCV
    img_flipped = cv2.flip(img, 1)  # 1 = horizontal flip
    
    # Add green hue by enhancing the green channel
    # Convert to HSV for better color manipulation
    img_hsv = cv2.cvtColor(img_flipped, cv2.COLOR_RGB2HSV)
    
    # Enhance green hue (hue value around 60 degrees for green in OpenCV HSV)
    # Adjust the hue towards green and increase saturation
    green_hue = 60  # Green hue in HSV
    hue_shift = intensity * 30  # Maximum hue shift of 30 degrees towards green
    
    # Shift hue towards green
    img_hsv[:, :, 0] = ((img_hsv[:, :, 0] + hue_shift) % 180).astype(np.uint8)
    
    # Increase saturation to make the green more vibrant
    saturation_boost = intensity * 50  # Boost saturation by up to 50
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] + saturation_boost, 0, 255).astype(np.uint8)
    
    # Convert back to RGB
    img_green = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    
    # Convert back to torch tensor
    if was_normalized:
        img_result = img_green.astype(np.float32) / 255.0
    else:
        img_result = img_green.astype(np.float32)
    
    # Convert back to original tensor format
    if was_chw:
        # Convert HWC back to CHW
        result_tensor = torch.from_numpy(img_result).permute(2, 0, 1)
    else:
        result_tensor = torch.from_numpy(img_result)
    
    # Add batch dimension back if it was originally present
    if had_batch_dim:
        result_tensor = result_tensor.unsqueeze(0)
    
    # Move to same device as original tensor
    result_tensor = result_tensor.to(frame.tensor.device)
    
    return frame.replace_tensor(result_tensor)

def update_params(params: dict):
    """Update green hue intensity (0.0 to 1.0)."""
    global intensity, delay
    if "intensity" in params:
        old = intensity
        intensity = max(0.0, min(1.0, float(params["intensity"])))
        if old != intensity:
            logger.info(f"Green hue intensity: {old:.2f} → {intensity:.2f}")
    if "delay" in params:
        old = delay
        delay = max(0.0, float(params["delay"]))
        if old != delay:
            logger.info(f"Processing delay: {old:.2f} → {delay:.2f}")

# Create and run StreamProcessor
if __name__ == "__main__":
    processor = StreamProcessor(
        video_processor=process_video,
        model_loader=load_model,
        param_updater=update_params,
        on_stream_stop=on_stream_stop,
        name="green-processor",
        port=8000
    )
    processor.run()