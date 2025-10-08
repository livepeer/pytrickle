#!/usr/bin/env python3
"""
Loading Overlay Processor using StreamProcessor

This example demonstrates how to create loading overlay frames
that can be toggled via parameter updates.

When the 'show_loading' parameter is True, video frames will be replaced with
animated loading overlay frames instead of showing the original video.
"""

import asyncio
import logging
import time
import torch
from pytrickle import StreamProcessor
from pytrickle.frames import VideoFrame
from pytrickle.frame_skipper import FrameSkipConfig
import numpy as np
from pytrickle.video_utils import create_loading_frame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
show_loading = False
loading_message = "Loading..."
ready = False
processor = None
background_tasks = []
background_task_started = False

# Loading animation state
frame_counter = 0

async def load_model(**kwargs):
    """Initialize processor state - called during model loading phase."""
    global show_loading, ready, processor
    
    logger.info(f"load_model called with kwargs: {kwargs}")
    
    # Set processor variables from kwargs or use defaults
    show_loading = kwargs.get('show_loading', False)
    
    # Simulate model loading time
    if kwargs.get('simulate_loading', False):
        logger.info("Simulating model loading...")
        await asyncio.sleep(2)  # Simulate loading time
    
    # Note: Cannot start background tasks here as event loop isn't running yet
    # Background task will be started when first frame is processed
    ready = True
    logger.info(f"✅ Loading Overlay processor ready (show_loading: {show_loading}, ready: {ready})")

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
                    "show_loading": show_loading,
                    "loading_message": loading_message,
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

async def on_stream_stop():
    """Called when stream stops - cleanup background tasks."""
    global background_tasks, background_task_started
    logger.info("Stream stopped, cleaning up background tasks")
    
    for task in background_tasks:
        if not task.done():
            task.cancel()
            logger.info("Cancelled background task")
    
    background_tasks.clear()
    background_task_started = False  # Reset flag for next stream
    logger.info("All background tasks cleaned up")

async def process_video(frame: VideoFrame) -> VideoFrame:
    """Process video frame - show loading overlay if enabled, otherwise passthrough."""
    global show_loading, ready, frame_counter, loading_message
    
    # Start background task on first frame (when event loop is running)
    start_background_task()
    
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
    
    # Create loading overlay frame using utility
    loading_frame = create_loading_frame(width, height, loading_message, frame_counter)
    
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

async def update_params(params: dict):
    """Update loading overlay parameters."""
    global show_loading, loading_message
    
    if "show_loading" in params:
        old = show_loading
        show_loading = bool(params["show_loading"])
        if old != show_loading:
            status = "enabled" if show_loading else "disabled"
            logger.info(f"Loading overlay: {status}")
    
    if "loading_message" in params:
        old = loading_message
        loading_message = str(params["loading_message"])
        if old != loading_message:
            logger.info(f"Loading message: {old} → {loading_message}")

# Create and run StreamProcessor
if __name__ == "__main__":
    logger.info("Starting Loading Overlay Processor")
    logger.info("Parameters you can update:")
    logger.info("  - show_loading: true/false - Enable/disable loading overlay")
    logger.info("  - loading_message: string - Custom loading message")
    logger.info("")
    logger.info("Example parameter updates:")
    logger.info('  {"show_loading": true, "loading_message": "Processing video..."}')
    logger.info('  {"show_loading": false}')
    
    processor = StreamProcessor(
        video_processor=process_video,
        model_loader=load_model,
        param_updater=update_params,
        on_stream_stop=on_stream_stop,
        name="loading-overlay",
        port=8001,  # Different port from process_video_example
        frame_skip_config=FrameSkipConfig(),
    )
    processor.run()