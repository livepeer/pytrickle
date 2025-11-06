#!/usr/bin/env python3
"""OpenCV Green Processor using decorator-based handlers.

This example demonstrates:
- Video processing with OpenCV (horizontal flip + green hue)
- Background tasks (periodic status updates)
- Parameter updates at runtime
- Proper lifecycle management with decorators

Run with:
    python -m pytrickle.examples.process_video_example
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import torch

from pytrickle import StreamProcessor, VideoFrame
from pytrickle.decorators import (
    model_loader,
    on_stream_start,
    on_stream_stop,
    param_updater,
    video_handler,
)
from pytrickle.frame_skipper import FrameSkipConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class GreenProcessorConfig:
    """Configuration for the green processor."""
    intensity: float = 0.8
    delay: float = 0.0
    ready: bool = False


class GreenProcessorHandlers:
    """Handlers for applying green hue and horizontal flip with background tasks."""

    def __init__(self) -> None:
        self.cfg = GreenProcessorConfig()
        self.processor = None
        self.background_tasks: List[asyncio.Task] = []
        self.background_task_started = False

    @model_loader
    async def load(self, **kwargs: dict) -> None:
        """Initialize processor state - called during model loading phase."""
        logger.info(f"Model loader called with kwargs: {kwargs}")
        
        # Set processor variables from kwargs or use defaults
        self.cfg.intensity = kwargs.get('intensity', 0.5)
        self.cfg.intensity = max(0.0, min(1.0, self.cfg.intensity))
        
        # Load the model here if needed
        # self.model = torch.load('my_model.pth')
        
        self.cfg.ready = True
        logger.info(f"✅ OpenCV Green processor ready (intensity: {self.cfg.intensity})")

    def _start_background_task(self) -> None:
        """Start the background task if not already started."""
        if not self.background_task_started and self.processor:
            task = asyncio.create_task(self._send_periodic_status())
            self.background_tasks.append(task)
            self.background_task_started = True
            logger.info("Started background status task")

    async def _send_periodic_status(self) -> None:
        """Background task that sends periodic status updates."""
        counter = 0
        try:
            while True:
                await asyncio.sleep(5.0)  # Send status every 5 seconds
                counter += 1
                if self.processor:
                    status_data = {
                        "type": "status_update",
                        "counter": counter,
                        "intensity": self.cfg.intensity,
                        "ready": self.cfg.ready,
                        "timestamp": time.time()
                    }
                    success = await self.processor.send_data(str(status_data))
                    if success:
                        logger.info(f"Sent status update #{counter}")
                    else:
                        logger.warning(f"Failed to send status update #{counter}, stopping")
                        break
        except asyncio.CancelledError:
            logger.info("Background status task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in background status task: {e}")

    @on_stream_start
    async def on_start(self) -> None:
        """Called when stream starts - initialize resources."""
        logger.info("Stream started, initializing resources")
        self.background_task_started = False
        logger.info("Stream initialization complete")

    @on_stream_stop
    async def on_stop(self) -> None:
        """Called when stream stops - cleanup background tasks."""
        logger.info("Stream stopped, cleaning up background tasks")
        
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
                logger.info("Cancelled background task")
        
        self.background_tasks.clear()
        self.background_task_started = False
        logger.info("All background tasks cleaned up")

    @video_handler
    async def handle_video(self, frame: VideoFrame) -> VideoFrame:
        """Apply horizontal flip and green hue using OpenCV."""
        # Start background task on first frame (when event loop is running)
        self._start_background_task()
        
        # Simulated processing time
        if self.cfg.delay > 0:
            await asyncio.sleep(self.cfg.delay)

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
        hue_shift = self.cfg.intensity * 30  # Maximum hue shift of 30 degrees towards green
        
        # Shift hue towards green
        img_hsv[:, :, 0] = ((img_hsv[:, :, 0] + hue_shift) % 180).astype(np.uint8)
        
        # Increase saturation to make the green more vibrant
        saturation_boost = self.cfg.intensity * 50  # Boost saturation by up to 50
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

    @param_updater
    async def update_params(self, params: dict) -> None:
        """Update green hue intensity (0.0 to 1.0) and processing delay."""
        if "intensity" in params:
            old = self.cfg.intensity
            self.cfg.intensity = max(0.0, min(1.0, float(params["intensity"])))
            if old != self.cfg.intensity:
                logger.info(f"Green hue intensity: {old:.2f} → {self.cfg.intensity:.2f}")
        if "delay" in params:
            old = self.cfg.delay
            self.cfg.delay = max(0.0, float(params["delay"]))
            if old != self.cfg.delay:
                logger.info(f"Processing delay: {old:.2f} → {self.cfg.delay:.2f}")


# Standalone functions for backward compatibility with tests
async def load_model(**kwargs: dict) -> None:
    """Standalone model loader for testing."""
    pass  # Tests don't need actual model loading


async def process_video(frame: VideoFrame) -> VideoFrame:
    """Standalone video processor for testing - simple passthrough."""
    return frame


async def update_params(params: dict) -> None:
    """Standalone param updater for testing."""
    pass  # Tests don't need actual param updates


async def main() -> None:
    """Main entry point - creates and runs the stream processor."""
    handlers = GreenProcessorHandlers()
    processor = StreamProcessor.from_handlers(
        handlers,
        name="green-processor",
        port=8000,
        frame_skip_config=FrameSkipConfig(),  # Optional frame skipping
    )
    
    # Store processor reference for background tasks
    handlers.processor = processor
    
    logger.info("🚀 Green processor started on port 8000")
    logger.info("   OpenCV will apply: horizontal flip + green hue")
    logger.info("   Update params: POST http://localhost:8000/control")
    
    await processor.run_forever()


if __name__ == "__main__":
    asyncio.run(main())