#!/usr/bin/env python3
"""
Simple TrickleApp Example

This example demonstrates the simplest way to build an AI video processing
application with PyTrickle using the TrickleApp class. The app adds a red 
tint to video frames and provides an HTTP API for stream management.
"""

import asyncio
import logging
import torch
from pytrickle import TrickleApp, RegisterCapability
from pytrickle.frames import VideoFrame, AudioFrame, VideoOutput, AudioOutput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def red_tint_processor(frame):
    """Add a red tint to video frames, pass through audio unchanged."""
    if isinstance(frame, VideoFrame):
        # Clone the tensor to avoid modifying the original
        tensor = frame.tensor.clone()
        
        # Add red tint by increasing the red channel
        tensor[:, :, :, 0] = torch.clamp(tensor[:, :, :, 0] + 0.3, 0, 1)
        
        # Create new frame with processed tensor
        new_frame = frame.replace_tensor(tensor)
        return VideoOutput(new_frame, "red_tint_processor")
    
    elif isinstance(frame, AudioFrame):
        # Pass through audio unchanged
        return AudioOutput([frame], "audio_passthrough")

async def main():
    """Main function to run the TrickleApp server."""
    # Optionally register as a worker with orchestrator
    RegisterCapability.register(
        logger,
        capability_name="simple-red-tint-worker",
        capability_desc="Simple red tint video processor"
    )
    
    logger.info("Starting TrickleApp with red tint processor...")
    logger.info("The app will listen on port 8080 for stream requests")
    logger.info("")
    logger.info("To start processing, send a POST request to:")
    logger.info("http://localhost:8080/api/stream/start")
    logger.info("")
    logger.info("Example request:")
    logger.info('curl -X POST http://localhost:8080/api/stream/start \\')
    logger.info('  -H "Content-Type: application/json" \\')
    logger.info('  -d \'{"subscribe_url": "http://localhost:3389/sample", "publish_url": "http://localhost:3389/sample-output", "gateway_request_id": "red_tint_demo"}\'')
    logger.info("")
    
    try:
        # Create and run TrickleApp
        app = TrickleApp(
            frame_processor=red_tint_processor,
            port=8080
        )
        await app.run_forever()
    except KeyboardInterrupt:
        logger.info("TrickleApp shutdown requested by user")

if __name__ == "__main__":
    asyncio.run(main()) 