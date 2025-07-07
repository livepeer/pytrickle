#!/usr/bin/env python3
"""
Simple Stream Processing Example

This example demonstrates how to use trickle-app to process video streams
with a simple frame transformation (adding a color tint).
"""

import asyncio
import logging
import torch
from trickle_app import SimpleTrickleClient
from trickle_app.frames import VideoFrame, VideoOutput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def red_tint_processor(frame: VideoFrame) -> VideoOutput:
    """Add a red tint to video frames."""
    # Clone the tensor to avoid modifying the original
    tensor = frame.tensor.clone()
    
    # Add red tint by increasing the red channel
    tensor[:, :, :, 0] = torch.clamp(tensor[:, :, :, 0] + 0.3, 0, 1)
    
    # Create new frame with processed tensor
    new_frame = frame.replace_tensor(tensor)
    return VideoOutput(new_frame, "red_tint_processor")

async def main():
    """Main function to run the stream processing."""
    # Configure trickle URLs (assumes trickle server is running on localhost:3389)
    subscribe_url = "http://localhost:3389/sample"
    publish_url = "http://localhost:3389/sample-output"
    
    # Create simple trickle client
    client = SimpleTrickleClient(subscribe_url, publish_url)
    
    logger.info("Starting stream processing with red tint...")
    logger.info(f"Subscribe URL: {subscribe_url}")
    logger.info(f"Publish URL: {publish_url}")
    
    try:
        # Process the stream with red tint
        await client.process_stream(
            frame_processor=red_tint_processor,
            request_id="simple_red_tint",
            width=704,
            height=384
        )
    except KeyboardInterrupt:
        logger.info("Stream processing interrupted by user")
    except Exception as e:
        logger.error(f"Error during stream processing: {e}")
    finally:
        await client.stop()
        logger.info("Stream processing stopped")

if __name__ == "__main__":
    asyncio.run(main()) 