#!/usr/bin/env python3
"""
HTTP Server Example

This example demonstrates how to run the pytrickle HTTP server with
custom frame processing logic.
"""

import asyncio
import logging
import torch
import json
from pytrickle import RegisterCapability, TrickleApp
from pytrickle.frames import VideoFrame, VideoOutput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFrameProcessor:
    """Advanced frame processor with configurable effects."""
    
    def __init__(self):
        self.current_params = {
            "effect": "none",
            "intensity": 0.5,
            "color_shift": [0.0, 0.0, 0.0]
        }
        self.request_id = "default"
    
    def update_params(self, params):
        """Update processing parameters."""
        self.current_params.update(params)
        logger.info(f"Updated processing params: {self.current_params}")
    
    def process_frame(self, frame: VideoFrame) -> VideoOutput:
        """Process frame with current parameters."""
        tensor = frame.tensor.clone()
        
        effect = self.current_params.get("effect", "none")
        intensity = self.current_params.get("intensity", 0.5)
        
        if effect == "red_tint":
            # Add red tint
            tensor[:, :, :, 0] = torch.clamp(tensor[:, :, :, 0] + intensity, 0, 1)
        
        elif effect == "blue_tint":
            # Add blue tint
            tensor[:, :, :, 2] = torch.clamp(tensor[:, :, :, 2] + intensity, 0, 1)
        
        elif effect == "green_tint":
            # Add green tint
            tensor[:, :, :, 1] = torch.clamp(tensor[:, :, :, 1] + intensity, 0, 1)
        
        elif effect == "grayscale":
            # Convert to grayscale
            gray = 0.299 * tensor[:, :, :, 0] + 0.587 * tensor[:, :, :, 1] + 0.114 * tensor[:, :, :, 2]
            tensor[:, :, :, 0] = gray
            tensor[:, :, :, 1] = gray
            tensor[:, :, :, 2] = gray
        
        elif effect == "invert":
            # Invert colors
            tensor = 1.0 - tensor
        
        elif effect == "brightness":
            # Adjust brightness
            tensor = torch.clamp(tensor + (intensity - 0.5), 0, 1)
        
        elif effect == "color_shift":
            # Apply color shift
            color_shift = self.current_params.get("color_shift", [0.0, 0.0, 0.0])
            for i, shift in enumerate(color_shift):
                if i < 3:
                    tensor[:, :, :, i] = torch.clamp(tensor[:, :, :, i] + shift, 0, 1)
        
        # Create new frame with processed tensor
        new_frame = frame.replace_tensor(tensor)
        return VideoOutput(new_frame, self.request_id)

# Global processor instance
processor = AdvancedFrameProcessor()

def custom_frame_processor(frame: VideoFrame) -> VideoOutput:
    """Wrapper function for the frame processor."""
    return processor.process_frame(frame)

async def main():
    """Main function to run the HTTP server."""
    # Optionally register as a worker with orchestrator
    RegisterCapability.register(
        logger,
        capability_name="pytrickle-example-server",
        capability_desc="PyTrickle HTTP server example with video effects"
    )
    
    # Create trickle app with custom frame processor
    app = TrickleApp(frame_processor=custom_frame_processor)
    
    logger.info("Starting pytrickle HTTP server on port 8080")
    logger.info("Available effects: red_tint, blue_tint, green_tint, grayscale, invert, brightness, color_shift")
    logger.info("")
    logger.info("API Endpoints:")
    logger.info("  POST /api/stream/start - Start a new stream")
    logger.info("  POST /api/stream/stop - Stop the current stream")
    logger.info("  POST /api/stream/params - Update processing parameters")
    logger.info("  GET /api/stream/status - Get stream status")
    logger.info("  GET /health - Health check")
    logger.info("")
    logger.info("Example curl commands:")
    logger.info('  curl -X POST http://localhost:8080/api/stream/start \\')
    logger.info('    -H "Content-Type: application/json" \\')
    logger.info('    -d \'{"subscribe_url": "http://localhost:3389/sample", "publish_url": "http://localhost:3389/sample-output", "gateway_request_id": "example", "params": {"width": 704, "height": 384, "effect": "red_tint", "intensity": 0.3}}\'')
    logger.info("")
    logger.info("  # Optional: Add control and events URLs")
    logger.info('  curl -X POST http://localhost:8080/api/stream/start \\')
    logger.info('    -H "Content-Type: application/json" \\')
    logger.info('    -d \'{"subscribe_url": "http://localhost:3389/sample", "publish_url": "http://localhost:3389/sample-output", "control_url": "http://localhost:3389/control", "events_url": "http://localhost:3389/events", "gateway_request_id": "example_with_control", "params": {"width": 704, "height": 384, "effect": "green_tint"}}\'')
    logger.info("")
    logger.info('  curl -X POST http://localhost:8080/api/stream/params \\')
    logger.info('    -H "Content-Type: application/json" \\')
    logger.info('    -d \'{"effect": "blue_tint", "intensity": 0.5}\'')
    logger.info("")
    
    try:
        # Run the server forever
        await app.run_forever()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")

if __name__ == "__main__":
    asyncio.run(main()) 