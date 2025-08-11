#!/usr/bin/env python3
"""
Simple Async Example - Basic usage of SimpleAsyncProcessor.

This example demonstrates the simplest way to use async processing
with PyTrickle using the built-in SimpleAsyncProcessor.
"""

import asyncio
import logging

from pytrickle import (
    TrickleApp,
    SimpleAsyncProcessor,
    RegisterCapability
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main function demonstrating SimpleAsyncProcessor usage."""
    
    logger.info("=== Simple Async Processor Example ===")
    
    # Create a simple async processor with a red tint effect
    processor = SimpleAsyncProcessor(
        effect="red_tint",
        processing_time=0.02,  # 20ms simulated processing time
        queue_maxsize=25
    )
    
    # Start the async processor
    await processor.start()
    
    logger.info(f"Processor started with configuration:")
    config = processor.get_current_config()
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Create TrickleApp with the async processor
    app = TrickleApp(
        frame_processor=processor.create_sync_bridge(),
        port=8080
    )
    
    # Optional: Register with orchestrator
    RegisterCapability.register(
        logger,
        capability_name="simple-video-processor",
        capability_desc="Simple async video processor with configurable effects"
    )
    
    logger.info("Simple async processor ready!")
    logger.info("Available endpoints:")
    logger.info("  POST /api/stream/start - Start video processing")
    logger.info("  POST /api/stream/params - Update effect parameters")
    logger.info("  GET /api/stream/status - Get processing status")
    logger.info("  POST /api/stream/stop - Stop processing")
    logger.info("")
    logger.info("Example parameter update:")
    logger.info("  curl -X POST http://localhost:8080/api/stream/params \\")
    logger.info("       -H 'Content-Type: application/json' \\")
    logger.info("       -d '{\"effect\": \"blue_tint\", \"processing_time\": 0.05}'")
    
    try:
        # Run the application
        # In a real scenario, this would run forever
        # For this example, we'll just simulate running briefly
        
        logger.info("Starting TrickleApp server...")
        # await app.run_forever()
        
        # For demo purposes, let's show how to update parameters
        logger.info("\nDemonstrating parameter updates:")
        
        effects_to_try = ["blue_tint", "brightness", "sepia", "contrast", "passthrough"]
        
        for effect in effects_to_try:
            logger.info(f"Setting effect to: {effect}")
            processor.update_params({"effect": effect})
            await asyncio.sleep(1)  # Simulate some processing time
        
        logger.info("Example completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Example interrupted by user")
    
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        await processor.stop()
        logger.info("Processor stopped")


if __name__ == "__main__":
    """
    Run the simple async processor example.
    
    This example shows:
    1. Creating a SimpleAsyncProcessor with basic video effects
    2. Starting the async processor
    3. Creating a TrickleApp with the processor
    4. Updating parameters dynamically
    5. Proper cleanup
    
    Usage:
        python simple_async_example.py
    """
    
    asyncio.run(main())
