#!/usr/bin/env python3
"""
Async Processor Example - Demonstrates AsyncFrameProcessor usage.

This example shows how to use the AsyncFrameProcessor to integrate
async AI models with PyTrickle applications.
"""

import asyncio
import logging
from typing import Optional
import torch

from pytrickle import (
    TrickleApp, 
    AsyncFrameProcessor, 
    SimpleAsyncProcessor,
    RegisterCapability
)
from pytrickle.frames import VideoFrame, AudioFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIVideoProcessor(AsyncFrameProcessor):
    """
    Example AI video processor that demonstrates async processing.
    
    This processor simulates an AI model that takes time to process
    each frame and shows how to handle async operations properly.
    """
    
    def __init__(
        self,
        model_name: str = "example_ai_model",
        processing_mode: str = "enhancement",
        ai_processing_time: float = 0.05,
        **kwargs
    ):
        """
        Initialize AI video processor.
        
        Args:
            model_name: Name of the AI model to simulate
            processing_mode: Type of processing (enhancement, style_transfer, etc.)
            ai_processing_time: Simulated AI processing time in seconds
            **kwargs: Additional arguments for AsyncFrameProcessor
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.processing_mode = processing_mode
        self.ai_processing_time = ai_processing_time
        self.model_loaded = False
        
        logger.info(f"AIVideoProcessor initialized: {model_name}")
    
    async def start(self):
        """Start the processor and load the AI model."""
        await super().start()
        
        # Simulate AI model loading
        if not self.model_loaded:
            logger.info(f"Loading AI model: {self.model_name}")
            await asyncio.sleep(1.0)  # Simulate model loading time
            self.model_loaded = True
            logger.info("AI model loaded successfully")
    
    async def process_video_async(self, frame: VideoFrame) -> Optional[VideoFrame]:
        """
        Process video frame with simulated AI model.
        
        Args:
            frame: Input video frame
            
        Returns:
            Processed video frame or None if processing failed
        """
        try:
            # Simulate AI model processing time
            await asyncio.sleep(self.ai_processing_time)
            
            # Apply processing based on mode
            tensor = frame.tensor.clone()
            
            if self.processing_mode == "enhancement":
                # Image enhancement - increase brightness and contrast
                tensor = tensor * 1.2
                tensor = tensor.clamp(0, 1)
                
            elif self.processing_mode == "style_transfer":
                # Style transfer simulation - add artistic effect
                # Apply a simple color transformation
                tensor[:, :, :, 0] = tensor[:, :, :, 0] * 1.3  # Enhance red
                tensor[:, :, :, 2] = tensor[:, :, :, 2] * 0.8  # Reduce blue
                tensor = tensor.clamp(0, 1)
                
            elif self.processing_mode == "edge_detection":
                # Edge detection simulation
                # Apply edge enhancement using simple difference
                edges_h = torch.abs(tensor[:, 1:, :, :] - tensor[:, :-1, :, :])
                edges_v = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :])
                
                # Pad to maintain original size
                edges_h = torch.cat([edges_h, edges_h[:, -1:, :, :]], dim=1)
                edges_v = torch.cat([edges_v, edges_v[:, :, -1:, :]], dim=2)
                
                # Combine with original
                tensor = tensor + (edges_h + edges_v) * 0.5
                tensor = tensor.clamp(0, 1)
            
            return frame.replace_tensor(tensor)
            
        except Exception as e:
            logger.error(f"Error in AI video processing: {e}")
            return None
    
    async def process_audio_async(self, frame: AudioFrame) -> Optional[list[AudioFrame]]:
        """
        Process audio frame (pass-through for this example).
        
        Args:
            frame: Input audio frame
            
        Returns:
            List containing the original audio frame
        """
        # For this example, we just pass audio through unchanged
        return [frame]
    
    def update_params(self, params):
        """
        Update processing parameters.
        
        Args:
            params: Dictionary of parameters to update
        """
        if 'processing_mode' in params:
            self.processing_mode = params['processing_mode']
            logger.info(f"Processing mode updated to: {self.processing_mode}")
        
        if 'ai_processing_time' in params:
            self.ai_processing_time = float(params['ai_processing_time'])
            logger.info(f"AI processing time updated to: {self.ai_processing_time}s")


async def example_with_custom_ai_processor():
    """Example using custom AI processor."""
    
    logger.info("=== Custom AI Processor Example ===")
    
    # Create AI processor
    ai_processor = AIVideoProcessor(
        model_name="advanced_video_ai",
        processing_mode="enhancement",
        ai_processing_time=0.03,  # 30ms simulated processing
        queue_maxsize=20
    )
    
    # Start the processor (loads AI model)
    await ai_processor.start()
    
    # Create TrickleApp with async processor (using new clean method)
    app = TrickleApp(
        frame_processor=ai_processor.create_sync_bridge(),
        port=8080
    )
    
    # Register with orchestrator (optional)
    RegisterCapability.register(
        logger,
        capability_name="ai-video-processor",
        capability_desc="AI-powered video enhancement processor"
    )
    
    logger.info("AI video processor ready!")
    logger.info("Processing mode: enhancement")
    logger.info("Simulated AI processing time: 30ms per frame")
    
    # In a real application, you would run the app
    # await app.run_forever()
    logger.info("Example completed (would run app.run_forever() in real usage)")
    
    # Cleanup
    await ai_processor.stop()


async def example_with_simple_processor():
    """Example using the built-in SimpleAsyncProcessor."""
    
    logger.info("=== Simple Async Processor Example ===")
    
    # Create simple processor with different effects
    processor = SimpleAsyncProcessor(
        effect="red_tint",
        processing_time=0.01,  # 10ms processing time
        queue_maxsize=30
    )
    
    # Start the processor
    await processor.start()
    
    # Create TrickleApp (using new clean method)
    app = TrickleApp(
        frame_processor=processor.create_sync_bridge(),
        port=8081
    )
    
    logger.info("Simple async processor ready!")
    logger.info("Effect: red_tint")
    logger.info("Processing time: 10ms per frame")
    
    # Cleanup
    await processor.stop()


async def main():
    """Main function to run examples."""
    
    try:
        # Run custom AI processor example
        await example_with_custom_ai_processor()
        
        # Run simple processor example
        await example_with_simple_processor()
        
    except KeyboardInterrupt:
        logger.info("Examples interrupted by user")
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    """
    Run async processor examples.
    
    This example demonstrates:
    1. Creating custom AsyncFrameProcessor subclasses
    2. Implementing async AI model integration
    3. Using the create_async_processor_bridge utility
    4. Integrating with TrickleApp
    5. Proper error handling and cleanup
    """
    
    asyncio.run(main())
