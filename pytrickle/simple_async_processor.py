"""
Simple Async Processor - Example implementation of AsyncFrameProcessor.

This module provides a ready-to-use example implementation of AsyncFrameProcessor
with basic video effects for demonstration and testing purposes.
"""

import asyncio
import logging
from typing import Optional, Dict, Any

from .async_processor import AsyncFrameProcessor
from .frames import VideoFrame, AudioFrame

logger = logging.getLogger(__name__)


class SimpleAsyncProcessor(AsyncFrameProcessor):
    """
    Simple example async processor for demonstration.
    
    This processor shows how to implement the AsyncFrameProcessor interface
    with basic video effects and audio pass-through. It's useful for:
    - Learning the AsyncFrameProcessor pattern
    - Testing async processing setups
    - Basic video effects demonstrations
    - Performance testing with configurable processing time
    """
    
    def __init__(
        self, 
        effect: str = "passthrough", 
        processing_time: float = 0.01,
        **kwargs
    ):
        """
        Initialize simple async processor.
        
        Args:
            effect: Video effect to apply (passthrough, red_tint, blue_tint, brightness)
            processing_time: Simulated processing time in seconds
            **kwargs: Additional arguments for AsyncFrameProcessor
        """
        super().__init__(**kwargs)
        self.effect = effect
        self.processing_time = processing_time
        
        logger.info(f"SimpleAsyncProcessor initialized with effect: {effect}")
    
    async def process_video_async(self, frame: VideoFrame) -> Optional[VideoFrame]:
        """
        Process video with the configured effect.
        
        Args:
            frame: Input video frame
            
        Returns:
            Processed video frame or None if processing failed
        """
        try:
            # Simulate processing time
            await asyncio.sleep(self.processing_time)
            
            tensor = frame.tensor.clone()
            
            if self.effect == "red_tint":
                # Enhance red channel
                tensor[:, :, :, 0] = tensor[:, :, :, 0].clamp(0, 1) * 1.2
                tensor = tensor.clamp(0, 1)
                
            elif self.effect == "blue_tint":
                # Enhance blue channel
                tensor[:, :, :, 2] = tensor[:, :, :, 2].clamp(0, 1) * 1.2
                tensor = tensor.clamp(0, 1)
                
            elif self.effect == "brightness":
                # Increase brightness
                tensor = (tensor + 0.1).clamp(0, 1)
                
            elif self.effect == "green_tint":
                # Enhance green channel
                tensor[:, :, :, 1] = tensor[:, :, :, 1].clamp(0, 1) * 1.2
                tensor = tensor.clamp(0, 1)
                
            elif self.effect == "contrast":
                # Increase contrast
                tensor = ((tensor - 0.5) * 1.3 + 0.5).clamp(0, 1)
                
            elif self.effect == "sepia":
                # Simple sepia effect
                # Convert to grayscale then tint
                gray = tensor.mean(dim=3, keepdim=True)
                tensor[:, :, :, 0] = (gray * 1.0).clamp(0, 1).squeeze(3)  # Red
                tensor[:, :, :, 1] = (gray * 0.8).clamp(0, 1).squeeze(3)  # Green
                tensor[:, :, :, 2] = (gray * 0.6).clamp(0, 1).squeeze(3)  # Blue
                
            # passthrough - no changes for any other effect
            
            return frame.replace_tensor(tensor)
        
        except Exception as e:
            logger.error(f"Error processing video frame: {e}")
            return None
    
    async def process_audio_async(self, frame: AudioFrame) -> Optional[list[AudioFrame]]:
        """
        Process audio frame (pass-through for this example).
        
        Args:
            frame: Input audio frame
            
        Returns:
            List containing the original audio frame
        """
        # For this simple processor, just pass audio through unchanged
        return [frame]
    
    def update_params(self, params: Dict[str, Any]):
        """
        Update processing parameters.
        
        Args:
            params: Dictionary of parameters to update
        """
        if 'effect' in params:
            old_effect = self.effect
            self.effect = params['effect']
            logger.info(f"Updated effect from '{old_effect}' to '{self.effect}'")
        
        if 'processing_time' in params:
            old_time = self.processing_time
            self.processing_time = float(params['processing_time'])
            logger.info(f"Updated processing time from {old_time}s to {self.processing_time}s")
    
    def get_available_effects(self) -> list[str]:
        """
        Get list of available video effects.
        
        Returns:
            List of available effect names
        """
        return [
            "passthrough",
            "red_tint", 
            "blue_tint",
            "green_tint",
            "brightness",
            "contrast",
            "sepia"
        ]
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get current processor configuration.
        
        Returns:
            Dictionary with current configuration
        """
        return {
            "effect": self.effect,
            "processing_time": self.processing_time,
            "available_effects": self.get_available_effects(),
            "is_started": self.is_started,
            "frame_count": self.frame_count
        }
