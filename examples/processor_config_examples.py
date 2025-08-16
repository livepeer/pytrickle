#!/usr/bin/env python3
"""
StreamProcessor Configuration Examples

This file demonstrates the different ways to configure StreamProcessor
with the new abstraction that properly handles processor dependencies.
"""

import logging
import torch
from pytrickle import (
    StreamProcessor, 
    VideoProcessorConfig, 
    AudioProcessorConfig, 
    AudioPassthrough
)
from pytrickle.frames import VideoFrame, AudioFrame
from typing import Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example processor functions
async def simple_video_processor(frame: VideoFrame) -> VideoFrame:
    """Simple video processor that adds a red tint."""
    tensor = frame.tensor
    if tensor is not None and tensor.dtype == torch.float32:
        # Add red tint
        if len(tensor.shape) >= 3:
            tensor[..., 0] = torch.clamp(tensor[..., 0] + 0.1, 0.0, 1.0)
        return frame.replace_tensor(tensor)
    return frame

async def simple_audio_processor(frame: AudioFrame) -> List[AudioFrame]:
    """Simple audio processor that applies a volume boost."""
    tensor = frame.tensor
    if tensor is not None:
        # Apply 10% volume boost
        boosted = tensor * 1.1
        return [frame.replace_tensor(boosted)]
    return [frame]

def load_model(**kwargs):
    """Model loader function."""
    logger.info("Model loaded successfully")

def update_params(params: dict):
    """Parameter updater function."""
    logger.info(f"Parameters updated: {params}")

def example_1_video_only_simple():
    """Example 1: Video processing only, simple function style."""
    logger.info("=== Example 1: Video Only (Simple) ===")
    
    processor = StreamProcessor(
        video_processor=simple_video_processor,  # Just pass the function
        audio_processor=AudioPassthrough(),      # Explicit passthrough
        model_loader=load_model,
        param_updater=update_params,
        name="video-only-simple",
        port=8001
    )
    return processor

def example_2_video_only_configured():
    """Example 2: Video processing only, with custom configuration."""
    logger.info("=== Example 2: Video Only (Configured) ===")
    
    processor = StreamProcessor(
        video_processor=VideoProcessorConfig(
            processor=simple_video_processor,
            queue_size=16,      # Larger queue for high throughput
            concurrency=2       # Multiple workers for parallel processing
        ),
        audio_processor=AudioPassthrough(),  # Explicit passthrough
        model_loader=load_model,
        param_updater=update_params,
        name="video-only-configured",
        port=8002
    )
    return processor

def example_3_audio_only():
    """Example 3: Audio processing only."""
    logger.info("=== Example 3: Audio Only ===")
    
    processor = StreamProcessor(
        video_processor=None,  # No video processing
        audio_processor=AudioProcessorConfig(
            processor=simple_audio_processor,
            queue_size=64,      # Larger queue for audio buffering
            concurrency=1       # Single worker for audio
        ),
        model_loader=load_model,
        param_updater=update_params,
        name="audio-only",
        port=8003
    )
    return processor

def example_4_both_processors():
    """Example 4: Both video and audio processing."""
    logger.info("=== Example 4: Both Video and Audio ===")
    
    processor = StreamProcessor(
        video_processor=VideoProcessorConfig(
            processor=simple_video_processor,
            queue_size=8,
            concurrency=1
        ),
        audio_processor=AudioProcessorConfig(
            processor=simple_audio_processor,
            queue_size=32,
            concurrency=1
        ),
        model_loader=load_model,
        param_updater=update_params,
        name="both-processors",
        port=8004
    )
    return processor

def example_5_backward_compatible():
    """Example 5: Backward compatible with old API style."""
    logger.info("=== Example 5: Backward Compatible ===")
    
    processor = StreamProcessor(
        video_processor=simple_video_processor,      # Function directly
        audio_processor=simple_audio_processor,      # Function directly  
        model_loader=load_model,
        param_updater=update_params,
        name="backward-compatible",
        port=8005
    )
    return processor

def example_6_high_performance():
    """Example 6: High performance configuration."""
    logger.info("=== Example 6: High Performance ===")
    
    processor = StreamProcessor(
        video_processor=VideoProcessorConfig(
            processor=simple_video_processor,
            queue_size=32,      # Large queue for buffering
            concurrency=4       # Multiple workers for parallel processing
        ),
        audio_processor=AudioProcessorConfig(
            processor=simple_audio_processor,
            queue_size=128,     # Very large audio buffer
            concurrency=2       # Dual audio workers
        ),
        model_loader=load_model,
        param_updater=update_params,
        name="high-performance",
        port=8006
    )
    return processor

if __name__ == "__main__":
    # Demonstrate all examples
    examples = [
        example_1_video_only_simple,
        example_2_video_only_configured,
        example_3_audio_only,
        example_4_both_processors,
        example_5_backward_compatible,
        example_6_high_performance
    ]
    
    print("StreamProcessor Configuration Examples")
    print("=" * 50)
    
    for example_func in examples:
        try:
            processor = example_func()
            print(f"✅ {example_func.__name__} - Created successfully")
            print(f"   Name: {processor.name}")
            print(f"   Port: {processor.port}")
            print()
        except Exception as e:
            print(f"❌ {example_func.__name__} - Failed: {e}")
            print()
    
    print("To run a specific example:")
    print("processor = example_1_video_only_simple()")
    print("processor.run()")
