#!/usr/bin/env python3
"""
Audio and Video Processing Example

This example demonstrates how to process both audio and video frames
using the updated TrickleClient with audio processing support.
"""

import asyncio
import logging
import torch
import numpy as np
from pytrickle import SimpleTrickleClient
from pytrickle.frames import VideoFrame, VideoOutput, AudioFrame, AudioOutput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def red_tint_video_processor(frame: VideoFrame) -> VideoOutput:
    """Add a red tint to video frames."""
    tensor = frame.tensor.clone()
    tensor[:, :, :, 0] = torch.clamp(tensor[:, :, :, 0] + 0.3, 0, 1)
    new_frame = frame.replace_tensor(tensor)
    return VideoOutput(new_frame, "red_tint")

def volume_boost_audio_processor(frame: AudioFrame) -> AudioOutput:
    """Boost audio volume by 50%."""
    # Clone the samples to avoid modifying the original
    new_samples = frame.samples.copy().astype(np.float32)
    
    # Apply volume scaling
    volume_factor = 1.5
    new_samples = new_samples * volume_factor
    
    # Clip to prevent overflow based on format
    if frame.format in ['s16', 's16p']:
        new_samples = np.clip(new_samples, -32768, 32767)
        new_samples = new_samples.astype(np.int16)
    elif frame.format in ['s32', 's32p']:
        new_samples = np.clip(new_samples, -2147483648, 2147483647)
        new_samples = new_samples.astype(np.int32)
    elif frame.format in ['flt', 'fltp']:
        new_samples = np.clip(new_samples, -1.0, 1.0)
        new_samples = new_samples.astype(np.float32)
    else:
        # For other formats, just convert back to original dtype
        new_samples = new_samples.astype(frame.samples.dtype)
    
    # Create new frame with modified samples
    new_frame = frame.replace_samples(new_samples)
    
    return AudioOutput([new_frame], "volume_boost")

def silence_audio_processor(frame: AudioFrame) -> AudioOutput:
    """Replace audio with silence."""
    # Create silent samples of the same shape and dtype
    silent_samples = np.zeros_like(frame.samples)
    new_frame = frame.replace_samples(silent_samples)
    return AudioOutput([new_frame], "silenced")

async def main():
    """Main function to run the audio/video processing example."""
    
    # You can choose different audio processors
    audio_processors = {
        "volume_boost": volume_boost_audio_processor,
        "silence": silence_audio_processor,
        "passthrough": lambda frame: AudioOutput([frame], "passthrough")
    }
    
    # Select which audio processor to use
    selected_audio_effect = "volume_boost"  # Change this to test different effects
    
    logger.info(f"Starting audio/video processing with audio effect: {selected_audio_effect}")
    
    client = SimpleTrickleClient(
        subscribe_url="http://localhost:3389/sample",
        publish_url="http://localhost:3389/sample-output"
    )
    
    try:
        await client.process_stream(
            frame_processor=red_tint_video_processor,
            audio_processor=audio_processors[selected_audio_effect],
            request_id="audio_video_example",
            width=704,
            height=384
        )
    except KeyboardInterrupt:
        logger.info("Received interrupt, stopping...")
    except Exception as e:
        logger.error(f"Error in processing: {e}")
    finally:
        await client.stop()

if __name__ == "__main__":
    asyncio.run(main())
