#!/usr/bin/env python3
"""
Video/Audio Passthrough with Text Publishing using StreamProcessor

This example demonstrates:
- Video passthrough (no processing)
- Audio passthrough (no processing)
- Text publishing every 400 audio frames using simple text queue
"""

import logging
import json
import time
from pytrickle import StreamProcessor
from pytrickle.frames import VideoFrame, AudioFrame
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
audio_frame_count = 0
text_publish_interval = 400
ready = False
start_time = None
_stream_processor = None  # Reference to StreamProcessor for text publishing

def load_model(**kwargs):
    """Initialize processor state - called during model loading phase."""
    global text_publish_interval, ready, start_time
    
    logger.info(f"load_model called with kwargs: {kwargs}")
    
    # Set processor variables from kwargs or use defaults
    text_publish_interval = kwargs.get('text_publish_interval', 400)
    text_publish_interval = max(1, int(text_publish_interval))
    
    ready = True
    start_time = time.time()
    logger.info(f"✅ Video/Audio passthrough with text publishing ready (interval: {text_publish_interval} frames)")

async def process_video(frame: VideoFrame) -> VideoFrame:
    """Pass through video frames unchanged."""
    global ready
    
    if not ready:
        return frame
    
    # Simply pass through the video frame without any processing
    return frame

async def process_audio(frame: AudioFrame) -> List[AudioFrame]:
    """Pass through audio frames and publish text data periodically."""
    global audio_frame_count, text_publish_interval, ready, start_time, _stream_processor
    
    if not ready:
        return [frame]
    
    # Increment frame counter
    audio_frame_count += 1
    
    # Check if we should publish text data
    if audio_frame_count % text_publish_interval == 0:
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Create JSONL data with audio processing statistics
        text_data = {
            "type": "audio_stats",
            "timestamp": time.time(),
            "elapsed_time_seconds": round(elapsed_time, 2),
            "total_audio_frames": audio_frame_count,
            "frames_per_second": round(audio_frame_count / elapsed_time, 2) if elapsed_time > 0 else 0,
            "frame_shape": list(frame.samples.shape) if hasattr(frame, 'samples') else None,
            "sample_rate": getattr(frame, 'sample_rate', None),
            "channels": getattr(frame, 'channels', None),
            "message": f"Processed {audio_frame_count} audio frames in {elapsed_time:.2f} seconds"
        }
        
        # Publish as JSONL - just add text to the queue!
        jsonl_line = json.dumps(text_data)
        if _stream_processor:
            await _stream_processor.publish_data_output(jsonl_line)
        
        logger.info(f"📊 Published stats: {audio_frame_count} frames, {elapsed_time:.2f}s elapsed")
    
    # Pass through the audio frame unchanged
    return [frame]

def update_params(params: dict):
    """Update text publishing interval."""
    global text_publish_interval
    
    if "text_publish_interval" in params:
        old = text_publish_interval
        text_publish_interval = max(1, int(params["text_publish_interval"]))
        if old != text_publish_interval:
            logger.info(f"Text publish interval: {old} → {text_publish_interval} frames")
    
    if "reset_counter" in params and params["reset_counter"]:
        global audio_frame_count, start_time
        audio_frame_count = 0
        start_time = time.time()
        logger.info("🔄 Reset audio frame counter and timer")


# Create and run StreamProcessor
if __name__ == "__main__":
    processor = StreamProcessor(
        video_processor=process_video,
        audio_processor=process_audio,
        model_loader=load_model,
        param_updater=update_params,
        name="passthrough-with-text",
        port=8000
    )
    
    # Set reference for text publishing
    _stream_processor = processor
    
    logger.info("🚀 Starting passthrough processor with text publishing...")
    logger.info(f"📝 Will publish JSONL stats every {text_publish_interval} audio frames")
    logger.info("🔧 Update parameters via /api/update_params:")
    logger.info("   - text_publish_interval: number of frames between text publications")
    logger.info("   - reset_counter: true to reset frame counter and timer")
    
    processor.run()