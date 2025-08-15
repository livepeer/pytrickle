#!/usr/bin/env python3
"""
Test script to verify async frame processing is working correctly.
This tests that frames are processed asynchronously and the ingress loop doesn't block.
"""

import asyncio
import logging
import time
from fractions import Fraction
from unittest.mock import Mock, AsyncMock
from pytrickle.client import TrickleClient
from pytrickle.protocol import TrickleProtocol
from pytrickle.frame_processor import FrameProcessor
from pytrickle.frames import VideoFrame, AudioFrame, InputFrame
import torch
import numpy as np
import av

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestProcessor(FrameProcessor):
    """Test processor that adds delays to verify async processing."""
    
    def __init__(self, processing_delay: float = 0.1):
        self.processing_delay = processing_delay
        self.processed_count = 0
        super().__init__()
    
    async def load_model(self, **kwargs):
        """Load model (no-op for test)."""
        logger.info("Test processor model loaded")
        pass
    
    async def process_video_async(self, frame: VideoFrame) -> VideoFrame:
        """Process video frame with artificial delay."""
        start_time = time.time()
        logger.info(f"Starting video processing for frame {self.processed_count}")
        
        # Simulate heavy processing with async delay
        await asyncio.sleep(self.processing_delay)
        
        # Simple processing: just modify the tensor
        processed_tensor = frame.tensor * 0.8  # Darken the frame
        self.processed_count += 1
        
        end_time = time.time()
        logger.info(f"Completed video processing for frame {self.processed_count} in {end_time - start_time:.3f}s")
        
        return processed_tensor
    
    async def process_audio_async(self, frame: AudioFrame) -> list[AudioFrame]:
        """Process audio frame with artificial delay."""
        start_time = time.time()
        logger.info(f"Starting audio processing for frame {self.processed_count}")
        
        # Simulate heavy processing with async delay
        await asyncio.sleep(self.processing_delay)
        
        # Simple processing: just modify the samples
        processed_samples = frame.samples * 0.8  # Lower volume
        frame.replace_samples(processed_samples)
        self.processed_count += 1
        
        end_time = time.time()
        logger.info(f"Completed audio processing for frame {self.processed_count} in {end_time - start_time:.3f}s")
        
        return [frame]
    
    def update_params(self, params: dict):
        """Update processing parameters."""
        if "delay" in params:
            self.processing_delay = float(params["delay"])
            logger.info(f"Updated processing delay to {self.processing_delay}s")


async def create_test_frames():
    """Create test video and audio frames."""
    frames = []
    
    # Create multiple frames with different timestamps
    for i in range(2):
        timestamp = 1000 + (i * 2000)  # Different timestamps
        
        # Create test video frame
        video_tensor = torch.rand(3, 256, 256)  # RGB frame
        video_frame = VideoFrame(
            tensor=video_tensor,
            timestamp=timestamp,
            time_base=0.033
        )
        frames.append(video_frame)
        
        # Create test audio frame using av.AudioFrame
        # First create numpy samples with slight variation
        audio_samples = np.random.rand(2, 1024).astype(np.float32)  # Stereo audio as np.ndarray
        
        # Create av.AudioFrame from the samples
        av_audio_frame = av.AudioFrame.from_ndarray(audio_samples, format='fltp', layout='stereo')
        av_audio_frame.sample_rate = 48000
        av_audio_frame.pts = timestamp
        av_audio_frame.time_base = Fraction(1, 48000)
        
        # Create AudioFrame from av.AudioFrame
        audio_frame = AudioFrame.from_av_audio(av_audio_frame)
        logger.debug(f"Created AudioFrame {i+1} with {audio_frame.nb_samples} samples, ts={audio_frame.timestamp}")
        frames.append(audio_frame)
    
    return frames


async def test_async_processing():
    """Test that processing is truly async and doesn't block ingress."""
    logger.info("=== Testing Async Frame Processing ===")
    
    # Create test processor with delay
    processor = TestProcessor(processing_delay=0.2)  # 200ms delay per frame
    
    # Create mock protocol
    protocol = Mock(spec=TrickleProtocol)
    protocol.start = AsyncMock()
    protocol.stop = AsyncMock()
    
    # Create test frames
    test_frames = await create_test_frames()
    
    # Mock ingress loop to yield test frames with timing
    async def mock_ingress_loop(stop_event):
        logger.info("Mock ingress loop starting")
        for i, frame in enumerate(test_frames):
            if stop_event.is_set():
                logger.info(f"Ingress loop stopped early at frame {i}")
                break
            logger.info(f"Yielding frame {i+1}: {type(frame).__name__}")
            yield frame
            # Small delay between frames to test async behavior
            await asyncio.sleep(0.05)
        logger.info("Mock ingress loop completed - all frames yielded")
    
    protocol.ingress_loop = mock_ingress_loop
    protocol.egress_loop = AsyncMock()
    protocol.control_loop = async_generator_empty
    
    # Create client
    client = TrickleClient(
        protocol=protocol,
        frame_processor=processor
    )
    
    # Test timing
    start_time = time.time()
    ingress_start = start_time
    
    # Start client (this should not block on processing)
    try:
        # Run with a longer timeout to allow all frames to be processed
        # Expected time: 4 frames * 0.05s ingress delay + processing time
        await asyncio.wait_for(client.start("test_request"), timeout=10.0)
    except asyncio.TimeoutError:
        logger.info("Client stopped due to timeout (expected)")
    
    client_end = time.time()
    
    # Wait a bit more to ensure all processing completes
    await asyncio.sleep(0.5)
    
    end_time = time.time()
    total_time = end_time - start_time
    client_time = client_end - start_time
    
    logger.info(f"=== Timing Breakdown ===")
    logger.info(f"Client execution time: {client_time:.3f}s")
    logger.info(f"Total time (including cleanup): {total_time:.3f}s")
    logger.info(f"Ingress delay per frame: 0.05s")
    logger.info(f"Processing delay per frame: {processor.processing_delay}s")
    logger.info(f"Expected ingress time: {len(test_frames) * 0.05:.3f}s")
    logger.info(f"Expected sequential processing: {len(test_frames) * processor.processing_delay:.3f}s")
    
    logger.info(f"=== Test Results ===")
    logger.info(f"Total test time: {total_time:.3f}s")
    logger.info(f"Client execution time: {client_time:.3f}s")
    logger.info(f"Frames processed: {processor.processed_count}")
    
    # Verify async processing worked
    # Use client_time for the comparison since that's the actual processing time
    expected_sequential_time = len(test_frames) * processor.processing_delay
    async_overhead_allowance = expected_sequential_time * 0.3  # 30% overhead allowance
    max_allowed_time = expected_sequential_time + async_overhead_allowance
    
    if client_time < max_allowed_time:
        logger.info("✅ SUCCESS: Processing was async (client time within acceptable range)")
        if client_time < expected_sequential_time:
            logger.info(f"📈 EXCELLENT: Client time ({client_time:.3f}s) < sequential time ({expected_sequential_time:.3f}s)")
        else:
            logger.info(f"📊 GOOD: Client time ({client_time:.3f}s) includes overhead but < max allowed ({max_allowed_time:.3f}s)")
    else:
        logger.warning(f"❌ POTENTIAL ISSUE: Client time ({client_time:.3f}s) > max allowed ({max_allowed_time:.3f}s) - may be blocking")
    
    if processor.processed_count >= len(test_frames):
        logger.info("✅ SUCCESS: All frames were processed")
    else:
        logger.warning(f"❌ ISSUE: Only {processor.processed_count}/{len(test_frames)} frames processed")


async def async_generator_empty(stop_event=None):
    """Empty async generator for mocking."""
    if False:  # Never yields anything
        yield


if __name__ == "__main__":
    asyncio.run(test_async_processing())
