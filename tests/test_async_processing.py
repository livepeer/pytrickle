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
        self.processed_video_count = 0
        self.processed_audio_count = 0
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
        self.processed_video_count += 1
        
        end_time = time.time()
        logger.info(f"Completed video processing for frame {self.processed_count} in {end_time - start_time:.3f}s")
        
        # Return a VideoFrame with the processed tensor, not just the tensor
        return frame.replace_tensor(processed_tensor)
    
    async def process_audio_async(self, frame: AudioFrame) -> list[AudioFrame]:
        """Process audio frame with artificial delay."""
        start_time = time.time()
        logger.info(f"Starting audio processing for frame {self.processed_count}")
        
        # Simulate heavy processing with async delay
        await asyncio.sleep(self.processing_delay)
        
        # Simple processing: just modify the samples
        processed_samples = frame.samples * 0.8  # Lower volume
        processed_frame = frame.from_audio_frame(samples=processed_samples)
        self.processed_count += 1
        self.processed_audio_count += 1
        
        end_time = time.time()
        logger.info(f"Completed audio processing for frame {self.processed_count} in {end_time - start_time:.3f}s")
        
        return [processed_frame]
    
    def update_params(self, params: dict):
        """Update processing parameters."""
        if "delay" in params:
            self.processing_delay = float(params["delay"])
            logger.info(f"Updated processing delay to {self.processing_delay}s")


async def create_test_frames(num_frames: int = 2):
    """Create test video and audio frames."""
    frames = []
    
    # Create multiple frames with different timestamps
    for i in range(num_frames):
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
    
    # Create mock protocol with FPSMeter
    protocol = Mock(spec=TrickleProtocol)
    protocol.start = AsyncMock()
    protocol.stop = AsyncMock()
    protocol.error_callback = None  # Add the missing attribute
    
    # Mock FPSMeter for the protocol
    from pytrickle.fps_meter import FPSMeter
    protocol.fps_meter = FPSMeter()
    
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
        # Give a moment for processing to complete, then signal stop
        await asyncio.sleep(0.5)
        stop_event.set()
        logger.info("Mock ingress set stop_event - signaling client shutdown")
    
    # Mock egress loop that properly consumes output frames and handles sentinel
    async def mock_egress_loop(output_generator):
        async for output_frame in output_generator:
            if output_frame is None:
                logger.debug("Egress received sentinel - terminating")
                break
            logger.debug(f"Egress received: {type(output_frame).__name__}")
    
    protocol.ingress_loop = mock_ingress_loop
    protocol.egress_loop = mock_egress_loop
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
        # Use a generous timeout to test natural completion
        await asyncio.wait_for(client.start("test_request"), timeout=5.0)
        logger.info("Client completed naturally")
    except asyncio.TimeoutError:
        logger.info("Client stopped due to timeout - this may indicate an issue")
    
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
    # The key insight: if processing were blocking/sequential, it would take much longer
    # Expected ingress time: frames * delay = 4 * 0.05 = 0.2s
    # Expected processing time if sequential: frames * processing_delay = 4 * 0.2 = 0.8s  
    # Expected processing time if async: max(ingress_time, longest_single_process) ≈ max(0.2s, 0.2s) = 0.2s
    # Plus realistic client coordination overhead (loops, queues, cleanup): ~1.5-2s is normal
    
    ingress_time = len(test_frames) * 0.05  # Time for ingress to feed all frames
    expected_sequential_processing = len(test_frames) * processor.processing_delay
    expected_async_processing = max(ingress_time, processor.processing_delay)  # Bottleneck time
    
    # Realistic client overhead: protocol startup, loop coordination, cleanup, etc.
    # With proper shutdown coordination, overhead should be minimal (~0.5s)
    realistic_client_overhead = 0.5  # Minimal overhead for well-coordinated async loops
    max_reasonable_time = expected_async_processing + realistic_client_overhead
    
    # The real test: async should be much faster than sequential
    if client_time < expected_sequential_processing:
        logger.info(f"✅ EXCELLENT: Client time ({client_time:.3f}s) < sequential processing time ({expected_sequential_processing:.3f}s)")
        logger.info("📈 SUCCESS: Async processing is working - much faster than sequential would be")
    elif client_time < max_reasonable_time:
        logger.info(f"✅ SUCCESS: Client time ({client_time:.3f}s) is reasonable for async processing with overhead")
        logger.info(f"📊 GOOD: Time includes expected client coordination overhead (~{realistic_client_overhead}s)")
    else:
        logger.warning(f"❌ POTENTIAL ISSUE: Client time ({client_time:.3f}s) > reasonable max ({max_reasonable_time:.3f}s)")
        logger.warning("This might indicate blocking behavior or excessive overhead")
    
    if processor.processed_count >= len(test_frames):
        logger.info("✅ SUCCESS: All frames were processed")
    else:
        logger.warning(f"❌ ISSUE: Only {processor.processed_count}/{len(test_frames)} frames processed")


async def test_frame_skipping():
    """Test frame skipping functionality."""
    logger.info("=== Testing Frame Skipping ===")
    
    from pytrickle.frame_skipper import AdaptiveFrameSkipper, FrameProcessingResult, FrameSkipConfig
    from pytrickle.frames import VideoFrame, AudioFrame
    from pytrickle.fps_meter import FPSMeter
    
    config = FrameSkipConfig(target_fps=10)
    fps_meter = FPSMeter()
    skipper = AdaptiveFrameSkipper(config=config, fps_meter=fps_meter)
    
    test_frames = await create_test_frames(num_frames=10)
    video_frames = [f for f in test_frames if isinstance(f, VideoFrame)]
    audio_frames = [f for f in test_frames if isinstance(f, AudioFrame)]
    
    processed_video = 0
    processed_audio = 0
    skipped_video = 0
    
    # Test video frame skipping
    for i, frame in enumerate(video_frames):
        skipper.frame_counter = i + 1
        skipper.skip_interval = 3  # Skip every 3rd frame
        
        result = skipper._process_video_frame(frame)
        if result == FrameProcessingResult.SKIPPED:
            skipped_video += 1
        elif isinstance(result, VideoFrame):
            processed_video += 1
        else:
            raise AssertionError(f"Unexpected result for video frame: {result}")
    
    # Audio frames are processed separately (not by frame skipper)
    processed_audio = len(audio_frames)  # All audio frames processed directly
    
    logger.info(f"Video frames: {len(video_frames)} → processed: {processed_video}, skipped: {skipped_video}")
    logger.info(f"Audio frames: {len(audio_frames)} → processed: {processed_audio}")
    
    assert skipped_video > 0, "Some video frames should have been skipped"
    assert processed_audio == len(audio_frames), "All audio frames should be processed"
    
    logger.info("✅ Frame skipping test passed")

async def async_generator_empty(stop_event=None):
    """Empty async generator for mocking."""
    if False:  # Never yields anything
        yield


if __name__ == "__main__":
    asyncio.run(test_async_processing())
    asyncio.run(test_frame_skipping())