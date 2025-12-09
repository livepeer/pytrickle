#!/usr/bin/env python3
"""
Model Loading and Overlay Example

Demonstrates:
1. Non-blocking model loading with configurable delay
2. Server health state transitions (LOADING -> IDLE)
3. Manual loading overlay activation during processing delays
4. Real-time parameter updates

The server starts immediately and is available for /health checks while
the model loads in the background. The loading overlay is manually activated
during processing delays, pausing frame processing to save resources.

To test:
1. Run: python examples/overlay_example.py
2. Check health: curl http://localhost:8000/health
3. Simulate a 15s processing stall:
   curl -X POST http://localhost:8000/update_params \
     -H "Content-Type: application/json" \
     -d '{"processing_delay": 15}'
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any

from pytrickle.frames import AudioFrame, VideoFrame
from pytrickle.frame_skipper import FrameSkipConfig
from pytrickle.frame_overlay import OverlayConfig, OverlayMode
from pytrickle.stream_processor import StreamProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoadingOverlayProcessor(StreamProcessor):
    """Encapsulate the overlay example in a reusable StreamProcessor subclass."""

    def __init__(
        self,
        *,
        model_load_delay: float = 10.0,
        startup_block_seconds: float = 15.0,
    ):
        self.model_load_delay = model_load_delay
        self.model_loaded = False
        self.pending_startup_block = startup_block_seconds
        self._model_load_start_time: Optional[float] = None

        super().__init__(
            video_processor=self.process_video,
            audio_processor=self.process_audio,
            model_loader=self.load_model,
            param_updater=self.update_params,
            on_stream_start=self.on_stream_start,
            on_stream_stop=self.on_stream_stop,
            name="model-loading-demo",
            port=8000,
            frame_skip_config=FrameSkipConfig(),
            overlay_config=OverlayConfig(
                mode=OverlayMode.PROGRESSBAR,
                message="Loading...",
                enabled=True,
                auto_timeout_seconds=1.0,
            ),
            route_prefix="/",
        )

    async def load_model(self, **kwargs):
        """Simulate non-blocking model loading."""
        self._model_load_start_time = time.time()
        logger.info("🔄 Model loading started...")

        load_delay = kwargs.get("load_delay", self.model_load_delay)
        if load_delay > 0:
            logger.info(f"Simulating model load for {load_delay:.1f}s...")
            await asyncio.sleep(load_delay)

        self.model_loaded = True
        load_duration = time.time() - (self._model_load_start_time or time.time())
        logger.info(f"✅ Model loading complete in {load_duration:.2f}s")

    async def on_stream_start(self, params: Dict[str, Any]):
        """Called when a stream starts."""
        logger.info("🎬 Stream started")
        if not self.model_loaded:
            logger.warning("⚠️  Model not loaded yet - frames will pass through until ready")

        if self.pending_startup_block > 0:
            block = self.pending_startup_block
            self.pending_startup_block = 0.0
            self._activate_manual_overlay(block)
            logger.info("Startup block scheduled for %.1fs", block)

    async def on_stream_stop(self):
        """Called when stream stops."""
        logger.info("🛑 Stream stopped")

    async def process_video(self, frame: VideoFrame):
        """
        Process video frames.

        When manual loading overlay is active (`set_manual_loading(True)`),
        the client skips calling this method, so the processor can simply echo
        the frame.
        """
        return frame

    async def process_audio(self, frame: AudioFrame) -> list[AudioFrame]:
        """Pass-through audio processing."""
        return [frame]

    async def update_params(self, params: dict):
        """
        Handle parameter updates.

        Parameters:
        - processing_delay: seconds to hold video output for the active stream
        - simulate_startup_block: seconds to hold output for the next stream start
        """
        logger.info(f"Custom parameters updated: {params}")

        if "simulate_startup_block" in params:
            self.pending_startup_block = max(0.0, float(params["simulate_startup_block"]))
            if self.pending_startup_block > 0:
                logger.info("Next stream will withhold frames for %.1fs", self.pending_startup_block)
            else:
                logger.info("Startup block cleared for next stream")

        processing_delay = float(params.get("processing_delay", 0.0))
        if processing_delay > 0:
            logger.info("Simulating processing delay for %.1fs", processing_delay)
            self._activate_manual_overlay(processing_delay)

    def _activate_manual_overlay(self, duration: float):
        if self.set_loading_overlay(True):
            asyncio.create_task(self._disable_overlay_after(duration))

    async def _disable_overlay_after(self, duration: float):
        """Disable manual overlay after the delay completes."""
        try:
            await asyncio.sleep(duration)
            self.set_loading_overlay(False)
            logger.info("Processing delay complete - overlay disabled")
        except asyncio.CancelledError:
            logger.debug("Processing delay logging cancelled")


# Create and run StreamProcessor
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Model Loading & Loading Overlay Example")
    logger.info("=" * 60)
    logger.info("")
    processor = ModelLoadingOverlayProcessor()
    logger.info("Server: http://localhost:8000")
    logger.info(f"Model loading delay: {processor.model_load_delay}s")
    logger.info("")
    logger.info("Test endpoints:")
    logger.info("  curl http://localhost:8000/health")
    logger.info("  curl -X POST http://localhost:8000/update_params \\")
    logger.info('    -H "Content-Type: application/json" \\')
    logger.info('    -d \'{"processing_delay": 15}\'')
    logger.info("")
    processor.run()