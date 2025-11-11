"""Minimal passthrough example for PyTrickle.

This example demonstrates the basic structure of a PyTrickle streaming app.
Video and audio frames pass through unchanged. Use this as a starting point
for your own processing pipeline.

Run directly:
    python -m pytrickle.examples.passthrough_example

Or generate a customized copy:
    pytrickle init my_app --port 8000 --out ./my_app.py
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import List

from pytrickle import StreamProcessor, VideoFrame, AudioFrame
from pytrickle.decorators import (
    audio_handler,
    model_loader,
    on_stream_stop,
    param_updater,
    video_handler,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class AppConfig:
    """Configuration that can be updated at runtime via /control endpoint."""
    enabled: bool = True
    # Add your parameters here, e.g.:
    # strength: float = 0.5


class PassthroughHandlers:
    """Minimal handlers that pass frames through unchanged.
    
    Modify these handlers to add your own processing logic.
    """

    def __init__(self) -> None:
        self.cfg = AppConfig()
        # Initialize your state here, e.g.:
        # self.model = None

    @model_loader
    async def load(self, **kwargs: dict) -> None:
        """Called once at stream start to load models or initialize resources.
        
        Args:
            **kwargs: Stream metadata (stream_name, etc.)
        """
        logger.info("Initializing passthrough handlers (no model to load)")

    @video_handler
    async def handle_video(self, frame: VideoFrame) -> VideoFrame:
        """Process video frame.
        
        Args:
            frame: Input video frame with .tensor property
            
        Returns:
            Processed video frame (or original for passthrough)
        """
        if not self.cfg.enabled:
            return frame

        # Add your video processing here, e.g.:
        # tensor = frame.tensor
        # processed = your_model(tensor)
        # return frame.replace_tensor(processed)

        return frame  # Passthrough

    @audio_handler
    async def handle_audio(self, frame: AudioFrame) -> List[AudioFrame]:
        """Process audio frame.
        
        Args:
            frame: Input audio frame with .samples property
            
        Returns:
            List of processed audio frames (or [original] for passthrough)
        """
        if not self.cfg.enabled:
            return [frame]

        # Add your audio processing here, e.g.:
        # samples = frame.samples
        # processed = your_audio_model(samples)
        # return [frame.replace_samples(processed)]

        return [frame]  # Passthrough

    @param_updater
    async def update_params(self, params: dict) -> None:
        """Update configuration at runtime via POST /control endpoint.
        
        Args:
            params: Dictionary of parameter updates
        """
        if "enabled" in params:
            self.cfg.enabled = bool(params["enabled"])
            logger.info(f"Processing {'enabled' if self.cfg.enabled else 'disabled'}")
        
        # Handle your custom parameters, e.g.:
        # if "strength" in params:
        #     self.cfg.strength = float(params["strength"])

    @on_stream_stop
    async def on_stop(self) -> None:
        """Called when stream stops - cleanup resources."""
        logger.info("Stream stopped, releasing resources")
        # Cleanup your resources here, e.g.:
        # del self.model


async def main() -> None:
    """Main entry point - creates and runs the stream processor."""
    handlers = PassthroughHandlers()
    processor = StreamProcessor.from_handlers(
        handlers,
        name="passthrough-example",
        port=8000,
    )
    
    logger.info("Send video to: http://localhost:8000/stream")
    logger.info("Update params: POST http://localhost:8000/control")
    
    await processor.run_forever()


if __name__ == "__main__":
    asyncio.run(main())

