#!/usr/bin/env python3
"""
Minimal Async FrameProcessor Demo (queue_mode + output drain)

This example shows the new pattern where a custom `FrameProcessor` runs in
queue_mode so the Trickle client drains processed outputs via its output drain
task. Audio is passed through unchanged (audio_concurrency=0), while video is
processed asynchronously with a subtle Accent Green color tint.

HTTP API (served by StreamServer):
- POST /api/stream/start
- POST /api/stream/params
- GET  /api/stream/status

Usage:
    python async_output_drain_example.py

Then start a stream with something like:
    curl -X POST http://localhost:8000/api/stream/start \
      -H 'Content-Type: application/json' \
      -d '{
            "subscribe_url": "http://127.0.0.1:3389/sample",
            "publish_url":   "http://127.0.0.1:3389/output",
            "gateway_request_id": "demo",
            "params": {"intensity": 0.5}
          }'
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List

import torch

from pytrickle import FrameProcessor, StreamServer
from pytrickle.frames import VideoFrame, AudioFrame


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MinimalAsyncProcessor(FrameProcessor):
    """Tiny async video processor with audio passthrough.

    - Runs with queue_mode enabled so outputs are drained by the client
    - Video: applies a subtle Accent Green (#18794E) color tint with adjustable intensity
    - Audio: passthrough (audio_concurrency=0)
    """

    def __init__(
        self,
        intensity: float = 0.5,
        **kwargs
    ):
        # Log errors via a simple callback
        def log_error(error_type: str, exception: Optional[Exception] = None):
            logger.warning(f"Processing error: {error_type} - {exception}")

        self.intensity = float(intensity)
        self.ready = False

        super().__init__(
            error_callback=log_error,
            queue_mode=True,
            video_queue_size=8,
            audio_queue_size=32,
            video_concurrency=1,
            audio_concurrency=0,
            **kwargs
        )

    def initialize(self, **kwargs):
        # Allow intensity override from kwargs
        if "intensity" in kwargs:
            try:
                self.intensity = float(kwargs["intensity"])
            except Exception:
                pass
        # Clamp between 0 and 1
        self.intensity = max(0.0, min(1.0, self.intensity))
        self.ready = True
        logger.info(f"✅ MinimalAsyncProcessor ready (intensity={self.intensity:.3f})")

    async def process_video_async(self, frame: VideoFrame) -> Optional[VideoFrame]:
        if not self.ready:
            return None
        try:
            tensor = frame.tensor
            if tensor is None:
                return None
            if tensor.dtype != torch.float32:
                tensor = tensor.float()
            # Ensure 4D (N H W C)
            if hasattr(tensor, "ndim") and tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)

            # Apply Accent Green tint (#18794E -> rgb approx [0.094, 0.475, 0.306])
            if self.intensity > 0.0:
                target = torch.tensor([0.094, 0.475, 0.306], device=tensor.device, dtype=tensor.dtype)
                tinted = torch.clamp(tensor + (target - 0.5) * 0.4, 0.0, 1.0)
                tensor = tensor * (1.0 - self.intensity) + tinted * self.intensity
                tensor = torch.clamp(tensor, 0.0, 1.0)

            return frame.replace_tensor(tensor)
        except Exception as e:
            if self.error_callback:
                try:
                    self.error_callback("video_processing_error", e)
                except Exception:
                    pass
            return None

    async def process_audio_async(self, frame: AudioFrame) -> Optional[List[AudioFrame]]:
        # Passthrough audio unchanged
        return [frame]

    def update_params(self, params: Dict[str, Any]):
        if not params:
            return
        if "intensity" in params:
            try:
                new_i = float(params["intensity"])
                new_i = max(0.0, min(1.0, new_i))
                if new_i != self.intensity:
                    old = self.intensity
                    self.intensity = new_i
                    logger.info(f"Intensity: {old:.3f} → {self.intensity:.3f}")
            except Exception:
                pass


async def main():
    server = None
    try:
        processor = MinimalAsyncProcessor(
            intensity=0.5
        )

        server = StreamServer(
            frame_processor=processor,
            port=8000,
            host="0.0.0.0",
            capability_name="minimal-async-output-drain"
        )

        logger.info("🌐 Service ready at http://localhost:8000")
        logger.info("API: /api/stream/start, /api/stream/params, /api/stream/status")
        await server.run_forever()
    except KeyboardInterrupt:
        logger.info("🛑 Service stopped")
    finally:
        if server is not None:
            await server.stop()


if __name__ == "__main__":
    asyncio.run(main())


