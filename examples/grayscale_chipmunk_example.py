"""Example demonstrating decorator-based handlers that stylize video and audio.

Run with:
    python -m pytrickle.examples.grayscale_chipmunk_example

The example wires up a ``StreamProcessor`` using the declarative decorators. Video
frames are converted to grayscale, and audio frames are pitch-shifted upward for a
"chipmunk" effect.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import torch

from pytrickle import StreamProcessor, VideoFrame, AudioFrame
from pytrickle.decorators import (
    audio_handler,
    model_loader,
    on_stream_start,
    on_stream_stop,
    param_updater,
    video_handler,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class EffectConfig:
    """Mutable configuration that can be updated at runtime."""

    grayscale_enabled: bool = True
    chipmunk_factor: float = 1.6  # >1.0 raises the pitch


class GrayscaleChipmunkHandlers:
    """Handlers that stylize incoming media using the decorator API."""

    def __init__(self) -> None:
        self.config = EffectConfig()

    @model_loader
    async def load(self, **_: dict) -> None:
        logger.info("Model loader invoked; nothing to warm up for this demo.")

    @on_stream_start
    async def on_start(self, params: Dict[str, Any]) -> None:
        """Capture initial parameter values for the effect."""
        if not params:
            return
        if "grayscale" in params:
            self.config.grayscale_enabled = bool(params["grayscale"])
            logger.info("Stream start grayscale set to %s", self.config.grayscale_enabled)
        if "chipmunk_factor" in params:
            try:
                value = float(params["chipmunk_factor"])
                if value > 0:
                    self.config.chipmunk_factor = value
                    logger.info("Stream start chipmunk_factor set to %.2f", value)
            except (TypeError, ValueError):
                logger.warning("Invalid chipmunk_factor %s at stream start", params["chipmunk_factor"])

    @video_handler
    async def handle_video(self, frame: VideoFrame) -> VideoFrame:
        """Apply a grayscale conversion while keeping the tensor layout intact."""

        if not self.config.grayscale_enabled:
            return frame

        tensor = frame.tensor

        squeeze_batch = tensor.dim() == 4 and tensor.shape[0] == 1
        if squeeze_batch:
            tensor = tensor.squeeze(0)

        if tensor.dim() != 3 or tensor.shape[-1] not in (1, 3):
            logger.debug("Unsupported video tensor shape %s; skipping grayscale", tuple(tensor.shape))
            return frame

        if tensor.shape[-1] == 1:
            grayscale_rgb = tensor.repeat(1, 1, 3)
        else:
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=tensor.device, dtype=tensor.dtype)
            grayscale = (tensor[..., :3] * weights).sum(dim=-1, keepdim=True)
            grayscale_rgb = grayscale.repeat(1, 1, 3)

        if squeeze_batch:
            grayscale_rgb = grayscale_rgb.unsqueeze(0)

        return frame.replace_tensor(grayscale_rgb)

    @audio_handler
    async def handle_audio(self, frame: AudioFrame) -> List[AudioFrame]:
        """Raise audio pitch by compressing the time axis and resampling to original length."""

        pitch = max(self.config.chipmunk_factor, 1.0)
        samples = np.asarray(frame.samples)
        orig_dtype = samples.dtype

        if samples.ndim == 1:
            channels = 1
            samples_mono = samples[np.newaxis, :]
            layout = "mono"
        else:
            # Normalize to channel-first layout for processing
            if samples.shape[0] <= samples.shape[1]:
                samples_mono = samples
                layout = "channels_first"
            else:
                samples_mono = samples.T
                layout = "channels_last"
            channels = samples_mono.shape[0]

        # Convert to float32 in [-1, 1]
        max_val = None
        if np.issubdtype(orig_dtype, np.integer):
            max_val = float(np.iinfo(orig_dtype).max)
            samples_float = samples_mono.astype(np.float32) / max_val
        else:
            samples_float = samples_mono.astype(np.float32, copy=False)

        if samples_float.shape[1] < 2:
            logger.debug("Audio frame too short for pitch shifting; returning original samples")
            return [frame]

        indices = np.arange(samples_float.shape[1], dtype=np.float32)
        shifted_idx = np.linspace(0.0, indices[-1], samples_float.shape[1], dtype=np.float32) / pitch
        shifted_idx = np.clip(shifted_idx, 0.0, indices[-1])

        processed = np.empty_like(samples_float)
        for ch in range(channels):
            processed[ch] = np.interp(shifted_idx, indices, samples_float[ch])

        # Restore dtype and original layout
        if np.issubdtype(orig_dtype, np.integer) and max_val is not None:
            processed = np.clip(processed * max_val, np.iinfo(orig_dtype).min, np.iinfo(orig_dtype).max)
            processed = processed.astype(orig_dtype)
        else:
            processed = processed.astype(orig_dtype, copy=False)

        if layout == "mono":
            processed = processed.squeeze(0)
        elif layout == "channels_last":
            processed = processed.T

        chipmunk_frame = frame.replace_samples(processed)
        return [chipmunk_frame]

    @param_updater
    async def update_config(self, params: dict ) -> None:
        if "grayscale" in params:
            self.config.grayscale_enabled = bool(params["grayscale"])
        if "chipmunk_factor" in params:
            try:
                value = float(params["chipmunk_factor"])
                if value > 0:
                    self.config.chipmunk_factor = value
            except (TypeError, ValueError):
                logger.warning("Invalid chipmunk_factor %s; keeping %s", params["chipmunk_factor"], self.config.chipmunk_factor)

    @on_stream_stop
    async def on_stop(self) -> None:
        logger.info("Stream stopped; restoring defaults.")
        self.config = EffectConfig()


async def main() -> None:
    handlers = GrayscaleChipmunkHandlers()
    processor = StreamProcessor.from_handlers(
        handlers,
        name="grayscale-chipmunk-demo",
        port=8000,
    )
    
    # Explicitly call load_model to initialize the handlers
    logger.info("Initializing handlers...")
    await processor._frame_processor.load_model()
    
    await processor.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
