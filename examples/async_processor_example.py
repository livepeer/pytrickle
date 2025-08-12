#!/usr/bin/env python3
"""
AsyncFrameProcessor BYOC Demo - Accent Green Tinting

Simple BYOC service that applies Accent Green (#18794E) color tinting
with real-time intensity control via HTTP API.

Environment Variables:
- CAPABILITY_URL: URL to register with orchestrator (optional)
- PORT: HTTP server port (default: 8080)
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any, List
import torch

from pytrickle import AsyncFrameProcessor, TrickleApp, RegisterCapability
from pytrickle.frames import VideoFrame, AudioFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccentGreenProcessor(AsyncFrameProcessor):
    """Applies Accent Green (#18794E) color tinting with adjustable intensity."""
    
    def __init__(self, intensity: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.intensity = max(0.0, min(1.0, intensity))
        self.ready = False
    
    async def initialize(self):
        """Initialize and warm up the processor."""
        # Simple warmup
        dummy_frame = VideoFrame(torch.rand(1, 256, 256, 3), 0, 30, {})
        await self._apply_tint(dummy_frame.tensor)
        
        self.ready = True
        logger.info(f"✅ Accent Green processor ready (intensity: {self.intensity})")
    
    async def process_video_async(self, frame: VideoFrame) -> Optional[VideoFrame]:
        """Apply Accent Green tinting to video frame."""
        if not self.ready:
            return frame
        
        tinted_tensor = await self._apply_tint(frame.tensor)
        return frame.replace_tensor(tinted_tensor)
    
    async def _apply_tint(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply Accent Green tint (#18794E: rgb(24,121,78) -> (0.094,0.475,0.306))."""
        # Accent Green target color
        target = torch.tensor([0.094, 0.475, 0.306], device=tensor.device)
        
        # Create tinted version
        tinted = tensor.clone()
        for c in range(3):
            tinted[:, :, :, c] = torch.clamp(
                tensor[:, :, :, c] + (target[c] - 0.5) * 0.4, 0, 1
            )
        
        # Blend based on intensity
        return tensor * (1.0 - self.intensity) + tinted * self.intensity
    
    async def process_audio_async(self, frame: AudioFrame) -> Optional[List[AudioFrame]]:
        """Pass through audio unchanged."""
        return [frame]
    
    def update_params(self, params: Dict[str, Any]):
        """Update tint intensity (0.0 to 1.0)."""
        if "intensity" in params:
            old = self.intensity
            self.intensity = max(0.0, min(1.0, float(params["intensity"])))
            if old != self.intensity:
                logger.info(f"Intensity: {old:.2f} → {self.intensity:.2f}")

async def main():
    """Start the Accent Green tinting service."""
    capability_url = os.getenv("CAPABILITY_URL")
    # Parse port from capability_url if provided, otherwise use PORT env var or default to 8080
    port = 8080
    if capability_url:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(capability_url)
            if parsed.port:
                port = parsed.port
        except Exception:
            pass
    if not capability_url or not port or port == 8080:
        port = int(os.getenv("PORT", "8080"))
    
    logger.info(f"🎨 Starting Accent Green Tinting Service on port {port}")
    
    try:
        # Create and initialize processor
        processor = AccentGreenProcessor(intensity=0.5)
        await processor.start()  # Initializes processor and calls initialize() hook
        
        # Register with orchestrator if URL provided
        if capability_url:
            try:
                RegisterCapability.register(
                    logger,
                    capability_name="accent-green-processor",
                    capability_desc="Accent Green color tinting service",
                    capability_url=capability_url
                )
                logger.info("✅ Registered with orchestrator")
            except Exception as e:
                logger.warning(f"Registration failed: {e}")
        
        # Create TrickleApp with native async processor support
        logger.info(f"🌐 Service ready at http://localhost:{port}")
        logger.info("API: /api/stream/start, /api/stream/params, /api/stream/status")
        logger.info(f"Update intensity: curl -X POST http://localhost:{port}/api/stream/params \\")
        logger.info("  -H 'Content-Type: application/json' -d '{\"intensity\": 0.8}'")
        
        # Create and run TrickleApp with the async processor
        app = TrickleApp(
            frame_processor=processor,
            port=port,
            capability_name="accent-green-processor"
        )
        await app.run_forever()
        
    except KeyboardInterrupt:
        logger.info("🛑 Service stopped")
    finally:
        if 'processor' in locals():
            await processor.stop()


if __name__ == "__main__":
    """
    Accent Green Tinting BYOC Service
    
    Simple BYOC service demonstrating AsyncFrameProcessor with:
    - Accent Green (#18794E) color tinting
    - Real-time intensity control (0.0 to 1.0)  
    - Orchestrator registration (optional)
    - HTTP API for stream management
    - Clean separation: AsyncFrameProcessor + TrickleApp
    
    Usage:
        # HTTP API server (recommended)
        python async_processor_example.py
        CAPABILITY_URL=http://orchestrator:8080/caps PORT=9090 python async_processor_example.py
        
        # Direct client usage (advanced)
        from pytrickle import TrickleProtocol, TrickleClient
        processor = AccentGreenProcessor()
        await processor.start()
        protocol = TrickleProtocol(
            subscribe_url="http://localhost:3389/sample",
            publish_url="http://localhost:3389/output"
        )
        client = TrickleClient(protocol=protocol, frame_processor=processor)
        await client.start("request_id")
    """
    asyncio.run(main())
