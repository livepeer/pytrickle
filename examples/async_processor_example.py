#!/usr/bin/env python3
"""
FrameProcessor BYOC Demo - Accent Green Tinting

BYOC service that applies Accent Green (#18794E) color tinting
with real-time intensity control, error handling, and configurable initialization.

Environment Variables:
- CAPABILITY_URL: URL to register with orchestrator (optional)
- PORT: HTTP server port (default: 8000)
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any, List
import torch

from pytrickle import FrameProcessor, TrickleApp, RegisterCapability
from pytrickle.frames import VideoFrame, AudioFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccentGreenProcessor(FrameProcessor):
    """Applies Accent Green (#18794E) color tinting with adjustable intensity."""
    
    def __init__(self, intensity: float = 0.5, **kwargs):
        # Simple error callback that logs processing errors
        def log_processing_error(error_type: str, exception: Optional[Exception] = None):
            logger.warning(f"Processing error: {error_type} - {exception}")
            logger.info("Fallback frame will be sent automatically")
        
        self.intensity = intensity
        self.ready = False
        
        super().__init__(error_callback=log_processing_error, **kwargs)
    
    def initialize(self, **kwargs):
        """Initialize and warm up the processor."""
        # Allow intensity override from kwargs
        self.intensity = kwargs.get('intensity', self.intensity)
        self.intensity = max(0.0, min(1.0, self.intensity))
        
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
    app = None
    port = 8000
    try:
        # Create processor with initialization kwargs
        processor = AccentGreenProcessor(
            intensity=0.5
        )
        
        # Register with orchestrator if URL provided
        try:
            result = await RegisterCapability.register(
                logger,
                capability_name="accent-green-processor",
                capability_desc="Accent Green color tinting service"
            )
            if result and result != False:
                # Extract port from returned capability URL
                if result.port:
                    port = result.port
                    logger.info(f"✅ Registered with orchestrator, using port {port} from returned URL")
                else:
                    logger.info("✅ Registered with orchestrator")
            else:
                logger.warning("Registration failed")
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
        if 'app' in locals() and app is not None:
            await app.stop()


if __name__ == "__main__":
    """
    Accent Green Tinting BYOC Service
    
    Simple BYOC service demonstrating FrameProcessor with:
    - Accent Green (#18794E) color tinting
    - Real-time intensity control (0.0 to 1.0)  
    - Error handling with error callbacks
    - Configurable initialization via kwargs
    - Orchestrator registration (optional)
    - HTTP API for stream management
    - Clean separation: FrameProcessor + TrickleApp
    
    Usage:
        # HTTP API server (recommended)
        python async_processor_example.py
        CAPABILITY_URL=http://processor:8000/caps PORT=9090 python async_processor_example.py
        
        # Direct client usage with kwargs
        from pytrickle import TrickleProtocol, TrickleClient
        processor = AccentGreenProcessor(intensity=0.8)
        protocol = TrickleProtocol(
            subscribe_url="http://localhost:3389/sample",
            publish_url="http://localhost:3389/output"
        )
        client = TrickleClient(protocol=protocol, frame_processor=processor)
        await client.start("request_id")
    """
    asyncio.run(main())
