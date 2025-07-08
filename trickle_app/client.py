"""
Trickle Client for easy interaction with trickle streams.

Provides a high-level client interface for connecting to trickle streams,
processing video frames, and handling stream lifecycle.
"""

import asyncio
import logging
import queue
from typing import Optional, Callable, AsyncGenerator, Dict, Any

from protocol import TrickleProtocol
from frames import VideoFrame, VideoOutput, InputFrame, OutputFrame

logger = logging.getLogger(__name__)

class TrickleClient:
    """High-level client for trickle streaming."""
    
    def __init__(
        self,
        subscribe_url: str,
        publish_url: str,
        control_url: Optional[str] = None,
        events_url: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        frame_processor: Optional[Callable[[VideoFrame], VideoOutput]] = None
    ):
        self.protocol = TrickleProtocol(
            subscribe_url=subscribe_url,
            publish_url=publish_url, 
            control_url=control_url,
            events_url=events_url,
            width=width,
            height=height
        )
        self.frame_processor = frame_processor or self._default_frame_processor
        self.stop_event = asyncio.Event()
        self.running = False
        self.request_id = "default"
        self.output_queue = queue.Queue()
        
    def _default_frame_processor(self, frame: VideoFrame) -> VideoOutput:
        """Default frame processor that passes frames through unchanged."""
        return VideoOutput(frame, self.request_id)
    
    async def start(self, request_id: str = "default"):
        """Start the trickle client."""
        if self.running:
            raise RuntimeError("Client is already running")
            
        self.request_id = request_id
        self.stop_event.clear()
        
        logger.info(f"Starting trickle client with request_id={request_id}")
        
        # Start the protocol
        await self.protocol.start()
        
        # Start processing loops
        self.running = True
        
        try:
            await asyncio.gather(
                self._ingress_loop(),
                self._egress_loop(),
                self._control_loop(),
                return_exceptions=True
            )
        finally:
            self.running = False
            await self.protocol.stop()
    
    async def stop(self):
        """Stop the trickle client."""
        if not self.running:
            return
            
        logger.info("Stopping trickle client")
        self.stop_event.set()
        
        # Send sentinel value to stop egress loop
        try:
            self.output_queue.put_nowait(None)
        except queue.Full:
            pass
    
    async def _ingress_loop(self):
        """Process incoming frames."""
        try:
            frame_count = 0
            async for frame in self.protocol.ingress_loop(self.stop_event):
                if isinstance(frame, VideoFrame):
                    frame_count += 1
                    logger.debug(f"Processing video frame {frame_count}: {frame.tensor.shape}")
                    
                    # Process the frame
                    output = self.frame_processor(frame)
                    if output:
                        # Send to egress
                        await self._send_output(output)
                        logger.debug(f"Sent processed frame {frame_count} to egress")
                    else:
                        logger.warning(f"Frame processor returned None for frame {frame_count}")
                else:
                    logger.debug(f"Received non-video frame: {type(frame)}")
        except Exception as e:
            logger.error(f"Error in ingress loop: {e}")
    
    async def _egress_loop(self):
        """Handle outgoing frames."""
        try:
            async def output_generator() -> AsyncGenerator[OutputFrame, None]:
                while not self.stop_event.is_set():
                    try:
                        # Try to get a frame from the queue with timeout
                        frame = await asyncio.to_thread(self.output_queue.get, timeout=0.1)
                        if frame is not None:
                            yield frame
                    except queue.Empty:
                        continue  # No frame available, continue loop
                    except Exception as e:
                        logger.error(f"Error getting frame from output queue: {e}")
                        continue
                    
            await self.protocol.egress_loop(output_generator())
        except Exception as e:
            logger.error(f"Error in egress loop: {e}")
    
    async def _control_loop(self):
        """Handle control messages."""
        try:
            async for control_data in self.protocol.control_loop(self.stop_event):
                await self._handle_control_message(control_data)
        except Exception as e:
            logger.error(f"Error in control loop: {e}")
    
    async def _send_output(self, output: OutputFrame):
        """Send output frame to the egress queue."""
        try:
            self.output_queue.put_nowait(output)
            logger.debug(f"Queued output frame: {type(output)} timestamp={output.timestamp}")
        except queue.Full:
            logger.warning("Output queue is full, dropping frame")
        except Exception as e:
            logger.error(f"Error sending output frame: {e}")
    
    async def _handle_control_message(self, control_data: Dict[str, Any]):
        """Handle incoming control messages."""
        logger.info(f"Received control message: {control_data}")
        # Override this method to handle control messages
    
    async def emit_event(self, event: Dict[str, Any], event_type: str = "client_event"):
        """Emit a monitoring event."""
        await self.protocol.emit_monitoring_event(event, event_type)

class SimpleTrickleClient:
    """Simplified trickle client for basic use cases."""
    
    def __init__(self, subscribe_url: str, publish_url: str):
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.client: Optional[TrickleClient] = None
    
    async def process_stream(
        self, 
        frame_processor: Callable[[VideoFrame], VideoOutput],
        request_id: str = "simple_client",
        width: int = 512,
        height: int = 512
    ):
        """Process a stream with the given frame processor."""
        self.client = TrickleClient(
            subscribe_url=self.subscribe_url,
            publish_url=self.publish_url,
            width=width,
            height=height,
            frame_processor=frame_processor
        )
        
        try:
            await self.client.start(request_id)
        finally:
            if self.client:
                await self.client.stop()
    
    async def stop(self):
        """Stop the client."""
        if self.client:
            await self.client.stop() 