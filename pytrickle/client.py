"""
Trickle Client for easy interaction with trickle streams.

Provides a high-level client interface for connecting to trickle streams,
processing video frames, and handling stream lifecycle.
"""

import asyncio
import logging
import queue
from typing import Optional, Callable, AsyncGenerator, Dict, Any

from .protocol import TrickleProtocol
from .frames import VideoFrame, VideoOutput, InputFrame, OutputFrame, AudioOutput
from .exceptions import ErrorPropagator

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
        frame_processor: Optional[Callable[[VideoFrame], VideoOutput]] = None,
        max_retries: int = 3
    ):
        self.protocol = TrickleProtocol(
            subscribe_url=subscribe_url,
            publish_url=publish_url, 
            control_url=control_url,
            events_url=events_url,
            width=width,
            height=height,
            max_retries=max_retries
        )
        self.frame_processor = frame_processor or self._default_frame_processor
        self.stop_event = asyncio.Event()
        self.running = False
        self.request_id = "default"
        self.output_queue = queue.Queue()
        self.error_propagator = ErrorPropagator()
        
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
        
        try:
            # Start the protocol
            await self.protocol.start()
            
            # Start processing loops
            self.running = True
            
            # Add error callback to handle protocol errors
            self.protocol.error_propagator.add_error_callback(self._handle_protocol_error)
            
            # Start the processing loops
            await asyncio.gather(
                self._ingress_loop(),
                self._egress_loop(),
                self._control_loop(),
                self._error_monitor_loop(),
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Error starting client: {e}")
            raise
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
    
    async def _handle_protocol_error(self, error: Exception, source: str):
        """Handle errors from the protocol layer."""
        logger.error(f"Protocol error from {source}: {error}")
        # Trigger our own shutdown
        self.stop_event.set()
    
    async def _error_monitor_loop(self):
        """Monitor for errors and trigger shutdown when needed."""
        try:
            # Wait for shutdown signal from protocol or our own stop event
            await asyncio.wait(
                [
                    asyncio.create_task(self.protocol.error_propagator.wait_for_shutdown()),
                    asyncio.create_task(self.stop_event.wait())
                ],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            logger.info("Shutdown signal received in error monitor")
            self.stop_event.set()
            
        except Exception as e:
            logger.error(f"Error in error monitor loop: {e}")
            self.stop_event.set()
    
    async def _ingress_loop(self):
        """Process incoming frames."""
        try:
            frame_count = 0
            async for frame in self.protocol.ingress_loop(self.stop_event):
                if self.stop_event.is_set():
                    break
                    
                if isinstance(frame, VideoFrame):
                    frame_count += 1
                    logger.debug(f"Processing video frame {frame_count}: {frame.tensor.shape}")
                    
                    try:
                        # Process the frame
                        output = self.frame_processor(frame)
                        if output:
                            # Send to egress
                            await self._send_output(output)
                            logger.debug(f"Sent processed frame {frame_count} to egress")
                        else:
                            logger.warning(f"Frame processor returned None for frame {frame_count}")
                    except Exception as e:
                        logger.error(f"Error processing frame {frame_count}: {e}")
                        # Continue processing other frames
                        continue
                else:
                    logger.debug(f"Received non-video frame: {type(frame)}")
                    
        except Exception as e:
            logger.error(f"Error in ingress loop: {e}")
            self.stop_event.set()
    
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
                        else:
                            # Sentinel value received, stop
                            break
                    except queue.Empty:
                        continue  # No frame available, continue loop
                    except Exception as e:
                        logger.error(f"Error getting frame from output queue: {e}")
                        continue
                    
            await self.protocol.egress_loop(output_generator())
            
        except Exception as e:
            logger.error(f"Error in egress loop: {e}")
            self.stop_event.set()
    
    async def _control_loop(self):
        """Handle control messages."""
        try:
            async for control_data in self.protocol.control_loop(self.stop_event):
                if self.stop_event.is_set():
                    break
                await self._handle_control_message(control_data)
                
        except Exception as e:
            logger.error(f"Error in control loop: {e}")
            # Control loop errors are not fatal, just log them
    
    async def _send_output(self, output: OutputFrame):
        """Send output frame to the egress queue."""
        try:
            self.output_queue.put_nowait(output)
            # Only log timestamp for VideoOutput since OutputFrame doesn't have timestamp
            if isinstance(output, VideoOutput):
                logger.debug(f"Queued output frame: {type(output)} timestamp={output.timestamp}")
            else:
                logger.debug(f"Queued output frame: {type(output)}")
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
        try:
            await self.protocol.emit_monitoring_event(event, event_type)
        except Exception as e:
            logger.debug(f"Error emitting event: {e}")

class SimpleTrickleClient:
    """Simplified trickle client for basic use cases."""
    
    def __init__(self, subscribe_url: str, publish_url: str, max_retries: int = 3):
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.max_retries = max_retries
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
            frame_processor=frame_processor,
            max_retries=self.max_retries
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