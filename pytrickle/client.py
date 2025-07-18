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
from .frames import VideoFrame, VideoOutput, AudioFrame, AudioOutput, InputFrame, OutputFrame
from . import ErrorCallback

logger = logging.getLogger(__name__)

class TrickleClient:
    """High-level client for trickle streaming."""
    
    def __init__(
        self,
        subscribe_url: str,
        publish_url: str,
        control_url: Optional[str] = None,
        events_url: Optional[str] = None,
        text_url: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        frame_processor: Optional[Callable[[VideoFrame], VideoOutput]] = None,
        error_callback: Optional[ErrorCallback] = None,
        audio_processor: Optional[Callable[[AudioFrame], AudioOutput]] = None
    ):
        self.protocol = TrickleProtocol(
            subscribe_url=subscribe_url,
            publish_url=publish_url, 
            control_url=control_url,
            events_url=events_url,
            text_url=text_url,
            width=width,
            height=height,
            error_callback=self._on_protocol_error
        )
        self.frame_processor = frame_processor or self._default_frame_processor
        self.audio_processor = audio_processor or self._default_audio_processor
        self.stop_event = asyncio.Event()
        self.running = False
        self.request_id = "default"
        self.output_queue = queue.Queue()
        self.error_callback = error_callback
        self.error_event = asyncio.Event()  # Use Event instead of boolean
        self.shutdown_event = asyncio.Event()  # Event to signal shutdown
        
    def _default_frame_processor(self, frame: VideoFrame) -> VideoOutput:
        """Default frame processor that passes frames through unchanged."""
        return VideoOutput(frame, self.request_id)
    
    def _default_audio_processor(self, frame: AudioFrame) -> AudioOutput:
        """Default audio processor that passes frames through unchanged."""
        return AudioOutput([frame], self.request_id)
    
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
            audio_count = 0
            async for frame in self.protocol.ingress_loop(self.stop_event):
                # Check for error state
                if self.error_event.is_set() or self.stop_event.is_set():
                    logger.info("Stopping ingress loop due to error or stop signal")
                    break
                    
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
                
                elif isinstance(frame, AudioFrame):
                    audio_count += 1
                    logger.debug(f"Processing audio frame {audio_count}: {frame.nb_samples} samples")
                    
                    # Process the audio frame
                    output = self.audio_processor(frame)
                    if output:
                        # Send to egress
                        await self._send_output(output)
                        logger.debug(f"Sent processed audio frame {audio_count} to egress")
                    else:
                        logger.warning(f"Audio processor returned None for frame {audio_count}")
                
                else:
                    logger.debug(f"Received unknown frame type: {type(frame)}")
        except Exception as e:
            logger.error(f"Error in ingress loop: {e}")
            # Set error state and stop event to trigger other loops to stop
            self.error_event.set()
            self.stop_event.set()
            # Notify parent if error callback is set
            if self.error_callback:
                try:
                    if asyncio.iscoroutinefunction(self.error_callback):
                        await self.error_callback("ingress_loop_error", e)
                    else:
                        self.error_callback("ingress_loop_error", e)
                except Exception as cb_error:
                    logger.error(f"Error in error callback: {cb_error}")
    
    async def _egress_loop(self):
        """Handle outgoing frames."""
        try:
            async def output_generator() -> AsyncGenerator[OutputFrame, None]:
                while not self.stop_event.is_set() and not self.error_event.is_set():
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
            # Set error state and stop event to trigger other loops to stop
            self.error_event.set()
            self.stop_event.set()
            # Notify parent if error callback is set
            if self.error_callback:
                try:
                    if asyncio.iscoroutinefunction(self.error_callback):
                        await self.error_callback("egress_loop_error", e)
                    else:
                        self.error_callback("egress_loop_error", e)
                except Exception as cb_error:
                    logger.error(f"Error in error callback: {cb_error}")
    
    async def _control_loop(self):
        """Handle control messages."""
        try:
            async for control_data in self.protocol.control_loop(self.stop_event):
                # Check for error state
                if self.error_event.is_set() or self.stop_event.is_set():
                    logger.info("Stopping control loop due to error or stop signal")
                    break
                await self._handle_control_message(control_data)
        except Exception as e:
            logger.error(f"Error in control loop: {e}")
            # Set error state and stop event to trigger other loops to stop
            self.error_event.set()
            self.stop_event.set()
            # Notify parent if error callback is set
            if self.error_callback:
                try:
                    if asyncio.iscoroutinefunction(self.error_callback):
                        await self.error_callback("control_loop_error", e)
                    else:
                        self.error_callback("control_loop_error", e)
                except Exception as cb_error:
                    logger.error(f"Error in error callback: {cb_error}")
    
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

    async def _on_protocol_error(self, error_type: str, exception: Optional[Exception] = None):
        """Handle errors from the protocol layer."""
        logger.error(f"Protocol error: {error_type} - {exception}")
        self.error_event.set()  # Set error event
        self.stop_event.set()  # Signal all loops to stop
        
        # Notify parent component
        if self.error_callback:
            try:
                if asyncio.iscoroutinefunction(self.error_callback):
                    await self.error_callback(error_type, exception)
                else:
                    self.error_callback(error_type, exception)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    async def publish_text(self, text_data: str):
        """Publish text data via the text publisher."""
        await self.protocol.publish_text(text_data)

class SimpleTrickleClient:
    """Simplified trickle client for basic use cases."""
    
    def __init__(self, subscribe_url: str, publish_url: str):
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.client: Optional[TrickleClient] = None
    
    async def process_stream(
        self, 
        frame_processor: Callable[[VideoFrame], VideoOutput],
        audio_processor: Optional[Callable[[AudioFrame], AudioOutput]] = None,
        request_id: str = "simple_client",
        width: int = 512,
        height: int = 512
    ):
        """Process a stream with the given frame and audio processors."""
        self.client = TrickleClient(
            subscribe_url=self.subscribe_url,
            publish_url=self.publish_url,
            width=width,
            height=height,
            frame_processor=frame_processor,
            audio_processor=audio_processor
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