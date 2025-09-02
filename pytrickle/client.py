"""
Trickle Client for processing video streams.

Coordinates ingress, egress, and control loops with proper shutdown handling
to ensure all components stop when subscription ends.
"""

import asyncio
import queue
import logging
import json
from typing import Callable, Optional, Union, Deque, Any
from collections import deque

from .protocol import TrickleProtocol
from .frames import VideoFrame, AudioFrame, VideoOutput, AudioOutput
from . import ErrorCallback
from .frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


class TrickleClient:
    """High-level client for trickle stream processing with native async support."""
    
    def __init__(
        self,
        protocol: TrickleProtocol,
        frame_processor: 'FrameProcessor',
        control_handler: Optional[Callable] = None,
        send_data_interval: Optional[float] = 0.333,
        error_callback: Optional[ErrorCallback] = None
    ):
        """Initialize TrickleClient.
        
        Args:
            protocol: TrickleProtocol instance
            frame_processor: FrameProcessor for native async processing
            control_handler: Optional control message handler
            error_callback: Optional error callback (if None, uses frame_processor.error_callback)
        """
        self.protocol = protocol
        self.frame_processor = frame_processor
        self.control_handler = control_handler
        self.send_data_interval = send_data_interval

        # Use provided error_callback, or fall back to frame_processor's error_callback
        self.error_callback = error_callback or frame_processor.error_callback
        
        # Connect protocol error callback to client error handling
        if not self.protocol.error_callback:
            self.protocol.error_callback = self._on_protocol_error
        
        # Client state
        self.running = False
        self.request_id = "default"
        
        # Coordination events
        self.stop_event = asyncio.Event()
        self.error_event = asyncio.Event()
        
        # Output queues
        self.output_queue = queue.Queue()
        self.data_queue: Deque[Any] = deque(maxlen=1000)

    def process_frame(self, frame: Union[VideoFrame, AudioFrame]) -> Optional[Union[VideoOutput, AudioOutput]]:
        """Process a single frame and return the output."""
        if not frame:
            return None
        if isinstance(frame, VideoFrame):
            return VideoOutput(frame, self.request_id)
        elif isinstance(frame, AudioFrame):
            return AudioOutput([frame], self.request_id)
    
    async def start(self, request_id: str = "default"):
        """Start the trickle client."""
        if self.running:
            raise RuntimeError("Client is already running")
            
        self.request_id = request_id
        self.stop_event.clear()
        self.error_event.clear()
        
        logger.info(f"Starting trickle client with request_id={request_id}")
        
        # Start the protocol
        await self.protocol.start()
        
        # Start processing loops
        self.running = True
        
        try:
            # Run all loops concurrently
            results = await asyncio.gather(
                self._ingress_loop(),
                self._egress_loop(),
                self._control_loop(),
                self._send_data_loop(),
                return_exceptions=True
            )
            
            # Check if any loop had an exception that is not a cancelled error
            for i, result in enumerate(results):
                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                    loop_names = ["ingress", "egress", "control", "send_data"]
                    logger.error(f"{loop_names[i]} loop failed: {result}")
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in client loops: {e}")
        finally:
            self.running = False
            logger.info("Stopping protocol due to client loops ending")
            
            # Call the optional on_stream_stop callback before stopping protocol
            if self.frame_processor.on_stream_stop:
                try:
                    await self.frame_processor.on_stream_stop()
                    logger.info("Stream stop callback executed successfully")
                except Exception as e:
                    logger.error(f"Error in stream stop callback: {e}")
            
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
            self.data_queue.append(None)
        except queue.Full:
            pass
    
    async def publish_data(self, data: str):
        """Publish data via the protocol's data publisher."""
        self.data_queue.append(data)

    async def _on_protocol_error(self, error_type: str, exception: Optional[Exception] = None):
        """Handle protocol errors and shutdown events."""
        logger.info(f"Protocol event received: {error_type} - {exception}")
        
        # Set appropriate event based on error type
        if error_type in ("protocol_shutdown", "subscription_ended"):
            # Clean shutdown - set stop event
            self.stop_event.set()
            logger.debug(f"Set stop_event due to {error_type}")
        else:
            # Error condition - set error event
            self.error_event.set()
            logger.debug(f"Set error_event due to {error_type}")
        
        # Also call the client's error callback if available
        if self.error_callback:
            try:
                await self.error_callback(error_type, exception)
            except Exception as e:
                logger.error(f"Error in client error callback: {e}")

    async def _ingress_loop(self):
        """Process incoming frames with native async support."""
        try:
            async for frame in self.protocol.ingress_loop(self.stop_event):
                # Check for error state or stop signal
                if self.error_event.is_set() or self.stop_event.is_set():
                    logger.info("Stopping ingress loop due to error or stop signal")
                    break
                
                # Process frames directly in ingress loop
                try:
                    if isinstance(frame, VideoFrame):
                        logger.debug(f"Processing video frame with frame processor: {frame.tensor.shape}")
                        
                        # Direct async processing
                        processed_frame = await self.frame_processor.process_video_async(frame)
                        if processed_frame:
                            output = VideoOutput(processed_frame, self.request_id)
                            await self._send_output(output)
                            logger.debug(f"Sent async processed video frame to egress")
                        else:
                            logger.warning(f"Frame processor returned None for video frame")
                            
                    elif isinstance(frame, AudioFrame):
                        logger.debug(f"Processing audio frame with frame processor: {frame.samples.shape}")
                        
                        # Direct async processing for audio
                        processed_frames = await self.frame_processor.process_audio_async(frame)
                        if processed_frames:
                            output = AudioOutput(processed_frames, self.request_id)
                            await self._send_output(output)
                            logger.debug(f"Sent async processed audio frame to egress")
                        else:
                            logger.warning(f"Frame processor returned None for audio frame")
                    else:
                        logger.debug(f"Received unknown frame type: {type(frame)}")
                        
                except Exception as e:
                    logger.error(f"Error in async frame processing: {e}")
                    
                    # Notify frame processor about the error
                    if self.error_callback:
                        try:
                            await self.error_callback("frame_processing_error", e)
                        except Exception as cb_error:
                            logger.error(f"Error in frame processing error callback: {cb_error}")
                    
                    # Still send the original frame as fallback
                    if isinstance(frame, VideoFrame):
                        fallback_output = VideoOutput(frame, self.request_id)
                        await self._send_output(fallback_output)
                    elif isinstance(frame, AudioFrame):
                        fallback_output = AudioOutput([frame], self.request_id)
                        await self._send_output(fallback_output)
            
            # Send sentinel to signal egress loop to complete
            logger.info("Ingress loop completed, sending sentinel to egress loop")
            await self._send_output(None)
            
            # Set stop event to signal other loops to complete
            self.stop_event.set()
            
        except Exception as e:
            logger.error(f"Error in ingress loop: {e}")
            # Set error state and stop event to trigger other loops to stop
            self.error_event.set()
            self.stop_event.set()
            # Notify parent if error callback is set
            if self.error_callback:
                try:
                    await self.error_callback("ingress_loop_error", e)
                except Exception as cb_error:
                    logger.error(f"Error in error callback: {cb_error}")
    
    async def _egress_loop(self):
        """Handle outgoing frames."""
        try:
            async def output_generator():
                """Generate output frames from the queue."""
                while not self.stop_event.is_set() and not self.error_event.is_set():
                    try:
                        # Try to get a frame from the queue with timeout
                        frame = await asyncio.to_thread(self.output_queue.get, timeout=0.1)
                        if frame is not None:
                            logger.debug("pulled frame from output queue")
                            yield frame
                        else:
                            # None frame indicates shutdown
                            break
                    except queue.Empty:
                        continue  # No frame available, continue loop
                    except Exception as e:
                        logger.error(f"Error getting frame from output queue: {e}")
                        continue
                    
            await self.protocol.egress_loop(output_generator())
            logger.info("Egress loop completed")
        except Exception as e:
            logger.error(f"Error in egress loop: {e}")
            # Set error state and stop event to trigger other loops to stop
            self.error_event.set()
            self.stop_event.set()
            # Notify parent if error callback is set
            if self.error_callback:
                try:
                    await self.error_callback("egress_loop_error", e)
                except Exception as cb_error:
                    logger.error(f"Error in error callback: {cb_error}")
    
    async def _control_loop(self):
        """Handle control messages."""
        try:
            async for control_data in self.protocol.control_loop(self.stop_event):
                # Check for error state or stop signal
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
                    await self.error_callback("control_loop_error", e)
                except Exception as cb_error:
                    logger.error(f"Error in error callback: {cb_error}")
    
    async def _send_data_loop(self):
        """Send data to the server every 333ms, batching all available items."""
        try:
            while not self.stop_event.is_set() and not self.error_event.is_set():
                # Wait for send_data_interval or until stop/error event is set
                if await self._wait_for_interval(self.send_data_interval):
                    break  # Stop or error event was set, exit loop
                # Pull all available items from the data_queue
                data_items = []
                while len(self.data_queue) > 0 and not self.stop_event.is_set() and not self.error_event.is_set():
                    data = self.data_queue.popleft()
                    if data is None:
                        # Sentinel value to stop loop
                        if data_items:
                            # Send any remaining items before stopping
                            break
                        else:
                            return  # No items to send, just stop
                    else:
                        data_items.append(data)
                
                # Send all collected data items
                if len(data_items) > 0:
                    try:
                        data_str = json.dumps(data_items) + "\n"
                    except Exception as e:
                        logger.error(f"Error serializing data items: {e}")
                        continue

                    await self.protocol.publish_data(data_str)
                
        except Exception as e:
            logger.error(f"Error in data sending loop: {e}")
            

    async def _handle_control_message(self, control_data: dict):
        """Handle a control message."""
        if self.control_handler:
            try:
                if asyncio.iscoroutinefunction(self.control_handler):
                    await self.control_handler(control_data)
                else:
                    self.control_handler(control_data)
            except Exception as e:
                logger.error(f"Error in control handler: {e}")
    
    async def _send_output(self, output):
        """Send output to the egress queue."""
        try:
            await asyncio.to_thread(self.output_queue.put, output, timeout=1.0)
        except queue.Full:
            logger.warning("Output queue is full, dropping frame")
        except Exception as e:
            logger.error(f"Error sending output: {e}")

    async def _wait_for_interval(self, interval: float):
        """Wait for the specified interval or until stop/error event is set.
        
        Returns:
            bool: True if stop/error event is set, False if timeout occurred (should continue)
        """
        try:
            done, pending = await asyncio.wait(
                [asyncio.create_task(self.stop_event.wait()), 
                asyncio.create_task(self.error_event.wait())],
                timeout=interval,
                return_when=asyncio.FIRST_COMPLETED
            )
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
            
            # Return True if any event is set (done set has completed tasks)
            return len(done) > 0
        except asyncio.TimeoutError:
            # Timeout means no event was set, should continue processing
            return False
        except Exception as e:
            logger.error(f"Error in wait_for_interval: {e}")
            # On error, signal to stop the loop
            return True
