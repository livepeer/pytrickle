"""
Trickle Client for processing video streams.

Coordinates ingress, egress, and control loops with proper shutdown handling
to ensure all components stop when subscription ends.
"""

import asyncio
import queue
import logging
from typing import Callable, Optional, Union, List
from collections import deque

from .protocol import TrickleProtocol
from .frames import VideoFrame, AudioFrame, VideoOutput, AudioOutput, TextOutput
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
        error_callback: Optional[ErrorCallback] = None,
        text_queue: Optional[deque] = None
    ):
        """Initialize TrickleClient.
        
        Args:
            protocol: TrickleProtocol instance
            frame_processor: FrameProcessor for native async processing
            control_handler: Optional control message handler
            error_callback: Optional error callback (if None, uses frame_processor.error_callback)
            text_queue: Optional deque of strings to publish as text data
        """
        self.protocol = protocol
        self.frame_processor = frame_processor
        self.control_handler = control_handler
        # Use provided error_callback, or fall back to frame_processor's error_callback
        self.error_callback = error_callback or frame_processor.error_callback
        self.text_queue = text_queue  # Optional text queue for apps to use
        
        # Client state
        self.running = False
        self.request_id = "default"
        
        # Coordination events
        self.stop_event = asyncio.Event()
        self.error_event = asyncio.Event()
        
        # Output queue for processed frames
        self.output_queue = queue.Queue()
        
        
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
                return_exceptions=True
            )
            
            # Check if any loop had an exception that is not a cancelled error
            for i, result in enumerate(results):
                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                    loop_names = ["ingress", "egress", "control"]
                    logger.error(f"{loop_names[i]} loop failed: {result}")
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in client loops: {e}")
        finally:
            self.running = False
            logger.info("Stopping protocol due to client loops ending")
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
    
    async def publish_data(self, data: str):
        """Publish data via the protocol's data publisher."""
        return await self.protocol.publish_data(data)
    
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
                        logger.debug(
                            f"Processing video frame with frame processor: {frame.tensor.shape}"
                        )

                        # Direct async processing
                        processed_frame = (
                            await self.frame_processor.process_video_async(frame)
                        )
                        if processed_frame:
                            output = VideoOutput(processed_frame, self.request_id)
                            await self._send_output(output)
                            logger.debug(f"Sent async processed video frame to egress")
                        else:
                            logger.warning(f"Frame processor returned None for video frame")
                            
                        # Check for text outputs from app text queue
                        await self._process_app_text_queue()
                            
                    elif isinstance(frame, AudioFrame):
                        logger.debug(
                            f"Processing audio frame with frame processor: {frame.samples.shape}"
                        )

                        # Direct async processing for audio
                        processed_frames = (
                            await self.frame_processor.process_audio_async(frame)
                        )
                        if processed_frames:
                            output = AudioOutput(processed_frames, self.request_id)
                            await self._send_output(output)
                            logger.debug(f"Sent async processed audio frame to egress")
                        else:
                            logger.warning(
                                f"Frame processor returned None for audio frame"
                            )
                            
                        # Check for text outputs from app text queue
                        await self._process_app_text_queue()

                    else:
                        logger.debug(f"Received unknown frame type: {type(frame)}")
                        
                except Exception as e:
                    logger.error(f"Error in async frame processing: {e}")
                    
                    # Notify frame processor about the error
                    if self.error_callback:
                        try:
                            if asyncio.iscoroutinefunction(self.error_callback):
                                await self.error_callback("frame_processing_error", e)
                            else:
                                self.error_callback("frame_processing_error", e)
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
            async def output_generator():
                """Generate output frames from the queue, filtering out TextOutput frames."""
                while not self.stop_event.is_set() and not self.error_event.is_set():
                    try:
                        # Try to get a frame from the queue with timeout
                        frame = await asyncio.to_thread(self.output_queue.get, timeout=0.1)
                        if frame is None:
                            # None frame indicates shutdown
                            break
                        elif isinstance(frame, TextOutput):
                            # Handle TextOutput frames separately
                            await self._handle_text_output(frame)
                        else:
                            # Yield video/audio frames to protocol egress
                            yield frame
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
                    if asyncio.iscoroutinefunction(self.error_callback):
                        await self.error_callback("control_loop_error", e)
                    else:
                        self.error_callback("control_loop_error", e)
                except Exception as cb_error:
                    logger.error(f"Error in error callback: {cb_error}")
    
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
    
    async def _process_app_text_queue(self):
        """Process any queued text strings from the app's text queue and send to egress."""
        if not self.text_queue:
            return
            
        try:
            # Process all available text strings from the app's queue
            texts_processed = 0
            while self.text_queue and not self.stop_event.is_set() and not self.error_event.is_set():
                try:
                    text_string = self.text_queue.popleft()
                    if text_string and text_string.strip():
                        # Create TextOutput and send to egress
                        text_output = TextOutput(text=text_string, request_id=self.request_id)
                        await self._send_output(text_output)
                        texts_processed += 1
                        logger.debug(f"Publishing: {text_string[:50]}...")
                except IndexError:
                    # Queue is empty
                    break
            
            if texts_processed > 0:
                logger.debug(f"Processed {texts_processed} text outputs from app queue")
                
        except Exception as e:
            logger.error(f"Error processing app text queue: {e}")
    
    async def _handle_text_output(self, text_output: TextOutput):
        """Handle TextOutput frame by publishing via protocol."""
        try:
            if self.protocol.data_publisher:
                await self.publish_data(text_output.text)
                logger.info(f"✅ Published to data URL: {text_output.text[:50]}...")
            else:
                logger.warning(f"❌ No data URL configured - cannot publish: {text_output.text[:50]}...")
        except Exception as e:
            logger.error(f"Error handling text output: {e}")