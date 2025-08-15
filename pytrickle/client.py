"""
Trickle Client for processing video streams.

Coordinates ingress, egress, and control loops with proper shutdown handling
to ensure all components stop when subscription ends.
"""

import asyncio
import logging
from typing import Callable, Optional, Union

from .protocol import TrickleProtocol
from .frames import VideoFrame, AudioFrame, VideoOutput, AudioOutput
from . import ErrorCallback
from .frame_processor import FrameProcessor
from .frame_skipper import AdaptiveFrameSkipper

logger = logging.getLogger(__name__)


class TrickleClient:
    """High-level client for trickle stream processing with native async support."""
    
    def __init__(
        self,
        protocol: TrickleProtocol,
        frame_processor: 'FrameProcessor',
        control_handler: Optional[Callable] = None,
        error_callback: Optional[ErrorCallback] = None,
        enable_frame_skipping: bool = True,
        target_fps: float = 30.0,
        frame_skip_config: Optional[dict] = None
    ):
        """Initialize TrickleClient.
        
        Args:
            protocol: TrickleProtocol instance
            frame_processor: FrameProcessor for native async processing
            control_handler: Optional control message handler
            error_callback: Optional error callback (if None, uses frame_processor.error_callback)
            enable_frame_skipping: Whether to enable adaptive frame skipping
            target_fps: Target output FPS for frame skipping
            frame_skip_config: Additional configuration for frame skipper
        """
        self.protocol = protocol
        self.frame_processor = frame_processor
        self.control_handler = control_handler
        # Use provided error_callback, or fall back to frame_processor's error_callback
        self.error_callback = error_callback or frame_processor.error_callback
        
        # Frame skipping setup
        self.enable_frame_skipping = enable_frame_skipping
        if self.enable_frame_skipping:
            skip_config = frame_skip_config or {}
            self.frame_skipper = AdaptiveFrameSkipper(
                target_fps=target_fps,
                **skip_config
            )
        else:
            self.frame_skipper = None
        
        # Client state
        self.running = False
        self.request_id = "default"
        
        # Coordination events
        self.stop_event = asyncio.Event()
        self.error_event = asyncio.Event()
        
        # Queues for frame processing pipeline with limited capacity
        self.input_queue = asyncio.Queue(maxsize=120)  # Raw frames from ingress
        self.output_queue = asyncio.Queue(maxsize=120)  # Processed frames for egress
        
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
            # Run all loops concurrently - added processing loop
            results = await asyncio.gather(
                self._ingress_loop(),
                self._processing_loop(),  # New async processing loop
                self._egress_loop(),
                self._control_loop(),
                return_exceptions=True
            )
            
            # Check if any loop had an exception that is not a cancelled error
            for i, result in enumerate(results):
                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                    loop_names = ["ingress", "processing", "egress", "control"]
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
        
        # Send sentinel values to stop processing and egress loops
        try:
            await self._put_with_overflow_protection(self.input_queue, None, "input_queue")  # Stop processing loop
            await self._put_with_overflow_protection(self.output_queue, None, "output_queue")  # Stop egress loop
        except Exception:
            pass
    
    async def publish_data(self, data: str):
        """Publish data via the protocol's data publisher."""
        return await self.protocol.publish_data(data)
    
    def get_frame_skip_statistics(self) -> Optional[dict]:
        """Get frame skipping performance statistics."""
        if self.frame_skipper:
            return self.frame_skipper.get_statistics()
        return None
    
    async def _put_with_overflow_protection(self, queue: asyncio.Queue, item, queue_name: str = "queue"):
        """Put item in queue, removing oldest items if queue is full. Preserves audio frames."""
        if queue.full():
            # Try to find and remove the oldest video frame instead of any frame
            removed_item = None
            temp_items = []
            
            # Extract items to find a video frame to remove
            while not queue.empty():
                try:
                    temp_item = queue.get_nowait()
                    if (removed_item is None and 
                        hasattr(temp_item, '__class__') and 
                        'Video' in temp_item.__class__.__name__):
                        # Found a video frame to remove
                        removed_item = temp_item
                        logger.debug(f"Removed oldest video item from {queue_name} to prevent overflow")
                        break
                    else:
                        temp_items.append(temp_item)
                except asyncio.QueueEmpty:
                    break
            
            # Put back the items we didn't remove
            for temp_item in temp_items:
                try:
                    queue.put_nowait(temp_item)
                except asyncio.QueueFull:
                    # If we can't put it back, we have a serious problem
                    logger.warning(f"Lost item during overflow protection in {queue_name}")
                    break
            
            # If we couldn't find a video frame and queue is still full, 
            # only then consider removing other items (but still prefer non-audio)
            if removed_item is None and queue.full():
                try:
                    removed_item = queue.get_nowait()
                    if (hasattr(removed_item, '__class__') and 
                        'Audio' in removed_item.__class__.__name__):
                        logger.warning(f"Had to remove audio item from {queue_name} - queue critically full")
                    else:
                        logger.debug(f"Removed oldest item from {queue_name} to prevent overflow")
                except asyncio.QueueEmpty:
                    pass
        
        try:
            await queue.put(item)
        except asyncio.QueueFull:
            # If queue is still full and the new item is audio, try harder to make space
            if (hasattr(item, '__class__') and 'Audio' in item.__class__.__name__):
                # For audio, try removing one more video frame
                try:
                    temp_items = []
                    removed_video = False
                    
                    while not queue.empty() and not removed_video:
                        temp_item = queue.get_nowait()
                        if (hasattr(temp_item, '__class__') and 
                            'Video' in temp_item.__class__.__name__):
                            removed_video = True
                            logger.debug(f"Removed video frame from {queue_name} to make space for audio")
                            break
                        else:
                            temp_items.append(temp_item)
                    
                    # Put back non-video items
                    for temp_item in temp_items:
                        try:
                            queue.put_nowait(temp_item)
                        except asyncio.QueueFull:
                            break
                    
                    await queue.put(item)
                    logger.debug(f"Successfully queued audio frame after making space in {queue_name}")
                    
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    logger.error(f"Could not queue audio frame in {queue_name} - dropping audio!")
                    # As last resort, drop the audio frame
            else:
                # For non-audio items (video), just drop the new item
                logger.warning(f"Could not add video item to {queue_name} due to overflow - dropping frame")
                # Drop the new item to prevent blocking
    
    async def _ingress_loop(self):
        """Receive incoming frames and queue them for async processing."""
        try:
            async for frame in self.protocol.ingress_loop(self.stop_event):
                # Check for error state or stop signal
                if self.error_event.is_set() or self.stop_event.is_set():
                    logger.info("Stopping ingress loop due to error or stop signal")
                    break
                
                # Queue frame for async processing with overflow protection
                try:
                    await self._put_with_overflow_protection(self.input_queue, frame, "input_queue")
                    logger.debug(f"Queued {type(frame).__name__} for async processing")
                except Exception as e:
                    logger.error(f"Error queueing frame for processing: {e}")
            
            # Send sentinel to signal processing loop to complete
            logger.info("Ingress loop completed, sending sentinel to processing loop")
            await self._put_with_overflow_protection(self.input_queue, None, "input_queue")
            
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

    async def _processing_loop(self):
        """Process frames asynchronously from the input queue."""
        try:
            while not self.stop_event.is_set() and not self.error_event.is_set():
                try:
                    # Use frame skipper's intelligent queue processing if available
                    if self.frame_skipper:
                        frame = await self.frame_skipper.process_queue_with_skipping(self.input_queue, timeout=5)
                    else:
                        # Fallback to standard queue processing
                        frame = await asyncio.wait_for(self.input_queue.get(), timeout=5)
                    
                    if frame is None:
                        # None frame indicates shutdown
                        logger.info("Processing loop received sentinel, ending")
                        break
                    
                    # Process frames asynchronously
                    if isinstance(frame, VideoFrame):
                        queue_size = self.input_queue.qsize()
                        output_queue_size = self.output_queue.qsize()
                        
                        logger.debug(f"Processing video frame with frame processor: {frame.tensor.shape}")
                        
                        # Log queue status more frequently when queues are filling up
                        if queue_size > 50 or output_queue_size > 50:
                            logger.warning(f"Queues filling up - input: {queue_size}/120, output: {output_queue_size}/120")
                        elif queue_size > 20 or output_queue_size > 20:
                            logger.info(f"Queue status - input: {queue_size}/120, output: {output_queue_size}/120")
                        else:
                            logger.debug(f"Queue status - input: {queue_size}/120, output: {output_queue_size}/120")
                        
                        # Frame skipping already handled in process_queue_with_skipping for video frames
                        # Async processing
                        processed_frame = await self.frame_processor.process_video_async(frame)
                        if processed_frame:
                            output = VideoOutput(processed_frame, self.request_id)
                            await self._send_output(output)
                            logger.debug(f"Sent async processed video frame to egress")
                        else:
                            logger.warning(f"Frame processor returned None for video frame")
                            
                    elif isinstance(frame, AudioFrame):
                        logger.debug(f"Processing audio frame with frame processor: {frame.samples.shape}")
                        
                        # Audio frames are never skipped - always process them
                        processed_frames = await self.frame_processor.process_audio_async(frame)
                        if processed_frames:
                            output = AudioOutput(processed_frames, self.request_id)
                            await self._send_output(output)
                            logger.debug(f"Sent async processed audio frame to egress")
                        else:
                            logger.warning(f"Frame processor returned None for audio frame")
                    else:
                        logger.debug(f"Received unknown frame type: {type(frame)}")
                        
                except asyncio.TimeoutError:
                    continue  # No frame available, continue loop
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
                    
                    # Still send the original frame as fallback if we have it
                    if 'frame' in locals() and frame is not None:
                        if isinstance(frame, VideoFrame):
                            fallback_output = VideoOutput(frame, self.request_id)
                            await self._send_output(fallback_output)
                        elif isinstance(frame, AudioFrame):
                            fallback_output = AudioOutput([frame], self.request_id)
                            await self._send_output(fallback_output)
            
            # Send sentinel to signal egress loop to complete
            logger.info("Processing loop completed, sending sentinel to egress loop")
            await self._send_output(None)
            
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
            # Set error state and stop event to trigger other loops to stop
            self.error_event.set()
            self.stop_event.set()
            # Notify parent if error callback is set
            if self.error_callback:
                try:
                    if asyncio.iscoroutinefunction(self.error_callback):
                        await self.error_callback("processing_loop_error", e)
                    else:
                        self.error_callback("processing_loop_error", e)
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
                        frame = await asyncio.wait_for(self.output_queue.get(), timeout=0.5)
                        if frame is not None:
                            yield frame
                        else:
                            # None frame indicates shutdown
                            break
                    except asyncio.TimeoutError:
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
        """Send output to the egress queue with overflow protection."""
        try:
            await self._put_with_overflow_protection(self.output_queue, output, "output_queue")
        except Exception as e:
            logger.error(f"Error sending output: {e}")

 