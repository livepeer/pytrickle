"""
Trickle Client for processing video streams.

Coordinates ingress, egress, and control loops with proper shutdown handling
to ensure all components stop when subscription ends.
"""

import asyncio
import logging
import time
import json
from typing import Callable, Optional, Union, Deque, Any
from collections import deque

from .protocol import TrickleProtocol
from .frames import VideoFrame, AudioFrame, VideoOutput, AudioOutput
from . import ErrorCallback
from .frame_processor import FrameProcessor
from .decoder import DEFAULT_MAX_FRAMERATE
from .frame_skipper import AdaptiveFrameSkipper
from .monotonic_audio import MonotonicAudioTracker
from .frame_processing_types import FrameProcessingResult, is_processing_result, is_frame

logger = logging.getLogger(__name__)


class TrickleClient:
    """High-level client for trickle stream processing with native async support."""
    
    def __init__(
        self,
        protocol: TrickleProtocol,
        frame_processor: 'FrameProcessor',
        control_handler: Optional[Callable] = None,
        send_data_interval: Optional[float] = 0.333,
        error_callback: Optional[ErrorCallback] = None,
        max_queue_size: int = 300,
        enable_frame_skipping: bool = True,
        target_fps: Optional[float] = None,
        auto_target_fps: bool = True
    ):
        """Initialize TrickleClient with optional AdaptiveFrameSkipper for intelligent frame management.
        
        Args:
            protocol: TrickleProtocol instance
            frame_processor: FrameProcessor for native async processing
            control_handler: Optional control message handler
            error_callback: Optional error callback (if None, uses frame_processor.error_callback)
            max_queue_size: Maximum size for frame queues
            enable_frame_skipping: Whether to enable intelligent frame skipping
            target_fps: Target FPS for intelligent skipping (None = auto-detect)
            auto_target_fps: Whether to automatically detect and use ingress FPS as target
        """
        self.protocol = protocol
        self.frame_processor = frame_processor
        self.control_handler = control_handler
        self.send_data_interval = send_data_interval

        # Use provided error_callback, or fall back to frame_processor's error_callback
        self.error_callback = error_callback or frame_processor.error_callback
        
        # Queue configuration
        self.enable_frame_skipping = enable_frame_skipping
        self.target_fps = target_fps
        self.auto_target_fps = auto_target_fps
        self.max_queue_size = max_queue_size
        
        # Connect protocol error callback to client error handling
        if not self.protocol.error_callback:
            self.protocol.error_callback = self._on_protocol_error
        
        # Client state
        self.running = False
        self.request_id = "default"
        
        # Coordination events
        self.stop_event = asyncio.Event()
        self.error_event = asyncio.Event()
        
        # Frame processing queues
        self.input_queue = asyncio.Queue(maxsize=max_queue_size)
        self.output_queue = asyncio.Queue(maxsize=200)
        
        # Data queue
        self.data_queue: Deque[Any] = deque(maxlen=1000)
        
        # Adaptive frame skipper for intelligent video frame management (optional)
        if enable_frame_skipping:
            self.frame_skipper = AdaptiveFrameSkipper(
                target_fps=target_fps or float(DEFAULT_MAX_FRAMERATE),
                auto_target_fps=auto_target_fps,
                skip_pattern="uniform",  # Use uniform skipping pattern
                adaptation_window=2.0    # 2 second adaptation window
            )
        else:
            self.frame_skipper = None
        
        # Monotonic audio timeline tracker for automatic A/V sync
        self.audio_timeline_tracker = MonotonicAudioTracker(frame_duration_ms=20)
        
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
                self._processing_loop(),
                self._egress_loop(),
                self._control_loop(),
                self._send_data_loop(),
                return_exceptions=True
            )
            
            # Check if any loop had an exception that is not a cancelled error
            for i, result in enumerate(results):
                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                    loop_names = ["ingress", "processing", "egress", "control", "send_data"]
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
        
        # Send sentinel values to stop processing and egress loops
        try:
            self.output_queue.put_nowait(None)
            self.data_queue.append(None)
        except queue.Full:
            pass
    
    async def publish_data(self, data: str):
        """Publish data via the protocol's data publisher."""
        self.data_queue.append(data)

    def get_statistics(self) -> dict:
        """Get comprehensive processing statistics."""
        stats = {
            "input_queue_size": self.input_queue.qsize(),
            "output_queue_size": self.output_queue.qsize()
        }

        return stats
    
    def set_target_fps(self, target_fps: float):
        """Manually set the target FPS for intelligent frame skipping.
        
        Args:
            target_fps: New target FPS value
        """
        if self.frame_skipper:
            self.frame_skipper.update_target_fps(target_fps)
            logger.info(f"Manually set target FPS to {target_fps}")
        else:
            logger.warning("Frame skipping is disabled, cannot set target FPS")
    
    def enable_auto_target_fps(self):
        """Re-enable automatic target FPS detection."""
        if self.frame_skipper:
            self.frame_skipper.enable_auto_target_fps()
        else:
            logger.warning("Frame skipping is disabled, cannot enable auto target FPS")

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
        """Receive incoming frames and queue them with automatic audio timeline correction."""
        try:
            async for frame in self.protocol.ingress_loop(self.stop_event):
                # Check for error state or stop signal
                if self.error_event.is_set() or self.stop_event.is_set():
                    logger.info("Stopping ingress loop due to error or stop signal")
                    break
                
                # Apply automatic monotonic audio timeline correction
                if isinstance(frame, AudioFrame):
                    frame = self.audio_timeline_tracker.process_audio_frame(frame)
                    logger.debug("Queued audio frame with corrected monotonic timestamp")
                else:
                    logger.debug(f"Queued {type(frame).__name__} for async processing")
                
                # Queue frames for processing - audio frames now have correct timestamps
                try:
                    await self.input_queue.put(frame)
                except Exception as e:
                    logger.error(f"Error queueing frame for processing: {e}")
            
            # Send sentinel to signal processing loop to complete
            logger.info("Ingress loop completed, sending sentinel to processing loop")
            await self.input_queue.put(None)
            
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
                    if self.frame_skipper:
                        # Use frame skipper for intelligent frame management
                        frame_or_result = await self.frame_skipper.process_queue_with_skipping(self.input_queue, timeout=5)
                        
                        # Handle processing results (non-frame returns)
                        if is_processing_result(frame_or_result):
                            if frame_or_result == FrameProcessingResult.SHUTDOWN:
                                # Sentinel value received, break out of processing loop
                                logger.info("Processing loop received shutdown signal, ending")
                                break
                            elif frame_or_result == FrameProcessingResult.FRAME_SKIPPED:
                                # Frame was skipped, continue to get next frame
                                continue
                            elif frame_or_result == FrameProcessingResult.TIMEOUT:
                                # Timeout occurred, continue to try again
                                continue
                            else:
                                # Unknown processing result, log and continue
                                logger.warning(f"Unknown frame processing result: {frame_or_result}")
                                continue
                        
                        # At this point, we have a valid frame
                        frame = frame_or_result
                    else:
                        # No frame skipping - process all frames directly
                        try:
                            frame = await asyncio.wait_for(self.input_queue.get(), timeout=5.0)
                            if frame is None:
                                # Sentinel value received, break out of processing loop
                                logger.info("Processing loop received shutdown signal, ending")
                                break
                        except asyncio.TimeoutError:
                            # Timeout occurred, continue to try again
                            continue
                    
                    # Process frames asynchronously
                    if isinstance(frame, VideoFrame):
                        logger.debug(f"Processing video frame with frame processor: {frame.tensor.shape}")
                        
                        # Monitor queue sizes for potential issues
                        queue_size = self.input_queue.qsize()
                        output_queue_size = self.output_queue.qsize()
                        if queue_size > self.max_queue_size * 0.8 or output_queue_size > self.max_queue_size * 0.8:
                            logger.warning(f"High queue usage - input: {queue_size}/{self.max_queue_size}, output: {output_queue_size}/{self.max_queue_size}")
                        
                        # Frame skipper already handled intelligent video frame skipping
                        # Async processing
                        processed_frame = await self.frame_processor.process_video_async(frame)
                        if processed_frame:
                            output = VideoOutput(processed_frame, self.request_id)
                            await self.output_queue.put(output)
                            logger.debug(f"Sent async processed video frame to egress")
                        else:
                            logger.warning(f"Frame processor returned None for video frame")
                            
                    elif isinstance(frame, AudioFrame):
                        logger.debug(f"Processing audio frame with frame processor: {frame.samples.shape}")
                        
                        # Audio frames are never skipped - always process them
                        processed_frames = await self.frame_processor.process_audio_async(frame)
                        if processed_frames:
                            output = AudioOutput(processed_frames, self.request_id)
                            await self.output_queue.put(output)
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
                            await self.error_callback("frame_processing_error", e)
                        except Exception as cb_error:
                            logger.error(f"Error in frame processing error callback: {cb_error}")
                    
                    # Still send the original frame as fallback if we have it
                    if 'frame' in locals() and frame is not None:
                        if isinstance(frame, VideoFrame):
                            fallback_output = VideoOutput(frame, self.request_id)
                            await self.output_queue.put(fallback_output)
                        elif isinstance(frame, AudioFrame):
                            fallback_output = AudioOutput([frame], self.request_id)
                            await self.output_queue.put(fallback_output)
            
            # Send sentinel to signal egress loop to complete
            logger.info("Processing loop completed, sending sentinel to egress loop")
            await self.output_queue.put(None)
            
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
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
                """Generate output frames from the output queue."""
                while not self.stop_event.is_set() and not self.error_event.is_set():
                    try:
                        # Get frame from output queue
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
