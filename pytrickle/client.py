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
from .frame_skipper import AdaptiveFrameSkipper, FrameSkipConfig, FrameProcessingResult



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
        frame_skip_config: Optional[FrameSkipConfig] = None
    ):
        """Initialize TrickleClient with optional AdaptiveFrameSkipper for intelligent frame management.
        
        Args:
            protocol: TrickleProtocol instance
            frame_processor: FrameProcessor for native async processing
            control_handler: Optional control message handler
            error_callback: Optional error callback (if None, uses frame_processor.error_callback)
            max_queue_size: Maximum size for frame queues
            frame_skip_config: Optional frame skipping configuration (None = no frame skipping)
        """
        self.protocol = protocol
        self.frame_processor = frame_processor
        self.control_handler = control_handler
        self.send_data_interval = send_data_interval

        # Use provided error_callback, or fall back to frame_processor's error_callback
        self.error_callback = error_callback or frame_processor.error_callback
        
        # Queue configuration
        self.frame_skip_config = frame_skip_config
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
        
        # Separate frame processing queues for clean audio/video handling
        self.video_input_queue = asyncio.Queue(maxsize=max_queue_size)  # Video frames can be dropped
        self.audio_input_queue = asyncio.Queue(maxsize=max_queue_size * 4)  # Audio frames never dropped, larger buffer
        self.output_queue = asyncio.Queue(maxsize=200)
        
        # Data queue
        self.data_queue: Deque[Any] = deque(maxlen=1000)
        
        # Adaptive frame skipper for intelligent video frame management (optional)
        if frame_skip_config is not None:
            self.frame_skipper = AdaptiveFrameSkipper(
                config=frame_skip_config,
                fps_meter=protocol.fps_meter
            )
        else:
            self.frame_skipper = None
        

        
        # Frame sequence tracking for ordered output
        self._frame_sequence_counter = 0
        self._pending_outputs = {}  # sequence_id -> (output, timestamp)
        self._next_expected_sequence = 0
        self._sequence_lock = asyncio.Lock()
        self._sequence_timeout = 5.0  # Max seconds to wait for missing sequence
        
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
        except asyncio.QueueFull:
            pass
        
        # Send sentinel to data queue (deque doesn't raise QueueFull)
        self.data_queue.append(None)
    
    async def publish_data(self, data: str):
        """Publish data via the protocol's data publisher."""
        self.data_queue.append(data)

    def get_statistics(self) -> dict:
        """Get comprehensive processing statistics."""
        stats = {
            "video_input_queue_size": self.video_input_queue.qsize(),
            "audio_input_queue_size": self.audio_input_queue.qsize(),
            "output_queue_size": self.output_queue.qsize()
        }

        return stats
    
    def set_target_fps(self, target_fps: Optional[float]):
        """Set the target FPS for intelligent frame skipping.
        
        Args:
            target_fps: Target FPS value (None = auto-detect from ingress)
        """
        if self.frame_skipper:
            self.frame_skipper.set_target_fps(target_fps)
        else:
            logger.warning("Frame skipping is disabled, cannot set target FPS")
    
    def _get_next_sequence_id(self) -> int:
        """Get the next sequence ID for frame ordering."""
        sequence_id = self._frame_sequence_counter
        self._frame_sequence_counter += 1
        return sequence_id
    
    async def _queue_ordered_output(self, output, sequence_id: int):
        """Queue output with sequence ordering to maintain frame order."""
        async with self._sequence_lock:
            self._pending_outputs[sequence_id] = (output, time.time())
            
            # Output all consecutive frames starting from next expected sequence
            await self._flush_consecutive_outputs()
    
    async def _flush_consecutive_outputs(self):
        """Flush all consecutive outputs and handle timeouts for missing sequences."""
        current_time = time.time()
        
        # Output all consecutive frames starting from next expected sequence
        while self._next_expected_sequence in self._pending_outputs:
            pending_output, _ = self._pending_outputs.pop(self._next_expected_sequence)
            await self.output_queue.put(pending_output)
            self._next_expected_sequence += 1
        
        # Handle timeout for missing sequences - skip them to prevent blocking
        if self._pending_outputs:
            oldest_sequence = min(self._pending_outputs.keys())
            if oldest_sequence > self._next_expected_sequence:
                # Check if we've waited too long for the missing sequence
                oldest_timestamp = min(timestamp for _, timestamp in self._pending_outputs.values())
                if current_time - oldest_timestamp > self._sequence_timeout:
                    logger.warning(f"Sequence timeout: skipping missing sequence {self._next_expected_sequence}, jumping to {oldest_sequence}")
                    self._next_expected_sequence = oldest_sequence
                    # Flush consecutive outputs after advancing (non-recursive)
                    while self._next_expected_sequence in self._pending_outputs:
                        pending_output, _ = self._pending_outputs.pop(self._next_expected_sequence)
                        await self.output_queue.put(pending_output)
                        self._next_expected_sequence += 1
    
    async def _process_frame_with_ordering(self, frame: Union[VideoFrame, AudioFrame], sequence_id: int):
        """Process a frame and handle ordering, with proper error recovery."""
        try:
            if isinstance(frame, VideoFrame):
                logger.debug(f"Processing video frame with frame processor: {frame.tensor.shape} (seq: {sequence_id})")
                
                processed_frame = await self.frame_processor.process_video_async(frame)
                if processed_frame:
                    output = VideoOutput(processed_frame, self.request_id)
                    await self._queue_ordered_output(output, sequence_id)
                else:
                    logger.warning(f"Frame processor returned None for video frame (seq: {sequence_id})")
                    await self._advance_sequence_if_next(sequence_id)
                    
            elif isinstance(frame, AudioFrame):
                logger.debug(f"Processing audio frame with frame processor: {frame.samples.shape} (seq: {sequence_id})")
                
                processed_frames = await self.frame_processor.process_audio_async(frame)
                if processed_frames:
                    output = AudioOutput(processed_frames, self.request_id)
                    await self._queue_ordered_output(output, sequence_id)
                else:
                    logger.warning(f"Frame processor returned None for audio frame (seq: {sequence_id})")
                    await self._advance_sequence_if_next(sequence_id)
            else:
                logger.warning(f"Received unknown frame type: {type(frame)} (seq: {sequence_id})")
                await self._advance_sequence_if_next(sequence_id)
                
        except Exception as e:
            logger.error(f"Error processing frame (seq: {sequence_id}): {e}")
            
            # Send original frame as fallback with proper ordering
            try:
                if isinstance(frame, VideoFrame):
                    fallback_output = VideoOutput(frame, self.request_id)
                    await self._queue_ordered_output(fallback_output, sequence_id)
                elif isinstance(frame, AudioFrame):
                    fallback_output = AudioOutput([frame], self.request_id)
                    await self._queue_ordered_output(fallback_output, sequence_id)
                else:
                    # Unknown frame type - just advance sequence
                    await self._advance_sequence_if_next(sequence_id)
            except Exception as fallback_error:
                logger.error(f"Error in fallback handling (seq: {sequence_id}): {fallback_error}")
                await self._advance_sequence_if_next(sequence_id)
    
    async def _advance_sequence_if_next(self, sequence_id: int):
        """Advance sequence counter if this is the next expected sequence."""
        async with self._sequence_lock:
            if sequence_id == self._next_expected_sequence:
                self._next_expected_sequence += 1
                # Try to flush any subsequent consecutive outputs
                await self._flush_consecutive_outputs()
    


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
                
                # Route frames to appropriate queues based on type
                try:
                    if isinstance(frame, VideoFrame):
                        await self.video_input_queue.put(frame)
                    elif isinstance(frame, AudioFrame):
                        await self.audio_input_queue.put(frame)
                    else:
                        logger.warning(f"Unknown frame type received: {type(frame)}")
                except Exception as e:
                    logger.error(f"Error queueing frame for processing: {e}")
            
            # Send sentinels to both queues to signal processing loop to complete
            logger.info("Ingress loop completed, sending sentinels to processing loops")
            await self.video_input_queue.put(None)
            await self.audio_input_queue.put(None)
            
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
        """Process frames asynchronously from separate video and audio queues."""
        try:
            # Run video and audio processing concurrently
            await asyncio.gather(
                self._process_video_frames(),
                self._process_audio_frames(),
                return_exceptions=True
            )
            
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
                    await self.error_callback("processing_loop_error", e)
                except Exception as cb_error:
                    logger.error(f"Error in error callback: {cb_error}")
    
    async def _process_video_frames(self):
        """Process video frames with real-time adaptive skipping."""
        try:
            processing_times = []  # Track recent processing times for adaptive skipping
            
            while not self.stop_event.is_set() and not self.error_event.is_set():
                try:
                    # Get video frame with short timeout for responsiveness
                    try:
                        frame = await asyncio.wait_for(self.video_input_queue.get(), timeout=0.1)
                        if frame is None:
                            logger.info("Video processing received shutdown signal")
                            break
                    except asyncio.TimeoutError:
                        continue  # No video frame available
                    
                    # Real-time skipping decision based on current conditions
                    should_skip = False
                    
                    if self.frame_skipper:
                        # Check frame skipper pattern first
                        skip_result = self.frame_skipper._process_frame(frame)
                        if skip_result == FrameProcessingResult.SKIPPED:
                            should_skip = True
                        else:
                            # Additional adaptive skipping based on real-time conditions
                            queue_pressure = self.video_input_queue.qsize() / self.frame_skipper.config.max_queue_size
                            avg_processing_time = sum(processing_times[-10:]) / len(processing_times) if processing_times else 0.1
                            
                            # Skip if queue is building up or processing is slow
                            if queue_pressure > 0.6:  # Start skipping earlier
                                should_skip = True
                            elif avg_processing_time > 0.2:  # Skip if processing is taking too long
                                should_skip = True
                    
                    if should_skip:
                        continue  # Skip this video frame for smooth motion
                    
                    # Process video frame and track timing
                    start_time = time.time()
                    logger.debug(f"Processing video frame: {frame.tensor.shape}")
                    processed_frame = await self.frame_processor.process_video_async(frame)
                    processing_time = time.time() - start_time
                    
                    # Track processing times for adaptive behavior
                    processing_times.append(processing_time)
                    if len(processing_times) > 20:  # Keep recent history
                        processing_times.pop(0)
                    
                    if processed_frame:
                        output = VideoOutput(processed_frame, self.request_id)
                        await self.output_queue.put(output)
                    
                except Exception as e:
                    logger.error(f"Error processing video frame: {e}")
                    
        except Exception as e:
            logger.error(f"Error in video processing loop: {e}")
    
    async def _process_audio_frames(self):
        """Process audio frames without any skipping or reordering."""
        try:
            while not self.stop_event.is_set() and not self.error_event.is_set():
                try:
                    # Get audio frame - prioritize audio processing for continuity
                    try:
                        frame = await asyncio.wait_for(self.audio_input_queue.get(), timeout=0.05)
                        if frame is None:
                            logger.info("Audio processing received shutdown signal")
                            break
                    except asyncio.TimeoutError:
                        continue  # No audio frame available
                    
                    # Process audio frame immediately (no skipping)
                    logger.debug(f"Processing audio frame: {frame.samples.shape}")
                    processed_frames = await self.frame_processor.process_audio_async(frame)
                    if processed_frames:
                        output = AudioOutput(processed_frames, self.request_id)
                        await self.output_queue.put(output)
                    
                except Exception as e:
                    logger.error(f"Error processing audio frame: {e}")
                    
        except Exception as e:
            logger.error(f"Error in audio processing loop: {e}")
    
    async def _cleanup_video_queue_simple(self):
        """Light cleanup - only when queue is severely overflowing."""
        queue_size = self.video_input_queue.qsize()
        max_size = self.frame_skipper.config.max_queue_size
        
        # Only do emergency cleanup when severely overflowing
        if queue_size <= max_size * 1.5:
            return
        
        # Light cleanup - just drop a few frames to prevent total overflow
        frames_to_drop = min(5, queue_size - max_size)  # Drop max 5 frames at a time
        dropped = 0
        
        for _ in range(frames_to_drop):
            try:
                frame = self.video_input_queue.get_nowait()
                if frame is None:
                    # Sentinel - put it back and stop
                    await self.video_input_queue.put(frame)
                    break
                # Drop video frame (don't put it back)
                dropped += 1
            except asyncio.QueueEmpty:
                break
        
        if dropped > 0:
            logger.debug(f"Emergency cleanup: dropped {dropped} video frames (queue: {queue_size} → {self.video_input_queue.qsize()})")

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
                            # No audio timestamp modifications - preserve original timing
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
