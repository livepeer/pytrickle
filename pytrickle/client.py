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

logger = logging.getLogger(__name__)


class TrickleClient:
    """High-level client for trickle stream processing with native async support."""
    
    def __init__(
        self,
        protocol: TrickleProtocol,
        frame_processor: 'FrameProcessor',
        control_handler: Optional[Callable] = None,
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
        # Use provided error_callback, or fall back to frame_processor's error_callback
        self.error_callback = error_callback or frame_processor.error_callback
        
        # Client state
        self.running = False
        self.request_id = "default"
        
        # Coordination events
        self.stop_event = asyncio.Event()
        self.error_event = asyncio.Event()
        
        # Stream timing state (reset between streams to prevent A/V sync issues)
        self._stream_start_time = None
        self._first_video_timestamp = None
        self._first_audio_timestamp = None
        self._timestamp_offset = 0.0
        self._first_video_ts_sec: Optional[float] = None
        
        # Output queue for processed frames - use asyncio.Queue for async compatibility
        self.output_queue = asyncio.Queue()
        # Optional background drain task when frame_processor runs in queue mode
        self._processor_drain_task: Optional[asyncio.Task] = None
        
        # Egress coordination state
        self._video_emit_baseline: Optional[int] = None
        self._saw_first_video_output: bool = False
        self._buffered_audio_outputs: list = []
        self._audio_emit_offset_sec: Optional[float] = None
        # Ingress audio buffering before first video
        self._pre_video_audio_buffer: list = []
        
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
        # Drain any stale frames (including sentinel) from previous stream
        try:
            while not self.output_queue.empty():
                self.output_queue.get_nowait()
                self.output_queue.task_done()
        except Exception:
            pass
        
        logger.info(f"Starting trickle client with request_id={request_id}")
        
        # If the frame processor supports queue mode, start its queue workers
        try:
            if getattr(self.frame_processor, "queue_mode", False):
                await self.frame_processor.start_queue_workers()
                logger.info("Started frame processor queue workers")
        except Exception as e:
            logger.warning(f"Unable to start frame processor queue workers: {e}")

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
            
            # When protocol ends, pause frame processor to prevent processing stale frames
            try:
                if hasattr(self.frame_processor, 'pause_inputs'):
                    await self.frame_processor.pause_inputs()
                    logger.info("Frame processor inputs paused due to protocol end")
            except Exception as e:
                logger.warning(f"Could not pause frame processor on protocol end: {e}")
            
            await self.protocol.stop()
    
    async def stop(self):
        """Stop the trickle client."""
        if not self.running:
            return
            
        logger.info("Stopping trickle client")
        self.stop_event.set()
        
        # Send sentinel value to stop egress loop
        try:
            await self.output_queue.put(None)
        except asyncio.QueueFull:
            pass
            
        # Cancel drain task if running
        if self._processor_drain_task and not self._processor_drain_task.done():
            self._processor_drain_task.cancel()
            try:
                await self._processor_drain_task
            except asyncio.CancelledError:
                pass
                
        # Stop frame processor workers if running in queue mode
        try:
            if getattr(self.frame_processor, "queue_mode", False):
                await self.frame_processor.stop_queue_workers()
        except Exception as e:
            logger.warning(f"Unable to stop frame processor workers: {e}")
            
        # Stop the frame processor if it has a stop method (for ComfyStream integration)
        # Use stop() for model preservation during stream switches
        try:
            if hasattr(self.frame_processor, 'stop') and callable(self.frame_processor.stop):
                await self.frame_processor.stop()
                logger.info("Frame processor stopped (models preserved)")
        except Exception as e:
            logger.warning(f"Unable to stop frame processor: {e}")

    def _reset_timing_state(self):
        """Reset client-level timing state to prevent cross-stream timestamp conflicts."""
        # Clear all timing-related state to ensure fresh start
        self._stream_start_time = None
        self._first_video_timestamp = None
        self._first_audio_timestamp = None
        self._timestamp_offset = 0.0
        self._first_video_ts_sec = None
        
        # Clear any additional timing state that might exist
        if hasattr(self, '_last_video_timestamp'):
            self._last_video_timestamp = None
        if hasattr(self, '_last_audio_timestamp'):
            self._last_audio_timestamp = None
        # Reset egress gating state
        self._video_emit_baseline = None
        self._saw_first_video_output = False
        self._buffered_audio_outputs = []
        self._audio_emit_offset_sec = None
            
        logger.info("Client timing state reset for new stream")
    
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
                
                # Normalize timestamps to start from 0 for each stream type
                try:
                    if isinstance(frame, VideoFrame):
                        if self._first_video_timestamp is None:
                            self._first_video_timestamp = frame.timestamp
                        # Rebase PTS ensuring non-negative
                        frame.timestamp = max(0, frame.timestamp - self._first_video_timestamp)
                    elif isinstance(frame, AudioFrame):
                        if self._first_audio_timestamp is None:
                            self._first_audio_timestamp = frame.timestamp
                        # Rebase PTS ensuring non-negative
                        frame.timestamp = max(0, frame.timestamp - self._first_audio_timestamp)
                except Exception as e:
                    logger.debug(f"Timestamp normalization skipped due to error: {e}")
                
                # Process frames directly in ingress loop
                try:
                    if isinstance(frame, VideoFrame):
                        if getattr(self.frame_processor, "queue_mode", False):
                            # Enqueue for background processing
                            await self.frame_processor.enqueue_video_frame(frame)
                        else:
                            # Direct async processing
                            processed_frame = await self.frame_processor.process_video_async(frame)
                            if processed_frame:
                                output = VideoOutput(processed_frame, self.request_id)
                                await self._send_output(output)
                                logger.debug(f"Sent async processed video frame to egress")
                            else:
                                logger.warning(f"Frame processor returned None for video frame")
                            
                    elif isinstance(frame, AudioFrame):
                        if getattr(self.frame_processor, "queue_mode", False):
                            # Check if audio workers are disabled (passthrough mode)
                            audio_concurrency = self.frame_processor._audio_concurrency
                            if audio_concurrency == 0:
                                # Direct passthrough for audio when no workers - skip queuing
                                processed_frames = await self.frame_processor.process_audio_async(frame)
                                if processed_frames:
                                    output = AudioOutput(processed_frames, self.request_id)
                                    await self._send_output(output)
                                    logger.debug(f"Sent passthrough audio frame to egress (no workers)")
                                else:
                                    logger.warning(f"Frame processor returned None for passthrough audio frame")
                            else:
                                # Normal queue mode with audio workers
                                await self.frame_processor.enqueue_audio_frame(frame)
                        else:
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
            # If processor is running in queue mode, start a drain task to push outputs into output_queue
            if getattr(self.frame_processor, "queue_mode", False) and not self._processor_drain_task:
                self._processor_drain_task = asyncio.create_task(self._drain_processor_outputs())

            async def output_generator():
                """Generate output frames from the queue."""
                def _shift_audio_output(ao):
                    # Apply a constant offset (in seconds) to each audio frame timestamp
                    try:
                        if self._audio_emit_offset_sec is None:
                            return ao
                        from .frames import AudioFrame, AudioOutput
                        shifted_frames = []
                        for af in ao.frames:
                            tb = float(af.time_base)
                            if tb <= 0:
                                shifted_frames.append(af)
                                continue
                            shift_pts = int(round(self._audio_emit_offset_sec / tb))
                            new_ts = max(0, af.timestamp + shift_pts)
                            new_af = AudioFrame._from_existing_with_timestamp(af, new_ts)
                            shifted_frames.append(new_af)
                        return AudioOutput(shifted_frames, ao.request_id)
                    except Exception:
                        return ao
                while not self.stop_event.is_set() and not self.error_event.is_set():
                    try:
                        # Try to get a frame from the queue with timeout
                        frame = await asyncio.wait_for(self.output_queue.get(), timeout=0.1)
                        if frame is not None:
                            # Gate audio until first video is emitted to avoid encoder A/V mismatch on startup
                            from .frames import AudioOutput, VideoOutput
                            if isinstance(frame, VideoOutput):
                                # First time we see video, flip the flag and flush any buffered audio
                                if not self._saw_first_video_output:
                                    self._saw_first_video_output = True
                                    # Establish audio offset relative to first video and first buffered audio (if any)
                                    try:
                                        video_ts_sec = float(frame.timestamp * frame.time_base)
                                    except Exception:
                                        video_ts_sec = 0.0
                                    if self._buffered_audio_outputs:
                                        try:
                                            first_ao = self._buffered_audio_outputs[0]
                                            first_af = first_ao.frames[0] if first_ao.frames else None
                                            if first_af is not None:
                                                first_audio_ts_sec = float(first_af.timestamp * first_af.time_base)
                                                # Shift audio forward so first audio aligns with first video
                                                self._audio_emit_offset_sec = max(0.0, video_ts_sec - first_audio_ts_sec)
                                        except Exception:
                                            self._audio_emit_offset_sec = None
                                    yield frame
                                    # Flush buffered audio after first video frame
                                    while self._buffered_audio_outputs:
                                        buffered = self._buffered_audio_outputs.pop(0)
                                        yield _shift_audio_output(buffered)
                                else:
                                    yield frame
                            elif isinstance(frame, AudioOutput):
                                if not self._saw_first_video_output:
                                    # Buffer limited amount of audio until video arrives
                                    if len(self._buffered_audio_outputs) > 200:
                                        # Drop oldest to keep buffer bounded (~4s at 20ms)
                                        self._buffered_audio_outputs.pop(0)
                                    self._buffered_audio_outputs.append(frame)
                                else:
                                    yield _shift_audio_output(frame)
                            else:
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

    async def _drain_processor_outputs(self):
        """Drain processed frames from the FrameProcessor queues into output_queue."""
        try:
            while not self.stop_event.is_set() and not self.error_event.is_set():
                # Create tasks for next available outputs
                tasks = []
                
                # Always try to get video output
                try:
                    tasks.append(asyncio.create_task(self.frame_processor.get_next_processed_video()))
                except Exception:
                    pass
                
                # Only try to get audio output if audio workers are enabled
                audio_concurrency = self.frame_processor._audio_concurrency
                if audio_concurrency > 0:
                    try:
                        tasks.append(asyncio.create_task(self.frame_processor.get_next_processed_audio()))
                    except Exception:
                        pass
                else:
                    logger.debug("Skipping audio drain - audio workers disabled (passthrough mode)")
                
                if not tasks:
                    await asyncio.sleep(0.01)
                    continue
                    
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                for t in done:
                    try:
                        result = await t
                        if result is None:
                            continue
                        # Determine type based on method returned
                        if isinstance(result, list):
                            # Audio
                            output = AudioOutput(result, self.request_id)
                        else:
                            # Video
                            # Rebase video timestamps relative to first emitted video to start at 0
                            try:
                                if self._video_emit_baseline is None:
                                    self._video_emit_baseline = getattr(result, 'timestamp', 0)
                                # Adjust in-place to avoid extra allocations
                                if hasattr(result, 'timestamp') and isinstance(getattr(result, 'timestamp'), (int, float)):
                                    adjusted = int(max(0, result.timestamp - self._video_emit_baseline))
                                    result.timestamp = adjusted
                            except Exception:
                                pass
                            output = VideoOutput(result, self.request_id)
                        await self._send_output(output)
                    except asyncio.CancelledError:
                        return
                    except Exception as e:
                        logger.debug(f"Processor drain error: {e}")
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error(f"Drain task error: {e}")
    
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
        # Built-in processor control commands
        try:
            cmd = control_data.get("command") if isinstance(control_data, dict) else None
            if cmd and getattr(self.frame_processor, "queue_mode", False):
                if cmd == "processor.start_workers":
                    if hasattr(self.frame_processor, "start_queue_workers"):
                        if not getattr(self.frame_processor, '_workers_started', False):
                            await self.frame_processor.start_queue_workers()
                            self.frame_processor._workers_started = True
                        return
                elif cmd == "processor.stop_workers":
                    if hasattr(self.frame_processor, "stop_queue_workers"):
                        await self.frame_processor.stop_queue_workers()
                        self.frame_processor._workers_started = False
                        return
                elif cmd == "processor.reset_timing":
                    if hasattr(self.frame_processor, "reset_timing"):
                        await self.frame_processor.reset_timing()
                        return
                elif cmd == "processor.reset_state":
                    if hasattr(self.frame_processor, "reset_state"):
                        await self.frame_processor.reset_state()
                        return
                elif cmd == "processor.pause_inputs":
                    if hasattr(self.frame_processor, "pause_inputs"):
                        await self.frame_processor.pause_inputs()
                        return
                elif cmd == "processor.resume_inputs":
                    if hasattr(self.frame_processor, "resume_inputs"):
                        await self.frame_processor.resume_inputs()
                        return
        except Exception as e:
            logger.error(f"Error handling built-in control command: {e}")

        # User-provided control handler fallback
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
            await self.output_queue.put(output)
        except asyncio.QueueFull:
            logger.warning("Output queue is full, dropping frame")
        except Exception as e:
            logger.error(f"Error sending output: {e}")

 