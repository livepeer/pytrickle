"""
Trickle Client for processing video streams.

Coordinates ingress, egress, and control loops with proper shutdown handling
to ensure all components stop when subscription ends.
"""

import asyncio
import queue
import logging
from typing import Callable, Optional

from .protocol import TrickleProtocol
from .frames import VideoFrame, OutputFrame
from . import ErrorCallback

logger = logging.getLogger(__name__)


class TrickleClient:
    """High-level client for trickle stream processing."""
    
    def __init__(
        self,
        protocol: TrickleProtocol,
        frame_processor: Callable,
        control_handler: Optional[Callable] = None,
        error_callback: Optional[ErrorCallback] = None
    ):
        self.protocol = protocol
        self.frame_processor = frame_processor
        self.control_handler = control_handler
        self.error_callback = error_callback
        
        # Client state
        self.running = False
        self.request_id = "default"
        
        # Coordination events
        self.stop_event = asyncio.Event()
        self.error_event = asyncio.Event()
        
        # Output queue for processed frames
        self.output_queue = queue.Queue()
        
    def process_frame(self, frame) -> Optional[OutputFrame]:
        """Process a single frame and return the output."""
        if not frame:
            return None
        return VideoOutput(frame, self.request_id)
    
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
            
            # Check if any loop had an exception
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    loop_names = ["ingress", "egress", "control"]
                    logger.error(f"{loop_names[i]} loop failed: {result}")
                    
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
        """Process incoming frames."""
        try:
            frame_count = 0
            async for frame in self.protocol.ingress_loop(self.stop_event):
                # Check for error state or stop signal
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
                else:
                    logger.debug(f"Received non-video frame: {type(frame)}")
            
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
                """Generate output frames from the queue."""
                while not self.stop_event.is_set() and not self.error_event.is_set():
                    try:
                        # Try to get a frame from the queue with timeout
                        frame = await asyncio.to_thread(self.output_queue.get, timeout=0.1)
                        if frame is not None:
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


class VideoOutput:
    """Video output frame wrapper."""
    
    def __init__(self, frame, request_id: str):
        self.frame = frame
        self.request_id = request_id 