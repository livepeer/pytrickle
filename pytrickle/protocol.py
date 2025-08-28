"""
Trickle Protocol implementation for video streaming.

Provides a high-level interface for trickle-based video streaming,
integrating subscription, publishing, and media processing.
"""

import asyncio
import json
import queue
import logging
from typing import Optional, AsyncGenerator

from .base import TrickleComponent, ComponentState, setup_asyncio_exception_handler
from .subscriber import TrickleSubscriber
from .publisher import TricklePublisher
from .media import run_subscribe, run_publish
from .frames import InputFrame, OutputFrame, AudioFrame, VideoFrame, AudioOutput, VideoOutput, DEFAULT_WIDTH, DEFAULT_HEIGHT
from .decoder import DEFAULT_MAX_FRAMERATE
from .encoder import default_output_metadata
from .cache import LastValueCache
from .fps_meter import FPSMeter
from . import ErrorCallback

logger = logging.getLogger(__name__)

class TrickleProtocol(TrickleComponent):
    """Trickle protocol coordinator that manages subscription, publishing, control, and events."""
    
    def __init__(
        self,
        subscribe_url: str,
        publish_url: str,
        control_url: str = "",
        events_url: str = "",
        data_url: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        max_framerate: Optional[int] = None,
        heartbeat_interval: float = 10.0,
        publisher_timeout: Optional[float] = None,
        subscriber_timeout: Optional[float] = None,
        error_callback: Optional[ErrorCallback] = None,
    ):
        super().__init__(error_callback, component_name="protocol")
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.control_url = control_url
        self.events_url = events_url
        self.data_url = data_url
        self.width = width
        self.height = height
        self.max_framerate = max_framerate
        self.heartbeat_interval = heartbeat_interval
        self.publisher_timeout = publisher_timeout
        self.subscriber_timeout = subscriber_timeout
        
        # Tasks and components
        self.subscribe_task: Optional[asyncio.Task] = None
        self.publish_task: Optional[asyncio.Task] = None
        self.control_subscriber: Optional[TrickleSubscriber] = None
        self.events_publisher: Optional[TricklePublisher] = None
        self.data_publisher: Optional[TricklePublisher] = None
        
        # Queues are initialized during start(), but attributes must always exist
        self.subscribe_queue = None
        self.publish_queue = None

        # Background tasks
        self.subscribe_task: Optional[asyncio.Task] = None
        self.publish_task: Optional[asyncio.Task] = None
        
        # Coordination events
        self.subscription_ended = asyncio.Event()
        self._monitor_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # FPS tracking
        self.fps_meter = FPSMeter()

    async def _run_task_with_error_handling(self, task_func, task_name: str, *args, **kwargs):
        """Generic wrapper for running tasks with error handling."""
        try:
            await task_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{task_name} task failed: {e}")
            # Trigger error handling which will propagate to client
            await self._notify_error(f"{task_name.lower()}_task_error", e)
            raise

    async def _monitor_subscription_end(self):
        """Monitor for subscription task completion and trigger immediate shutdown."""
        try:
            if self.subscribe_task:
                # Wait for subscription task to complete
                await self.subscribe_task
                logger.info("Subscription task completed, triggering immediate shutdown of events publisher")
                
                # Set shutdown and subscription ended events first
                self.shutdown_event.set()
                self.subscription_ended.set()
                
                # Notify client of subscription end via error callback
                if self.error_callback:
                    try:
                        await self.error_callback("subscription_ended", None)
                    except Exception as e:
                        logger.error(f"Error in subscription end callback: {e}")
                
                # Immediately signal shutdown to events publisher to stop background tasks
                if self.events_publisher:
                    await self.events_publisher.shutdown()
                    logger.info("Events publisher shutdown signaled due to subscription end")
                # Stop heartbeat task on subscription end
                if self._heartbeat_task and not self._heartbeat_task.done():
                    self._heartbeat_task.cancel()
                    try:
                        await self._heartbeat_task
                    except asyncio.CancelledError:
                        pass
                
                # Also signal shutdown to control subscriber
                if self.control_subscriber:
                    await self.control_subscriber.shutdown()
                    logger.info("Control subscriber shutdown signaled due to subscription end")
                
        except Exception as e:
            logger.error(f"Error in subscription monitor: {e}")

    async def start(self):
        """Start the trickle protocol."""
        self._update_state(ComponentState.STARTING)
        logger.info(f"Starting trickle protocol: subscribe={self.subscribe_url}, publish={self.publish_url}")
        
        # Setup global asyncio exception handler to suppress aiohttp connection reset errors
        setup_asyncio_exception_handler()
        
        # Initialize queues
        self.subscribe_queue = queue.Queue()
        self.publish_queue = queue.Queue()
        
        # Metadata cache to pass video metadata from decoder to encoder
        metadata_cache = LastValueCache()
        
        # Start subscribe and publish tasks with error monitoring
        if self.subscribe_url and self.subscribe_url.strip():
            self.subscribe_task = asyncio.create_task(
                self._run_task_with_error_handling(
                    run_subscribe,
                    "Subscribe",
                    self.subscribe_url, 
                    self.subscribe_queue.put, 
                    metadata_cache.put, 
                    self.emit_monitoring_event, 
                    self.width or DEFAULT_WIDTH, 
                    self.height or DEFAULT_HEIGHT,
                    self.max_framerate or DEFAULT_MAX_FRAMERATE,
                    self.subscriber_timeout,
                )
            )
        else:
            #add default metadata to allow encoder to startup
            logger.info("setting default metadata for encoder")
            metadata_cache.put(default_output_metadata(self.width or DEFAULT_WIDTH, self.height or DEFAULT_HEIGHT))

        if self.publish_url and self.publish_url.strip():
            self.publish_task = asyncio.create_task(
                self._run_task_with_error_handling(
                    run_publish,
                    "Publish",
                    self.publish_url, 
                    self.publish_queue.get, 
                    metadata_cache.get, 
                    self.emit_monitoring_event,
                    self.publisher_timeout,
                )
            )
        
        # Initialize control subscriber if URL provided
        if self.control_url and self.control_url.strip():
            self.control_subscriber = TrickleSubscriber(
                self.control_url,
                error_callback=self._notify_error,
                connect_timeout_seconds=self.subscriber_timeout,
            )
            
        # Initialize events publisher if URL provided
        if self.events_url and self.events_url.strip():
            self.events_publisher = TricklePublisher(self.events_url, "application/json", error_callback=self._notify_error)
            if self.publisher_timeout is not None:
                self.events_publisher.connect_timeout_seconds = self.publisher_timeout
            await self.events_publisher.start()
            # Start protocol-level heartbeat loop if enabled
            if self.heartbeat_interval and self.heartbeat_interval > 0:
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
        # Initialize data publisher if URL provided
        if self.data_url and self.data_url.strip():
            self.data_publisher = TricklePublisher(self.data_url, "application/jsonl", error_callback=self._notify_error)
            if self.publisher_timeout is not None:
                self.data_publisher.connect_timeout_seconds = self.publisher_timeout
            await self.data_publisher.start()
        
        # Start monitoring subscription end for immediate cleanup
        self._monitor_task = asyncio.create_task(self._monitor_subscription_end())
        
        self._update_state(ComponentState.RUNNING)

    async def stop(self):
        """Stop the trickle protocol."""
        self._update_state(ComponentState.STOPPING)
        logger.info("Stopping trickle protocol")
        
        # Signal shutdown immediately to all components
        self.shutdown_event.set()
        
        # Notify client of protocol shutdown via error callback
        if self.error_callback:
            try:
                await self.error_callback("protocol_shutdown", None)
            except Exception as e:
                logger.error(f"Error in shutdown callback: {e}")
        
        # Stop heartbeat task first
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        self._heartbeat_task = None

        # Stop monitoring task
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Immediately signal shutdown to events publisher to stop background tasks
        if self.events_publisher:
            await self.events_publisher.shutdown()
        
        # Immediately signal shutdown to control subscriber
        if self.control_subscriber:
            await self.control_subscriber.shutdown()
        
        # Send sentinel None values to stop the trickle tasks gracefully if queues exist
        try:
            if self.subscribe_queue is not None:
                self.subscribe_queue.put(None)
        except Exception as e:
            logger.error(f"Error in subscribe queue shutdown: {e}")
            pass
        try:
            if self.publish_queue is not None:
                self.publish_queue.put(None)
        except Exception as e:
            logger.error(f"Error in publish queue shutdown: {e}")
            pass

        # Close control and events components
        if self.control_subscriber:
            await self.control_subscriber.close()
            self.control_subscriber = None

        if self.events_publisher:
            await self.events_publisher.close()
            self.events_publisher = None

        if self.data_publisher:
            await self.data_publisher.close()
            self.data_publisher = None

        # Wait for tasks to complete with timeout
        tasks = []
        if self.subscribe_task:
            tasks.append(self.subscribe_task)
        if self.publish_task:
            tasks.append(self.publish_task)
            
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
                logger.info("All protocol tasks completed gracefully")
            except asyncio.TimeoutError:
                logger.warning("Tasks did not complete within timeout, canceling...")
                for task in tasks:
                    if not task.done():
                        task.cancel()
                
                # Wait a bit more for cancellation to take effect
                try:
                    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2.0)
                    logger.info("Tasks cancelled successfully")
                except asyncio.TimeoutError:
                    logger.error("Some tasks failed to cancel - forcing cleanup")
                except Exception as e:
                    logger.warning(f"Expected errors during task cancellation: {e}")

        self.subscribe_task = None
        self.publish_task = None
        self._update_state(ComponentState.STOPPED)



    async def ingress_loop(self, done: asyncio.Event) -> AsyncGenerator[InputFrame, None]:
        """Generate frames from the ingress stream."""
        def dequeue_frame():
            try:
                frame = self.subscribe_queue.get(timeout=0.1)
                return frame if frame else None
            except queue.Empty:
                return None

        while not done.is_set() and not self.shutdown_event.is_set():
            frame = await asyncio.to_thread(dequeue_frame)
            if frame is None:
                continue
            if frame is None:  # Sentinel value
                break
            
            # Record ingress FPS for the frame
            if isinstance(frame, VideoFrame):
                self.fps_meter.record_ingress_video_frame()
            elif isinstance(frame, AudioFrame):
                self.fps_meter.record_ingress_audio_frame()
            
            # Send all frames to frame processor
            yield frame

    async def egress_loop(self, output_frames: AsyncGenerator[OutputFrame, None]):
        """Consume output frames and send them to the publish queue."""
        def enqueue_frame(frame: OutputFrame):
            self.publish_queue.put(frame)

        try:
            async for frame in output_frames:
                # Check if subscription has ended or shutdown is signaled
                if self.subscription_ended.is_set() or self.shutdown_event.is_set():
                    logger.info("Subscription ended or shutdown signaled, ending egress loop")
                    break
                
                # Record egress FPS for the frame
                if isinstance(frame, VideoOutput):
                    self.fps_meter.record_egress_video_frame()
                elif isinstance(frame, AudioOutput):
                    self.fps_meter.record_egress_audio_frame()
                    
                await asyncio.to_thread(enqueue_frame, frame)
        except Exception as e:
            logger.error(f"Error in egress loop: {e}")
            # Re-raise to trigger error handling in the client
            raise

    async def emit_monitoring_event(self, event: dict, queue_event_type: str = "ai_stream_events"):
        """Emit monitoring events via the events publisher."""
        if not self.events_publisher or self.shutdown_event.is_set():
            return
        try:
            event_json = json.dumps({"event": event, "queue_event_type": queue_event_type})
            async with await self.events_publisher.next() as segment:
                await segment.write(event_json.encode())
        except Exception as e:
            logger.error(f"Error reporting status: {e}")

    async def _heartbeat_loop(self):
        """Emit minimal liveness events at a fixed cadence."""
        try:
            while not self.shutdown_event.is_set():
                try:
                    heartbeat_event = {
                        "type": "heartbeat",
                        "timestamp": __import__("time").time(),
                    }
                    # Use milliseconds for timestamp consistency
                    heartbeat_event["timestamp"] = int(heartbeat_event["timestamp"] * 1000)
                    heartbeat_event.update({
                        "urls": {
                            "subscribe": self.subscribe_url,
                            "publish": self.publish_url,
                            "control": self.control_url or "",
                            "events": self.events_url or "",
                            "data": self.data_url or "",
                        },
                        "dimensions": {
                            "width": self.width or DEFAULT_WIDTH,
                            "height": self.height or DEFAULT_HEIGHT,
                        },
                        "fps": self.fps_meter.get_fps_stats(),
                    })
                    await self.emit_monitoring_event(heartbeat_event, queue_event_type="stream_heartbeat")
                except Exception as e:
                    logger.warning(f"Heartbeat emission error: {e}")
                # Wait for interval or shutdown
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=self.heartbeat_interval)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass

    async def control_loop(self, done: asyncio.Event) -> AsyncGenerator[dict, None]:
        """Generate control messages from the control stream."""
        if not self.control_subscriber:
            logger.warning("No control-url provided, inference won't get updates from the control trickle subscription")
            return

        logger.info("Starting Control subscriber at %s", self.control_url)
        keepalive_message = {"keep": "alive"}

        while not done.is_set() and not self.shutdown_event.is_set():
            try:
                segment = await self.control_subscriber.next()
                if not segment or segment.eos():
                    return

                params = await segment.read()
                if not params:
                    continue
                data = json.loads(params)
                if data == keepalive_message:
                    # Ignore periodic keepalive messages
                    continue

                logger.info("Received control message with params: %s", data)
                yield data

            except Exception:
                logger.error(f"Error in control loop", exc_info=True)
                continue

    async def publish_data(self, data: str):
        """Publish data via the data publisher."""
        if not self.data_publisher:
            return
        try:
            async with await self.data_publisher.next() as segment:
                await segment.write(data.encode('utf-8'))
        except Exception as e:
            logger.error(f"Error publishing data: {e}")