"""
Frame Processor - Async processing utilities for PyTrickle.
This module provides base classes and utilities for async frame processing,
making it easy to integrate AI models and async pipelines with PyTrickle.

Enhancements:
- Optional internal queue/buffer mode for decoupled ingress/egress
- Built-in worker tasks that consume input queues and produce output queues
  while calling the subclass processing methods
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List
from .frames import VideoFrame, AudioFrame
from . import ErrorCallback

logger = logging.getLogger(__name__)


class FrameProcessor(ABC):
    """
    Base class for async frame processors.

    This class provides native async frame processing for PyTrickle. It handles:
    - initialization and warmup
    - async processing video and audio frames

    Lifecycle:
    1. Processing begins automatically when streams start
    2. Processing stops automatically when streams end

    Usage patterns:

    # HTTP server with StreamServer (recommended)
    processor = MyProcessor()
    app = StreamServer(frame_processor=processor, port=8000)
    await app.run_forever()

    # Direct client usage (advanced)
    protocol = TrickleProtocol(subscribe_url="...", publish_url="...")
    client = TrickleClient(protocol=protocol, frame_processor=processor)
    await client.start("request_id")

    Subclass this to implement your async AI processing logic.
    """

    def __init__(
        self,
        error_callback: Optional[ErrorCallback] = None,
        # Queue/Buffer mode (default enabled)
        queue_mode: bool = True,
        video_queue_size: int = 8,
        audio_queue_size: int = 32,
        video_concurrency: int = 1,
        audio_concurrency: int = 0,  # Default to audio passthrough mode=0
        **init_kwargs
    ):
        """Initialize the frame processor.
        
        Args:
            error_callback: Optional error callback for processing errors
            queue_mode: Enable internal queued processing with worker tasks
            video_queue_size: Max buffered video frames
            audio_queue_size: Max buffered audio frames
            video_concurrency: Number of concurrent video workers
            audio_concurrency: Number of concurrent audio workers
            **init_kwargs: Additional kwargs passed to initialize() method
        """
        self.error_callback = error_callback
        # Queue mode configuration
        self.queue_mode = bool(queue_mode)
        self._video_in_q: Optional[asyncio.Queue] = None
        self._audio_in_q: Optional[asyncio.Queue] = None
        self._video_out_q: Optional[asyncio.Queue] = None
        self._audio_out_q: Optional[asyncio.Queue] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._video_workers: List[asyncio.Task] = []
        self._audio_workers: List[asyncio.Task] = []
        self._video_queue_size = int(video_queue_size)
        self._audio_queue_size = int(audio_queue_size)
        self._video_concurrency = max(1, int(video_concurrency))
        self._audio_concurrency = max(1, int(audio_concurrency))

        self.load_model(**init_kwargs)

    # ========= Optional queued IO API =========
    async def start_queue_workers(self) -> None:
        """Start internal processing workers when queue_mode is enabled.

        In queue mode, TrickleClient can enqueue frames via enqueue_* and
        read processed frames via get_next_processed_* without blocking ingress.
        """
        if not self.queue_mode:
            return
        # Clear any stale items before starting to avoid cross-stream jitter
        try:
            for q in [self._video_in_q, self._audio_in_q, self._video_out_q, self._audio_out_q]:
                if isinstance(q, asyncio.Queue):
                    while not q.empty():
                        q.get_nowait()
                        q.task_done()
        except Exception:
            pass

        # Always recreate a fresh stop event on (re)start so workers run after previous stop
        self._stop_event = asyncio.Event()
        # Initialize queues lazily
        if self._video_in_q is None:
            self._video_in_q = asyncio.Queue(maxsize=self._video_queue_size)
        if self._audio_in_q is None:
            self._audio_in_q = asyncio.Queue(maxsize=self._audio_queue_size)
        if self._video_out_q is None:
            self._video_out_q = asyncio.Queue(maxsize=self._video_queue_size)
        if self._audio_out_q is None:
            self._audio_out_q = asyncio.Queue(maxsize=self._audio_queue_size)
        # Spawn workers
        for _ in range(self._video_concurrency):
            self._video_workers.append(asyncio.create_task(self._video_worker()))
        for _ in range(self._audio_concurrency):
            self._audio_workers.append(asyncio.create_task(self._audio_worker()))

    async def stop_queue_workers(self) -> None:
        """Stop internal processing workers and drain queues."""
        if not self.queue_mode:
            return
        if self._stop_event:
            self._stop_event.set()
        # Cancel workers
        for task in self._video_workers + self._audio_workers:
            task.cancel()
        for task in self._video_workers + self._audio_workers:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"Worker ended with error: {e}")
        self._video_workers.clear()
        self._audio_workers.clear()

        # Best-effort drain
        for q in [self._video_in_q, self._audio_in_q, self._video_out_q, self._audio_out_q]:
            if isinstance(q, asyncio.Queue):
                try:
                    while not q.empty():
                        q.get_nowait()
                        q.task_done()
                except Exception:
                    pass

    async def reset_state(self) -> None:
        """Reset internal queued state (drain queues, clear stop flag).

        Safe to call between streams to ensure clean start.
        """
        if not self.queue_mode:
            return
        # Do not cancel workers here; just clear queues and reset stop flag
        try:
            for q in [self._video_in_q, self._audio_in_q, self._video_out_q, self._audio_out_q]:
                if isinstance(q, asyncio.Queue):
                    while not q.empty():
                        try:
                            q.get_nowait()
                            q.task_done()
                        except asyncio.QueueEmpty:
                            break
                        except Exception:
                            break
        except Exception:
            pass
        # Reset stop event so workers run on next start
        if self._stop_event is None or self._stop_event.is_set():
            self._stop_event = asyncio.Event()

    # ===== Optional lifecycle hooks; subclasses may override =====
    async def reset_timing(self) -> None:
        """Optional hook to reset timing state (e.g., frame counters) between streams."""
        return

    async def enqueue_video_frame(self, frame: VideoFrame) -> None:
        if not self.queue_mode or self._video_in_q is None:
            raise RuntimeError("Queue mode is disabled on FrameProcessor")
        try:
            await self._video_in_q.put(frame)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Failed to enqueue video frame: {e}")

    async def enqueue_audio_frame(self, frame: AudioFrame) -> None:
        if not self.queue_mode or self._audio_in_q is None:
            raise RuntimeError("Queue mode is disabled on FrameProcessor")
        try:
            await self._audio_in_q.put(frame)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Failed to enqueue audio frame: {e}")

    async def get_next_processed_video(self) -> Optional[VideoFrame]:
        if not self.queue_mode or self._video_out_q is None:
            raise RuntimeError("Queue mode is disabled on FrameProcessor")
        try:
            return await self._video_out_q.get()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Failed to dequeue processed video: {e}")
            return None

    async def get_next_processed_audio(self) -> Optional[List[AudioFrame]]:
        if not self.queue_mode or self._audio_out_q is None:
            raise RuntimeError("Queue mode is disabled on FrameProcessor")
        try:
            return await self._audio_out_q.get()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Failed to dequeue processed audio: {e}")
            return None

    async def _video_worker(self) -> None:
        assert self._video_in_q is not None and self._video_out_q is not None
        while self._stop_event and not self._stop_event.is_set():
            try:
                frame = await self._video_in_q.get()
                processed = await self.process_video_async(frame)
                if processed is not None:
                    await self._video_out_q.put(processed)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Video worker error: {e}")
                if self.error_callback:
                    try:
                        if asyncio.iscoroutinefunction(self.error_callback):
                            await self.error_callback("video_worker_error", e)
                        else:
                            self.error_callback("video_worker_error", e)
                    except Exception:
                        pass

    async def _audio_worker(self) -> None:
        assert self._audio_in_q is not None and self._audio_out_q is not None
        while self._stop_event and not self._stop_event.is_set():
            try:
                frame = await self._audio_in_q.get()
                processed_list = await self.process_audio_async(frame)
                if processed_list:
                    await self._audio_out_q.put(processed_list)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audio worker error: {e}")
                if self.error_callback:
                    try:
                        if asyncio.iscoroutinefunction(self.error_callback):
                            await self.error_callback("audio_worker_error", e)
                        else:
                            self.error_callback("audio_worker_error", e)
                    except Exception:
                        pass

    @abstractmethod
    def load_model(self, *kwargs):
        """
        Load the model.

        This method should be implemented to load any required models or resources.
        It is called automatically during initialization.
        
        Args:
            *kwargs: Additional parameters for model loading
        """
        pass

    @abstractmethod
    async def process_video_async(self, frame: VideoFrame) -> Optional[VideoFrame]:
        """
        Process a video frame asynchronously.

        Args:
            frame: Input video frame

        Returns:
            Processed video frame or None if processing failed
        """
        pass

    @abstractmethod
    async def process_audio_async(self, frame: AudioFrame) -> Optional[List[AudioFrame]]:
        """
        Process an audio frame asynchronously.

        Args:
            frame: Input audio frame

        Returns:
            List of processed audio frames or None if processing failed
        """
        pass

    @abstractmethod
    def update_params(self, params: Dict[str, Any]):
        """
        Update processing parameters (optional override).

        Args:
            params: Dictionary of parameters to update
        """
        pass
