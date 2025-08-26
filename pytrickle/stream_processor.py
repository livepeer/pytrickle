import asyncio
import inspect
import logging
from typing import Optional, Callable, Dict, Any, List, Union, Awaitable

from .frames import VideoFrame, AudioFrame
from .frame_processor import FrameProcessor
from .server import StreamServer

logger = logging.getLogger(__name__)

# Type aliases for processing functions
VideoProcessor = Callable[[VideoFrame], Awaitable[Optional[VideoFrame]]]
AudioProcessor = Callable[[AudioFrame], Awaitable[Optional[List[AudioFrame]]]]

class StreamProcessor:
    def __init__(
        self,
        video_processor: Optional[VideoProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        model_loader: Optional[Callable[[], None]] = None,
        param_updater: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_stream_stop: Optional[Callable[[], None]] = None,
        send_data_interval: Optional[float] = 0.333,
        name: str = "stream-processor",
        port: int = 8000,
        enable_frame_skipping: bool = True,
        target_fps: Optional[float] = None,
        auto_target_fps: bool = True,
        **server_kwargs
    ):
        """
        Initialize StreamProcessor with async processing functions.
        
        Args:
            video_processor: Async function that processes VideoFrame objects
            audio_processor: Async function that processes AudioFrame objects  
            model_loader: Optional function called during load_model phase
            param_updater: Optional function called when parameters update
            send_data_interval: Interval for sending data
            on_stream_stop: Optional function called when stream stops/client disconnects
            name: Processor name
            port: Server port
            enable_frame_skipping: Whether to enable intelligent frame skipping
            target_fps: Target FPS for frame skipping (None = auto-detect)
            auto_target_fps: Whether to automatically detect target FPS
            **server_kwargs: Additional arguments passed to StreamServer
        """
        # Validate that processors are async functions
        if video_processor is not None and not inspect.iscoroutinefunction(video_processor):
            raise ValueError("video_processor must be an async function")
        if audio_processor is not None and not inspect.iscoroutinefunction(audio_processor):
            raise ValueError("audio_processor must be an async function")
            
        self.video_processor = video_processor
        self.audio_processor = audio_processor
        self.model_loader = model_loader
        self.param_updater = param_updater
        self.on_stream_stop = on_stream_stop
        self.send_data_interval = send_data_interval
        self.name = name
        self.port = port
        self.enable_frame_skipping = enable_frame_skipping
        self.target_fps = target_fps
        self.auto_target_fps = auto_target_fps
        self.server_kwargs = server_kwargs
        
        # Create internal frame processor
        self._frame_processor = _InternalFrameProcessor(
            video_processor=video_processor,
            audio_processor=audio_processor,
            model_loader=model_loader,
            param_updater=param_updater,
            on_stream_stop=on_stream_stop,
            name=name
        )
        
        # Create and start server
        self.server = StreamServer(
            frame_processor=self._frame_processor,
            port=port,
            enable_frame_skipping=enable_frame_skipping,
            target_fps=target_fps,
            auto_target_fps=auto_target_fps,
            **server_kwargs
        )
    
    async def send_data(self, data: str):
        """Send data to the server."""
        if self.server.current_client is None:
            logger.warning("No active client connection, cannot send data")
            return False
        
        # Check if client is in error state
        if self.server.current_client.error_event.is_set():
            logger.debug("Client is in error state, not sending data")
            return False
            
        try:
            await self.server.current_client.publish_data(data)
            return True
        except Exception as e:
            logger.error(f"Error sending data: {e}")
            return False

    async def run_forever(self):
        """Run the stream processor server forever."""
        await self.server.run_forever()
    
    def run(self):
        """Run the stream processor server (blocking)."""
        asyncio.run(self.run_forever())

class _InternalFrameProcessor(FrameProcessor):
    """Internal frame processor that wraps user-provided functions."""
    
    def __init__(
        self,
        video_processor: Optional[VideoProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        model_loader: Optional[Callable[[], None]] = None,
        param_updater: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_stream_stop: Optional[Callable[[], None]] = None,
        name: str = "internal-processor"
    ):
        # Set attributes first before calling parent
        self.video_processor = video_processor
        self.audio_processor = audio_processor
        self.model_loader = model_loader
        self.param_updater = param_updater
        self.on_stream_stop = on_stream_stop
        self._ready = False
        self.name = name
        
        # Frame skipping is handled at the TrickleClient level
        # Having multiple frame skippers causes interference and double-counting
        self.enable_frame_skipping = False  # Always False to prevent conflicts
        self.frame_skipper = None
        
        # Initialize parent with error_callback=None, which will call load_model
        super().__init__(error_callback=None)
    
    def load_model(self, **kwargs):
        """Load model using provided function."""
        if self.model_loader:
            try:
                self.model_loader(self, **kwargs)
                logger.info(f"StreamProcessor '{self.name}' model loaded successfully")
            except Exception as e:
                logger.error(f"Error in model loader: {e}")
                raise
        self._ready = True
    
    async def process_video_async(self, frame: VideoFrame) -> Optional[VideoFrame]:
        """Process video frame using provided async function."""
        if not self._ready or not self.video_processor:
            logger.info("Processor not ready or no video processor defined, passing frame unchanged")
            return frame
        
        
        # NOTE: Frame skipping is handled at the client level (TrickleClient)
        # Do NOT apply additional frame skipping here as it causes double-skipping
        # and FPS measurement interference
            
        try:
            result = await self.video_processor(frame)
            return result if isinstance(result, VideoFrame) else frame
        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            return frame
    
    async def process_audio_async(self, frame: AudioFrame) -> Optional[List[AudioFrame]]:
        """Process audio frame using provided async function."""
        if not self._ready or not self.audio_processor:
            return [frame]
            
        try:
            result = await self.audio_processor(frame)
            if isinstance(result, AudioFrame):
                return [result]
            elif isinstance(result, list):
                return result
            elif result is None:
                return [frame]
            return [frame]
        except Exception as e:
            logger.error(f"Error in audio processing: {e}")
            return [frame]
    
    def update_params(self, params: Dict[str, Any]):
        """Update parameters using provided function."""
        if self.param_updater:
            try:
                self.param_updater(params)
                logger.info(f"Parameters updated: {params}")
            except Exception as e:
                logger.error(f"Error updating parameters: {e}")