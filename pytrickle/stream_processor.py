import asyncio
import inspect
import logging
from typing import Optional, Callable, Dict, Any, List, Union, Awaitable, Coroutine

from .frames import VideoFrame, AudioFrame, VideoOutput, AudioOutput
from .frame_processor import FrameProcessor
from .server import StreamServer
from .frame_skipper import FrameSkipConfig
from .security import SecurityConfig

logger = logging.getLogger(__name__)

# Type aliases for processing functions
VideoProcessor = Callable[[VideoFrame], Awaitable[Optional[VideoFrame]]]
AudioProcessor = Callable[[AudioFrame], Awaitable[Optional[List[AudioFrame]]]]
ModelLoader = Callable[[Dict[str, Any]], Awaitable[None]]
ParamUpdater = Callable[[Dict[str, Any]], Awaitable[None]]
OnStreamStop = Callable[[], Awaitable[None]]

class StreamProcessor:
    def __init__(
        self,
        video_processor: Optional[VideoProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        model_loader: Optional[ModelLoader] = None,
        param_updater: Optional[ParamUpdater] = None,
        on_stream_stop: Optional[OnStreamStop] = None,
        send_data_interval: Optional[float] = 0.333,
        name: str = "stream-processor",
        port: int = 8000,
        frame_skip_config: Optional[FrameSkipConfig] = None,
        security_config: Optional[SecurityConfig] = None,
        enable_security: bool = True,
        **server_kwargs
    ):
        """
        Initialize StreamProcessor with async processing functions.
        
        Args:
            video_processor: Async function that processes VideoFrame objects
            audio_processor: Async function that processes AudioFrame objects  
            model_loader: Optional async function called during load_model phase
            param_updater: Optional async function called when parameters update
            send_data_interval: Interval for sending data
            name: Processor name
            port: Server port
            frame_skip_config: Optional frame skipping configuration (None = no frame skipping)
            security_config: Optional security configuration (uses defaults if None)
            enable_security: Whether to enable security middleware (default: True)
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
        self.frame_skip_config = frame_skip_config
        self.security_config = security_config
        self.enable_security = enable_security
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
            frame_skip_config=frame_skip_config,
            security_config=security_config,
            enable_security=enable_security,
            **server_kwargs
        )
    
    async def send_data(self, data: str):
        """Send data to the server."""
        if self.server.current_client is None:
            logger.warning("No active client connection, cannot send data")
            return False
        
        client = self.server.current_client
        
        # Check if client is in error state or stopping
        if client.error_event.is_set() or client.stop_event.is_set():
            logger.debug("Client is in error/stop state, not sending data")
            return False
            
        try:
            await client.publish_data(data)
            return True
        except Exception as e:
            logger.error(f"Error sending data: {e}")
            return False

    async def send_input_frame(self, frame: Union[VideoFrame, AudioFrame]):
        """Send a video or audio frame to the input processing pipeline."""
        if self.server.current_client is None:
            logger.debug("No active client connection, cannot send input frame")
            return False

        client = self.server.current_client
        # Check if client is in error state or stopping
        if client.error_event.is_set() or client.stop_event.is_set():
            logger.debug("Client is in error/stop state, not sending input frame")
            return False

        try:
            logger.debug("sending input frame to client")
            if isinstance(frame, VideoFrame):
                await client.video_input_queue.put(frame)
            elif isinstance(frame, AudioFrame):
                await client.audio_input_queue.put(frame)
            return True
        except Exception as e:
            logger.error(f"Error sending input frame: {e}")
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
        model_loader: Optional[ModelLoader] = None,
        param_updater: Optional[ParamUpdater] = None,
        on_stream_stop: Optional[OnStreamStop] = None,
        name: str = "internal-processor"
    ):
        # Set attributes first before calling parent
        self.video_processor = video_processor
        self.audio_processor = audio_processor
        self.model_loader = model_loader
        self.param_updater = param_updater
        self.on_stream_stop = on_stream_stop
        self.name = name
        
        # Frame skipping is handled at TrickleClient level
        self.frame_skip_config = None
        self.frame_skipper = None
        
        super().__init__(error_callback=None)
    
    async def load_model(self, **kwargs):
        """Load model using provided async function."""
        if self.model_loader:
            try:
                await self.model_loader(**kwargs)
                logger.info(f"StreamProcessor '{self.name}' model loaded successfully")
            except Exception as e:
                logger.error(f"Error in model loader: {e}")
                raise
    
    async def process_video_async(self, frame: VideoFrame) -> Optional[VideoFrame]:
        """Process video frame using provided async function."""
        if not self.video_processor:
            logger.debug("No video processor defined, passing frame unchanged")
            return frame
            
        try:
            result = await self.video_processor(frame)
            return result if isinstance(result, VideoFrame) else frame
        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            return frame
    
    async def process_audio_async(self, frame: AudioFrame) -> Optional[List[AudioFrame]]:
        """Process audio frame using provided async function."""
        if not self.audio_processor:
            logger.debug("No audio processor defined, passing frame unchanged")
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
    
    async def update_params(self, params: Dict[str, Any]):
        """Update parameters using provided async function."""
        if self.param_updater:
            try:
                await self.param_updater(params)
                logger.info(f"Parameters updated: {params}")
            except Exception as e:
                logger.error(f"Error updating parameters: {e}")
