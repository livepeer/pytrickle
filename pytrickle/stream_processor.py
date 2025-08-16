import asyncio
import logging
from typing import Optional, Callable, Dict, Any, List, Awaitable


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
        enable_frame_caching: bool = True,
        name: str = "stream-processor",
        port: int = 8000,
        **server_kwargs
    ):
        """
        Initialize StreamProcessor with processing functions.
        
        Args:
            video_processor: Function that processes VideoFrame objects
            audio_processor: Function that processes AudioFrame objects
            model_loader: Optional function called during load_model phase
            param_updater: Optional function called when parameters update
            enable_frame_caching: Enable frame caching for better fallback behavior
            name: Processor name
            port: Server port
            **server_kwargs: Additional arguments passed to StreamServer
        """
        
        # Validate that at least one processor is provided
        if video_processor is None and audio_processor is None:
            raise ValueError("At least one of video or audio processor must be provided")
        self.video_processor = video_processor
        self.audio_processor = audio_processor
        self.model_loader = model_loader
        self.param_updater = param_updater
        self.name = name
        self.port = port
        self.server_kwargs = server_kwargs
        
        # Create internal frame processor with resolved parameters
        self._frame_processor = _InternalFrameProcessor(
            video_processor=video_processor,
            audio_processor=audio_processor,
            model_loader=model_loader,
            param_updater=param_updater,
            enable_frame_caching=enable_frame_caching,
            name=name
        )
        
        # Create and start server
        self.server = StreamServer(
            frame_processor=self._frame_processor,
            port=port,
            **server_kwargs
        )
    
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
        name: str = "internal-processor",
        enable_frame_caching: bool = True,
        **kwargs
    ):
        # Set attributes first before calling parent
        self.video_processor = video_processor
        self.audio_processor = audio_processor
        self.model_loader = model_loader
        self.param_updater = param_updater
        self._ready = False
        self.name = name
        
        # Initialize parent with error_callback=None, which will call load_model
        super().__init__(
            error_callback=None, 
            enable_frame_caching=enable_frame_caching,
            **kwargs
        )
    
    def load_model(self, **kwargs):
        """Load model using provided function."""
        if self.model_loader:
            try:
                self.model_loader(**kwargs)
                logger.info(f"StreamProcessor '{self.name}' model loaded successfully")
            except Exception as e:
                logger.error(f"Error in model loader: {e}")
                raise
        self._ready = True
    
    async def process_video_async(self, frame: VideoFrame) -> Optional[VideoFrame]:
        """Process video frame using provided function."""
        if not self._ready or not self.video_processor:
            return frame
        
        try:
            result = await self.video_processor(frame)
            return result if isinstance(result, VideoFrame) else None
        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            return None
    
    async def process_audio_async(self, frame: AudioFrame) -> Optional[List[AudioFrame]]:
        """Process audio frame using provided function."""
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