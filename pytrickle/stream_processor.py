import asyncio
import inspect
import logging
from typing import Optional, Callable, Dict, Any, List, Union, Awaitable

from .frames import VideoFrame, AudioFrame
from .frame_processor import FrameProcessor
from .server import StreamServer
from .frame_skipper import AdaptiveFrameSkipper

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
        name: str = "stream-processor",
        port: int = 8000,
        enable_frame_skipping: bool = True,
        target_fps: float = 30.0,
        frame_skip_config: Optional[dict] = None,
        **server_kwargs
    ):
        """
        Initialize StreamProcessor with async processing functions.
        
        Args:
            video_processor: Async function that processes VideoFrame objects
            audio_processor: Async function that processes AudioFrame objects  
            model_loader: Optional function called during load_model phase
            param_updater: Optional function called when parameters update
            name: Processor name
            port: Server port
            enable_frame_skipping: Whether to enable adaptive frame skipping
            target_fps: Target output FPS for frame skipping
            frame_skip_config: Additional configuration for frame skipper
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
        self.name = name
        self.port = port
        self.server_kwargs = server_kwargs
        
        # Frame skipping setup
        self.enable_frame_skipping = enable_frame_skipping
        self.frame_skip_config = frame_skip_config or {}
        self.target_fps = target_fps
        
        # Create internal frame processor
        self._frame_processor = _InternalFrameProcessor(
            video_processor=video_processor,
            audio_processor=audio_processor,
            model_loader=model_loader,
            param_updater=param_updater,
            name=name,
            enable_frame_skipping=self.enable_frame_skipping,
            target_fps=self.target_fps,
            frame_skip_config=self.frame_skip_config
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
    
    def get_frame_skip_statistics(self) -> Optional[dict]:
        """Get frame skipping performance statistics."""
        return self._frame_processor.get_frame_skip_statistics()

class _InternalFrameProcessor(FrameProcessor):
    """Internal frame processor that wraps user-provided functions."""
    
    def __init__(
        self,
        video_processor: Optional[VideoProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        model_loader: Optional[Callable[[], None]] = None,
        param_updater: Optional[Callable[[Dict[str, Any]], None]] = None,
        name: str = "internal-processor",
        enable_frame_skipping: bool = True,
        target_fps: float = 24.0,
        frame_skip_config: Optional[dict] = None
    ):
        # Set attributes first
        self.video_processor = video_processor
        self.audio_processor = audio_processor
        self.model_loader = model_loader
        self.param_updater = param_updater
        self._ready = False
        self.name = name
        
        # Frame skipping is handled at the TrickleClient level
        # Having multiple frame skippers causes interference and double-counting
        self.enable_frame_skipping = False  # Always False to prevent conflicts
        self.frame_skipper = None
        
        # Set error_callback like parent constructor but skip load_model call
        self.error_callback = None
        
        # Call load_model manually after attributes are set
        self.load_model()
    
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
        """Process audio frame using provided async function. Audio frames are never skipped."""
        if not self._ready or not self.audio_processor:
            return [frame]
        
        # Audio frames are never skipped - always process them    
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
    
    def get_frame_skip_statistics(self) -> Optional[dict]:
        """Get frame skipping performance statistics."""
        if self.frame_skipper:
            return self.frame_skipper.get_statistics()
        return None