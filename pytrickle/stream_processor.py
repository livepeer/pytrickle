import asyncio
import logging
from typing import Optional, Callable, Dict, Any, List, Union, Awaitable

from pytrickle.state import PipelineState

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
        model_loader: Optional[Union[Callable[[], None], Callable[[], Awaitable[None]]]] = None,
        param_updater: Optional[Union[Callable[[Dict[str, Any]], None], Callable[[Dict[str, Any]], Awaitable[None]]]] = None,
        send_data_interval: Optional[float] = 0.333,
        name: str = "stream-processor",
        port: int = 8000,
        **server_kwargs
    ):
        """
        Initialize StreamProcessor with processing functions.
        
        Args:
            video_processor: Async function that processes VideoFrame objects
            audio_processor: Async function that processes AudioFrame objects  
            model_loader: Optional function (sync or async) called during load_model phase
            param_updater: Optional function (sync or async) called when parameters update
            send_data_interval: Interval for sending data
            name: Processor name
            port: Server port
            **server_kwargs: Additional arguments passed to StreamServer
        """
        self.video_processor = video_processor
        self.audio_processor = audio_processor
        self.model_loader = model_loader
        self.param_updater = param_updater
        self.send_data_interval = send_data_interval
        self.name = name
        self.port = port
        self.server_kwargs = server_kwargs
        
        # Store pipeline reference if available
        self._pipeline = None
        
        # Create internal frame processor
        self._frame_processor = _InternalFrameProcessor(
            video_processor=video_processor,
            audio_processor=audio_processor,
            model_loader=model_loader,
            param_updater=param_updater,
            name=name
        )
        
        # Store pipeline reference for passing to model_loader
        self._frame_processor._pipeline = self._pipeline
        
        # Create startup handler to ensure load_model is called
        async def startup_handler(app):
            """Ensure load_model is called during server startup."""
            if not self._frame_processor._load_model_scheduled:
                logger.info("🔧 Calling load_model during server startup...")
                await self._frame_processor.load_model()
                self._frame_processor._load_model_scheduled = True
                logger.info("✅ load_model completed during server startup")
        
        # Add our startup handler to any existing ones
        existing_startup = server_kwargs.get('on_startup', [])
        if not isinstance(existing_startup, list):
            existing_startup = [existing_startup] if existing_startup else []
        startup_handlers = [startup_handler] + existing_startup
        server_kwargs['on_startup'] = startup_handlers
        
        # Create and start server
        self.server = StreamServer(
            frame_processor=self._frame_processor,
            port=port,
            **server_kwargs
        )
    
    async def send_data(self, data: str):
        """Send data to the server."""
        if self.server.current_client is None:
            logger.warning("No active client connection, cannot send data")
            return False
        await self.server.current_client.publish_data(data)
        return True

    async def run_forever(self):
        """Run the stream processor server forever."""
        await self.server.run_forever()
    
    def set_pipeline(self, pipeline):
        """Set the pipeline reference for passing to model_loader."""
        self._pipeline = pipeline
        self._frame_processor._pipeline = pipeline
    
    def run(self):
        """Run the stream processor server (blocking)."""
        asyncio.run(self.run_forever())

class _InternalFrameProcessor(FrameProcessor):
    """Internal frame processor that wraps user-provided functions."""
    
    def __init__(
        self,
        video_processor: Optional[VideoProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        model_loader: Optional[Union[Callable[[], None], Callable[[], Awaitable[None]]]] = None,
        param_updater: Optional[Union[Callable[[Dict[str, Any]], None], Callable[[Dict[str, Any]], Awaitable[None]]]] = None,
        name: str = "internal-processor"
    ):
        # Set attributes first before calling parent
        self.video_processor = video_processor
        self.audio_processor = audio_processor
        self.model_loader = model_loader
        self.param_updater = param_updater
        self._ready = False
        self.name = name
        
        # Initialize parent with error_callback=None
        super().__init__(error_callback=None)
        
        # Schedule load_model to be called during server startup
        # This ensures warmup happens during initialization, not on first request
        self._load_model_scheduled = False
    
    async def load_model(self, **kwargs):
        """Load model using provided function (fully async)."""
        # Prevent duplicate loading
        if self._load_model_scheduled and self._ready:
            logger.info("Model already loaded, skipping duplicate load_model call")
            return
            
        if self.model_loader:
            try:
                # Check if model_loader is async
                if inspect.iscoroutinefunction(self.model_loader):
                    # Run async model loader
                    await self.model_loader(**kwargs)
                else:
                    # Run synchronous model loader
                    self.model_loader(**kwargs)
                logger.info(f"StreamProcessor '{self.name}' model loaded successfully")
            except Exception as e:
                logger.error(f"Error in model loader: {e}")
                raise
        else:
            # No external model_loader, but we need to ensure that any FrameProcessor
            # instances that were created get their load_model called
            logger.info("No external model loader, checking for FrameProcessor initialization")
        
        self._ready = True
        self._load_model_scheduled = True
    
    async def process_video_async(self, frame: VideoFrame) -> Optional[VideoFrame]:
        """Process video frame using provided function."""
        if not self._ready or not self.video_processor:
            return frame
        
        try:
            result = await self.video_processor(frame)
            return result if isinstance(result, VideoFrame) else frame
        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            return frame
    
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
    
    async def update_params(self, params: Dict[str, Any]):
        """Update parameters using provided function (supports both sync and async)."""
        if self.param_updater:
            try:
                # Check if param_updater is async
                if inspect.iscoroutinefunction(self.param_updater):
                    # Run async param updater
                    await self.param_updater(params)
                else:
                    # Run synchronous param updater
                    self.param_updater(params)
                logger.info(f"Parameters updated: {params}")
            except Exception as e:
                logger.error(f"Error updating parameters: {e}")