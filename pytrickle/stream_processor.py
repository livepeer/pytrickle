import asyncio
import logging
from typing import Optional, Callable, Dict, Any, List, Union, Awaitable
from dataclasses import dataclass

from .frames import VideoFrame, AudioFrame
from .frame_processor import FrameProcessor
from .server import StreamServer

logger = logging.getLogger(__name__)

# Type aliases for processing functions
VideoProcessor = Callable[[VideoFrame], Awaitable[Optional[VideoFrame]]]
AudioProcessor = Callable[[AudioFrame], Awaitable[Optional[List[AudioFrame]]]]

# Default configuration constants
class ProcessorDefaults:
    """Default configuration values for processors."""
    
    # Video processing defaults
    VIDEO_QUEUE_SIZE = 8
    VIDEO_CONCURRENCY = 1
    
    # Audio processing defaults  
    AUDIO_QUEUE_SIZE = 32
    AUDIO_CONCURRENCY = 1
    
    # Passthrough/disabled mode
    PASSTHROUGH_CONCURRENCY = 0
    
    # Server defaults
    DEFAULT_PORT = 8000
    DEFAULT_NAME = "stream-processor"
    INTERNAL_PROCESSOR_NAME = "internal-processor"
    
    # Queue mode default
    QUEUE_MODE_ENABLED = True

@dataclass
class ProcessorConfig:
    """Base configuration for a processor."""
    queue_size: int
    concurrency: int
    
    @classmethod
    def video_defaults(cls) -> 'ProcessorConfig':
        """Create default video processor configuration."""
        return cls(
            queue_size=ProcessorDefaults.VIDEO_QUEUE_SIZE,
            concurrency=ProcessorDefaults.VIDEO_CONCURRENCY
        )
    
    @classmethod 
    def audio_defaults(cls) -> 'ProcessorConfig':
        """Create default audio processor configuration."""
        return cls(
            queue_size=ProcessorDefaults.AUDIO_QUEUE_SIZE,
            concurrency=ProcessorDefaults.AUDIO_CONCURRENCY
        )

@dataclass
class VideoProcessorConfig(ProcessorConfig):
    """Configuration for video processing."""
    processor: VideoProcessor
    
    def __init__(
        self, 
        processor: VideoProcessor,
        queue_size: int = ProcessorDefaults.VIDEO_QUEUE_SIZE,
        concurrency: int = ProcessorDefaults.VIDEO_CONCURRENCY
    ):
        self.processor = processor
        super().__init__(queue_size=queue_size, concurrency=concurrency)

@dataclass
class AudioProcessorConfig(ProcessorConfig):
    """Configuration for audio processing."""
    processor: AudioProcessor
    
    def __init__(
        self,
        processor: AudioProcessor, 
        queue_size: int = ProcessorDefaults.AUDIO_QUEUE_SIZE,
        concurrency: int = ProcessorDefaults.AUDIO_CONCURRENCY
    ):
        self.processor = processor
        super().__init__(queue_size=queue_size, concurrency=concurrency)

class AudioPassthrough:
    """Marker class to indicate audio passthrough mode (concurrency=0)."""
    pass

class ProcessorConfigResolver:
    """Utility class to resolve processor configurations."""
    
    @staticmethod
    def get_video_config(
        video_processor: Optional[Union[VideoProcessor, VideoProcessorConfig]]
    ) -> tuple[Optional[VideoProcessor], int, int]:
        """
        Resolve video configuration to (processor, queue_size, concurrency).
        
        Returns:
            tuple: (processor_func, queue_size, concurrency)
        """
        if video_processor is None:
            return None, ProcessorDefaults.VIDEO_QUEUE_SIZE, ProcessorDefaults.PASSTHROUGH_CONCURRENCY
        elif isinstance(video_processor, VideoProcessorConfig):
            return video_processor.processor, video_processor.queue_size, video_processor.concurrency
        elif callable(video_processor):
            # Backward compatibility: treat as processor function
            return video_processor, ProcessorDefaults.VIDEO_QUEUE_SIZE, ProcessorDefaults.VIDEO_CONCURRENCY
        else:
            raise ValueError(f"Invalid video configuration: {type(video_processor)}")
    
    @staticmethod
    def get_audio_config(
        audio_processor: Optional[Union[AudioProcessor, AudioProcessorConfig, AudioPassthrough]]
    ) -> tuple[Optional[AudioProcessor], int, int]:
        """
        Resolve audio configuration to (processor, queue_size, concurrency).
        
        Returns:
            tuple: (processor_func, queue_size, concurrency)
        """
        if audio_processor is None or isinstance(audio_processor, AudioPassthrough):
            return None, ProcessorDefaults.AUDIO_QUEUE_SIZE, ProcessorDefaults.PASSTHROUGH_CONCURRENCY
        elif isinstance(audio_processor, AudioProcessorConfig):
            return audio_processor.processor, audio_processor.queue_size, audio_processor.concurrency
        elif callable(audio_processor):
            # Backward compatibility: treat as processor function
            return audio_processor, ProcessorDefaults.AUDIO_QUEUE_SIZE, ProcessorDefaults.AUDIO_CONCURRENCY
        else:
            raise ValueError(f"Invalid audio configuration: {type(audio_processor)}")
    
    @staticmethod
    def validate_configuration(video_concurrency: int, audio_concurrency: int) -> None:
        """Validate that at least one processor is enabled."""
        if video_concurrency == ProcessorDefaults.PASSTHROUGH_CONCURRENCY and \
           audio_concurrency == ProcessorDefaults.PASSTHROUGH_CONCURRENCY:
            raise ValueError("At least one of video or audio processing must be enabled")

class StreamProcessor:
    def __init__(
        self,
        video_processor: Optional[Union[VideoProcessor, VideoProcessorConfig]] = None,
        audio_processor: Optional[Union[AudioProcessor, AudioProcessorConfig, AudioPassthrough]] = None,
        model_loader: Optional[Callable[[], None]] = None,
        param_updater: Optional[Callable[[Dict[str, Any]], None]] = None,
        queue_mode: bool = ProcessorDefaults.QUEUE_MODE_ENABLED,
        name: str = ProcessorDefaults.DEFAULT_NAME,
        port: int = ProcessorDefaults.DEFAULT_PORT,
        **server_kwargs
    ):
        """
        Initialize StreamProcessor with processing functions.
        
        Args:
            video_processor: Video processor function or VideoProcessorConfig
            audio_processor: Audio processor function, AudioProcessorConfig, or AudioPassthrough()
            model_loader: Optional function called during load_model phase
            param_updater: Optional function called when parameters update
            queue_mode: Enable internal queued processing with worker tasks
            name: Processor name
            port: Server port
            **server_kwargs: Additional arguments passed to StreamServer
        
        Examples:
            # Simple functional style (backward compatible)
            StreamProcessor(video_processor=process_video_func, audio_processor=AudioPassthrough())
            
            # With custom configuration
            StreamProcessor(
                video_processor=VideoProcessorConfig(process_video_func, queue_size=16, concurrency=1),
                audio_processor=AudioProcessorConfig(process_audio_func, queue_size=64, concurrency=0)
            )
        """
        self.model_loader = model_loader
        self.param_updater = param_updater
        self.name = name
        self.port = port
        self.server_kwargs = server_kwargs
        
        # Resolve configurations using the consolidated resolver
        video_processor_func, video_queue_size, video_concurrency = \
            ProcessorConfigResolver.get_video_config(video_processor)
        
        audio_processor_func, audio_queue_size, audio_concurrency = \
            ProcessorConfigResolver.get_audio_config(audio_processor)
        
        # Validate configuration
        ProcessorConfigResolver.validate_configuration(video_concurrency, audio_concurrency)
        
        # Create internal frame processor with resolved parameters
        self._frame_processor = _InternalFrameProcessor(
            video_processor=video_processor_func,
            audio_processor=audio_processor_func,
            model_loader=model_loader,
            param_updater=param_updater,
            queue_mode=queue_mode,
            video_queue_size=video_queue_size,
            audio_queue_size=audio_queue_size,
            video_concurrency=video_concurrency,
            audio_concurrency=audio_concurrency,
            name=name,
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
        name: str = ProcessorDefaults.INTERNAL_PROCESSOR_NAME,
        queue_mode: bool = ProcessorDefaults.QUEUE_MODE_ENABLED,
        video_queue_size: int = ProcessorDefaults.VIDEO_QUEUE_SIZE,
        audio_queue_size: int = ProcessorDefaults.AUDIO_QUEUE_SIZE,
        video_concurrency: int = ProcessorDefaults.VIDEO_CONCURRENCY,
        audio_concurrency: int = ProcessorDefaults.PASSTHROUGH_CONCURRENCY,
        **kwargs
    ):
        # Set attributes first
        self.video_processor = video_processor
        self.audio_processor = audio_processor
        self.model_loader = model_loader
        self.param_updater = param_updater
        self._ready = False
        self.name = name
        
        # Log errors via a simple callback
        def log_error(error_type: str, exception: Optional[Exception] = None):
            logger.warning(f"Processing error: {error_type} - {exception}")
        
        # Call parent constructor with queue parameters
        super().__init__(
            error_callback=log_error,
            queue_mode=queue_mode,
            video_queue_size=video_queue_size,
            audio_queue_size=audio_queue_size,
            video_concurrency=video_concurrency,
            audio_concurrency=audio_concurrency,
            **kwargs
        )
    
    def initialize(self, **kwargs):
        """Initialize processor using provided model_loader function."""
        if self.model_loader:
            try:
                self.model_loader(**kwargs)
                logger.info(f"StreamProcessor '{self.name}' model loaded successfully")
            except Exception as e:
                logger.error(f"Error in model loader: {e}")
                raise
        self._ready = True
    
    def load_model(self, **kwargs):
        """Load model using provided function - required by FrameProcessor ABC."""
        self.initialize(**kwargs)
    
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
    
    def update_params(self, params: Dict[str, Any]):
        """Update parameters using provided function."""
        if self.param_updater:
            try:
                self.param_updater(params)
                logger.info(f"Parameters updated: {params}")
            except Exception as e:
                logger.error(f"Error updating parameters: {e}")