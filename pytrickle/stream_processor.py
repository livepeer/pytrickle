import asyncio
import inspect
import logging
from typing import Optional, Callable, Dict, Any, List, Union, Awaitable

from .registry import HandlerRegistry
from .frames import VideoFrame, AudioFrame
from .frame_processor import FrameProcessor
from .server import StreamServer
from .frame_skipper import FrameSkipConfig
from .warmup_config import WarmupConfig

logger = logging.getLogger(__name__)

# Type aliases for processing functions
VideoProcessor = Callable[[VideoFrame], Awaitable[Optional[VideoFrame]]]
AudioProcessor = Callable[[AudioFrame], Awaitable[Optional[List[AudioFrame]]]]

# Using var-args here provides flexibility for future configuration expansion
# without breaking existing handler implementations. Once a stable configuration
# contract is agreed upon, we can replace this with a TypedDict/Protocol.
ModelLoader = Callable[..., Awaitable[None]]
WarmupHandler = Callable[..., Awaitable[None]]

ParamUpdater = Callable[[Dict[str, Any]], Awaitable[None]]
OnStreamStart = Callable[[], Awaitable[None]]
OnStreamStop = Callable[[], Awaitable[None]]

class StreamProcessor:
    @classmethod
    def from_handlers(
        cls,
        handler_instance: Any,
        *,
        send_data_interval: Optional[float] = 0.333,
        name: str = "stream-processor",
        port: int = 8000,
        frame_skip_config: Optional[FrameSkipConfig] = None,
        validate_signature: bool = True,
        **server_kwargs
    ):
        """Construct a StreamProcessor by discovering handlers on *handler_instance*."""

        registry = HandlerRegistry()

        for attr_name in dir(handler_instance):
            if attr_name.startswith("_"):
                continue
            attr = getattr(handler_instance, attr_name)
            info = getattr(attr, "_trickle_handler_info", None)
            if not info:
                func_obj = getattr(attr, "__func__", attr)
                func_obj = inspect.unwrap(func_obj)
                info = getattr(func_obj, "_trickle_handler_info", None)
            if not info:
                continue
            registry.register(attr, info)

        video_processor = registry.get("video")
        audio_processor = registry.get("audio")
        model_loader = registry.get("model_loader")
        param_updater = registry.get("param_updater")
        on_stream_start = registry.get("stream_start")
        on_stream_stop = registry.get("stream_stop")
        warmup_handler = registry.get("warmup")

        # If handler_instance is a FrameProcessor, extract its warmup_config
        warmup_config = None
        if isinstance(handler_instance, FrameProcessor):
            warmup_config = getattr(handler_instance, 'warmup_config', None)
            if warmup_config:
                logger.debug(f"Extracted warmup_config from {handler_instance.__class__.__name__}")

        processor = cls(
            video_processor=video_processor,
            audio_processor=audio_processor,
            model_loader=model_loader,
            param_updater=param_updater,
            on_stream_start=on_stream_start,
            on_stream_stop=on_stream_stop,
            warmup_handler=warmup_handler,
            send_data_interval=send_data_interval,
            name=name,
            port=port,
            frame_skip_config=frame_skip_config,
            warmup_config=warmup_config,
            validate_signature=validate_signature,
            **server_kwargs
        )

        processor._handler_registry = registry

        return processor

    def __init__(
        self,
        video_processor: Optional[VideoProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        model_loader: Optional[ModelLoader] = None,
        param_updater: Optional[ParamUpdater] = None,
        on_stream_start: Optional[OnStreamStart] = None,
        on_stream_stop: Optional[OnStreamStop] = None,
        warmup_handler: Optional[WarmupHandler] = None,
        send_data_interval: Optional[float] = 0.333,
        name: str = "stream-processor",
        port: int = 8000,
        frame_skip_config: Optional[FrameSkipConfig] = None,
        warmup_config: Optional[WarmupConfig] = None,
        validate_signature: bool = True,
        **server_kwargs
    ):
        """
        Initialize StreamProcessor with async processing functions.
        
        Args:
            video_processor: Async function that processes VideoFrame objects
            audio_processor: Async function that processes AudioFrame objects  
            model_loader: Optional async function called during load_model phase
            param_updater: Optional async function called when parameters update
            on_stream_start: Optional async function called when stream starts
            on_stream_stop: Optional async function called when stream stops
            send_data_interval: Interval for sending data
            name: Processor name
            port: Server port
            frame_skip_config: Optional frame skipping configuration (None = no frame skipping)
            **server_kwargs: Additional arguments passed to StreamServer
        """
        # Validate that processors are async functions
        for name_label, processor_fn in {
            "video_processor": video_processor,
            "audio_processor": audio_processor,
            "model_loader": model_loader,
            "param_updater": param_updater,
            "on_stream_start": on_stream_start,
            "on_stream_stop": on_stream_stop,
            "warmup_handler": warmup_handler,
        }.items():
            if processor_fn is None:
                continue
            # unwrap descriptors/wrappers and validate coroutine-ness
            candidate = getattr(processor_fn, "__func__", processor_fn)
            candidate = inspect.unwrap(candidate)
            if not inspect.iscoroutinefunction(candidate):
                raise ValueError(f"{name_label} must be an async function")
            if validate_signature:
                self._validate_signature_shape(name_label, processor_fn)
            
        self.video_processor = video_processor
        self.audio_processor = audio_processor
        self.model_loader = model_loader
        self.param_updater = param_updater
        self.on_stream_start = on_stream_start
        self.on_stream_stop = on_stream_stop
        self.warmup_handler = warmup_handler
        self.send_data_interval = send_data_interval
        self.name = name
        self.port = port
        self.frame_skip_config = frame_skip_config
        self.server_kwargs = server_kwargs
        self._handler_registry: Optional[HandlerRegistry] = None
        
        # Create internal frame processor
        self._frame_processor = _InternalFrameProcessor(
            video_processor=video_processor,
            audio_processor=audio_processor,
            model_loader=model_loader,
            param_updater=param_updater,
            on_stream_start=on_stream_start,
            on_stream_stop=on_stream_stop,
            warmup_handler=warmup_handler,
            name=name,
            warmup_config=warmup_config
        )
        
        # Create and start server
        self.server = StreamServer(
            frame_processor=self._frame_processor,
            port=port,
            frame_skip_config=frame_skip_config,
            **server_kwargs
        )

        # Attach server state to processor for health transitions
        self._frame_processor.attach_state(self.server.state)
    
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
        try:
            if self.model_loader:
                # Trigger model loading for processors with models (non-blocking due to async lock)
                # The server will start and be available while model loading happens
                asyncio.create_task(self._trigger_model_loading())
            else:
                # For processors without models, mark startup complete immediately
                self.server.state.set_startup_complete()
            
            # Run server (this will start immediately and be available for /health checks)
            await self.server.run_forever()
            
        except Exception as e:
            logger.error(f"Error in StreamProcessor: {e}")
            raise
    
    async def _trigger_model_loading(self):
        """Trigger model loading via parameter update.
        
        If model loading fails, the error is logged and propagated to ensure
        the server state reflects the failure properly.
        """
        try:
            await self._frame_processor.update_params({"load_model": True})
            logger.debug(f"Model loading triggered via parameter update for '{self.name}'")
        except Exception as e:
            logger.error(f"Failed to trigger model loading: {e}")
            self.server.state.set_error(f"Model loading failed: {e}")
            raise
    
    def run(self):
        """Run the stream processor server (blocking)."""
        asyncio.run(self.run_forever())

    @staticmethod
    def _validate_signature_shape(name_label: str, fn: Callable[..., Any]) -> None:
        """Best-effort validation of handler signatures.

        - video_processor(frame: VideoFrame) -> Awaitable[Optional[VideoFrame]]
        - audio_processor(frame: AudioFrame) -> Awaitable[Optional[List[AudioFrame]]]
        - model_loader(...) -> Awaitable[None]
        - warmup_handler(...) -> Awaitable[None]
        - param_updater(params: Dict[str, Any]) -> Awaitable[None]
        - on_stream_start() -> Awaitable[None]
        - on_stream_stop() -> Awaitable[None]
        """
        try:
            base_fn = getattr(fn, "__func__", fn)
            sig = inspect.signature(base_fn)
            params = list(sig.parameters.values())

            if params and params[0].name == "self":
                params = params[1:]

            if name_label in {"on_stream_start", "on_stream_stop", "model_loader", "warmup_handler"}:
                # allow any for model_loader; strict zero-arg for lifecycle
                if name_label.startswith("on_stream") and any(p.kind == p.POSITIONAL_OR_KEYWORD for p in params):
                    logger.warning("%s expected no parameters, got %s", name_label, [p.name for p in params])
                return
            if name_label == "param_updater":
                if not params:
                    logger.warning("param_updater expected a 'params' argument")
                return
            if name_label in {"video_processor", "audio_processor"}:
                if not params:
                    logger.warning("%s expected a 'frame' argument", name_label)
                return
        except Exception:
            logger.debug("Could not validate signature for %s", name_label, exc_info=True)

class _InternalFrameProcessor(FrameProcessor):
    """Internal frame processor that wraps user-provided functions."""
    
    def __init__(
        self,
        video_processor: Optional[VideoProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        model_loader: Optional[ModelLoader] = None,
        param_updater: Optional[ParamUpdater] = None,
        on_stream_start: Optional[OnStreamStart] = None,
        on_stream_stop: Optional[OnStreamStop] = None,
        warmup_handler: Optional[WarmupHandler] = None,
        name: str = "internal-processor",
        warmup_config: Optional['WarmupConfig'] = None
    ):
        # Set attributes first before calling parent
        self.video_processor = video_processor
        self.audio_processor = audio_processor
        self.model_loader = model_loader
        self.param_updater = param_updater
        self.on_stream_start_callback = on_stream_start
        self.on_stream_stop_callback = on_stream_stop
        self.warmup_handler = warmup_handler
        self.name = name
        
        # Frame skipping is handled at TrickleClient level
        self.frame_skip_config = None
        self.frame_skipper = None
        
        # Pass warmup_config to base class so it knows not to call set_startup_complete early
        super().__init__(error_callback=None, warmup_config=warmup_config)

    async def warmup(self, **kwargs):
        """Run warmup using provided async function."""
        if self.warmup_handler:
            try:
                await self.warmup_handler(**kwargs)
                logger.info(f"StreamProcessor '{self.name}' warmup completed successfully")
            except Exception as e:
                logger.error(f"Error in warmup handler: {e}")
                raise

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
    
    async def _handle_control_sentinels(self, params: Dict[str, Any]) -> None:
        """Process internal sentinel parameters like load_model and warmup."""
        if not params:
            return

        if params.pop("load_model", False):
            try:
                await self.ensure_model_loaded()
                logger.info(f"Model loaded via parameter update for '{self.name}'")
            except Exception as e:
                logger.error(f"Error loading model via parameter update: {e}")
                raise

        warmup_params = params.pop("warmup", None)
        if warmup_params is not None:
            kwargs = warmup_params if isinstance(warmup_params, dict) else {}
            try:
                await self.trigger_warmup(wait=False, **kwargs)
                logger.info(f"Warmup triggered via parameter update for '{self.name}'")
            except Exception as e:
                logger.error(f"Error triggering warmup: {e}")
                raise
    
    async def update_params(self, params: Dict[str, Any]):
        """Update parameters using provided async function.
        
        Handles the 'load_model' sentinel parameter internally by extracting it
        and triggering model loading, while passing through all other parameters
        to the user's param_updater callback.
        """
        await self._handle_control_sentinels(params)

        # Normal parameter updates (will include other params even if load_model was present)
        if self.param_updater and params:
            try:
                await self.param_updater(params)
                logger.info(f"Parameters updated: {params}")
            except Exception as e:
                logger.error(f"Error updating parameters: {e}")
    
    async def on_stream_start(self):
        """Call user-provided on_stream_start callback."""
        if self.on_stream_start_callback:
            try:
                await self.on_stream_start_callback()
                logger.info(f"StreamProcessor '{self.name}' stream start callback executed successfully")
            except Exception as e:
                logger.error(f"Error in stream start callback: {e}")
    
    async def on_stream_stop(self):
        """Call user-provided on_stream_stop callback."""
        if self.on_stream_stop_callback:
            try:
                await self.on_stream_stop_callback()
                logger.info(f"StreamProcessor '{self.name}' stream stop callback executed successfully")
            except Exception as e:
                logger.error(f"Error in stream stop callback: {e}")
