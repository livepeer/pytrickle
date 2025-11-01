"""Decorators and handler registry for PyTrickle."""

from __future__ import annotations

import asyncio
import inspect
import logging
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    overload,
    Protocol,
    get_type_hints,
    ParamSpec,
    cast,
    Concatenate,
    TypeVar,
)

from .frames import VideoFrame, AudioFrame
from .registry import HandlerFn, HandlerInfo, HandlerKind

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

# Protocol definitions for decorated handler signatures
class VideoHandlerProtocol(Protocol):
    """Protocol for decorated video handler functions."""
    async def __call__(self, *args: VideoFrame) -> Optional[VideoFrame]: ...

class AudioHandlerProtocol(Protocol):
    """Protocol for decorated audio handler functions."""
    async def __call__(self, *args: AudioFrame) -> Optional[List[AudioFrame]]: ...

class ModelLoaderProtocol(Protocol):
    """Protocol for decorated model loader functions."""
    async def __call__(self, *args: Any, **kwargs: Any) -> None: ...

class ParamUpdaterProtocol(Protocol):
    """Protocol for decorated parameter updater functions."""
    async def __call__(self, params: Dict[str, Any]) -> None: ...

class LifecycleProtocol(Protocol):
    """Protocol for decorated lifecycle functions (start/stop)."""
    async def __call__(self) -> None: ...


def _validate_signature(func: HandlerFn, handler_type: HandlerKind) -> None:
    signature = inspect.signature(func)
    parameters = list(signature.parameters.values())

    # Strip self if method handler
    if parameters and parameters[0].name == "self":
        parameters = parameters[1:]

    if handler_type in ("video", "audio"):
        expected_type = VideoFrame if handler_type == "video" else AudioFrame

        # Try annotations
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        type_ok = any(
            hints.get(p.name) is expected_type
            for p in parameters
        )

        name_ok = any(
            p.name == "frame" for p in parameters
        )

        if not (type_ok or name_ok):
            logger.warning(
                "%s handler '%s' has no parameter matching VideoFrame/AudioFrame. "
                "Expected a param annotated as %s or named 'frame'. Got: %s",
                handler_type,
                func.__name__,
                expected_type.__name__,
                [p.name for p in parameters],
            )

    # param_updater
    elif handler_type == "param_updater":
        if not parameters:
            logger.warning(
                "Param updater '%s' has no parameters. Expected something like (params: dict).",
                func.__name__,
            )


def trickle_handler(
    handler_type: HandlerKind,
    description: Optional[str] = None,
    validate_signature: bool = True,
) -> Callable[[HandlerFn], HandlerFn]:
    """Base decorator that tags a function as a PyTrickle handler."""

    def decorator(func: HandlerFn) -> HandlerFn:
        if validate_signature:
            _validate_signature(func, handler_type)

        desc = description or f"{handler_type} handler: {func.__name__}"
        info = HandlerInfo(
            handler_type=handler_type,
            description=desc,
            signature=str(inspect.signature(func)),
        )

        setattr(func, "_trickle_handler", True)
        setattr(func, "_trickle_handler_type", handler_type)
        setattr(func, "_trickle_handler_info", info)
        return func

    return decorator


def _is_coro_fn(func: HandlerFn) -> bool:
    """Return True if *func* is an async function."""

    return inspect.iscoroutinefunction(func)


async def _maybe_await(func: HandlerFn, *args: Any, **kwargs: Any) -> Any:
    """Await *func* if necessary, running sync call in a thread."""
    if _is_coro_fn(func):
        return await func(*args, **kwargs)
    return await asyncio.to_thread(func, *args, **kwargs)


def _wrap_handler(
    handler_type: HandlerKind,
    description: Optional[str] = None,
    validate_signature: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Awaitable[Any]]]:
    """Common decorator factory for handler wrappers."""

    def outer(func: HandlerFn) -> Callable[..., Awaitable[Any]]:
        trickle_handler(
            handler_type,
            description=description,
            validate_signature=validate_signature,
        )(func)

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await _maybe_await(func, *args, **kwargs)

        setattr(wrapper, "_trickle_handler", True)
        setattr(wrapper, "_trickle_handler_type", handler_type)
        setattr(wrapper, "_trickle_handler_info", getattr(func, "_trickle_handler_info"))
        return wrapper

    return outer


@overload
def video_handler(
    func: Callable[Concatenate[VideoFrame, P], Any],
) -> Callable[Concatenate[VideoFrame, P], Awaitable[Optional[VideoFrame]]]: ...

@overload
def video_handler(
    func: Callable[Concatenate[T, VideoFrame, P], Any],
) -> Callable[Concatenate[T, VideoFrame, P], Awaitable[Optional[VideoFrame]]]: ...

def video_handler(
    func: HandlerFn,
) -> Any:
    """Decorator for video frame handlers with output normalisation.
    
    The decorated function must have a VideoFrame parameter (named 'frame' or type-annotated).
    The description is automatically generated from the function name.
    
    Type checking: This decorator enforces that the decorated function has a VideoFrame
    parameter. Type checkers will report errors if you use AudioFrame or other types.
    
    Examples:
        @video_handler
        async def process(frame: VideoFrame) -> VideoFrame:
            ...
        
        @video_handler
        async def blur(self, frame: VideoFrame) -> VideoFrame:
            ...
    """
    import numpy as np
    import torch

    base_wrapper = _wrap_handler(
        "video",
        description=f"Video handler: {func.__name__}",
        validate_signature=True,
    )(func)

    shape_warn_count = 0

    def _maybe_warn_shape_mismatch(result_shape: Any, orig_shape: Any) -> None:
        nonlocal shape_warn_count
        shape_warn_count += 1
        # Rate-limit warnings: first 3 occurrences, then every 100th
        if shape_warn_count <= 3 or shape_warn_count % 100 == 0:
            logger.warning(
                "Video handler '%s' returned output with unexpected shape %s; "
                "expected same ndim and channels in (1, 3) as input %s. (seen=%d)",
                func.__name__, result_shape, orig_shape, shape_warn_count,
            )

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Optional[VideoFrame]:
        frame = kwargs.get("frame") if "frame" in kwargs else (args[-1] if args else None)
        result = await base_wrapper(*args, **kwargs)

        if result is None:
            return None
        if isinstance(result, VideoFrame):
            return result
        if frame is None or not isinstance(frame, VideoFrame):
            return None

        # Validate basic shape compatibility
        try:
            original_frame = frame
            orig = original_frame.tensor

            orig_ndim = orig.dim() if hasattr(orig, "dim") else getattr(orig, "ndim", -1)

            if torch is not None and torch.is_tensor(result):
                if result.dim() == getattr(orig, "dim", lambda: -1)() and result.shape[-1] in (1, 3):
                    return original_frame.replace_tensor(result)
                _maybe_warn_shape_mismatch(getattr(result, "shape", None), getattr(orig, "shape", None))
                return None
            if np is not None and isinstance(result, np.ndarray):
                if result.ndim == orig_ndim and result.shape[-1] in (1, 3):
                    if torch is not None:
                        return original_frame.replace_tensor(torch.from_numpy(result))
                    # Torch not available: cannot produce a Tensor for replace_tensor
                    logger.warning("Received numpy array but torch is unavailable; dropping frame")
                    return None
                _maybe_warn_shape_mismatch(getattr(result, "shape", None), getattr(orig, "shape", None))
                return None
        except Exception:
            logger.exception("Error normalizing video handler output")
            return None
        return None

    setattr(wrapper, "_trickle_handler", True)
    setattr(wrapper, "_trickle_handler_type", "video")
    setattr(wrapper, "_trickle_handler_info", getattr(func, "_trickle_handler_info", None))
    return cast(Callable[..., Awaitable[Optional[VideoFrame]]], wrapper)

@overload
def audio_handler(
    func: Callable[Concatenate[AudioFrame, P], Any],
) -> Callable[Concatenate[AudioFrame, P], Awaitable[Optional[List[AudioFrame]]]]: ...

@overload
def audio_handler(
    func: Callable[Concatenate[T, AudioFrame, P], Any],
) -> Callable[Concatenate[T, AudioFrame, P], Awaitable[Optional[List[AudioFrame]]]]: ...

def audio_handler(
    func: HandlerFn,
) -> Any:
    """Decorator for audio frame handlers with output normalisation.
    
    The decorated function must have an AudioFrame parameter (named 'frame' or type-annotated).
    The description is automatically generated from the function name.
    
    Type checking: This decorator enforces that the decorated function has an AudioFrame
    parameter. Type checkers will report errors if you use VideoFrame or other types.
    
    Examples:
        @audio_handler
        async def process(frame: AudioFrame) -> List[AudioFrame]:
            ...
        
        @audio_handler
        async def echo(self, frame: AudioFrame) -> List[AudioFrame]:
            ...
    """
    import numpy as np
    import torch

    base_wrapper = _wrap_handler(
        "audio",
        description=f"Audio handler: {func.__name__}",
        validate_signature=True,
    )(func)

    type_warn_count = 0

    def _maybe_warn_invalid_type(result: Any) -> None:
        nonlocal type_warn_count
        type_warn_count += 1
        if type_warn_count <= 3 or type_warn_count % 100 == 0:
            logger.warning(
                "Audio handler '%s' returned unsupported type %s; expected List[AudioFrame], AudioFrame, torch.Tensor, or numpy.ndarray. (seen=%d)",
                func.__name__, type(result).__name__, type_warn_count,
            )

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Optional[List[AudioFrame]]:
        frame = kwargs.get("frame") if "frame" in kwargs else (args[-1] if args else None)
        result = await base_wrapper(*args, **kwargs)

        if result is None:
            return None
        if isinstance(result, list):
            return result
        if isinstance(result, AudioFrame):
            return [result]
        if frame is None or not isinstance(frame, AudioFrame):
            return None

        try:
            original_frame: AudioFrame = frame
            if torch is not None and torch.is_tensor(result):
                samples = result.detach().cpu().numpy()
                return [original_frame.replace_samples(samples)]
            if np is not None and isinstance(result, np.ndarray):
                return [original_frame.replace_samples(result)]
            # Unsupported type: issue a rate-limited warning for parity with video handler
            _maybe_warn_invalid_type(result)
        except Exception:
            logger.exception("Error normalizing audio handler output")
            return None
        return None

    setattr(wrapper, "_trickle_handler", True)
    setattr(wrapper, "_trickle_handler_type", "audio")
    setattr(wrapper, "_trickle_handler_info", getattr(func, "_trickle_handler_info", None))
    return cast(Callable[..., Awaitable[Optional[List[AudioFrame]]]], wrapper)


def model_loader(
    func: HandlerFn,
) -> ModelLoaderProtocol:
    """Decorator for model loader handlers.
    
    The description is automatically generated from the function name.
    """
    base_wrapper = _wrap_handler(
        "model_loader",
        description=f"Model loader: {func.__name__}",
        validate_signature=True,
    )(func)

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> None:
        await base_wrapper(*args, **kwargs)
        return None

    setattr(wrapper, "_trickle_handler", True)
    setattr(wrapper, "_trickle_handler_type", "model_loader")
    setattr(wrapper, "_trickle_handler_info", getattr(func, "_trickle_handler_info", None))
    return wrapper


def param_updater(
    func: HandlerFn,
) -> ParamUpdaterProtocol:
    """Decorator for parameter update handlers.
    
    The description is automatically generated from the function name.
    """
    base_wrapper = _wrap_handler(
        "param_updater",
        description=f"Parameter updater: {func.__name__}",
        validate_signature=True,
    )(func)

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> None:
        await base_wrapper(*args, **kwargs)
        return None

    setattr(wrapper, "_trickle_handler", True)
    setattr(wrapper, "_trickle_handler_type", "param_updater")
    setattr(wrapper, "_trickle_handler_info", getattr(func, "_trickle_handler_info", None))
    return wrapper


def on_stream_start(
    func: HandlerFn,
) -> LifecycleProtocol:
    """Decorator for stream start lifecycle handler.
    
    The description is automatically generated from the function name.
    """
    base_wrapper = _wrap_handler(
        "stream_start",
        description=f"Stream start handler: {func.__name__}",
        validate_signature=True,
    )(func)

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> None:
        await base_wrapper(*args, **kwargs)
        return None

    setattr(wrapper, "_trickle_handler", True)
    setattr(wrapper, "_trickle_handler_type", "stream_start")
    setattr(wrapper, "_trickle_handler_info", getattr(func, "_trickle_handler_info", None))
    return wrapper


def on_stream_stop(
    func: HandlerFn,
) -> LifecycleProtocol:
    """Decorator for stream stop lifecycle handler.
    
    The description is automatically generated from the function name.
    """
    base_wrapper = _wrap_handler(
        "stream_stop",
        description=f"Stream stop handler: {func.__name__}",
        validate_signature=True,
    )(func)

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> None:
        await base_wrapper(*args, **kwargs)
        return None

    setattr(wrapper, "_trickle_handler", True)
    setattr(wrapper, "_trickle_handler_type", "stream_stop")
    setattr(wrapper, "_trickle_handler_info", getattr(func, "_trickle_handler_info", None))
    return wrapper
