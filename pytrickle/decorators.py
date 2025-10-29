"""Decorators and handler registry for PyTrickle."""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

HandlerFn = Callable[..., Any]


@dataclass
class HandlerInfo:
    """Metadata captured for each registered handler."""

    handler_type: str
    function_name: str
    description: Optional[str] = None
    signature: Optional[str] = None


class HandlerRegistry:
    """Registry that stores the active handlers for a component."""

    def __init__(self) -> None:
        self._handlers: Dict[str, HandlerFn] = {}
        self._info: Dict[str, HandlerInfo] = {}

    def register(self, handler: HandlerFn, info: HandlerInfo) -> None:
        """Register a handler, warning if another handler is replaced."""

        if info.handler_type in self._handlers:
            previous = self._info[info.handler_type].function_name
            logger.warning(
                "Overwriting handler '%s': %s -> %s",
                info.handler_type,
                previous,
                info.function_name,
            )
        self._handlers[info.handler_type] = handler
        self._info[info.handler_type] = info

    def get(self, handler_type: str) -> Optional[HandlerFn]:
        """Return the handler for the given type, if registered."""

        return self._handlers.get(handler_type)


def _validate_signature(func: HandlerFn, handler_type: str) -> None:
    """Log a warning if the handler signature looks unexpected."""

    signature = inspect.signature(func)
    parameters = list(signature.parameters.keys())

    expected_args: Dict[str, List[str]] = {
        "video": ["frame"],
        "audio": ["frame"],
        "model_loader": [],
        "param_updater": ["params"],
        "stream_start": [],
        "stream_stop": [],
    }
    expected = expected_args.get(handler_type, [])

    if parameters and parameters[0] == "self":
        expected = ["self", *expected]

    missing = [name for name in expected if name not in parameters]
    if missing:
        logger.warning(
            "Handler '%s' (%s) missing params %s. Expected %s, got %s",
            func.__name__,
            handler_type,
            missing,
            expected,
            parameters,
        )


def trickle_handler(
    handler_type: str,
    *,
    description: Optional[str] = None,
    validate_signature: bool = False,
) -> Callable[[HandlerFn], HandlerFn]:
    """Base decorator that tags a function as a PyTrickle handler."""

    def decorator(func: HandlerFn) -> HandlerFn:
        if validate_signature:
            _validate_signature(func, handler_type)

        info = HandlerInfo(
            handler_type=handler_type,
            function_name=func.__name__,
            description=description,
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
    handler_type: str,
    *,
    description: Optional[str] = None,
    validate_signature: bool = False,
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


def video_handler(
    func: Optional[HandlerFn] = None,
    *,
    description: Optional[str] = None,
    validate_signature: bool = False,
) -> Callable[[HandlerFn], Callable[..., Awaitable[Any]]]:
    """Decorator for video frame handlers with output normalisation."""

    def decorate(target: HandlerFn) -> Callable[..., Awaitable[Any]]:
        from .frames import VideoFrame
        import numpy as np
        import torch

        base_wrapper = _wrap_handler(
            "video",
            description=description or f"Video handler: {target.__name__}",
            validate_signature=validate_signature,
        )(target)

        @wraps(target)
        async def wrapper(*args: Any, **kwargs: Any) -> Optional[VideoFrame]:
            frame = kwargs.get("frame") if "frame" in kwargs else (args[-1] if args else None)
            result = await base_wrapper(*args, **kwargs)

            if result is None:
                return None
            if isinstance(result, VideoFrame):
                return result
            if frame is None or not isinstance(frame, VideoFrame):
                return None

            original_frame: VideoFrame = frame
            if isinstance(result, torch.Tensor):
                return original_frame.replace_tensor(result)
            if isinstance(result, np.ndarray):
                tensor = torch.from_numpy(result)
                return original_frame.replace_tensor(tensor)
            return None

        setattr(wrapper, "_trickle_handler", True)
        setattr(wrapper, "_trickle_handler_type", "video")
        setattr(wrapper, "_trickle_handler_info", getattr(target, "_trickle_handler_info"))
        return wrapper

    if func is None:
        return decorate
    return decorate(func)


def audio_handler(
    func: Optional[HandlerFn] = None,
    *,
    description: Optional[str] = None,
    validate_signature: bool = False,
) -> Callable[[HandlerFn], Callable[..., Awaitable[Any]]]:
    """Decorator for audio frame handlers with output normalisation."""

    def decorate(target: HandlerFn) -> Callable[..., Awaitable[Any]]:
        from .frames import AudioFrame
        import numpy as np
        import torch

        base_wrapper = _wrap_handler(
            "audio",
            description=description or f"Audio handler: {target.__name__}",
            validate_signature=validate_signature,
        )(target)

        @wraps(target)
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

            original_frame: AudioFrame = frame
            if isinstance(result, torch.Tensor):
                samples = result.detach().cpu().numpy()
                return [original_frame.replace_samples(samples)]
            if isinstance(result, np.ndarray):
                return [original_frame.replace_samples(result)]
            return None

        setattr(wrapper, "_trickle_handler", True)
        setattr(wrapper, "_trickle_handler_type", "audio")
        setattr(wrapper, "_trickle_handler_info", getattr(target, "_trickle_handler_info"))
        return wrapper

    if func is None:
        return decorate
    return decorate(func)


def model_loader(
    func: Optional[HandlerFn] = None,
    *,
    description: Optional[str] = None,
    validate_signature: bool = False,
) -> Callable[[HandlerFn], Callable[..., Awaitable[Any]]]:
    """Decorator for model loader handlers."""

    def decorate(target: HandlerFn) -> Callable[..., Awaitable[Any]]:
        base_wrapper = _wrap_handler(
            "model_loader",
            description=description or f"Model loader: {target.__name__}",
            validate_signature=validate_signature,
        )(target)

        return base_wrapper

    if func is None:
        return decorate
    return decorate(func)


def param_updater(
    func: Optional[HandlerFn] = None,
    *,
    description: Optional[str] = None,
    validate_signature: bool = False,
) -> Callable[[HandlerFn], Callable[..., Awaitable[Any]]]:
    """Decorator for parameter update handlers."""

    def decorate(target: HandlerFn) -> Callable[..., Awaitable[Any]]:
        base_wrapper = _wrap_handler(
            "param_updater",
            description=description or f"Parameter updater: {target.__name__}",
            validate_signature=validate_signature,
        )(target)

        return base_wrapper

    if func is None:
        return decorate
    return decorate(func)


def on_stream_start(
    func: Optional[HandlerFn] = None,
    *,
    description: Optional[str] = None,
    validate_signature: bool = False,
) -> Callable[[HandlerFn], Callable[..., Awaitable[Any]]]:
    """Decorator for stream start lifecycle handler."""

    def decorate(target: HandlerFn) -> Callable[..., Awaitable[Any]]:
        base_wrapper = _wrap_handler(
            "stream_start",
            description=description or f"Stream start handler: {target.__name__}",
            validate_signature=validate_signature,
        )(target)

        return base_wrapper

    if func is None:
        return decorate
    return decorate(func)


def on_stream_stop(
    func: Optional[HandlerFn] = None,
    *,
    description: Optional[str] = None,
    validate_signature: bool = False,
) -> Callable[[HandlerFn], Callable[..., Awaitable[Any]]]:
    """Decorator for stream stop lifecycle handler."""

    def decorate(target: HandlerFn) -> Callable[..., Awaitable[Any]]:
        base_wrapper = _wrap_handler(
            "stream_stop",
            description=description or f"Stream stop handler: {target.__name__}",
            validate_signature=validate_signature,
        )(target)

        return base_wrapper

    if func is None:
        return decorate
    return decorate(func)
