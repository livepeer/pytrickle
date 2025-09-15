from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Optional, cast

try:
    # Pydantic is optional but available in requirements
    from pydantic import BaseModel  # type: ignore
    _PYDANTIC_AVAILABLE = True
except Exception:  # pragma: no cover - if pydantic missing at runtime
    BaseModel = object  # sentinel
    _PYDANTIC_AVAILABLE = False


def trickle_handler(handler_type: str):
    """
    Base decorator to mark methods as trickle handlers. Lightweight marker only.

    Args:
        handler_type: One of 'video' | 'audio' | 'model_loader' | 'param_updater' | 'stream_stop'.

    Returns:
        The same function with marker attributes set for discovery.

    Notes:
        - This does not wrap the callable or change semantics.
        - Convenience typed decorators below build on this to add DX ergonomics.
    """

    def decorator(func):
        setattr(func, "_trickle_handler_type", handler_type)
        setattr(func, "_trickle_handler", True)
        return func

    return decorator


# ---------- DX helpers: async bridging and return normalization ----------

def _is_coro_fn(fn: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(fn)


async def _maybe_await(fn: Callable[..., Any], *args, **kwargs):
    """Call fn and await if it returns a coroutine or is a coroutine function.

    If fn is synchronous, execute it in a thread to avoid blocking the event loop.
    """
    if _is_coro_fn(fn):
        return await fn(*args, **kwargs)
    # If call returns a coroutine due to dynamic wrappers, still await it
    result = await asyncio.to_thread(fn, *args, **kwargs)
    return result


def video_handler(func: Callable[..., Any]):
    """
    Decorator for video handlers with DX niceties:
    - Accepts sync or async functions (sync runs in thread pool)
    - Accepts flexible return types from user logic:
        * None -> pass-through (internal falls back to original frame)
        * VideoFrame -> used as-is
        * torch.Tensor / numpy.ndarray -> converted via frame.replace_tensor

    Signature expected by framework: (self, frame: VideoFrame) -> Optional[VideoFrame]
    """

    from .frames import VideoFrame  # local import to avoid cycles at import time
    import torch  # type: ignore
    import numpy as np  # type: ignore

    @trickle_handler("video")
    async def wrapper(*args, **kwargs):
        # Expect last positional arg or kw 'frame'
        frame = kwargs.get("frame") if "frame" in kwargs else (args[-1] if args else None)
        result = await _maybe_await(func, *args, **kwargs)

        if result is None:
            return None  # internal will pass-through
        if isinstance(result, VideoFrame):
            return result
        if frame is None:
            return None
        frame_v = cast(VideoFrame, frame)
        if isinstance(result, torch.Tensor):
            return frame_v.replace_tensor(result)
        if isinstance(result, np.ndarray):
            tensor = torch.from_numpy(result)
            return frame_v.replace_tensor(tensor)
        # Unknown type -> pass-through
        return None

    # Keep metadata for debugging/testing
    wrapper.__name__ = getattr(func, "__name__", "video_handler")
    return wrapper


def audio_handler(func: Callable[..., Any]):
    """
    Decorator for audio handlers with DX niceties:
    - Accepts sync or async functions (sync runs in thread pool)
    - Flexible return types:
        * None -> pass-through (becomes [original])
        * AudioFrame -> [that]
        * List[AudioFrame] -> as-is
        * numpy.ndarray / torch.Tensor -> replace samples keeping metadata

    Signature expected by framework: (self, frame: AudioFrame) -> Optional[List[AudioFrame]]
    """

    from .frames import AudioFrame
    import torch  # type: ignore
    import numpy as np  # type: ignore

    @trickle_handler("audio")
    async def wrapper(*args, **kwargs):
        frame = kwargs.get("frame") if "frame" in kwargs else (args[-1] if args else None)
        result = await _maybe_await(func, *args, **kwargs)

        if result is None:
            return None  # internal will convert to [frame]
        if isinstance(result, AudioFrame):
            return [result]
        if isinstance(result, list):
            return result
        # Tensor/ndarray samples
        if frame is None:
            return None
        frame_a = cast(AudioFrame, frame)
        if isinstance(result, torch.Tensor):
            samples = result.detach().cpu().numpy()
            return [frame_a.replace_samples(samples)]
        if isinstance(result, np.ndarray):
            return [frame_a.replace_samples(result)]
        return None

    wrapper.__name__ = getattr(func, "__name__", "audio_handler")
    return wrapper


def model_loader(func: Callable[..., Any]):
    """Decorator for model loader; allows sync or async functions."""

    @trickle_handler("model_loader")
    async def wrapper(*args, **kwargs):
        return await _maybe_await(func, *args, **kwargs)

    wrapper.__name__ = getattr(func, "__name__", "model_loader")
    return wrapper


def param_updater(func: Optional[Callable[..., Any]] = None, *, model: Any | None = None):
    """
    Decorator for parameter updates with optional Pydantic model validation.

    Usage:
        @param_updater
        async def update_params(self, params: Dict[str, Any]): ...

        @param_updater(model=MyParamsModel)
        async def update_params(self, params: MyParamsModel): ...
    """

    def _decorate(fn: Callable[..., Any]):
        @trickle_handler("param_updater")
        async def wrapper(*args, **kwargs):
            # Pull params (positional last or kw)
            has_kw = "params" in kwargs
            incoming = kwargs["params"] if has_kw else (args[-1] if args else None)

            parsed = incoming
            # Parse with Pydantic model if provided and available
            if (
                model is not None
                and _PYDANTIC_AVAILABLE
                and inspect.isclass(model)
                and issubclass(model, BaseModel)  # type: ignore[arg-type]
            ):
                try:
                    if isinstance(incoming, model):  # already validated instance
                        parsed = incoming
                    else:
                        mv = getattr(model, "model_validate", None)
                        if callable(mv):  # Pydantic v2
                            parsed = mv(incoming)
                        elif isinstance(incoming, dict):  # Pydantic v1 expects kwargs
                            parsed = model(**incoming)
                        else:
                            parsed = incoming
                except Exception:
                    # Fall back to original incoming on validation error to avoid breaking stream
                    parsed = incoming

            # Rebuild args/kwargs replacing only the params argument
            new_args = list(args)
            if has_kw:
                kwargs["params"] = parsed
            else:
                if new_args:
                    new_args[-1] = parsed
                else:
                    new_args.append(parsed)

            return await _maybe_await(fn, *tuple(new_args), **kwargs)

        # Attach for discovery/testing
        setattr(wrapper, "_trickle_param_model", model)
        wrapper.__name__ = getattr(fn, "__name__", "param_updater")
        return wrapper

    # Support bare and called decorator forms
    if func is None:
        return _decorate
    return _decorate(func)


def on_stream_stop(func: Callable[..., Any]):
    """Decorator for stream stop callback; allows sync or async functions."""

    @trickle_handler("stream_stop")
    async def wrapper(*args, **kwargs):
        return await _maybe_await(func, *args, **kwargs)

    wrapper.__name__ = getattr(func, "__name__", "on_stream_stop")
    return wrapper

