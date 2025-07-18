"""
Tensor utility functions for pytrickle.
"""

import torch
import numpy as np
import av


def tensor_to_av_frame(tensor: torch.Tensor) -> av.VideoFrame:
    """
    Convert a tensor to av.VideoFrame for use in video pipelines.
    Handles [B, H, W, C] or [H, W, C] formats, float or uint8, and grayscale/RGB.
    """
    try:
        # Handle tensor format conversion - trickle uses [B, H, W, C] or [H, W, C]
        if tensor.dim() == 4:
            # Expected format: [B, H, W, C] where B=1
            if tensor.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got {tensor.shape[0]}")
            tensor = tensor.squeeze(0)  # Remove batch dimension: [H, W, C]
        elif tensor.dim() == 3:
            # Already in [H, W, C] format
            pass
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D tensor with shape {tensor.shape}")

        # Validate tensor format
        if tensor.dim() != 3:
            raise ValueError(f"Expected 3D tensor after conversion, got {tensor.dim()}D")
        if tensor.shape[2] not in [1, 3, 4]:
            raise ValueError(f"Expected 1, 3, or 4 channels, got {tensor.shape[2]}")

        # Convert tensor to numpy array for av.VideoFrame
        # Handle different tensor value ranges
        if tensor.dtype in [torch.float32, torch.float64]:
            if tensor.max() <= 1.0:
                # Tensor is in [0, 1] range, convert to [0, 255]
                tensor_np = (tensor * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
            else:
                # Tensor is already in [0, 255] range
                tensor_np = tensor.clamp(0, 255).to(torch.uint8).cpu().numpy()
        elif tensor.dtype == torch.uint8:
            tensor_np = tensor.cpu().numpy()
        else:
            # Convert other types to uint8
            tensor_np = tensor.clamp(0, 255).to(torch.uint8).cpu().numpy()

        # Ensure numpy array is contiguous
        if not tensor_np.flags.c_contiguous:
            tensor_np = np.ascontiguousarray(tensor_np)

        # Handle grayscale to RGB conversion if needed
        if tensor_np.shape[2] == 1:
            tensor_np = np.repeat(tensor_np, 3, axis=2)

        # Create av.VideoFrame from numpy array
        av_frame = av.VideoFrame.from_ndarray(tensor_np, format="rgb24")

        return av_frame

    except Exception as e:
        # Optionally, you could log here if logger is available
        raise


def tensor_to_av_audio_frame(tensor: torch.Tensor, sample_rate: int = 48000, layout: str = 'mono') -> av.AudioFrame:
    """
    Convert a tensor to av.AudioFrame for use in audio pipelines.
    
    Args:
        tensor: Audio tensor in format [samples] for mono or [samples, channels] for multi-channel
        sample_rate: Sample rate for the audio (default: 48000 Hz)
        layout: Audio channel layout ('mono', 'stereo', etc.)
    
    Returns:
        av.AudioFrame: Audio frame ready for encoding
    """
    try:
        # Handle tensor format conversion
        if tensor.dim() == 1:
            # Mono audio: [samples]
            if layout != 'mono':
                # Convert mono to multi-channel by repeating
                if layout == 'stereo':
                    tensor = tensor.unsqueeze(1).repeat(1, 2)  # [samples, 2]
                else:
                    raise ValueError(f"Cannot convert mono tensor to layout '{layout}'")
        elif tensor.dim() == 2:
            # Multi-channel audio: [samples, channels]
            if layout == 'mono' and tensor.shape[1] > 1:
                # Convert multi-channel to mono by averaging channels
                tensor = tensor.mean(dim=1)  # [samples]
            elif layout == 'stereo' and tensor.shape[1] == 1:
                # Convert mono to stereo
                tensor = tensor.repeat(1, 2)  # [samples, 2]
        else:
            raise ValueError(f"Expected 1D or 2D tensor, got {tensor.dim()}D tensor with shape {tensor.shape}")
        
        # Convert tensor to numpy array for av.AudioFrame
        # Handle different tensor value ranges and types
        if tensor.dtype in [torch.float32, torch.float64]:
            # Float tensors should be in [-1.0, 1.0] range for audio
            tensor_np = tensor.clamp(-1.0, 1.0).cpu().numpy()
            audio_format = 'fltp'  # planar float
        elif tensor.dtype in [torch.int16]:
            # 16-bit integer audio
            tensor_np = tensor.cpu().numpy()
            audio_format = 's16p'  # planar 16-bit signed integer
        elif tensor.dtype in [torch.int32]:
            # 32-bit integer audio
            tensor_np = tensor.cpu().numpy()
            audio_format = 's32p'  # planar 32-bit signed integer
        else:
            # Convert other types to float32
            if tensor.max() > 1.0 or tensor.min() < -1.0:
                # Normalize if outside [-1, 1] range
                max_abs = max(abs(tensor.max()), abs(tensor.min()))
                tensor = tensor / max_abs
            tensor_np = tensor.float().clamp(-1.0, 1.0).cpu().numpy()
            audio_format = 'fltp'
        
        # Ensure numpy array is contiguous
        if not tensor_np.flags.c_contiguous:
            tensor_np = np.ascontiguousarray(tensor_np)
        
        # Handle channel layout for av.AudioFrame
        if tensor_np.ndim == 1:
            # Mono: reshape to [1, samples] for planar format
            tensor_np = tensor_np.reshape(1, -1)
        elif tensor_np.ndim == 2:
            # Multi-channel: transpose to [channels, samples] for planar format
            tensor_np = tensor_np.T
        
        # Create av.AudioFrame from numpy array
        av_frame = av.AudioFrame.from_ndarray(
            tensor_np,
            format=audio_format,
            layout=layout
        )
        av_frame.sample_rate = sample_rate
        
        return av_frame
        
    except Exception as e:
        # Optionally, you could log here if logger is available
        raise
