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
