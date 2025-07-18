#!/usr/bin/env python3
"""
Audio Tensor Example

Demonstrates how to use tensor_to_av_audio_frame and AudioFrame.from_tensor
for creating and processing audio using PyTorch tensors.
"""

import torch
import numpy as np
import math
from pytrickle.frames import AudioFrame, AudioOutput
from pytrickle.tensors import tensor_to_av_audio_frame

def generate_sine_wave_tensor(frequency: float = 440.0, duration: float = 1.0, 
                             sample_rate: int = 48000, amplitude: float = 0.5) -> torch.Tensor:
    """Generate a sine wave as a PyTorch tensor."""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    sine_wave = amplitude * torch.sin(2 * math.pi * frequency * t)
    return sine_wave

def tensor_audio_processor(frame: AudioFrame) -> AudioOutput:
    """
    Example audio processor that works with tensors.
    Applies a simple gain and adds some harmonics.
    """
    # Convert to tensor
    audio_tensor = frame.to_tensor()
    
    # Apply gain
    processed = audio_tensor * 1.2
    
    # Add second harmonic at lower amplitude
    if len(audio_tensor) > 1:
        # Simple harmonic generation by doubling frequency (taking every other sample)
        harmonic = audio_tensor[::2]  # Downsample to double frequency
        # Pad to original length
        harmonic_padded = torch.zeros_like(audio_tensor)
        harmonic_padded[:len(harmonic)] = harmonic * 0.1
        processed = processed + harmonic_padded
    
    # Clamp to prevent clipping
    processed = torch.clamp(processed, -1.0, 1.0)
    
    # Create new audio frame from tensor
    new_frame = AudioFrame.from_tensor(
        processed, 
        sample_rate=frame.rate,
        layout=frame.layout,
        timestamp=frame.timestamp
    )
    
    return AudioOutput([new_frame], "tensor_processed")

def demo_audio_tensor_functions():
    """Demonstrate audio tensor functions."""
    print("=== Audio Tensor Demo ===")
    
    # Generate a 440Hz sine wave (A note)
    print("1. Generating 440Hz sine wave tensor...")
    sine_tensor = generate_sine_wave_tensor(frequency=440.0, duration=0.5)
    print(f"   Shape: {sine_tensor.shape}")
    print(f"   Range: [{sine_tensor.min():.3f}, {sine_tensor.max():.3f}]")
    
    # Convert tensor to AudioFrame using the new from_tensor method
    print("2. Converting tensor to AudioFrame...")
    audio_frame = AudioFrame.from_tensor(sine_tensor, sample_rate=48000, layout='mono')
    print(f"   AudioFrame - samples: {audio_frame.nb_samples}, rate: {audio_frame.rate}")
    print(f"   Format: {audio_frame.format}, Layout: {audio_frame.layout}")
    
    # Process the audio frame using tensor operations
    print("3. Processing audio with tensor operations...")
    processed_output = tensor_audio_processor(audio_frame)
    processed_frame = processed_output.frames[0]
    print(f"   Processed frame - samples: {processed_frame.nb_samples}")
    
    # Convert back to tensor to verify
    print("4. Converting back to tensor...")
    result_tensor = processed_frame.to_tensor()
    print(f"   Result tensor shape: {result_tensor.shape}")
    print(f"   Result range: [{result_tensor.min():.3f}, {result_tensor.max():.3f}]")
    
    # Test stereo conversion
    print("5. Testing stereo conversion...")
    stereo_tensor = sine_tensor.unsqueeze(1).repeat(1, 2)  # [samples, 2]
    print(f"   Stereo tensor shape: {stereo_tensor.shape}")
    
    stereo_frame = AudioFrame.from_tensor(stereo_tensor, sample_rate=48000, layout='stereo')
    print(f"   Stereo AudioFrame - format: {stereo_frame.format}, layout: {stereo_frame.layout}")
    
    # Test direct av.AudioFrame creation
    print("6. Testing direct tensor_to_av_audio_frame...")
    av_frame = tensor_to_av_audio_frame(sine_tensor, sample_rate=48000, layout='mono')
    print(f"   av.AudioFrame - samples: {av_frame.samples}, rate: {av_frame.sample_rate}")
    print(f"   av.AudioFrame format: {av_frame.format.name}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    demo_audio_tensor_functions()
