#!/usr/bin/env python3
"""
Audio Pitch Shifter using StreamProcessor

This example demonstrates real-time audio pitch shifting using torchaudio.transforms.PitchShift.
Supports pitch shifts from -12 to +12 semitones with dynamic parameter updates.

See docs/audio_pitch_shifting.md for detailed documentation.

Usage:
    python examples/process_audio_example.py
    
API:
    POST http://localhost:8001/update_params
    {"pitch_shift": 5.0}  # +5 semitones (perfect fourth up)
"""

import logging
import torch
import torchaudio
from pytrickle import StreamProcessor
from pytrickle.frames import AudioFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
pitch_shift = 0.0  # Pitch shift in semitones (-12 to +12)
pitch_shifter = None
ready = False

def load_model(**kwargs):
    """Initialize processor state - called during model loading phase."""
    global pitch_shift, pitch_shifter, ready
    
    # Set processor variables from kwargs or use defaults
    pitch_shift = kwargs.get('pitch_shift', 0.0)
    pitch_shift = max(-12.0, min(12.0, pitch_shift))
    
    # Initialize the pitch shifter transform
    # Note: Sample rate will be set when we process the first frame
    pitch_shifter = None
    
    ready = True
    logger.info(f"✅ Audio Pitch Shifter ready (pitch_shift: {pitch_shift:.1f} semitones)")

async def process_audio(frame: AudioFrame) -> AudioFrame:
    """Apply pitch shifting to audio frame using torchaudio.transforms.PitchShift."""
    global pitch_shift, pitch_shifter, ready
    
    # Return frames unchanged if not ready or no pitch shift
    if not ready or abs(pitch_shift) < 0.01:
        return frame
    
    try:
        # Initialize pitch shifter if needed
        if pitch_shifter is None:
            pitch_shifter = torchaudio.transforms.PitchShift(
                sample_rate=frame.rate,
                n_steps=pitch_shift
            )
        else:
            # Update pitch shift if needed
            if abs(pitch_shifter.n_steps - pitch_shift) > 0.01:
                pitch_shifter = torchaudio.transforms.PitchShift(
                    sample_rate=frame.rate,
                    n_steps=pitch_shift
                )
        
        # Convert AudioFrame to tensor
        audio_tensor = frame.to_tensor()
        
        # Apply pitch shifting
        shifted_tensor = pitch_shifter(audio_tensor)
        
        # Convert back to AudioFrame
        return AudioFrame.from_tensor(
            shifted_tensor,
            format=frame.format,
            layout=frame.layout,
            sample_rate=frame.rate,
            timestamp=frame.timestamp,
            time_base=frame.time_base
        )
        
    except Exception as e:
        logger.warning(f"Pitch shifting failed: {e}, returning original frame")
        return frame

def update_params(params: dict):
    """Update pitch shift (-12.0 to +12.0 semitones)."""
    global pitch_shift, pitch_shifter
    if "pitch_shift" in params:
        old = pitch_shift
        pitch_shift = max(-12.0, min(12.0, float(params["pitch_shift"])))
        if abs(old - pitch_shift) > 0.01:
            # Reset pitch shifter to pick up new parameters
            pitch_shifter = None
            logger.info(f"Pitch shift: {old:.1f} → {pitch_shift:.1f} semitones")

# Create and run StreamProcessor
if __name__ == "__main__":
    processor = StreamProcessor(
        audio_processor=process_audio,
        model_loader=load_model,
        param_updater=update_params,
        name="audio-pitch-shifter",
        port=8001
    )
    processor.run()