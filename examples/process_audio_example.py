#!/usr/bin/env python3
"""
Audio Effects Processor using StreamProcessor

This example demonstrates real audio modification with PyTrickle, including:
- Volume adjustment
- Low-pass filtering 
- Echo/reverb effects
- Channel manipulation
"""

import logging
import numpy as np
from scipy import signal
from pytrickle import StreamProcessor
from pytrickle.frames import AudioFrame
import time
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
volume = 1.0
echo_delay = 0.1  # seconds
echo_decay = 0.3  # echo strength (0.0 to 1.0)
lowpass_cutoff = 8000  # Hz
enable_effects = True
delay = 0.0
ready = False

# Echo buffer for storing previous samples
echo_buffer = {}  # Will store buffers per sample rate

def load_model(**kwargs):
    """Initialize audio processor state - called during model loading phase."""
    global volume, echo_delay, echo_decay, lowpass_cutoff, enable_effects, ready
    
    logger.info(f"load_model called with kwargs: {kwargs}")
    
    # Set processor variables from kwargs or use defaults
    volume = max(0.0, min(2.0, kwargs.get('volume', 1.0)))
    echo_delay = max(0.0, min(1.0, kwargs.get('echo_delay', 0.1)))
    echo_decay = max(0.0, min(1.0, kwargs.get('echo_decay', 0.3)))
    lowpass_cutoff = max(100, min(20000, kwargs.get('lowpass_cutoff', 8000)))
    enable_effects = kwargs.get('enable_effects', True)
    
    ready = True
    logger.info(f"✅ Audio effects processor ready:")
    logger.info(f"   📢 Volume: {volume:.2f}")
    logger.info(f"   🔉 Echo delay: {echo_delay:.2f}s, decay: {echo_decay:.2f}")
    logger.info(f"   🎛️ Low-pass cutoff: {lowpass_cutoff} Hz")
    logger.info(f"   ⚡ Effects enabled: {enable_effects}")

async def process_audio(frame: AudioFrame) -> List[AudioFrame]:
    """Apply audio effects including volume, echo, and filtering."""
    global volume, echo_delay, echo_decay, lowpass_cutoff, enable_effects, ready, delay, echo_buffer
    
    # Simulated processing time
    if delay > 0:
        time.sleep(delay)
    
    if not ready or not enable_effects:
        # Pass through unchanged
        return [frame]
    
    try:
        # Get audio samples - they're already numpy arrays
        samples = frame.samples.copy().astype(np.float32)
        sample_rate = frame.rate
        
        # Normalize integer samples to float range [-1, 1] if needed
        if frame.format in ['s16', 's16p']:
            samples = samples / 32768.0
        elif frame.format in ['s32', 's32p']:
            samples = samples / 2147483648.0
        # float formats are already in [-1, 1] range
        
        # Handle different audio layouts
        if samples.ndim == 1:
            # Mono audio
            channels = 1
            samples = samples.reshape(1, -1)
        elif samples.ndim == 2:
            # Multi-channel audio
            if frame.format.endswith('p'):
                # Planar format: (channels, samples)
                channels = samples.shape[0]
            else:
                # Packed format: (samples, channels) -> transpose to (channels, samples)
                samples = samples.T
                channels = samples.shape[0]
        else:
            logger.warning(f"Unexpected audio shape: {samples.shape}")
            return [frame]
        
        # Initialize echo buffer for this sample rate if needed
        buffer_key = f"{sample_rate}_{channels}"
        if buffer_key not in echo_buffer:
            # Create buffer to store echo_delay seconds of audio
            buffer_size = int(sample_rate * echo_delay)
            echo_buffer[buffer_key] = np.zeros((channels, buffer_size), dtype=np.float32)
        
        current_buffer = echo_buffer[buffer_key]
        
        # Process each channel
        processed_samples = np.zeros_like(samples)
        
        for ch in range(channels):
            channel_samples = samples[ch]
            
            # Apply volume adjustment
            channel_samples = channel_samples * volume
            
            # Apply echo effect
            if echo_decay > 0 and current_buffer.shape[1] > 0:
                # Add delayed samples from buffer
                buffer_samples = current_buffer[ch]
                echo_samples = buffer_samples * echo_decay
                
                # Mix echo with current samples
                mix_length = min(len(channel_samples), len(echo_samples))
                channel_samples[:mix_length] += echo_samples[:mix_length]
                
                # Update buffer with current samples for next frame
                if len(channel_samples) >= len(buffer_samples):
                    # Current samples are longer than buffer
                    current_buffer[ch] = channel_samples[-len(buffer_samples):]
                else:
                    # Shift buffer and add new samples
                    shift_amount = len(channel_samples)
                    current_buffer[ch] = np.roll(current_buffer[ch], -shift_amount)
                    current_buffer[ch][-shift_amount:] = channel_samples
            
            # Apply low-pass filter
            if lowpass_cutoff < sample_rate / 2:
                # Design Butterworth low-pass filter
                nyquist = sample_rate / 2
                normalized_cutoff = lowpass_cutoff / nyquist
                b, a = signal.butter(4, normalized_cutoff, btype='low')
                
                # Apply filter
                channel_samples = signal.filtfilt(b, a, channel_samples)
            
            # Clip to prevent overflow
            channel_samples = np.clip(channel_samples, -1.0, 1.0)
            
            processed_samples[ch] = channel_samples
        
        # Convert back to original format
        if frame.format in ['s16', 's16p']:
            processed_samples = (processed_samples * 32767).astype(np.int16)
        elif frame.format in ['s32', 's32p']:
            processed_samples = (processed_samples * 2147483647).astype(np.int32)
        # float formats stay as float32
        
        # Convert back to original layout
        if frame.format.endswith('p'):
            # Keep planar format: (channels, samples)
            final_samples = processed_samples
        else:
            # Convert back to packed format: (samples, channels)
            if channels == 1:
                final_samples = processed_samples.squeeze(0)  # Remove channel dimension for mono
            else:
                final_samples = processed_samples.T
        
        # Create new AudioFrame with modified samples
        # We'll create it manually since there's no replace_samples method
        new_frame = AudioFrame.__new__(AudioFrame)
        new_frame.samples = final_samples
        new_frame.nb_samples = frame.nb_samples
        new_frame.format = frame.format
        new_frame.rate = frame.rate
        new_frame.layout = frame.layout
        new_frame.timestamp = frame.timestamp
        new_frame.time_base = frame.time_base
        new_frame.log_timestamps = frame.log_timestamps.copy()
        new_frame.side_data = frame.side_data
        
        logger.debug(f"🎵 Processed audio: {channels} channels, {len(final_samples)} samples, "
                    f"volume={volume:.2f}, echo={echo_decay:.2f}, lpf={lowpass_cutoff}Hz")
        
        return [new_frame]
        
    except Exception as e:
        logger.error(f"Error in audio processing: {e}")
        # Return original frame on error
        return [frame]

def update_params(params: dict):
    """Update audio effect parameters."""
    global volume, echo_delay, echo_decay, lowpass_cutoff, enable_effects, delay, echo_buffer
    
    if "volume" in params:
        old = volume
        volume = max(0.0, min(2.0, float(params["volume"])))
        if old != volume:
            logger.info(f"📢 Volume: {old:.2f} → {volume:.2f}")
    
    if "echo_delay" in params:
        old = echo_delay
        echo_delay = max(0.0, min(1.0, float(params["echo_delay"])))
        if old != echo_delay:
            logger.info(f"⏱️ Echo delay: {old:.2f}s → {echo_delay:.2f}s")
            # Clear echo buffers when delay changes
            echo_buffer.clear()
    
    if "echo_decay" in params:
        old = echo_decay
        echo_decay = max(0.0, min(1.0, float(params["echo_decay"])))
        if old != echo_decay:
            logger.info(f"🔉 Echo decay: {old:.2f} → {echo_decay:.2f}")
    
    if "lowpass_cutoff" in params:
        old = lowpass_cutoff
        lowpass_cutoff = max(100, min(20000, int(params["lowpass_cutoff"])))
        if old != lowpass_cutoff:
            logger.info(f"🎛️ Low-pass cutoff: {old} Hz → {lowpass_cutoff} Hz")
    
    if "enable_effects" in params:
        old = enable_effects
        enable_effects = bool(params["enable_effects"])
        if old != enable_effects:
            logger.info(f"⚡ Effects: {'ON' if enable_effects else 'OFF'}")
    
    if "delay" in params:
        old = delay
        delay = max(0.0, float(params["delay"]))
        if old != delay:
            logger.info(f"⏳ Processing delay: {old:.2f}s → {delay:.2f}s")
    
    if "clear_echo_buffer" in params and params["clear_echo_buffer"]:
        echo_buffer.clear()
        logger.info("🧹 Echo buffer cleared")

# Create and run StreamProcessor
if __name__ == "__main__":
    processor = StreamProcessor(
        audio_processor=process_audio,
        model_loader=load_model,
        param_updater=update_params,
        name="audio-effects-processor",
        port=8000
    )
    
    logger.info("🚀 Starting audio effects processor...")
    logger.info("🎵 Available effects: volume, echo, low-pass filter")
    logger.info("🔧 Update parameters via /api/update_params:")
    logger.info("   - volume: 0.0 to 2.0 (1.0 = normal)")
    logger.info("   - echo_delay: 0.0 to 1.0 seconds")
    logger.info("   - echo_decay: 0.0 to 1.0 (echo strength)")
    logger.info("   - lowpass_cutoff: 100 to 20000 Hz")
    logger.info("   - enable_effects: true/false")
    logger.info("   - delay: processing delay in seconds")
    logger.info("   - clear_echo_buffer: true to reset echo buffer")
    
    processor.run()
