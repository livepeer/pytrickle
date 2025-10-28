
# PyTrickle [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/livepeer/pytrickle)


A high-performance Python package for real-time video streaming and processing over the trickle protocol. Built for maximum throughput, reliability, and ease of integration into video processing applications.

## Overview

PyTrickle provides a complete Python framework for real-time video and audio streaming with custom processing. Built on the trickle protocol, it enables you to:

- **Process live streams in real-time** with your custom Python functions
- **Build HTTP streaming services** with REST APIs for remote control
- **Handle both video and audio** with automatic format detection and conversion
- **Scale from simple filters to complex AI pipelines** with async processing support
- **Integrate easily** into existing Python applications with minimal code

Perfect for building AI-powered video processing services, real-time filters, streaming analytics, and more.

## Features

- 🚀 **High Performance**: Optimized for maximum throughput with asyncio and efficient tensor operations
- 📹 **Video Processing**: Real-time frame processing with PyTorch tensors
- 🔄 **Stream Management**: Start, stop, and monitor streams via HTTP API
- ⚙️ **Dynamic Parameters**: Update processing parameters in real-time
- 🔧 **Extensible**: Easy to add custom frame processing algorithms
- 📊 **Monitoring**: Built-in monitoring and event reporting
- 🛡️ **Reliable**: Automatic reconnection and error recovery
- 🎵 **Audio Support**: Handles mono, stereo, and multi-channel audio

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- FFmpeg (for video encoding/decoding)

### Install PyTrickle

```bash
pip install -r requirements.txt
pip install -e .
```

### Install http-trickle (for testing)

```bash
git clone https://github.com/livepeer/http-trickle.git ~/repos/http-trickle
cd ~/repos/http-trickle
make build
```

## Quick Start

PyTrickle is designed for simplicity. You can create a powerful real-time video processor by providing a few async functions to the `StreamProcessor`. This handles the server, state management, and model loading for you.

See a complete, runnable example in `examples/model_loading_example.py`.

### Simple Video Processor

Here's how to create a basic video processor that applies a green tint to video frames:

```python
import asyncio
import logging
from pytrickle import StreamProcessor
from pytrickle.frames import VideoFrame

logging.basicConfig(level=logging.INFO)

# 1. Define your model loading logic
async def load_my_model(**kwargs):
    """Simulate loading a model. Runs in the background without blocking the server."""
    logging.info("🔄 Model loading started...")
    await asyncio.sleep(5)  # Simulate a 5-second model load time
    logging.info("✅ Model loaded successfully!")

# 2. Define your video processing function
async def apply_green_tint(frame: VideoFrame) -> VideoFrame:
    """Apply a simple green tint effect to the video."""
    tensor = frame.tensor.clone()
    tensor[1] = tensor[1] * 2.0  # Increase green channel
    return frame.replace_tensor(tensor)

# 3. Define a function to handle real-time parameter updates
async def handle_parameter_updates(params: dict):
    """Handle incoming parameters to adjust processing in real-time."""
    logging.info(f"Parameters updated: {params}")

# 4. Create and run the StreamProcessor
if __name__ == "__main__":
    processor = StreamProcessor(
        video_processor=apply_green_tint,
        model_loader=load_my_model,
        param_updater=handle_parameter_updates,
        port=8000,
        name="green-tint-processor"
    )
    processor.run()
```

## Model Loading

PyTrickle features a non-blocking model loading mechanism that allows the server to start immediately and be available for health checks while your model loads in the background.

- **Non-Blocking Startup**: The server starts instantly and responds to `/health` requests with a `LOADING` status.
- **Automatic Loading**: Your `model_loader` function is triggered automatically after startup.
- **State Transition**: Once your `model_loader` completes, the server's health status transitions from `LOADING` to `IDLE`, indicating it's ready to process streams.

This robust pattern is ideal for managed environments (like Docker or Kubernetes) where immediate health checks are critical for service orchestration. It is the recommended way to handle model initialization.

## HTTP API

PyTrickle automatically provides a REST API for your video processor:

### Start Processing

```bash
curl -X POST http://localhost:8000/api/stream/start \
  -H "Content-Type: application/json" \
  -d '{
    "subscribe_url": "http://localhost:3389/input",
    "publish_url": "http://localhost:3389/output",
    "gateway_request_id": "demo_stream",
    "params": {
      "width": 704,
      "height": 384,
      "intensity": 0.7
    }
  }'
```

### Update Parameters

```bash
curl -X POST http://localhost:8000/api/stream/params \
  -H "Content-Type: application/json" \
  -d '{
    "intensity": 0.9,
    "effect": "enhanced"
  }'
```

### Check Status

```bash
curl http://localhost:8000/api/stream/status
```

### Stop Processing

```bash
curl -X POST http://localhost:8000/api/stream/stop
```

## Advanced Usage

### GPU Processing

```python
class GPUProcessor(FrameProcessor):
    """GPU-accelerated video processor."""
    
    async def process_video_async(self, frame: VideoFrame) -> Optional[VideoFrame]:
        tensor = frame.tensor
        
        # Move to GPU if available
        if torch.cuda.is_available() and not tensor.is_cuda:
            tensor = tensor.cuda()
        
        # Apply GPU processing
        processed = await self.gpu_model(tensor)
        
        return frame.replace_tensor(processed)
```

### Direct Client Integration

For applications that need direct control without HTTP, see the TrickleClient documentation and `examples/async_processor_example.py` for advanced usage patterns.

## Testing

### Quick Test

```bash
# Install and test
make install
make test

# Run the example processor
python examples/async_processor_example.py
```

### Full Integration Test

1. **Start trickle server**:
```bash
cd ~/repos/http-trickle && make trickle-server addr=0.0.0.0:3389
```

2. **Start the example processor**:
```bash
python examples/async_processor_example.py
```

3. **Start video stream**:
```bash
cd ~/repos/http-trickle && make publisher-ffmpeg in=video.mp4 stream=input url=http://127.0.0.1:3389
```

4. **Begin processing**:
```bash
curl -X POST http://localhost:8000/api/stream/start \
  -H "Content-Type: application/json" \
  -d '{
    "subscribe_url": "http://127.0.0.1:3389/input",
    "publish_url": "http://127.0.0.1:3389/output",
    "gateway_request_id": "test",
    "params": {"intensity": 0.7}
  }'
```

5. **Update parameters in real-time**:
```bash
curl -X POST http://localhost:8000/api/stream/params \
  -H "Content-Type: application/json" \
  -d '{"intensity": 0.9}'
```

6. **View processed stream**:
```bash
cd ~/repos/http-trickle && go run cmd/read2pipe/*.go --url http://127.0.0.1:3389/ --stream output | ffplay -
```

## Performance Tips

### Optimization

- Use GPU processing when available
- Minimize tensor copying with efficient PyTorch operations
- Process frames in batches for AI models
- Use async/await for I/O operations

### Memory Management

- PyTrickle automatically handles CUDA memory
- Tensors are moved between CPU/GPU as needed
- Frame metadata is preserved during processing

### Monitoring

Built-in performance tracking includes:
- Frame processing times
- Input/output FPS
- Memory usage
- Error rates

### Frame Rate Configuration

PyTrickle allows you to control the maximum frame rate for video processing:

**Set framerate when starting a stream:**
```bash
curl -X POST http://localhost:8000/api/stream/start \
  -H "Content-Type: application/json" \
  -d '{
    "subscribe_url": "http://127.0.0.1:3389/",
    "publish_url": "http://127.0.0.1:3389/",
    "gateway_request_id": "test",
    "params": {
      "width": 512,
      "height": 512,
      "max_framerate": 30
    }
  }'
```



**Framerate options:**
- **Default**: 24 FPS (balanced performance)
- **Low**: 15 FPS (reduced CPU usage)
- **Standard**: 30 FPS (smooth video)
- **High**: 60 FPS (ultra-smooth, higher resource usage)
- **Custom**: Any positive integer value from 1 to 60 FPS
- **Maximum**: 60 FPS (values above 60 will be rejected)

The framerate setting controls the maximum number of frames processed per second, helping balance performance and resource usage.

## Architecture

PyTrickle consists of several key components:

- **StreamServer**: HTTP server for API-based integration
- **FrameProcessor**: Base class for async AI processors
- **TrickleClient**: Direct client for custom applications
- **TrickleProtocol**: High-level protocol implementation

### Data Flow

```
Input Stream → Decoder → Frame Processor → Encoder → Output Stream
                               ↓
                       Parameter Updates & Monitoring
```

## Examples

The `examples/` directory contains:

- `async_processor_example.py`: Complete FrameProcessor with green tint processing and real-time parameter updates

## Troubleshooting

### Common Issues

**CUDA out of memory**
- Use smaller frame dimensions
- Process on CPU instead of GPU

**Connection refused**
- Ensure trickle server is running on correct port
- Check firewall settings

**Low performance**
- Use GPU processing when available
- Optimize your processing algorithms
- Check network bandwidth

### Audio Issues

PyTrickle automatically handles different audio formats. If you encounter audio-related errors, the SDK will automatically detect and convert between mono, stereo, and multi-channel configurations.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[MIT License](LICENSE)

---

**Get started with PyTrickle today and build powerful real-time video processing applications!**