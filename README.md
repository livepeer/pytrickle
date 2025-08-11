# PyTrickle

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
- 🎛️ **Multi-Channel Audio**: Handles mono, stereo, and multi-channel audio automatically

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- FFmpeg (for video encoding/decoding)
- http-trickle Go package (for testing)

### Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Install http-trickle (for testing)

```bash
# Clone the http-trickle repository
git clone https://github.com/livepeer/http-trickle.git ~/repos/http-trickle
cd ~/repos/http-trickle
make build
```

## Quick Start

### Simple Stream Processing

```python
import asyncio
import torch
from pytrickle import TrickleClient
from pytrickle.frames import VideoFrame, VideoOutput

def red_tint_processor(frame: VideoFrame) -> VideoOutput:
    """Add a red tint to video frames."""
    tensor = frame.tensor.clone()
    tensor[:, :, :, 0] = torch.clamp(tensor[:, :, :, 0] + 0.3, 0, 1)
    new_frame = frame.replace_tensor(tensor)
    return VideoOutput(new_frame, "red_tint")

async def main():
    client = TrickleClient(
        subscribe_url="http://localhost:3389/sample",
        publish_url="http://localhost:3389/sample-output",
        width=704,
        height=384,
        frame_processor=red_tint_processor
    )
    
    await client.start("simple_example")

asyncio.run(main())
```

### HTTP Server Integration (TrickleApp)

The simplest way to integrate PyTrickle into your application is using `TrickleApp`:

```python
from pytrickle import create_app

# Create app with custom frame processor
app = create_app(frame_processor=your_processor_function)

# Run the server
await app.run_forever()
```

## API Reference

### HTTP Endpoints

#### Start Stream
```bash
POST /api/stream/start
POST /live-video-to-video  # Alias
```

**Request:**
```json
{
    "subscribe_url": "http://localhost:3389/sample",
    "publish_url": "http://localhost:3389/sample-output",
    "control_url": "http://localhost:3389/control",     // Optional
    "events_url": "http://localhost:3389/events",       // Optional
    "data_url": "http://localhost:3389/data",           // Optional
    "gateway_request_id": "stream_001",
    "params": {                                         // Optional
        "width": 1920,
        "height": 1080,
        "effect": "enhancement",
        "quality": "high"
    }
}
```

**Parameters:**
- `subscribe_url` (required): Source URL for incoming video stream
- `publish_url` (required): Destination URL for processed video stream  
- `control_url` (optional): URL for receiving control messages
- `events_url` (optional): URL for publishing monitoring events
- `data_url` (optional): URL for publishing text/data output
- `gateway_request_id` (required): Unique identifier for the stream request
- `params` (optional): Processing parameters (any key-value pairs)

#### Update Parameters
```bash
POST /api/stream/params
```

**Request:**
```json
{
    "params": {
        "effect": "blue_tint",
        "intensity": 0.5,
        "width": 1024,
        "height": 768
    }
}
```

#### Get Status
```bash
GET /api/stream/status
```

#### Stop Stream
```bash
POST /api/stream/stop
```

#### Health Check
```bash
GET /health
```

### Python API

#### TrickleClient (Direct Integration)

```python
from pytrickle import TrickleClient

client = TrickleClient(
    subscribe_url="http://localhost:3389/input",
    publish_url="http://localhost:3389/output",
    control_url="http://localhost:3389/control",  # Optional
    events_url="http://localhost:3389/events",    # Optional
    width=1920,
    height=1080,
    frame_processor=your_processor
)

await client.start("request_id")
```

#### TrickleApp (HTTP Server)

```python
from pytrickle import TrickleApp

app = TrickleApp(frame_processor=your_processor, port=8080)
await app.run_forever()
```

## Frame Processing

### Custom Frame Processors

Create custom frame processors by implementing a function that takes a `VideoFrame` and returns a `VideoOutput`:

```python
def custom_processor(frame: VideoFrame) -> VideoOutput:
    # Get the tensor (shape: [1, height, width, 3])
    tensor = frame.tensor.clone()
    
    # Apply your processing
    processed_tensor = your_model(tensor)
    
    # Create new frame with processed tensor
    new_frame = frame.replace_tensor(processed_tensor)
    return VideoOutput(new_frame, "custom_processor")
```

### GPU Processing

```python
def gpu_processor(frame: VideoFrame) -> VideoOutput:
    tensor = frame.tensor
    
    # Move to GPU if not already there
    if not tensor.is_cuda and torch.cuda.is_available():
        tensor = tensor.cuda()
    
    # GPU processing
    processed = your_gpu_model(tensor)
    
    new_frame = frame.replace_tensor(processed)
    return VideoOutput(new_frame, "gpu_processor")
```

### Audio Processing

PyTrickle automatically handles different audio channel configurations:

```python
def audio_processor(frame: AudioFrame) -> AudioOutput:
    # Audio tensor is automatically converted to the right format
    # Supports mono, stereo, and multi-channel audio
    tensor = frame.tensor.clone()
    
    # Apply audio processing
    processed = apply_audio_filter(tensor)
    
    new_frame = frame.replace_tensor(processed)
    return AudioOutput(new_frame, "audio_processed")
```

## Testing

### Quick Test with Make

```bash
# Install and run basic tests
make install
make test

# Run integration tests (requires http-trickle)
make test-integration

# Start example server
make run-server

# Start simple processing example
make run-example
```

### Manual Testing with http-trickle

1. **Start the trickle server**:
```bash
cd ~/repos/http-trickle && make trickle-server addr=0.0.0.0:3389
```

2. **Start the pytrickle server**:
```bash
make run-server
```

3. **Start a video stream**:
```bash
cd ~/repos/http-trickle && make publisher-ffmpeg in=bbb_sunflower_1080p_30fps_normal.mp4 stream=sample url=http://127.0.0.1:3389
```

4. **Start processing via API**:
```bash
curl -X POST http://localhost:8080/api/stream/start \
  -H "Content-Type: application/json" \
  -d '{
    "subscribe_url": "http://127.0.0.1:3389/sample",
    "publish_url": "http://127.0.0.1:3389/sample-output",
    "gateway_request_id": "test",
    "params": {
      "width": 704,
      "height": 384,
      "effect": "red_tint",
      "intensity": 0.3
    }
  }'
```

5. **Play the processed stream**:
```bash
cd ~/repos/http-trickle && go run cmd/read2pipe/*.go --url http://127.0.0.1:3389/ --stream sample-output | ffplay -probesize 64 -
```

## Performance Optimization

### Throughput Optimization

- Use GPU processing when available
- Minimize tensor copying with `.clone()` only when necessary
- Process frames in batches when possible
- Use efficient PyTorch operations

### Memory Management

- The package automatically handles CUDA memory management
- Tensors are automatically moved between CPU/GPU as needed
- Frame timestamps and metadata are preserved during processing

### Monitoring Performance

The package includes built-in performance monitoring:

- Frame processing timestamps
- Input/output FPS tracking
- Memory usage monitoring
- Error rate tracking

## Architecture

### Components

- **TrickleClient**: High-level client interface for direct integration
- **TrickleApp**: HTTP server application for API-based integration
- **TrickleStreamHandler**: Complete stream handler with built-in protocol management (NEW)
- **TrickleSubscriber**: Handles subscription to trickle streams
- **TricklePublisher**: Handles publishing to trickle streams  
- **TrickleProtocol**: High-level protocol implementation
- **Decoder**: Video decoding with PyAV
- **Encoder**: Video encoding with PyAV
- **StreamHealthManager**: Health monitoring and management
- **BaseStreamManager/TrickleStreamManager**: Stream lifecycle management

### Data Flow

```
Input Stream → Subscriber → Decoder → Frame Processor → Encoder → Publisher → Output Stream
                                            ↓
                                    Events/Monitoring
```

The package uses asyncio for concurrent processing and PyTorch tensors for efficient frame manipulation.

## Troubleshooting

### Common Issues

**Error: "http-trickle not found"**
- Ensure http-trickle is cloned to `~/repos/http-trickle`
- Run `make build` in the http-trickle directory

**Error: "CUDA out of memory"**
- Reduce frame dimensions
- Process frames on CPU instead of GPU
- Ensure proper tensor cleanup

**Error: "Connection refused"**
- Check that the trickle server is running
- Verify the correct host and port
- Check firewall settings

**Error: "cannot reshape array of size X into shape (2)"**
- This audio error is now automatically handled
- The package detects and properly processes mono, stereo, and multi-channel audio

### Performance Issues

**Low FPS**
- Use GPU processing if available
- Optimize frame processor algorithms
- Check network bandwidth
- Monitor CPU/memory usage

**High Latency**
- Reduce processing complexity
- Use smaller frame dimensions
- Check network latency
- Optimize encoder settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[MIT License](LICENSE)

## Recent Updates

### Version 0.1.1
- **Unified API Parameters**: StreamStartRequest now uses `params` field directly (matching update requests)
- **Enhanced Audio Support**: Automatic handling of mono, stereo, and multi-channel audio configurations
- **Improved Integration**: Better support for different integration patterns
- **Runtime Fixes**: Resolved async/sync compatibility issues in metadata caching
- **Robust Error Handling**: Enhanced encoder metadata validation and error recovery

## Acknowledgments

- Built on top of the http-trickle protocol
- Uses PyAV for video encoding/decoding
- Leverages PyTorch for tensor operations
- Based on architecture from ai-runner