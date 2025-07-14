# Trickle App

A high-performance Python package for real-time video streaming and processing over the trickle protocol. Built for maximum throughput, reliability, and ease of use.

## Overview

Trickle App provides a complete solution for subscribing to and publishing video streams using the trickle protocol. It includes:

- **High-performance streaming**: Optimized for maximum throughput and low latency
- **Real-time video processing**: Process video frames in real-time with custom algorithms
- **HTTP API**: REST API for stream management and parameter updates
- **Reliability**: Built-in retry mechanisms and error handling
- **Flexibility**: Support for custom frame processors and effects

## Features

- 🚀 **High Performance**: Optimized for maximum throughput with asyncio and efficient tensor operations
- 📹 **Video Processing**: Real-time frame processing with PyTorch tensors
- 🔄 **Stream Management**: Start, stop, and monitor streams via HTTP API
- ⚙️ **Dynamic Parameters**: Update processing parameters in real-time
- 🔧 **Extensible**: Easy to add custom frame processing algorithms
- 📊 **Monitoring**: Built-in monitoring and event reporting
- 🛡️ **Reliable**: Automatic reconnection and error recovery

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- FFmpeg (for video encoding/decoding)
- http-trickle Go package (for testing)

### Install Dependencies

```bash
pip install -r requirements.txt
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
from pytrickle import SimpleTrickleClient
from pytrickle.frames import VideoFrame, VideoOutput

def red_tint_processor(frame: VideoFrame) -> VideoOutput:
    """Add a red tint to video frames."""
    tensor = frame.tensor.clone()
    tensor[:, :, :, 0] = torch.clamp(tensor[:, :, :, 0] + 0.3, 0, 1)
    new_frame = frame.replace_tensor(tensor)
    return VideoOutput(new_frame, "red_tint")

async def main():
    client = SimpleTrickleClient(
        subscribe_url="http://localhost:3389/sample",
        publish_url="http://localhost:3389/sample-output"
    )
    
    await client.process_stream(
         frame_processor=red_tint_processor,
         request_id="simple_example",
         width=704,
         height=384
     )

asyncio.run(main())
```

### HTTP Server

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
```

Example request (minimal):
```json
{
    "subscribe_url": "http://localhost:3389/sample",
    "publish_url": "http://localhost:3389/sample-output",
    "gateway_request_id": "example",
    "params": {
        "width": 704,
        "height": 384,
        "effect": "red_tint",
        "intensity": 0.3
    }
}
```

Example request (with optional control and events URLs):
```json
{
    "subscribe_url": "http://localhost:3389/sample",
    "publish_url": "http://localhost:3389/sample-output",
    "control_url": "http://localhost:3389/control",
    "events_url": "http://localhost:3389/events",
    "gateway_request_id": "example",
    "params": {
        "width": 704,
        "height": 384,
        "effect": "red_tint",
        "intensity": 0.3
    }
}
```

**Parameters:**
- `subscribe_url` (required): Source URL for incoming video stream
- `publish_url` (required): Destination URL for processed video stream  
- `control_url` (optional): URL for receiving control messages
- `events_url` (optional): URL for publishing monitoring events
- `gateway_request_id` (optional): Identifier for the request
- `params` (optional): Processing parameters

#### Update Parameters
```bash
POST /api/stream/params
```

Example request:
```json
{
    "effect": "blue_tint",
    "intensity": 0.5
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

#### TrickleClient

```python
from pytrickle import TrickleClient

# Minimal setup
client = TrickleClient(
    subscribe_url="http://localhost:3389/input",
    publish_url="http://localhost:3389/output",
    width=704,
    height=384,
    frame_processor=your_processor
)

# With optional control and events
client_with_control = TrickleClient(
    subscribe_url="http://localhost:3389/input",
    publish_url="http://localhost:3389/output",
    control_url="http://localhost:3389/control",  # Optional
    events_url="http://localhost:3389/events",    # Optional
    width=704,
    height=384,
    frame_processor=your_processor
)

await client.start("request_id")
```

#### SimpleTrickleClient

```python
from pytrickle import SimpleTrickleClient

client = SimpleTrickleClient(
    subscribe_url="http://localhost:3389/input",
    publish_url="http://localhost:3389/output"
)

 await client.process_stream(
     frame_processor=your_processor,
     request_id="example",
     width=704,
     height=384
 )
```

#### TrickleApp

```python
from pytrickle import TrickleApp

app = TrickleApp(frame_processor=your_processor, port=8080)
await app.run_forever()
```

## Testing

### Running Tests

```bash
# Run basic tests
pytest tests/test_basic_streaming.py -v

# Run integration tests (requires http-trickle)
pytest tests/test_integration.py -v -m integration
```

### Manual Testing with http-trickle

1. **Start the trickle server**:
```bash
cd ~/repos/http-trickle && make trickle-server addr=0.0.0.0:3389
```

2. **Start the pytrickle server**:
```bash
python examples/http_server_example.py
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

### Automated Test Workflow

Run the complete test workflow:

```bash
python scripts/test_workflow.py
```

This script will automatically:
- Start the trickle server
- Start the pytrickle server  
- Start a video stream
- Process the stream with various effects
- Monitor the stream status
- Clean up all processes

## Frame Processing

### Built-in Effects

The example server includes several built-in effects:

- `red_tint`: Add red tint to frames
- `blue_tint`: Add blue tint to frames  
- `green_tint`: Add green tint to frames
- `grayscale`: Convert to grayscale
- `invert`: Invert colors
- `brightness`: Adjust brightness
- `color_shift`: Apply RGB color shifts

### Custom Frame Processors

Create custom frame processors by implementing a function that takes a `VideoFrame` and returns a `VideoOutput`:

```python
def custom_processor(frame: VideoFrame) -> VideoOutput:
    # Get the tensor (shape: [1, height, width, 3])
    tensor = frame.tensor.clone()
    
    # Apply your processing
    # ... your custom logic here ...
    
    # Create new frame with processed tensor
    new_frame = frame.replace_tensor(processed_tensor)
    return VideoOutput(new_frame, "custom_processor")
```

### GPU Processing

The package supports CUDA tensors for GPU acceleration:

```python
def gpu_processor(frame: VideoFrame) -> VideoOutput:
    tensor = frame.tensor
    
    # Move to GPU if not already there
    if not tensor.is_cuda and torch.cuda.is_available():
        tensor = tensor.cuda()
    
    # GPU processing
    processed = tensor * 1.2  # Example operation
    
    new_frame = frame.replace_tensor(processed)
    return VideoOutput(new_frame, "gpu_processor")
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

- **TrickleSubscriber**: Handles subscription to trickle streams
- **TricklePublisher**: Handles publishing to trickle streams  
- **TrickleProtocol**: High-level protocol implementation
- **Decoder**: Video decoding with PyAV
- **Encoder**: Video encoding with PyAV
- **TrickleClient**: High-level client interface
- **TrickleApp**: HTTP server application

### Data Flow

```
Input Stream → Subscriber → Decoder → Frame Processor → Encoder → Publisher → Output Stream
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

## Recent Fixes

### Runtime Error Fixes (v1.1)
- **Fixed `LastValueCache` async/sync compatibility**: Converted from async locks to thread locks to work properly in mixed sync/async environments
- **Enhanced encoder metadata validation**: Added robust error handling for malformed metadata to prevent `TypeError: 'coroutine' object is not subscriptable`
- **Fixed encoder metadata timing**: Encoder now waits for metadata from decoder instead of immediately exiting
- **Fixed output publishing**: TrickleClient now properly queues and publishes processed frames
- **Improved thread safety**: All components now work correctly in multi-threaded environments
- **Made control_url and events_url fully optional**: Can be `None` or omitted entirely for basic streaming functionality

### Error Fixes Addressed:
- `TypeError: 'coroutine' object is not subscriptable` in encoder
- `RuntimeWarning: coroutine 'LastValueCache.get' was never awaited`
- `RuntimeWarning: coroutine 'LastValueCache.put' was never awaited`
- Thread safety issues in metadata cache
- Output publishing not working (frames not reaching encoder)
- Race condition between encoder and decoder startup

## Acknowledgments

- Built on top of the http-trickle protocol
- Uses PyAV for video encoding/decoding
- Leverages PyTorch for tensor operations
- Based on architecture from ai-runner 