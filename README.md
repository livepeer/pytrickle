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
import asyncio
import torch
from typing import Union
from pytrickle import TrickleApp
from pytrickle.frames import VideoFrame, AudioFrame, VideoOutput, AudioOutput

def my_frame_processor(frame: Union[VideoFrame, AudioFrame]) -> Union[VideoOutput, AudioOutput]:
    """Your custom video and audio processing logic."""
    
    if isinstance(frame, VideoFrame):
        # Process video frames
        tensor = frame.tensor.clone()
        
        # Example: Add a blue tint
        tensor[:, :, :, 2] = torch.clamp(tensor[:, :, :, 2] + 0.3, 0, 1)
        
        new_frame = frame.replace_tensor(tensor)
        return VideoOutput(new_frame, "blue_tint_processor")
    
    elif isinstance(frame, AudioFrame):
        # Process audio frames (automatically handles mono/stereo/multi-channel)
        tensor = frame.tensor.clone()
        
        # Example: Apply simple gain
        tensor = tensor * 0.8  # Reduce volume by 20%
        
        new_frame = frame.replace_tensor(tensor)
        return AudioOutput([new_frame], "audio_gain_processor")

async def main():
    # Create and run the TrickleApp
    app = TrickleApp(
        frame_processor=my_frame_processor,
        port=8080,
        host="0.0.0.0"
    )
    
    # This starts the HTTP server and runs forever
    await app.run_forever()

if __name__ == "__main__":
    asyncio.run(main())
```

### Async AI Integration

For AI models and async processing, use the `AsyncFrameProcessor` to seamlessly integrate async operations:

```python
import asyncio
from pytrickle import TrickleApp, AsyncFrameProcessor
from pytrickle.frames import VideoFrame, AudioFrame

class MyAIProcessor(AsyncFrameProcessor):
    async def process_video_async(self, frame: VideoFrame):
        # Your async AI processing here
        result = await my_ai_model(frame.tensor)
        return frame.replace_tensor(result)
    
    async def process_audio_async(self, frame: AudioFrame):
        return [frame]  # Pass through audio

async def main():
    # Create and start async processor
    processor = MyAIProcessor()
    await processor.start()
    
    # Create TrickleApp with async processor
    app = TrickleApp(
        frame_processor=processor.create_sync_bridge(),
        port=8080
    )
    
    await app.run_forever()

asyncio.run(main())
```

For quick testing and demos, use the built-in `SimpleAsyncProcessor`:

```python
from pytrickle import TrickleApp, SimpleAsyncProcessor

async def main():
    # Create simple processor with video effects
    processor = SimpleAsyncProcessor(
        effect="red_tint",  # or blue_tint, brightness, sepia, etc.
        processing_time=0.02
    )
    await processor.start()
    
    # Clean, organized usage
    app = TrickleApp(frame_processor=processor.create_sync_bridge())
    await app.run_forever()

asyncio.run(main())
```

That's it! Your server will now be running on port 8080 with these endpoints:
- `POST /api/stream/start` - Start processing a video stream
- `POST /api/stream/params` - Update processing parameters in real-time
- `GET /api/stream/status` - Get current stream status
- `POST /api/stream/stop` - Stop the current stream

**📚 For comprehensive integration examples, see:**
- [**PyTrickle Integration Guide**](examples/INTEGRATION_GUIDE.md) - Complete guide with three integration methods and examples
- [Method 1: Simple TrickleApp](examples/method1_simple_trickle_app.py) - Minimal setup for AI applications
- [Method 2: Custom Server](examples/method2_custom_server.py) - Production-ready custom server integration  
- [Method 3: Direct Publishers/Subscribers](examples/method3_direct_trickle.py) - Direct protocol control for specialized use cases

## Integration Approaches

PyTrickle provides three distinct integration patterns, each optimized for different use cases and complexity levels. **See the [Integration Guide](examples/INTEGRATION_GUIDE.md) for detailed examples and implementation patterns.**

### 1. TrickleApp - Simple HTTP Server Integration

**Best for:** Simple AI applications, prototypes, microservices

The simplest way to build your own AI application with a custom frame processor. TrickleApp provides a complete HTTP server with minimal setup.

**Key Features:**
- ✅ Complete HTTP server with REST API endpoints
- ✅ Built-in stream management and lifecycle handling
- ✅ Automatic error handling and recovery
- ✅ Real-time parameter updates via API
- ✅ Minimal code required (just implement frame processor)

**Example:** [method1_simple_trickle_app.py](examples/method1_simple_trickle_app.py)

```python
import asyncio
import torch
from pytrickle import TrickleApp, RegisterCapability
from pytrickle.frames import VideoFrame, AudioFrame, VideoOutput, AudioOutput

def my_frame_processor(frame):
    """Your custom frame processing logic."""
    if isinstance(frame, VideoFrame):
        # Process video frames - example: flip upside down
        tensor = frame.tensor.clone()
        flipped = tensor.flip(dims=[1])
        return VideoOutput(frame.replace_tensor(flipped), "upside_down")
    else:
        # Pass through audio unchanged
        return AudioOutput([frame], "audio_passthrough")

async def main():
                # Optionally register as a worker with orchestrator
            registry = RegisterCapability(logger)
            registry.register_capability()
    
    # Create and run the TrickleApp
    app = TrickleApp(
        frame_processor=my_frame_processor,
        port=8080
    )
    await app.run_forever()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Custom Server Integration - Full Control

**Best for:** Production applications, complex pipelines, custom protocols

Integrate PyTrickle into your existing server framework (aiohttp, FastAPI, etc.) for full control over the application architecture. This is how ComfyStream integrates PyTrickle.

**Key Features:**
- ✅ Integrate with existing server applications
- ✅ Custom stream managers and handlers
- ✅ Advanced async processing patterns
- ✅ Multiple concurrent streams
- ✅ Custom health monitoring and metrics
- ✅ Text output channels for AI workflows

**Example:** [method2_custom_server.py](examples/method2_custom_server.py)

```python
from pytrickle.manager import TrickleStreamManager, StreamHandler
from pytrickle import RegisterCapability

# Custom stream handler with your application logic
class MyStreamHandler(StreamHandler):
    def __init__(self, pipeline, request_id, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = pipeline
        self.request_id = request_id
        self.processor = MyAsyncProcessor(pipeline, request_id)
    
    async def start(self):
        await self.processor.start_processing()
        return await super().start()
    
    async def stop(self):
        await self.processor.stop_processing()
        return await super().stop()

# Custom stream manager
class MyStreamManager(TrickleStreamManager):
    def __init__(self, app_context):
        super().__init__(app_context=app_context)
        
        # Set up stream handler factory
        async def factory(request_id, **kwargs):
            return MyStreamHandler(
                pipeline=kwargs['pipeline'],
                request_id=request_id,
                **kwargs
            )
        self.set_stream_handler_factory(factory)

# Integrate with your existing aiohttp/FastAPI server
async def setup_app(app):
                # Optionally register capability
            registry = RegisterCapability(logger)
            await registry.register_capability_async()
    
    # Setup stream manager
    stream_manager = MyStreamManager(app_context={'pipeline': my_pipeline})
    app['stream_manager'] = stream_manager
    
    # Setup API routes for stream control
    app.router.add_post("/stream/start", start_stream_handler)
    app.router.add_post("/stream/stop", stop_stream_handler)
```

### 3. Direct Publisher/Subscriber Integration

**Best for:** Specialized pipelines, batch processing, research applications

Use TricklePublisher and TrickleSubscriber directly for maximum flexibility. This approach gives you direct control over the trickle protocol without stream management overhead.

**Key Features:**
- ✅ Direct control over trickle protocol
- ✅ Custom processing pipelines
- ✅ Batch processing workflows
- ✅ Research and experimentation
- ✅ Specialized use cases (audio-to-text, transcription)

**Example:** [method3_direct_trickle.py](examples/method3_direct_trickle.py)

```python
import asyncio
from pytrickle import TrickleSubscriber, TricklePublisher

class MyVideoPipeline:
    def __init__(self, config):
        self.config = config
        self.input_queue = asyncio.Queue(maxsize=10)
        self.output_queue = asyncio.Queue(maxsize=10)
    
    async def run(self):
        """Run the pipeline with direct trickle integration."""
        tasks = await asyncio.gather(
            self._subscriber_task(),
            self._processor_task(),
            self._publisher_task(),
            return_exceptions=True
        )
    
    async def _subscriber_task(self):
        """Subscribe to input stream and queue segments."""
        async with TrickleSubscriber(self.config.subscribe_url) as subscriber:
            while self.running:
                segment = await subscriber.next()
                if segment:
                    await self.input_queue.put(segment)
    
    async def _processor_task(self):
        """Process segments through your custom pipeline."""
        while self.running:
            segment = await self.input_queue.get()
            
            # Your custom processing logic
            processed_segment = await self.process_segment(segment)
            
            await self.output_queue.put(processed_segment)
    
    async def _publisher_task(self):
        """Publish processed segments to output stream."""
        async with TricklePublisher(self.config.publish_url) as publisher:
            while self.running:
                segment = await self.output_queue.get()
                
                # Publish to trickle endpoint
                writer = await publisher.preconnect()
                if writer:
                    await writer.put(segment)
                    
                publisher.idx += 1

# Usage
pipeline = MyVideoPipeline(config)
await pipeline.run()
```

## Capability Registration

PyTrickle includes built-in capability registration for orchestrator-managed deployments. Workers can automatically register their capabilities on startup:

```python
from pytrickle import RegisterCapability

# Simplest usage - uses all environment variables
RegisterCapability.register()

# Custom parameters with environment variables for the rest
RegisterCapability.register(
    capability_name="my-ai-worker",
    capability_desc="Custom AI video processor"
)

# With custom logger
RegisterCapability.register(logger, capability_name="my-ai-worker")

# Instance-based usage (if you need the instance)
registry = RegisterCapability(logger)
registry.register_capability(capability_name="my-ai-worker")
```

**Environment Variables:**
- `ORCH_URL`: Orchestrator URL (required)
- `ORCH_SECRET`: Authentication secret (required)
- `CAPABILITY_NAME`: Worker capability name (default: pytrickle-worker)
- `CAPABILITY_DESCRIPTION`: Worker description (default: PyTrickle video processing worker)
- `CAPABILITY_URL`: Worker endpoint URL (default: http://localhost:8080)
- `CAPABILITY_CAPACITY`: Max concurrent streams (default: 1)
- `CAPABILITY_PRICE_PER_UNIT`: Price per processing unit (default: 0)
- `CAPABILITY_PRICE_SCALING`: Price scaling factor (default: 1)

## Choosing the Right Approach

| Approach | Best For | Complexity | Control | Performance |
|----------|----------|------------|---------|-------------|
| **TrickleApp** | Simple AI apps, prototypes | Low | Medium | High |
| **Custom Server** | Production apps, complex pipelines | High | Full | Highest |
| **Direct Publisher/Subscriber** | Research, specialized workflows | Medium | Full | Medium |

**Example Applications:**
- **TrickleApp** ([method1_simple_trickle_app.py](examples/method1_simple_trickle_app.py)): Real-time filters, simple AI effects, microservices
- **Custom Server** ([method2_custom_server.py](examples/method2_custom_server.py)): Production streaming services, complex AI pipelines
- **Direct Integration** ([method3_direct_trickle.py](examples/method3_direct_trickle.py)): Transcription services, batch processing, research pipelines

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