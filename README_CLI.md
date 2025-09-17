# PyTrickle CLI Reference

The PyTrickle CLI provides commands to scaffold and run streaming applications using the PyTrickle framework.

## Installation

Use your preferred Python environment, then install this package:

```bash
pip install -e .
```

After installation, the `pytrickle` command will be available globally.

## Commands Overview

```bash
pytrickle --help                    # Show main help
pytrickle init --help               # Show init command help  
pytrickle run --help                # Show run command help
```

## `pytrickle init` - Scaffold a New App

Creates a new PyTrickle application with a complete project structure.

### Usage

```bash
pytrickle init [PATH] [OPTIONS]
```

### Arguments

- `PATH` (optional): Target directory for the new app (default: current directory `.`)

### Options

- `--package NAME`: Package name to use (default: derived from folder name)
- `--force`: Overwrite existing files without prompting

### Examples

```bash
# Create app in current directory
pytrickle init

# Create app in specific directory
pytrickle init my_streaming_app

# Create with custom package name
pytrickle init my_app --package custom_name

# Force overwrite existing files
pytrickle init my_app --force
```

### Generated Structure

```
my_app/
├── my_app/
│   ├── __init__.py         # Package initializer
│   ├── __main__.py         # Entry point for 'python -m my_app'
│   └── handlers.py         # Main handlers with decorators
└── README.md               # Basic usage instructions
```

### Handler Template

The generated `handlers.py` includes:

- `@model_loader` - Load your model/resources on startup
- `@video_handler` - Process video frames (returns None for pass-through)
- `@param_updater` - Handle parameter updates during streaming
- `@on_stream_stop` - Cleanup when stream ends

## `pytrickle run` - Run an Existing App

Convenience command to run a Python module/package.

### Usage

```bash
pytrickle run --module MODULE_NAME
```

### Options

- `--module MODULE_NAME` (required): Name of the Python module/package to run

### Examples

```bash
# Run a scaffolded app
pytrickle run --module my_app

# Run any Python module with __main__.py
pytrickle run --module my_existing_package
```

This is equivalent to `python -m MODULE_NAME` but provides consistent CLI experience.

## Quick Start Workflow

1. **Create a new app:**
   ```bash
   pytrickle init video_processor
   cd video_processor
   ```

2. **Customize your handlers:**
   Edit `video_processor/handlers.py` to implement your processing logic.

3. **Run the app:**
   ```bash
   python -m video_processor
   ```
   or
   ```bash
   pytrickle run --module video_processor
   ```

4. **Test the streaming endpoint:**
   The server starts on port 8000. Send a POST request to `/api/stream/start`:
   
   ```json
   {
     "subscribe_url": "http://localhost:3389/sample",
     "publish_url": "http://localhost:3389/output", 
     "gateway_request_id": "demo-1",
     "params": {"width": 512, "height": 512}
   }
   ```

## Server Endpoints

Once running, your app exposes these endpoints:

- `POST /api/stream/start` - Start streaming with parameters
- `POST /api/stream/stop` - Stop active stream
- `POST /api/stream/params` - Update stream parameters
- `GET /api/stream/status` - Get current stream status
- `GET /health` - Health check
- `GET /version` - App version info

## Development Tips

- **Debugging**: Add logging to your handlers to trace execution
- **Parameters**: Use the `@param_updater` to modify behavior during streaming
- **Frame Processing**: Return `None` from handlers for pass-through behavior
- **Error Handling**: Exceptions in handlers are caught and logged automatically
- **Model Loading**: Use `@model_loader` for one-time initialization on startup

## Advanced Usage

### Custom Port

Modify the `main()` function in your generated `handlers.py`:

```python
processor = StreamProcessor.from_handlers(
    handlers,
    name="my-app",
    port=9000,  # Custom port
    frame_skip_config=FrameSkipConfig(),
)
```

### Multiple Handler Types

Add more decorators to your handler class:

```python
@audio_handler
async def process_audio(self, frame):
    # Process audio frames
    return None
```

### Custom Frame Skipping

Configure frame skipping in your main function:

```python
from pytrickle.frame_skipper import FrameSkipConfig

config = FrameSkipConfig(
    max_queue_size=10,
    skip_threshold=5
)
processor = StreamProcessor.from_handlers(
    handlers,
    frame_skip_config=config
)
```
