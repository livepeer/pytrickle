# Trickle Media Examples

This guide shows how to reproduce the FFmpeg/FFplay workflows from
[`http-trickle`](https://github.com/J0sh/http-trickle) using the utilities that ship
with PyTrickle. The goal is to stream a sample video into a trickle channel, let a
PyTrickle processor consume/process the stream, and then play the output in
real-time.

## Prerequisites

- `ffmpeg` and `ffplay`
- A running trickle server (e.g. `cd ~/repos/http-trickle && make trickle-server addr=0.0.0.0:3389`)
- PyTrickle installed (`pip install -e .`)

## 1. Start a PyTrickle processor

Reuse any existing example—`passthrough_example` keeps frames unchanged and is
handy for smoke tests:

```bash
python -m pytrickle.examples.passthrough_example
```

## 2. Publish a video into trickle

`examples/trickle_media_tools.py` mimics the `publisher-ffmpeg` sample. Pipe an
FFmpeg MPEG-TS payload into the helper and it will upload the bytes into the
`input` channel:

```bash
ffmpeg -re -stream_loop -1 -i tests/demo.mp4 -c copy -f mpegts - \
  | python -m examples.trickle_media_tools \
      publisher --url http://127.0.0.1:3389 --stream input
```

Use `python -m examples.trickle_media_tools commands demo.mp4` to print
the exact pipeline commands for your sample file.

## 3. Tell PyTrickle to read from trickle

When the publish stream is live, start PyTrickle’s HTTP API and point it at the
`input`/`output` channels:

```bash
curl -X POST http://localhost:8000/stream/start \
  -H "Content-Type: application/json" \
  -d '{
    "subscribe_url": "http://127.0.0.1:3389/input",
    "publish_url": "http://127.0.0.1:3389/output",
    "gateway_request_id": "demo"
  }'
```

## 4. Subscribe and render with ffplay

Pipe the subscriber helper into FFplay. It mirrors the `make play` workflow from
http-trickle:

```bash
python -m examples.trickle_media_tools \
  subscriber --url http://127.0.0.1:3389 --stream output \
  | ffplay -probesize 32 -fflags nobuffer -flags low_delay -
```

## 5. Direct TrickleProtocol passthrough

The `examples/trickle_protocol_passthrough.py` script demonstrates how to work
straight against `TrickleProtocol`, matching the style of the upstream Go
examples. It subscribes to one channel and republishes the frames to another:

```bash
python -m examples.trickle_protocol_passthrough \
  --subscribe http://127.0.0.1:3389/input \
  --publish http://127.0.0.1:3389/output
```

Use this when you want a lightweight relay or pre/post-processing stage without
running the full StreamProcessor/HTTP server stack.

