# PyTrickle CLI quickstart

This repository now includes a lightweight CLI to scaffold and run apps.

## Install (editable)

Use your preferred environment, then install this package:

```bash
pip install -e .
```

## Scaffold a new app

```bash
pytrickle init my_app
cd my_app
python -m my_app
```

This will start a server on port 8000 with a minimal handlers skeleton.

Then POST to `/api/stream/start` with a JSON body, e.g.:

```json
{
  "subscribe_url": "http://localhost:3389/sample",
  "publish_url": "http://localhost:3389/output",
  "gateway_request_id": "demo-1",
  "params": {"width": 512, "height": 512}
}
```

## Run an existing module

```bash
pytrickle run --module my_app
```
