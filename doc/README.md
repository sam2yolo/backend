# SAM-to-YOLO API documentation

This directory documents the backend API and contains runnable client examples.

## Documents

- [API.md](API.md) — complete HTTP and WebSocket API reference.
- [INFERENCE.md](INFERENCE.md) — model payloads, result formats, and inference workflows.
- [examples/README.md](examples/README.md) — setup and commands for all examples.

## Runnable examples

| Model | Input | Script |
|---|---|---|
| YOLO | Single image | `examples/yolo_single_image.py` |
| YOLO | Image batch | `examples/yolo_batch_images.py` |
| YOLO | Video | `examples/yolo_video.py` |
| SAM3 | Single image | `examples/sam_single_image.py` |
| SAM3 | Image batch | `examples/sam_batch_images.py` |
| SAM3 | Video frames | `examples/sam_video.py` |

The examples use the public backend API on port `8000`. They also work through
an FRP URL such as:

```bash
export BACKEND_URL=http://163.61.236.112:20000
```

Install client dependencies:

```bash
python -m pip install -r doc/examples/requirements.txt
```

The SAM service on port `8001` is normally internal. The main backend's
`samHandler` communicates with it.

## Create a test room

Run the terminal room creator from the repository root:

```bash
./create-room-tui
```

It creates a room and copies a command like this to the clipboard:

```bash
wget "https://raw.githubusercontent.com/sam2yolo/backend/refs/heads/main/run.sh" && bash run.sh <room-id> <room-secret> <notebook-name>
```

The notebook name is generated automatically in a Reddit-style format such as
`curious-otter-0427`.

After creating the room, the TUI continuously polls it. When the notebook adds
a tunnel, the TUI displays its public address, for example:

```text
http://163.61.236.112:20000
```

Set `TUNNEL_HOST` if the public FRP host differs from the broker hostname.

Override the broker configuration when necessary:

```bash
BROKER_URL=http://broker:7001 API_TOKEN=token ./create-room-tui
```
