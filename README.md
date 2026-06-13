# SamToYOLO Backend

A WebSocket-based RPC server that provides YOLO model inference (segmentation/detection) on images and videos, with file download management and training support.

## Architecture Overview

```
┌─────────────┐     WebSocket      ┌──────────────────┐
│   Client     │ ◄──────────────►  │  FastAPI Server  │
│  (Web UI /   │    /ws endpoint   │  (port 8000)     │
│   Test App)  │                   │                  │
└─────────────┘                    └────────┬─────────┘
                                            │
              ┌─────────────────────────────┼──────────────────┐
              │           Context           │                  │
              │  ┌────────────────────┐     │                  │
              │  │  Action Handlers   │     │                  │
              │  │  (@register'd)     │     │                  │
              │  ├────────────────────┤     │                  │
              │  │  Model Handler     │     │                  │
              │  │  (YOLO inference)  │     │                  │
              │  ├────────────────────┤     │                  │
              │  │  Response Queue    │     │                  │
              │  │  (thread-safe)     │     │                  │
              │  └────────────────────┘     │                  │
              └─────────────────────────────┘──────────────────┘
```

### Key Components

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server entry point; WebSocket endpoint, HTTP routes, lifecycle management |
| `actionhandler.py` | WebSocket action dispatcher — maps action names to handler functions |
| `context.py` | Global state: model registry, file registry, response queue, worker thread |
| `modelhandler.py` | YOLO model wrapper — inference (video/image), training, task queuing |
| `commons.py` | Thread-safe `send_action()` helper for asynchronous message delivery |
| `returnablequeue.py` | Custom retry queue for reliable message delivery |
| `run.sh` | Deployment/startup script (WIP) |
| `test_front/` | Standalone Flask web UI for testing WebSocket interactions |
| `test_sequence.py` | Integration test: init model → download file → run inference |
| `test_download_disconnect.py` | Integration test: download + disconnect/reconnect behavior |

---

## Prerequisites

- **Python 3.9+**
- **Conda** (recommended) or `venv`
- **FFmpeg** (for video frame extraction)
- **gdown** (for Google Drive downloads)

### Install System Dependencies

```bash
# FFmpeg
sudo apt update && sudo apt install ffmpeg -y

# gdown (for Google Drive file downloads)
pip install gdown
```

---

## Setup

### 1. Clone & Enter the Project

```bash
cd samtoyolo/backend
```

### 2. Create and Activate a Conda Environment (Recommended)

```bash
conda create -n samtoyolo python=3.9 -y
conda activate samtoyolo
```

Or use `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If no `requirements.txt` exists yet, install the core packages:
> ```bash
> pip install fastapi uvicorn[standard] websockets ultralytics opencv-python-headless ffmpeg-python numpy requests flask
> ```

### 4. Verify Installation

```bash
python -c "from ultralytics import YOLO; print('YOLO ready')"
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"
```

---

## Running the Server

### Start the Backend Server

```bash
conda activate samtoyolo
python app.py
```

The server starts on **`http://0.0.0.0:8000`**.

- **HTTP**: Open `http://localhost:8000` in your browser for the built-in WebSocket test UI (served from `test_front/templates/index.html`).
- **WebSocket**: Connect to `ws://localhost:8000/ws`.
- **REST Endpoints**:
  - `GET /` — HTML test client
  - `GET /inference_result?id=<chunk_id>` — download a saved inference result file

### Start the Test Frontend (Optional)

A standalone Flask UI is available for more structured testing:

```bash
cd test_front
python app.py     # runs on port 5000
```

---

## WebSocket Protocol

All communication uses JSON messages over a single WebSocket connection (`/ws`).

### Message Format

**Request** (client → server):
```json
{
  "action": "<action_name>",
  "payload": { ... }
}
```

**Response** (server → client):
```json
{
  "action": "<response_action>",
  "payload": { ... }
}
```

Only **one WebSocket connection** is allowed at a time. If a new client connects while another is active, the old connection is replaced and its pending messages are flushed to the new client.

---

## Available Actions

### General

| Action | Payload | Response | Description |
|--------|---------|----------|-------------|
| `ping` | `{}` | `pong` | Health check |
| `list_models` | `{}` | `list_models_response` | List registered model handlers |

### Model Management

| Action | Payload | Response(s) | Description |
|--------|---------|-------------|-------------|
| `init_model` | `{"model_name": "yolo", "variant_name": "yolo26n"}` | `model_setup_started` → `model_setup_completed` → `model_init_started` → `model_init_completed` (or `model_init_error`) | Install dependencies (`ultralytics`) and load the YOLO model variant |
| `destroy_model` | `{"model_name": "yolo"}` | `model_destroyed` | Unload model and delete cached weights |

**Supported variants**: `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`, `yolo26n`, etc. (the `-seg` suffix is appended automatically for segmentation models).

### File Management

| Action | Payload | Response(s) | Description |
|--------|---------|-------------|-------------|
| `download_file_wget` | `{"url": "https://..."}` | `file_download_initiated` → `download_progress` → `file_download_completed` (or `download_failed`) | Download a file via HTTP(S) with progress tracking |
| `download_file_google_drive` | `{"url": "https://drive.google.com/..."}` | Same as above | Download from Google Drive (uses `gdown` internally) |
| `list_files` | `{}` | `file_list` | List all downloaded files |
| `delete_file` | `{"file_id": "..."}` | `delete_file_success` / `delete_file_failed` | Remove a downloaded file |

### Inference (Task Queue)

Inference uses a task queue system. Tasks are added to a queue and processed sequentially.

| Action | Payload | Response(s) | Description |
|--------|---------|-------------|-------------|
| `create_inference_task` | `{"file_id": "...", "file_type": "video" \| "image", "conf": 0.25, "iou": 0.45, "frames": [...], "batch": 128, ...}` | `task_added` | Add an inference task to the queue |
| `delete_inference_task` | `{"id": "..."}` | `task_deleted` / `task_cancelled` / `task_not_found` | Remove a task from the queue |
| `start_inference_from_queue` | `{}` | `work_started` / `queue_empty` / `already_working` | Start processing the task queue |

#### Inference Task Payload Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `file_id` | ✅ | — | ID of a previously downloaded file |
| `file_type` | ✅ | — | `"video"` or `"image"` |
| `conf` | ❌ | `0.25` | Detection confidence threshold |
| `iou` | ❌ | `0.45` | NMS IoU threshold |
| `classes` | ❌ | `None` | List of class IDs to filter (e.g., `[0, 2]`) |
| `imgsz` | ❌ | model default | Inference image size |
| `frames` | ❌ | all frames | Specific frame indices for video (e.g., `[0, 10, 20]`) |
| `batch` | ❌ | `128` | Number of frames to process per batch (video only) |
| `temporal_downsampling` | ❌ | `False` | Enable temporal downsampling |
| `drop_rate` | ❌ | `0.01` | Fraction of frames to keep (if downsampling) |
| `persist` | ❌ | `None` | YOLO tracking: persist tracks across frames |
| `tracker` | ❌ | `None` | YOLO tracking: tracker configuration file |
| `save_dir` | ❌ | `results/{task_id}` | Custom output directory |

#### Inference Progress & Results

During inference, the server sends periodic progress updates:

- `inference_stage_plus_progress` — `{"task_id": "...", "progress": 45, "stage": "Processing frame 45/100"}`
- `inference_task_chunk_result` — Emitted for each completed batch chunk, containing:
  ```json
  {
    "chunk_id": "abcdef12",
    "chunk_index": 0,
    "task_id": "...",
    "frame_count": 10
  }
  ```
- `inference_completed` — All frames processed
- `inference_task_error` — Task failed
- `task_cancelled` — Task was cancelled

#### Fetching Inference Results

| Action | Payload | Response | Description |
|--------|---------|----------|-------------|
| `fetch_inference_chunk` | `{"task_id": "...", "chunk_id": "..."}` | `inference_chunk_data` | Load a specific chunk's results |
| `fetch_inference_chunk` | `{"task_id": "..."}` | `inference_chunk_list` | List all chunks for a task |

#### Deleting Inference Results

| Action | Payload | Response | Description |
|--------|---------|----------|-------------|
| `delete_inference_result` | `{"chunk_id": "..."}` | `inference_result_deleted` / `inference_result_delete_error` | Delete a single result chunk |
| `delete_inference_result_of_task` | `{"task_id": "..."}` | `inference_results_of_task_deleted` / `inference_result_delete_error` | Delete all results for a task |

### Training

| Action | Payload | Response(s) | Description |
|--------|---------|-------------|-------------|
| `create_inference_task` with `type: "train"` | `{"type": "train", "dataset": "...", "epochs": 100, ...}` | `training_started` → `training_completed` (or `task_failed`) | Train a YOLO model |

Training payload fields: `dataset` (path or `.zip` file), `epochs` (default `100`), `batch_size` (default `16`), `imgsz` (default `640`), `workers` (default `8`), `device` (default `"cpu"`), `project` (default `"runs/train"`), `name`.

---

## Running Tests

### Integration Test: Full Pipeline

```bash
python test_sequence.py
```

This runs: **init model → download video from Google Drive → create inference task → process frames**.

### Disconnect/Reconnect Test

```bash
python test_download_disconnect.py
```

Tests that file downloads survive WebSocket disconnections and pending messages are correctly flushed upon reconnection.

---

## Output Structure

Inference results are saved to the `results/` directory:

```
results/
└── {task_id}/
    ├── chunk-0.pkl          # Pickled YOLO Results for batch 0
    ├── chunk-1.pkl          # Pickled YOLO Results for batch 1
    ├── ...
    ├── annotations.json     # All frame annotations (if processed)
    ├── dataset_summary.json # Dataset statistics
    └── dataset/
        ├── images/          # Extracted/copied frames
        └── masks/           # Segmentation masks (PNG)
```

Downloaded files are stored in:

```
files/
├── {file_id}               # Raw downloaded file (named by UUID)
└── ...
```

---

## HTTP Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the WebSocket test UI (`index.html`) |
| `GET` | `/inference_result?id=<chunk_id>` | Downloads the pickle file for a given inference chunk |

---

## Customization

### Adding a New Action

1. Add a handler function in `actionhandler.py` decorated with `@register("action_name")`:

```python
@register("my_action")
async def handle_my_action(websocket: WebSocket, data: dict, context: Context):
    result = do_something(data)
    await websocket.send_text(json.dumps({
        "action": "my_action_response",
        "payload": result
    }))
```

2. Send `{"action": "my_action", "payload": {...}}` from the client.

### Adding a New Model Handler

1. Subclass `ModelHandler` in `modelhandler.py`.
2. Implement `setup()`, `init()`, `handle_inference_task()`, and optionally `handle_train_task()`.
3. Register it in the context:
   - In `context.py` → `load_models()` or
   - In `app.py` → `context.models_dict['my_model'] = MyModelHandler`
