# SAM-to-YOLO Backend API v1.0

This backend exposes one JSON-RPC 2.0 WebSocket endpoint for control messages and
small payloads, plus HTTP endpoints for large file transfer.

- WebSocket RPC: `/v1/ws`
- Uploads: `POST /v1/projects/{project_id}/uploads/{kind}`
- Downloads: `GET /v1/projects/{project_id}/downloads/...`
- Health: `GET /health`
- Method index: `GET /v1/methods`

Production deployments should expose the server through HTTPS/WSS. In remote
mode, the backend can launch `cloudflared`, register the tunnel URL with the
Tunnelbroker peer registry, and refresh that peer record periodically.

## Normalized Terminology

The rough specification used several mixed naming styles. v1.0 standardizes
public method and event names to `snake_case`, except namespaced notifications
that intentionally use dots.

| Draft term | Canonical term | Notes |
| --- | --- | --- |
| `endpoint: /v1/<method>` | `/v1/ws` | JSON-RPC method name is inside each WebSocket message. |
| `backend-init` | `backend_init` | Event and method use underscore form. |
| `backend-init-progress` | `backend_init_progress` | Event. |
| `backend-ready` | `backend_ready` | Event. |
| `video_upload` | `upload_video` | `video_upload` remains as an alias. |
| `image_upload` | `upload_image` | Alias preserved. |
| `dataset_upload` | `upload_dataset` | Alias preserved. |
| `gdrive_upload` | `upload_from_google_drive` | Alias preserved. |
| `mega_upload` | `upload_from_mega` | Alias preserved for public Mega links. |
| `register_mega_credential` / `register_mega` | `register_mega_credentials` | Aliases preserved. |
| `train-yolo` | `train_yolo` | Alias preserved. |
| `train` | `train_model` | Alias preserved. |
| `inference-to-yolo` | `convert_inference_to_yolo` | Aliases preserved. |
| `getInferenceResult` | `download_inference_result` | Alias preserved. |
| `getModel` | `download_model` | Alias preserved. |
| `downsample` / `temporal_downsample` | `sample_interval_seconds` | Legacy keys are accepted. Values below `1` are treated as a frame rate, so `0.1` means one frame every 10 seconds. |

## JSON-RPC Envelope

Client request:

```json
{
  "jsonrpc": "2.0",
  "id": "req-1",
  "method": "create_project",
  "params": {
    "project_id": "demo"
  }
}
```

Successful response:

```json
{
  "jsonrpc": "2.0",
  "id": "req-1",
  "result": {
    "project_id": "demo"
  }
}
```

Error response:

```json
{
  "jsonrpc": "2.0",
  "id": "req-1",
  "error": {
    "code": -32602,
    "message": "project_id is required"
  }
}
```

Server notification:

```json
{
  "jsonrpc": "2.0",
  "method": "task_progress",
  "params": {
    "task_id": "task_abc",
    "progress": 42.5,
    "message": "processed batch 3/8"
  }
}
```

## Project Workspace

Every project lives under `projects/{project_id}` by default. Override this with
`SAMTOYOLO_PROJECT_ROOT`.

```text
projects/{project_id}/
  session.json
  uploads/
  frames/
  inference_results/
  datasets/
  models/
  checkpoints/
  tmp/
```

`session.json` is the persistent project/session record. It stores tasks,
uploads, inference artifacts, datasets, trained models, pending client sync
requests, and non-secret secret metadata. Mega credentials are stored only in
memory.

On startup, incomplete tasks from all project sessions are moved back to the
queue. Real SAM/training adapters should inspect their checkpoint directories
and skip completed batches or epochs.

## Task Model

Long-running work is submitted as a task and executed by GPU worker coroutines.
The number of workers defaults to the detected CUDA GPU count, then falls back to
one. Override it with `SAMTOYOLO_GPU_WORKERS`.

Task status values:

- `queued`
- `running`
- `succeeded`
- `failed`
- `cancelled`

All task-submitting methods return:

```json
{
  "task_id": "task_123",
  "status": "queued"
}
```

## Methods

### Backend

#### `backend_init()`

Checks runtime readiness and emits:

- `backend_init`
- `backend_init_progress`
- `backend_ready`

Result:

```json
{
  "ready": true,
  "project_root": "/abs/path/projects",
  "gpu_workers": 1,
  "mode": "local",
  "runtime_environment": {
    "bootstrap_enabled": true,
    "bootstrapped": true,
    "conda_env_name": "samtoyolo-backend",
    "in_conda_env": true
  }
}
```

### Capabilities

#### `models()`

Lists supported inference and training model names. SAM 3.1 is reached through
an isolated model server so its CUDA/PyTorch/SAM dependencies do not pollute the
backend environment. The SAM 3.1 entry includes `model_source_url`,
`model_download_url`, `model_file_id`, `model_filename`, and
`model_server_url`.

Aliases: `list_models`

### Projects

#### `create_project(project_id, display_name?)`

Creates the project directory and `session.json` if needed.

Alias: `project.create`

#### `get_project_list()`

Returns all known projects.

Alias: `project.list`

#### `get_project_info(project_id)`

Returns project metadata, uploads, datasets, models, and inference results.

Alias: `project.info`

### Uploads

Large binary uploads should use HTTP. The WebSocket upload methods either return
the HTTP endpoint or register an already-present server-side file path.

#### `upload_video(project_id, source_path?, filename?)`

Alias: `video_upload`

If `source_path` is omitted:

```json
{
  "transport": "http",
  "method": "POST",
  "upload_endpoint": "/v1/projects/demo/uploads/video",
  "field": "file"
}
```

If `source_path` is supplied, it must be inside the project directory and a task
is queued to register it.

#### `upload_image(project_id, source_path?, filename?)`

Alias: `image_upload`

#### `upload_dataset(project_id, source_path?, filename?, format?)`

Alias: `dataset_upload`

Uploaded datasets are registered in the project dataset list. v1 expects zip
datasets for download and merge/map operations.

#### `upload_from_google_drive(project_id, url, kind?, filename?, format?)`

Downloads a public Google Drive file. The backend converts common
`drive.google.com/file/d/...` links to direct `uc?export=download&id=...` links.

Aliases: `gdrive_upload`, `download_gdrive`

#### `upload_from_url(project_id, url, kind?, filename?, format?)`

Downloads a direct public URL.

Aliases: `direct_link_upload`, `download_url`

#### `upload_from_mega(project_id, url, kind?, filename?, format?)`

Queues a Mega public-link download. The current executor returns a task failure
unless a Mega public-link adapter/CLI is installed and wired in.

Aliases: `mega_upload`, `download_mega`

### Inference

#### `inference_sam3(...)`

Runs SAM 3.1 text-prompt segmentation.

```json
{
  "project_id": "demo",
  "media_path": "uploads/video/input.mp4",
  "prompts": ["red car", "person"],
  "prompt_to_class": {
    "red car": "car",
    "person": "person"
  },
  "sample_interval_seconds": 10,
  "batch_size": 4,
  "save_to_mega": false,
  "prepare_model": true
}
```

Aliases: `inference.sam3`

Notes:

- `media_path` must point inside the project directory.
- Video inputs may use common OpenCV-readable formats such as MP4, MOV, AVI,
  MKV, WEBM, and DAV.
- `sample_interval_seconds` controls video frame sampling.
- Legacy keys `temporal_downsample` and `downsample` are accepted.
- By default, the backend downloads the configured SAM 3.1 zip model artifact
  to `SAMTOYOLO_MODEL_CACHE_DIR`, extracts it once, and reuses the extracted
  directory on later inference runs.
- Real inference is sent to the configured SAM model server
  (`model_server_url` or `SAMTOYOLO_SAM3_SERVER_URL`). That server loads
  `sam3.1_multiplex.pt` from the extracted artifact using the official `sam3`
  video predictor. The requested GPU is the task's assigned `gpu_index`.
- Set `use_stub_inference=true` only for lightweight developer smoke tests.
- Optional SAM runtime keys include `output_prob_thresh`,
  `confidence_threshold`, `include_masks`, `sam3_max_num_objects`,
  `sam3_multiplex_count`, `sam3_use_fa3`, `sam3_compile`,
  `sam3_async_loading_frames`, `sam3_cache_model`, and
  `sam3_allow_partial_checkpoint`.
- The backend refuses to continue if the SAM checkpoint reports missing or
  unexpected keys during load, because that would produce non-real predictions.
  `sam3_allow_partial_checkpoint=true` is only for debugging incompatible SAM
  package/checkpoint pairs.
- Each batch writes a checkpoint JSON file.

Output artifact:

```text
inference_results/{task_id}.zip
  frames/
  annotations.json
  metadata.json
  checkpoints/
```

#### `inference_yolo(project_id, media_path, batch_size?, sample_interval_seconds?)`

Runs detector inference through the same task/checkpoint pipeline.

Aliases: `inference.yolo`

### Dataset Operations

#### `convert_inference_to_yolo(project_id, inference_task_id, target_format="yolo")`

Converts an inference result zip into a YOLO dataset zip.

Aliases: `inference_to_dataset`, `inference_to_yolo`, `inference-to-yolo`

#### `merge_dataset(project_id, dataset_ids, class_map?)`

Merges multiple YOLO datasets and remaps class names.

Example:

```json
{
  "project_id": "demo",
  "dataset_ids": ["dataset_a", "dataset_b"],
  "class_map": {
    "cat": "feline",
    "kitty": "feline",
    "dog": "canine"
  }
}
```

Alias: `dataset.merge`

#### `map_dataset(project_id, dataset_id, class_map)`

Creates a new dataset with renamed/remapped classes.

Alias: `dataset.map`

#### `list_datasets(project_id)`

Returns registered datasets.

Alias: `dataset.list`

### Training

#### `train_model(project_id, model_name, dataset_id, config?)`

Supported names:

- `yolov8`
- `rt_detr`
- `grounding_dino`
- `efficientdet`
- `ssd`
- `retinanet`
- `faster_rcnn`
- `detectron2`
- `mediapipe`

Example:

```json
{
  "project_id": "demo",
  "model_name": "yolov8",
  "dataset_id": "dataset_123",
  "config": {
    "epochs": 50,
    "learning_rate": 0.001,
    "checkpoint_interval": 5
  }
}
```

Alias: `train`

#### `train_yolo(project_id, dataset_id, config?)`

Shortcut for `train_model` with `model_name="yolov8"`.

Alias: `train-yolo`

### Mega

#### `check_mega_credentials(email, password, project_id?)`

Checks that credentials were supplied and emits
`mega_credential_check_result`. Live Mega login validation is an adapter point.

Alias: `mega.credentials.check`

#### `register_mega_credentials(project_id, email, password)`

Stores credentials in process memory only and records a non-secret marker in
`session.json`.

Aliases: `register_mega_credential`, `register_mega`

### Task Info

#### `get_task_list(project_id?)`

Lists tasks for one project, or all projects if `project_id` is omitted.

Alias: `task.list`

#### `get_task_status(task_id)`

Returns the full task record.

Alias: `task.status`

### Downloads

#### `download_inference_result(task_id, delete_after_download?)`

Returns the HTTP download URL and artifact metadata. If
`delete_after_download=true`, the HTTP URL includes a query flag that deletes
the zip after the response is sent.

Aliases: `getInferenceResult`, `get_inference_result`

#### `download_dataset(project_id, dataset_id)`

Returns the dataset zip URL.

Aliases: `getDataset`, `get_dataset`

#### `download_model(project_id, model_id)`

Returns the model file URL.

Aliases: `getModel`, `get_model`

### Control

#### `cancel_task(task_id)`

Marks queued or running tasks as cancelled. Executors check cancellation between
batches/epochs.

Alias: `task.cancel`

#### `client_response(requestId, project_id?, status?, data?)`

Records the frontend response to a `client.ask` notification.
`request_id` is accepted as an alias for `requestId`. If `project_id` is omitted,
the backend searches project sessions for the pending request.

Alias: `client.response`

### Tunnel

#### `get_tunnel_status()`

Returns local/remote mode, public endpoint, registration state, and last error.

Alias: `tunnel.status`

#### `restart_tunnel()`

Restarts `cloudflared` and re-registers the peer. In local mode this is a no-op
with a status message.

Alias: `tunnel.restart`

## Events

| Event | Payload |
| --- | --- |
| `backend_init` | `{}` |
| `backend_init_progress` | `{ "percent": 0-100, "message": "..." }` |
| `backend_ready` | `{}` |
| `task_started` | `{ "task_id": "...", "description": "..." }` |
| `task_progress` | `{ "task_id": "...", "progress": 0-100, "message": "...", "metrics": {} }` |
| `task_complete` | `{ "task_id": "...", "result": {} }` |
| `task_failed` | `{ "task_id": "...", "error": "..." }` |
| `task_cancelled` | `{ "task_id": "..." }` |
| `inference_result_ready` | `{ "task_id": "...", "file_path_or_url": "/v1/..." }` |
| `training_complete` | `{ "task_id": "...", "model_id": "...", "metrics": {} }` |
| `new_dataset_from_merge` | `{ "task_id": "...", "dataset_id": "..." }` |
| `new_dataset_from_map` | `{ "task_id": "...", "dataset_id": "..." }` |
| `mega_credential_check_result` | `{ "valid": true/false/null, "message": "..." }` |
| `mega_mount_success` | `{ "task_id": "..." }` |
| `mega_upload_success` | `{ "task_id": "...", "mega_path": "..." }` |
| `upload_success` | `{ "task_id": "...", "filename": "...", "path": "..." }` |
| `session_expiring` | `{ "remaining_seconds": 900 }` |
| `client.ask` | `{ "requestId": "...", "type": "sync_file", "data": {} }` |
| `server.notification` | `{ "level": "info|warning|error", "message": "..." }` |
| `tunnel_ready` | `{ "endpoint": "https://...", "server_name": "..." }` |

## HTTP File Transfer

### Upload

```bash
curl -F "file=@input.mp4" \
  http://localhost:8000/v1/projects/demo/uploads/video
```

Response:

```json
{
  "task_id": "task_123",
  "status": "queued",
  "filename": "input.mp4",
  "path": "uploads/video/input.mp4",
  "size_bytes": 123456
}
```

### Download Inference Result

```bash
curl -OJ \
  "http://localhost:8000/v1/projects/demo/downloads/inference/task_123"
```

Delete after download:

```bash
curl -OJ \
  "http://localhost:8000/v1/projects/demo/downloads/inference/task_123?delete_after_download=true"
```

### Download Dataset

```bash
curl -OJ \
  "http://localhost:8000/v1/projects/demo/downloads/datasets/dataset_123"
```

### Download Model

```bash
curl -OJ \
  "http://localhost:8000/v1/projects/demo/downloads/models/model_123"
```

## Deployment Environment

Common variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SAMTOYOLO_PROJECT_ROOT` | `projects` | Project/session storage root. |
| `SAMTOYOLO_MODE` | `local` | `local` or `remote`. |
| `SAMTOYOLO_HOST` | `0.0.0.0` | Uvicorn host. |
| `SAMTOYOLO_PORT` | `8000` | Uvicorn port. |
| `SAMTOYOLO_GPU_WORKERS` | auto | Number of GPU worker coroutines. |
| `SAMTOYOLO_INSTANCE_TTL_SECONDS` | `42600` | 11h50m cloud-session timer. |
| `SAMTOYOLO_EXPIRY_NOTICE_SECONDS` | `900` | When to emit `session_expiring`. |
| `SAMTOYOLO_ALLOW_STUB_ML` | `true` | Allows stub inference/training artifacts before real adapters are installed. |
| `SAMTOYOLO_SAM3_MODEL_URL` | Google Drive SAM 3.1 archive | Default SAM 3.1 model artifact source. |
| `SAMTOYOLO_SAM3_MODEL_FILENAME` | `sam3.1.zip` | Expected SAM 3.1 archive filename. |
| `SAMTOYOLO_SAM3_SERVER_URL` | `ws://127.0.0.1:8101/v1/ws` | SAM 3.1 model server WebSocket endpoint. |
| `SAMTOYOLO_MODEL_CACHE_DIR` | `{project_root}/_models` | Shared model artifact cache directory. |
| `SAMTOYOLO_CONDA_BOOTSTRAP` | `true` | Create/re-enter the backend conda environment on startup. |
| `SAMTOYOLO_CONDA_ENV_NAME` | `samtoyolo-backend` | Backend runtime conda environment name. |
| `SAMTOYOLO_CONDA_PYTHON` | `3.12` | Python version used when creating the env. |
| `SAMTOYOLO_CONDA_INSTALL_PREFIX` | `~/.samtoyolo/miniforge3` | Conda install path when conda is missing. |
| `SAMTOYOLO_TORCH_INDEX_URL` | PyTorch CUDA 12.8 index | Torch wheel index used by model-server setup scripts. |
| `SAMTOYOLO_INSTALL_TORCH` | `false` | Optional legacy backend torch install. Model servers install their own torch. |
| `SAMTOYOLO_REQUIREMENTS_FILE` | `requirements.txt` | Requirements file installed into the env. |
| `TUNNELBROKER_URL` | unset | Peer registry base URL. |
| `TUNNELBROKER_GROUP` | unset | Peer registry group. |
| `TUNNELBROKER_GROUP_TOKEN` | unset | Optional group read/write token. |
| `TUNNELBROKER_PEER_SECRET` | unset | Peer-owned write secret. |
| `CLOUDFLARED_PATH` | `cloudflared` | Cloudflared executable. |

Run locally:

```bash
python -m samtoyolo_backend.run
```

On first run, the backend checks for the configured conda environment. If it is
missing, the backend creates it, installs the repository requirements, then
re-execs itself inside that environment. If conda is not installed, Miniforge is
installed first.

Direct uvicorn startup is also supported and performs the same bootstrap:

```bash
uvicorn samtoyolo_backend.main:app --host 0.0.0.0 --port 8000
```

Remote mode:

```bash
SAMTOYOLO_MODE=remote \
TUNNELBROKER_URL=https://tunnelbroker.example.workers.dev \
TUNNELBROKER_GROUP=teamA \
TUNNELBROKER_PEER_SECRET=peer-owned-secret \
uvicorn samtoyolo_backend.main:app --host 0.0.0.0 --port 8000
```

## Model Server Boundaries

Each heavy model runs in a separate FastAPI model server with its own conda
environment and WebSocket JSON-RPC endpoint. The backend keeps project state,
task queues, frame extraction, result packaging, checkpoint events, and client
sync. Model servers own CUDA runtimes and model-specific packages.

In v1.0, model servers are intended to run on the same machine or a shared
filesystem with the backend. The backend passes sampled frame directories and
model-checkpoint directories by path. Remote model-server hosts should add a
file-staging layer before inference.

SAM 3.1 ships with:

- setup: `model_servers/sam3/setup.sh`
- run script: `model_servers/sam3/run.sh`
- endpoint: `ws://127.0.0.1:8101/v1/ws`
- RPC method: `sam3.infer_video_text`

The backend adapter points are:

- `run_sam3_video_text_inference` in
  `samtoyolo_backend/executors/sam3_adapter.py`
- SAM model-server implementation in
  `samtoyolo_model_servers/sam3/server.py`
- official SAM runtime implementation in
  `samtoyolo_model_servers/sam3/inference.py`
- `execute_inference_sam3` in `samtoyolo_backend/executors/inference.py`
- `execute_inference_yolo` in `samtoyolo_backend/executors/inference.py`
- `execute_train_model` in `samtoyolo_backend/executors/training.py`
- upload/download logic in `samtoyolo_backend/executors/upload.py`
- Mega mount/upload adapter logic in `samtoyolo_backend/handlers/mega_handlers.py`

Replace the stub sections with real model runners while preserving:

- task checkpoint writes,
- `task_progress` events,
- cancellation checks,
- final artifact registration in `session.json`,
- client twin sync via `client.ask`.

## SAM 3.1 Model Artifact

The default SAM 3.1 model source is:

```text
https://drive.google.com/file/d/1U_SBWxdyRFx-519v_UQZh48cm4y4qLVm/view?usp=sharing
```

The backend exposes this through `models()` as `model_source_url` and derives a
direct Google Drive URL as `model_download_url`. Override it with
`SAMTOYOLO_SAM3_MODEL_URL` when running with a different artifact.

When `inference_sam3` starts, the server prepares that artifact before frame
processing:

```text
{SAMTOYOLO_MODEL_CACHE_DIR}/sam3/downloads/sam3.1.zip
{SAMTOYOLO_MODEL_CACHE_DIR}/sam3/extracted/sam3.1/
```

The inference result metadata includes a `model_asset` object with the archive
path, extract directory, manifest path, size, and whether the asset was reused.
