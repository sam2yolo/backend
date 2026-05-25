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

Compatibility/discovery method for SAM 3.1 text-prompt segmentation. The main
backend prepares the SAM3 artifact, ensures the SAM3 model server is available,
and returns the model-server endpoint and RPC method to call. Model servers own
the CUDA runtime and the actual inference call.

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
  "output_mode": "both",
  "include_masks": true,
  "visualize": true,
  "save_to_mega": false,
  "prepare_model": true
}
```

Aliases: `inference.sam3`

Parameters:

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `project_id` | string | required | Project workspace id. |
| `media_path` | string | required | Project-relative image, image directory, or video path. |
| `prompts` | string[] | `[]` | Text noun phrases passed to SAM3, such as `Road` or `Person`. |
| `prompt_to_class` | object | `{}` | Maps each prompt to the output class name. Missing prompts map to themselves. |
| `sample_interval_seconds` | number | `1` | Sequential video sampling interval. Ignored when random sampling is requested. |
| `temporal_downsample` / `downsample` | number | optional | Compatibility aliases for sampling interval. Values `< 1` mean `1 / value` seconds. |
| `sample_strategy` | string | sequential | Use `random` to sample random frames from a video. |
| `max_frames` | integer | unset | Maximum number of video frames to extract. When set with a video, random sampling is used unless `sample_strategy` says otherwise. |
| `random_seed` | integer | unset | Optional deterministic seed for random frame selection. |
| `max_frame_width` | integer | unset | Optional resize width while extracting sampled video frames. Keeps aspect ratio. |
| `batch_size` | integer | `4` | Checkpoint batch size. Values greater than `4` are capped to `4`. |
| `output_mode` | string | `both` | `bbox`, `segmentation`, or `both`. `both` emits boxes and masks. |
| `include_masks` | boolean | `true` | Compatibility switch. If `false`, segmentation output is disabled. |
| `visualize` | boolean | `true` | Adds `visualizations/` images to the result zip by default. |
| `inference_backend` | string | `video` | SAM3 backend selector. Use `video` for the video tracker or `image` for per-frame text segmentation. |
| `sam3_backend` | string | optional | Alias for `inference_backend`. |
| `confidence_threshold` / `output_prob_thresh` | number | `0.5` | SAM3 object confidence threshold. |
| `sam3_max_num_objects` | integer | `16` | Video backend object slots. The SAM3.1 multiplex checkpoint expects `16`. |
| `sam3_multiplex_count` | integer | `16` | Video backend multiplex count. |
| `sam3_use_fa3` | boolean | `false` | Enables FA3 when supported by GPU/runtime. |
| `sam3_compile` | boolean | `false` | Enables PyTorch compile for SAM3. |
| `sam3_cache_model` | boolean | `true` | Keeps the loaded model in the model-server process for later requests. |
| `sam3_allow_partial_checkpoint` | boolean | `false` | Debug-only video backend option for non-clean checkpoint loads. |
| `prepare_model` | boolean | `true` | Downloads/extracts the SAM3 archive before inference. |
| `model_source_url` | string | configured SAM3 URL | Override SAM3 model archive source. |
| `model_download_url` | string | derived | Override direct archive download URL. |
| `model_filename` | string | `sam3.1.zip` | Archive filename used for the cache. |
| `save_to_mega` | boolean | `false` | Uploads final result to Mega when Mega support is configured. |

Notes:

- Returned `model_server.public_ws_url` is the public Cloudflare/Tunnelbroker
  WebSocket endpoint when remote mode is enabled; otherwise use
  `model_server.local_ws_url`.
- Call `inference_sam3` or `sam3.infer_video_text` on the model-server endpoint
  for real inference.
- The main backend keeps upload/project/dataset management; model servers own
  inference and training runtimes.
- The `video` backend performs SAM3 video propagation and can emit segmentation
  when the GPU has enough memory. The `image` backend runs SAM3 text
  segmentation independently per sampled frame and is the practical option for
  smaller GPUs such as T4.
- `output_mode="bbox"` writes only `bbox`/`bbox_format`; `segmentation` writes
  only `segmentation`/`area`; `both` writes both.

Output artifact:

```text
inference_results/{task_id}.zip
  frames/
  annotations.json
  metadata.json
  checkpoints/
  visualizations/        # present when visualize=true
```

`annotations.json` shape:

```json
{
  "frames": [
    {
      "file_name": "000000.jpg",
      "width": 640,
      "height": 360,
      "objects": [
        {
          "object_id": "frame0_prompt0_0",
          "prompt": "Road",
          "class_name": "Road",
          "confidence": 0.91,
          "bbox": [0.0, 180.2, 640.0, 179.8],
          "bbox_format": "xywh_abs",
          "segmentation": {
            "format": "rle",
            "encoding": "uncompressed_coco_rle",
            "size": [360, 640],
            "counts": [1200, 53, 11]
          },
          "area": 82411
        }
      ]
    }
  ]
}
```

Clear end-to-end example for 20 random frames, max batch size 4, boxes plus
segmentation, and default visualizations:

```json
{
  "jsonrpc": "2.0",
  "id": "sam3-demo-1",
  "method": "inference_sam3",
  "params": {
    "project_id": "cctv-demo",
    "media_path": "uploads/video/cctv.dav",
    "prompts": ["Road", "Building", "Person", "Vehicle", "Dog"],
    "prompt_to_class": {
      "Road": "Road",
      "Building": "Building",
      "Person": "Person",
      "Vehicle": "Vehicle",
      "Dog": "Dog"
    },
    "sample_strategy": "random",
    "max_frames": 20,
    "random_seed": 20260525,
    "max_frame_width": 640,
    "batch_size": 4,
    "inference_backend": "image",
    "output_mode": "both",
    "include_masks": true,
    "visualize": true,
    "confidence_threshold": 0.35
  }
}
```

The corresponding model-server call uses the returned model-server WebSocket:

```json
{
  "jsonrpc": "2.0",
  "id": "sam3-model-call-1",
  "method": "sam3.infer_video_text",
  "params": {
    "frames_dir": "/workspace/projects/cctv-demo/inference_results/task_123/frames",
    "frames": [
      "/workspace/projects/cctv-demo/inference_results/task_123/frames/000000.jpg"
    ],
    "prompts": ["Road", "Building", "Person", "Vehicle", "Dog"],
    "prompt_to_class": {
      "Road": "Road",
      "Building": "Building",
      "Person": "Person",
      "Vehicle": "Vehicle",
      "Dog": "Dog"
    },
    "model_extract_dir": "/workspace/projects/_models/sam3/extracted/sam3.1",
    "gpu_index": 0,
    "inference_backend": "image",
    "output_mode": "both",
    "include_masks": true,
    "output_prob_thresh": 0.35,
    "cache_model": true
  }
}
```

During model-server inference, progress is sent as JSON-RPC notifications:

```json
{
  "jsonrpc": "2.0",
  "method": "model.progress",
  "params": {
    "progress": 42.5,
    "message": "SAM image prompt 4/5 frame 9/20",
    "metrics": {
      "gpu_index": 0,
      "prompt": "Vehicle",
      "frame_index": 8
    }
  }
}
```

After the backend emits `inference_result_ready`, download the zip with:

```bash
curl -OJ \
  "http://localhost:8000/v1/projects/cctv-demo/downloads/inference/task_123"
```

Clients can summarize `annotations.json` after download. For example, a
20-frame run might produce:

```json
{
  "total_objects": 354,
  "objects_by_class": {
    "Building": 152,
    "Person": 62,
    "Road": 89,
    "Vehicle": 51
  },
  "bbox_count": 354,
  "segmentation_count": 354
}
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

Compatibility discovery method for training. The main backend no longer trains
models directly; it returns the matching model-server endpoint and the
`train_model` RPC method to call. Model-server training support is implemented
per model.

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
| `model.progress` | `{ "progress": 0-100, "message": "...", "metrics": {} }` from a model-server WebSocket. |
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

| Variable                                | Default                      | Purpose                                                                       |
| --------------------------------------- | ---------------------------- | ----------------------------------------------------------------------------- |
| `SAMTOYOLO_PROJECT_ROOT`                | `projects`                   | Project/session storage root.                                                 |
| `SAMTOYOLO_MODE`                        | `local`                      | `local` or `remote`.                                                          |
| `SAMTOYOLO_HOST`                        | `0.0.0.0`                    | Uvicorn host.                                                                 |
| `SAMTOYOLO_PORT`                        | `8000`                       | Uvicorn port.                                                                 |
| `SAMTOYOLO_GPU_WORKERS`                 | auto                         | Number of GPU worker coroutines.                                              |
| `SAMTOYOLO_INSTANCE_TTL_SECONDS`        | `42600`                      | 11h50m cloud-session timer.                                                   |
| `SAMTOYOLO_EXPIRY_NOTICE_SECONDS`       | `900`                        | When to emit `session_expiring`.                                              |
| `SAMTOYOLO_ALLOW_STUB_ML`               | `true`                       | Allows stub inference/training artifacts before real adapters are installed.  |
| `SAMTOYOLO_SAM3_MODEL_URL`              | Google Drive SAM 3.1 archive | Default SAM 3.1 model artifact source.                                        |
| `SAMTOYOLO_SAM3_MODEL_FILENAME`         | `sam3.1.zip`                 | Expected SAM 3.1 archive filename.                                            |
| `SAMTOYOLO_SAM3_SERVER_URL`             | `ws://127.0.0.1:8101/v1/ws`  | SAM 3.1 model server WebSocket endpoint.                                      |
| `SAMTOYOLO_MODEL_SERVERS_AUTO_START`    | `true`                       | Set up and start model servers on backend startup.                            |
| `SAMTOYOLO_MODEL_SERVERS_PUBLIC_TUNNEL` | `false`                      | Create public tunnels for model servers even outside remote mode.             |
| `SAMTOYOLO_SAM3_SERVER_PORT`            | `8101`                       | Local SAM3 model server port.                                                 |
| `SAMTOYOLO_MODEL_CACHE_DIR`             | `{project_root}/_models`     | Shared model artifact cache directory.                                        |
| `SAMTOYOLO_CONDA_BOOTSTRAP`             | `true`                       | Create/re-enter the backend conda environment on startup.                     |
| `SAMTOYOLO_CONDA_ENV_NAME`              | `samtoyolo-backend`          | Backend runtime conda environment name.                                       |
| `SAMTOYOLO_CONDA_PYTHON`                | `3.12`                       | Python version used when creating the env.                                    |
| `SAMTOYOLO_CONDA_CHANNEL`               | `conda-forge`                | Conda channel used for new backend/model-server environments.                 |
| `SAMTOYOLO_CONDA_AUTO_ACCEPT_TOS`        | `true`                       | Accepts Anaconda default-channel TOS when an existing conda install requires it. |
| `SAMTOYOLO_CONDA_INSTALL_PREFIX`        | `~/.samtoyolo/miniforge3`    | Conda install path when conda is missing.                                     |
| `SAMTOYOLO_TORCH_INDEX_URL`             | PyTorch CUDA 12.8 index      | Torch wheel index used by model-server setup scripts.                         |
| `SAMTOYOLO_INSTALL_TORCH`               | `false`                      | Optional legacy backend torch install. Model servers install their own torch. |
| `SAMTOYOLO_REQUIREMENTS_FILE`           | `requirements.txt`           | Requirements file installed into the env.                                     |
| `TUNNELBROKER_URL`                      | `https://tunnelbroker.sam2yolo.workers.dev` | Peer registry base URL.                                                       |
| `TUNNELBROKER_GROUP`                    | unset                        | Peer registry group.                                                          |
| `TUNNELBROKER_GROUP_TOKEN`              | unset                        | Optional group read/write token.                                              |
| `TUNNELBROKER_PEER_SECRET`              | unset                        | Peer-owned write secret.                                                      |
| `CLOUDFLARED_PATH`                      | `cloudflared`                | Cloudflared executable.                                                       |

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
TUNNELBROKER_URL=https://tunnelbroker.sam2yolo.workers.dev \
TUNNELBROKER_GROUP=teamA \
TUNNELBROKER_PEER_SECRET=peer-owned-secret \
uvicorn samtoyolo_backend.main:app --host 0.0.0.0 --port 8000
```

In remote mode, the main backend also starts model servers, creates a
Cloudflare Tunnel for each model-server HTTP endpoint, and registers each model
server as a Tunnelbroker peer using `TUNNELBROKER_PEER_SECRET`. Model-server
peer metadata includes `model`, `capabilities`, and `ws_endpoint`.

## Model Server Boundaries

Each heavy model runs in a separate FastAPI model server with its own conda
environment and WebSocket JSON-RPC endpoint. The backend sets up, starts, and
publishes model servers. Model servers own inference/training RPCs, CUDA
runtimes, and model-specific packages.

In v1.0, model servers are intended to run on the same machine or a shared
filesystem with the backend. The backend passes sampled frame directories and
model-checkpoint directories by path. Remote model-server hosts should add a
file-staging layer before inference.

SAM 3.1 ships with:

- setup: `model_servers/sam3/setup.sh`
- run script: `model_servers/sam3/run.sh`
- local endpoint: `ws://127.0.0.1:8101/v1/ws`
- public endpoint: stored in Tunnelbroker metadata as `ws_endpoint`
- inference RPC methods: `inference_sam3`, `sam3.infer_video_text`
- training RPC method: `train_model` (currently reports not implemented for SAM3)

SAM3 model-server inference backends:

| Backend | Selector | Output | When to use |
| --- | --- | --- | --- |
| Video tracker | `"video"` | Video propagation, boxes, optional RLE masks | Larger GPUs where SAM3 video propagation fits in memory. |
| Image processor | `"image"` | Per-frame text segmentation, boxes, RLE masks | Smaller GPUs or random-frame dataset generation. |

SAM3 output modes:

| `output_mode` | Object fields |
| --- | --- |
| `bbox` | `bbox`, `bbox_format` |
| `segmentation` | `segmentation`, `area` |
| `both` | `bbox`, `bbox_format`, `segmentation`, `area` |

Main-backend model server management RPCs:

- `get_model_server_list`
- `get_model_server_status`
- `setup_model_server`
- `restart_model_server`

Model-server implementation points:

- SAM model-server API in
  `samtoyolo_model_servers/sam3/server.py`
- official SAM runtime implementation in
  `samtoyolo_model_servers/sam3/inference.py`
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
