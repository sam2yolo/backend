# Backend API reference

## Base URLs

The main backend listens on port `8000`.

```text
HTTP:      http://HOST:8000
WebSocket: ws://HOST:8000/ws
```

When FRP is used, replace `HOST:8000` with the broker host and allocated remote
port. For example:

```text
HTTP:      http://163.61.236.112:20000
WebSocket: ws://163.61.236.112:20000/ws
```

The internal SAM worker service listens on port `8001`.

## Important behavior

- Only one control WebSocket may be connected at a time.
- Only one model handler is active at a time. Destroy or replace the current
  model before switching between YOLO and SAM.
- Files, tasks, and result registrations are kept in memory. Restarting the
  backend loses their IDs even if files remain on disk.
- Most asynchronous responses include `payload.worker_id`.
- Inference output is stored as Python pickle. Only unpickle data from a trusted
  backend.

## HTTP endpoints

### `GET /`

Returns the bundled test interface.

### `POST /upload`

Uploads one image or video.

Content type:

```text
multipart/form-data
```

Form field:

| Field | Type | Required | Description |
|---|---|---:|---|
| `file` | binary | yes | Image or video file |

Example:

```bash
curl -F "file=@image.jpg" "$BACKEND_URL/upload"
```

Response:

```json
{
  "status": 200,
  "file_id": "4f8aa487-9d5f-45"
}
```

### `GET /inference_result?id={chunk_id}`

Downloads a result chunk as `application/octet-stream`.

```bash
curl -o result.pkl \
  "$BACKEND_URL/inference_result?id=30ebfe0d"
```

Responses:

- `200`: pickle data.
- `404`: `{"error":"File not found"}`.

## WebSocket protocol

Connect to `/ws`. Client messages use:

```json
{
  "action": "action_name",
  "payload": {}
}
```

Server messages generally use the same envelope.

### `ping`

Request:

```json
{"action":"ping","payload":{}}
```

Response:

```json
{"action":"pong","payload":{}}
```

### `list_models`

Request:

```json
{"action":"list_models","payload":{}}
```

Response:

```json
{
  "action": "list_models_response",
  "payload": {"models": ["yolo", "sam"]}
}
```

### `init_model`

YOLO:

```json
{
  "action": "init_model",
  "payload": {
    "model_name": "yolo",
    "variant_name": "yolo11n"
  }
}
```

The handler appends `-seg`, producing `yolo11n-seg`.

SAM:

```json
{
  "action": "init_model",
  "payload": {
    "model_name": "sam",
    "base_url": "http://127.0.0.1:8001"
  }
}
```

Possible events:

- `model_setup_started`
- `model_setup_completed`
- `model_setup_error`
- `model_init_started`
- `model_init_completed`
- `model_init_error`

### `destroy_model`

Request:

```json
{"action":"destroy_model","payload":{}}
```

Events:

- `model_destroyed`
- `model_distroyed` — retained misspelling in the current public protocol.
- `no_model_loaded_error`

### `download_file_wget`

Downloads a directly accessible URL on the backend.

```json
{
  "action": "download_file_wget",
  "payload": {"url": "https://example.com/video.mp4"}
}
```

Events:

- `file_download_initiated`
- `download_progress`
- `file_download_completed`
- `download_failed`

### `download_file_google_drive`

Downloads a Google Drive URL using `gdown`.

```json
{
  "action": "download_file_google_drive",
  "payload": {
    "url": "https://drive.google.com/file/d/FILE_ID/view"
  }
}
```

The response events are the same as `download_file_wget`.

### `list_files`

Request:

```json
{"action":"list_files","payload":{}}
```

Response:

```json
{
  "action": "file_list",
  "payload": {
    "files": [
      {
        "id": "4f8aa487-9d5f-45",
        "path": "files/4f8aa487-9d5f-45",
        "name": "image.jpg"
      }
    ]
  }
}
```

### `delete_file`

```json
{
  "action": "delete_file",
  "payload": {"file_id": "4f8aa487-9d5f-45"}
}
```

Responses:

- `delete_file_success`
- `delete_file_failed`

### `create_inference_task`

Adds an inference task to the active model's local queue.

```json
{
  "action": "create_inference_task",
  "payload": {
    "file_id": "first-file-id",
    "file_ids": ["first-file-id"],
    "file_type": "image"
  }
}
```

Success:

```json
{
  "action": "task_added",
  "payload": {
    "id": "task-id",
    "...": "original task fields"
  }
}
```

Errors include:

- `model_handler_not_loaded_error`
- `create_inference_task_error`
- `create_infrerence_task_error` — retained misspelling in one current path.

See [INFERENCE.md](INFERENCE.md) for model-specific fields.

### `start_inference_from_queue`

```json
{"action":"start_inference_from_queue","payload":{}}
```

Events:

- `work_started`
- `already_working`
- `queue_empty`
- `inference_stage_plus_progress`
- `inference_task_chunk_result`
- `inference_completed`
- `inference_task_error`
- `task_failed`
- `queue_completed`

Each `inference_task_chunk_result` contains a `chunk_id`. Use that ID with
`GET /inference_result`.

### `delete_inference_task`

Deletes a queued task, or marks the current task cancelled.

```json
{
  "action": "delete_inference_task",
  "payload": {"id": "task-id"}
}
```

Events:

- `task_deleted`
- `task_cancelled`
- `task_not_found`
- `task_delete_error`

### `stop_inference_task`

Stops the current inference loop.

```json
{
  "action": "stop_inference_task",
  "payload": {"task_id": "task-id"}
}
```

`task_id` is optional. Events include `task_cancelled`, `no_task_running`, and
`task_stop_error`.

### `fetch_inference_chunk`

Requests task chunk metadata or decoded chunk data over WebSocket.

```json
{
  "action": "fetch_inference_chunk",
  "payload": {
    "task_id": "task-id",
    "chunk_id": "optional-chunk-id"
  }
}
```

Responses:

- `inference_chunk_list`
- `inference_chunk_data`
- `inference_chunks_error`

For large output, prefer `GET /inference_result`. In the current handlers,
chunk registrations are stored by global `chunk_id`, while this action looks
under the task-level entry. Consequently, the HTTP download endpoint using the
`chunk_id` emitted by `inference_task_chunk_result` is the reliable retrieval
path.

### `delete_inference_result`

```json
{
  "action": "delete_inference_result",
  "payload": {"chunk_id": "30ebfe0d"}
}
```

Responses:

- `inference_result_deleted`
- `inference_result_delete_error`

### `delete_inference_result_of_task`

```json
{
  "action": "delete_inference_result_of_task",
  "payload": {"task_id": "task-id"}
}
```

Responses:

- `inference_results_of_task_deleted`
- `inference_result_delete_error`

## Internal SAM service

The main backend's `samHandler` calls this API. External clients normally do not
need it.

### `GET /health`

```json
{
  "status": "ready",
  "port": 8001,
  "gpu_count": 2,
  "ready_workers": 2,
  "queue_size": 0
}
```

### `POST /add_to_infererence_queue`

The endpoint name intentionally retains its current spelling.

Multipart fields:

| Field | Type | Required |
|---|---|---:|
| `images` | one or more image files | yes |
| `text_prompt` | string | yes |
| `confidence_threshold` | float from 0 to 1 | no |

Response status: `202`.

Example:

```bash
curl -X POST http://127.0.0.1:8001/add_to_infererence_queue \
  -F "images=@first.jpg" \
  -F "images=@second.jpg" \
  -F "text_prompt=person" \
  -F "confidence_threshold=0.5"
```

```json
{
  "job_id": "180e976a94064d06b639314096ccd622",
  "status": "queued",
  "websocket_path": "/ws/180e976a94064d06b639314096ccd622"
}
```

### `GET /jobs/{job_id}`

Returns the current job snapshot.

### `WS /ws/{job_id}`

Streams job snapshots until status is `completed` or `failed`.

Statuses:

- `queued`
- `running`
- `completed`
- `failed`
- `not_found`
