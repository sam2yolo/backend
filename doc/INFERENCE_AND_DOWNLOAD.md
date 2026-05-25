# Inference And Download Tutorial

This guide shows the complete production flow for starting the backend from a
GPU notebook, discovering it through Tunnelbroker, running SAM 3.1 inference,
and downloading the generated project artifacts.

The important production rule is: **the client does not need SSH access**. SSH
is only useful for development/debugging. In production, notebook code starts
the backend, the backend creates public Cloudflare Tunnel URLs, announces those
URLs to Tunnelbroker, and the frontend client uses only HTTPS/WSS.

## Architecture

There are two runtime layers:

| Layer | Purpose | Public endpoint |
| --- | --- | --- |
| Main backend | Projects, uploads, task state, model-server setup, downloads, Tunnelbroker registration | `https://...trycloudflare.com` and `wss://.../v1/ws` |
| Model server | Model-specific CUDA runtime for inference/training, such as SAM3 | separate `https://...trycloudflare.com` and `wss://.../v1/ws` |

In remote mode the backend creates Cloudflare tunnels for both layers:

- main backend peer: `SAMTOYOLO_SERVER_NAME`
- SAM3 model-server peer: `SAMTOYOLO_SERVER_NAME-sam3`

The SAM3 peer metadata includes its public WebSocket endpoint as
`metadata.ws_endpoint`.

## Required Values

The notebook startup code must provide:

```bash
SAMTOYOLO_MODE=remote
SAMTOYOLO_SERVER_NAME=<unique-session-name>
TUNNELBROKER_URL=https://tunnelbroker.sam2yolo.workers.dev
TUNNELBROKER_GROUP=<group-name>
TUNNELBROKER_PEER_SECRET=<peer-owned-secret>
TUNNELBROKER_GROUP_TOKEN=<optional-group-token>
```

If `cloudflared` is missing on the notebook image, the backend downloads the
Linux binary to `~/.samtoyolo/bin/cloudflared` automatically.

## Notebook Startup

The production notebook should clone the backend and start it. Example:

```bash
git clone git@github.com:sam2yolo/backend.git samtoyolo-backend
cd samtoyolo-backend

export SAMTOYOLO_MODE=remote
export SAMTOYOLO_SERVER_NAME="gpu-session-001"
export SAMTOYOLO_PROJECT_ROOT="/kaggle/working/projects"
export SAMTOYOLO_PORT=8000

export TUNNELBROKER_URL="https://tunnelbroker.sam2yolo.workers.dev"
export TUNNELBROKER_GROUP="samtoyolo-demo"
export TUNNELBROKER_PEER_SECRET="replace-with-peer-secret"
export TUNNELBROKER_GROUP_TOKEN="replace-with-group-token-if-required"

python -m samtoyolo_backend.run
```

On first run the backend:

1. Installs Miniforge if conda is missing.
2. Creates/reuses the backend conda environment.
3. Installs backend requirements.
4. Starts FastAPI/uvicorn.
5. Sets up and starts the SAM3 model server.
6. Opens Cloudflare Tunnel URLs.
7. Registers the backend and model-server peers in Tunnelbroker.

## Tunnelbroker Discovery

The frontend discovers active peers by calling Tunnelbroker.

```bash
curl -H "Authorization: Bearer replace-with-group-token-if-required" \
  "https://tunnelbroker.sam2yolo.workers.dev/v1/groups/samtoyolo-demo/peers"
```

Example response shape:

```json
{
  "peers": [
    {
      "peer": "gpu-session-001",
      "contacts": [
        {
          "endpoint": "https://main-example.trycloudflare.com",
          "label": "primary",
          "priority": 10
        }
      ],
      "metadata": {
        "app": "samtoyolo-backend",
        "mode": "remote"
      }
    },
    {
      "peer": "gpu-session-001-sam3",
      "contacts": [
        {
          "endpoint": "https://sam3-example.trycloudflare.com",
          "label": "primary",
          "priority": 10
        }
      ],
      "metadata": {
        "app": "samtoyolo-model-server",
        "model": "sam3",
        "display_name": "Meta SAM 3.1",
        "capabilities": ["inference", "training"],
        "ws_endpoint": "wss://sam3-example.trycloudflare.com/v1/ws",
        "main_server": "gpu-session-001"
      }
    }
  ]
}
```

Derived client URLs:

```text
backend_http = https://main-example.trycloudflare.com
backend_ws   = wss://main-example.trycloudflare.com/v1/ws
sam3_ws      = wss://sam3-example.trycloudflare.com/v1/ws
```

## JSON-RPC WebSocket Format

All WebSocket calls use JSON-RPC 2.0:

```json
{
  "jsonrpc": "2.0",
  "id": "request-1",
  "method": "method_name",
  "params": {}
}
```

Responses:

```json
{
  "jsonrpc": "2.0",
  "id": "request-1",
  "result": {}
}
```

Server events are JSON-RPC notifications:

```json
{
  "jsonrpc": "2.0",
  "method": "task_progress",
  "params": {
    "task_id": "task_123",
    "progress": 45.0,
    "message": "processing"
  }
}
```

## Step 1: Create Project

Connect to the main backend WebSocket:

```text
wss://main-example.trycloudflare.com/v1/ws
```

Request:

```json
{
  "jsonrpc": "2.0",
  "id": "create-project-1",
  "method": "create_project",
  "params": {
    "project_id": "cctv-demo",
    "display_name": "CCTV Demo"
  }
}
```

Result:

```json
{
  "jsonrpc": "2.0",
  "id": "create-project-1",
  "result": {
    "project_id": "cctv-demo",
    "display_name": "CCTV Demo"
  }
}
```

Creating or querying a project also binds that WebSocket connection to the
project, so project-specific task events are delivered to the client.

## Step 2: Upload Or Import Video

Use HTTP for large files.

```bash
curl -X POST \
  -F "file=@/path/to/cctv.dav" \
  "https://main-example.trycloudflare.com/v1/projects/cctv-demo/uploads/video"
```

When uploading through Cloudflare Tunnel, keep each HTTP request below the
provider upload limit. For large videos, use the chunked upload endpoints:

```bash
curl -X POST \
  -H "content-type: application/json" \
  -d '{"filename":"cctv.dav","size_bytes":104857600}' \
  "https://main-example.trycloudflare.com/v1/projects/cctv-demo/uploads/video/chunked/init"
```

Upload chunks to:

```text
/v1/projects/cctv-demo/uploads/video/chunked/{upload_id}/chunks/{chunk_index}
```

Then finalize:

```bash
curl -X POST \
  -H "content-type: application/json" \
  -d '{"filename":"cctv.dav","chunk_count":13,"size_bytes":104857600}' \
  "https://main-example.trycloudflare.com/v1/projects/cctv-demo/uploads/video/chunked/{upload_id}/complete"
```

Example result:

```json
{
  "task_id": "task_upload_001",
  "status": "queued",
  "filename": "cctv.dav",
  "path": "uploads/video/cctv.dav",
  "size_bytes": 104857600
}
```

For a public Google Drive video, call the main backend RPC:

```json
{
  "jsonrpc": "2.0",
  "id": "gdrive-upload-1",
  "method": "upload_from_google_drive",
  "params": {
    "project_id": "cctv-demo",
    "url": "https://drive.google.com/file/d/FILE_ID/view?usp=sharing",
    "kind": "video",
    "filename": "cctv.dav"
  }
}
```

Watch for:

```json
{
  "jsonrpc": "2.0",
  "method": "upload_success",
  "params": {
    "task_id": "task_upload_001",
    "filename": "cctv.dav",
    "path": "uploads/video/cctv.dav"
  }
}
```

The `path` value is the project-relative media path used by inference.

## Step 3: Ensure SAM3 Model Server

The main backend can set up and start the SAM3 model server:

```json
{
  "jsonrpc": "2.0",
  "id": "setup-sam3-1",
  "method": "setup_model_server",
  "params": {
    "name": "sam3"
  }
}
```

Example result:

```json
{
  "jsonrpc": "2.0",
  "id": "setup-sam3-1",
  "result": {
    "name": "sam3",
    "display_name": "Meta SAM 3.1",
    "local_ws_url": "ws://127.0.0.1:8101/v1/ws",
    "public_ws_url": "wss://sam3-example.trycloudflare.com/v1/ws",
    "tunnel_registered": true,
    "server_running": true,
    "tunnel_running": true,
    "capabilities": ["inference", "training"]
  }
}
```

The client should use `public_ws_url` in remote mode.

## Step 4: Prepare Inference Parameters

SAM3 accepts text prompts. Each prompt should map to a class name.

For random-frame dataset generation:

```json
{
  "project_id": "cctv-demo",
  "media_path": "uploads/video/cctv.dav",
  "prompts": ["Road", "Building", "Person", "Vehicle", "Dog"],
  "prompt_to_class": {
    "Road": "road",
    "Building": "building",
    "Person": "person",
    "Vehicle": "vehicle",
    "Dog": "dog"
  },
  "sample_strategy": "random",
  "max_frames": 20,
  "random_seed": 20260526,
  "max_frame_width": 640,
  "batch_size": 4,
  "inference_backend": "image",
  "output_mode": "both",
  "include_masks": true,
  "visualize": true,
  "confidence_threshold": 0.35
}
```

Important fields:

| Field | Recommended value | Meaning |
| --- | --- | --- |
| `sample_strategy` | `random` | Pick frames randomly from the video. |
| `max_frames` | `20` | Number of sampled frames. |
| `batch_size` | `4` | Maximum checkpoint batch size. Larger values are capped to `4`. |
| `inference_backend` | `image` | Per-frame SAM3 text segmentation. Usually best for T4/smaller GPUs. |
| `output_mode` | `both` | Produces bounding boxes and segmentation masks. |
| `visualize` | `true` | Adds rendered overlay images in `visualizations/`. |

`output_mode` options:

| Value | Output fields |
| --- | --- |
| `bbox` | `bbox`, `bbox_format` |
| `segmentation` | `segmentation`, `area` |
| `both` | all bbox and segmentation fields |

## Step 5: Start Inference

There are two ways to start SAM3 inference.

### Backend Compatibility Call

Call `inference_sam3` on the main backend. This ensures the SAM3 model server is
available and returns the model-server endpoint/method to use.

```json
{
  "jsonrpc": "2.0",
  "id": "inference-discovery-1",
  "method": "inference_sam3",
  "params": {
    "project_id": "cctv-demo",
    "media_path": "uploads/video/cctv.dav",
    "prompts": ["Road", "Building", "Person", "Vehicle", "Dog"],
    "prompt_to_class": {
      "Road": "road",
      "Building": "building",
      "Person": "person",
      "Vehicle": "vehicle",
      "Dog": "dog"
    },
    "sample_strategy": "random",
    "max_frames": 20,
    "batch_size": 4,
    "output_mode": "both",
    "include_masks": true,
    "visualize": true
  }
}
```

Example result:

```json
{
  "jsonrpc": "2.0",
  "id": "inference-discovery-1",
  "result": {
    "delegated": true,
    "model": "sam3",
    "model_server": {
      "public_ws_url": "wss://sam3-example.trycloudflare.com/v1/ws",
      "server_running": true,
      "tunnel_registered": true
    },
    "rpc_method": "inference_sam3",
    "compat_rpc_method": "sam3.infer_video_text"
  }
}
```

### Model-Server Inference Call

Connect to the returned SAM3 WebSocket:

```text
wss://sam3-example.trycloudflare.com/v1/ws
```

Then call:

```json
{
  "jsonrpc": "2.0",
  "id": "sam3-run-1",
  "method": "sam3.infer_video_text",
  "params": {
    "frames_dir": "/kaggle/working/projects/cctv-demo/inference_results/task_001/frames",
    "frames": [
      "/kaggle/working/projects/cctv-demo/inference_results/task_001/frames/frame_00000000.jpg"
    ],
    "prompts": ["Road", "Building", "Person", "Vehicle", "Dog"],
    "prompt_to_class": {
      "Road": "road",
      "Building": "building",
      "Person": "person",
      "Vehicle": "vehicle",
      "Dog": "dog"
    },
    "model_extract_dir": "/kaggle/working/projects/_models/sam3/extracted/sam3.1",
    "gpu_index": 0,
    "inference_backend": "image",
    "output_mode": "both",
    "include_masks": true,
    "output_prob_thresh": 0.35,
    "cache_model": true
  }
}
```

The model-server RPC is intentionally low-level: it expects paths on the server
machine. A frontend should normally let the main backend stage media/frames and
return those paths, then call the model server with those returned paths. This
keeps the client SSH-free while still letting the model server run in its own
environment.

Progress notifications from the model server:

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

Final model-server result:

```json
{
  "jsonrpc": "2.0",
  "id": "sam3-run-1",
  "result": {
    "frames": [
      {
        "file_name": "frame_00000000.jpg",
        "width": 640,
        "height": 360,
        "objects": [
          {
            "object_id": "frame0_prompt0_0",
            "prompt": "Road",
            "class_name": "road",
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
    ],
    "metadata": {
      "sam3_backend": "official_sam3_image_processor",
      "sam3_real_inference": true,
      "sam3_output_mode": "both"
    }
  }
}
```

## Step 6: Download Inference Result

When inference is packaged by the backend task pipeline, it emits:

```json
{
  "jsonrpc": "2.0",
  "method": "inference_result_ready",
  "params": {
    "task_id": "task_infer_001",
    "file_path_or_url": "https://main-example.trycloudflare.com/v1/projects/cctv-demo/downloads/inference/task_infer_001"
  }
}
```

You can also ask for the download URL:

```json
{
  "jsonrpc": "2.0",
  "id": "download-inference-1",
  "method": "download_inference_result",
  "params": {
    "task_id": "task_infer_001",
    "delete_after_download": false
  }
}
```

Example result:

```json
{
  "jsonrpc": "2.0",
  "id": "download-inference-1",
  "result": {
    "project_id": "cctv-demo",
    "task_id": "task_infer_001",
    "download_url": "https://main-example.trycloudflare.com/v1/projects/cctv-demo/downloads/inference/task_infer_001",
    "relative_download_url": "/v1/projects/cctv-demo/downloads/inference/task_infer_001"
  }
}
```

Download:

```bash
curl -L -OJ \
  "https://main-example.trycloudflare.com/v1/projects/cctv-demo/downloads/inference/task_infer_001"
```

To delete the server-side zip after a successful download:

```bash
curl -L -OJ \
  "https://main-example.trycloudflare.com/v1/projects/cctv-demo/downloads/inference/task_infer_001?delete_after_download=true"
```

## Result Zip Layout

The inference zip contains:

```text
inference_results/task_infer_001/
  frames/
    frame_00000000.jpg
    frame_00000001.jpg
  annotations.json
  metadata.json
  checkpoints/
    batch_000001.json
    batch_000002.json
  visualizations/
    frame_00000000.jpg
    frame_00000001.jpg
```

`annotations.json` contains one row per sampled frame:

```json
{
  "version": "samtoyolo.inference.v1",
  "model_name": "sam3",
  "frames": [
    {
      "file_name": "frame_00000000.jpg",
      "width": 640,
      "height": 360,
      "objects": []
    }
  ]
}
```

`metadata.json` stores the inference parameters, model artifact info, GPU
worker, sampled-frame strategy, and SAM3 runtime metadata.

## Download Project Metadata

Use `get_project_info` to list current uploads, datasets, models, and inference
results:

```json
{
  "jsonrpc": "2.0",
  "id": "project-info-1",
  "method": "get_project_info",
  "params": {
    "project_id": "cctv-demo"
  }
}
```

Use `get_task_list` to show all project tasks:

```json
{
  "jsonrpc": "2.0",
  "id": "task-list-1",
  "method": "get_task_list",
  "params": {
    "project_id": "cctv-demo"
  }
}
```

For generated YOLO datasets:

```json
{
  "jsonrpc": "2.0",
  "id": "download-dataset-1",
  "method": "download_dataset",
  "params": {
    "project_id": "cctv-demo",
    "dataset_id": "dataset_001"
  }
}
```

For trained models:

```json
{
  "jsonrpc": "2.0",
  "id": "download-model-1",
  "method": "download_model",
  "params": {
    "project_id": "cctv-demo",
    "model_id": "model_001"
  }
}
```

Both methods return public `download_url` values when the backend tunnel is
active.

## Complete Python Client Example

This example performs discovery, creates a project, asks the backend to import a
Google Drive video, waits for upload completion, ensures SAM3 is running, and
prints the public endpoints needed for inference/download.

```python
import asyncio
import json
from itertools import count

import httpx
import websockets


TUNNELBROKER_URL = "https://tunnelbroker.sam2yolo.workers.dev"
GROUP = "samtoyolo-demo"
GROUP_TOKEN = "replace-with-group-token-if-required"
PROJECT_ID = "cctv-demo"
GDRIVE_VIDEO_URL = "https://drive.google.com/file/d/FILE_ID/view?usp=sharing"


async def rpc(ws, method, params=None, request_id=None):
    request_id = request_id or method
    await ws.send(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }
        )
    )
    while True:
        message = json.loads(await ws.recv())
        if message.get("id") == request_id:
            if "error" in message:
                raise RuntimeError(message["error"])
            return message["result"]
        print("event:", message)


async def discover():
    headers = {}
    if GROUP_TOKEN:
        headers["Authorization"] = f"Bearer {GROUP_TOKEN}"
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            f"{TUNNELBROKER_URL.rstrip('/')}/v1/groups/{GROUP}/peers",
            headers=headers,
        )
        response.raise_for_status()
        peers = response.json()["peers"]

    backend = next(
        peer for peer in peers if peer.get("metadata", {}).get("app") == "samtoyolo-backend"
    )
    sam3 = next(
        peer
        for peer in peers
        if peer.get("metadata", {}).get("app") == "samtoyolo-model-server"
        and peer.get("metadata", {}).get("model") == "sam3"
    )
    backend_http = backend["contacts"][0]["endpoint"].rstrip("/")
    backend_ws = backend_http.replace("https://", "wss://") + "/v1/ws"
    sam3_ws = sam3["metadata"]["ws_endpoint"]
    return backend_http, backend_ws, sam3_ws


async def main():
    backend_http, backend_ws, sam3_ws = await discover()
    print("backend:", backend_http)
    print("backend ws:", backend_ws)
    print("sam3 ws:", sam3_ws)

    async with websockets.connect(backend_ws, max_size=None) as ws:
        await rpc(
            ws,
            "create_project",
            {"project_id": PROJECT_ID, "display_name": "CCTV Demo"},
            "create-project",
        )

        upload = await rpc(
            ws,
            "upload_from_google_drive",
            {
                "project_id": PROJECT_ID,
                "url": GDRIVE_VIDEO_URL,
                "kind": "video",
                "filename": "cctv.dav",
            },
            "upload-video",
        )
        print("upload task:", upload["task_id"])

        setup = await rpc(ws, "setup_model_server", {"name": "sam3"}, "setup-sam3")
        print("sam3 public ws:", setup["public_ws_url"])

        discovery = await rpc(
            ws,
            "inference_sam3",
            {
                "project_id": PROJECT_ID,
                "media_path": "uploads/video/cctv.dav",
                "prompts": ["Road", "Building", "Person", "Vehicle", "Dog"],
                "prompt_to_class": {
                    "Road": "road",
                    "Building": "building",
                    "Person": "person",
                    "Vehicle": "vehicle",
                    "Dog": "dog",
                },
                "sample_strategy": "random",
                "max_frames": 20,
                "batch_size": 4,
                "output_mode": "both",
                "include_masks": True,
                "visualize": True,
                "inference_backend": "image",
            },
            "inference-discovery",
        )
        print("call model server:", discovery["model_server"]["public_ws_url"])


asyncio.run(main())
```

## Troubleshooting

### No peers appear in Tunnelbroker

Check the notebook startup environment:

```bash
echo "$SAMTOYOLO_MODE"
echo "$TUNNELBROKER_URL"
echo "$TUNNELBROKER_GROUP"
```

`SAMTOYOLO_MODE` must be `remote`, and the Tunnelbroker URL/group/peer secret
must be set.

### Peer exists but client cannot connect

Check `get_tunnel_status` on the backend:

```json
{
  "jsonrpc": "2.0",
  "id": "tunnel-status-1",
  "method": "get_tunnel_status",
  "params": {}
}
```

Expected:

```json
{
  "registered": true,
  "cloudflared_running": true,
  "endpoint": "https://main-example.trycloudflare.com"
}
```

Check SAM3:

```json
{
  "jsonrpc": "2.0",
  "id": "sam3-status-1",
  "method": "get_model_server_status",
  "params": {
    "name": "sam3"
  }
}
```

Expected:

```json
{
  "server_running": true,
  "tunnel_running": true,
  "tunnel_registered": true,
  "public_ws_url": "wss://sam3-example.trycloudflare.com/v1/ws"
}
```

### Download URL is relative

This means the backend has no active public base URL. Start it with
`SAMTOYOLO_MODE=remote`, or set:

```bash
SAMTOYOLO_PUBLIC_BASE_URL=https://main-example.trycloudflare.com
```

### Model server cannot see the files

In v1.0 the backend and model server are expected to run on the same machine or
shared filesystem. The model-server RPC receives server-side file paths. If the
model server runs on a different machine, add file staging before calling the
model-server RPC.

## Minimal Checklist

1. Start backend in remote mode from notebook.
2. Backend registers itself and SAM3 to Tunnelbroker.
3. Client discovers backend and SAM3 peers.
4. Client opens backend WSS.
5. Client creates project.
6. Client uploads/imports video.
7. Client ensures SAM3 model server is running.
8. Client starts SAM3 inference with `output_mode="both"` and `visualize=true`.
9. Client watches progress events.
10. Client downloads the inference zip from the public backend URL.
