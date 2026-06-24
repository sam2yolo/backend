# Inference guide

## Shared workflow

1. Upload or remotely download input files.
2. Open `/ws`.
3. Initialize one model.
4. Send `create_inference_task`.
5. Wait for `task_added` and save its `id`.
6. Send `start_inference_from_queue`.
7. Collect every `chunk_id` from `inference_task_chunk_result`.
8. Wait for `inference_completed`.
9. Download each chunk with `GET /inference_result?id=...`.

## YOLO segmentation

Initialize:

```json
{
  "action": "init_model",
  "payload": {
    "model_name": "yolo",
    "variant_name": "yolo11n"
  }
}
```

### Single image

The current YOLO handler validates `file_id` and reads images from `file_ids`,
so include both:

```json
{
  "action": "create_inference_task",
  "payload": {
    "file_id": "image-id",
    "file_ids": ["image-id"],
    "file_type": "image",
    "batch": 1,
    "conf": 0.25,
    "iou": 0.45,
    "imgsz": 640
  }
}
```

### Image batch

```json
{
  "action": "create_inference_task",
  "payload": {
    "file_id": "first-image-id",
    "file_ids": [
      "first-image-id",
      "second-image-id",
      "third-image-id"
    ],
    "file_type": "image",
    "batch": 16,
    "conf": 0.25,
    "iou": 0.45,
    "classes": [0, 2]
  }
}
```

YOLO image fields:

| Field | Default | Description |
|---|---:|---|
| `file_id` | required | First image ID; currently required for validation |
| `file_ids` | required | Images to infer |
| `file_type` | required | `"image"` |
| `batch` | `128` | Images per output chunk |
| `conf` | `0.25` | Confidence threshold |
| `iou` | `0.45` | NMS IoU threshold |
| `imgsz` | model default | Inference size |
| `classes` | all | Optional class ID list |

### Video

```json
{
  "action": "create_inference_task",
  "payload": {
    "file_id": "video-id",
    "file_type": "video",
    "frames": [0, 900, 1800, 2700],
    "batch": 16,
    "conf": 0.25,
    "iou": 0.45
  }
}
```

Video fields:

| Field | Default | Description |
|---|---:|---|
| `file_id` | required | Uploaded or downloaded video |
| `file_type` | required | `"video"` |
| `frames` | all frames | Explicit zero-based frame indexes |
| `batch` | `128` | Frames per chunk |
| `temporal_downsampling` | false | Enables legacy drop-rate sampling |
| `drop_rate` | `0.01` | Fraction used by legacy sampling |
| `conf`, `iou`, `imgsz`, `classes` | see image table | YOLO prediction options |

For predictable time sampling, calculate explicit frame indexes:

```python
interval_frames = round(video_fps * seconds_between_frames)
frames = list(range(0, total_frames, interval_frames))
```

### YOLO result format

Each downloaded pickle is a list of `ultralytics.engine.results.Results`
objects. Install a compatible `ultralytics` package before unpickling:

```python
import pickle

with open("result.pkl", "rb") as stream:
    results = pickle.load(stream)

for result in results:
    print(result.boxes.xyxy)
    print(result.boxes.conf)
    print(result.boxes.cls)
    if result.masks is not None:
        print(result.masks.data)
```

## SAM3 text-prompted segmentation

Initialize:

```json
{
  "action": "init_model",
  "payload": {
    "model_name": "sam",
    "base_url": "http://127.0.0.1:8001"
  }
}
```

### Single image

```json
{
  "action": "create_inference_task",
  "payload": {
    "file_ids": ["image-id"],
    "file_type": "image",
    "text_prompt": "person",
    "conf": 0.5,
    "batch": 1
  }
}
```

### Image batch

```json
{
  "action": "create_inference_task",
  "payload": {
    "file_ids": [
      "first-image-id",
      "second-image-id",
      "third-image-id"
    ],
    "file_type": "image",
    "text_prompt": "vehicle",
    "conf": 0.5,
    "batch": 2
  }
}
```

SAM fields:

| Field | Default | Description |
|---|---:|---|
| `file_ids` | required | Images to infer |
| `file_id` | none | Accepted as a single-image alias |
| `file_type` | `"image"` | SAM currently accepts images only |
| `text_prompt` | required | Object description |
| `conf` | `0.5` | Detection threshold |
| `batch` | `2` | Images sent to one GPU worker job |

Keep SAM batches small because images are resized to `1008×1008`. A batch of
`1` or `2` is recommended on a 16 GB T4.

### Video with SAM

SAM does not accept a raw video task. Extract frames, upload them as images, and
submit an image batch. The example `sam_video.py` does this automatically.

For one frame every 30 seconds:

```python
interval_frames = round(fps * 30)
frame_indexes = range(0, total_frames, interval_frames)
```

### SAM result format

Each pickle contains:

```python
{
    "job_id": "...",
    "text_prompt": "person",
    "gpu_id": 0,
    "images": [
        {
            "filename": "...",
            "width": 1920,
            "height": 1080,
            "boxes": [[x0, y0, x1, y1]],
            "scores": [0.91],
            "masks": [
                {
                    "shape": [1, 1080, 1920],
                    "encoding": "zlib+base64+packbits",
                    "data": "..."
                }
            ]
        }
    ]
}
```

Decode a mask:

```python
import base64
import zlib
import numpy as np

def decode_mask(encoded):
    shape = tuple(encoded["shape"])
    packed = zlib.decompress(base64.b64decode(encoded["data"]))
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
    return bits[:np.prod(shape)].reshape(shape).astype(bool)
```

An empty `boxes`, `scores`, and `masks` list is a valid result when no object
matches the prompt and threshold.

## Security

Python pickle can execute code during deserialization. Never unpickle output
from an untrusted or impersonated backend.
