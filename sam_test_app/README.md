# SAM 3.1 Testing App

A standalone tool for exercising the **SAM 3.1** text-prompted segmentation
model served by the model-manager backend, with a realtime web UI plus a CLI.

SAM runs on the backend (model manager on port `8000`, publicly
`http://163.61.236.112:20000`), which proxies to the SAM 3.1 inference server on
`localhost:8001`. **SAM accepts images only** — video is sampled into frames
client-side at the chosen FPS, then each frame is segmented.

## Contents

| File | Purpose |
|------|---------|
| `sam_client.py` | Backend protocol client + mask decode + overlay rendering + video frame sampler |
| `server.py` + `static/index.html` | Realtime web app — upload image/video, stream annotated results live |
| `smoke_test.py` | CLI end-to-end test (image or video) that saves annotated frames |
| `assemble_video.py` | Stitch annotated frames into an mp4 |

## Install

```bash
pip install -r requirements.txt
```

## Realtime web app

```bash
python server.py          # http://127.0.0.1:7860
```

Open the page, pick **Image** or **Video**, set a text prompt (e.g. `person`,
`car`, `bus`), choose sampling FPS for video, and hit **Run**. Annotated frames
stream into the page as each batch completes.

## CLI

```bash
# image
python smoke_test.py image path/to.jpg --prompt car

# video @ 2 fps
python smoke_test.py video path/to.mp4 --prompt person --fps 2 --batch 8

# stitch annotated frames into a video
python assemble_video.py outputs/video --fps 6
```

Annotated frames are written to `outputs/{image,video}/`.

## Protocol (for reference)

1. WS `init_model` `{model_name:"sam", base_url:"http://127.0.0.1:8001"}` → `model_init_completed`
2. HTTP `POST /upload` (multipart `file`) → `{file_id}`
3. WS `create_inference_task` `{file_ids, file_type:"image", text_prompt, conf, batch}` → `task_added`
4. WS `start_inference_from_queue` `{}`
5. `inference_task_chunk_result` (per batch) … `inference_completed`
6. HTTP `GET /inference_result?id=<chunk_id>` → pickle

Pickle per chunk:
```
{job_id, text_prompt, images:[
   {filename, width, height,
    boxes:[[x1,y1,x2,y2]], scores:[...],
    masks:[{shape:[1,H,W], encoding:"zlib+base64+packbits", data:b64}]}]}
```
Decode a mask: `base64decode → zlib.decompress → np.frombuffer(uint8) →
np.unpackbits → reshape(shape) → squeeze` to 2D `(H, W)`.

## Notes

- The client does **not** depend on `inference_completed` to finish; it also
  completes when all expected chunks (`ceil(frames/batch)`) have arrived, and
  always closes the WebSocket cleanly. This avoids hanging — a hung client
  behind the FRP tunnel can wedge the backend's event loop.
- Only **one** WebSocket connection to the backend is allowed at a time.
