"""Realtime SAM 3.1 testing app.

A local FastAPI server that drives the model-manager backend for SAM 3.1
inference and streams annotated results to the browser in realtime.

Pipeline (run in a background thread per job):
    image:  upload -> infer -> download chunk -> decode masks -> render -> stream
    video:  sample frames @ fps -> (same as above, batched)

Run:
    cd sam_test_app
    python server.py            # serves http://127.0.0.1:7860
"""

import base64
import queue
import threading
import time
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from sam_client import SamBackendClient, overlay_result, sample_frames

APP_DIR = Path(__file__).parent
WORK_DIR = APP_DIR / "work"
WORK_DIR.mkdir(exist_ok=True)

DEFAULT_BACKEND = "http://163.61.236.112:20000"
DEFAULT_SAM = "http://127.0.0.1:8001"

app = FastAPI(title="SAM 3.1 Tester")

# job_id -> {"q": Queue, "params": {...}, "upload": Path}
JOBS = {}


def _emit(job_id, event, **data):
    job = JOBS.get(job_id)
    if job:
        job["q"].put({"event": event, **data})


def _encode_jpg(image_bgr, max_side=900):
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("ascii") if ok else None


def run_job(job_id):
    job = JOBS[job_id]
    p = job["params"]
    upload_path = job["upload"]
    backend_url = p["backend_url"]
    sam_url = p["sam_url"]
    prompt = p["prompt"]
    conf = p["conf"]
    batch = p["batch"]
    mode = p["mode"]
    fps = p["fps"]
    max_frames = p.get("max_frames")

    t0 = time.time()
    try:
        # 1. Prepare frame list
        if mode == "video":
            _emit(job_id, "status", message=f"Sampling video at {fps} fps…")
            frame_dir = WORK_DIR / job_id / "frames"
            paths, src_fps, step = sample_frames(
                upload_path, frame_dir, target_fps=fps, max_frames=max_frames
            )
            _emit(job_id, "status",
                  message=f"Sampled {len(paths)} frames (source {src_fps:.1f} fps, every {step} frames)")
        else:
            paths = [str(upload_path)]

        if not paths:
            _emit(job_id, "error", message="No frames to process")
            return

        _emit(job_id, "meta", total=len(paths), prompt=prompt, conf=conf, batch=batch, mode=mode)

        # 2. Connect + init SAM
        client = SamBackendClient(backend_url)
        client.connect()
        try:
            _emit(job_id, "status", message="Initializing SAM 3.1 on backend…")
            info = client.init_sam(sam_url)
            _emit(job_id, "status", message=f"SAM ready: {info.get('server')}")

            # 3. Upload all frames (concurrent — uploads are latency-bound)
            _emit(job_id, "status", message=f"Uploading {len(paths)} image(s)…")
            file_ids = client.upload_many(
                paths, workers=8,
                progress=lambda d, t: _emit(job_id, "upload_progress", done=d, total=t),
            )

            # 4. Run inference; download+render each chunk as it arrives
            chunk_meta = {}   # chunk_index -> start offset

            def on_event(msg):
                action = msg.get("action")
                data = msg.get("payload", {})
                if action == "inference_stage_plus_progress":
                    _emit(job_id, "status",
                          message=f"{data.get('stage')} ({data.get('progress')}%)")
                elif action == "inference_task_chunk_result":
                    chunk_id = data["chunk_id"]
                    chunk_index = data.get("chunk_index", 0)
                    start = chunk_index * batch
                    try:
                        result = client.download_chunk(chunk_id)
                    except Exception as exc:
                        _emit(job_id, "status", message=f"chunk download failed: {exc}")
                        return
                    for j, img_res in enumerate(result.get("images", [])):
                        global_idx = start + j
                        if global_idx >= len(paths):
                            break
                        src = cv2.imread(paths[global_idx])
                        if src is None:
                            continue
                        annotated = overlay_result(src, img_res)
                        b64 = _encode_jpg(annotated)
                        ndet = len(img_res.get("masks", []) or [])
                        # also persist
                        out_dir = WORK_DIR / job_id / "out"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(out_dir / f"{global_idx:05d}_det{ndet}.jpg"), annotated)
                        _emit(job_id, "frame",
                              index=global_idx, total=len(paths),
                              detections=ndet,
                              scores=[round(float(s), 3) for s in img_res.get("scores", [])],
                              image=b64)

            task_id, chunks = client.run_inference(
                file_ids, prompt, conf=conf, batch=batch, on_event=on_event
            )
            elapsed = time.time() - t0
            _emit(job_id, "done", task_id=task_id, chunks=len(chunks),
                  frames=len(paths), seconds=round(elapsed, 1))
        finally:
            client.close()
    except Exception as exc:
        import traceback
        _emit(job_id, "error", message=f"{type(exc).__name__}: {exc}",
              trace=traceback.format_exc())


@app.get("/")
async def index():
    return HTMLResponse((APP_DIR / "static" / "index.html").read_text())


@app.post("/api/start")
async def start(
    file: UploadFile = File(...),
    mode: str = Form("image"),
    prompt: str = Form(...),
    fps: float = Form(2.0),
    conf: float = Form(0.5),
    batch: int = Form(4),
    max_frames: int = Form(0),
    backend_url: str = Form(DEFAULT_BACKEND),
    sam_url: str = Form(DEFAULT_SAM),
):
    job_id = uuid.uuid4().hex[:12]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix or (".mp4" if mode == "video" else ".jpg")
    upload_path = job_dir / f"input{suffix}"
    upload_path.write_bytes(await file.read())

    JOBS[job_id] = {
        "q": queue.Queue(),
        "upload": upload_path,
        "params": {
            "mode": mode, "prompt": prompt, "fps": fps, "conf": conf,
            "batch": max(1, batch), "max_frames": max_frames or None,
            "backend_url": backend_url.rstrip("/"), "sam_url": sam_url.rstrip("/"),
        },
    }
    threading.Thread(target=run_job, args=(job_id,), daemon=True).start()
    return {"job_id": job_id}


@app.websocket("/api/ws/{job_id}")
async def ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    job = JOBS.get(job_id)
    if not job:
        await websocket.send_json({"event": "error", "message": "unknown job"})
        await websocket.close()
        return
    q = job["q"]
    try:
        while True:
            try:
                msg = q.get(timeout=0.25)
            except queue.Empty:
                await websocket.send_json({"event": "ping"})
                continue
            await websocket.send_json(msg)
            if msg["event"] in ("done", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")
