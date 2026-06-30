"""Client + helpers for the SAM 3.1 model-manager backend.

The model manager (port 8000, publicly 163.61.236.112:20000) proxies to the
SAM 3.1 inference server on localhost:8001. SAM accepts IMAGES only, so video
is sampled into frames client-side before upload.

Protocol (WebSocket /ws + HTTP):
    1. init_model {model_name:"sam", base_url:...}  -> model_init_completed
    2. POST /upload (multipart file)                -> {file_id}
    3. create_inference_task {file_ids, file_type:"image", text_prompt, conf, batch} -> task_added
    4. start_inference_from_queue {}
    5. inference_task_chunk_result (chunk_id) ... inference_completed
    6. GET /inference_result?id=<chunk_id>          -> pickle bytes

Pickle (per chunk):
    {job_id, text_prompt, images:[{filename,width,height,
        boxes:[[x1,y1,x2,y2]], scores:[...],
        masks:[{shape:[h,w], encoding:"zlib+base64+packbits", data:b64}]}]}
"""

import base64
import json
import os
import pickle
import zlib
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import websocket


# ----------------------------------------------------------------------------
# Mask decoding + visualization
# ----------------------------------------------------------------------------

def decode_mask(mask_obj):
    """Decode a SAM mask dict into a uint8 HxW binary (0/1) numpy array.

    SAM returns masks with a leading singleton dim, e.g. shape [1, H, W];
    we squeeze to a plain 2D (H, W) array.
    """
    shape = tuple(mask_obj["shape"])
    raw = zlib.decompress(base64.b64decode(mask_obj["data"]))
    packed = np.frombuffer(raw, dtype=np.uint8)
    flat = np.unpackbits(packed)
    n = int(np.prod(shape))
    flat = flat[:n]
    mask = flat.reshape(shape).astype(np.uint8)
    mask = np.squeeze(mask)
    if mask.ndim > 2:          # multiple stacked planes — collapse to 2D
        mask = mask.reshape(mask.shape[-2], mask.shape[-1])
    return mask


# Distinct BGR colors for overlaying multiple detections.
_PALETTE = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
    (255, 128, 0), (128, 255, 0), (0, 255, 128), (255, 0, 128),
]


def overlay_result(image_bgr, image_result, alpha=0.5, draw_boxes=True):
    """Overlay all masks/boxes/scores from one image_result onto a BGR image.

    Returns a new annotated BGR image.
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]
    masks = image_result.get("masks", []) or []
    boxes = image_result.get("boxes", []) or []
    scores = image_result.get("scores", []) or []

    for i, mask_obj in enumerate(masks):
        color = _PALETTE[i % len(_PALETTE)]
        try:
            mask = decode_mask(mask_obj)
        except Exception:
            continue
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        sel = mask.astype(bool)
        if sel.any():
            colored = np.zeros_like(out)
            colored[:] = color
            out[sel] = cv2.addWeighted(out, 1 - alpha, colored, alpha, 0)[sel]
            # draw contour for crisp edges
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, color, 2)

    if draw_boxes:
        for i, box in enumerate(boxes):
            color = _PALETTE[i % len(_PALETTE)]
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{scores[i]:.2f}" if i < len(scores) else ""
            if label:
                cv2.putText(
                    out, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
                )

    n = len(masks)
    cv2.putText(
        out, f"{n} det", (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA,
    )
    cv2.putText(
        out, f"{n} det", (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA,
    )
    return out


# ----------------------------------------------------------------------------
# Video frame sampling
# ----------------------------------------------------------------------------

def sample_frames(video_path, out_dir, target_fps=2.0, max_frames=None, max_side=1024):
    """Sample frames from a video at target_fps. Returns (paths, src_fps, step).

    Frames are downscaled so the longest side <= max_side (SAM resizes every
    image to ~1008px square internally, so this is lossless for inference but
    cuts upload bandwidth ~4x for 1080p sources). Pass max_side=None to keep
    full resolution.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if src_fps <= 0:
        src_fps = 30.0
    step = max(1, round(src_fps / float(target_fps)))
    wanted = set(range(0, total if total > 0 else 10**9, step))

    paths = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            if max_side:
                h, w = frame.shape[:2]
                scale = min(1.0, max_side / max(h, w))
                if scale < 1.0:
                    frame = cv2.resize(frame, (round(w * scale), round(h * scale)))
            p = out_dir / f"frame-{idx:08d}.jpg"
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            paths.append(str(p))
            if max_frames and len(paths) >= max_frames:
                break
        idx += 1
    cap.release()
    return paths, src_fps, step


# ----------------------------------------------------------------------------
# Backend client
# ----------------------------------------------------------------------------

class SamBackendClient:
    """Talks to the model-manager backend for SAM 3.1 inference."""

    def __init__(self, base_url, timeout=3600):
        self.base_url = base_url.rstrip("/")
        parsed = urlparse(self.base_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        self.ws_url = f"{ws_scheme}://{parsed.netloc}/ws"
        self.timeout = timeout
        self.ws = None

    def connect(self):
        self.ws = websocket.create_connection(self.ws_url, timeout=self.timeout)
        return self

    def close(self):
        if self.ws is not None:
            try:
                self.ws.close()
            finally:
                self.ws = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *_):
        self.close()

    def send(self, action, payload=None):
        self.ws.send(json.dumps({"action": action, "payload": payload or {}}))

    def receive(self):
        raw = self.ws.recv()
        if not raw:
            raise ConnectionError("Backend WebSocket closed")
        return json.loads(raw)

    def wait_for(self, actions, on_event=None):
        if isinstance(actions, str):
            actions = {actions}
        while True:
            msg = self.receive()
            if on_event:
                on_event(msg)
            if msg.get("action") in actions:
                return msg

    def init_sam(self, sam_url="http://127.0.0.1:8001", on_event=None):
        self.send("init_model", {"model_name": "sam", "base_url": sam_url})
        res = self.wait_for({"model_init_completed", "model_init_error"}, on_event)
        if res["action"] != "model_init_completed":
            raise RuntimeError(f"SAM init failed: {res}")
        return res["payload"]

    def upload(self, path):
        path = Path(path)
        with path.open("rb") as fh:
            r = requests.post(
                f"{self.base_url}/upload",
                files={"file": (path.name, fh)},
                timeout=300,
            )
        r.raise_for_status()
        return r.json()["file_id"]

    def upload_many(self, paths, workers=8, progress=None):
        """Upload many files concurrently, preserving input order.

        Uploads are latency-bound over the FRP tunnel (~1s RTT each), so
        parallelising gives a near-linear speedup. Returns file_ids in the
        same order as ``paths``.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        paths = list(paths)
        results = [None] * len(paths)
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(self.upload, p): i for i, p in enumerate(paths)}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
                done += 1
                if progress:
                    progress(done, len(paths))
        return results

    def run_inference(self, file_ids, text_prompt, conf=0.5, batch=2, on_event=None,
                      idle_timeout=120):
        """Create + start a task. Calls on_event for every backend message.
        Returns (task_id, [chunk_ids]).

        Robust completion: the backend's ``inference_completed`` event is not
        reliably delivered, so we also treat "all expected chunks received" as
        done. A recv idle-timeout prevents hanging forever (which would wedge
        the server through the FRP tunnel). The caller is expected to close()
        the connection cleanly afterwards.
        """
        import math
        expected_chunks = math.ceil(len(file_ids) / max(1, batch))

        self.send("create_inference_task", {
            "file_ids": file_ids,
            "file_type": "image",
            "text_prompt": text_prompt,
            "conf": conf,
            "batch": batch,
        })
        added = self.wait_for({"task_added", "create_inference_task_error"}, on_event)
        if added["action"] != "task_added":
            raise RuntimeError(f"create task failed: {added}")
        task_id = added["payload"]["id"]

        self.send("start_inference_from_queue", {})
        chunks = []
        self.ws.settimeout(idle_timeout)
        while True:
            try:
                msg = self.receive()
            except (websocket.WebSocketTimeoutError, TimeoutError):
                if chunks:           # got results but no completion event
                    return task_id, chunks
                raise
            if on_event:
                on_event(msg)
            action = msg.get("action")
            data = msg.get("payload", {})
            cand = data.get("task_id") or data.get("id")
            if cand not in (None, task_id):
                continue
            if action == "inference_task_chunk_result":
                chunks.append(data["chunk_id"])
                if len(chunks) >= expected_chunks:
                    # All chunks in. Wait briefly for inference_completed but
                    # don't block forever on it.
                    self.ws.settimeout(8)
                    try:
                        while True:
                            tail = self.receive()
                            if on_event:
                                on_event(tail)
                            if tail.get("action") == "inference_completed":
                                break
                    except Exception:
                        pass
                    return task_id, chunks
            elif action in {"inference_task_error", "task_failed"}:
                raise RuntimeError(f"inference error: {msg}")
            elif action == "inference_completed":
                return task_id, chunks

    def download_chunk(self, chunk_id):
        r = requests.get(
            f"{self.base_url}/inference_result",
            params={"id": chunk_id}, timeout=600,
        )
        r.raise_for_status()
        return pickle.loads(r.content)
