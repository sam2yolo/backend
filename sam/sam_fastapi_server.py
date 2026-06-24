import argparse
import asyncio
import base64
import multiprocessing as mp
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zlib
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket


PORT = 8001
PROJECT_DIR = Path(os.environ.get("SAM3_PROJECT_DIR", "/root/sam3")).resolve()
CHECKPOINT_PATH = Path(
    os.environ.get(
        "SAM3_CHECKPOINT_PATH",
        PROJECT_DIR / "hf/sam3.1_multiplex_mapped.pt",
    )
).resolve()
UPLOAD_ROOT = Path(
    os.environ.get("SAM3_UPLOAD_DIR", tempfile.gettempdir())
) / "sam3-api"

app = FastAPI(title="SAM3 inference server")
jobs = {}
jobs_lock = threading.Lock()
task_queue = None
result_queue = None
workers = []
collector_thread = None
cloudflared_process = None


def encode_mask(mask):
    mask = mask.detach().to("cpu").numpy().astype("uint8", copy=False)
    packed = __import__("numpy").packbits(mask.reshape(-1))
    return {
        "shape": list(mask.shape),
        "encoding": "zlib+base64+packbits",
        "data": base64.b64encode(zlib.compress(packed.tobytes())).decode("ascii"),
    }


def build_datapoint(image, prompt, query_id):
    from sam3.train.data.sam3_image_dataset import (
        Datapoint,
        FindQueryLoaded,
        Image as SAMImage,
        InferenceMetadata,
    )

    width, height = image.size
    return Datapoint(
        find_queries=[
            FindQueryLoaded(
                query_text=prompt,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=query_id,
                    original_image_id=query_id,
                    original_category_id=1,
                    original_size=[height, width],
                    object_id=0,
                    frame_index=0,
                ),
            )
        ],
        images=[SAMImage(data=image, objects=[], size=[height, width])],
    )


def gpu_worker(gpu_id, incoming, outgoing):
    os.chdir(PROJECT_DIR)
    torch.cuda.set_device(gpu_id)

    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.eval.postprocessors import PostProcessImage
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api as collate
    from sam3.train.transforms.basic_for_api import (
        ComposeAPI,
        NormalizeAPI,
        RandomResizeAPI,
        ToTensorAPI,
    )

    device = torch.device(f"cuda:{gpu_id}")
    model = build_sam3_image_model(checkpoint_path=CHECKPOINT_PATH)
    model = model.to(device).eval()
    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=1008,
                max_size=1008,
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    outgoing.put({"kind": "worker_ready", "gpu_id": gpu_id})

    while True:
        task = incoming.get()
        if task is None:
            return

        job_id = task["job_id"]
        outgoing.put({"kind": "running", "job_id": job_id, "gpu_id": gpu_id})
        try:
            datapoints = []
            metadata = []
            for query_id, image_info in enumerate(task["images"], start=1):
                with Image.open(image_info["path"]) as image:
                    rgb_image = image.convert("RGB")
                    metadata.append(
                        {
                            "filename": image_info["filename"],
                            "width": rgb_image.width,
                            "height": rgb_image.height,
                        }
                    )
                    datapoints.append(
                        transform(
                            build_datapoint(
                                rgb_image,
                                task["text_prompt"],
                                query_id,
                            )
                        )
                    )

            batch = collate(datapoints, dict_key="batch")["batch"]
            batch = copy_data_to_device(batch, device, non_blocking=True)
            postprocessor = PostProcessImage(
                max_dets_per_img=-1,
                iou_type="segm",
                use_original_sizes_box=True,
                use_original_sizes_mask=True,
                convert_mask_to_rle=False,
                detection_threshold=task["confidence_threshold"],
                to_cpu=False,
            )

            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.float16
            ):
                output = model(batch)

            processed = postprocessor.process_results(
                output, batch.find_metadatas
            )
            image_results = []
            for query_id, image_info in enumerate(metadata, start=1):
                result = processed[query_id]
                image_results.append(
                    {
                        **image_info,
                        "boxes": result["boxes"].detach().cpu().tolist(),
                        "scores": result["scores"].detach().float().cpu().tolist(),
                        "masks": [
                            encode_mask(mask)
                            for mask in result["masks"]
                        ],
                    }
                )

            outgoing.put(
                {
                    "kind": "completed",
                    "job_id": job_id,
                    "gpu_id": gpu_id,
                    "result": {
                        "job_id": job_id,
                        "text_prompt": task["text_prompt"],
                        "gpu_id": gpu_id,
                        "images": image_results,
                    },
                }
            )
        except Exception as exc:
            outgoing.put(
                {
                    "kind": "failed",
                    "job_id": job_id,
                    "gpu_id": gpu_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            shutil.rmtree(task["upload_dir"], ignore_errors=True)


def collect_worker_messages():
    while True:
        message = result_queue.get()
        if message is None:
            return
        kind = message["kind"]
        if kind == "worker_ready":
            with jobs_lock:
                jobs.setdefault("_workers", {})[str(message["gpu_id"])] = "ready"
            continue

        job_id = message["job_id"]
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                continue
            job["status"] = kind
            job["gpu_id"] = message.get("gpu_id")
            job["updated_at"] = time.time()
            if kind == "completed":
                job["result"] = message["result"]
            elif kind == "failed":
                job["error"] = message["error"]


def job_snapshot(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        return dict(job) if job else None


def start_cloudflare_tunnel():
    global cloudflared_process
    cloudflared = shutil.which("cloudflared")
    if cloudflared is None:
        raise RuntimeError(
            "cloudflared is not installed; install it or run without --tunnel"
        )
    cloudflared_process = subprocess.Popen(
        [
            cloudflared,
            "tunnel",
            "--no-autoupdate",
            "--url",
            f"http://127.0.0.1:{PORT}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def print_tunnel_output():
        url_pattern = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")
        for line in cloudflared_process.stdout:
            print(line, end="", flush=True)
            match = url_pattern.search(line)
            if match:
                print(
                    f"SAM_SERVER_URL={match.group(0)}",
                    flush=True,
                )

    threading.Thread(target=print_tunnel_output, daemon=True).start()


@app.on_event("startup")
def startup():
    global task_queue, result_queue, workers, collector_thread
    if not CHECKPOINT_PATH.is_file():
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("No CUDA GPUs were found")

    for gpu_id in range(gpu_count):
        process = ctx.Process(
            target=gpu_worker,
            args=(gpu_id, task_queue, result_queue),
            daemon=True,
        )
        process.start()
        workers.append(process)

    collector_thread = threading.Thread(
        target=collect_worker_messages,
        daemon=True,
    )
    collector_thread.start()


@app.on_event("shutdown")
def shutdown():
    for _ in workers:
        task_queue.put(None)
    result_queue.put(None)
    for process in workers:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
    if cloudflared_process is not None:
        cloudflared_process.terminate()


@app.get("/health")
def health():
    with jobs_lock:
        ready_workers = len(jobs.get("_workers", {}))
    return {
        "status": "ready" if ready_workers == len(workers) else "loading",
        "port": PORT,
        "gpu_count": len(workers),
        "ready_workers": ready_workers,
        "queue_size": task_queue.qsize() if task_queue is not None else 0,
    }


@app.post("/add_to_infererence_queue", status_code=202)
async def add_to_infererence_queue(
    images: list[UploadFile] = File(...),
    text_prompt: str = Form(...),
    confidence_threshold: float = Form(0.5),
):
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required")
    if not text_prompt.strip():
        raise HTTPException(status_code=400, detail="text_prompt is required")
    if not 0 <= confidence_threshold <= 1:
        raise HTTPException(
            status_code=400,
            detail="confidence_threshold must be between 0 and 1",
        )

    job_id = uuid.uuid4().hex
    upload_dir = UPLOAD_ROOT / job_id
    upload_dir.mkdir(parents=True)
    saved_images = []
    try:
        for index, upload in enumerate(images):
            suffix = Path(upload.filename or "").suffix or ".jpg"
            path = upload_dir / f"{index:06d}{suffix}"
            with path.open("wb") as destination:
                while chunk := await upload.read(1024 * 1024):
                    destination.write(chunk)
            saved_images.append(
                {
                    "filename": upload.filename or path.name,
                    "path": str(path),
                }
            )

        now = time.time()
        with jobs_lock:
            jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "gpu_id": None,
                "created_at": now,
                "updated_at": now,
                "image_count": len(saved_images),
            }
        task_queue.put_nowait(
            {
                "job_id": job_id,
                "upload_dir": str(upload_dir),
                "images": saved_images,
                "text_prompt": text_prompt.strip(),
                "confidence_threshold": confidence_threshold,
            }
        )
    except queue.Full:
        shutil.rmtree(upload_dir, ignore_errors=True)
        with jobs_lock:
            jobs.pop(job_id, None)
        raise HTTPException(status_code=503, detail="Inference queue is full")
    except Exception:
        shutil.rmtree(upload_dir, ignore_errors=True)
        with jobs_lock:
            jobs.pop(job_id, None)
        raise

    return {
        "job_id": job_id,
        "status": "queued",
        "websocket_path": f"/ws/{job_id}",
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = job_snapshot(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.websocket("/ws/{job_id}")
async def job_websocket(websocket: WebSocket, job_id: str):
    await websocket.accept()
    last_update = None
    try:
        while True:
            job = job_snapshot(job_id)
            if job is None:
                await websocket.send_json(
                    {"job_id": job_id, "status": "not_found"}
                )
                return
            if job["updated_at"] != last_update:
                await websocket.send_json(job)
                last_update = job["updated_at"]
            if job["status"] in {"completed", "failed"}:
                return
            await asyncio.sleep(0.25)
    finally:
        await websocket.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tunnel",
        action="store_true",
        help="Expose port 8001 using a Cloudflare quick tunnel",
    )
    args = parser.parse_args()
    if args.tunnel:
        start_cloudflare_tunnel()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    mp.freeze_support()
    main()
