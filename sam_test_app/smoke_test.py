"""End-to-end smoke test for SAM 3.1 via the model-manager backend.

Usage:
    python smoke_test.py image  <image_path>  --prompt person
    python smoke_test.py video  <video_path>  --prompt person --fps 2
"""

import argparse
import sys
from pathlib import Path

import cv2

from sam_client import SamBackendClient, overlay_result, sample_frames

BACKEND_URL = "http://163.61.236.112:20000"
SAM_URL = "http://127.0.0.1:8001"
OUT = Path(__file__).parent / "outputs"


def log_event(msg):
    a = msg.get("action")
    p = msg.get("payload", {})
    if a == "inference_stage_plus_progress":
        print(f"  [progress] {p.get('progress')}%  {p.get('stage')}")
    elif a == "inference_task_chunk_result":
        print(f"  [chunk] {p.get('chunk_id')} idx={p.get('chunk_index')} frames={p.get('frame_count')}")
    elif a in ("model_init_completed", "task_added", "inference_completed",
               "model_init_error", "inference_task_error", "task_failed"):
        print(f"  [{a}] {p}")


def visualize(client, file_paths, task_id, chunks, tag):
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    flat_idx = 0
    total_det = 0
    for chunk_id in chunks:
        data = client.download_chunk(chunk_id)
        for img_res in data["images"]:
            src = file_paths[flat_idx]
            img = cv2.imread(str(src))
            annotated = overlay_result(img, img_res)
            n = len(img_res.get("masks", []))
            total_det += n
            dst = out_dir / f"{flat_idx:04d}_det{n}.jpg"
            cv2.imwrite(str(dst), annotated)
            print(f"    saved {dst.name}  ({n} detections)")
            flat_idx += 1
    print(f"  -> {flat_idx} frames, {total_det} total detections in {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["image", "video"])
    ap.add_argument("path")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    if args.mode == "video":
        print(f"Sampling video at {args.fps} fps ...")
        frame_dir = OUT / "frames"
        paths, src_fps, step = sample_frames(
            args.path, frame_dir, target_fps=args.fps, max_frames=args.max_frames
        )
        print(f"  src_fps={src_fps:.2f} step={step} -> {len(paths)} frames")
    else:
        paths = [args.path]

    client = SamBackendClient(BACKEND_URL)
    client.connect()
    try:
        print("Init SAM ...")
        info = client.init_sam(SAM_URL, on_event=log_event)
        print(f"  server: {info.get('server')}")

        print(f"Uploading {len(paths)} file(s) ...")
        file_ids = [client.upload(p) for p in paths]

        print(f"Running inference (prompt='{args.prompt}', conf={args.conf}, batch={args.batch}) ...")
        task_id, chunks = client.run_inference(
            file_ids, args.prompt, conf=args.conf, batch=args.batch, on_event=log_event
        )
        print(f"Done. task_id={task_id}, {len(chunks)} chunks")

        print("Downloading + visualizing ...")
        visualize(client, paths, task_id, chunks, tag=args.mode)
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
