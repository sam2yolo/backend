import argparse
import os

import cv2

from client import BackendClient


parser = argparse.ArgumentParser()
parser.add_argument("video")
parser.add_argument("--every-seconds", type=float, default=30)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument(
    "--url",
    default=os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"),
)
args = parser.parse_args()

capture = cv2.VideoCapture(args.video)
fps = capture.get(cv2.CAP_PROP_FPS)
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
capture.release()
if fps <= 0 or total_frames <= 0:
    raise RuntimeError("Could not read video metadata")

step = max(1, round(fps * args.every_seconds))
frames = list(range(0, total_frames, step))

with BackendClient(args.url) as client:
    client.init_yolo("yolo11n")
    file_id = client.upload(args.video)
    task_id, chunks = client.run_inference(
        {
            "file_id": file_id,
            "file_type": "video",
            "frames": frames,
            "batch": args.batch,
            "conf": 0.25,
            "iou": 0.45,
        }
    )
    for index, chunk_id in enumerate(chunks):
        client.download_chunk(
            chunk_id,
            f"downloads/{task_id}/chunk-{index}.pkl",
        )

