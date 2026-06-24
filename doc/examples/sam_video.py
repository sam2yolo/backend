import argparse
import os
import tempfile
from pathlib import Path

import cv2

from client import BackendClient


parser = argparse.ArgumentParser()
parser.add_argument("video")
parser.add_argument("--prompt", required=True)
parser.add_argument("--every-seconds", type=float, default=30)
parser.add_argument("--max-frames", type=int)
parser.add_argument("--batch", type=int, default=2)
parser.add_argument(
    "--url",
    default=os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"),
)
parser.add_argument(
    "--sam-url",
    default=os.environ.get("SAM_SERVER_URL", "http://127.0.0.1:8001"),
)
args = parser.parse_args()

capture = cv2.VideoCapture(args.video)
fps = capture.get(cv2.CAP_PROP_FPS)
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
if fps <= 0 or total_frames <= 0:
    raise RuntimeError("Could not read video metadata")

step = max(1, round(fps * args.every_seconds))
indexes = list(range(0, total_frames, step))
if args.max_frames:
    indexes = indexes[: args.max_frames]

with tempfile.TemporaryDirectory(prefix="sam-video-") as directory:
    directory = Path(directory)
    frame_paths = []
    wanted = set(indexes)
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in wanted:
            path = directory / f"frame-{frame_index:08d}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"Could not write {path}")
            frame_paths.append(path)
        frame_index += 1
    capture.release()

    with BackendClient(args.url) as client:
        client.init_sam(args.sam_url)
        file_ids = client.upload_many(frame_paths)
        task_id, chunks = client.run_inference(
            {
                "file_ids": file_ids,
                "file_type": "image",
                "text_prompt": args.prompt,
                "conf": 0.5,
                "batch": args.batch,
            }
        )
        for index, chunk_id in enumerate(chunks):
            client.download_chunk(
                chunk_id,
                f"downloads/{task_id}/chunk-{index}.pkl",
            )

