import argparse
import os

from client import BackendClient


parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="+")
parser.add_argument("--batch", type=int, default=16)
parser.add_argument(
    "--url",
    default=os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"),
)
args = parser.parse_args()

with BackendClient(args.url) as client:
    client.init_yolo("yolo11n")
    file_ids = client.upload_many(args.images)
    task_id, chunks = client.run_inference(
        {
            "file_id": file_ids[0],
            "file_ids": file_ids,
            "file_type": "image",
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

