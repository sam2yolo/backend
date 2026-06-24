import argparse
import os

from client import BackendClient


parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument(
    "--url",
    default=os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"),
)
args = parser.parse_args()

with BackendClient(args.url) as client:
    client.init_yolo("yolo11n")
    file_id = client.upload(args.image)
    task_id, chunks = client.run_inference(
        {
            "file_id": file_id,
            "file_ids": [file_id],
            "file_type": "image",
            "batch": 1,
            "conf": 0.25,
            "iou": 0.45,
        }
    )
    for index, chunk_id in enumerate(chunks):
        client.download_chunk(
            chunk_id,
            f"downloads/{task_id}/chunk-{index}.pkl",
        )

