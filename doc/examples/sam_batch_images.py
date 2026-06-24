import argparse
import os

from client import BackendClient


parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="+")
parser.add_argument("--prompt", required=True)
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

with BackendClient(args.url) as client:
    client.init_sam(args.sam_url)
    file_ids = client.upload_many(args.images)
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

