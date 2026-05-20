# SAM-to-YOLO Backend

FastAPI/uvicorn backend scaffold for project-based SAM inference, dataset
generation, and transfer-learning tasks.

## Run

```bash
python -m samtoyolo_backend.run
```

On first run the backend checks for a conda environment named
`samtoyolo-sam3`. If it does not exist, the backend creates it, installs a
CUDA-enabled PyTorch build, installs `requirements.txt`, and re-runs itself
inside that environment. If conda is not installed, it installs Miniforge under
`~/.samtoyolo/miniforge3`.

You can still call uvicorn directly:

```bash
uvicorn samtoyolo_backend.main:app --host 0.0.0.0 --port 8000
```

The uvicorn import path performs the same conda bootstrap before loading
FastAPI or SAM dependencies.

The JSON-RPC WebSocket endpoint is `/v1/ws`. Large files use HTTP upload and
download endpoints under `/v1/projects/{project_id}`.

See [doc/API.md](doc/API.md) for the v1.0 API contract.
