# SAM-to-YOLO Backend

FastAPI/uvicorn backend scaffold for project-based SAM inference, dataset
generation, and transfer-learning tasks.

## Run

```bash
pip install -r requirements.txt
uvicorn samtoyolo_backend.main:app --host 0.0.0.0 --port 8000
```

The JSON-RPC WebSocket endpoint is `/v1/ws`. Large files use HTTP upload and
download endpoints under `/v1/projects/{project_id}`.

See [doc/API.md](doc/API.md) for the v1.0 API contract.
