# SAM-to-YOLO Backend

FastAPI/uvicorn backend scaffold for project-based SAM inference, dataset
generation, and transfer-learning tasks.

## Run

```bash
python -m samtoyolo_backend.run
```

On first run the backend checks for a conda environment named
`samtoyolo-backend`. If it does not exist, the backend creates it, installs
`requirements.txt`, and re-runs itself inside that environment. If conda is not
installed, it installs Miniforge under `~/.samtoyolo/miniforge3`.

Model runtimes are isolated into separate model servers. On backend startup,
the main server can set up and start model servers, then publish their
Cloudflare Tunnel endpoints to Tunnelbroker in remote mode. You can also start
the SAM 3.1 model server manually:

```bash
model_servers/sam3/run.sh
```

Clients should call inference and training methods on the model-server
WebSocket endpoint returned by `models()` or `get_model_server_list()`.

## Remote Notebook Startup

In production notebook environments, the client should not need SSH access. The
notebook only starts the backend with remote settings; the backend then creates
Cloudflare public tunnels and announces itself to Tunnelbroker.

```bash
SAMTOYOLO_MODE=remote \
SAMTOYOLO_SERVER_NAME=my-gpu-session \
TUNNELBROKER_URL=https://your-tunnelbroker.example \
TUNNELBROKER_GROUP=my-group \
TUNNELBROKER_PEER_SECRET=peer-secret \
TUNNELBROKER_GROUP_TOKEN=group-token \
python -m samtoyolo_backend.run
```

If `cloudflared` is not installed, the backend downloads a Linux binary into
`~/.samtoyolo/bin` before opening the tunnel. The main backend registers as
`SAMTOYOLO_SERVER_NAME`; the SAM3 model server registers as
`SAMTOYOLO_SERVER_NAME-sam3` with its public WebSocket endpoint in metadata.
The client discovers these peers from Tunnelbroker and then uses public WSS/HTTP
URLs for RPC, uploads, inference, and result downloads.

You can still call uvicorn directly:

```bash
uvicorn samtoyolo_backend.main:app --host 0.0.0.0 --port 8000
```

The uvicorn import path performs the same backend conda bootstrap before
loading FastAPI.

The JSON-RPC WebSocket endpoint is `/v1/ws`. Large files use HTTP upload and
download endpoints under `/v1/projects/{project_id}`.

See [doc/API.md](doc/API.md) for the v1.0 API contract.
