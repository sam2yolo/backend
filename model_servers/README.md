# Model Servers

Each model runs in its own process and conda environment. The main backend sets
up and starts model servers, then publishes public Cloudflare Tunnel endpoints
to Tunnelbroker in remote mode. Clients call inference/training RPCs on model
servers directly over JSON-RPC 2.0 WebSockets.

Model servers currently expect access to the same project/model-cache filesystem
as the backend because RPC calls pass frame and checkpoint paths. If a model
server is moved to another machine, add a file-staging adapter for that model.

## SAM 3.1

```bash
model_servers/sam3/setup.sh
model_servers/sam3/run.sh
```

Default endpoint:

```text
ws://127.0.0.1:8101/v1/ws
```

Point the backend at a different SAM server with:

```bash
export SAMTOYOLO_SAM3_SERVER_URL=ws://host:8101/v1/ws
```

The SAM server exposes:

- `health`
- `inference_sam3`
- `sam3.infer_video_text`
- `train_model`

Long-running calls emit `model.progress` JSON-RPC notifications while the final
response contains `{ "frames": [...], "metadata": {...} }`.
