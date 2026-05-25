# SAM-to-YOLO Demo Test App

This Flask app is a browser-based test client for the remote inference flow in
`doc/INFERENCE_AND_DOWNLOAD.md`.

It can:

- discover backend/model-server peers from Tunnelbroker,
- create a project,
- upload a local video with upload progress,
- forward local videos to the remote backend using chunked HTTP uploads to avoid
  Cloudflare `413 Payload Too Large` limits,
- import a Google Drive video and show backend task progress,
- configure SAM3 model-server setup,
- run the configured SAM3 inference request,
- show backend task/model progress events,
- download the final inference zip with download progress when the backend
  returns a packaged inference task.

Install and run:

```bash
cd demo_test_app
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5055
```

The app intentionally runs as a client. It does not require SSH access to the
GPU notebook; it uses only Tunnelbroker, HTTPS, and WSS.
