# Inference examples

Run commands from this directory:

```bash
cd doc/examples
python -m pip install -r requirements.txt
export BACKEND_URL=http://127.0.0.1:8000
```

For an FRP deployment:

```bash
export BACKEND_URL=http://163.61.236.112:20000
```

`SAM_SERVER_URL` is interpreted by the backend process, not by the machine
running these examples. For the usual same-host deployment:

```bash
export SAM_SERVER_URL=http://127.0.0.1:8001
```

## YOLO

Single image:

```bash
python yolo_single_image.py image.jpg
```

Image batch:

```bash
python yolo_batch_images.py images/*.jpg --batch 16
```

Video sampled every 30 seconds:

```bash
python yolo_video.py video.mp4 --every-seconds 30 --batch 16
```

## SAM3

Single image:

```bash
python sam_single_image.py image.jpg --prompt person
```

Image batch:

```bash
python sam_batch_images.py images/*.jpg --prompt vehicle --batch 2
```

Video frames sampled every 30 seconds:

```bash
python sam_video.py video.mp4 \
  --prompt person \
  --every-seconds 30 \
  --batch 2
```

Limit the SAM video workflow to 100 frames:

```bash
python sam_video.py video.mp4 \
  --prompt person \
  --every-seconds 3 \
  --max-frames 100 \
  --batch 2
```

Downloaded pickle files are saved under `downloads/{task_id}/`.

YOLO pickle files require `ultralytics` when being loaded. SAM pickle files
contain ordinary dictionaries, lists, strings, and numbers.

