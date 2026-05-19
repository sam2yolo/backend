For version 1.0

Stack: FASTAPI + uvicorn

endpoint: `/v1/<method>`
All communication is throw `wss`.

Purpose: A backend server for transfer learning. Creates dataset from inference result of SAM 3.1 (also other potential models with similar accuracy) and trains the result on smaller model like yolo, RT-DETR, Grounding DINO, SSD (Single Shot MultiBox Detector), EfficientDet, RetinaNet, Faster R-CNN, Detectron2, MediaPipe. Generates training dataset from inference.

Project based workspace. Inference result is saved in own format by zip compression. 

### Features:
- While installing installs necessary requirements
- It is the server side front end side can be running on another pc
- Each inference, upload, training, dataset merging, and dataset mapping is a task
- Can connect and mount to mega taking login credential. Credentials are entered for each task. Runs on task add a time. Mounts task specefic mega drive
- Files can be upload or can be collected using google drive, mega or direct link
- Inference SAM 3.1 with necessary parameters. Parameters will be send through endpoints
- Inference progress will be streamed through rpc
- Inference will be based on per gpu worker
- Inference supports temporal downsampling
- Inference will batch inference
- after a batch Inference, server will ask the client to download result (result will contain information so that a training dataset can be made) if remote saving is not saved (i.e. save to mega drive)
- If client downloads delete inference result
- If client gets disconnected to at any moment keep the inference running client may again try to connect
- Inference and training will primary run on gpu
- if inference output is saved to mega-drive save to mega drive. Make a project folder in mega drive. 
- Session information should be saved so that if for any reason server crashes and new server is connected the backend will init the server setting all things up and resume from where it left off. For this server may ask client for necessary information
- Use openCV for frame conversion but this part shouldn't take more than 3gb of cpu memory.
- Inference task may have multiple prompts. Prompts will be mapped to classes this mapping will be provided from api calling. client will provide information like: `prompt` -> `classname`.
- Dataset can be merged. During merging classes with similar name will be merged to new dataset. Classes with not matching name will be added as usual. user maps classes for this purpose.
- client should be able to train model from generated dataset or directly from inference result creating a dataset all information will be provided by client.
- server can send notification to client
- server can ask for information to client. client can have popup which generated for server.
- remote servers generally has 11hr 50 min session. If a inference task is larger than that new session will be created by server. 
- make project space twin system one in server and one in client. Client side will be minimum enough to save result and resume if server changed
- Client side will be managed by frontend client app so add necessary endpoints and events for front end app.
- Server is under a shared IP. Server uses cloudflared tunnel to create endpoint. server and client uses a tunnelbroker program see documentation. If server is running but cloudflared tunnel stopped create a new tunnel and and update in broker. Server will be installed with shared secrets (if installed as remote mode). If server is installed as local machine mode no need for this.


Sum possible API endpoints and events but not limited to it. Add if necessary:

# Events:

- `backend-init` - when backend started to initialize (only during first run, install requirements, start server )
- `backend-init-progress` - sends percentage progress
- `backend-ready` - when initialization is complete
-  `new-dataset-from-merge`
- `new-dataset-from-map`
- `mega-credential-check-result`
- `mega-mount-success`
- `upload-success` (make endpoints for each variants)
- ``


# Methods

- `models` - list of model available for inference

## Tasks
- `inference_sam3` 
- `inference_yolo`
- `video_upload` - upload video
- `image_upload` - upload image
- `dataset_upload` - upload .zip dataset
- `gdrive_upload` - Google drive public link of video or zip file will be given 
- `mega_upload` - mega public link will be given
-  `merge_dataset`
- `map_dataset`
- `check_mega_credentials`
- `register_mega_credential`
- `train-yolo`
- `train`
- `inference-to-yolo` converts inference result to yolo dataset

## Info methods
- `get_task_list`

## Inference
- `getInferenceResult`
- ``
## Download
- `getModel`



Instruction: 
1. Identify potential revision and reviews. Ask me a question if needed. Refine the design specification
2. Change every terms to suitable names. Make a handler function for every events and methods. Keep Each handlers in separate `py` file. Make it modular so that events and method can be added easily. Make a detailed API documentation