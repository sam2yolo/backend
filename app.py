from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from actionhandler import handle_action
import io
import mimetypes
import os

from modelhandler import yoloHandler

import asyncio
from context import *
from contextlib import asynccontextmanager
from video_utils import maybe_convert_dav, storage_path_for_file

# Configure the logging system
logging.basicConfig(
    level=logging.DEBUG,  # Capture INFO, WARNING, ERROR, and CRITICAL logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# create a context instance to hold the global variables and constants used in the application and also to register the action handlers
context = Context()
context.models_dict['yolo'] = yoloHandler





@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: store the event loop reference and start the response dispatcher."""
    context.set_loop(asyncio.get_event_loop())
    context.start_worker()
    logging.info("Server started, response dispatcher running")
    yield
    """Shutdown: stop the response dispatcher and signal running workers."""
    context.stop_event.set()
    context.stop_worker()
    logging.info("Server shutting down, workers signalled")


app = FastAPI(lifespan=lifespan)

@app.get("/")
async def get():
    with open("test_front/templates/index.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # If there is already a connection refuse to connect
    if context.active_websocket:
        try:
            # Check if the old websocket is still alive
            old_ws = context.active_websocket
            if hasattr(old_ws, 'client_state') and old_ws.client_state.name == 'CONNECTED':
                # only one web socket is allowed
                await websocket.close()
                logging.info('Websocket closed (connection already exists).')
                return
            else:
                # Old websocket is dead, clear it and allow new connection
                logging.info('Clearing stale websocket connection')
                context.active_websocket = None
        except Exception as e:
            logging.error(f'Error checking old websocket: {e}')
            context.active_websocket = None

    # Register new websocket
    context.active_websocket = websocket
    logging.info('New websocket connection established')

    # Re-enable the stop event so new workers can run
    context.stop_event.clear()

    # Flush any pending responses from the previous session
    await context.flush_pending_responses()

    try:
        while True:
            # Receive text data from the client
            data = await websocket.receive_text()

            # check if the received data is a valid JSON string
            # expected format: {"action": "some_action", "payload": {"key": "value"}}
            try:
                import json
                data_dict = json.loads(data)
                action = data_dict.get("action")
                payload = data_dict.get("payload")

                # action and payload should also be dictionaries
                if not isinstance(action, str) or not isinstance(payload, dict):
                    logging.warning(f"Received invalid action or payload: {data_dict}")
                    await websocket.send_text(json.dumps({"error": "Invalid action or payload format"}))
                    continue

                logging.info(f"Received action: {action} with payload: {payload}")
                # Call the action handler and send the response back to the client
                await handle_action(websocket, action, payload, context)
            except json.JSONDecodeError:
                logging.warning(f"Received non-JSON message: {data}")
                # send error json message back to client
                await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))

    except WebSocketDisconnect:
        logging.info("Client disconnected")

        # Clear the active websocket so background threads don't try to send
        context.active_websocket = None

    except Exception as e:
        logging.error(f'Error in websocket handler: {type(e).__name__}: {e}')

        context.active_websocket = None


@app.get('/inference_result')
async def get_inference_result(id:str):

    save_file_path = context.get_inference_result_path(id)
    if not save_file_path:
        return JSONResponse(status_code=404, content={"error": "File not found"})
    
    with open(save_file_path, "rb") as f:
        data = f.read()

    byte_io = io.BytesIO(data)

    return StreamingResponse(
            byte_io, 
            media_type="application/octet-stream"
        )                            


@app.get('/file')
async def get_uploaded_file(id: str):
    file_info = context.files_dict.get(id)
    if not file_info:
        return JSONResponse(status_code=404, content={"error": "File not found"})

    file_path = file_info["path"] if isinstance(file_info, dict) else file_info
    if not file_path or not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})

    file_name = file_info.get("name") if isinstance(file_info, dict) else os.path.basename(file_path)
    media_type = mimetypes.guess_type(file_name or file_path)[0] or "application/octet-stream"
    return FileResponse(file_path, media_type=media_type, filename=file_name or os.path.basename(file_path))


@app.post('/upload')
async def handle_file_upload(file:UploadFile = File(...)):
    
    # generate file_id

    # ensure the storage directory exists (the download handlers create it, but
    # a fresh checkout may reach /upload first — without this, open() 500s)
    os.makedirs("files", exist_ok=True)

    file_name = file.filename

    file_id = str(uuid.uuid4())[:16]
    while (
        os.path.exists(f"files/{file_id}") or
        os.path.exists(f"files/{file_id}.dav") or
        os.path.exists(f"files/{file_id}.mp4")
    ):
        file_id = str(uuid.uuid4())[:16]

    # Create file path
    file_path = storage_path_for_file(file_id, file_name)

    # read file
    content = await file.read()

    # write file
    with open(file_path, "wb") as f:
            f.write(content)

    file_meta = await asyncio.to_thread(maybe_convert_dav, file_id, file_path, file_name)

    # save file meta to context:
    context.files_dict[file_id] = file_meta


    return JSONResponse({
        'status':200,
        'file_id':file_id,
        'file_path': file_meta["path"],
        'file_name': file_meta["name"],
        'original_name': file_meta.get("original_name"),
        'converted': file_meta.get("converted", False),
        'conversion_error': file_meta.get("conversion_error")
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
#!/bin/bash

# script is non interactive

# check if `samtoyolo_conda_environment` environment exists
# if does not exist
#   install latest miniconda with python version 3.9
#   activate environment

# run pip install -r requirements.txt
