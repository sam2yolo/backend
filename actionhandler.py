import json
from multiprocessing import context
import asyncio
import subprocess
from context import Context
from fastapi import WebSocket
from typing import Awaitable, Callable, Dict, Optional
import logging
import time
import uuid
import requests
import os
import re
import shutil
import pickle
from modelhandler import ModelHandler
from commons import send_action
import threading

# List of actions accepeted by the server
# ping: {"action": "ping", "payload": {}}
# list_models: {"action": "list_models", "payload": {}}
# init_model: {"action": "init_model", "payload": {"model_name": "yolov11-seg"}}
# destroy_model: {"action": "destroy_model", "payload": {"model_name": "yolov11-seg"}}


# download_file_wget: {"action": "download_file_wget", "payload": {"url": "https://example.com/video.mp4"}} -> returns a unique video_id that can be used to track the download progress also video id can be used to get video download path
# download_file_google_drive: {"action": "download_file_google_drive", "payload": {"file_id": "1a2b3c4d5e6f7g8h9i0j"}} -> returns a unique video_id that can be used to track the download progress also video id can be used to get video download path
# upload_file: {"action":"upload_file"}

# list_files: {"action": "list_files", "payload": {}} -> returns a list of all downloaded files with their file_id, file_name and file_path
# delete_file: {"action": "delete_file", "payload": {"file_id": "1234"}} -> deletes the file with the given file_id from the server and also removes the entry from files_dict in context


# create_inference_task: {"action": "create_inference_task", "payload": {"model_name": "yolov11-seg", "file_id": "1234"}} -> creates an inference task for the given model and file and returns a unique task_id that can be used to track the inference progress
# delete_inference_task: {"action": "delete_inference_task", "payload": {"task_id": "1234"}} -> deletes the inference task with the given task_id from the server and also removes any associated resources 
# start_inference_from_queue: {"action": "start_inference_from_queue", "payload": {}} -> starts the next inference task in the queue and returns the task_id of the started task or a message indicating that the queue is empty
# stop_inference_task

# delete_inference_result: {"action": "delete_inference_result", "payload": {"chunk_id": "1234"}} -> deletes the inference result for the given chunk_id from the server
# delete_inference_result_of_task: {"action": "delete_inference_result_of_task", "payload": {"task_id": "1234"}} -> deletes the inference result for the given task_id from the server



# List of actions that the server can send to the client
# download_progress: {"action": "download_progress", "payload": {"video_id": "1234", "progress": 50}} -> sent periodically to update the client about the download progress of a video
# git_clone_initiated: {"action": "git_clone_initiated", "payload": {"model_name": "yolov11-seg"}} -> sent when the git clone process for a model is initiated
# git_clone_completed: {"action": "git_clone_completed", "payload": {"model_name": "yolov11-seg"}} -> sent when the git clone process for a model is completed
# git_clone_failed: {"action": "git_clone_failed", "payload": {"model_name": "yolov11-seg", "error": "error message"}} -> sent when the git clone process for a model fails
# init_script_initiated: {"action": "init_script_initiated", "payload": {"model_name": "yolov11-seg"}} -> sent when the init.sh script execution for a model is initiated
# init_script_completed: {"action": "init_script_completed", "payload": {"model_name": "yolov11-seg"}} -> sent when the init.sh script execution for a model is completed   
# init_script_failed: {"action": "init_script_failed", "payload": {"model_name": "yolov11-seg", "error": "error message"}} -> sent when the init.sh script execution for a model fails

# model_not_found: {"action": "model_not_found", "payload": {"model_name": "yolov11-seg"}} -> sent when the requested model is not found in the servers model list

# inference_stage_plus_progress: {"action": "inference_stage_plus_progress", "payload": {"task_id": "1234", "progress": 50, "stage": "stage_name"}} -> sent periodically to update the client about the inference progress of a task
# inference_completed: {"action": "inference_completed", "payload": {"task_id": "1234", "result": "path to result file"}} -> sent when an inference task is completed with the path to the result file
# inference_failed: {"action": "inference_failed", "payload": {"task_id": "1234", "error": "error message"}} -> sent when an inference task fails with the error message    





handlers: Dict[str, Callable[[WebSocket, dict, Context], Awaitable[None]]] = {}




def register(action: str):
    def decorator(func: Callable[[WebSocket, dict, Context], Awaitable[None]]):
        handlers[action] = func
        return func
    return decorator


async def handle_action(websocket: WebSocket, action: str, data: dict, context: Context):
    handler = handlers.get(action)
    if handler:
        await handler(websocket, data, context)
        logging.info(f"Handled request for action: {action}")
    else:
        await websocket.send_text(json.dumps({"action": "error", "payload": {"message": "Unknown action"}}))


@register("ping")
async def handle_ping(websocket: WebSocket, data: dict, context: Context):
    response = {"action": "pong", "payload": {}}
    await websocket.send_text(json.dumps(response))

@register("list_models")
async def handle_list_models(websocket: WebSocket, data: dict, context: Context):
    # This is a placeholder implementation. You should replace it with actual logic to list available models.
    models = list(context.models_dict.keys())
    response = {"action": "list_models_response", "payload": {"models": models}}

    logging.debug(f"List of models: {models}")

    await websocket.send_text(json.dumps(response))


@register("init_model")
async def handle_init_model(websocket: WebSocket, data: dict, context: Context):
    
    def worker(data: dict, context: Context):
    
        model_name = data.get("model_name", None)

        # check if model sent
        if not model_name:
            send_action(context, "model_init_error",{'error':'model_name_not_provided'})
            return

        # check if model exists
        if model_name not in context.models_dict:
            send_action(context,"model_init_error",{
                'model_name':model_name,
                'error':'model_not_found'
            })
            logging.error(f'{model_name} not found')
            return
        ModelHandlerClass = context.models_dict[model_name]

        # check if handler found
        if not ModelHandlerClass:
            send_action(context,"model_init_error",{
                'model_name':model_name,
                'error':'handler_not_found'
            })
            logging.error("ModelHandler not found")
            return

        context.modelHandler = ModelHandlerClass(context, data)
        context.modelHandler.setup(data, context)
        context.modelHandler.init(data, context)

    thread = threading.Thread(target=worker, args=(data,context))
    thread.daemon = True
    thread.start()

    
@register("destroy_model")
async def handle_destroy_model(websocket: WebSocket, data: dict, context: Context):
    if not context.modelHandler:
        send_action(context, "no_model_loaded_error",{})
        return

    context.modelHandler.destroy(data,context)
    model_name = context.modelHandler.model_name

    context.modelHandler = None

    send_action(context,"model_distroyed",{'model_name':model_name})



@register("download_file_wget")
async def handle_download_file_wget(websocket: WebSocket, data: dict, context: Context):
    url = data.get("url")

    def worker(url: str, context: Context):
        # generate a 16 char uuid for the file check if the uuid already exists in the files directory if it does generate a new one until we get a unique uuid this is to avoid file name conflicts in the files directory
        file_id = str(uuid.uuid4())[:16]
        while os.path.exists(f"files/{file_id}"):
            file_id = str(uuid.uuid4())[:16]

        # get acutal file name from url
        file_name = url.split("/")[-1]
        os.makedirs("files", exist_ok=True)

        # send file_download_initiated message to client
        send_action(context, "file_download_initiated", {
            "file_id": file_id,
            "expected_path": f"files/{file_id}",
            "file_name": file_name
        })

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logging.info(f"Received download_file_wget action for url {url} at {current_time} with file_id {file_id}")

        # use requests and tqdm to download the file and send download_progress messages
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            with open(f"files/{file_id}", "wb") as f:
                for chunk in response.iter_content(block_size):
                    if context.stop_event.is_set():
                        logging.info(f"Download {file_id} cancelled by stop event")
                        os.remove(f"files/{file_id}")
                        return
                    f.write(chunk)
                    # send download_progress message to client via queue
                    progress = int((f.tell() / total_size) * 100) if total_size else None
                    send_action(context, "download_progress", {
                        "file_id": file_id,
                        "progress": progress,
                        "timestamp": current_time,
                        "total_size": total_size
                    })
                # add entry to files_dict in context
                context.files_dict[file_id] = {"path": f"files/{file_id}", "name": file_name}
                send_action(context, "file_download_completed", {
                    "file_id": file_id,
                    "file_path": f"files/{file_id}"
                })

        except Exception as e:
            logging.error(f"File download failed for url {url}: {e}")
            send_action(context, "download_failed", {
                "file_id": file_id,
                "error": str(e)
            })
            return

    thread = threading.Thread(target=worker, args=(url, context))
    thread.daemon = True
    thread.start()




def handle_log_google_drive(line: str) -> Optional[dict]:
    line = line.strip()
    if not line:
        return None

    # 1. Check if it's a progress line (contains percentage followed by '|')
    if re.search(r'^\d+%', line):
        pct_match = re.search(r'(\d+)%', line)
        size_match = re.search(r'([\d\.]+[A-Za-z]*)/([\d\.]+[A-Za-z]*)', line)
        eta_match = re.search(r'<([^,\]]+)', line)
        
        progress_data = {
            "percentage": int(pct_match.group(1)) if pct_match else None,
            "downloaded_size": size_match.group(1) if size_match else None,
            "total_size": size_match.group(2) if size_match else None,
            "eta": eta_match.group(1) if eta_match else None
        }
        return {'type': 'progress', 'data': progress_data}

    # 2. Check if it's the target file path line
    if line.startswith("To:"):
        file_path = line.replace("To:", "").strip()
        return {'type': 'file_path', 'data': {'path': file_path}}

    # 3. For all other meta lines (e.g., 'Downloading...', URLs), return meta type
    return {'type': 'meta', 'data': {'text': line}}
""" example output for handle_log_google_drive function:

{'type': 'meta', 'data': {'text': 'Downloading...'}}
{'type': 'meta', 'data': {'text': 'From (original): https://drive.google.com/uc?id=1U_SBWxdyRFx-519v_UQZh48cm4y4qLVm'}}
{'type': 'meta', 'data': {'text': 'From (redirected): https://drive.google.com/uc?id=1U_SBWxdyRFx-519v_UQZh48cm4y4qLVm&confirm=t&uuid=f06142e3-6dce-488b-b36f-1c360d94ddbe'}}
{'type': 'file_path', 'data': {'path': '/kaggle/working/sam3.1.zip'}}
None,
{'type': 'progress', 'data': {'percentage': 0, 'downloaded_size': '0.00', 'total_size': '3.25G', 'eta': '?'}}
{'type': 'progress', 'data': {'percentage': 0, 'downloaded_size': '4.72M', 'total_size': '3.25G', 'eta': '01:37'}}"""



def _download_google_drive_worker(url: str, context: Context):
    """Background worker for Google Drive download.
    
    Runs in a daemon thread. Uses send_action() instead of direct websocket calls,
    so it survives WebSocket disconnections.
    """
    # call a subprocess to run gdown with the provided url and parse the output
    process = subprocess.Popen(
        ["gdown", "--fuzzy", url],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    file_id = str(uuid.uuid4())[:16]
    os.makedirs("files", exist_ok=True)
    while os.path.exists(f"files/{file_id}"):
        file_id = str(uuid.uuid4())[:16]

    expected_path = None
    for line in process.stdout:
        if context.stop_event.is_set():
            process.terminate()
            logging.info(f"Google Drive download {file_id} cancelled by stop event")
            return

        log_data = handle_log_google_drive(line)
        if log_data:
            if log_data['type'] == 'file_path':
                expected_path = log_data['data']['path']
                send_action(context, "file_download_initiated", {
                    "file_id": file_id,
                    "expected_path": expected_path
                })
            elif log_data['type'] == 'progress':
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                send_action(context, "download_progress", {
                    "file_id": file_id,
                    "progress": log_data['data']['percentage'],
                    "timestamp": current_time,
                    "total_size": log_data['data']['total_size']
                })

    process.wait()

    if process.returncode != 0:
        send_action(context, "download_failed", {
            "file_id": file_id,
            "error": f"gdown exited with status {process.returncode}"
        })
        return

    if not expected_path:
        send_action(context, "download_failed", {
            "file_id": file_id,
            "error": "Failed to determine downloaded file path"
        })
        return

    final_path = f"files/{file_id}"
    original_name = os.path.basename(expected_path)
    try:
        shutil.move(expected_path, final_path)
    except Exception as e:
        logging.error(f"Failed to move downloaded file {expected_path}: {e}")
        send_action(context, "download_failed", {
            "file_id": file_id,
            "error": str(e)
        })
        return

    context.files_dict[file_id] = {"path": final_path, "name": original_name}
    send_action(context, "file_download_completed", {
        "file_id": file_id,
        "file_path": final_path,
        "file_name": original_name
    })


@register("download_file_google_drive")
async def handle_download_file_google_drive(websocket: WebSocket, data: dict, context: Context):
    url = data.get("url")
    if not url:
        send_action(context, "download_failed", {"error": "No url provided"})
        return

    thread = threading.Thread(target=_download_google_drive_worker, args=(url, context))
    thread.daemon = True
    thread.start()

    

@register("list_files")
async def handle_list_files(websocket: WebSocket, data: dict, context: Context):
    file_list = [{"id": fid, "path": info["path"], "name": info["name"]} for fid, info in context.files_dict.items()]
    await websocket.send_text(json.dumps({"action": "file_list", "payload": {"files": file_list}}))

@register("delete_file")
async def handle_delete_file(websocket: WebSocket, data: dict, context: Context):
    file_id = data.get("file_id")
    if file_id in context.files_dict:
        file_path = context.files_dict[file_id]["path"]
        try:
            os.remove(file_path)
            del context.files_dict[file_id]
            await websocket.send_text(json.dumps({"action": "delete_file_success", "payload": {"file_id": file_id}}))
        except Exception as e:
            logging.error(f"Failed to delete file {file_path}: {e}")
            await websocket.send_text(json.dumps({"action": "delete_file_failed", "payload": {"file_id": file_id, "error": str(e)}}))
    else:
        await websocket.send_text(json.dumps({"action": "delete_file_failed", "payload": {"file_id": file_id, "error": "File ID not found"}}))



@register("create_inference_task")
async def handle_create_inference_task(websocket: WebSocket, data: dict, context: Context):
    if not context.modelHandler:
        send_action(context, 'model_handler_not_loaded_error')
        logging.error('Model handler not loaded yet task is being created')
        return

    '''
    parameters for payload:
        file_id*
        file_type*
        conf
        iou
        classes
        device
        verbose
        imgsz

        persist
        tracker
    '''

    # Create the inference task - it will be queued internally by the model handler
    context.modelHandler.add_task(data, context)


@register("delete_inference_task")
async def handle_delete_inference_task(websocket: WebSocket, data: dict, context: Context):
    if not context.modelHandler:
        send_action(context, 'model_handler_not_loaded_error')
        logging.error('Model handler not loaded')
        return
    context.modelHandler.delete_task(data, context)


@register("start_inference_from_queue")
async def handle_start_inference_from_queue(websocket: WebSocket, data: dict, context: Context):
    if not context.modelHandler:
        send_action(context, 'model_handler_not_loaded_error')
        logging.error('Model handler not loaded')
        return
    context.modelHandler.start_working(data, context)


@register("fetch_inference_chunk")
async def handle_fetch_inference_chunk(websocket: WebSocket, data: dict, context: Context):
    """Fetch inference chunk results for a given task.
    
    Payload:
        task_id (str): ID of the inference task
        chunk_id (str, optional): Specific chunk to fetch.
            If omitted, returns the list of all chunks for the task.
    
    Response actions:
        inference_chunk_list: list of all chunks for the task
        inference_chunk_data: data for a specific chunk (loaded from pickle)
        inference_chunks_error: error message
    """
    task_id = data.get("task_id")
    if not task_id:
        await websocket.send_text(json.dumps({
            "action": "inference_chunks_error",
            "payload": {"error": "No task_id provided"}
        }))
        return

    def _as_info(entry):
        # Chunk metadata may be stored as a dict (SAM) or a [dict] list (YOLO).
        if isinstance(entry, list):
            return entry[0] if entry else None
        if isinstance(entry, dict):
            return entry
        return None

    chunk_id = data.get("chunk_id")
    if chunk_id:
        # SAM stores chunks at the top level keyed by chunk_id; YOLO may nest
        # them under inference_results[task_id][chunk_id]. Handle both.
        chunk_info = _as_info(context.inference_results.get(chunk_id))
        if chunk_info is None:
            nested = context.inference_results.get(task_id)
            if isinstance(nested, dict):
                chunk_info = _as_info(nested.get(chunk_id))
        if chunk_info is None:
            await websocket.send_text(json.dumps({
                "action": "inference_chunks_error",
                "payload": {"error": f"Chunk {chunk_id} not found for task {task_id}"}
            }))
            return

        save_file = chunk_info.get("save_file")
        chunk_data = []
        if save_file and os.path.exists(save_file):
            with open(save_file, 'rb') as f:
                chunk_data = pickle.load(f)

        await websocket.send_text(json.dumps({
            "action": "inference_chunk_data",
            "payload": {
                "task_id": task_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_info.get("chunk_index", 0),
                "frame_count": chunk_info.get("frame_count", 0),
                "frames": chunk_info.get("frames", []),
                "data": chunk_data
            }
        }))
    else:
        # List all chunks for the task — search both storage shapes.
        chunk_list = []
        for cid, entry in context.inference_results.items():
            info = _as_info(entry)
            if info and info.get("task_id") == task_id:
                chunk_list.append({**info, "chunk_id": cid})
        nested = context.inference_results.get(task_id)
        if isinstance(nested, dict):
            for cid, entry in nested.items():
                info = _as_info(entry)
                if info:
                    chunk_list.append({**info, "chunk_id": cid})

        await websocket.send_text(json.dumps({
            "action": "inference_chunk_list",
            "payload": {
                "task_id": task_id,
                "chunks": chunk_list
            }
        }))


@register("delete_inference_result")
async def handle_delete_inference_result(websocket: WebSocket, data: dict, context: Context):
    """Delete a single inference result chunk by chunk_id.

    Payload:
        chunk_id (str): ID of the chunk to delete.

    Response actions:
        inference_result_deleted: chunk was deleted
        inference_result_delete_error: error message
    """
    chunk_id = data.get("chunk_id")
    if not chunk_id:
        send_action(context, "inference_result_delete_error", {
            "error": "No chunk_id provided"
        })
        return

    def worker(chunk_id: str, context: Context):
        chunk_info = context.inference_results.get(chunk_id)
        if not chunk_info:
            send_action(context, "inference_result_delete_error", {
                "error": f"Chunk {chunk_id} not found"
            })
            return

        task_id = chunk_info.get("task_id")

        # Delete the pickle file on disk
        save_file = chunk_info.get("save_file")
        if save_file and os.path.exists(save_file):
            try:
                os.remove(save_file)
            except Exception as e:
                logging.error(f"Failed to delete chunk file {save_file}: {e}")

        # Remove from in-memory dict
        del context.inference_results[chunk_id]
        logging.info(f"Deleted inference result chunk {chunk_id} for task {task_id}")

        send_action(context, "inference_result_deleted", {
            "chunk_id": chunk_id,
            "task_id": task_id
        })

    thread = threading.Thread(target=worker, args=(chunk_id, context))
    thread.daemon = True
    thread.start()


@register("delete_inference_result_of_task")
async def handle_delete_inference_result_of_task(websocket: WebSocket, data: dict, context: Context):
    """Delete all inference results for a given task_id.

    Payload:
        task_id (str): ID of the task whose results should be deleted.

    Response actions:
        inference_results_of_task_deleted: all results for the task were deleted
        inference_result_delete_error: error message
    """
    task_id = data.get("task_id")
    if not task_id:
        send_action(context, "inference_result_delete_error", {
            "error": "No task_id provided"
        })
        return

    def worker(task_id: str, context: Context):
        # Find all chunks belonging to this task
        chunks_to_delete = []
        for cid, info in list(context.inference_results.items()):
            if isinstance(info, dict) and info.get("task_id") == task_id:
                chunks_to_delete.append((cid, info))

        if not chunks_to_delete:
            send_action(context, "inference_result_delete_error", {
                "error": f"No results found for task {task_id}"
            })
            return

        deleted_count = 0
        for cid, info in chunks_to_delete:
            # Delete the pickle file on disk
            save_file = info.get("save_file")
            if save_file and os.path.exists(save_file):
                try:
                    os.remove(save_file)
                except Exception as e:
                    logging.error(f"Failed to delete chunk file {save_file}: {e}")

            # Remove from in-memory dict
            del context.inference_results[cid]
            deleted_count += 1

        # Also remove the results directory if it exists
        results_dir = os.path.join("results", task_id)
        if os.path.exists(results_dir):
            try:
                shutil.rmtree(results_dir)
                logging.info(f"Removed results directory {results_dir}")
            except Exception as e:
                logging.error(f"Failed to remove results directory {results_dir}: {e}")

        # Also clean up the task-level dict entry if it exists
        if task_id in context.inference_results:
            del context.inference_results[task_id]

        logging.info(f"Deleted {deleted_count} inference result chunks for task {task_id}")
        send_action(context, "inference_results_of_task_deleted", {
            "task_id": task_id,
            "deleted_count": deleted_count
        })

    thread = threading.Thread(target=worker, args=(task_id, context))
    thread.daemon = True
    thread.start()


@register("stop_inference_task")
async def handle_stop_inference_task(websocket: WebSocket, data: dict, context: Context):
    """Stop the currently running inference task.

    Cancels the task that is actively being processed.
    Does not remove queued tasks — only stops the running inference loop.

    Payload:
        task_id (str, optional): ID of the task to stop.
            If omitted, stops whatever is currently running.

    Response actions:
        task_cancelled: the running task was stopped
        no_task_running: no task was currently running
    """
    if not context.modelHandler:
        send_action(context, 'model_handler_not_loaded_error')
        logging.error('Model handler not loaded')
        return

    task_id = data.get("task_id")
    current = context.modelHandler.current_task

    # If a specific task_id was requested, verify it's the one running
    if task_id:
        if not current or current.get("id") != task_id:
            send_action(context, "task_stop_error", {
                "error": f"Task {task_id} is not currently running"
            })
            return

    if not current:
        send_action(context, "no_task_running", {
            "message": "No task is currently running"
        })
        return

    running_task_id = current.get("id")
    context.modelHandler.cancelled_tasks.add(running_task_id)
    context.modelHandler.running = False
    context.stop_event.set()

    logging.info(f"Stopping inference task {running_task_id}")
    send_action(context, "task_cancelled", {"id": running_task_id})
