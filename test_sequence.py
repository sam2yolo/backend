"""
Test sequence:
1. init model named `yolo`
2. download file from google drive
3. create inference task
4. run inference task
5. check log and check if inference task is completed
"""

import asyncio
import json
import websockets
import time
import sys

WS_URL = "ws://localhost:8000/ws"
GDRIVE_URL = "https://drive.google.com/file/d/1TtqAIblLhsAk7vOcVpBTrR3w2EDrZRcL/view?usp=drive_link"

async def recv_with_timeout(ws, timeout=5):
    """Receive a message with timeout."""
    try:
        msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
        return json.loads(msg)
    except asyncio.TimeoutError:
        return None

async def test_sequence():
    print("=" * 70)
    print("🧪 SEQUENCE TEST: Init → Download → Create Task → Run Inference")
    print("=" * 70)

    # --- Step 1: Connect ---
    print("\n[1] Connecting to WebSocket...")
    ws = await websockets.connect(WS_URL)
    print("✅ Connected")
    
    # Consume any initial messages (e.g., flushed pending from previous)
    await asyncio.sleep(0.5)
    while True:
        msg = await recv_with_timeout(ws, timeout=0.5)
        if msg is None:
            break
        print(f"   ⏪ Flushed: action={msg.get('action')}")

    # --- Step 2: Init model "yolo" ---
    print("\n[2] Initializing model 'yolo'...")
    await ws.send(json.dumps({
        "action": "init_model",
        "payload": {"model_name": "yolo", "variant_name": "yolo26n"}
    }))
    print("   ✅ init_model action sent")

    # Wait for model init responses
    print("   Waiting for model setup/init responses...")
    init_completed = False
    for _ in range(60):  # up to 60 seconds
        msg = await recv_with_timeout(ws, timeout=3)
        if msg is None:
            continue
        action = msg.get("action", "")
        payload = msg.get("payload", {})
        print(f"   📨 Received: action={action}")
        
        if action == "model_setup_completed":
            print(f"      ✅ Setup completed for model: {payload.get('model_name')}")
        elif action == "model_init_completed":
            print(f"      ✅ Init completed for model: {payload.get('model_name')}")
            init_completed = True
            break
        elif action == "model_init_error" or action == "model_setup_started":
            pass  # continue waiting
    
    if not init_completed:
        print("❌ Model init not completed within timeout")
        await ws.close()
        return False

    # --- Step 3: Download file from Google Drive ---
    print("\n[3] Downloading file from Google Drive...")
    await ws.send(json.dumps({
        "action": "download_file_google_drive",
        "payload": {"url": GDRIVE_URL}
    }))
    print("   ✅ download_file_google_drive action sent")

    # Wait for download to complete
    file_id = None
    download_completed = False
    print("   Waiting for download to complete...")
    for _ in range(60):  # up to 60 seconds
        msg = await recv_with_timeout(ws, timeout=3)
        if msg is None:
            continue
        action = msg.get("action", "")
        payload = msg.get("payload", {})
        
        if action == "file_download_initiated":
            file_id = payload.get("file_id")
            print(f"   📥 Download initiated: file_id={file_id}, path={payload.get('expected_path')}")
        elif action == "download_progress":
            progress = payload.get("progress")
            print(f"   📊 Download progress: {progress}%", end="\r")
        elif action == "file_download_completed":
            file_id = payload.get("file_id")
            file_path = payload.get("file_path")
            print(f"\n   ✅ Download completed: file_id={file_id}, path={file_path}")
            download_completed = True
            break
        elif action == "download_failed":
            print(f"\n   ❌ Download failed: {payload.get('error')}")
            await ws.close()
            return False

    if not download_completed or not file_id:
        print("❌ Download did not complete")
        await ws.close()
        return False

    # --- Step 4: Create inference task ---
    print("\n[4] Creating inference task...")
    
    # Get video file info - check actual duration to pick frames
    import subprocess
    file_path = f"files/{file_id}"
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", file_path],
        capture_output=True, text=True
    )
    duration_str = result.stdout.strip()
    try:
        duration = float(duration_str)
        print(f"   Video duration: {duration:.2f}s")
    except:
        duration = 0
        print("   Could not determine video duration")

    # Get frame count
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_packets", "-show_entries", "stream=nb_read_packets",
         "-of", "default=noprint_wrappers=1:nokey=1", file_path],
        capture_output=True, text=True
    )
    frame_count_str = result.stdout.strip()
    try:
        total_frames = int(frame_count_str)
        print(f"   Total frames: {total_frames}")
    except:
        total_frames = 0
        print("   Could not determine frame count")

    # For a short video, we can process a subset of frames
    # If the video is short (< 30 frames), use all frames
    # Otherwise, sample every Nth frame
    if total_frames > 0:
        if total_frames <= 30:
            frame_list = list(range(total_frames))
        else:
            # Take 20 evenly spaced frames
            step = max(1, total_frames // 20)
            frame_list = list(range(0, total_frames, step))[:20]
    else:
        frame_list = list(range(0, 200, 10))  # default sample

    print(f"   Using {len(frame_list)} frames for inference")

    inference_payload = {
        "file_id": file_id,
        "file_type": "video",
        "frames": frame_list,
        "conf": 0.25,
        "iou": 0.45,
        "batch": 10
    }

    await ws.send(json.dumps({
        "action": "create_inference_task",
        "payload": inference_payload
    }))
    print("   ✅ create_inference_task action sent")

    # Wait for task_added confirmation
    task_id = None
    for _ in range(10):
        msg = await recv_with_timeout(ws, timeout=2)
        if msg is None:
            continue
        action = msg.get("action", "")
        payload = msg.get("payload", {})
        print(f"   📨 Received: action={action}")
        if action == "task_added":
            task_id = payload.get("id")
            print(f"      ✅ Task added: id={task_id}")
            break
        elif action == "model_handler_not_loaded_error":
            print("      ❌ Model handler not loaded")
            await ws.close()
            return False
        elif action == "create_infrerence_task_error" or action == "create_inference_task_error":
            print(f"      ❌ Task creation error: {payload}")
            await ws.close()
            return False

    if not task_id:
        print("❌ Task was not created")
        await ws.close()
        return False

    # --- Step 5: Run inference task ---
    print(f"\n[5] Running inference task {task_id}...")
    await ws.send(json.dumps({
        "action": "start_inference_from_queue",
        "payload": {}
    }))
    print("   ✅ start_inference_from_queue action sent")

    # Wait for inference to complete
    inference_completed = False
    print("   Waiting for inference to complete...")
    for _ in range(120):  # up to 120 seconds
        msg = await recv_with_timeout(ws, timeout=3)
        if msg is None:
            continue
        action = msg.get("action", "")
        payload = msg.get("payload", {})
        
        if action == "work_started":
            print(f"   🔄 Work started: {payload.get('message')}")
        elif action == "inference_stage_plus_progress":
            print(f"   📊 Inference progress: {payload.get('progress')}% - {payload.get('stage')}")
        elif action == "inferece_task_chunk_result":
            print(f"   📦 Chunk result: chunk_id={payload.get('chunk_id')}, task_id={payload.get('task_id')}")
        elif action == "inference_completed":
            print(f"   ✅ Inference completed! save_dir={payload.get('save_dir')}")
            inference_completed = True
            break
        elif action == "task_failed":
            print(f"   ❌ Task failed: {payload.get('error')}")
            await ws.close()
            return False
        elif action == "task_cancelled":
            print(f"   ❌ Task cancelled")
            await ws.close()
            return False
        elif action == "inference_task_error":
            print(f"   ❌ Inference error: {payload.get('error')}")
            await ws.close()
            return False

    if not inference_completed:
        print("❌ Inference did not complete within timeout")
        await ws.close()
        return False

    # --- Done ---
    print("\n" + "=" * 70)
    print("✅ SEQUENCE TEST PASSED!")
    print("=" * 70)
    
    await ws.close()
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_sequence())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
