"""
Test scenario:
1. Download a file using Google Drive link
2. Disconnect client during downloading
3. Reconnect and check if pending messages are flushed
"""

import asyncio
import json
import websockets
import time

WS_URL = "ws://localhost:8000/ws"
GDRIVE_URL = "https://drive.google.com/file/d/1TtqAIblLhsAk7vOcVpBTrR3w2EDrZRcL/view?usp=drive_link"

async def test_disconnect_reconnect():
    print("=" * 60)
    print("🔌 TEST: Google Drive download with disconnect/reconnect")
    print("=" * 60)

    # --- Step 1: Connect ---
    print("\n[1] Connecting to WebSocket...")
    ws1 = await websockets.connect(WS_URL)
    print("✅ Connected (ws1)")
    
    # Wait a moment for the connection to establish
    await asyncio.sleep(0.5)

    # --- Step 2: Start Google Drive download ---
    print(f"\n[2] Sending download_file_google_drive action...")
    await ws1.send(json.dumps({
        "action": "download_file_google_drive",
        "payload": {
            "url": GDRIVE_URL
        }
    }))
    print("✅ Download action sent")

    # Give the download a moment to start and get some progress updates
    print("[3] Waiting for download to start (receiving messages for 5 seconds)...")
    download_started = False
    for i in range(10):  # Try to receive messages for up to 5 seconds
        try:
            msg = await asyncio.wait_for(ws1.recv(), timeout=1)
            data = json.loads(msg)
            action = data.get("action", "")
            payload = data.get("payload", {})
            
            if action == "file_download_initiated":
                download_started = True
                print(f"   📥 Download initiated: file_id={payload.get('file_id')}, path={payload.get('expected_path')}")
            elif action == "download_progress":
                print(f"   📊 Progress: {payload.get('progress', 'N/A')}% - ETA: {payload.get('eta', 'N/A')} - Size: {payload.get('total_size', 'N/A')}")
            else:
                print(f"   📨 Received: action={action}, payload={payload}")
        except asyncio.TimeoutError:
            print(f"   ⏱️  No message received in {i+1}s...")
    
    if not download_started:
        print("   ⚠️  Download may not have started. Checking pending messages on reconnect...")
    
    # --- Step 3: Disconnect ---
    print("\n[4] 💥 Disconnecting client (closing WebSocket)...")
    await ws1.close()
    print("✅ Disconnected")
    
    # Wait a bit while download continues in background
    print("[5] Waiting 3 seconds while download continues in background...")
    await asyncio.sleep(3)
    
    # --- Step 4: Reconnect ---
    print("\n[6] Reconnecting...")
    try:
        ws2 = await asyncio.wait_for(websockets.connect(WS_URL), timeout=10)
        print("✅ Reconnected (ws2)")
    except asyncio.TimeoutError:
        print("❌ Connection timed out. Server may be stuck.")
        return
    
    # Check if pending messages from the ongoing download are flushed
    print("[7] Checking for flushed pending messages (receiving for 15 seconds)...")
    pending_messages = []
    for i in range(15):  # Try for up to 15 seconds
        try:
            msg = await asyncio.wait_for(ws2.recv(), timeout=1)
            data = json.loads(msg)
            action = data.get("action", "")
            payload = data.get("payload", {})
            pending_messages.append(data)
            
            if action == "download_progress":
                print(f"   📊 Flushed progress: {payload.get('progress', 'N/A')}% - file_id={payload.get('file_id')}")
            elif action == "file_download_completed":
                print(f"   ✅ Flushed download completed: file_id={payload.get('file_id')}, path={payload.get('file_path')}")
            elif action == "download_failed":
                print(f"   ❌ Flushed download failed: {payload.get('error')}")
            else:
                print(f"   📨 Flushed message: action={action}, payload={payload}")
        except asyncio.TimeoutError:
            print(f"   ⏱️  No more messages ({i+1}s timeout)")
            break
    
    # --- Step 5: Summary ---
    print("\n" + "=" * 60)
    if pending_messages:
        print(f"✅ PASS: {len(pending_messages)} pending message(s) flushed after reconnect!")
        for m in pending_messages:
            a = m.get("action", "")
            p = m.get("payload", {})
            if a == "file_download_completed":
                print(f"   🎯 Download completed! file_id={p.get('file_id')}, path={p.get('file_path')}")
            elif a == "download_failed":
                print(f"   ❌ Download failed: {p.get('error')}")
    else:
        print("ℹ️  No pending messages received. This could mean:")
        print("   - Download completed during disconnection")
        print("   - Download hasn't started yet")
        print("   - Download failed before disconnect")
        print("   Check server logs for details.")
    print("=" * 60)
    
    if ws2:
        await ws2.close()
        print("✅ Test connection closed")

if __name__ == "__main__":
    asyncio.run(test_disconnect_reconnect())
