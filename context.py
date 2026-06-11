# This is a data context holder for the application
# This contains only the global variables and constants used in the application

import uuid
import json
import queue
import threading
import asyncio
from returnablequeue import WorkObject
from returnablequeue import ReturnableQueue
import logging

class Context:
    def __init__(self):
        self.models_dict = {}
        self.files_dict = {}
        self.id = None
        self.send_action_queue:ReturnableQueue = ReturnableQueue()
        # Internal queue for workers to submit responses without touching the websocket
        self.response_queue: queue.Queue = queue.Queue()

        self.model_backend_base_url = "http://localhost:8001"  # base URL for the model backend server
        
        self.modelHandler = None
        self._active_websocket = None

        # Event to signal running workers to stop (e.g., on disconnect)
        self.stop_event = threading.Event()

        # Pending responses that couldn't be delivered (no active websocket)
        self.pending_responses = []
        self._pending_lock = threading.Lock()

        # Reference to the asyncio event loop (set by app.py on startup)
        self._loop = None

        '''
        mega_credentials: a list of dictionaries containing the credentials for mega.nz accounts to be used for downloading files from mega.nz links, each dictionary should have the following keys: "email" and "password"
        Example: [{"email": "mymail@mymail.com", "password": "mypassword"}, {"email": "mysecondmail@mymail.com", "password": "mysecondpassword"}]
        '''
        
        self.mega_credentials = []
        self.inference_tasks = {}
        self.inference_results = {}

        self.load_models()

        # generate a unique 16 char UUID for this server instance to identify it in the model backend server
        self.id = str(uuid.uuid4())[:16]

        # Shared thread pool for long-running workers
        self._worker_threads = []
        self._running = False

    def get_inference_result_path(self, id):
        if not id in self.inference_results:
            return None
        
        return self.inference_results[id]['save_file']

    def start_worker(self):
        """Start the response dispatcher thread.
        
        This thread reads from both the internal response_queue and the
        send_action_queue (ReturnableQueue), and dispatches messages to the
        active WebSocket via the asyncio event loop.
        """
        if self._running:
            return
        self._running = True
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._response_dispatcher, daemon=True)
        self.worker_thread.start()
        logging.info("Response dispatcher worker started")

    def stop_worker(self):
        """Signal the response dispatcher to stop."""
        self._running = False
        self.stop_event.set()
        self.response_queue.put(None)  # unblock if waiting
        logging.info("Response dispatcher worker stopping")

    def _response_dispatcher(self):
        """Bridge between worker threads and the async WebSocket.
        
        Reads messages from response_queue and send_action_queue, then
        dispatches them to the active WebSocket via the asyncio loop.
        If no WebSocket is active, messages are stored in pending_responses
        and flushed when a new connection arrives.
        """
        while self._running:
            # Check send_action_queue first
            msg = self.send_action_queue.unload()
            if msg is None:
                # Check internal response queue
                try:
                    msg = self.response_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
            
            if msg is None:
                continue

            # Try to send to active websocket
            ws = self._active_websocket
            if ws is not None and self._loop is not None:
                try:
                    # Verify websocket is in CONNECTED state
                    if hasattr(ws, 'client_state') and ws.client_state.name != 'CONNECTED':
                        logging.debug("WebSocket not connected, queuing message for later delivery")
                        with self._pending_lock:
                            self.pending_responses.append(msg)
                        self.send_action_queue.return_object(msg, success=False)
                        continue
                    
                    # Schedule the async send on the event loop
                    future = asyncio.run_coroutine_threadsafe(
                        ws.send_text(msg.data), self._loop
                    )
                    future.result(timeout=10)  # wait for send to complete
                    logging.debug(f"Sent message: {msg.data}")
                    self.send_action_queue.return_object(msg, success=True)
                except Exception as e:
                    logging.error(f'Error sending message via websocket: {type(e).__name__}: {e}')
                    # Don't retry - store as pending
                    with self._pending_lock:
                        self.pending_responses.append(msg)
                    self.send_action_queue.return_object(msg, success=False)
            else:
                # No active websocket — store for later delivery
                with self._pending_lock:
                    self.pending_responses.append(msg)
                logging.debug(f"Queued message (no active websocket): {msg.data[:60]}...")

    async def flush_pending_responses(self):
        """Send all queued pending responses to the newly connected websocket.

        Called from the async WebSocket handler (event loop thread), so we
        await ws.send_text() directly instead of using run_coroutine_threadsafe
        (which would deadlock when called from the same event loop).

        Failed sends are put back into pending_responses for a later retry.
        """
        ws = self._active_websocket
        if ws is None or self._loop is None:
            return

        # Check if websocket is still connected before attempting to flush
        try:
            if not hasattr(ws, 'client_state') or ws.client_state.name != 'CONNECTED':
                logging.warning("WebSocket not in CONNECTED state, deferring flush")
                return
        except Exception as e:
            logging.warning(f"Could not verify WebSocket state: {e}")
            return

        with self._pending_lock:
            pending = self.pending_responses.copy()
            self.pending_responses.clear()

        still_failed = []
        for msg in pending:
            try:
                await ws.send_text(msg.data)
                logging.debug(f"Flushed pending message: {msg.data[:60]}...")
            except Exception as e:
                logging.error(f'Error flushing pending message: {type(e).__name__}: {e}')
                still_failed.append(msg)

        # Put back any that failed so they can be retried later
        if still_failed:
            with self._pending_lock:
                self.pending_responses.extend(still_failed)
            logging.info(f"Re-queued {len(still_failed)} pending messages for later delivery")

    @property
    def active_websocket(self):
        return self._active_websocket
    
    @active_websocket.setter
    def active_websocket(self, ws):
        self._active_websocket = ws

    def set_loop(self, loop):
        """Store the asyncio event loop for use by background threads."""
        self._loop = loop

    def load_models(self):
        # lazy import to avoid circular dependency
        from modelhandler import yoloHandler
        self.models_dict['yolo'] = yoloHandler

    def get_file_path_from_id(self, id):
        return self.files_dict.get(id, None)
    







