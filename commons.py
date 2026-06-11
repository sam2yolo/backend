import json
import logging
from returnablequeue import WorkObject

def send_action(context, action, payload={}):
    """Queue a message for async delivery to the active WebSocket.
    
    This is sync (no await) and can be called from any thread. The message
    will be delivered by the response dispatcher thread when a WebSocket is active.
    """
    payload['worker_id'] = context.id 
    logging.debug(f"send_action: action: {action}")
    obj = WorkObject(data=json.dumps({"action":action,'payload':payload}))
    context.send_action_queue.put(obj)
    

