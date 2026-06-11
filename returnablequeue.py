import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
import uuid

class ObjectStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class WorkObject:
    data: str
    id: int = field(default_factory=lambda: uuid.uuid4().int)
    status: ObjectStatus = ObjectStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3

class ReturnableQueue:
    """Queue that allows returning objects to their original position"""
    
    def __init__(self):
        self.queue = queue.Queue()
        self.processing_objects = {}  # Track objects being processed
        self.lock = threading.Lock()
    
    def put(self, obj: WorkObject):
        """Add object to queue"""   

        self.queue.put(obj)
        print(f"📥 Added to queue: {obj}")
    
    def unload(self, timeout: Optional[float] = None) -> Optional[WorkObject]:
        """Remove object from queue for processing"""
        try:
            obj = self.queue.get(timeout=timeout)
            obj.status = ObjectStatus.PROCESSING
            
            with self.lock:
                self.processing_objects[obj.id] = obj
            
            print(f"📤 Unloaded: {obj}")
            return obj
        except queue.Empty:
            return None
    
    def return_object(self, obj: WorkObject, success: bool = True):
        """Return object to queue if needed"""
        with self.lock:
            if obj.id in self.processing_objects:
                del self.processing_objects[obj.id]
        
        if success:
            obj.status = ObjectStatus.COMPLETED
            print(f"✅ Completed: {obj}")
            # Don't put back completed objects
        else:
            obj.retry_count += 1
            if obj.retry_count <= obj.max_retries:
                obj.status = ObjectStatus.PENDING
                self.queue.put(obj)
                print(f"🔄 Returned to queue (retry {obj.retry_count}/{obj.max_retries}): {obj}")
            else:
                obj.status = ObjectStatus.FAILED
                print(f"❌ Failed after {obj.max_retries} retries: {obj}")
    
    def is_empty(self) -> bool:
        return self.queue.empty() and len(self.processing_objects) == 0

# Example usage
if __name__ == "__main__":
    basic_example()
    print("\nAll objects processed!")
    
    print("\nAll objects processed!")

def basic_example():
    print("=== Basic Returnable Queue Example ===\n")
    
    rq = ReturnableQueue()
    
    # Add objects to queue
    for i in range(5):
        rq.put(WorkObject(id=i, data=f"Task-{i}"))
    
    print(f"\nQueue size: {rq.queue.qsize()}")
    
    # Process objects
    while not rq.is_empty():
        obj = rq.unload(timeout=1)
        if obj:
            # Simulate processing
            time.sleep(0.5)
            
            # Decide whether to return (e.g., if failed)
            if obj.id % 3 == 0:  # Return every 3rd object
                print(f"⚠️  Processing failed for {obj.data}")
                rq.return_object(obj, success=False)
            else:
                print(f"✓ Successfully processed {obj.data}")
                rq.return_object(obj, success=True)

if __name__ == "__main__":
    basic_example()
    print("\nAll objects processed!")