import subprocess
import json
import logging
from fastapi import WebSocket
import uuid
import threading
import os
import time
from pathlib import Path
import pickle
from typing import TYPE_CHECKING
import cv2
import ffmpeg
from commons import send_action
import numpy as np

if TYPE_CHECKING:
    from context import Context

'''
Task object
    properties
        type: inference | train
        file_type = image | video
        image_files = [] list dir of image files on which inferece will run
        batch_size = int batch size for the task

        video_file = dir of video file
        target_frames = specefic frames on which inference will run. If None inference runs on full frame


        dataset: .zip file of the dataset on which training will run



inference task:
    runs inference sends inference progress through websocket. Saves inference result to specified destination (either send through websocket or save to mega)

trains task:
    train model, update progress, save to specified destination.
'''




class ModelHandler:
    def __init__(self, context):
        self.context = context
        self.tasks = []          # list of task payload dicts
        self.current_task = None # the task dict currently being processed
        self.running = False     # whether the queue processor is active
        self.cancelled_tasks = set()  # set of task IDs to skip

    def setup(self, payload, context):
        raise NotImplementedError("Subclasses must implement setup()")

    def init(self, payload, context):
        raise NotImplementedError("Subclasses must implement init()")

    def add_task(self, payload, context):
        payload["id"] = str(uuid.uuid4())[:16]
        self.tasks.append(payload)
        send_action(context, "task_added", payload)

    def delete_task(self, data, context):
        task_id = data.get("id")
        if not task_id:
            send_action(context, "task_delete_error", {"error": "No task id provided", "payload": data})
            return
        if self.current_task and self.current_task.get("id") == task_id:
            self.cancelled_tasks.add(task_id)
            self.running = False
            send_action(context, "task_cancelled", {"id": task_id})
            return
        for i, task in enumerate(self.tasks):
            if task.get("id") == task_id:
                self.tasks.pop(i)
                send_action(context, "task_deleted", {"id": task_id})
                return
        send_action(context, "task_not_found", {"id": task_id})

    def start_working(self, payload, context):
        if self.running:
            send_action(context, "already_working", {"message": "Already processing tasks"})
            return
        if not self.tasks:
            send_action(context, "queue_empty", {"message": "No tasks in queue"})
            return
        self.running = True
        thread = threading.Thread(target=self._process_queue, args=(context,))
        thread.daemon = True
        thread.start()
        send_action(context, "work_started", {"message": "Started processing task queue"})

    def _process_queue(self, context):
        while self.tasks and self.running:
            if context.stop_event.is_set():
                self.running = False
                break
            task = self.tasks.pop(0)
            task_id = task.get("id")
            if task_id in self.cancelled_tasks:
                self.cancelled_tasks.discard(task_id)
                continue
            self.current_task = task
            try:
                task_type = task.get("type", "inference")
                if task_type == "train":
                    self.handle_train_task(task, context)
                else:
                    self.handle_inference_task(task, context)
            except Exception as e:
                logging.error(f"Task {task_id} failed: {e}")
                send_action(context, "task_failed", {"id": task_id, "error": str(e)})
            finally:
                self.current_task = None
        self.running = False
        send_action(context, "queue_completed", {"message": "All tasks processed"})

    def handle_train_task(self, payload, context):
        raise NotImplementedError("Subclasses must implement handle_train_task()")

    def handle_inference_task(self, payload, context):
        raise NotImplementedError("Subclasses must implement handle_inference_task()")

    def destroy(self, payload, context):
        pass


class yoloHandler(ModelHandler):
    def __init__(self, context, payload):
        super().__init__(context)
        self.model = None
        self.model_name = ''

    def setup(self, payload, context):
        send_action(context, 'model_setup_started', {'model_name': self.model_name})
        subprocess.call(['pip', 'install', '-y', 'ultralytics'], shell=False)
        send_action(context, 'model_setup_completed', {'model_name': self.model_name})

    def init(self, payload, context):
        self.model_name = payload.get('variant_name', 'yolo26n') + '-seg'
        logging.debug(f'initiating model {self.model_name}')
        send_action(context, 'model_init_started', {'model_name': self.model_name})
        from ultralytics import YOLO
        self.model = YOLO(self.model_name)
        send_action(context, 'model_init_completed', {'model_name': self.model_name})

    def destroy(self, payload, context):
        pt_path = self.model_name + '.pt'
        if os.path.exists(pt_path):
            os.remove(pt_path)
        self.model = None
        send_action(context, 'model_destroyed', {'model_name': self.model_name})

    def add_task(self, payload, context):
        payload["id"] = str(uuid.uuid4())[:16]

        '''
        parameters for payload:

        file_id*
        file_type*
        frames
        conf
        iou
        classes
        imgsz
        persist
        tracker
        '''

        file_id = payload.get("file_id",None)
        file_type = payload.get("file_type",None)

        # check if file id is sent
        if not file_id:
            # file_id is required send error
            send_action(context, "create_inference_task_error", {"error": "no_file_id_provided"})
            return

        # check if file type is sent
        if not file_type:
            # file_id is required send error
            send_action(context, "create_inference_task_error", {"error": "no_file_id_provided"})
            return


        # check if file exists
        if not file_id in context.files_dict:
            # file in not found
            send_action(context, "create_infrerence_task_error", {"error": "file_not_found"})
            return
        
        # check if file type is valid

        if not (file_type == 'video' or file_type == 'image'):
            # file type is not valid for inference
            # send error
            send_action(context,'create_inference_task_error',{"error":"file_type_not_valid_error"})

        # all ok for now add the task in queue

        self.tasks.append(payload)
        send_action(context, "task_added", payload)

    def delete_task(self, data, context):
        task_id = data.get("id")
        if not task_id:
            send_action(context, "task_delete_error", {"error": "No task id provided"})
            return
        if self.current_task and self.current_task.get("id") == task_id:
            self.cancelled_tasks.add(task_id)
            self.running = False
            send_action(context, "task_cancelled", {"id": task_id})
            return
        for i, task in enumerate(self.tasks):
            if task.get("id") == task_id:
                self.tasks.pop(i)
                send_action(context, "task_deleted", {"id": task_id})
                return
        send_action(context, "task_not_found", {"id": task_id})

    def start_working(self, payload, context):
        if self.running:
            send_action(context, "already_working", {"message": "Already processing tasks"})
            return
        if not self.tasks:
            send_action(context, "queue_empty", {"message": "No tasks in queue"})
            return
        self.running = True
        thread = threading.Thread(target=self._process_queue, args=(context,))
        thread.daemon = True
        thread.start()
        send_action(context, "work_started", {"message": "Started processing task queue"})

    def handle_train_task(self, payload, context):
        task_id = payload.get("id")
        dataset = payload.get("dataset")
        epochs = payload.get("epochs", 100)
        batch_size = payload.get("batch_size", 16)
        imgsz = payload.get("imgsz", 640)
        workers = payload.get("workers", 8)
        device = payload.get("device", "cpu")
        project = payload.get("project", "runs/train")
        name = payload.get("name", f"exp_{task_id}")

        if not dataset:
            send_action(context, "task_failed", {"id": task_id, "error": "No dataset provided"})
            return

        dataset_path = dataset
        if isinstance(dataset, str) and dataset.endswith(".zip"):
            import zipfile
            extract_dir = f"datasets/{task_id}"
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(dataset, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            yaml_files = list(Path(extract_dir).rglob("data.yaml"))
            dataset_path = str(yaml_files[0]) if yaml_files else extract_dir

        send_action(context, "training_started", {"id": task_id, "epochs": epochs, "dataset": dataset_path})

        self.model.train(
            data=dataset_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            workers=workers,
            device=device,
            project=project,
            name=name,
            verbose=True
        )

        results_path = os.path.join(project, name)
        if not os.path.exists(results_path):
            train_dirs = sorted(Path(project).iterdir(), key=os.path.getmtime)
            if train_dirs:
                results_path = str(train_dirs[-1])

        send_action(context, "training_completed", {"id": task_id, "result_path": results_path})

    def handle_inference_task(self, payload, context):

        '''
        parameters for payload:

        file_id*
        file_type*
        frames
        conf
        iou
        classes
        imgsz
        persist
        tracker
        '''
              
        task_id = payload.get("id")
        file_type = payload.get("file_type", None)
        conf = payload.get("conf", 0.25)
        iou = payload.get("iou", 0.45)
        file_id = payload.get('file_id',None)

        if not file_type:
            send_action(context, "inference_task_error", {"id": task_id, 'error': 'FILE_TYPE_NOT_FOUND'})
            return
        if not file_id:
            send_action(context, "inference_task_error", {"id": task_id, 'error': 'FILE_ID_NOT_FOUND'})
            return

        save_dir = payload.get("save_dir", f"results/{task_id}")
        os.makedirs(save_dir, exist_ok=True)

        if file_type == "video":
            self._infer_video(payload, context, save_dir)
        elif file_type == "image":
            self._infer_image(payload, context, save_dir)
        else:
            send_action(context, "inference_task_error", {
                "id": task_id, "error": f"UNSUPPORTED_FILE_TYPE: {file_type}"
            })

    # ------------------------------------------------------------------
    #  Video inference
    # ------------------------------------------------------------------
    def _infer_video(self, payload, context, save_dir):
        # parse all parameters
        task_id = payload.get("id")
        file_id = payload.get("file_id")
        conf = payload.get("conf", 0.25)
        iou = payload.get("iou", 0.45)
        frames_param = payload.get("frames", None)
        imgsz = payload.get('imgsz',None)
        classes = payload.get('classes',None)

        # get file path from file id
        file_info = context.get_file_path_from_id(file_id)
        if not file_info:
            send_action(context, "inference_task_error", {"id": task_id, 'error': 'FILE_PATH_NOT_FOUND'})
            return
        file_path = file_info["path"] if isinstance(file_info, dict) else file_info

        logging.info(f'file_path: {file_path}')

        # Prepare output directories for training-ready dataset format
        dataset_dir = os.path.join(save_dir, "dataset")
        images_dir = os.path.join(dataset_dir, "images")
        masks_dir = os.path.join(dataset_dir, "masks")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        # Determine which frames to process
        if frames_param is not None:
            frame_indices = frames_param
            total = len(frame_indices)
            send_action(context, "inference_stage_plus_progress", {
                "task_id": task_id, "progress": 0,
                "stage": f"Processing {total} specific frames from video"
            })
        else:
            # Get total frame count first
            try:
                total_frames = self._get_frame_count(file_path)
                if total_frames is None:
                    raise ValueError("Could not count frames")
                frame_indices = list(range(total_frames))
                total = total_frames
            except Exception as e:
                send_action(context, "inference_task_error", {
                    "id": task_id, 'error': 'frame_count_failed', 'trace': str(e)
                })
                return

        # Temporal downsampling
        temporal_downsampling = payload.get('temporal_downsampling', None)
        if temporal_downsampling:
            drop_rate = payload.get('drop_rate', 0.01)
            if drop_rate > 1:
                drop_rate = 1
                send_action(context, "inference_task_warning", {
                    "task_id": task_id, "warning": "Drop rate is > 1. Setting to 1"
                })
            frame_indices = self._drop_items_evenly(frame_indices, drop_rate)

        # Chunk frame indices into batches — only `batch` frames in memory at once
        batch_size = payload.get('batch', 128)
        frame_batches = self._chunk_frames(frame_indices, batch_size)
        total_frames_to_process = len(frame_indices)
        processed_count = 0
        all_annotations = []
        chunk_index = 0

        # Track chunks in context for later retrieval
        if task_id not in context.inference_results:
            context.inference_results[task_id] = {}

        for batch_indices in frame_batches:
            if context.stop_event.is_set():
                send_action(context, "task_cancelled", {"id": task_id})
                logging.info('Task cancelled')
                return

            # --- Extract batch frames from video (memory-bound) ---
            logging.debug(f'{task_id}: Extracing frames for batch {chunk_index}')
            batch_frames = self._extract_frames_from_video(file_path, batch_indices)
            if not batch_frames:
                continue

            # --- Run inference on this batch ---
            logging.debug(f'{task_id}: Running inference for batch {chunk_index}')
            
            predict_kwargs = dict(conf=conf, iou=iou,verbose=True, classes=classes)
            if imgsz is not None:
                predict_kwargs['imgsz'] = imgsz
            # Separate frame indices and arrays — YOLO only accepts arrays
            
            batch_frame_indices = [idx for idx, _ in batch_frames]
            batch_frame_arrays = [arr for _, arr in batch_frames]

            batch_results = self.model.predict(batch_frame_arrays, **predict_kwargs)
            
            
            processed_count += len(batch_frames)
            progress = int((processed_count / total_frames_to_process) * 100)
            send_action(context, "inference_stage_plus_progress", {
                "task_id": task_id,
                "progress": progress,
                "stage": f"Processing frame {processed_count}/{total_frames_to_process}"
            })

            logging.debug(f'saving result as pickle')

            

            # --- Save batch result as pickle and register chunk ---
            chunk_id = str(uuid.uuid4())[:8]
            save_file = os.path.join(save_dir, f'chunk-{chunk_index}.pkl')
            with open(save_file, 'wb') as f:
                pickle.dump(batch_results, f)

            context.inference_results[chunk_id] = {
                'task_id': task_id,
                'chunk_index': chunk_index,
                'save_file': save_file,
                'frame_count': len(batch_results),
                'frames': batch_frame_indices
            }

            send_action(context, 'inference_task_chunk_result', {
                'chunk_id': chunk_id,
                'chunk_index': chunk_index,
                'task_id': task_id,
                'frame_count': len(batch_results)
            })

            chunk_index += 1
            # batch_frames goes out of scope → memory freed

        send_action(context, "inference_completed", {
            "task_id": task_id,
            "save_dir": save_dir,
            "result_count": len(all_annotations)
        })

    @staticmethod
    def _extract_frames_from_video(video_path, frame_indices):
        """Extract specific frames from a video file.
        
        Returns list of (frame_index, frame_numpy) tuples.
        Only loads requested frames into memory at once.
        
        Uses sequential scan only — seeking via cap.set(CAP_PROP_POS_FRAMES)
        breaks on HEVC/H.265 videos because reference frames are missing.
        """
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        frames = []
        frame_set = set(frame_indices)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in frame_set:
                frames.append((frame_idx, frame.copy()))
            frame_idx += 1

        cap.release()
        return frames

    # ------------------------------------------------------------------
    #  Image inference
    # ------------------------------------------------------------------
    def _infer_image(self, payload, context, save_dir):
        task_id = payload.get("id")
        file_ids = payload.get("file_ids", [])
        conf = payload.get("conf", 0.25)
        iou = payload.get("iou", 0.45)
        imgsz = payload.get('imgsz', None)
        classes = payload.get('classes', None)

        if not file_ids:
            send_action(context, "inference_task_error", {"id": task_id, 'error': 'NO_FILE_IDS_PROVIDED'})
            return

        # Resolve all file paths
        file_paths = []
        for fid in file_ids:
            file_info = context.get_file_path_from_id(fid)
            if not file_info:
                send_action(context, "inference_task_error", {"id": task_id, 'error': f'FILE_PATH_NOT_FOUND for {fid}'})
                return
            file_path = file_info["path"] if isinstance(file_info, dict) else file_info
            file_paths.append(file_path)

        total = len(file_paths)
        processed_count = 0
        chunk_index = 0

        # Track chunks in context for later retrieval
        if task_id not in context.inference_results:
            context.inference_results[task_id] = {}

        # Process in batches (same logic as video)
        batch_size = payload.get('batch', 128)
        file_batches = self._chunk_frames(file_paths, batch_size)

        for batch_paths in file_batches:
            if context.stop_event.is_set():
                send_action(context, "task_cancelled", {"id": task_id})
                return

            # --- Run inference on this batch of images ---
            predict_kwargs = dict(conf=conf, iou=iou, verbose=True, classes=classes)
            if imgsz is not None:
                predict_kwargs['imgsz'] = imgsz

            batch_results = self.model.predict(batch_paths, **predict_kwargs)

            processed_count += len(batch_paths)
            progress = int((processed_count / total) * 100)
            send_action(context, "inference_stage_plus_progress", {
                "task_id": task_id,
                "progress": progress,
                "stage": f"Processing image {processed_count}/{total}"
            })

            # --- Save batch result as pickle and register chunk ---
            chunk_id = str(uuid.uuid4())[:8]
            save_file = os.path.join(save_dir, f'chunk-{chunk_index}.pkl')
            with open(save_file, 'wb') as f:
                pickle.dump(batch_results, f)

            context.inference_results[chunk_id] = {
                'task_id': task_id,
                'chunk_index': chunk_index,
                'save_file': save_file,
                'frame_count': len(batch_results),
                'frames': list(range(processed_count - len(batch_paths), processed_count))
            }

            send_action(context, 'inference_task_chunk_result', {
                'chunk_id': chunk_id,
                'chunk_index': chunk_index,
                'task_id': task_id,
                'frame_count': len(batch_results)
            })

            chunk_index += 1

        send_action(context, "inference_completed", {
            "task_id": task_id,
            "save_dir": save_dir,
            "result_count": processed_count
        })

    # ------------------------------------------------------------------
    #  Annotation extraction helpers
    # ------------------------------------------------------------------
    def _extract_frame_annotation(self, result, frame_idx, images_dir, masks_dir,
                                   save_dir):
        """Convert a single YOLO result into a structured annotation dict,
        save the original image copy and mask images."""
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return None

        frame_key = f"frame_{frame_idx:06d}"
        img_filename = f"{frame_key}.jpg"
        img_path = os.path.join(images_dir, img_filename)
        orig_img = result.orig_img
        if orig_img is not None:
            cv2.imwrite(img_path, cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
        h, w = orig_img.shape[:2] if orig_img is not None else (0, 0)

        boxes = result.boxes
        masks_data = result.masks
        annotations = []

        for det_idx in range(len(boxes)):
            cls_id = int(boxes.cls[det_idx])
            cls_name = result.names.get(cls_id, f"class_{cls_id}")
            confidence = float(boxes.conf[det_idx])
            xyxy = boxes.xyxy[det_idx].cpu().numpy().tolist()
            track_id = int(boxes.id[det_idx]) if boxes.id is not None else None
            bbox = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]

            seg_polygons = []
            mask_save_path = None
            if masks_data is not None:
                seg_mask = masks_data.data[det_idx].cpu().numpy().astype(np.uint8)
                mask_filename = f"{frame_key}_det{det_idx:03d}_{cls_name}.png"
                mask_save_path = os.path.join(masks_dir, mask_filename)
                cv2.imwrite(mask_save_path, seg_mask * 255)
                if masks_data.xy is not None and len(masks_data.xy) > det_idx:
                    poly = masks_data.xy[det_idx]
                    if len(poly) > 0:
                        seg_polygons = poly.flatten().tolist()

            annotations.append({
                "id": det_idx,
                "image_id": frame_idx,
                "category_id": cls_id,
                "category_name": cls_name,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "segmentation": seg_polygons,
                "mask_path": mask_save_path,
                "confidence": confidence,
                "track_id": track_id
            })

        combined_mask = self._build_combined_mask(result, masks_data, h, w)
        combined_mask_path = None
        if combined_mask is not None:
            combined_mask_path = os.path.join(masks_dir, f"{frame_key}_combined.png")
            cv2.imwrite(combined_mask_path, combined_mask)

        return {
            "image_id": frame_idx,
            "image_file": img_filename,
            "image_path": img_path,
            "width": w,
            "height": h,
            "annotations": annotations,
            "combined_mask": combined_mask_path
        }

    def _build_combined_mask(self, result, masks_data, h, w):
        """Single mask image where each pixel = class_id+1 (0=background).

        Resizes mask from inference resolution to original image (h, w).
        """
        if masks_data is None or result.boxes is None:
            return None
        boxes = result.boxes
        combined = np.zeros((h, w), dtype=np.uint8)
        for det_idx in range(len(boxes)):
            cls_id = int(boxes.cls[det_idx])
            mask = masks_data.data[det_idx].cpu().numpy().astype(np.uint8)
            # Resize mask to match original image dimensions
            mask_h, mask_w = mask.shape
            if mask_h != h or mask_w != w:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            combined = np.where(mask > 0, cls_id + 1, combined)
        return combined

    def _extract_tracking(self, result):
        """Extract track IDs + bboxes from a YOLO result."""
        if result is None or result.boxes is None or result.boxes.id is None:
            return None
        boxes = result.boxes
        tracks = []
        for i in range(len(boxes)):
            tracks.append({
                "track_id": int(boxes.id[i]),
                "class_id": int(boxes.cls[i]),
                "class_name": result.names.get(int(boxes.cls[i]), "unknown"),
                "bbox": boxes.xyxy[i].cpu().numpy().tolist()
            })
        return tracks

    def _save_tracking_data(self, tracks, save_dir, frame_idx):
        tracking_file = os.path.join(save_dir, "tracking.jsonl")
        entry = {"frame": frame_idx, "tracks": tracks}
        with open(tracking_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _extract_reid(self, result, det_idx):
        """Placeholder for ReID feature extraction.
        YOLO does not natively produce re-identification features.
        Override this method with your own embedding extraction."""
        return None



    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _get_frame_count(video_path):
        try:
            
            cmd = [
                "ffprobe",
                "-v", "error",
                "-count_frames",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames",
                "-of", "json",
                video_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            data = json.loads(result.stdout)
            return int(data["streams"][0]["nb_read_frames"])

        except Exception as e:
            logging.error(f"Failed to get frame count: {e}")
            return None

    @staticmethod
    def _drop_items_evenly(data_list, drop_rate):
        if not 0 < drop_rate < 1:
            raise ValueError("Drop rate must be strictly between 0 and 1.")
        interval = 1 / drop_rate
        return [
            item for i, item in enumerate(data_list)
            if not abs(((i + 1) % interval)) < 1e-9
               and not abs(((i + 1) % interval) - interval) < 1e-9
        ]

    @staticmethod
    def _chunk_frames(frame_list, n):
        return [frame_list[i:i + n] for i in range(0, len(frame_list), n)]







