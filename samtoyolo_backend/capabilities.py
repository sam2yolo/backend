from __future__ import annotations


INFERENCE_MODELS = [
    {
        "name": "sam3",
        "display_name": "Meta SAM 3.1",
        "task_type": "segmentation",
        "status": "model_server_adapter",
        "text_prompt": True,
        "default_artifact": "sam3.1.zip",
        "server_kind": "sam3",
    },
    {
        "name": "yolov8",
        "display_name": "YOLOv8",
        "task_type": "detection",
        "status": "adapter_required",
        "text_prompt": False,
    },
]

TRAINING_MODELS = [
    {"name": "yolov8", "display_name": "YOLOv8", "framework": "ultralytics"},
    {"name": "rt_detr", "display_name": "RT-DETR", "framework": "pytorch"},
    {"name": "grounding_dino", "display_name": "Grounding DINO", "framework": "pytorch"},
    {"name": "efficientdet", "display_name": "EfficientDet", "framework": "pytorch"},
    {"name": "ssd", "display_name": "SSD", "framework": "pytorch"},
    {"name": "retinanet", "display_name": "RetinaNet", "framework": "pytorch"},
    {"name": "faster_rcnn", "display_name": "Faster R-CNN", "framework": "pytorch"},
    {"name": "detectron2", "display_name": "Detectron2", "framework": "detectron2"},
    {"name": "mediapipe", "display_name": "MediaPipe Model Maker", "framework": "mediapipe"},
]

TRAINING_MODEL_NAMES = {model["name"] for model in TRAINING_MODELS}
