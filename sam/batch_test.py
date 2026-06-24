from pathlib import Path

import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAMImage,
    InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_PATHS = sorted(PROJECT_DIR.glob("img_[1-4].jpg"))
PROMPT = "vehicle"
BATCH_SIZE = 2


def make_datapoint(image: Image.Image, query_id: int) -> Datapoint:
    """Create one SAM3 datapoint containing one image and one text query."""
    width, height = image.size
    datapoint = Datapoint(
        find_queries=[
            FindQueryLoaded(
                query_text=PROMPT,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=query_id,
                    original_image_id=query_id,
                    original_category_id=1,
                    original_size=[height, width],
                    object_id=0,
                    frame_index=0,
                ),
            )
        ],
        images=[SAMImage(data=image, objects=[], size=[height, width])],
    )
    return datapoint


def main() -> None:
    if not IMAGE_PATHS:
        raise FileNotFoundError("No img_1.jpg ... img_4.jpg files were found")

    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=1008,
                max_size=1008,
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=0.5,
        to_cpu=True,
    )

    model = build_sam3_image_model(
        checkpoint_path=PROJECT_DIR / "hf/sam3.1_multiplex_mapped.pt"
    )
    model = model.to("cuda").eval()

    for start in range(0, len(IMAGE_PATHS), BATCH_SIZE):
        paths = IMAGE_PATHS[start : start + BATCH_SIZE]
        datapoints = []
        query_ids = []

        for query_id, path in enumerate(paths, start=start + 1):
            with Image.open(path) as image:
                datapoints.append(make_datapoint(image.convert("RGB"), query_id))
            query_ids.append(query_id)

        datapoints = [transform(datapoint) for datapoint in datapoints]
        batch = collate(datapoints, dict_key="batch")["batch"]
        batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            output = model(batch)

        results = postprocessor.process_results(output, batch.find_metadatas)
        for path, query_id in zip(paths, query_ids):
            result = results[query_id]
            print(f"{path.name}: {len(result['scores'])} detections")
            print("boxes:", result["boxes"])
            print("scores:", result["scores"])


if __name__ == "__main__":
    main()