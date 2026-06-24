import torch
from pathlib import Path
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

PROJECT_DIR = Path(__file__).resolve().parent

# Load the model
model = build_sam3_image_model(
    checkpoint_path=PROJECT_DIR / "hf/sam3.1_multiplex_mapped.pt"
)
model.to(device="cuda")
processor = Sam3Processor(model)
# Load an image
image = Image.open(PROJECT_DIR / "img.jpg")
with torch.autocast("cuda", dtype=torch.float16):
    inference_state = processor.set_image(image)
    # Prompt the model with text
    output = processor.set_text_prompt(state=inference_state, prompt="vehicle")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]



print(boxes)
print(scores)
