import requests
import os
import argparse
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from sam2.sam2_image_predictor import SAM2ImagePredictor

def save_mask_as_rgb(mask: np.ndarray, output_path: str):
    """
    Convert a (1, H, W) binary mask to an RGB image and save it.
    Output will be white mask on black background.
    
    Args:
        mask: Binary mask numpy array of shape (1, height, width)
        output_path: Path to save the output image (e.g., 'output.png')
    """
    # Ensure binary (0 or 1) and convert to uint8
    mask = (mask.squeeze(0) > 0).astype(np.uint8) * 255
    
    # Create RGB by repeating the mask across 3 channels
    mask_rgb = np.stack([mask]*3, axis=-1)
    
    # Save
    Image.fromarray(mask_rgb).save(output_path)

# grounding dino inference
def grounding_dino_inference(processor, model, image_path, prompt):
    """
    Perform inference using the Grounding DINO model.
    This function is used to detect objects in an image and return their bounding boxes.
    The prompt should be "text queries need to be lowercased + end with a dot", aligning with the official guide
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    # Check for cats and remote controls
    # VERY important: prompt queries need to be lowercased + end with a dot
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch.device("cuda"))
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    boxes = []
    for subject in results:
        boxes.append(subject["boxes"].cpu().numpy())
    boxes = np.concatenate(boxes, axis=0)
    return image, image_np, boxes

def sam_2_inference(predictor, image_np, boxes):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image_np)
        masks, _, _ = predictor.predict(
            box=boxes,
            multimask_output=False
        )

    if len(masks.shape) == 3:
        masks = np.expand_dims(masks, axis=0)  # Add batch dimension if missing
    return masks

def extract_object_to_white_bg(image: np.ndarray, 
                             mask: np.ndarray, 
                             padding: int = 0) -> np.ndarray:
    """
    Extract object from image using binary mask and place on white background.
    
    Args:
        image: Input image array (h, w, 3)
        mask: Binary mask array (1, h, w) with values 0 or 1
        padding: Additional pixels around the object (default: 0)
    
    Returns:
        Extracted object on white background (new_h, new_w, 3)
    """
    # Remove channel dimension from mask and ensure binary
    mask = np.squeeze(mask, axis=0) > 0

    # import pdb;pdb.set_trace()
    
    # Find bounding box coordinates
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # Apply padding (clipped to image boundaries)
    h, w = mask.shape
    ymin = max(0, ymin - padding)
    ymax = min(h-1, ymax + padding)
    xmin = max(0, xmin - padding)
    xmax = min(w-1, xmax + padding)
    
    # Calculate object dimensions
    obj_height = ymax - ymin + 1
    obj_width = xmax - xmin + 1
    
    # Create white background
    result = np.ones((obj_height, obj_width, 3), dtype=np.uint8) * 255
    
    # Extract object region and mask
    obj_region = image[ymin:ymax+1, xmin:xmax+1]
    obj_mask = mask[ymin:ymax+1, xmin:xmax+1]
    
    # Place object on white background
    result[obj_mask] = obj_region[obj_mask]

    # import pdb;pdb.set_trace()
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Run Grounding DINO + SAM2 pipeline")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt, The prompt should be 'text queries need to be lowercased + end with a dot', aligning with the official guide")
    
    args = parser.parse_args()
    
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # grounding dino inference
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    # sam inference
    # multiinstance please refer to https://github.com/facebookresearch/sam2/issues/267
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

    # data 
    image_path = args.image_path
    prompt = args.prompt

    output_id = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs("output_masks", exist_ok=True)
    os.makedirs(os.path.join("output_masks", output_id), exist_ok=True)

    image, image_np, boxes = grounding_dino_inference(processor, model, image_path, prompt)
    print(f"Boxes detected: {boxes.shape[0]}")
    if boxes.shape[0] == 0:
        print("No objects detected. Exiting.")
        return
    
    # import pdb;pdb.set_trace()
    
    masks = sam_2_inference(predictor, image_np, boxes)

    # save image and masks
    # import pdb;pdb.set_trace()
    for idx in range(masks.shape[0]):
        mask = masks[idx]
        save_mask_as_rgb(mask, os.path.join("output_masks", output_id, f"mask_{idx}.png"))
        extraxt_subject = extract_object_to_white_bg(image_np, mask, padding=0)
        Image.fromarray(extraxt_subject).save(os.path.join("output_masks", output_id, f"subject_{idx}.png"))
    # save original image
    image.save(os.path.join("output_masks", output_id, "original_image.png"))
    
main()