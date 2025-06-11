import torch
import sys
import numpy as np
import cv2 as cv
import random
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
from PIL import Image

from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, DDIMScheduler
from huggingface_hub import hf_hub_download

from photomaker import PhotoMakerStableDiffusionXLPipeline
import torchvision.transforms as transforms

import argparse
import json

device = "cuda:0"

# ============== freecure package =================
# load the configuration files
with open("../freecure/cfg.json", "r") as load_f:
    cfg_freecure = json.load(load_f)
    machine = 'local_machine'
sys.path.append(cfg_freecure[machine]['repo']) # the package location of freecure
# freecure package loading
from freecure.fasa_module.fasa import (
    FoundationAwareSelfAttention, 
    register_fasa_to_model, 
    trace_attention_module
)
# face parsing models
from freecure.face_parsing.bisenet.utils import build_bisenet_model
bisenet = build_bisenet_model(cfg_freecure[machine]['bisenet'], torch.device(device)) # bisenet face parsing model
from freecure.face_parsing.efficientvit.utils import build_yolo_segment_model
yolo_model, sam = build_yolo_segment_model(cfg_freecure[machine]['sam'], torch.device(device)) # segment anything
from freecure.face_parsing.utils import resize_mask, mask_from_numpy_to_torch, obtain_mask_from_pillow_form_image, obtain_map_of_specific_parts, vis_parsing_maps, merge_masks

def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--pretrained_sdxl_model', default='stabilityai/stable-diffusion-xl-base-1.0', type=str)
    parser.add_argument('--output-folder', default='', type=str)
    parser.add_argument('--reference-folder', default='', type=str)
    parser.add_argument('--prompt', default='', type=str)
    parser.add_argument('--seed', default=20, type=int)
    return parser.parse_args()

args = parse_args()

# an example of bisetnet face parsing model's label correspondence
attribute_lut = {
    "hair": [17],
    "eye": [4, 5],
    "glasses": [6],
}

# base_model_path = 'SG161222/RealVisXL_V3.0'
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

save_path = args.output_folder
os.makedirs(save_path, exist_ok=True)
input_folder_name = args.reference_folder
output_folder_label = os.path.basename(input_folder_name)
target_output_folder = os.path.join(save_path, output_folder_label)
os.makedirs(target_output_folder, exist_ok=True)

num_images = 1 # 生成多少张图片

prompt = args.prompt
target_attribute = []
for key, value in attribute_lut.items():
    if key in prompt:
        target_attribute.extend(value)
negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch, monochrome)"
from huggingface_hub import hf_hub_download

photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")

pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="fp16",
).to(device)

pipe.load_photomaker_adapter(
    os.path.dirname(photomaker_ckpt),
    subfolder="",
    weight_name=os.path.basename(photomaker_ckpt),
    trigger_word="img"
)
pipe.id_encoder.to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe.set_adapters(["photomaker"], adapter_weights=[1.0])
pipe.fuse_lora()

# load the reference images
image_basename_list = os.listdir(input_folder_name)
image_path_list = sorted([os.path.join(input_folder_name, basename) for basename in image_basename_list])
input_id_images = []
for image_path in image_path_list:
    input_id_images.append(load_image(image_path))

num_steps = 50
style_strength_ratio = 20
start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
if start_merge_step > 30:
    start_merge_step = 30


print("********* Generation Information *********")
print(f"Prompt: {prompt}")
print(f"Negative Prompt: {negative_prompt}")
print(f"Reference Input: {input_folder_name}")
print(f"Target Attribute: {target_attribute}")
print("******************************************")
generator = torch.Generator(device=device).manual_seed(args.seed)

images_contents = pipe(
    prompt=prompt,
    input_id_images=input_id_images,
    negative_prompt=negative_prompt,
    num_images_per_prompt=num_images,
    num_inference_steps=num_steps,
    start_merge_step=start_merge_step,
    generator=generator,
)
images = images_contents.images

image_personalized_v1 = images[0]
image_base_v1 = images[1]

# 第二步，生成base 和 personalized的face parsing的map
guide_mask_base = obtain_mask_from_pillow_form_image(image_base_v1, bisenet)
guide_mask_base_binary = obtain_map_of_specific_parts(guide_mask_base, target_attribute) # hair: 17, mouth: 11 12 13, glasses: 6

guide_mask_personalized = obtain_mask_from_pillow_form_image(image_personalized_v1, bisenet)
guide_mask_personalized_binary = obtain_map_of_specific_parts(guide_mask_personalized, target_attribute) # hair

# # obtain the final guide mask
guide_mask_binary = merge_masks([guide_mask_base_binary, guide_mask_personalized_binary])

# Second Inference
attr_mask_foundation_target_mask = resize_mask(guide_mask_base_binary)
attr_mask_foundation_target_mask = mask_from_numpy_to_torch(attr_mask_foundation_target_mask).half().cuda() # 修正mask的格式，形状和数据类型都要align
# attr_mask_foundation_target_mask = torch.ones_like(attr_mask_foundation_target_mask)
freecure_attention = FoundationAwareSelfAttention(
    start_step = 0,
    end_step = 50, 
    layer_idx = list(range(50, 70, 1)),
    ref_masks = [attr_mask_foundation_target_mask],
    mask_weights = [1.5],
    style_fidelity = 1
)
register_fasa_to_model(pipe, freecure_attention)
print("FreeCure attention modules successfully modified")
generator = torch.Generator(device=device).manual_seed(args.seed) # remember to reset the identical seeds for generation
images_contents = pipe(
    prompt=prompt,
    input_id_images=input_id_images,
    negative_prompt=negative_prompt,
    num_images_per_prompt=num_images,
    num_inference_steps=num_steps,
    start_merge_step=start_merge_step,
    generator=generator,
)
# import pdb;pdb.set_trace()
images = images_contents.images
image_personalized_v2 = images[0]
image_base_v2 = images[1]

image_base_v1.save(os.path.join(target_output_folder, f"{args.prompt}_{args.seed}_foundation.png"))
image_personalized_v1.save(os.path.join(target_output_folder, f"{args.prompt}_{args.seed}_personalization.png"))
image_personalized_v2.save(os.path.join(target_output_folder, f"{args.prompt}_{args.seed}_refined_with_freecure.png"))

print("Generation Finished")
