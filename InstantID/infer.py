import os
import json
import copy
import sys
import cv2
import torch
import numpy as np
import argparse
from PIL import Image

from diffusers.utils import load_image
from diffusers.models import ControlNetModel

import torchvision.transforms as transforms

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

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
    parser.add_argument('--prompt', default='a man', type=str)
    parser.add_argument('--id-image', default='', type=str)
    parser.add_argument('--seed', default=25, type=int)
    return parser.parse_args()

# an example of bisetnet face parsing model's label correspondence
attribute_lut = {
    "hair": [17],
    "eye": [4, 5],
    "glasses": [6],
}

if __name__ == "__main__":

    args = parse_args()

    # Load face encoder
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Path to InstantID models
    face_adapter = f'./checkpoints/ip-adapter.bin'
    controlnet_path = f'./checkpoints/ControlNetModel'

    # Load pipeline
    # import pdb;pdb.set_trace()
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

    base_model_path = 'stabilityai/stable-diffusion-xl-base-1.0'

    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    pipe.cuda()
    pipe.load_ip_adapter_instantid(face_adapter)

    # Infer setting
    prompt = args.prompt
    target_attribute = []
    for key, value in attribute_lut.items():
        if key in prompt:
            target_attribute.extend(value)
    n_prompt = "monochromatic, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured, black-and-white"

    face_image = load_image(args.id_image)
    face_image = resize_img(face_image)

    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
    # import pdb;pdb.set_trace()
    face_emb = face_info['embedding']
    face_emb_no_id = np.zeros_like(face_emb)
    face_kps = draw_kps(face_image, face_info['kps'])

    # with id embedding
    # save directory
    save_dir = "./outputs"
    os.makedirs(save_dir, exist_ok=True)
    output_folder_label = os.path.basename(args.id_image).split('.')[0]  # use the image name as task id
    target_output_folder = os.path.join(save_dir, output_folder_label)
    if not os.path.exists(target_output_folder):
        os.makedirs(target_output_folder)
    
    print("********* Generation Information *********")
    print(f"Prompt: {prompt}")
    print(f"Negative Prompt: {n_prompt}")
    print(f"Reference Input: {args.id_image}")
    print(f"Target Attribute: {target_attribute}")
    print("******************************************")


    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    images = pipe(
        prompt=prompt,
        negative_prompt=n_prompt,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=0.8,
        num_inference_steps=30,
        guidance_scale=5,
        generator = generator
    ).images

    image_personalization_v1 = images[0]
    image_foundation_v1 = images[1]

    guide_mask_base = obtain_mask_from_pillow_form_image(image_foundation_v1, bisenet)
    guide_mask_base_binary = obtain_map_of_specific_parts(guide_mask_base, target_attribute) # hair: 17, mouth: 11 12 13, glasses: 6
    guide_mask_base_rgb = np.stack([guide_mask_base_binary]*3, axis=-1)

    guide_mask_personalized = obtain_mask_from_pillow_form_image(image_personalization_v1, bisenet)
    guide_mask_personalized_binary = obtain_map_of_specific_parts(guide_mask_personalized, target_attribute) # hair
    guide_mask_personalized_rgb = np.stack([guide_mask_personalized_binary]*3, axis=-1)

    # # 第三步：跑二次inference
    attr_mask_foundation_target_mask = resize_mask(guide_mask_base_binary)
    attr_mask_foundation_target_mask = mask_from_numpy_to_torch(attr_mask_foundation_target_mask).half().cuda() # 修正mask的格式，形状和数据类型都要align
    # attr_mask_foundation_target_mask = torch.ones_like(attr_mask_foundation_target_mask)
    freecure_attention = FoundationAwareSelfAttention(
        start_step = 0, # to be move to a cfg object
        end_step = 50, # to be move to a cfg object
        layer_idx = list(range(50, 70, 1)), # to be move to a cfg object
        ref_masks = [attr_mask_foundation_target_mask], # to be move to a cfg object
        mask_weights = [1.5], # to be move to a cfg object
        style_fidelity = 1, # to be move to a cfg object
    )
    register_fasa_to_model(pipe, freecure_attention)

    print("FreeCure attention modules successfully modified")
    generator = torch.Generator(device='cuda').manual_seed(args.seed) 

    images = pipe(
        prompt=args.prompt,
        negative_prompt=n_prompt,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=0.8,
        num_inference_steps=30,
        guidance_scale=5,
        generator = generator
    ).images

    image_personalization_v2 = images[0]
    image_foundation_v2 = images[1]

    image_foundation_v1.save(os.path.join(target_output_folder, f"{args.prompt}_{args.seed}_foundation.png"))
    image_personalization_v1.save(os.path.join(target_output_folder, f"{args.prompt}_{args.seed}_personalization.png"))
    image_personalization_v2.save(os.path.join(target_output_folder, f"{args.prompt}_{args.seed}_refined_with_freecure.png"))

    print("Generation Finished")
