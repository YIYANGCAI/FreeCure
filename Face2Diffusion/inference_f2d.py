import os
import sys
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import numpy as np
import cv2
import argparse
import face_alignment
from PIL import Image
import json
from transformers.models.clip.modeling_clip import CLIPTextTransformer,CLIPTextModel

# src: the original implementation of Face2Diffusion
from src import modules
from src import utils
from src.msid import msid_base_patch8_112
from src import mod
from src.pipeline import Face2DiffusionStableDiffusionPipeline

# freecure package
# step 1: load the freecure source code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('../freecure/cfg.json', 'r') as load_f:
	cfg = json.load(load_f)
cfg = cfg['local_machine']
sys.path.append(cfg['repo']) # the package location of freecure
# step 2: load freecure core algorithm related
from freecure.fasa_module.fasa import (
    FoundationAwareSelfAttention, 
    register_fasa_to_model, 
    trace_attention_module
)
# step 3: load face parsing module integrated in freecure packages, including bisenet and segment anything
from freecure.face_parsing.bisenet.utils import build_bisenet_model
bisenet = build_bisenet_model(cfg['bisenet'], device) # bisenet face parsing model 
from freecure.face_parsing.efficientvit.utils import build_yolo_segment_model
yolo_model, sam = build_yolo_segment_model(cfg['sam'], device) # segment anything
from freecure.face_parsing.utils import resize_mask, mask_from_numpy_to_torch, obtain_mask_from_pillow_form_image, obtain_map_of_specific_parts, vis_parsing_maps, merge_masks

def main(args):
# an example of bisetnet face parsing model's label correspondence
	attribute_lut = {
		"hair": [17],
		"eye": [4, 5],
		"glasses": [6],
	}
	output_dir = args.output
	os.makedirs(output_dir, exist_ok=True)
	
	pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"
	pipe = Face2DiffusionStableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16,safety_checker=None).to("cuda")
	pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

	#build f2d pipeline
	pipe.text_encoder.text_model.forward = mod.forward_texttransformer.__get__(pipe.text_encoder.text_model, CLIPTextTransformer)
	pipe.text_encoder.forward = mod.forward_textmodel.__get__(pipe.text_encoder, CLIPTextModel)

	img2text = modules.IMG2TEXTwithEXP(384*4,384*4,768)
	img2text.load_state_dict(torch.load(args.w_map,map_location='cpu'))
	img2text=img2text.to(device)
	img2text.eval()

	msid = msid_base_patch8_112(ext_depthes=[2,5,8,11])
	msid.load_state_dict(torch.load(args.w_msid))
	msid=msid.to(device)
	msid.eval()

	# 生成prompt embedding: with / without identity embedding
	prompt = args.prompt
	target_attribute = []
	for key, value in attribute_lut.items():
		if key in prompt:
			target_attribute.extend(value)
	prompt_foundation = prompt.replace('f l', 'a person')
	prompt_foundation = prompt_foundation.replace("person", args.gender)
	identifier='f'

	ids = pipe.tokenizer(
					prompt,
					padding="do_not_pad",
					truncation=True,
					max_length=pipe.tokenizer.model_max_length,
				).input_ids
	placeholder_token_id=pipe.tokenizer(
					identifier,
					padding="do_not_pad",
					truncation=True,
					max_length=pipe.tokenizer.model_max_length,
				).input_ids[1]
	assert placeholder_token_id in ids,'identifier does not exist in prompt'
	pos_id = ids.index(placeholder_token_id)

	input_ids = pipe.tokenizer.pad(
			{"input_ids": [ids]},
			padding="max_length",
			max_length=pipe.tokenizer.model_max_length,
			return_tensors="pt",
		).input_ids

	# 没有identifier (identity embedding) 的 prompt embedding,体现基模能力的embedding
	ids_foundation = pipe.tokenizer(
					prompt_foundation,
					padding="do_not_pad",
					truncation=True,
					max_length=pipe.tokenizer.model_max_length,
				).input_ids

	input_ids_foundation = pipe.tokenizer.pad(
			{"input_ids": [ids_foundation]},
			padding="max_length",
			max_length=pipe.tokenizer.model_max_length,
			return_tensors="pt",
		).input_ids	

	#identity encoding
	detector=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,flip_input=False,device='cuda' if torch.cuda.is_available() else 'cpu')
	lmk=np.array(detector.get_landmarks(args.input))[0]
	img = np.array(Image.open(args.input).convert('RGB'))
	with torch.no_grad():
		M=utils.align(lmk)
		img=utils.warp_img(img,M,(112,112))/255
		img=torch.tensor(img).permute(2,0,1).unsqueeze(0)
		img=(img-0.5)/0.5
		idvec = msid.extract_mlfeat(img.to(device).float(),[2,5,8,11])
		tokenized_identity_first, tokenized_identity_last = img2text(idvec,exp=None)
		hidden_states = utils.get_clip_hidden_states(input_ids.to(device),pipe.text_encoder).to(dtype=torch.float32)
		hidden_states[[0], [pos_id]]=tokenized_identity_first.to(dtype=torch.float32)
		hidden_states[[0], [pos_id+1]]=tokenized_identity_last.to(dtype=torch.float32)
		pos_eot = input_ids.to(dtype=torch.int, device=hidden_states.device).argmax(dim=-1)
		# 基模对应的prompt embedding
		hidden_states_foundation = utils.get_clip_hidden_states(input_ids_foundation.to(device),pipe.text_encoder).to(dtype=torch.float32)
	
	#text encoding
	with torch.autocast("cuda"):
		with torch.no_grad():
			encoder_hidden_states = pipe.text_encoder(hidden_states=hidden_states, pos_eot=pos_eot)[0]
			encoder_hidden_states_foundation = pipe.text_encoder(hidden_states=hidden_states_foundation, pos_eot=pos_eot)[0]
			encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_foundation], dim = 0)

	print("********* Generation Information *********")
	print(f"Prompt: {prompt}")
	print(f"Reference Input: {args.input}")
	print(f"Target Attribute: {target_attribute}")
	print("******************************************")

	seed = args.seed
	generator = torch.Generator(device).manual_seed(seed) # seed 确定
	# Step 1: dual inference 获得foundation output以及 personalized output
	image = pipe(
		prompt_embeds=encoder_hidden_states, 
		num_inference_steps=30, 
		guidance_scale=7,
		generator=generator,
		num_images_per_prompt=args.n_samples
	).images
	image_personalization_v1 = image[0]
	image_foundation_v1 = image[1]

	guide_mask_base = obtain_mask_from_pillow_form_image(image_foundation_v1, bisenet)
	guide_mask_base_binary = obtain_map_of_specific_parts(guide_mask_base, target_attribute) # hair: 17, mouth: 11 12 13, glasses: 6

	attr_mask_foundation_target_mask = resize_mask(guide_mask_base_binary)
	attr_mask_foundation_target_mask = mask_from_numpy_to_torch(attr_mask_foundation_target_mask).half().cuda() # 修正mask的格式，形状和数据类型都要align

	freecure_attention = FoundationAwareSelfAttention(
		start_step = 0,
		end_step = 50,
		layer_idx = [10, 11, 12, 13, 14, 15],
		ref_masks = [attr_mask_foundation_target_mask],
		mask_weights = [0.5],
		style_fidelity = 1,
	)
	register_fasa_to_model(pipe, freecure_attention)
	print("FreeCure attention modules successfully modified")

	generator = torch.Generator(device).manual_seed(seed) 
	image = pipe(
		prompt_embeds=encoder_hidden_states, 
		num_inference_steps=30, 
		guidance_scale=7,
		generator=generator,
		num_images_per_prompt=args.n_samples
	).images
	image_personalization_v2 = image[0]

	image_foundation_v1.save(os.path.join(output_dir, f"{seed}_foundation_out_v1.png"))
	image_personalization_v1.save(os.path.join(output_dir, f"{seed}_personalization.png"))
	image_personalization_v2.save(os.path.join(output_dir, f"{seed}_refined_with_freecure.png"))

if __name__=='__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('-p',dest='prompt',required=True)
	parser.add_argument('-g',dest='gender',required=True)
	parser.add_argument('-i',dest='input',required=True,help='path for the input facial image')
	parser.add_argument('--w_map',required=True,help='weight path for the mapping network')
	parser.add_argument('--w_msid',required=True,help='weight path for the msid encoder')
	parser.add_argument('-o',dest='output',required=True)
	parser.add_argument('-n',dest='n_samples',default=1,type=int)
	parser.add_argument('-s',dest='seed',default=0,type=int)
	args=parser.parse_args()
	main(args)