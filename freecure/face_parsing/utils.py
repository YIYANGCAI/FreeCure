from typing import Tuple, List
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
# try:
#     from inference.models import YOLOWorld
#     from face_parsing.efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
#     from face_parsing.efficientvit.sam_model_zoo import create_sam_model
#     import supervision as sv
# except:
#     print("YoloWorld can not be loaded")

# try:
#     from face_parsing.bisenet.bisenet_model import BiSeNet
# except:
#     print("BiSENet can not be loaded")

def obtain_mask_from_pillow_form_image(img, seg_model):
    # 定义 model 输入的 transformation, 
    # input: img: Image frame
    # outputs: the numpy data of int value (refer to different labels)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = seg_model(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0) # numpy, semantic map

    return parsing

def obtain_map_of_specific_parts(parsing_map, target_labels = []):
    # parsing_map: [xxx, yyy] (semantic segmentation maps, numpy)
    # input: numpy
    # output: numpy
    parsing_map = torch.from_numpy(parsing_map)
    parsing_map_filtered = torch.zeros_like(parsing_map)
    for target_label in target_labels:
        parsing_map_current = torch.where(parsing_map == target_label, 1, 0)
        parsing_map_filtered = parsing_map_filtered + parsing_map_current
    parsing_map_filtered = torch.where(parsing_map_filtered != 0, 1, 0) # normalized
    parsing_map_filtered = (parsing_map_filtered.numpy()*255).astype(np.uint8) # numpy 255

    return parsing_map_filtered

def vis_parsing_maps(im, parsing_anno, stride=1):
    # Colors for all 20 parts， 可视化face parsing的部分
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 127

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    # vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    return vis_parsing_anno_color

def merge_masks(masks):
    final_mask = np.zeros_like(masks[0])
    for mask in masks:
        final_mask += mask
    final_mask = (np.where(final_mask != 0, 1, 0)*255).astype(np.uint8)
    return final_mask

def resize_mask(mask, shape = (128, 128)):
	# (512, 512) (0, 255) => (128, 128, 3) (0, 255) numpy
	mask = np.stack([mask]*3, axis=-1)
	mask = cv2.resize(mask, shape)
	mask = np.where(mask == 255, 255, 0)
	return mask

def mask_from_torch_to_numpy(mask):
    # mask: torch [1, 1, 128, 128](0, 1 float32) => numpy [128, 128, 3](0, 256 uint8), 能够被存储成png的形式
    mask = mask[0][0]
    mask = torch.stack([mask] * 3, dim = -1).cpu().numpy()
    mask = mask * 255
    return mask

def mask_from_numpy_to_torch(mask):
    # mask: numpy [128, 128, 3](0, 255 uint8) => torch [1, 1, 128, 128](0, 1 uint8), 能够作为freecustom的结果
    mask = mask.astype(np.float32)[:, :, 0] / 255.0
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    return mask