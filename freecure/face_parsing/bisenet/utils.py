import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import freecure
try:
    from freecure.face_parsing.bisenet.bisenet_model import BiSeNet
except:
    print("BiSENet can not be loaded")

def build_bisenet_model(bisenet_path, device):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    # save_pth = os.path.join('res/cp', cp)
    net.load_state_dict(torch.load(bisenet_path))
    net.eval()
    return net

# bisenet inference
def bisenet_run_prediction(img, seg_model):
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

def bisenet_extract_attr(parsing_map, target_labels = []):
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