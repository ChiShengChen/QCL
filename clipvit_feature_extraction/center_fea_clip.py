"""
Package CLIP features for center images

"""

import argparse
import torch.nn as nn
import numpy as np
import torch
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print(torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--project_dir', default='/home/aidan/QuantumMUSE_EEG/Data/Things-EEG2/Image_set', type=str)
args = parser.parse_args()

print('Extract feature maps CLIP of images for center <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.use_deterministic_algorithms(True)

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
model = model.cuda()
model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])

# with open('./THINGS-EEG/MUSE_EEG/model/CLIP_img_ViT_info.txt', 'w') as file:
#     print(model, file=file)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

img_set_dir = os.path.join(args.project_dir, 'image_set/center_images/')
condition_list = os.listdir(img_set_dir)
condition_list.sort()

all_centers = []

for cond in condition_list:
    one_cond_dir = os.path.join(args.project_dir, 'image_set/center_images/', cond)
    cond_img_list = os.listdir(one_cond_dir)
    cond_img_list.sort()
    cond_center = []
    for img in cond_img_list:
        img_path = os.path.join(one_cond_dir, img)
        img = Image.open(img_path).convert('RGB')
        inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=img, return_tensors="pt", padding=True)
        inputs.data['pixel_values'].cuda()
        with torch.no_grad():
            outputs = model(**inputs).image_embeds
    # * for mean center
    #     cond_center.append(outputs.detach().cpu().numpy())
    # cond_center = np.mean(cond_center, axis=0)
    # all_centers.append(np.squeeze(cond_center))
    
        cond_center.append(np.squeeze(outputs.detach().cpu().numpy()))
    all_centers.append(np.array(cond_center))


all_centers = np.array(all_centers)
print(all_centers.shape)
np.save(os.path.join(args.project_dir, 'center_all_image_clip.npy'), all_centers)

# * for mean center
# np.save(os.path.join(args.project_dir, 'center__clip.npy'), all_centers)

