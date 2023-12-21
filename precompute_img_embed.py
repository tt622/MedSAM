#%% import packages
# precompute image embeddings and save them to disk for model training

import numpy as np
import os
join = os.path.join 
from skimage import io, segmentation, transform
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
import cv2

#%% parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img_path', type=str, default='/Road/deepGlobe/train_split/img', help='# and also Tr_Release_Part2 when part1 is done')
parser.add_argument('-a', '--mask_path', type=str, default='/Road/deepGlobe/train_split/mask', help='# and also Tr_Release_Part2 when part1 is done')
parser.add_argument('-o', '--save_path', type=str, default='data/Tr_npy_new', help='path to save the image embeddings')
parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
parser.add_argument('--checkpoint', type=str, default='work_dir/SAM/sam_vit_b_01ec64.pth', help='path to the pre-trained SAM model')
args = parser.parse_args()

pre_img_path = args.img_path 
pre_mask_path = args.mask_path
save_img_emb_path = join(args.save_path, 'npy_embs')
save_gt_path = join(args.save_path, 'npy_gts')
os.makedirs(save_img_emb_path, exist_ok=True)
os.makedirs(save_gt_path, exist_ok=True)
npz_files = sorted(os.listdir(pre_img_path))
#%% set up the model
sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to('cuda:0')
sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)

# compute image embeddings
for name in tqdm(npz_files):
    # img = np.load(join(pre_img_path, name)) # (256, 256, 3)
    # gt = np.load(join(pre_mask_path, name))
    img = cv2.imread(join(pre_img_path, name)) # (256, 256, 3)
    # print(img.shape)
    gt = cv2.imread(join(pre_mask_path, name))[:,:,0]
    resize_img = sam_transform.apply_image(img)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to('cuda:0')
    # model input: (1, 3, 1024, 1024)
    input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
    with torch.no_grad():
        embedding = sam_model.image_encoder(input_image)
    
    # 将gt resize到256*256
    gt = transform.resize(
        gt == 255,
        (256, 256),
        order=0,
        preserve_range=True,
        mode="constant",
    )

    # save as npy
    np.save(join(save_img_emb_path, name.split('.png')[0]+'.npy'), embedding.cpu().numpy()[0])
    np.save(join(save_gt_path, name.split('.png')[0]+'.npy'), gt)
    # # sanity check
    # img_idx = img.copy()
    # bd = segmentation.find_boundaries(gt, mode='inner')
    # img_idx[bd, :] = [255, 0, 0]
    # io.imsave(save_img_emb_path + '.png', img_idx)
