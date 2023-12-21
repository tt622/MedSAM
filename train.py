# %% set up environment
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
from torch import nn
import cv2
from skimage import io, segmentation, transform
# set seeds
torch.manual_seed(2023)
np.random.seed(2023)


#%% create a dataset class to load npz data and return back image embeddings and ground truth
class NpyDataset(Dataset): 
    def __init__(self, data_root):
        self.data_root = data_root
        self.gt_path = join(data_root, 'mask')
        self.embed_path = join(data_root, 'img')
        self.npy_files = sorted(os.listdir(self.gt_path))
    
    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, index):
        # gt2D = np.load(join(self.gt_path, self.npy_files[index]))
        # img_embed = np.load(join(self.embed_path, self.npy_files[index]))
        img = cv2.imread(join(self.embed_path, self.npy_files[index])) # (256, 256, 3)
        gt2D = cv2.imread(join(self.gt_path, self.npy_files[index]))[:,:,0]
        # 将gt resize到256*256
        gt2D = transform.resize(gt2D == 255,(256, 256),order=0,preserve_range=True,mode="constant",)
        resize_img = sam_transform.apply_image(img)
        img = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(img).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(bboxes).float()

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--data_root', type=str, default='/Road/deepGlobe/train_split', help='path to training npy files; two subfolders: img and mask')
parser.add_argument('--task_name', type=str, default='SAM-B-deepglobe-BCELoss-noFreeze')
parser.add_argument('--model_type', type=str, default='vit_b')
parser.add_argument('--checkpoint', type=str, default='work_dir/SAM/sam_vit_b_01ec64.pth')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--work_dir', type=str, default='work_dir_test')
# train
parser.add_argument('--num_epochs', type=int, default=24)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0)
args = parser.parse_args()


# %% set up model for fine-tuning 
device = args.device
model_save_path = join(args.work_dir, args.task_name)
os.makedirs(model_save_path, exist_ok=True)
sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
sam_model.train()

mylog = open(model_save_path+"/train.log",'w')

# Set up the optimizer, hyperparameter tuning will improve performance here
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
# mse_Loss=nn.MSELoss()
bce_Loss = nn.BCELoss()
m = nn.Sigmoid()
# regress loss for IoU/DSC prediction; (ignored for simplicity but will definitely included in the near future)
# regress_loss = torch.nn.MSELoss(reduction='mean')
#%% train
num_epochs = args.num_epochs
losses = []
best_loss = 1e10
train_dataset = NpyDataset(args.data_root)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
for epoch in range(num_epochs):
    epoch_loss = 0
    # Just train on the first 20 examples
    for step, (img, gt2D, boxes) in enumerate(tqdm(train_dataloader)):
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            # convert box to 1024x1024 grid
            box_np = boxes.numpy()
            sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
            box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                # boxes=box_torch,
                boxes=None,
                masks=None,
            )
        # do not freeze image encoder
        # model input: (1, 3, 1024, 1024)
        # print(img.shape)
        image_embedding = sam_model.image_encoder(img)
        # print("image_embedding:" + str(image_embedding.shape))
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          )
        # print(low_res_masks.shape)# (B, 1, 256, 256)
        # loss = seg_loss(low_res_masks, gt2D.to(device))
        # loss = mse_Loss(low_res_masks, gt2D.to(device).float())
        loss = bce_Loss(m(low_res_masks), m(gt2D).to(device).float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= step
    losses.append(epoch_loss)
    print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
    print(f'EPOCH: {epoch}, Loss: {epoch_loss}', file=mylog)
    # save the model checkpoint
    torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_latest.pth'))
    # save the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_best.pth'))

    # %% plot loss
    plt.plot(losses)
    plt.title('Dice + Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.show() # comment this line if you are running on a server
    plt.savefig(join(model_save_path, 'train_loss.png'))
    plt.close()
print("Finish!", file=mylog)
mylog.close()

