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
        self.img_path = args.img_path
        self.scribble_path = args.scribble_path
        self.gt_path = args.gt_path
        self.file_name = sorted(os.listdir(self.gt_path))
    
    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        img = cv2.imread(join(self.img_path, self.file_name[index])) # (1024, 1024, 3)
        scribble = cv2.imread(join(self.scribble_path, self.file_name[index]))[:,:,0]
        gt = cv2.imread(join(self.gt_path, self.file_name[index]))[:,:,0]
        resize_img = sam_transform.apply_image(img)
        img = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(img).float(), torch.tensor(scribble[None, :,:]).float(), torch.tensor(gt[None, :,:]).long()

# %% set up parser
parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--tr_npy_path', type=str, default='data/Tr_npy_new', help='path to training npy files; two subfolders: npy_gts and npy_embs')
parser.add_argument('--img_path', type=str, default="/Road/datasets/partial-osm/imagery", help='path to training images')
parser.add_argument('--scribble_path', type=str, default="/Road/datasets/partial-osm/masks_75", help='path to training scribble prompts')
parser.add_argument('--gt_path', type=str, default="/Road/datasets/partial-osm/masks", help='path to training gts')

parser.add_argument('--task_name', type=str, default='SAM-B-partialOSM-scribble-BCELoss')
parser.add_argument('--model_type', type=str, default='vit_b')
parser.add_argument('--checkpoint', type=str, default='work_dir/SAM/sam_vit_b_01ec64.pth')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--work_dir', type=str, default='work_dir_test')
# train
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=4)
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
optimizer = torch.optim.Adam(sam_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
train_dataset = NpyDataset(args.img_path)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
for epoch in range(num_epochs):
    epoch_loss = 0
    # Just train on the first 20 examples
    for step, (img, scribble, gt) in enumerate(tqdm(train_dataloader)):
        # do not compute gradients for image encoder and prompt encoder
        with torch.no_grad():
            # model input: (B, 3, 1024, 1024)
            image_embedding = sam_model.image_encoder(img)
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=scribble.to(device),
            )
            print("image_embedding:" + str(image_embedding.shape))
        print("sparse_embeddings:" + str(sparse_embeddings.shape))
        print("dense_embeddings:" + str(dense_embeddings.shape))
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )
        masks = sam_model.postprocess_masks(
            low_res_masks,
            input_size=img.shape[-2:],
            original_size=img.shape[-2:],
        )
        # print(low_res_masks.shape)# (B, 1, 256, 256)
        # loss = seg_loss(low_res_masks, gt2D.to(device))
        # loss = mse_Loss(low_res_masks, gt2D.to(device).float())
        loss = bce_Loss(m(masks), m(gt).to(device).float())
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

