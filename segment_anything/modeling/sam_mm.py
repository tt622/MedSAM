# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from torch.nn.functional import threshold, normalize
import torch
from torch import nn
from torch.nn import functional as F
from mmengine.visualization import Visualizer
import numpy as np

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from mmengine.model import BaseModel
from src.loss import ce_loss,bce_loss
from monai.losses import DiceCELoss
import cv2

class Sam_mm(BaseModel):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        # self.loss=nn.MSELoss()
        self.loss = bce_loss()
        # self.loss=DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.train_epoch=0
        self.val=False

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    # @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool=False,
        mode=None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([x["image"] for x in batched_input], dim=0)
        mask_labels=torch.stack([x["mask_label"] for x in batched_input], dim=0)
        # with torch.no_grad():
        #   image_embeddings = self.image_encoder(input_images)
        
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            # if "point_coords" in image_record:
            # if image_record["point_coords"]!= None:
            #   points = (image_record["point_coords"], image_record["point_labels"])
            # else:
            #   points = None
            points = None
            #with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            #print(curr_embedding.shape,sparse_embeddings.shape,dense_embeddings.shape)
            spatial_dense_embeddings = F.interpolate(dense_embeddings, size=(32, 32), mode='bilinear')
            spatial_dense_pe=self.prompt_encoder.get_dense_pe()
            spatial_dense_pe = F.interpolate(spatial_dense_pe, size=(32, 32), mode='bilinear')
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=spatial_dense_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=spatial_dense_embeddings,
                multimask_output=multimask_output,
            )
            outputs.append(low_res_masks)
        outputs=torch.stack(tuple(outputs), dim=0).squeeze(1)
        # outputs = normalize(threshold(outputs, 0.0, 0))
        # print(mode)
        if mode=='loss':
            self.val=False
            
            # outputs=F.interpolate(outputs, (512,512), mode="bilinear", align_corners=False)
            mask_labels = F.interpolate(
              mask_labels.unsqueeze(1),
              (128, 128),
              mode="nearest"
            )
            loss=self.loss(outputs,mask_labels.float())
            # import pdb;pdb.set_trace()
            
            return {'loss':loss}
        outputs=F.interpolate(outputs, (512,512), mode="bilinear")   #, align_corners=False)
        
       
        outputs=((outputs>0.5)*1).detach()
        if(not self.val and mask_labels[0].max()>0):
          self.train_epoch+=1
          visualizer = Visualizer.get_instance(
              name='vis',
              vis_backends=[dict(type='TensorboardVisBackend')],
              save_dir='temp_dir'
          )
          visualizer.add_image('img',np.array(batched_input[1]['image'].cpu().detach().numpy().transpose(1,2,0)*255,dtype=np.uint8),step=self.train_epoch)
          visualizer.add_image('mask',np.array(mask_labels[1].cpu().detach().numpy()[:,:,np.newaxis].repeat(3,axis=2)*255,dtype=np.uint8),step=self.train_epoch)
          visualizer.add_image('pred',np.array(outputs[1,0].cpu().detach().numpy()[:,:,np.newaxis].repeat(3,axis=2)*255,dtype=np.uint8),step=self.train_epoch)
           
        for i in range(outputs.shape[0]):
          img = np.array(batched_input[i]['image'].cpu().detach().numpy().transpose(1,2,0)*255,dtype=np.uint8)
          mask = np.array(mask_labels[i].cpu().detach().numpy()[:,:,np.newaxis].repeat(3,axis=2)*255,dtype=np.uint8)
          pred = np.array(outputs[i,0].cpu().detach().numpy()[:,:,np.newaxis].repeat(3,axis=2)*255,dtype=np.uint8)
          name = batched_input[i]['name']
          # cv2.imwrite('./result_ft/images/img{}.png'.format(name), img)
          # cv2.imwrite('./result_ft/masks/mask{}.png'.format(name), mask)
          cv2.imwrite('./result_ft/preds/pred{}.png'.format(name), pred)
            
         

          self.val=True
        
        # import pdb
        # pdb.set_trace()
        
        return outputs,mask_labels.unsqueeze(1)

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
