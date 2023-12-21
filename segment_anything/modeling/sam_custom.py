# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmengine.visualization import Visualizer
import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from mmengine.model import BaseModel
import numpy as np
from mmengine import MessageHub
class Sam(BaseModel):
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
        self.loss=nn.MSELoss()
        self.train_epoch=0
        self.val=False
        
    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    # @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output=False,
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
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        # import pdb;pdb.set_trace()
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        mask_labels=torch.stack([x["mask_label"] for x in batched_input], dim=0)
        # input_images=batched_input['images']
        # mask_labels=batched_input["mask_label"]
        B,C,H,W=input_images.shape
        # print("aaaa",input_images.shape)
        image_embeddings = self.image_encoder(input_images)
        
        message_hub = MessageHub.get_current_instance()
        epoch = message_hub.get_info('epoch')
        # print(epoch)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            pass
            if epoch>10 and "point_coords" in image_record:
            # if "point_coords" in image_record:
                # print('hhhhhh')
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            
            # prompt encoder freeze 
            # with torch.no_grad():
            #   sparse_embeddings, dense_embeddings = self.prompt_encoder(
            #       points=points,
            #       boxes=image_record.get("boxes", None),
            #       masks=image_record.get("mask_inputs", None),
            #   )
            # prompt encoder finutune
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            
            # masks = self.postprocess_masks(
            #     low_res_masks,
            #     input_size=image_record["image"].shape[-2:],
            #     original_size=image_record["original_size"],
            # )
            # masks = masks > self.mask_threshold
            # outputs.append(
            #     {
            #         "masks": masks,
            #         "iou_predictions": iou_predictions,
            #         "low_res_logits": low_res_masks,
            #     }
            # )
            # return outputs
            outputs.append(low_res_masks)
        outputs=torch.stack(tuple(outputs), dim=0).squeeze(1)
        if mode=='loss':
            self.val=False
             
            # import pdb;pdb.set_trace()
            loss2=self.loss(outputs,mask_labels.float())
            return {'loss':loss2}
            # return loss2*0.5+loss1*0.5
        
        outputs = F.interpolate(
              outputs,
              (1024, 1024),
              mode="bilinear",
              align_corners=False,
        )
        
        outputs=((outputs>0.5)*1).detach()
        if(not self.val and mask_labels[0].max()>0):
          self.train_epoch+=1
          visualizer = Visualizer.get_instance(
              name='vis',
              vis_backends=[dict(type='TensorboardVisBackend')],
              save_dir='temp_dir'
          )
          visualizer.add_image('img',np.array(batched_input[0]['image'].cpu().detach().numpy().transpose(1,2,0)*255,dtype=np.uint8),step=self.train_epoch)
          visualizer.add_image('mask',np.array(mask_labels[0].cpu().detach().numpy()[:,:,np.newaxis].repeat(3,axis=2)*255,dtype=np.uint8),step=self.train_epoch)
          visualizer.add_image('pred',np.array(outputs[0,0].cpu().detach().numpy()[:,:,np.newaxis].repeat(3,axis=2)*255,dtype=np.uint8),step=self.train_epoch)

          self.val=True
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
        # padh = 512 - h
        # padw = 512 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
