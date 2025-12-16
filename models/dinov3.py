# This code has been adapted from Meta’s DINOv3 (https://github.com/facebookresearch/dinov3)
import logging
import os
import torch
import torch.nn as nn
import torchvision.transforms as TF
import numpy as np
from safetensors.torch import load_file
from PIL import Image
from .encoder import Encoder
import pathlib
from typing import Optional
import logging


logger = logging.getLogger(__name__)

class DINOv3Encoder(Encoder):
    def setup(self, dino_size="l", repo_dir="./dinov3", ckpt: Optional[str] = None):
        """
        dino_size: 'b' (ViT-B/16) or 's' (ViT-S/16)
        repo_dir: local path to DINOv3 repo containing hubconf.py
        dino_ckpt: path to weights file (.pth) or None
        """
        self.dino_size = dino_size
        self.repo_dir = repo_dir
        self.dino_ckpt = ckpt
        self.arch_str = f"dinov3_vit{self.dino_size}16"

        self.model = torch.hub.load(
            self.repo_dir, self.arch_str, source="local", pretrained=False
        )

        if self.dino_ckpt is not None and os.path.exists(self.dino_ckpt):
            self.arch_str = pathlib.Path(self.dino_ckpt).stem
            if self.dino_ckpt.endswith(".safetensors"):
                state_dict = load_file(self.dino_ckpt)
            else:
                state_dict = torch.load(self.dino_ckpt, map_location="cpu")

            self.model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning(f"Initialized {self.arch_str} model with random weights. Checkpoint is either None or doesn't exist: {self.dino_ckpt=}")

        self.model.eval()


    def transform(self, img):
        """Transformacja obrazu do formatu wejściowego DINOv3"""
        # normalizacja taka jak dla ImageNet
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])

        cifar_mean = [0.4914, 0.4822, 0.4465]
        cifar_std = [0.2023, 0.1994, 0.2010]
        img = TF.Compose(
            [
                TF.Resize((224, 224), TF.InterpolationMode.BICUBIC),
                TF.ToTensor(),
                TF.Normalize(cifar_mean, cifar_std),
            ]
        )(img)
        return img
