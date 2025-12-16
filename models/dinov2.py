# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

# This code has been adapted from the sources above


import torchvision.transforms as TF
import numpy as np
import torch

import sys

from torch import Tensor

from .encoder import Encoder

from PIL import Image
from torchvision.transforms.functional import to_tensor


def pil_resize(x, output_size):
    s1, s2 = output_size

    def resize_single_channel(x):
        img = Image.fromarray(x, mode="F")
        img = img.resize(output_size, resample=Image.BICUBIC)
        return np.asarray(img).clip(0, 255).reshape(s2, s1, 1)

    x = np.array(x.convert("RGB")).astype(np.float32)
    x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
    x = np.concatenate(x, axis=2).astype(np.float32)
    return to_tensor(x) / 255


VALID_ARCHITECTURES = [
    "vits14",
    "vitb14",
    "vitl14",
    "vitg14",
]


class DINOv2Encoder(Encoder):
    def setup(self, arch=None, clean_resize: bool = False):
        if arch is None:
            arch = "vitl14"

        self.arch = arch

        self.arch_str = f"dinov2_{self.arch}"

        if self.arch not in VALID_ARCHITECTURES:
            sys.exit(
                f"arch={self.arch} is not a valid architecture. Choose from {VALID_ARCHITECTURES}"
            )

        self.model = torch.hub.load("facebookresearch/dinov2", self.arch_str)
        self.clean_resize = clean_resize

    def transform(self, img) -> Tensor:

        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])

        if self.clean_resize:
            img = pil_resize(img, (224, 224))
        else:
            img = TF.Compose(
                [
                    TF.Resize((224, 224), TF.InterpolationMode.BICUBIC),
                    TF.ToTensor(),
                ]
            )(img)

        return TF.Normalize(imagenet_mean, imagenet_std)(img)
