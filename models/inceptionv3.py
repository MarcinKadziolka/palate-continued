import torchvision.transforms as TF
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import inception_v3, Inception_V3_Weights
from PIL import Image
from torchvision.transforms.functional import to_tensor

from .encoder import Encoder


def pil_resize(x, output_size):
    """
    Manually implements BICUBIC resizing using PIL to ensure
    reproducibility across different versions of TorchVision.
    """
    s1, s2 = output_size

    def resize_single_channel(x):
        img = Image.fromarray(x, mode="F")
        img = img.resize(output_size, resample=Image.BICUBIC)
        return np.asarray(img).clip(0, 255).reshape(s2, s1, 1)

    x = np.array(x.convert("RGB")).astype(np.float32)
    x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
    x = np.concatenate(x, axis=2).astype(np.float32)
    return to_tensor(x) / 255


class InceptionV3Encoder(Encoder):
    def setup(self, clean_resize: bool = False):
        # Load the standard ImageNet-pretrained weights
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights)

        # This line is crucial for your main.py logging and file naming
        self.arch_str = "inception_v3"

        # We need the 2048-dim embedding (the "pool3" layer) for metrics like FID.
        # Replacing the final fully connected layer with Identity returns the
        # flattened features instead of 1000 class probabilities.
        model.fc = nn.Identity()

        # Inception has Batchnorm/Dropout; eval() ensures deterministic behavior.
        model.eval()

        self.model = model
        self.clean_resize = clean_resize

    def transform(self, img) -> Tensor:
        # CRITICAL: InceptionV3 was trained on 299x299, unlike 224x224 for DINO.
        # Using the wrong size will lead to incorrect FID/IS results.
        size = (299, 299)

        # Standard ImageNet normalization constants used by TorchVision models.
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        # Choose between high-fidelity PIL resizing or standard TorchVision resizing.
        if self.clean_resize:
            img = pil_resize(img, size)
        else:
            img = TF.Compose(
                [
                    TF.Resize(size, TF.InterpolationMode.BICUBIC),
                    TF.ToTensor(),
                ]
            )(img)

        return TF.Normalize(imagenet_mean, imagenet_std)(img)