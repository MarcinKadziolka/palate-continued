import subprocess
import sys
import torchvision.transforms as TF
import numpy as np
import torch
from torch import Tensor
from PIL import Image
from .encoder import Encoder


class CLIPEncoder(Encoder):
    def setup(self, arch: str = "ViT-B/32", clean_resize: bool = False):
        # --- Automatic Installation Check ---
        try:
            import clip
        except ImportError:
            print("CLIP library not found. Installing OpenAI CLIP... please wait.")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"])
            import clip
        # ------------------------------------

        self.arch = arch
        self.arch_str = f"clip_{arch.replace('/', '_')}"

        # Load model and its specific preprocess transform
        # We load to CPU first, main.py moves it to the device later
        model, preprocess = clip.load(self.arch, device="cpu")
        self.model = model.visual
        self.clip_preprocess = preprocess
        self.clean_resize = clean_resize

    def transform(self, img) -> Tensor:
        if self.clean_resize:
            # CLIP usually uses 224x224, except for the large 336px model
            size = 224 if "336px" not in self.arch else 336

            img = img.convert("RGB")
            img = img.resize((size, size), resample=Image.BICUBIC)
            img = TF.ToTensor()(img)

            # These are specific CLIP normalization values (different from ImageNet!)
            img = TF.Normalize((0.48145466, 0.4578275, 0.40821073),
                               (0.26862954, 0.26130258, 0.27577711))(img)
            return img

        return self.clip_preprocess(img)