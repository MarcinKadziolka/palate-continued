# This code has been adapted from: https://github.com/layer6ai-labs/dgm-eval/blob/master/dgm_eval/dataloaders.py

import os
import sys
import pathlib
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import logging
import torchvision
import torchvision.transforms
import io
import torch
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}

TORCHVISION_DATA_PATH = "./data/"


def apply_jpeg_compression(img, quality=75):
    # Convert the tensor to a PIL image
    img = torchvision.transforms.ToPILImage()(img)

    # Create a buffer to store the image data
    buffer = io.BytesIO()

    # Save the image as JPEG with the specified quality
    img.save(buffer, format="JPEG", quality=quality)

    # Load the image back from the buffer
    buffer.seek(0)
    img = Image.open(buffer)

    # Convert the PIL image back to a tensor
    img = torchvision.transforms.ToTensor()(img)

    return img

def get_files_at_path(path):
    """Return list of all files at path of type IMAGE_EXTENSIONS"""

    files = sorted([file for ext in IMAGE_EXTENSIONS for file in path.glob(f"*.{ext}")])

    return files


class ImagePathDataset(Dataset):
    """
    Create a custom dataset from a list of image files on disk

    Files must have image extensions specified in IMAGE_EXTENSIONS
    """

    def __init__(self, files, transform=None, distortion="none"):
        self.files = sorted(files)
        self.transform = transform
        self.distortion = distortion

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.distortion == "none":
            pass
        elif self.distortion == "posterize":
            posterize = torchvision.transforms.Lambda(
                lambda img: ((img * 255).to(torch.uint8) & 0b11110000).float() / 255.0)
            img = posterize(img)
        elif self.distortion == "blur":
            light_blur = torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(0.5))  # Light Gaussian blur
            img = light_blur(img)
        elif self.distortion == "heavy_blur":
            heavy_blur = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=1.4)  # Heavy Gaussian blur
            img = heavy_blur(img)
        elif self.distortion == "resize":
            resize = torchvision.transforms.Resize((196, 196))  # Resize to 224x224
            img = resize(img)
        elif self.distortion == "center_crop30":
            center_crop_30 = torchvision.transforms.CenterCrop(30)  # Center crop to 30x30
            img = center_crop_30(img)
        elif self.distortion == "center_crop28":
            center_crop_28 = torchvision.transforms.CenterCrop(28)  # Center crop to 30x30
            img = center_crop_28(img)
        elif self.distortion == "color_distort":
            color_distort = torchvision.transforms.ColorJitter()  # Random color distortion
            img = color_distort(img)
        elif self.distortion == "elastic_transform":
            elastic_transform = torchvision.transforms.ElasticTransform()  # Elastic transformation
            img = elastic_transform(img)
        elif self.distortion == "jpg75":
            img = apply_jpeg_compression(img, quality=75)
        elif self.distortion == "jpg90":
            img = apply_jpeg_compression(img, quality=90)

        return img


class CustomDataLoader:
    """
    Create Datasets and Dataloaders from ImagePathDataset and from torchvision.datasets.
    """

    def __init__(
        self,
        path: str,
        nsample: int = -1,
        transform=None,
        batch_size: int = 50,
        num_workers: int = 0,
        seed: int = 13579,
        random_sample: bool = True,
        sample_w_replacement: bool = False,
        distortion="none"
    ):
        logger.info(f"Initializing dataloader for path: {path}")
        self.path = path
        self.nsample = nsample
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        # for class conditional models, remember the labels as loading
        self.labels = []

        self.random_sample = random_sample
        self.sample_w_replacement = sample_w_replacement

        if sample_w_replacement:
            print(
                (
                    f"Warning: sample_w_replacement={sample_w_replacement}."
                    f"Sampling with replacement from path {path}"
                ),
                file=sys.stderr,
            )
            self.seed += 1

        self.distortion = distortion
        self.transform = transform
        if not transform:
            self.transform = torchvision.transforms.ToTensor()

        self.get_dataset()

        if (self.nsample > 0) and (len(self.data_set) > self.nsample):
            self.subsample_dataset()

        self.get_dataloader()

    def get_dataset(self):
        """
        Get dataset from local path or from torchvision.datasets
        """
        if os.path.exists(self.path):
            self.get_local_dataset()
        else:
            raise Exception(f"Path {self.path} does not exist.")

    def get_local_dataset(self):
        """
        Get dataset from disk

        Currently accepted formats:

        1.) Path to folder containing individual images of extension types in IMAGE_EXTENSIONS

        2.) Path to folder containing sub-folders for each image class,
            where each sub-folder contains individual images of extension types in IMAGE_EXTENSIONS
        """

        self.dataset_name = os.path.basename(os.path.normpath(self.path))

        image_path = pathlib.Path(self.path)

        self.files = get_files_at_path(image_path)
        class_idx = 0

        def get_order(file):
            filename = os.path.splitext(os.path.basename(file))[0]
            return int(filename)

        if not self.files:
            # Assume sub-folders for image classes
            class_dirs = sorted(
                image_path.glob("[0-9]*"), key=get_order
            )  # look for all subfolders in the numerical order
            logger.info(f"Found {len(class_dirs)} classes in {image_path}")
            self.files = []
            for f in class_dirs:
                files_in_path = get_files_at_path(f)
                self.files += files_in_path
                self.labels.extend([class_idx for _ in range(len(files_in_path))])
                class_idx += 1
        self.labels = np.array(self.labels, dtype=np.int32)
        # print(f'len labels {len(self.labels)}')

        # Confirm data at path is in proper format
        try:
            logger.info("Applying distortion: %s", self.distortion)
            self.data_set = ImagePathDataset(self.files, transform=self.transform, distortion=self.distortion)
            logger.info("Applied distortion: %s", self.distortion)
        except:
            raise RuntimeError(
                f"Images cannot be loaded from {self.path}. Expecting path full of images: {IMAGE_EXTENSIONS}"
            )

    def subsample_dataset(self):
        """subsample to desired size"""

        np.random.seed(self.seed)  # for consistent subsampling of datasets across runs

        if self.random_sample:
            self.inds_keep = sorted(
                np.random.choice(
                    len(self.data_set), self.nsample, replace=self.sample_w_replacement
                )
            )
        else:
            self.inds_keep = np.arange(self.nsample)

        if self.files:
            self.files = [self.files[i] for i in self.inds_keep]

        if self.labels is not None and len(self.labels) > 0:
            self.labels = self.labels[self.inds_keep]
        self.data_set = Subset(
            self.data_set,
            self.inds_keep,
        )

    def get_dataloader(self):
        """
        Create dataloader from dataset
        """
        self.nimages = len(self.data_set)
        if self.batch_size > self.nimages:
            print(
                (
                    "Warning: batch size is bigger than the data size. "
                    "Setting batch size to data size"
                )
            )
            self.batch_size = self.nimages

        self.data_loader = DataLoader(
            self.data_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )


def get_dataloader(
    path: str,
    nsample: int = -1,
    batch_size: int = 32,
    num_workers: int = 0,
    transform=None,
    seed: int = 13579,
    random_sample: bool = True,
    sample_w_replacement: bool = False,
    distortion="none"
) -> CustomDataLoader:
    """Deal with format of input path, and get relevant DataLoader"""
    data_loader = CustomDataLoader(
        path,
        nsample=nsample,
        batch_size=batch_size,
        num_workers=num_workers,
        transform=transform,
        seed=seed,
        random_sample=random_sample,
        sample_w_replacement=sample_w_replacement,
        distortion=distortion
    )

    return data_loader
