# This code has been adapted from: https://github.com/layer6ai-labs/dgm-eval/blob/master/dgm_eval/dataloaders.py

import os
import sys
import pathlib
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import logging
import torchvision
import torchvision.transforms

from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}

TORCHVISION_DATA_PATH = "./data/"


def get_files_at_path(path):
    """Return list of all files at path of type IMAGE_EXTENSIONS"""

    files = sorted([file for ext in IMAGE_EXTENSIONS for file in path.glob(f"*.{ext}")])

    return files


class ImagePathDataset(Dataset):
    """
    Create a custom dataset from a list of image files on disk

    Files must have image extensions specified in IMAGE_EXTENSIONS
    """

    def __init__(self, files, transform=None):
        self.files = sorted(files)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
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
                [p for p in image_path.iterdir() if p.is_dir() and p.name.isdigit()],
                key=get_order
            )
            logger.info(f"Found {len(class_dirs)} classes in {image_path}")
            self.files = []
            print(class_dirs, "ccc")
            for f in class_dirs:
                files_in_path = get_files_at_path(f)
                self.files += files_in_path
                self.labels.extend([class_idx for _ in range(len(files_in_path))])
                class_idx += 1
        self.labels = np.array(self.labels, dtype=np.int32)
        # print(f'len labels {len(self.labels)}')

        # Confirm data at path is in proper format
        try:
            self.data_set = ImagePathDataset(self.files, transform=self.transform)
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
    )

    return data_loader
