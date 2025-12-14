# This code has been adapted from: https://github.com/layer6ai-labs/dgm-eval/blob/master/dgm_eval/representations.py

import numpy as np
from tqdm import tqdm
import torch
from torch.nn.functional import adaptive_avg_pool2d
from models.load_encoder import DinoEncoder
from dataloader import CustomDataLoader
import logging
logger = logging.getLogger(__name__)

def get_representations(model: DinoEncoder, DataLoader: CustomDataLoader, device: torch.device, normalized: bool = False) -> np.ndarray:
    """Extracts features from all images in DataLoader given model.

    Params:
    -- model       : Instance of Encoder such as inception or CLIP or dinov2
    -- DataLoader  : DataLoader containing image files, or torchvision.dataset

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    logger.info(f"Computing representations for {DataLoader.path} using {model.__class__.__name__}.")
    model.eval()

    start_idx = 0

    for ibatch, batch in enumerate(tqdm(DataLoader.data_loader)):
        if isinstance(batch, list):
            # batch is likely list[array(images), array(labels)]
            batch = batch[0]

        if not torch.is_tensor(batch):
            # assume batch is then e.g. AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
            batch = batch["pixel_values"]
            batch = batch[:, 0]

        # Convert grayscale to RGB
        if batch.ndim == 3:
            batch.unsqueeze_(1)
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)

        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

            if not torch.is_tensor(pred):  # Some encoders output tuples or lists
                pred = pred[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.dim() > 2:
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2)

        if normalized:
            pred = torch.nn.functional.normalize(pred, dim=-1)
        pred = pred.cpu().numpy()

        if ibatch == 0:
            # initialize output array with full dataset size
            dims = pred.shape[-1]
            pred_arr = np.empty((DataLoader.nimages, dims))

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr
