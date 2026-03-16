import inspect
from typing import Union, TypeAlias
import torch
import logging

from .dinov2 import DINOv2Encoder
from .dinov3 import DINOv3Encoder
from .inceptionv3 import InceptionV3Encoder
from .clip import CLIPEncoder

logger = logging.getLogger(__name__)

# Update the TypeAlias to include CLIPEncoder
DinoEncoder: TypeAlias = Union[DINOv2Encoder, DINOv3Encoder, InceptionV3Encoder, CLIPEncoder]

# Add "clip" to the MODELS dictionary
MODELS: dict[str, type[DinoEncoder]] = {
    "dinov2": DINOv2Encoder,
    "dinov3": DINOv3Encoder,
    "inceptionv3": InceptionV3Encoder,
    "clip": CLIPEncoder,
}


def load_encoder(model_name: str, device: torch.device, **kwargs) -> DinoEncoder:
    """Load feature extractor"""
    model_cls: type[DinoEncoder] = MODELS[model_name]

    signature = inspect.signature(model_cls.setup)
    arguments = list(signature.parameters.keys())
    arguments = arguments[1:]  # Omit `self`

    encoder: DinoEncoder = model_cls()

    # Filter kwargs for the specific model's setup
    setup_args = {arg: kwargs[arg] for arg in arguments if arg in kwargs}
    encoder.setup(**setup_args)

    encoder.name = model_name
    logger.info(f"Loaded {model_cls.__name__}")

    # Move the underlying model to GPU/CPU
    encoder.model.to(device)

    return encoder