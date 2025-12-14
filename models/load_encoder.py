import inspect
from typing import Union
import torch

from .dinov2 import DINOv2Encoder
from .dinov3 import DINOv3Encoder
import logging
from typing import TypeAlias

logger = logging.getLogger(__name__)

DinoEncoder: TypeAlias = Union[DINOv2Encoder, DINOv3Encoder]

MODELS: dict[str, type[DinoEncoder]] = {
    "dinov2": DINOv2Encoder,
    "dinov3": DINOv3Encoder,
}


def load_encoder(model_name: str, device: torch.device, **kwargs) -> DinoEncoder:
    """Load feature extractor"""

    model_cls: type[DinoEncoder] = MODELS[model_name]
    # Get names of model_cls.setup arguments
    signature = inspect.signature(model_cls.setup)
    arguments = list(signature.parameters.keys())
    arguments = arguments[1:]  # Omit `self` arg

    # Initialize model using the `arguments` that have been passed in the `kwargs` dict
    encoder: DinoEncoder = model_cls(**{arg: kwargs[arg] for arg in arguments if arg in kwargs})
    encoder.name = model_name

    logger.info(f"Loaded {model_cls.__name__}")
    return encoder.to(device)