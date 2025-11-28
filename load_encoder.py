# This code has been adapted from: https://github.com/layer6ai-labs/dgm-eval/blob/master/dgm_eval/models/load_encoder.py

import inspect
from .dinov2 import DINOv2Encoder
#----zmiana-----
from .dinov3 import DINOv3Encoder #dodany import



MODELS = {
    "dinov2": DINOv2Encoder,
    "dinov3": DINOv3Encoder, #dodane
}


def load_encoder(model_name, device, **kwargs):
    """Load feature extractor"""

    model_cls = MODELS[model_name]

    # Get names of model_cls.setup arguments
    signature = inspect.signature(model_cls.setup)
    arguments = list(signature.parameters.keys())
    arguments = arguments[1:]  # Omit `self` arg

    # Initialize model using the `arguments` that have been passed in the `kwargs` dict
    encoder = model_cls(**{arg: kwargs[arg] for arg in arguments if arg in kwargs})
    encoder.name = model_name

    return encoder.to(device)
