import hashlib
import time
from dataclasses import dataclass
import dataclasses
import numpy as np
import sympy as sp
import logging

from jax import Array
from dmmd import dmmd_blockwise_jax
from dmmd import dmmd_blockwise

logger = logging.getLogger(__name__)

# =========================
# Symbolic definitions
# =========================

def compute_palate(
    *,
    train_representations,
    test_representations,
    gen_representations,
    gen_gt,
    sigma: float,
):
    sigma3 = sigma / 3

    # ====================================================
    # m_palate → sigma → gen_representations
    # ====================================================
    dmmd_train_sigma, _ = dmmd_blockwise(
        train_representations,
        gen_representations,
        sigma=sigma,
    )

    dmmd_test_sigma, denominator_scale = dmmd_blockwise(
        test_representations,
        gen_representations,
        sigma=sigma,
    )

    # ====================================================
    # palate → sigma / 3 → gen_gt
    # ====================================================
    dmmd_train_sigma3, _ = dmmd_blockwise_jax(
        train_representations,
        gen_gt,
        sigma=sigma3,
    )

    dmmd_test_sigma3, _ = dmmd_blockwise_jax(
        test_representations,
        gen_gt,
        sigma=sigma3,
    )

    # ====================================================
    # Metrics
    # ====================================================
    palate = dmmd_test_sigma3 / (
        dmmd_test_sigma3 + dmmd_train_sigma3
    )

    m_palate = (
        dmmd_test_sigma / (2 * denominator_scale)
        + 0.5 * palate
    )

    # ====================================================
    # Return everything
    # ====================================================
    return {
        # metrics
        "m_palate": m_palate,
        "palate": palate,

        # sigmas
        "sigma": sigma,
        "sigma3": sigma3,

        # sigma DMMDs
        "dmmd_train_sigma": dmmd_train_sigma,
        "dmmd_test_sigma": dmmd_test_sigma,
        "denominator_scale": denominator_scale,

        # sigma/3 DMMDs
        "dmmd_train_sigma3": dmmd_train_sigma3,
        "dmmd_test_sigma3": dmmd_test_sigma3,
    }

