import hashlib
import time
from dataclasses import dataclass
import dataclasses
import numpy as np
import sympy as sp
import logging
import time

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
    t_start = time.perf_counter()


    # ====================================================
    # m_palate → sigma → gen_representations
    # ====================================================
    t0 = time.perf_counter()
    dmmd_train_sigma, _ = dmmd_blockwise(
        train_representations,
        gen_representations,
        sigma=sigma,
    )
    t_dmmd_train_sigma = time.perf_counter() - t0

    t0 = time.perf_counter()
    dmmd_test_sigma, denominator_scale = dmmd_blockwise(
        test_representations,
        gen_representations,
        sigma=sigma,
    )
    t_dmmd_test_sigma = time.perf_counter() - t0

    # ====================================================
    # palate → sigma / 3 → gen_gt
    # ====================================================
    t0 = time.perf_counter()
    dmmd_train_sigma3, _ = dmmd_blockwise(
        train_representations,
        gen_gt,
        sigma=sigma3,
    )
    t_dmmd_train_sigma3 = time.perf_counter() - t0

    t0 = time.perf_counter()
    dmmd_test_sigma3, _ = dmmd_blockwise(
        test_representations,
        gen_gt,
        sigma=sigma3,
    )
    t_dmmd_test_sigma3 = time.perf_counter() - t0

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

    total_time = time.perf_counter() - t_start

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
        # timings (seconds)
        "time_dmmd_train_sigma": t_dmmd_train_sigma,
        "time_dmmd_test_sigma": t_dmmd_test_sigma,
        "time_dmmd_train_sigma3": t_dmmd_train_sigma3,
        "time_dmmd_test_sigma3": t_dmmd_test_sigma3,
        "time_dmmd_total": (
                t_dmmd_train_sigma
                + t_dmmd_test_sigma
                + t_dmmd_train_sigma3
                + t_dmmd_test_sigma3
        ),
        "time_palate_total": total_time,
    }

