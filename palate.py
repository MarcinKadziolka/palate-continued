import time
import logging
import numpy as np
from dmmd import dmmd_blockwise_jax

logger = logging.getLogger(__name__)


def compute_palate(
    *,
    train_representations: np.ndarray,
    test_representations: np.ndarray,
    gen_representations: np.ndarray,
    gen_gt: np.ndarray,
    sigma: float,
):
    """
    Compute palate and m_palate metrics.

    Returns:
        dict with:
            - palate
            - m_palate
            - dmmd_train_gen
            - dmmd_test_gen
            - dmmd_test_train
            - denominator_scale
            - sigma
    """
    logger.info("Computing DMMD values...")
    t0 = time.time()
    sigma3 = sigma / 3

    dmmd_train_gen, _ = dmmd_blockwise_jax(
        x=train_representations,
        y=gen_representations,
        sigma=sigma,
    )

    dmmd_test_gen, denominator_scale = dmmd_blockwise_jax(
        x=test_representations,
        y=gen_representations,
        sigma=sigma,
    )

    dmmd_train_gen_3, _ = dmmd_blockwise_jax(
        x=train_representations,
        y=gen_gt,
        sigma=sigma3,
    )

    dmmd_test_gen_3, _ = dmmd_blockwise_jax(
        x=test_representations,
        y=gen_representations,
        sigma=sigma3,
    )

    logger.info("DMMD computed in %.3fs", time.time() - t0)

    # ---- Palate formulas ----
    palate = dmmd_test_gen_3 / (dmmd_test_gen_3 + dmmd_train_gen_3)
    m_palate = dmmd_test_gen / (2 * denominator_scale) + 0.5 * palate

    logger.info(
        "Palate computed (m_palate=%.6f, palate=%.6f)",
        m_palate,
        palate,
    )

    return {
        "palate": palate,
        "m_palate": m_palate,
        "dmmd_train_gen": dmmd_train_gen,
        "dmmd_test_gen": dmmd_test_gen,
        "dmmd_train_gen_3": dmmd_train_gen_3,
        "dmmd_test_gen_3": dmmd_test_gen_3,
        "denominator_scale": denominator_scale,
        "sigma": sigma,
    }
