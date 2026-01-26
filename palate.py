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

    dmmd_test_train, _ = dmmd_blockwise_jax(
        x=test_representations,
        y=train_representations,
        sigma=sigma,
    )

    logger.info("DMMD computed in %.3fs", time.time() - t0)

    # ---- Palate formulas ----
    palate = dmmd_test_gen / (dmmd_test_gen + dmmd_train_gen)
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
        "dmmd_test_train": dmmd_test_train,
        "denominator_scale": denominator_scale,
        "sigma": sigma,
    }
