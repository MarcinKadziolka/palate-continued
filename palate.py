import time
import logging
import numpy as np
from dmmd import dmmd_blockwise_jax
import jax
import jax.numpy as jnp
from jax import lax

logger = logging.getLogger(__name__)

def pad_to_block(x, block_size):
    n, d = x.shape
    pad = (-n) % block_size
    x_pad = jnp.pad(x, ((0, pad), (0, 0)))
    mask = jnp.arange(n + pad) < n
    return x_pad, mask, n


def prepare(x, block_size):
    x_pad, xmask, nx = pad_to_block(x, block_size)
    x2 = jnp.sum(x_pad * x_pad, axis=1)
    return x_pad, x2, xmask, nx


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
    # prepare once
    train_p = prepare(train_representations)
    test_p = prepare(test_representations)
    gen_p = prepare(gen_representations)
    gt_p = prepare(gen_gt)

    # MMDs
    dmmd_test_gen, denom_scale = dmmd_fused_minimal(
        *test_p,
        *gen_p,
        sigma
    )

    dmmd_train_gt, _ = dmmd_fused_minimal(
        *train_p,
        *gt_p,
        sigma / 3
    )

    dmmd_test_gt, _ = dmmd_fused_minimal(
        *test_p,
        *gt_p,
        sigma / 3
    )

    # Final metrics
    palate = dmmd_test_gt / (dmmd_test_gt + dmmd_train_gt)

    m_palate = (
            dmmd_test_gen / (2.0 * denom_scale)
            + 0.5 * palate
    )

    '''
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
        y=gen_gt,
        sigma=sigma3,
    )
    '''
    logger.info("DMMD computed in %.3fs", time.time() - t0)

    # ---- Palate formulas ----

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