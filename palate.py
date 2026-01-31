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
    sigma3 = sigma/3.0
    train = jnp.asarray(train_representations, dtype=jnp.float32)
    test = jnp.asarray(test_representations, dtype=jnp.float32)
    gen = jnp.asarray(gen_representations, dtype=jnp.float32)
    gt = jnp.asarray(gen_gt, dtype=jnp.float32)

    # --- DMMDs ---
    dmmd_test_gen, denom = gaussian_mmd(test, gen, sigma)
    dmmd_train_gt, _ = gaussian_mmd(train, gt, sigma3)
    dmmd_test_gt, _ = gaussian_mmd(test, gt, sigma3)

    palate = dmmd_test_gt / (dmmd_test_gt + dmmd_train_gt)

    m_palate = (
            dmmd_test_gen / (2.0 * denom)
            + 0.5 * palate
    )

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
        "dmmd_test_gen": dmmd_test_gen,
        "dmmd_train_gt": dmmd_train_gt,
        "dmmd_test_gt": dmmd_test_gt,
        "sigma": sigma,
    }