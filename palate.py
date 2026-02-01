import time
import logging
import numpy as np
from dmmd import dmmd_blockwise_jax, compute_all_dmmd
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

    train = jnp.asarray(train_representations, dtype=jnp.float32)
    test = jnp.asarray(test_representations, dtype=jnp.float32)
    gen = jnp.asarray(gen_representations, dtype=jnp.float32)
    gt = jnp.asarray(gen_gt, dtype=jnp.float32)

    (
        palate,
        m_palate,
        dmmd_test_gen,
        dmmd_train_gen_3,
        dmmd_test_gen_3,
        denominator_scale,
    ) = compute_all_dmmd(
        train, test, gen, gt, sigma
    )

    return {
        "palate": float(palate),
        "m_palate": float(m_palate),
        "dmmd_test_gen": float(dmmd_test_gen),
        "dmmd_train_gen_3": float(dmmd_train_gen_3),
        "dmmd_test_gen_3": float(dmmd_test_gen_3),
        "denominator_scale": float(denominator_scale),
        "sigma": sigma,
        "fraction": len(gen_gt) / len(gen_representations),
    }
