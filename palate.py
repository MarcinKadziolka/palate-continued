import logging
import time

import jax.numpy as jnp
import numpy as np

from dmmd import dmmd_blockwise_jax

logger = logging.getLogger(__name__)


def prepare(x, block_size):
    x_pad, xmask, nx = pad_to_block(x, block_size)
    x2 = jnp.sum(x_pad * x_pad, axis=1)
    return x_pad, x2, xmask, nx


def pad_to_block(x, block_size=4096):
    n, d = x.shape
    pad = (-n) % block_size
    if pad == 0:
        return x
    return jnp.pad(x, ((0, pad), (0, 0)))


_BLOCK_SIZE = 4096


def compute_palate(
    *,
    train_representations: np.ndarray,
    test_representations: np.ndarray,
    gen_representations: np.ndarray,
    gen_gt: np.ndarray,
    sigma: float,
):
    """Compute palate and m_palate metrics."""

    logger.info("Computing DMMD values...")
    t0 = time.time()
    sigma3 = sigma / 3
    n_train = train_representations.shape[0]
    n_test = test_representations.shape[0]
    n_gen = gen_representations.shape[0]
    n_gt = gen_gt.shape[0]
    train_p = pad_to_block(jnp.asarray(train_representations))
    test_p = pad_to_block(jnp.asarray(test_representations))
    gen_p = pad_to_block(jnp.asarray(gen_representations))
    gt_p = pad_to_block(jnp.asarray(gen_gt))

    fraction = len(gen_gt) / len(gen_representations)

    # warmup
    _ = dmmd_blockwise_jax(
        train_p[:_BLOCK_SIZE],
        gen_p[:_BLOCK_SIZE],
        sigma,
        n_x=_BLOCK_SIZE,
        n_y=_BLOCK_SIZE,
    )

    # main values
    dmmd_test_gen, denominator_scale = dmmd_blockwise_jax(
        x=test_p,
        y=gen_p,
        sigma=sigma,
        n_x=n_test,
        n_y=n_gen,
    )
    if fraction > 0:
        dmmd_train_gen_3, _ = dmmd_blockwise_jax(
            x=train_p,
            y=gt_p,
            sigma=sigma3,
            n_x=n_train,
            n_y=n_gt,
        )

        dmmd_test_gen_3, _ = dmmd_blockwise_jax(
            x=test_p,
            y=gt_p,
            sigma=sigma3,
            n_x=n_test,
            n_y=n_gt,
        )
    else:
        return {
            "dmmd_test_gen": dmmd_test_gen,
            "denominator_scale": denominator_scale,
            "sigma": sigma,
            "fraction": fraction,
        }

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
        "dmmd_test_gen": dmmd_test_gen,
        "dmmd_train_gen_3": dmmd_train_gen_3,
        "dmmd_test_gen_3": dmmd_test_gen_3,
        "denominator_scale": denominator_scale,
        "sigma": sigma,
        "fraction": fraction,
    }
