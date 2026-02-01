import time
import logging
import numpy as np
from dmmd import dmmd_from_blocks, compute_all_kernels
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


def compute_palate(T, E, G, GT, sigma):
    K = compute_all_kernels(T, E, G, GT, sigma)

    dmmd_train_gen = dmmd_from_blocks(K["TT"], K["GG"], K["TG"])
    dmmd_test_gen  = dmmd_from_blocks(K["EE"], K["GG"], K["EG"])

    dmmd_train_gt  = dmmd_from_blocks(K["TT"], K["GTGT"], K["TGT"])
    dmmd_test_gt   = dmmd_from_blocks(K["EE"], K["GTGT"], K["EGT"])

    palate = dmmd_test_gt / (dmmd_test_gt + dmmd_train_gt)
    m_palate = dmmd_test_gen / 2 + 0.5 * palate

    return {
        "palate": palate,
        "m_palate": m_palate,
        "dmmd_train_gen": dmmd_train_gen,
        "dmmd_test_gen": dmmd_test_gen,
        "dmmd_train_gen_3": dmmd_train_gt,
        "dmmd_test_gen_3": dmmd_test_gt,
    }

