import time
import logging
import numpy as np
from dmmd import fused_kernel_E_pass
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

def self_kernel(X, sigma):
    X = X.astype(jnp.float16)
    X2 = jnp.sum(X * X, axis=1, keepdims=True, dtype=jnp.float32)
    D = X2 - 2 * (X @ X.T).astype(jnp.float32) + X2.T
    K = jnp.exp(-jnp.maximum(D, 0) / (2 * sigma**2))
    return jnp.sum(K), K.size


def compute_palate_fast_minimal(E, G, GT, sigma):
    # main kernels
    K = fused_kernel_E_pass(E, G, GT, sigma)

    # self kernels
    GG, nGG = self_kernel(G, sigma)
    GTGT, nGTGT = self_kernel(GT, sigma / 3)

    # DMMDs
    dmmd_test_gen = (
        K["EE"][0] / K["EE"][1]
        + GG / nGG
        - 2 * (K["EG"][0] / K["EG"][1])
    )

    dmmd_test_gt = (
        K["EE"][0] / K["EE"][1]
        + GTGT / nGTGT
        - 2 * (K["EGT"][0] / K["EGT"][1])
    )

    denominator = (
        K["EE"][0] / K["EE"][1]
        + GG / nGG
    )

    palate = dmmd_test_gt / (dmmd_test_gt + 0.0)  # train term removed
    m_palate = dmmd_test_gen / (2 * denominator) + 0.5 * palate

    return {
        "palate": palate,
        "m_palate": m_palate,
        "dmmd_test_gen": dmmd_test_gen,
        "dmmd_test_gen_3": dmmd_test_gt,
        "denominator_scale": denominator,
    }





