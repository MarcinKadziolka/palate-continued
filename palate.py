import time
import logging
import numpy as np
from dmmd import fused_E_pass
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


def train_gt_kernel(T, GT, sigma):
    T  = T.astype(jnp.float16)
    GT = GT.astype(jnp.float16)

    T2  = jnp.sum(T  * T,  axis=1, keepdims=True, dtype=jnp.float32)
    GT2 = jnp.sum(GT * GT, axis=1, keepdims=True, dtype=jnp.float32)

    D = T2 - 2 * (T @ GT.T).astype(jnp.float32) + GT2.T
    K = jnp.exp(-jnp.maximum(D, 0) / (2 * (sigma/3)**2))

    return jnp.sum(K), K.size


def compute_palate(E, G, GT, T, sigma):
    # Main fused pass
    K = fused_E_pass(E, G, GT, sigma)

    # Self kernels
    GG, nGG = self_kernel(G, sigma)
    GTGT, nGTGT = self_kernel(GT, sigma / 3)

    # Train × GT
    TGT, nTGT = train_gt_kernel(T, GT, sigma)

    # DMMDs
    dmmd_test_gen = (
        K["EE"][0] / K["EE"][1]
        + GG / nGG
        - 2 * (K["EG"][0] / K["EG"][1])
    )

    dmmd_train_gt = (
        K["EE"][0] / K["EE"][1]
        + GTGT / nGTGT
        - 2 * (TGT / nTGT)
    )

    palate = dmmd_train_gt / (dmmd_train_gt + 1e-12)
    m_palate = dmmd_test_gen / (2 * (K["EE"][0] / K["EE"][1] + GG / nGG)) + 0.5 * palate

    return {
        "palate": palate,
        "m_palate": m_palate,
        "dmmd_test_gen": dmmd_test_gen,
        "dmmd_train_gen_3": dmmd_train_gt,
    }






