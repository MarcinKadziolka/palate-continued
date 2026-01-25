import jax
import jax.numpy as jnp

# ============================================================
# Configuration
# ============================================================

_BLOCK_SIZE = 1000


# ============================================================
# Utilities
# ============================================================

def _pad(x, block):
    """Pad x to multiple of block size and return mask."""
    n = x.shape[0]
    pad = (-n) % block
    x_pad = jnp.pad(x, ((0, pad), (0, 0)))
    mask = jnp.arange(x_pad.shape[0]) < n
    return x_pad, mask


# ============================================================
# Core kernel computation
# ============================================================

@jax.jit
def _blockwise_kernel_mean(x, y, mx, my, sigma):
    """
    Computes mean RBF kernel between x and y using blockwise evaluation.
    Handles padding correctly via masks.
    """
    n = x.shape[0]
    B = _BLOCK_SIZE
    nb = n // B
    gamma = 1.0 / (2.0 * sigma**2)

    def body(i, acc):
        bi = i // nb
        bj = i % nb

        xs = jax.lax.dynamic_slice(x, (bi * B, 0), (B, x.shape[1]))
        ys = jax.lax.dynamic_slice(y, (bj * B, 0), (B, y.shape[1]))

        mxs = jax.lax.dynamic_slice(mx, (bi * B,), (B,))
        mys = jax.lax.dynamic_slice(my, (bj * B,), (B,))

        mask = mxs[:, None] & mys[None, :]

        x2 = jnp.sum(xs**2, axis=1)[:, None]
        y2 = jnp.sum(ys**2, axis=1)[None, :]
        k = jnp.exp(-gamma * (x2 + y2 - 2 * xs @ ys.T))

        return acc + jnp.sum(k * mask)

    total = jax.lax.fori_loop(0, nb * nb, body, 0.0)
    count = jnp.sum(mx) * jnp.sum(my)

    return total / count


# ============================================================
# Public API
# ============================================================

def dmmd_blockwise(x, y, sigma):
    """
    Computes D-MMD(x, y) with correct normalization.

    Returns:
        mmd_value, normalization_term
    """
    x, mx = _pad(x, _BLOCK_SIZE)
    y, my = _pad(y, _BLOCK_SIZE)

    kxx = _blockwise_kernel_mean(x, x, mx, mx, sigma)
    kyy = _blockwise_kernel_mean(y, y, my, my, sigma)
    kxy = _blockwise_kernel_mean(x, y, mx, my, sigma)

    return kxx + kyy - 2 * kxy, kxx + kyy