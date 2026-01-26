import jax
import jax.numpy as jnp

# ============================================================
# Configuration
# ============================================================

BLOCK_SIZE = 1024   # must be static


# ============================================================
# Utilities
# ============================================================

def pad_to_block(x, block_size):
    """
    Pads x to a multiple of block_size.
    Returns padded array and original length.
    """
    n, d = x.shape
    pad = (-n) % block_size
    if pad == 0:
        return x, n
    return jnp.pad(x, ((0, pad), (0, 0))), n


# ============================================================
# Blockwise kernel mean (FAST + SAFE)
# ============================================================

@jax.jit
def _kernel_mean_blockwise(x, y, sigma):
    """
    Computes mean RBF kernel between x and y.

    x, y must already be padded to multiples of BLOCK_SIZE.
    """
    B = BLOCK_SIZE
    nx = x.shape[0]
    ny = y.shape[0]

    nbx = nx // B
    nby = ny // B

    gamma = 1.0 / (2.0 * sigma * sigma)

    def body(i, acc):
        bi = i // nby
        bj = i % nby

        xb = jax.lax.dynamic_slice(
            x, (bi * B, 0), (B, x.shape[1])
        )
        yb = jax.lax.dynamic_slice(
            y, (bj * B, 0), (B, y.shape[1])
        )

        # squared distances
        x2 = jnp.sum(xb * xb, axis=1)[:, None]
        y2 = jnp.sum(yb * yb, axis=1)[None, :]
        dist2 = x2 + y2 - 2.0 * xb @ yb.T

        return acc + jnp.sum(jnp.exp(-gamma * dist2))

    total = jax.lax.fori_loop(
        0,
        nbx * nby,
        body,
        0.0
    )

    return total / (nx * ny)


# ============================================================
# Public API: D-MMD
# ============================================================

def dmmd_blockwise_jax(x, y, sigma):
    """
    Computes:

        MMD^2(x, y) = kxx + kyy - 2 kxy

    using blockwise exact RBF kernel evaluation.
    """

    # Pad once (outside JIT)
    x_pad, nx = pad_to_block(x, BLOCK_SIZE)
    y_pad, ny = pad_to_block(y, BLOCK_SIZE)

    kxx = _kernel_mean_blockwise(x_pad, x_pad, sigma)
    kyy = _kernel_mean_blockwise(y_pad, y_pad, sigma)
    kxy = _kernel_mean_blockwise(x_pad, y_pad, sigma)

    return kxx + kyy - 2.0 * kxy, kxx + kyy
