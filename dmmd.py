import jax
import jax.numpy as jnp
from jax import lax


# ------------------------------------------------------------
# Padding utilities
# ------------------------------------------------------------
def pad_to_block(x, block_size):
    n, d = x.shape
    pad = (-n) % block_size
    x_pad = jnp.pad(x, ((0, pad), (0, 0)))
    mask = jnp.arange(n + pad) < n
    return x_pad, mask, n


# ------------------------------------------------------------
# RBF kernel block
# ------------------------------------------------------------
@jax.jit
def _rbf_block(x, y, x2, y2, sigma):
    return jnp.exp(
        -(x2[:, None] + y2[None, :] - 2.0 * (x @ y.T))
        / (2.0 * sigma**2)
    )


# ------------------------------------------------------------
# Exact blockwise kernel mean (SAFE)
# ------------------------------------------------------------
def kernel_mean_blockwise(x, y, sigma, block_size=100):
    # Pad inputs
    x, xmask, nx = pad_to_block(x, block_size)
    y, ymask, ny = pad_to_block(y, block_size)

    n_blocks_x = x.shape[0] // block_size
    n_blocks_y = y.shape[0] // block_size

    x2 = jnp.sum(x * x, axis=1)
    y2 = jnp.sum(y * y, axis=1)

    def outer_loop(i, total):
        xb = lax.dynamic_slice(x, (i * block_size, 0),
                               (block_size, x.shape[1]))
        x2b = lax.dynamic_slice(x2, (i * block_size,), (block_size,))
        xm = lax.dynamic_slice(xmask, (i * block_size,), (block_size,))

        def inner_loop(j, subtotal):
            yb = lax.dynamic_slice(y, (j * block_size, 0),
                                   (block_size, y.shape[1]))
            y2b = lax.dynamic_slice(y2, (j * block_size,), (block_size,))
            ym = lax.dynamic_slice(ymask, (j * block_size,), (block_size,))

            k = _rbf_block(xb, yb, x2b, y2b, sigma)

            # Mask padded rows
            mask = xm[:, None] & ym[None, :]
            k = jnp.where(mask, k, 0.0)

            return subtotal + jnp.sum(k)

        return lax.fori_loop(0, n_blocks_y, inner_loop, total)

    total = lax.fori_loop(0, n_blocks_x, outer_loop, 0.0)

    # Correct normalization
    return total / (nx * ny)


# ------------------------------------------------------------
# MMD
# ------------------------------------------------------------
@jax.jit
def dmmd_blockwise_jax(x, y, sigma, block_size=1024):
    kxx = kernel_mean_blockwise(x, x, sigma, block_size)
    kyy = kernel_mean_blockwise(y, y, sigma, block_size)
    kxy = kernel_mean_blockwise(x, y, sigma, block_size)
    return kxx + kyy - 2.0 * kxy, kxx + kyy