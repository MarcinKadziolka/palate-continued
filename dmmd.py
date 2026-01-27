import jax
import jax.numpy as jnp
from jax import lax


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def pad_to_block(x, block_size):
    n, d = x.shape
    pad = (-n) % block_size
    x = jnp.pad(x, ((0, pad), (0, 0)))
    mask = jnp.arange(n + pad) < n
    return x, mask, n


@jax.jit
def _rbf_block(x, y, x2, y2, sigma):
    return jnp.exp(
        -(x2[:, None] + y2[None, :] - 2.0 * (x @ y.T))
        / (2.0 * sigma**2)
    )


# ------------------------------------------------------------
# Kernel mean (exact, masked, symmetric)
# ------------------------------------------------------------
def kernel_mean_blockwise(x, y, sigma, block_size=1024, symmetric=False):
    x, xm, nx = pad_to_block(x, block_size)
    y, ym, ny = pad_to_block(y, block_size)

    nbx = x.shape[0] // block_size
    nby = y.shape[0] // block_size

    x2 = jnp.sum(x * x, axis=1)
    y2 = jnp.sum(y * y, axis=1)

    def outer(i, acc):
        xb = lax.dynamic_slice(x, (i * block_size, 0), (block_size, x.shape[1]))
        x2b = lax.dynamic_slice(x2, (i * block_size,), (block_size,))
        xm_b = lax.dynamic_slice(xm, (i * block_size,), (block_size,))

        def inner(j, acc2):
            yb = lax.dynamic_slice(y, (j * block_size, 0), (block_size, y.shape[1]))
            y2b = lax.dynamic_slice(y2, (j * block_size,), (block_size,))
            ym_b = lax.dynamic_slice(ym, (j * block_size,), (block_size,))

            k = _rbf_block(xb, yb, x2b, y2b, sigma)

            mask = xm_b[:, None] & ym_b[None, :]
            k = jnp.where(mask, k, 0.0)

            # symmetry handling
            factor = jnp.where(
                (symmetric & (i != j)), 2.0, 1.0
            )

            return acc2 + factor * jnp.sum(k)

        j0 = i if symmetric else 0
        return lax.fori_loop(j0, nby, inner, acc)

    total = lax.fori_loop(0, nbx, outer, 0.0)

    denom = nx * ny
    return total / denom


# ------------------------------------------------------------
# MMD (fast + exact)
# ------------------------------------------------------------
@jax.jit
def dmmd_blockwise_jax(x, y, sigma, block_size=1024):
    kxx = kernel_mean_blockwise(x, x, sigma, block_size, symmetric=True)
    kyy = kernel_mean_blockwise(y, y, sigma, block_size, symmetric=True)
    kxy = kernel_mean_blockwise(x, y, sigma, block_size, symmetric=False)
    return kxx + kyy - 2.0 * kxy, kxx + kyy
