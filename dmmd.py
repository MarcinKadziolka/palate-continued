import jax
import jax.numpy as jnp
from jax import lax

_BLOCK_SIZE = 1024


@jax.jit
def kernel_mean_blockwise(x, y, sigma, n_real, m_real):
    n, d = x.shape
    m = y.shape[0]

    x2 = jnp.sum(x * x, axis=1)
    y2 = jnp.sum(y * y, axis=1)

    nb = n // _BLOCK_SIZE
    mb = m // _BLOCK_SIZE

    def outer_loop(bi, acc):
        total, count = acc
        i = bi * _BLOCK_SIZE

        xb = lax.dynamic_slice(x, (i, 0), (_BLOCK_SIZE, d))
        x2b = lax.dynamic_slice(x2, (i,), (_BLOCK_SIZE,))

        valid_x = (i + jnp.arange(_BLOCK_SIZE)) < n_real

        def inner_loop(bj, acc2):
            total2, count2 = acc2
            j = bj * _BLOCK_SIZE

            yb = lax.dynamic_slice(y, (j, 0), (_BLOCK_SIZE, d))
            y2b = lax.dynamic_slice(y2, (j,), (_BLOCK_SIZE,))

            valid_y = (j + jnp.arange(_BLOCK_SIZE)) < m_real

            k = jnp.exp(
                -(x2b[:, None] + y2b[None, :] - 2 * xb @ yb.T) / (2.0 * sigma**2)
            )

            mask = valid_x[:, None] * valid_y[None, :]

            return (
                total2 + jnp.sum(k * mask),
                count2 + jnp.sum(mask),
            )

        return lax.fori_loop(0, mb, inner_loop, (total, count))

    total, count = lax.fori_loop(0, nb, outer_loop, (0.0, 0.0))
    return total / count


@jax.jit
def dmmd_blockwise_jax(x, y, sigma, n_x, n_y):
    kxx = kernel_mean_blockwise(x, x, sigma, n_x, n_x)
    kyy = kernel_mean_blockwise(y, y, sigma, n_y, n_y)
    kxy = kernel_mean_blockwise(x, y, sigma, n_x, n_y)
    return kxx + kyy - 2.0 * kxy, kxx + kyy
