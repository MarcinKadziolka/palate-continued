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
                -(x2b[:, None] + y2b[None, :] - 2 * xb @ yb.T)
                / (2.0 * sigma**2)
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
def kernel_mean_full(x, y, sigma):
    x = x.astype(jnp.float32)
    y = y.astype(jnp.float32)
    sigma = jnp.float32(sigma)

    x2 = jnp.sum(x * x, axis=1, keepdims=True)      # [n,1]
    y2 = jnp.sum(y * y, axis=1, keepdims=True).T    # [1,m]

    dist = x2 + y2 - 2.0 * (x @ y.T)                # [n,m]
    k = jnp.exp(-dist / (2.0 * sigma * sigma))
    return jnp.mean(k)


@jax.jit
def dmmd_blockwise_jax(x, y, sigma, n_x, n_y):
    kxx = kernel_mean_blockwise(x, x, sigma, n_x, n_x)
    kyy = kernel_mean_blockwise(y, y, sigma, n_y, n_y)
    kxy = kernel_mean_blockwise(x, y, sigma, n_x, n_y)
    return kxx + kyy - 2.0 * kxy, kxx + kyy

def dmmd_auto(x, y, sigma, n_x, n_y, threshold=20000):
    if max(n_x, n_y) <= threshold:
        x_r = x[:n_x]
        y_r = y[:n_y]

        kxx = kernel_mean_full(x_r, x_r, sigma)
        kyy = kernel_mean_full(y_r, y_r, sigma)
        kxy = kernel_mean_full(x_r, y_r, sigma)

        dmmd = kxx + kyy - 2.0 * kxy
        denom = kxx + kyy
        return dmmd, denom

    return dmmd_blockwise_jax(x, y, sigma, n_x, n_y)

