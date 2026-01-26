import jax
import jax.numpy as jnp


@jax.jit
def _rbf_block(x, y, sigma):
    x_norm = jnp.sum(x**2, axis=1)[:, None]
    y_norm = jnp.sum(y**2, axis=1)[None, :]
    sq = x_norm + y_norm - 2.0 * x @ y.T
    return jnp.exp(-sq / (2.0 * sigma**2))


def _block_sum(x, y, sigma, block_size):
    n = x.shape[0]
    m = y.shape[0]

    def outer(i, acc):
        xi = x[i:i + block_size]

        def inner(j, acc2):
            yj = y[j:j + block_size]
            k = _rbf_block(xi, yj, sigma)
            return acc2 + jnp.sum(k)

        acc = jax.lax.fori_loop(
            0, (m + block_size - 1) // block_size,
            inner,
            acc,
        )
        return acc

    return jax.lax.fori_loop(
        0, (n + block_size - 1) // block_size,
        outer,
        0.0,
    )


def dmmd_blockwise_jax(x, y, sigma, block_size=1024):
    """
    Memory-efficient MMD^2 with RBF kernel.

    Args:
        x: [N, D]
        y: [M, D]
        sigma: RBF bandwidth
        block_size: block size for computation

    Returns:
        mmd2, kxx_plus_kyy
    """

    n = x.shape[0]
    m = y.shape[0]

    kxx = _block_sum(x, x, sigma, block_size) / (n * n)
    kyy = _block_sum(y, y, sigma, block_size) / (m * m)
    kxy = _block_sum(x, y, sigma, block_size) / (n * m)

    return kxx + kyy - 2.0 * kxy, kxx + kyy
