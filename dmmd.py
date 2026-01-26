import jax
import jax.numpy as jnp


@jax.jit
def _rbf_block(x, y, sigma):
    x_norm = jnp.sum(x**2, axis=1)[:, None]
    y_norm = jnp.sum(y**2, axis=1)[None, :]
    sq = x_norm + y_norm - 2.0 * x @ y.T
    return jnp.exp(-sq / (2.0 * sigma**2))


def _pad_to_block(x, block_size):
    n, d = x.shape
    pad = (-n) % block_size
    return jnp.pad(x, ((0, pad), (0, 0))), n


def _block_sum(x, y, sigma, block_size):
    x, nx = _pad_to_block(x, block_size)
    y, ny = _pad_to_block(y, block_size)

    n_blocks = x.shape[0] // block_size
    m_blocks = y.shape[0] // block_size

    def outer(i, acc):
        xi = x[i * block_size:(i + 1) * block_size]

        xi_mask = (i * block_size + jnp.arange(block_size)) < nx
        xi_mask = xi_mask[:, None]

        def inner(j, acc2):
            yj = y[j * block_size:(j + 1) * block_size]

            yj_mask = (j * block_size + jnp.arange(block_size)) < ny
            yj_mask = yj_mask[None, :]

            k = _rbf_block(xi, yj, sigma)
            k = k * xi_mask * yj_mask

            return acc2 + jnp.sum(k)

        return jax.lax.fori_loop(0, m_blocks, inner, acc)

    return jax.lax.fori_loop(0, n_blocks, outer, 0.0)


def dmmd_blockwise_jax(x, y, sigma, block_size=1024):
    n = x.shape[0]
    m = y.shape[0]

    kxx = _block_sum(x, x, sigma, block_size) / (n * n)
    kyy = _block_sum(y, y, sigma, block_size) / (m * m)
    kxy = _block_sum(x, y, sigma, block_size) / (n * m)

    return kxx + kyy - 2.0 * kxy, kxx + kyy
