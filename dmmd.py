import jax
import jax.numpy as jnp


@jax.jit
def _rbf_block(x, y, sigma):
    x_norm = jnp.sum(x**2, axis=1)[:, None]
    y_norm = jnp.sum(y**2, axis=1)[None, :]
    sq = x_norm + y_norm - 2.0 * x @ y.T
    return jnp.exp(-sq / (2.0 * sigma**2))


def _block_sum(x, y, sigma, block_size):
    n, d = x.shape
    m = y.shape[0]

    num_x_blocks = (n + block_size - 1) // block_size
    num_y_blocks = (m + block_size - 1) // block_size

    def outer(i, acc):
        xi = jax.lax.dynamic_slice(
            x,
            (i * block_size, 0),
            (block_size, d),
        )

        # mask for valid rows in last block
        xi_mask = (i * block_size + jnp.arange(block_size)) < n
        xi_mask = xi_mask[:, None]

        def inner(j, acc2):
            yj = jax.lax.dynamic_slice(
                y,
                (j * block_size, 0),
                (block_size, d),
            )

            yj_mask = (j * block_size + jnp.arange(block_size)) < m
            yj_mask = yj_mask[None, :]

            k = _rbf_block(xi, yj, sigma)

            # mask invalid entries
            k = k * xi_mask * yj_mask

            return acc2 + jnp.sum(k)

        return jax.lax.fori_loop(0, num_y_blocks, inner, acc)

    return jax.lax.fori_loop(0, num_x_blocks, outer, 0.0)


def dmmd_blockwise_jax(x, y, sigma, block_size=1024):
    n = x.shape[0]
    m = y.shape[0]

    kxx = _block_sum(x, x, sigma, block_size) / (n * n)
    kyy = _block_sum(y, y, sigma, block_size) / (m * m)
    kxy = _block_sum(x, y, sigma, block_size) / (n * m)

    return kxx + kyy - 2.0 * kxy, kxx + kyy
