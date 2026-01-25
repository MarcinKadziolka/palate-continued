import jax
import jax.numpy as jnp


@jax.jit
def _rbf_block(x, y, sigma):
    x2 = jnp.sum(x * x, axis=1)[:, None]
    y2 = jnp.sum(y * y, axis=1)[None, :]
    return jnp.exp(-(x2 + y2 - 2.0 * x @ y.T) / (2.0 * sigma**2))


@jax.jit
def kernel_mean_blockwise(x, y, sigma, block_size):
    nx, d = x.shape
    ny = y.shape[0]

    nbx = (nx + block_size - 1) // block_size
    nby = (ny + block_size - 1) // block_size
    total_blocks = nbx * nby

    def body(i, acc):
        bi = i // nby
        bj = i % nby

        xs = bi * block_size
        ys = bj * block_size

        xb = jax.lax.dynamic_slice(
            x,
            (xs, 0),
            (jnp.minimum(block_size, nx - xs), d),
        )

        yb = jax.lax.dynamic_slice(
            y,
            (ys, 0),
            (jnp.minimum(block_size, ny - ys), d),
        )

        k = _rbf_block(xb, yb, sigma)
        return acc + jnp.sum(k)

    total = jax.lax.fori_loop(
        0,
        total_blocks,
        body,
        0.0,
    )

    return total / (nx * ny)


@jax.jit
def dmmd_exact(x, y, sigma, block_size=1024):
    kxx = kernel_mean_blockwise(x, x, sigma, block_size)
    kyy = kernel_mean_blockwise(y, y, sigma, block_size)
    kxy = kernel_mean_blockwise(x, y, sigma, block_size)
    return kxx + kyy - 2.0 * kxy
