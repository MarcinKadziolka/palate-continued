import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def _rbf_block_precomputed(x, y, x2, y2, sigma):
    # x: [B1, D]
    # y: [B2, D]
    # x2: [B1]
    # y2: [B2]
    return jnp.exp(
        -(x2[:, None] + y2[None, :] - 2.0 * (x @ y.T))
        / (2.0 * sigma**2)
    )


def kernel_mean_blockwise(x, y, sigma, block_size=1024):
    n, m = x.shape[0], y.shape[0]

    x2 = jnp.sum(x * x, axis=1)
    y2 = jnp.sum(y * y, axis=1)

    def outer_loop(i, acc):
        total, count = acc

        xb = x[i:i + block_size]
        x2b = x2[i:i + block_size]

        def inner_loop(j, inner_acc):
            total_inner, count_inner = inner_acc

            yb = y[j:j + block_size]
            y2b = y2[j:j + block_size]

            k = _rbf_block_precomputed(xb, yb, x2b, y2b, sigma)
            return (
                total_inner + jnp.sum(k),
                count_inner + k.size,
            )

        total, count = lax.fori_loop(
            0,
            (m + block_size - 1) // block_size,
            lambda j, acc: inner_loop(j * block_size, acc),
            (total, count),
        )

        return total, count

    total, count = lax.fori_loop(
        0,
        (n + block_size - 1) // block_size,
        lambda i, acc: outer_loop(i * block_size, acc),
        (0.0, 0.0),
    )

    return total / count


@jax.jit
def dmmd_blockwise_jax(x, y, sigma, block_size=1024):
    kxx = kernel_mean_blockwise(x, x, sigma, block_size)
    kyy = kernel_mean_blockwise(y, y, sigma, block_size)
    kxy = kernel_mean_blockwise(x, y, sigma, block_size)
    return kxx + kyy - 2.0 * kxy, kxx + kyy
