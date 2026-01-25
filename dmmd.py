import jax
import jax.numpy as jnp

_BLOCK_SIZE = 1000

@jax.jit
def blockwise_kernel_mean(x, y, block_size=_BLOCK_SIZE):
    n_x = x.shape[0]
    n_y = y.shape[0]
    gamma = 1.0 / (2 * _SIGMA**2)

    def body_fun(i, acc):
        bx = i // num_blocks_y
        by = i % num_blocks_y

        x_start = bx * block_size
        y_start = by * block_size

        x_end = jnp.minimum(x_start + block_size, n_x)
        y_end = jnp.minimum(y_start + block_size, n_y)

        x_block = x[x_start:x_end]
        y_block = y[y_start:y_end]

        x_sq = jnp.sum(x_block ** 2, axis=1, keepdims=True)
        y_sq = jnp.sum(y_block ** 2, axis=1, keepdims=True)

        k = jnp.exp(
            -gamma * (
                x_sq
                - 2.0 * x_block @ y_block.T
                + y_sq.T
            )
        )

        return acc + jnp.sum(k)

    num_blocks_x = (n_x + block_size - 1) // block_size
    num_blocks_y = (n_y + block_size - 1) // block_size
    total_pairs = n_x * n_y

    total = jax.lax.fori_loop(
        0,
        num_blocks_x * num_blocks_y,
        body_fun,
        0.0,
    )

    return total / total_pairs



@jax.jit
def dmmd(x, y):
    kxx = blockwise_kernel_mean(x, x)
    kxy = blockwise_kernel_mean(x, y)
    kyy = blockwise_kernel_mean(y, y)
    return kxx + kyy - 2 * kxy, kxx + kyy

