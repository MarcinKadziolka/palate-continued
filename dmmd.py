import jax
import jax.numpy as jnp

_BLOCK_SIZE = 1000

@jax.jit
def blockwise_kernel_mean(x, y, sigma, block_size=1024):
    n_x = x.shape[0]
    n_y = y.shape[0]
    d = x.shape[1]

    gamma = 1.0 / (2.0 * sigma**2)

    num_blocks_x = (n_x + block_size - 1) // block_size
    num_blocks_y = (n_y + block_size - 1) // block_size

    def body_fun(i, acc):
        bx = i // num_blocks_y
        by = i % num_blocks_y

        x_start = bx * block_size
        y_start = by * block_size

        # Always slice full block_size
        x_block = jax.lax.dynamic_slice(
            x,
            (x_start, 0),
            (block_size, d),
        )
        y_block = jax.lax.dynamic_slice(
            y,
            (y_start, 0),
            (block_size, d),
        )

        # Mask for valid rows
        x_valid = (x_start + jnp.arange(block_size)) < n_x
        y_valid = (y_start + jnp.arange(block_size)) < n_y

        x_sq = jnp.sum(x_block**2, axis=1, keepdims=True)
        y_sq = jnp.sum(y_block**2, axis=1, keepdims=True)

        k = jnp.exp(
            -gamma * (x_sq - 2 * x_block @ y_block.T + y_sq.T)
        )

        # Mask invalid rows/cols
        k = k * x_valid[:, None] * y_valid[None, :]

        return acc + jnp.sum(k)

    total = jax.lax.fori_loop(
        0,
        num_blocks_x * num_blocks_y,
        body_fun,
        0.0,
    )

    return total / (n_x * n_y)




@jax.jit
def dmmd(x, y, sigma):
    kxx = blockwise_kernel_mean(x, x, sigma)
    kxy = blockwise_kernel_mean(x, y, sigma)
    kyy = blockwise_kernel_mean(y, y, sigma)
    return kxx + kyy - 2 * kxy, kxx + kyy

