import jax
import jax.numpy as jnp

_BLOCK_SIZE = 1000

def pad_to_block(x, block_size):
    n, d = x.shape
    pad = (-n) % block_size
    return jnp.pad(x, ((0, pad), (0, 0))), n


@jax.jit
def blockwise_kernel_mean(x, y, sigma, block_size=1024):
    x, n_x = pad_to_block(x, block_size)
    y, n_y = pad_to_block(y, block_size)

    d = x.shape[1]
    gamma = 1.0 / (2.0 * sigma**2)

    num_blocks_x = x.shape[0] // block_size
    num_blocks_y = y.shape[0] // block_size

    def body_fun(i, acc):
        bx = i // num_blocks_y
        by = i % num_blocks_y

        x_block = jax.lax.dynamic_slice(
            x,
            (bx * block_size, 0),
            (block_size, d),
        )
        y_block = jax.lax.dynamic_slice(
            y,
            (by * block_size, 0),
            (block_size, d),
        )

        x_sq = jnp.sum(x_block**2, axis=1, keepdims=True)
        y_sq = jnp.sum(y_block**2, axis=1, keepdims=True)

        k = jnp.exp(
            -gamma * (x_sq - 2 * x_block @ y_block.T + y_sq.T)
        )

        # Mask padded rows
        x_mask = (bx * block_size + jnp.arange(block_size)) < n_x
        y_mask = (by * block_size + jnp.arange(block_size)) < n_y
        k = k * x_mask[:, None] * y_mask[None, :]

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

