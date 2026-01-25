import jax
import jax.numpy as jnp

_BLOCK_SIZE = 1000


def _pad_to_block(x, block):
    n = x.shape[0]
    pad = (-n) % block
    return jnp.pad(x, ((0, pad), (0, 0))), n


@jax.jit
def _blockwise_kernel_mean(x, y, sigma):
    n = x.shape[0]
    B = _BLOCK_SIZE
    nb = n // B
    gamma = 1.0 / (2.0 * sigma**2)

    def body(i, acc):
        bi = i // nb
        bj = i % nb

        xs = jax.lax.dynamic_slice(x, (bi * B, 0), (B, x.shape[1]))
        ys = jax.lax.dynamic_slice(y, (bj * B, 0), (B, y.shape[1]))

        x2 = jnp.sum(xs**2, axis=1)[:, None]
        y2 = jnp.sum(ys**2, axis=1)[None, :]
        k = jnp.exp(-gamma * (x2 + y2 - 2 * xs @ ys.T))

        return acc + jnp.sum(k)

    total = jax.lax.fori_loop(0, nb * nb, body, 0.0)
    return total / jnp.asarray(n * n, dtype=total.dtype)

def dmmd_blockwise(x, y, sigma):
    """
    Fast, stable D-MMD.
    Works for different sizes.
    """

    x, nx = _pad_to_block(x, _BLOCK_SIZE)
    y, ny = _pad_to_block(y, _BLOCK_SIZE)

    kxx = _blockwise_kernel_mean(x, x, sigma)
    kyy = _blockwise_kernel_mean(y, y, sigma)
    kxy = _blockwise_kernel_mean(x, y, sigma)

    return kxx + kyy - 2 * kxy, kxx + kyy
