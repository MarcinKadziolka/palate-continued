import jax
import jax.numpy as jnp

BLOCK_SIZE = 1024


def pad_to_block(x, block_size):
    n, d = x.shape
    pad = (-n) % block_size
    if pad == 0:
        return x, n
    return jnp.pad(x, ((0, pad), (0, 0))), n


@jax.jit
def _kernel_mean_blockwise(x, y, sigma, nx, ny):
    B = BLOCK_SIZE
    nbx = x.shape[0] // B
    nby = y.shape[0] // B

    gamma = 1.0 / (2.0 * sigma * sigma)

    def body(i, acc):
        bi = i // nby
        bj = i % nby

        xb = jax.lax.dynamic_slice(
            x, (bi * B, 0), (B, x.shape[1])
        )
        yb = jax.lax.dynamic_slice(
            y, (bj * B, 0), (B, y.shape[1])
        )

        dist2 = (
            jnp.sum(xb * xb, axis=1)[:, None]
            + jnp.sum(yb * yb, axis=1)[None, :]
            - 2.0 * xb @ yb.T
        )

        return acc + jnp.sum(jnp.exp(-gamma * dist2))

    total = jax.lax.fori_loop(
        0, nbx * nby, body, 0.0
    )

    return total / (nx * ny)


def dmmd_blockwise_jax(x, y, sigma):
    x_pad, nx = pad_to_block(x, BLOCK_SIZE)
    y_pad, ny = pad_to_block(y, BLOCK_SIZE)

    kxx = _kernel_mean_blockwise(x_pad, x_pad, sigma, nx, nx)
    kyy = _kernel_mean_blockwise(y_pad, y_pad, sigma, ny, ny)
    kxy = _kernel_mean_blockwise(x_pad, y_pad, sigma, nx, ny)

    return kxx + kyy - 2.0 * kxy, kxx + kyy
