import jax
import jax.numpy as jnp


# ============================================================
# Utility: pad to block size (required for JAX correctness)
# ============================================================
def pad_to_block(x, block_size):
    n, d = x.shape
    pad = (-n) % block_size
    if pad == 0:
        return x
    return jnp.pad(x, ((0, pad), (0, 0)))


# ============================================================
# RBF kernel
# ============================================================
@jax.jit
def _rbf_block(x, y, sigma):
    x2 = jnp.sum(x * x, axis=1)[:, None]
    y2 = jnp.sum(y * y, axis=1)[None, :]
    return jnp.exp(-(x2 + y2 - 2.0 * x @ y.T) / (2.0 * sigma**2))


# ============================================================
# Blockwise kernel mean (exact, JIT-safe)
# ============================================================
@jax.jit
def kernel_mean_blockwise(x, y, sigma, block_size):
    nx = x.shape[0]
    ny = y.shape[0]

    nbx = nx // block_size
    nby = ny // block_size

    def body(i, acc):
        bi = i // nby
        bj = i % nby

        xb = jax.lax.dynamic_slice(
            x,
            (bi * block_size, 0),
            (block_size, x.shape[1])
        )
        yb = jax.lax.dynamic_slice(
            y,
            (bj * block_size, 0),
            (block_size, y.shape[1])
        )

        k = _rbf_block(xb, yb, sigma)
        return acc + jnp.sum(k)

    total = jax.lax.fori_loop(
        0,
        nbx * nby,
        body,
        0.0
    )

    return total / (nx * ny)


# ============================================================
# Exact D-MMD
# ============================================================
def dmmd_exact(x, y, sigma, block_size=1024):
    # Pad ONCE (important!)
    x = pad_to_block(x, block_size)
    y = pad_to_block(y, block_size)

    kxx = kernel_mean_blockwise(x, x, sigma, block_size)
    kyy = kernel_mean_blockwise(y, y, sigma, block_size)
    kxy = kernel_mean_blockwise(x, y, sigma, block_size)

    return kxx + kyy - 2.0 * kxy
