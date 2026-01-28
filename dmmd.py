import jax
import jax.numpy as jnp

# ------------------------------------------------------------
# RBF kernel block
# ------------------------------------------------------------
@jax.jit
def _rbf_block(x, y, sigma):
    x2 = jnp.sum(x * x, axis=1)[:, None]
    y2 = jnp.sum(y * y, axis=1)[None, :]
    return jnp.exp(-(x2 + y2 - 2.0 * x @ y.T) / (2.0 * sigma**2))


# ------------------------------------------------------------
# Exact blockwise kernel mean
# ------------------------------------------------------------
def kernel_mean_blockwise(x, y, sigma, block_size=1024):
    total = jnp.array(0.0)
    count = jnp.array(0.0)

    for i in range(0, x.shape[0], block_size):
        xb = x[i:i + block_size]
        for j in range(0, y.shape[0], block_size):
            yb = y[j:j + block_size]

            k = _rbf_block(xb, yb, sigma)
            total += jnp.sum(k)
            count += jnp.array(k.size, dtype=total.dtype)

    return total / count



# ------------------------------------------------------------
# MMD
# ------------------------------------------------------------
def dmmd_blockwise_jax(x, y, sigma, block_size=1024):
    kxx = kernel_mean_blockwise(x, x, sigma, block_size)
    kyy = kernel_mean_blockwise(y, y, sigma, block_size)
    kxy = kernel_mean_blockwise(x, y, sigma, block_size)
    return kxx + kyy - 2.0 * kxy, kxx + kyy