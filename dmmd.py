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

@jax.jit
def rbf_block(x, y, x2, y2, sigma):
    return jnp.exp(-(x2 + y2 - 2.0 * x @ y.T) / (2.0 * sigma**2))

# ------------------------------------------------------------
# Exact blockwise kernel mean
# ------------------------------------------------------------
def kernel_mean_blockwise(x, y, sigma, block_size=16384):
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

from jax import lax

@jax.jit
def kernel_mean_blockwise(x, y, sigma, block_size):
    n, d = x.shape
    m = y.shape[0]

    x2_all = jnp.sum(x * x, axis=1)
    y2_all = jnp.sum(y * y, axis=1)

    def outer_loop(i, carry):
        total, count = carry
        xb = x[i:i+block_size]
        x2b = x2_all[i:i+block_size]

        def inner_loop(j, inner_carry):
            total_inner, count_inner = inner_carry
            yb = y[j:j+block_size]
            y2b = y2_all[j:j+block_size]

            k = rbf_block(xb, yb, x2b[:, None], y2b[None, :], sigma)
            return (
                total_inner + jnp.sum(k),
                count_inner + k.size,
            )

        total, count = lax.fori_loop(
            0,
            m,
            inner_loop,
            (total, count),
            unroll=1
        )
        return total, count

    total, count = lax.fori_loop(
        0,
        n,
        outer_loop,
        (0.0, 0.0),
        unroll=1
    )

    return total / count

from jax import lax
import jax.numpy as jnp
_BLOCK_SIZE = 16384 

@jax.jit
def kernel_mean_blockwise(x, y, sigma, block_size):
    n, d = x.shape
    m = y.shape[0]

    x2_all = jnp.sum(x * x, axis=1)
    y2_all = jnp.sum(y * y, axis=1)

    nb = (n + block_size - 1) // block_size
    mb = (m + block_size - 1) // block_size

    def outer_loop(bi, carry):
        total, count = carry

        i = bi * block_size
        xb = lax.dynamic_slice(x, (i, 0), (block_size, d))
        x2b = lax.dynamic_slice(x2_all, (i,), (block_size,))

        def inner_loop(bj, inner):
            total_inner, count_inner = inner

            j = bj * block_size
            yb = lax.dynamic_slice(y, (j, 0), (block_size, d))
            y2b = lax.dynamic_slice(y2_all, (j,), (block_size,))

            k = jnp.exp(
                -(x2b[:, None] + y2b[None, :] - 2 * xb @ yb.T)
                / (2.0 * sigma**2)
            )

            return (
                total_inner + jnp.sum(k),
                count_inner + k.size,
            )

        total, count = lax.fori_loop(
            0, mb, inner_loop, (total, count)
        )

        return total, count

    total, count = lax.fori_loop(
        0, nb, outer_loop, (0.0, 0.0)
    )

    return total / count
import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def kernel_mean_blockwise(x, y, sigma):
    n, d = x.shape
    m = y.shape[0]

    x2_all = jnp.sum(x * x, axis=1)
    y2_all = jnp.sum(y * y, axis=1)

    nb = (n + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    mb = (m + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    def outer_loop(bi, carry):
        total, count = carry

        i = bi * _BLOCK_SIZE
        xb = lax.dynamic_slice(x, (i, 0), (_BLOCK_SIZE, d))
        x2b = lax.dynamic_slice(x2_all, (i,), (_BLOCK_SIZE,))

        def inner_loop(bj, inner):
            total_inner, count_inner = inner

            j = bj * _BLOCK_SIZE
            yb = lax.dynamic_slice(y, (j, 0), (_BLOCK_SIZE, d))
            y2b = lax.dynamic_slice(y2_all, (j,), (_BLOCK_SIZE,))

            k = jnp.exp(
                -(x2b[:, None] + y2b[None, :] - 2 * xb @ yb.T)
                / (2.0 * sigma**2)
            )

            return total_inner + jnp.sum(k), count_inner + k.size

        total, count = lax.fori_loop(0, mb, inner_loop, (total, count))
        return total, count

    total, count = lax.fori_loop(0, nb, outer_loop, (0.0, 0.0))
    return total / count
import jax
import jax.numpy as jnp
from jax import lax

_BLOCK_SIZE = 1024


@jax.jit
def kernel_mean_blockwise(x, y, sigma):
    n, d = x.shape
    m = y.shape[0]

    x2_all = jnp.sum(x * x, axis=1)
    y2_all = jnp.sum(y * y, axis=1)

    nb = (n + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    mb = (m + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    def outer_loop(bi, carry):
        total, count = carry

        i = bi * _BLOCK_SIZE

        # --- safe slice ---
        xb = lax.dynamic_slice(
            x,
            (i, 0),
            (_BLOCK_SIZE, d),
        )
        x2b = lax.dynamic_slice(
            x2_all,
            (i,),
            (_BLOCK_SIZE,),
        )

        # valid row mask
        valid_x = (i + jnp.arange(_BLOCK_SIZE)) < n
        valid_x = valid_x.astype(x.dtype)

        def inner_loop(bj, inner):
            total_inner, count_inner = inner
            j = bj * _BLOCK_SIZE

            yb = lax.dynamic_slice(
                y,
                (j, 0),
                (_BLOCK_SIZE, d),
            )
            y2b = lax.dynamic_slice(
                y2_all,
                (j,),
                (_BLOCK_SIZE,),
            )

            valid_y = (j + jnp.arange(_BLOCK_SIZE)) < m
            valid_y = valid_y.astype(y.dtype)

            k = jnp.exp(
                -(x2b[:, None] + y2b[None, :] - 2 * xb @ yb.T)
                / (2.0 * sigma**2)
            )

            # mask invalid rows/cols
            k = k * valid_x[:, None] * valid_y[None, :]

            return (
                total_inner + jnp.sum(k),
                count_inner + jnp.sum(valid_x[:, None] * valid_y[None, :]),
            )

        total, count = lax.fori_loop(0, mb, inner_loop, (total, count))
        return total, count

    total, count = lax.fori_loop(0, nb, outer_loop, (0.0, 0.0))
    return total / count

_BLOCK_SIZE = 1024


# ============================================================
# Utilities
# ============================================================

def pad_to_block(x, block_size=_BLOCK_SIZE):
    """Pad x to multiple of block_size along axis 0."""
    n, d = x.shape
    pad = (-n) % block_size
    if pad == 0:
        return x
    return jnp.pad(x, ((0, pad), (0, 0)))


# ============================================================
# Kernel Mean (Blockwise, JIT-safe)
# ============================================================

@jax.jit
def kernel_mean_blockwise(x, y, sigma):
    n, d = x.shape
    m = y.shape[0]

    x2 = jnp.sum(x * x, axis=1)
    y2 = jnp.sum(y * y, axis=1)

    nb = n // _BLOCK_SIZE
    mb = m // _BLOCK_SIZE

    def outer_loop(bi, acc):
        total, count = acc
        i = bi * _BLOCK_SIZE

        xb = lax.dynamic_slice(x, (i, 0), (_BLOCK_SIZE, d))
        x2b = lax.dynamic_slice(x2, (i,), (_BLOCK_SIZE,))

        def inner_loop(bj, acc2):
            total2, count2 = acc2
            j = bj * _BLOCK_SIZE

            yb = lax.dynamic_slice(y, (j, 0), (_BLOCK_SIZE, d))
            y2b = lax.dynamic_slice(y2, (j,), (_BLOCK_SIZE,))

            k = jnp.exp(
                -(x2b[:, None] + y2b[None, :] - 2 * xb @ yb.T)
                / (2.0 * sigma**2)
            )

            return (
                total2 + jnp.sum(k),
                count2 + k.size,
            )

        return lax.fori_loop(0, mb, inner_loop, (total, count))

    total, count = lax.fori_loop(0, nb, outer_loop, (0.0, 0.0))
    return total / count

import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def kernel_mean_blockwise(x, y, sigma):
    n, d = x.shape
    m = y.shape[0]

    x2 = jnp.sum(x * x, axis=1)
    y2 = jnp.sum(y * y, axis=1)

    nb = (n + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    mb = (m + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    def outer_loop(bi, acc):
        total, count = acc
        i = bi * _BLOCK_SIZE

        xb = lax.dynamic_slice(x, (i, 0), (_BLOCK_SIZE, d))
        x2b = lax.dynamic_slice(x2, (i,), (_BLOCK_SIZE,))

        valid_x = (i + jnp.arange(_BLOCK_SIZE)) < n
        valid_x = valid_x.astype(x.dtype)

        def inner_loop(bj, acc2):
            total2, count2 = acc2
            j = bj * _BLOCK_SIZE

            yb = lax.dynamic_slice(y, (j, 0), (_BLOCK_SIZE, d))
            y2b = lax.dynamic_slice(y2, (j,), (_BLOCK_SIZE,))

            valid_y = (j + jnp.arange(_BLOCK_SIZE)) < m
            valid_y = valid_y.astype(y.dtype)

            k = jnp.exp(
                -(x2b[:, None] + y2b[None, :] - 2 * xb @ yb.T)
                / (2.0 * sigma**2)
            )

            mask = valid_x[:, None] * valid_y[None, :]
            k = k * mask

            return (
                total2 + jnp.sum(k),
                count2 + jnp.sum(mask),
            )

        return lax.fori_loop(0, mb, inner_loop, (total, count))

    total, count = lax.fori_loop(0, nb, outer_loop, (0.0, 0.0))
    return total / count
def pad_to_block(x, block_size):
    n, d = x.shape
    pad = (-n) % block_size
    if pad == 0:
        return x
    return jnp.pad(x, ((0, pad), (0, 0)))
_BLOCK_SIZE = 1024

@jax.jit
def kernel_mean_blockwise(x, y, sigma):
    n, d = x.shape
    m = y.shape[0]

    x2 = jnp.sum(x * x, axis=1)
    y2 = jnp.sum(y * y, axis=1)

    nb = n // _BLOCK_SIZE
    mb = m // _BLOCK_SIZE

    def outer_loop(bi, acc):
        total, count = acc
        i = bi * _BLOCK_SIZE

        xb = lax.dynamic_slice(x, (i, 0), (_BLOCK_SIZE, d))
        x2b = lax.dynamic_slice(x2, (i,), (_BLOCK_SIZE,))

        def inner_loop(bj, acc2):
            total2, count2 = acc2
            j = bj * _BLOCK_SIZE

            yb = lax.dynamic_slice(y, (j, 0), (_BLOCK_SIZE, d))
            y2b = lax.dynamic_slice(y2, (j,), (_BLOCK_SIZE,))

            k = jnp.exp(
                -(x2b[:, None] + y2b[None, :] - 2 * xb @ yb.T)
                / (2 * sigma**2)
            )

            return total2 + jnp.sum(k), count2 + k.size

        return lax.fori_loop(0, mb, inner_loop, (total, count))

    total, count = lax.fori_loop(0, nb, outer_loop, (0.0, 0.0))
    return total / count

# ------------------------------------------------------------
# MMD
# ------------------------------------------------------------
@jax.jit
def kernel_mean_blockwise(x, y, sigma, n_real, m_real):
    n, d = x.shape
    m = y.shape[0]

    x2 = jnp.sum(x * x, axis=1)
    y2 = jnp.sum(y * y, axis=1)

    nb = n // _BLOCK_SIZE
    mb = m // _BLOCK_SIZE

    def outer_loop(bi, acc):
        total, count = acc
        i = bi * _BLOCK_SIZE

        xb = lax.dynamic_slice(x, (i, 0), (_BLOCK_SIZE, d))
        x2b = lax.dynamic_slice(x2, (i,), (_BLOCK_SIZE,))

        valid_x = (i + jnp.arange(_BLOCK_SIZE)) < n_real

        def inner_loop(bj, acc2):
            total2, count2 = acc2
            j = bj * _BLOCK_SIZE

            yb = lax.dynamic_slice(y, (j, 0), (_BLOCK_SIZE, d))
            y2b = lax.dynamic_slice(y2, (j,), (_BLOCK_SIZE,))

            valid_y = (j + jnp.arange(_BLOCK_SIZE)) < m_real

            k = jnp.exp(
                -(x2b[:, None] + y2b[None, :] - 2 * xb @ yb.T)
                / (2.0 * sigma**2)
            )

            mask = valid_x[:, None] * valid_y[None, :]

            return (
                total2 + jnp.sum(k * mask),
                count2 + jnp.sum(mask),
            )

        return lax.fori_loop(0, mb, inner_loop, (total, count))

    total, count = lax.fori_loop(0, nb, outer_loop, (0.0, 0.0))
    return total / count



@jax.jit
def dmmd_blockwise_jax(x, y, sigma):
    kxx = kernel_mean_blockwise(x, x, sigma)
    kyy = kernel_mean_blockwise(y, y, sigma)
    kxy = kernel_mean_blockwise(x, y, sigma)
    return kxx + kyy - 2.0 * kxy, kxx + kyy

@jax.jit
def dmmd_blockwise_jax(x, y, sigma):
    x = pad_to_block(x, _BLOCK_SIZE)
    y = pad_to_block(y, _BLOCK_SIZE)

    kxx = kernel_mean_blockwise(x, x, sigma)
    kyy = kernel_mean_blockwise(y, y, sigma)
    kxy = kernel_mean_blockwise(x, y, sigma)

    return kxx + kyy - 2.0 * kxy, kxx + kyy
@jax.jit
def dmmd_blockwise_jax(x, y, sigma, n_x, n_y):
    kxx = kernel_mean_blockwise(x, x, sigma, n_x, n_x)
    kyy = kernel_mean_blockwise(y, y, sigma, n_y, n_y)
    kxy = kernel_mean_blockwise(x, y, sigma, n_x, n_y)
    return kxx + kyy - 2.0 * kxy, kxx + kyy

