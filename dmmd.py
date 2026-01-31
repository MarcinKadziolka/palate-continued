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





import jax
import jax.numpy as jnp
from jax import lax


# ------------------------------------------------------------
# Padding utilities
# ------------------------------------------------------------
def pad_to_block(x, block_size):
    n, d = x.shape
    pad = (-n) % block_size
    x_pad = jnp.pad(x, ((0, pad), (0, 0)))
    mask = jnp.arange(n + pad) < n
    return x_pad, mask, n


# ------------------------------------------------------------
# RBF kernel block
# ------------------------------------------------------------
@jax.jit
def _rbf_block(x, y, x2, y2, sigma):
    return jnp.exp(
        -(x2[:, None] + y2[None, :] - 2.0 * (x @ y.T))
        / (2.0 * sigma**2)
    )


# ------------------------------------------------------------
# Exact blockwise kernel mean (SAFE)
# ------------------------------------------------------------
def kernel_mean_blockwise(x, y, sigma, block_size=1024):
    # Pad inputs
    x, xmask, nx = pad_to_block(x, block_size)
    y, ymask, ny = pad_to_block(y, block_size)

    n_blocks_x = x.shape[0] // block_size
    n_blocks_y = y.shape[0] // block_size

    x2 = jnp.sum(x * x, axis=1)
    y2 = jnp.sum(y * y, axis=1)

    def outer_loop(i, total):
        xb = lax.dynamic_slice(x, (i * block_size, 0),
                               (block_size, x.shape[1]))
        x2b = lax.dynamic_slice(x2, (i * block_size,), (block_size,))
        xm = lax.dynamic_slice(xmask, (i * block_size,), (block_size,))

        def inner_loop(j, subtotal):
            yb = lax.dynamic_slice(y, (j * block_size, 0),
                                   (block_size, y.shape[1]))
            y2b = lax.dynamic_slice(y2, (j * block_size,), (block_size,))
            ym = lax.dynamic_slice(ymask, (j * block_size,), (block_size,))

            k = _rbf_block(xb, yb, x2b, y2b, sigma)

            # Mask padded rows
            mask = xm[:, None] & ym[None, :]
            k = jnp.where(mask, k, 0.0)

            return subtotal + jnp.sum(k)

        return lax.fori_loop(0, n_blocks_y, inner_loop, total)

    total = lax.fori_loop(0, n_blocks_x, outer_loop, 0.0)

    # Correct normalization
    return total / (nx * ny)


# ------------------------------------------------------------
# MMD
# ------------------------------------------------------------
@jax.jit
def dmmd_blockwise_jax(x, y, sigma, block_size=1024):
    kxx = kernel_mean_blockwise(x, x, sigma, block_size)
    kyy = kernel_mean_blockwise(y, y, sigma, block_size)
    kxy = kernel_mean_blockwise(x, y, sigma, block_size)
    return kxx + kyy - 2.0 * kxy, kxx + kyy


@jax.jit
def dmmd_fused_minimal(
    x,  x2,  xm,
    y,  y2,  ym,
    sigma,
):
    """
    Returns:
        dmmd  = kxx + kyy - 2*kxy
        denom = kxx + kyy
    """

    BS = 1024

    def kernel_sum(a, a2, am, b, b2, bm):
        na = a.shape[0] // BS
        nb = b.shape[0] // BS

        def outer(i, acc):
            ab = lax.dynamic_slice(a, (i * BS, 0), (BS, a.shape[1]))
            a2b = lax.dynamic_slice(a2, (i * BS,), (BS,))
            amb = lax.dynamic_slice(am, (i * BS,), (BS,))

            def inner(j, acc2):
                bb = lax.dynamic_slice(b, (j * BS, 0), (BS, b.shape[1]))
                b2b = lax.dynamic_slice(b2, (j * BS,), (BS,))
                bmb = lax.dynamic_slice(bm, (j * BS,), (BS,))

                k = jnp.exp(
                    -(a2b[:, None] + b2b[None, :] - 2.0 * ab @ bb.T)
                    / (2.0 * sigma**2)
                )

                mask = amb[:, None] & bmb[None, :]
                return acc2 + jnp.sum(jnp.where(mask, k, 0.0))

            return lax.fori_loop(0, nb, inner, acc)

        return lax.fori_loop(0, na, outer, 0.0)

    # --- kernel sums ---
    kxx = kernel_sum(x, x2, xm, x, x2, xm)
    kyy = kernel_sum(y, y2, ym, y, y2, ym)
    kxy = kernel_sum(x, x2, xm, y, y2, ym)

    nx = xm.sum()
    ny = ym.sum()

    dmmd = kxx / (nx * nx) + kyy / (ny * ny) - 2.0 * kxy / (nx * ny)
    denom = kxx / (nx * nx) + kyy / (ny * ny)

    return dmmd, denom

@jax.jit
def kernel_mean_precomputed(x, x2, xm, y, y2, ym, sigma, block_size):
    nbx = x.shape[0] // block_size
    nby = y.shape[0] // block_size

    def outer(i, acc):
        xb = lax.dynamic_slice(x, (i*block_size, 0), (block_size, x.shape[1]))
        x2b = lax.dynamic_slice(x2, (i*block_size,), (block_size,))
        xm_b = lax.dynamic_slice(xm, (i*block_size,), (block_size,))

        def inner(j, acc2):
            yb = lax.dynamic_slice(y, (j*block_size, 0), (block_size, y.shape[1]))
            y2b = lax.dynamic_slice(y2, (j*block_size,), (block_size,))
            ym_b = lax.dynamic_slice(ym, (j*block_size,), (block_size,))

            k = jnp.exp(
                -(x2b[:, None] + y2b[None, :] - 2 * xb @ yb.T)
                / (2 * sigma**2)
            )

            mask = xm_b[:, None] & ym_b[None, :]
            return acc2 + jnp.sum(jnp.where(mask, k, 0.0))

        return lax.fori_loop(0, nby, inner, acc)

    total = lax.fori_loop(0, nbx, outer, 0.0)
    return total

@jax.jit
def dmmd_fast(xp, x2, xm, nx,
              yp, y2, ym, ny,
              sigma, block_size):

    kxx = kernel_mean_precomputed(xp, x2, xm, xp, x2, xm, sigma, block_size) / (nx * nx)
    kyy = kernel_mean_precomputed(yp, y2, ym, yp, y2, ym, sigma, block_size) / (ny * ny)
    kxy = kernel_mean_precomputed(xp, x2, xm, yp, y2, ym, sigma, block_size) / (nx * ny)

    return kxx + kyy - 2 * kxy, kxx + kyy

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


@jax.jit
def gaussian_mmd(x, y, sigma):
    """
    Exact Gaussian MMD:
        MMD² = E[k(x,x)] + E[k(y,y)] - 2E[k(x,y)]
    """

    inv_sigma2 = 1.0 / (2.0 * sigma * sigma)

    # norms
    x2 = jnp.sum(x * x, axis=1, keepdims=True)
    y2 = jnp.sum(y * y, axis=1, keepdims=True)

    # pairwise squared distance
    dxx = x2 + x2.T - 2 * (x @ x.T)
    dyy = y2 + y2.T - 2 * (y @ y.T)
    dxy = x2 + y2.T - 2 * (x @ y.T)

    kxx = jnp.exp(-inv_sigma2 * dxx).mean()
    kyy = jnp.exp(-inv_sigma2 * dyy).mean()
    kxy = jnp.exp(-inv_sigma2 * dxy).mean()

    return kxx + kyy - 2 * kxy, kxx + kyy
@jax.jit
def gaussian_mmd_fast(x, y, sigma, block=1024):
    inv = 1.0 / (2.0 * sigma * sigma)

    x2 = jnp.sum(x * x, axis=1)
    y2 = jnp.sum(y * y, axis=1)

    nx = x.shape[0]
    ny = y.shape[0]

    def kxx_body(i, acc):
        xi = x[i:i+block]
        xi2 = x2[i:i+block]

        def inner(j, acc2):
            xj = x[j:j+block]
            xj2 = x2[j:j+block]
            d = xi2[:, None] + xj2[None, :] - 2 * xi @ xj.T
            return acc2 + jnp.sum(jnp.exp(-inv * d))

        return jax.lax.fori_loop(0, nx, inner, acc)

    def kyy_body(i, acc):
        yi = y[i:i+block]
        yi2 = y2[i:i+block]

        def inner(j, acc2):
            yj = y[j:j+block]
            yj2 = y2[j:j+block]
            d = yi2[:, None] + yj2[None, :] - 2 * yi @ yj.T
            return acc2 + jnp.sum(jnp.exp(-inv * d))

        return jax.lax.fori_loop(0, ny, inner, acc)

    def kxy_body(i, acc):
        xi = x[i:i+block]
        xi2 = x2[i:i+block]

        def inner(j, acc2):
            yj = y[j:j+block]
            yj2 = y2[j:j+block]
            d = xi2[:, None] + yj2[None, :] - 2 * xi @ yj.T
            return acc2 + jnp.sum(jnp.exp(-inv * d))

        return jax.lax.fori_loop(0, ny, inner, acc)

    kxx = jax.lax.fori_loop(0, nx, kxx_body, 0.0) / (nx * nx)
    kyy = jax.lax.fori_loop(0, ny, kyy_body, 0.0) / (ny * ny)
    kxy = jax.lax.fori_loop(0, nx, kxy_body, 0.0) / (nx * ny)

    return kxx + kyy - 2 * kxy, kxx + kyy
import jax
import jax.numpy as jnp
from jax import lax

# -----------------------------
# BLOCKWISE GAUSSIAN MMD
# -----------------------------

@jax.jit
def _block_mmd(x, x2, y, y2, inv_sigma2, block):
    nx = x.shape[0]
    ny = y.shape[0]

    def outer(i, acc):
        xi = x[i:i+block]
        xi2 = x2[i:i+block]

        def inner(j, acc2):
            yj = y[j:j+block]
            yj2 = y2[j:j+block]

            d = xi2[:, None] + yj2[None, :] - 2.0 * xi @ yj.T
            return acc2 + jnp.sum(jnp.exp(-inv_sigma2 * d))

        return lax.fori_loop(0, ny, inner, acc)

    return lax.fori_loop(0, nx, outer, 0.0)


@jax.jit
def gaussian_mmd_blockwise(x, y, sigma, block=1024):
    """
    Exact Gaussian MMD² using blockwise accumulation.
    Fast, stable, GPU-friendly.
    """

    inv = 1.0 / (2.0 * sigma * sigma)

    x = jnp.asarray(x, jnp.float32)
    y = jnp.asarray(y, jnp.float32)

    x2 = jnp.sum(x * x, axis=1)
    y2 = jnp.sum(y * y, axis=1)

    nx = x.shape[0]
    ny = y.shape[0]

    kxx = _block_mmd(x, x2, x, x2, inv, block) / (nx * nx)
    kyy = _block_mmd(y, y2, y, y2, inv, block) / (ny * ny)
    kxy = _block_mmd(x, x2, y, y2, inv, block) / (nx * ny)

    return kxx + kyy - 2.0 * kxy, kxx + kyy
