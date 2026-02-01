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

import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=("sigma",))
def compute_all_dmmd(
    train,
    test,
    gen,
    gt,
    sigma,
):
    sigma3 = sigma / 3

    d_test_gen, denom = dmmd_blockwise_jax(test, gen, sigma)

    d_train_gt, _ = dmmd_blockwise_jax(train, gt, sigma3)
    d_test_gt, _ = dmmd_blockwise_jax(test, gt, sigma3)

    palate = d_test_gt / (d_test_gt + d_train_gt)
    m_palate = d_test_gen / (2 * denom) + 0.5 * palate

    return (
        palate,
        m_palate,
        d_test_gen,
        d_train_gt,
        d_test_gt,
        denom,
    )
import jax
import jax.numpy as jnp

@jax.jit
def kernel_sum_fast(X, Y, sigma):
    X = X.astype(jnp.float16)
    Y = Y.astype(jnp.float16)

    x2 = jnp.sum(X * X, axis=1, keepdims=True)
    y2 = jnp.sum(Y * Y, axis=1, keepdims=True)

    dist2 = x2 - 2 * X @ Y.T + y2.T
    K = jnp.exp(-dist2 / (2 * sigma**2))

    return jnp.sum(K), jnp.asarray(K.size, jnp.float32)


def kernel_sum_auto(X, Y, sigma, *, threshold=40_000, block=4096):
    n, m = X.shape[0], Y.shape[0]

    # =========================
    # FAST PATH (small matrices)
    # =========================
    if n <= threshold and m <= threshold:
        X = X.astype(jnp.float16)
        Y = Y.astype(jnp.float16)

        x2 = jnp.sum(X * X, axis=1, keepdims=True, dtype=jnp.float32)
        y2 = jnp.sum(Y * Y, axis=1, keepdims=True, dtype=jnp.float32)

        dist2 = x2 - 2 * (X @ Y.T).astype(jnp.float32) + y2.T
        dist2 = jnp.maximum(dist2, 0.0)

        K = jnp.exp(-dist2 / (2 * sigma**2))

        return (
            jnp.sum(K),
            jnp.asarray(K.size, dtype=jnp.float32),
        )

    # =========================
    # SAFE PATH (blocked)
    # =========================
    total = jnp.array(0.0, dtype=jnp.float32)
    count = jnp.array(0.0, dtype=jnp.float32)

    for i in range(0, n, block):
        Xi = X[i:i+block].astype(jnp.float16)
        Xi2 = jnp.sum(Xi * Xi, axis=1, keepdims=True, dtype=jnp.float32)

        for j in range(0, m, block):
            Yj = Y[j:j+block].astype(jnp.float16)
            Yj2 = jnp.sum(Yj * Yj, axis=1, keepdims=True, dtype=jnp.float32)

            dist2 = Xi2 - 2 * (Xi @ Yj.T).astype(jnp.float32) + Yj2.T
            dist2 = jnp.maximum(dist2, 0.0)

            K = jnp.exp(-dist2 / (2 * sigma**2))

            total += jnp.sum(K)
            count += jnp.asarray(K.size, dtype=jnp.float32)

    return total, count





def dmmd_from_blocks(Kxx, Kyy, Kxy):
    return (
        Kxx[0] / Kxx[1] +
        Kyy[0] / Kyy[1] -
        2 * (Kxy[0] / Kxy[1])
    )


def compute_all_kernels(T, E, G, GT, sigma):
    return {
        "TT": kernel_sum_auto(T, T, sigma),
        "EE": kernel_sum_auto(E, E, sigma),
        "GG": kernel_sum_auto(G, G, sigma),
        "GTGT": kernel_sum_auto(GT, GT, sigma),

        "TG": kernel_sum_auto(T, G, sigma),
        "EG": kernel_sum_auto(E, G, sigma),
        "TGT": kernel_sum_auto(T, GT, sigma),
        "EGT": kernel_sum_auto(E, GT, sigma),
    }


