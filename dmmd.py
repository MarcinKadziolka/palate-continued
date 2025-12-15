# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code has been adapted from the repository: https://github.com/google-research/google-research/blob/master/cmmd/main.py.

"""A memory-efficient MMD implementation in JAX."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxlib import xla_client
from jax import Array


# The bandwidth parameter for the Gaussian RBF kernel. See the paper for more
# details.
_SIGMA = 0.01 #nowa sigma


_BLOCK_SIZE = 1000


@jax.jit
def blockwise_kernel_mean(x, y, sigma) -> float:
    """Computes the mean of the kernel function in a blockwise manner without constructing full matrices.

    sigma: The bandwidth parameter for the Gaussian RBF kernel. See the paper for more details.
    """
    n = x.shape[0]
    num_blocks = n // _BLOCK_SIZE  # Ensure divisibility for simplicity
    gamma = 1 / (2 * sigma**2)

    def block_kernel_mean(i, mean_accum):
        row_start = (i // num_blocks) * _BLOCK_SIZE
        col_start = (i % num_blocks) * _BLOCK_SIZE

        # Slice blocks
        x_block = jax.lax.dynamic_slice(x, (row_start, 0), (_BLOCK_SIZE, x.shape[1]))
        y_block = jax.lax.dynamic_slice(y, (col_start, 0), (_BLOCK_SIZE, y.shape[1]))

        # Compute squared norms for blocks
        x_sq_block = jnp.diag(jnp.matmul(x_block, x_block.T))
        y_sq_block = jnp.diag(jnp.matmul(y_block, y_block.T))

        # Compute kernel matrix block
        k_block = jnp.exp(
            -gamma
            * (
                -2 * jnp.matmul(x_block, y_block.T)
                + jnp.expand_dims(x_sq_block, 1)
                + jnp.expand_dims(y_sq_block, 0)
            )
        )

        return mean_accum + jnp.mean(k_block)

    mean_sum = jax.lax.fori_loop(0, num_blocks**2, block_kernel_mean, 0.0)
    return mean_sum / (num_blocks**2)


@jax.jit
<<<<<<< Updated upstream
def dmmd_blockwise(x: np.ndarray, y: np.ndarray, sigma: float) -> tuple[xla_client.Array, xla_client.Array]:
=======
def dmmd_blockwise(x: np.ndarray, y: np.ndarray) -> tuple[Array, Array]:
>>>>>>> Stashed changes
    """Computes D-MMD using blockwise kernel computation."""
    mean_kxx = blockwise_kernel_mean(x, x, sigma)
    mean_kxy = blockwise_kernel_mean(x, y, sigma)
    mean_kyy = blockwise_kernel_mean(y, y, sigma)

    return mean_kxx + mean_kyy - 2 * mean_kxy, mean_kxx + mean_kyy
