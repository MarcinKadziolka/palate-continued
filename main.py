# This script contains modified parts of code from repository: https://github.com/layer6ai-labs/dgm-eval

import time
import csv
import dataclasses
import logging
import os
import pathlib
import uuid
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import Literal, Optional, Callable
from scipy.special import logsumexp

from jaxlib.xla_client import Array
import numpy as np
import torch
from palate_local_knn import compute_local_palate_knn
from palate_local_knn import compute_global_palate_fast
from palate_local_knn import compute_global_palate_fast_normalized
from palate_local_knn import compute_global_palate_fast_anisotropic
import jax
import jax.numpy as jnp
from jax import jit
from dataloader import CustomDataLoader
from dataloader import get_dataloader
from models.load_encoder import DinoEncoder
from models.load_encoder import MODELS, load_encoder
from palate import compute_palate
from representations import get_representations
from dmmd import dmmd_blockwise_jax

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "--model",
    type=str,
    default="dinov2",
    choices=MODELS.keys(),
    help="Encoder model used to generate representations.",
)

parser.add_argument(
    "--nsample",
    type=int,
    default=10000,
    help="Maximum number of images used per dataset.",
)

parser.add_argument(
    "--sigma",
    type=float,
    default=0.01,
    help="Sigma to use in blockwise_kernel_mean in dmmd.py",
)

parser.add_argument(
    "-bs",
    "--batch_size",
    type=int,
    default=50,
    help="Batch size to use. If needed, equals to min(batch_size, nsample).",
)

parser.add_argument(
    "--num-workers",
    type=int,
    help="Number of processes to use for data loading. "
    "Defaults to `min(8, num_cpus)`",
)

parser.add_argument(
    "--device", type=str, default=None, help="Device to use. Like cuda, cuda:0 or cpu"
)

parser.add_argument(
    "path",
    type=str,
    nargs="+",
    help="Paths to image datasets in order: train test gen_1 gen_2 ... gen_n. At least 3 paths are required.",
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="./output",
    help="Root directory for all experiment outputs.",
)

parser.add_argument(
    "--tau",
    type=float,
    default=-9.0,
    help="Explicit global KDE log-density threshold. Overrides --kde_percentile."
)

parser.add_argument(
    "--exp_dir",
    type=str,
    default=None,
    help="Name of the experiment directory. If not provided, a unique ID is generated. Parent is --output_dir",
)

parser.add_argument(
    "--dino_ckpt",
    type=str,
    default="/shared/results/gmdziarm/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    help="Path to dinov3 weights (used only if --model dinov3).",
)


parser.add_argument("--seed", type=int, default=13579, help="Random seed")

parser.add_argument(
    "--clean_resize", action="store_true", help="Use clean resizing (from pillow)"
)

parser.add_argument(
    "--load_npz",
    action="store_true",
    help="Run in image-free mode. Paths must be .npz files containing representations."
)

parser.add_argument(
    "--depth",
    type=int,
    default=0,
    help="Negative depth for internal layers, positive 1 for after projection head.",
)

parser.add_argument(
    "--kde_percentile",
    type=float,
    nargs="+",
    default=5.0,
    help="List of percentiles for global KDE threshold grid search (in %)."
)


parser.add_argument(
    "--repr_dir",
    type=str,
    default="./saved_representations",
    help="Directory for cached representations.",
)

parser.add_argument(
    "--save", action="store_true", help="Save computed representations to repr_dir."
)

parser.add_argument(
    "--load",
    action="store_true",
    help="Load representations from repr_dir instead of recomputing.",
)


def get_device_and_num_workers(
    device: Literal["cuda", "cpu"], num_workers: int
) -> tuple[torch.device, int]:
    if device is None:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(device)

    if num_workers is None:
        # num_avail_cpus = len(os.sched_getaffinity(0)) zmiana przez windows
        try:
            num_avail_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            num_avail_cpus = os.cpu_count()
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = num_workers

    logger.info(f"Device: {device}, num_workers: {num_workers}")

    return device, num_workers


def get_dataloader_from_path(
    path: str,
    model_transform: Callable,
    num_workers: int,
    args: Namespace,
    sample_w_replacement: bool = False,
) -> CustomDataLoader:
    dataloader = get_dataloader(
        path,
        args.nsample,
        args.batch_size,
        num_workers,
        seed=args.seed,
        sample_w_replacement=sample_w_replacement,
        transform=lambda x: model_transform(x),
    )
    return dataloader

def now():
    return time.perf_counter()

def create_unique_exp_dir() -> str:
    if os.getenv("OAR_JOB_ID"):
        unique_str = os.getenv("OAR_JOB_ID")
    else:
        unique_str = uuid.uuid4()
    exp_dir = str(unique_str)[:8]
    return exp_dir


def write_to_txt(
    scores: dict[str, Array | str],
    output_dir: str,
    model: DinoEncoder,
    train_path: str,
    test_path: str,
    gen_path: str,
    nsample: int,
    sigma,
):
    model_arch = model.arch_str if model is not None else "npz"
    out_file = "metrics_summary.txt"
    out_path = os.path.join(output_dir, out_file)

    with open(out_path, "a") as f:
        f.write(f"Model: {model_arch}\n")
        f.write(f"Train: {train_path}\nTest: {test_path}\nGen: {gen_path}\nnsample: {nsample}\nsigma: {sigma}\n")
        for key, value in scores.items():
            f.write(f"{key}: {value}\n")
        f.write("\n" + "=" * 50 + "\n\n")


def write_to_csv(
    scores: dict[str, Array | str],
    output_dir,
    model,
    train_name,
    test_name,
    gen_name,
    nsample,
    sigma,
):
    csv_file = os.path.join(output_dir, "metrics_summary.csv")
    file_exists = os.path.isfile(csv_file)

    model_arch = model.arch_str if model is not None else "npz"

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["model_arch", "train", "test", "ten", "nsample", "sigma"] + list(scores.keys())
            writer.writerow(header)
        row = [model_arch, train_name, test_name, gen_name, nsample, sigma] + list(scores.values())
        writer.writerow(row)


def get_last_x_dirs(path: str, x=2):
    parts = pathlib.Path(path).parts
    x = min(x, len(parts))
    return "_".join(parts[-x:])

def save_score(
        palate_components,
        output_dir,
        model,
        train_path,
        test_path,
        gen_path,
        nsample,
        sigma,
):
    train_name = get_last_x_dirs(train_path)
    test_name = get_last_x_dirs(test_path)
    gen_name = get_last_x_dirs(gen_path)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    scores = palate_components


    write_to_txt(scores, output_dir, model, train_path, test_path, gen_path, nsample, sigma)
    write_to_csv(scores, output_dir, model, train_name, test_name, gen_name, nsample, sigma)

    logger.info(
        f"Scores of:\ntrain: {train_path}\ntest: {test_path}\ngen: {gen_path}\nsaved to dir: {output_dir}"
    )


def get_model(args: Namespace, device: torch.device, ckpt: str) -> DinoEncoder:
    return load_encoder(
        args.model,
        device,
        ckpt=ckpt,
        arch=None,
        clean_resize=args.clean_resize,
        sinception=True if args.model == "sinception" else False,
        depth=args.depth,
    )


def compute_representations(
    path: str, model: DinoEncoder, num_workers: int, device, args: Namespace
) -> np.ndarray:
    """
    Compute or load representations for the given path.

    Args:
        path (str): Path to the data.
        model: The model used to compute representations.
        num_workers (int): Number of workers for the dataloader.
        device: The device to use for computation (e.g., "cuda" or "cpu").
        args: Command-line arguments or configuration object.

    Returns:
        np.ndarray: Computed or loaded representations.
    """

    if args.load:
        loaded_reps = load_reps_from_path(args.repr_dir, path, model, args.nsample)
        if loaded_reps is not None:
            return loaded_reps

    logger.warning("Load path doesn't exist.")
    dataloader: CustomDataLoader = get_dataloader_from_path(
        path, model.transform, num_workers, args
    )

    representations = get_representations(model, dataloader, device, normalized=False)

    if args.save:
        save_representations(
            args.repr_dir, path, representations, model, dataloader, args.nsample
        )

    return representations


def save_representations(
    output_dir: str,
    path: str,
    reps,
    model: DinoEncoder,
    dataloader: CustomDataLoader,
    nsample: int,
):
    """Save representations and other info to disk at file_path"""
    # Create a unique file path for saving
    out_path = get_path(output_dir, path, model, nsample)

    # Ensure the output directory exists
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare hyperparameters for saving
    hyperparams = vars(dataloader).copy()  # Remove keys that can't be pickled
    hyperparams.pop("transform", None)
    hyperparams.pop("data_loader", None)
    hyperparams.pop("data_set", None)

    logger.info(f"Saving representations to: {out_path}.")
    np.savez(out_path, model=model, reps=reps, hparams=hyperparams)


def load_reps_from_path(
    saved_dir: str, path: str, model: DinoEncoder, nsample: int
) -> Optional[np.ndarray]:
    """
    Load representations from a saved .npz file if it exists.

    Args:
        saved_dir (str): Directory where the representations are saved.
        model: The model used to generate the representations.
        dataloader: The dataloader used to generate the representations.
        nsample (int): Number of samples used to generate the representations.

    Returns:
        np.ndarray: Loaded representations if the file exists, otherwise None.
    """
    # Generate the file path
    load_path = get_path(saved_dir, path, model, nsample)
    logger.info(f"Checking if load path exists: {load_path}.")

    if os.path.exists(load_path):
        saved_file = np.load(f"{load_path}")
        reps = saved_file["reps"]
        logger.info(f"Loaded representations from {load_path}.")
        return reps
    else:
        return None

def load_reps_from_npz(path: str) -> np.ndarray:
    if not path.endswith(".npz"):
        raise ValueError(f"Expected .npz file, got: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Representation file not found: {path}")

    data = np.load(path)
    if "reps" not in data:
        raise KeyError(f"'reps' key not found in {path}")

    logger.info(f"Loaded representations from NPZ: {path}")
    return data["reps"]

def get_path(output_dir: str, path: str, model: DinoEncoder, nsample: int) -> str:
    """Generate a unique file path for saving representations"""

    dataset_name = get_last_x_dirs(path)
    model_arch = model.arch_str if model is not None else "npz"

    return os.path.join(output_dir, f"{model_arch}_{dataset_name}_{nsample}.npz")


def write_arguments(args: Namespace, output_dir: str, filename: str = "arguments.txt"):
    """
    Writes all arguments from the `args` object to a file, one argument per line.

    Args:
        args: The argparse.Namespace object containing the arguments.
        output_dir: The directory where the file will be saved.
        filename: The name of the file to write the arguments to.
    """
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, filename)

    with open(file_path, "a") as f:
        for arg in vars(args):
            value = getattr(args, arg)
            f.write(f"{arg}: {value}\n")
        f.write("\n" + "=" * 50 + "\n\n")

from scipy.special import logsumexp

@jax.jit
def _kde_chunk(query, data, inv_sigma2):
    q_norm = jnp.sum(query**2 * inv_sigma2, axis=1, keepdims=True)
    d_norm = jnp.sum(data**2 * inv_sigma2, axis=1)
    cross = (query * inv_sigma2) @ data.T
    return jax.scipy.special.logsumexp(
        -0.5 * (q_norm + d_norm - 2.0 * cross),
        axis=1
    )


def log_kde_jax(query, data, sigma, batch_size=1024):
    query = jnp.asarray(query, dtype=jnp.float32)
    data  = jnp.asarray(data, dtype=jnp.float32)

    inv_sigma2 = jnp.asarray(1.0 / (sigma ** 2), dtype=jnp.float32)
    logN = jnp.log(data.shape[0])

    outs = []
    for i in range(0, len(query), batch_size):
        q = query[i:i + batch_size]
        outs.append(_kde_chunk(q, data, inv_sigma2))

    return jnp.concatenate(outs) - logN


import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp


@jax.jit
def _kde_block(query, data, inv_sigma2):
    """
    Computes:
        logsumexp(-0.5 * ||q - x||^2)
    for a block of query points.
    """
    q_norm = jnp.sum(query**2 * inv_sigma2, axis=1, keepdims=True)
    d_norm = jnp.sum(data**2 * inv_sigma2, axis=1)
    cross  = (query * inv_sigma2) @ data.T

    dist = q_norm + d_norm - 2.0 * cross
    return logsumexp(-0.5 * dist, axis=1)


def log_kde_exact(query, data, sigma, batch_size=1024):
    """
    Exact KDE:
        log p(x) = log(1/N * sum_j exp(-||x - y_j||^2 / 2σ²))

    No approximation, no padding bias, numerically stable.
    """

    query = jnp.asarray(query, dtype=jnp.float32)
    data  = jnp.asarray(data,  dtype=jnp.float32)

    inv_sigma2 = 1.0 / (sigma ** 2)
    N = data.shape[0]

    n_blocks = (len(query) + batch_size - 1) // batch_size
    out = jnp.zeros((n_blocks, batch_size), dtype=jnp.float32)

    def body(i, out):
        q = lax.dynamic_slice(
            query,
            (i * batch_size, 0),
            (batch_size, query.shape[1])
        )
        logsum = _kde_block(q, data, inv_sigma2)
        return out.at[i].set(logsum)

    out = lax.fori_loop(0, n_blocks, body, out)

    return out.reshape(-1)[:len(query)] - jnp.log(N)

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp



@jax.jit
def log_kde_fast(query, data, sigma):
    inv_sigma2 = 1.0 / (sigma ** 2)

    q_norm = jnp.sum(query**2 * inv_sigma2, axis=1, keepdims=True)
    d_norm = jnp.sum(data**2 * inv_sigma2, axis=1)

    cross = (query * inv_sigma2) @ data.T
    dist = q_norm + d_norm - 2.0 * cross

    return logsumexp(-0.5 * dist, axis=1) - jnp.log(data.shape[0])

def use_fast_kde(n_query, n_data):
    return (n_query * n_data) <= 400_000_000
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


@jax.jit
def _kde_block(query, data, d_norm, inv_sigma2):
    """
    query: [B, D]
    data:  [N, D]
    d_norm: [N]
    inv_sigma2: [D]
    """
    q_norm = jnp.sum(query * query * inv_sigma2, axis=1, keepdims=True)
    cross = (query * inv_sigma2) @ data.T
    return logsumexp(-0.5 * (q_norm + d_norm - 2.0 * cross), axis=1)


def log_kde_fast_exact(query, data, sigma, batch_size=1024):
    """
    Fast, exact KDE.
    Numerically identical to naive implementation.
    """

    # ---- move to device once ----
    query = jnp.asarray(query, dtype=jnp.float32)
    data  = jnp.asarray(data,  dtype=jnp.float32)

    inv_sigma2 = jnp.asarray(1.0 / (sigma ** 2), dtype=jnp.float32)

    # ---- precompute constants ----
    d_norm = jnp.sum(data * data * inv_sigma2, axis=1)
    logN = jnp.log(data.shape[0])

    # ---- chunked evaluation ----
    outs = []
    for i in range(0, query.shape[0], batch_size):
        q = query[i:i + batch_size]
        out = _kde_block(q, data, d_norm, inv_sigma2)
        outs.append(out)

    return jnp.concatenate(outs) - logN
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


@jax.jit
def log_kde_gpu_maxspeed(query, data, sigma):
    """
    Fastest exact KDE possible on GPU.
    Requires enough VRAM for (Q × N) matrix.
    """

    query = jnp.asarray(query, dtype=jnp.float32)
    data  = jnp.asarray(data,  dtype=jnp.float32)

    inv_sigma2 = 1.0 / (sigma ** 2)

    # Precompute norms
    q_norm = jnp.sum(query * query * inv_sigma2, axis=1, keepdims=True)
    d_norm = jnp.sum(data  * data  * inv_sigma2, axis=1)

    # 🔥 Single massive GEMM
    cross = (query * inv_sigma2) @ data.T

    # Full KDE
    logp = logsumexp(-0.5 * (q_norm + d_norm - 2.0 * cross), axis=1)

    return logp - jnp.log(data.shape[0])


import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


@jax.jit
def log_kde_gpu_fast(query, data, inv_sigma, tau, block_size):
    Q = query.shape[0]
    N = data.shape[0]

    q_norm = jnp.sum(query * query * inv_sigma, axis=1, keepdims=True)
    logN = jnp.log(N)

    def body(i, state):
        acc = state

        d = jax.lax.dynamic_slice(
            data,
            (i * block_size, 0),
            (block_size, data.shape[1]),
        )

        d_norm = jnp.sum(d * d * inv_sigma, axis=1)

        cross = (query * inv_sigma) @ d.T

        contrib = logsumexp(
            -0.5 * (q_norm + d_norm - 2.0 * cross),
            axis=1,
        )

        acc = jnp.logaddexp(acc, contrib)

        return acc

    n_blocks = (N + block_size - 1) // block_size

    acc0 = jnp.full((Q,), -jnp.inf, dtype=jnp.float32)

    acc = jax.lax.fori_loop(0, n_blocks, body, acc0)

    return acc - logN

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np


@jax.jit
def _kde_block_fp16(query, data_block, inv_sigma2):
    # FP16 math, tensor cores
    cross = (query * inv_sigma2) @ data_block.T
    q_norm = jnp.sum(query * query * inv_sigma2, axis=1, keepdims=True)
    d_norm = jnp.sum(data_block * data_block * inv_sigma2, axis=1)

    return logsumexp(
        -0.5 * (q_norm + d_norm - 2.0 * cross),
        axis=1,
    )


@jax.jit
def _kde_scan(query, data, inv_sigma2, tau, block_size):
    Q = query.shape[0]
    N = data.shape[0]

    n_blocks = (N + block_size - 1) // block_size

    def body(i, state):
        acc = state

        d = jax.lax.dynamic_slice(
            data,
            (i * block_size, 0),
            (block_size, data.shape[1]),
        )

        contrib = _kde_block_fp16(query, d, inv_sigma2)
        acc = jnp.logaddexp(acc, contrib)

        return acc

    acc0 = jnp.full((Q,), -jnp.inf, dtype=jnp.float16)
    acc = jax.lax.fori_loop(0, n_blocks, body, acc0)

    return acc

@jax.jit
def kde_early_exit(query, data, inv_sigma2, tau, block_size):
    Q = query.shape[0]
    N = data.shape[0]

    acc = jnp.full((Q,), -jnp.inf, dtype=jnp.float16)
    q_norm = jnp.sum(query * query * inv_sigma2, axis=1, keepdims=True)

    n_blocks = (N + block_size - 1) // block_size

    def body(i, state):
        acc, done = state

        d = jax.lax.dynamic_slice(
            data,
            (i * block_size, 0),
            (block_size, data.shape[1]),
        )

        d_norm = jnp.sum(d * d * inv_sigma2, axis=1)
        cross = (query * inv_sigma2) @ d.T

        contrib = jax.scipy.special.logsumexp(
            -0.5 * (q_norm + d_norm - 2 * cross),
            axis=1
        )

        acc = jnp.logaddexp(acc, contrib)
        done = jnp.logical_or(done, acc >= tau)

        return acc, done

    acc0 = jnp.full((Q,), -jnp.inf, dtype=jnp.float16)
    done0 = jnp.zeros((Q,), dtype=bool)

    acc, _ = jax.lax.fori_loop(
        0, n_blocks, body, (acc0, done0)
    )

    return acc
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

import jax
import jax.numpy as jnp
import numpy as np

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

BLOCK = 4096  # fixed


@jax.jit
def fast_reject(query, data, inv_sigma2, tau):
    q_scaled = query * inv_sigma2
    q_norm = jnp.sum(query * q_scaled, axis=1, keepdims=True)

    # compute only max kernel value
    cross = q_scaled @ data.T
    d_norm = jnp.sum(data * data * inv_sigma2, axis=1)

    max_val = jnp.max(-0.5 * (q_norm + d_norm - 2 * cross), axis=1)

    return max_val + jnp.log(data.shape[0]) >= tau


def filter_gen_by_global_kde(gen, D, inv_sigma, tau):
    gen = jnp.asarray(gen, dtype=jnp.float16)
    D   = jnp.asarray(D,   dtype=jnp.float16)
    inv = jnp.asarray(inv_sigma, dtype=jnp.float16)

    # FAST REJECT
    keep_mask = fast_reject(gen, D, inv, tau)
    keep_idx = np.where(np.asarray(keep_mask))[0]

    if len(keep_idx) == 0:
        return keep_mask, ~keep_mask, None

    # EXACT KDE ONLY FOR SURVIVORS
    logp = kde_filter_exact(
        gen[keep_idx],
        D,
        inv,
        tau
    )

    final_mask = np.zeros(len(gen), dtype=bool)
    final_mask[keep_idx] = logp

    return final_mask, ~final_mask, None





def main():
    logger.info("Starting main function.")
    args: Namespace = parser.parse_args()
    logger.info(f"Arguments: {args}")
    device, num_workers = get_device_and_num_workers(args.device, args.num_workers)
    if len(args.path) < 3:
        logger.error(
            "At least three paths are required: train, test, and one or more generated."
        )
        return

    train_path = args.path[0]
    test_path = args.path[1]
    gen_paths = args.path[2:]
    logger.info(f"Training path: {train_path}")
    logger.info(f"Testing path: {test_path}")
    logger.info(f"Gen paths: {gen_paths}")

    model: DinoEncoder = get_model(args, device, args.dino_ckpt)

    if args.exp_dir:
        exp_dir = args.exp_dir
    else:
        exp_dir = create_unique_exp_dir()
    output_experiment_dir = os.path.join(args.output_dir, exp_dir)
    logger.info(f"Experiment directory: {output_experiment_dir}")
    write_arguments(args, output_experiment_dir)
    if args.load_npz:
        logger.info("Loading representations from NPZ files")

        train_representations = load_reps_from_npz(train_path)
        test_representations = load_reps_from_npz(test_path)
    else:

        train_representations = compute_representations(
            train_path, model, num_workers, device, args
        )
        logger.info("Finished loading/computing train representations")

        test_representations = compute_representations(
            test_path, model, num_workers, device, args
        )
        logger.info("Finished loading/computing test representations")
        logger.info(f"Enumerating paths to generated samples: {gen_paths}")

    for gen_path in gen_paths:

        if args.load_npz:
            gen_representations = load_reps_from_npz(gen_path)
        else:
            gen_representations = compute_representations(
                gen_path, model, num_workers, device, args
            )

        # ==============================
        # KDE FILTERING (timed)
        # ==============================
        t0 = time.perf_counter()

        D = np.vstack([train_representations, test_representations])
        sigma_D = np.std(D, axis=0).astype(np.float32)

        tau = float(args.tau)

        mask_keep, mask_low, _ = filter_gen_by_global_kde(
            gen_representations,
            D,
            sigma_D,
            tau,
        )

        gen_gt = gen_representations[mask_keep]

        kde_time = time.perf_counter() - t0

        # ==============================
        # PALATE (timed)
        # ==============================
        t1 = time.perf_counter()

        palate_components = compute_palate(
            train_representations=train_representations,
            test_representations=test_representations,
            gen_representations=gen_representations,
            gen_gt=gen_gt,
            sigma=args.sigma,
        )

        palate_time = time.perf_counter() - t1
        total_time = kde_time + palate_time
        palate_components["time_kde_sec"] = kde_time
        palate_components["time_palate_sec"] = palate_time
        palate_components["time_total_sec"] = total_time
        print(kde_time, "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")

        save_score(
            palate_components,
            output_experiment_dir,
            model,
            train_path,
            test_path,
            gen_path,
            args.nsample,
            args.sigma,
        )


if __name__ == "__main__":
    main()