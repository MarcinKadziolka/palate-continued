# This script contains modified parts of code from repository: https://github.com/layer6ai-labs/dgm-eval

import csv
import dataclasses
import logging
import os
import pathlib
import uuid
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import Literal, Optional, Callable

from jaxlib.xla_client import Array
import numpy as np
import torch
from palate_local_knn import compute_local_palate_knn


from dataloader import CustomDataLoader
from dataloader import get_dataloader
from models.load_encoder import DinoEncoder
from models.load_encoder import MODELS, load_encoder
from palate import compute_palate, PalateComponents, flatten_dataclass
from representations import get_representations

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
    "--depth",
    type=int,
    default=0,
    help="Negative depth for internal layers, positive 1 for after projection head.",
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
    out_file = "metrics_summary.txt"
    out_path = os.path.join(output_dir, out_file)

    with open(out_path, "a") as f:
        f.write(f"Model: {model.arch_str}\n")
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

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["model_arch", "train", "test", "ten", "nsample", "sigma"] + list(scores.keys())
            writer.writerow(header)
        row = [model.arch_str, train_name, test_name, gen_name, nsample, sigma] + list(scores.values())
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
        extra_scores,
):
    train_name = get_last_x_dirs(train_path)
    test_name = get_last_x_dirs(test_path)
    gen_name = get_last_x_dirs(gen_path)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    scores = flatten_dataclass(palate_components)

    if extra_scores is not None:
        scores.update(extra_scores)

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


def get_path(output_dir: str, path: str, model: DinoEncoder, nsample: int) -> str:
    """Generate a unique file path for saving representations"""

    dataset_name = get_last_x_dirs(path)

    return os.path.join(output_dir, f"{model.arch_str}_{dataset_name}_{nsample}.npz")


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

        gen_representations = compute_representations(
            gen_path, model, num_workers, device, args
        )

        palate_components: PalateComponents = compute_palate(
            train_representations=train_representations,
            test_representations=test_representations,
            gen_representations=gen_representations,
            sigma=args.sigma,
        )

        local_scores = compute_local_palate_knn(
            train_representations, test_representations, gen_representations, k=50, sigma=None
        )

        local_summary = {
            "local_palate_mean": float(local_scores.mean()),
            "local_palate_median": float(np.median(local_scores)),
            "local_palate_std": float(local_scores.std()),
            "local_palate_frac_gt_0.5": float((local_scores > 0.5).mean()),
        }

        save_score(
            palate_components,
            output_experiment_dir,
            model,
            train_path,
            test_path,
            gen_path,
            args.nsample,
            args.sigma,
            extra_scores=local_summary,  # ← plugs straight in
        )


if __name__ == "__main__":
    main()
