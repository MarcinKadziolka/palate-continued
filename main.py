# This script contains modified parts of code from repository: https://github.com/layer6ai-labs/dgm-eval

import csv
import logging
import os
import pathlib
import uuid
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import Literal, Optional, Callable

import numpy as np
import torch
from jaxlib.xla_client import Array

from palate_local_knn import compute_local_palate_knn
from dataloader import CustomDataLoader, get_dataloader
from models.load_encoder import DinoEncoder, MODELS, load_encoder
from palate import compute_palate, PalateComponents, flatten_dataclass
from representations import get_representations

# =====================================================
# LOGGING
# =====================================================

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# =====================================================
# ARGUMENTS
# =====================================================

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--model", type=str, default="dinov2", choices=MODELS.keys())
parser.add_argument("--nsample", type=int, default=10000)
parser.add_argument("--sigma", type=float, default=0.01)
parser.add_argument("-bs", "--batch_size", type=int, default=50)
parser.add_argument("--num-workers", type=int)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("path", type=str, nargs="+")
parser.add_argument("--output_dir", type=str, default="./output")
parser.add_argument("--exp_dir", type=str, default=None)
parser.add_argument("--dino_ckpt", type=str, default=None)
parser.add_argument("--seed", type=int, default=13579)
parser.add_argument("--clean_resize", action="store_true")
parser.add_argument("--depth", type=int, default=0)
parser.add_argument("--repr_dir", type=str, default="./saved_representations")
parser.add_argument("--save", action="store_true")
parser.add_argument("--load", action="store_true")

# =====================================================
# UTILITIES
# =====================================================

def get_device_and_num_workers(device, num_workers):
    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)

    logger.info(f"Device: {device}, num_workers: {num_workers}")
    return device, num_workers


def get_last_x_dirs(path: str, x=2):
    parts = pathlib.Path(path).parts
    return "_".join(parts[-min(x, len(parts)):])

# =====================================================
# IO
# =====================================================

def write_to_txt(scores, output_dir, model, train_path, test_path, gen_path, nsample, sigma):
    path = os.path.join(output_dir, "metrics_summary.txt")
    with open(path, "a") as f:
        f.write(f"Model: {model.arch_str}\n")
        f.write(f"Train: {train_path}\nTest: {test_path}\nGen: {gen_path}\n")
        f.write(f"nsample: {nsample}\nsigma: {sigma}\n")
        for k, v in scores.items():
            f.write(f"{k}: {v}\n")
        f.write("\n" + "=" * 60 + "\n\n")


def write_to_csv(scores, output_dir, model, train_name, test_name, gen_name, nsample, sigma):
    path = os.path.join(output_dir, "metrics_summary.csv")
    exists = os.path.isfile(path)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                ["model", "train", "test", "gen", "nsample", "sigma"] + list(scores.keys())
            )
        writer.writerow(
            [model.arch_str, train_name, test_name, gen_name, nsample, sigma]
            + list(scores.values())
        )


def save_score(
    palate_components: PalateComponents,
    output_dir: str,
    model,
    train_path,
    test_path,
    gen_path,
    nsample,
    sigma,
    extra_scores: dict | None = None,
):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    scores = flatten_dataclass(palate_components)
    if extra_scores is not None:
        scores.update(extra_scores)

    write_to_txt(scores, output_dir, model, train_path, test_path, gen_path, nsample, sigma)
    write_to_csv(
        scores,
        output_dir,
        model,
        get_last_x_dirs(train_path),
        get_last_x_dirs(test_path),
        get_last_x_dirs(gen_path),
        nsample,
        sigma,
    )

# =====================================================
# REPRESENTATIONS
# =====================================================

def get_model(args: Namespace, device: torch.device, ckpt: str) -> DinoEncoder:
    return load_encoder(
        args.model,
        device,
        ckpt=ckpt,
        clean_resize=args.clean_resize,
        depth=args.depth,
    )


def compute_representations(
    path: str, model: DinoEncoder, num_workers: int, device, args: Namespace
) -> np.ndarray:

    if args.load:
        loaded = load_reps_from_path(args.repr_dir, path, model, args.nsample)
        if loaded is not None:
            return loaded

    dataloader = get_dataloader(
        path,
        args.nsample,
        args.batch_size,
        num_workers,
        seed=args.seed,
        transform=lambda x: model.transform(x),
    )

    reps = get_representations(model, dataloader, device, normalized=False)

    if args.save:
        save_representations(args.repr_dir, path, reps, model, dataloader, args.nsample)

    return reps


def save_representations(output_dir, path, reps, model, dataloader, nsample):
    out_path = get_path(output_dir, path, model, nsample)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.savez(out_path, reps=reps)


def load_reps_from_path(saved_dir, path, model, nsample) -> Optional[np.ndarray]:
    load_path = get_path(saved_dir, path, model, nsample)
    if os.path.exists(load_path):
        return np.load(load_path)["reps"]
    return None


def get_path(output_dir, path, model, nsample):
    return os.path.join(
        output_dir,
        f"{model.arch_str}_{get_last_x_dirs(path)}_{nsample}.npz",
    )

# =====================================================
# MAIN
# =====================================================

def main():
    args = parser.parse_args()
    device, num_workers = get_device_and_num_workers(args.device, args.num_workers)

    train_path, test_path, *gen_paths = args.path
    model = get_model(args, device, args.dino_ckpt)

    exp_dir = args.exp_dir or str(uuid.uuid4())[:8]
    output_dir = os.path.join(args.output_dir, exp_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_repr = compute_representations(train_path, model, num_workers, device, args)
    test_repr = compute_representations(test_path, model, num_workers, device, args)

    for gen_path in gen_paths:
        gen_repr = compute_representations(gen_path, model, num_workers, device, args)

        palate_components = compute_palate(
            train_representations=train_repr,
            test_representations=test_repr,
            gen_representations=gen_repr,
            sigma=args.sigma,
        )

        local_scores = compute_local_palate_knn(
            train_repr, test_repr, gen_repr, k=50, sigma=None
        )

        local_summary = {
            "local_palate_mean": float(local_scores.mean()),
            "local_palate_median": float(np.median(local_scores)),
            "local_palate_std": float(local_scores.std()),
            "local_palate_frac_gt_0.5": float((local_scores > 0.5).mean()),
        }

        save_score(
            palate_components,
            output_dir,
            model,
            train_path,
            test_path,
            gen_path,
            args.nsample,
            args.sigma,
            extra_scores=local_summary,
        )

        np.save(
            os.path.join(output_dir, f"local_palate_{get_last_x_dirs(gen_path)}.npy"),
            local_scores,
        )


if __name__ == "__main__":
    main()
