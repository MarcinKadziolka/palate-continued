# main.py — palate runner

---

## Command-line Arguments

### Positional arguments

| Argument | Type                | Description                                                                                                  |
| -------- | ------------------- | ------------------------------------------------------------------------------------------------------------ |
| `path`   | `str` (one or more) | Paths to image datasets **in order**: `train test gen_1 gen_2 ... gen_n`. At least **3 paths** are required. |

---

### Optional arguments

#### Model & representations

| Argument      | Default                                     | Description                                                                             |
| ------------- | ------------------------------------------- | --------------------------------------------------------------------------------------- |
| `--model`     | `dinov2`                                    | Encoder model used to generate representations. Choices are taken from `MODELS.keys()`. |
| `--dino_ckpt` | path to dinov3 pth checkpoint on the server | Path to DINOv3 weights (used only if `--model dinov3`).                                 |
| `--sigma`     | `0.01`                                      | Kernel bandwidth used in palate / dmmd computation.                                     |

---

#### Sampling & performance

| Argument              | Default            | Description                                                            |
| --------------------- | ------------------ | ---------------------------------------------------------------------- |
| `--nsample`           | `10000`            | Maximum number of images used per dataset.                             |
| `--batch_size`, `-bs` | `50`               | Batch size for representation extraction.                              |
| `--num-workers`       | `min(8, num_cpus)` | Number of workers for data loading.                                    |
| `--device`            | auto               | Device to use (`cuda`, `cuda:0`, `cpu`). Auto-selects if not provided. |
| `--seed`              | `13579`            | Random seed for sampling.                                              |

---

#### Representation caching

| Argument     | Default                   | Description                                                  |
| ------------ | ------------------------- | ------------------------------------------------------------ |
| `--repr_dir` | `./saved_representations` | Directory for cached representations.                        |
| `--save`     | `False`                   | Save computed representations to `repr_dir`.                 |
| `--load`     | `False`                   | Load representations from `repr_dir` instead of recomputing. |

---

#### Experiment output

| Argument       | Default          | Description                                                                  |
| -------------- | ---------------- | ---------------------------------------------------------------------------- |
| `--output_dir` | `./output`       | Root directory for all experiment outputs.                                   |
| `--exp_dir`    | random 8-char ID | Name of the experiment directory. If not provided, a unique ID is generated. |

---

## Example Usage

### Example (single generated dataset)

```bash
python main.py \
  "/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10/train" \
  "/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10/test" \
  "/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-MHGAN" \
  --save \
  --load
```

---

Output Structure

### 1. Experiment directory layout

```
output/
└── <--exp_dir>/
    ├── metrics_summary.csv
    ├── metrics_summary.txt
    └── arguments.txt
```

If `--exp_dir` is **not** provided, it is an auto-generated 8-character ID (UUID or cluster job ID).

---

### 2. `metrics_summary.csv`

One row **per generated dataset**.

Each run will **append** to the existing file, meaning that for one sbatch
the folder can be _static_.

Example:

```
Train,Test,Gen,Nsample,m_palate,palate,...
CIFAR10_train,CIFAR10_test,CIFAR10_CIFAR10-ACGAN-Mod,10000,0.74487555,0.4919243,...
CIFAR10_train,CIFAR10_test,CIFAR10_CIFAR10-MHGAN,10000,0.7304801,0.49030876,...
```

- `Train`, `Test`, `Gen` are derived from the **last two path components**
- Metric columns correspond to fields of `PalateComponents` in the exact order.

---

### 3. `metrics_summary.txt`

Human-readable log of results, appended per generated dataset:

```
Model: dinov2_vitl14
Train: /home/mubuntu/datasets/CIFAR10/CIFAR10/train
Test: /home/mubuntu/datasets/CIFAR10/CIFAR10/test
Gen: /home/mubuntu/datasets/CIFAR10/CIFAR10-ACGAN-Mod
nsample: 10000
m_palate: 0.7448755502700806
palate: 0.4919242858886719
train_gen: 0.0002973441150970757
test_gen: 0.00028789174393750727
test_train: 0.000209193371119909
denominator_scale: 0.0002885187277570367
sigma: 10.0
m_palate_formula: palate/2 + dmmd_test/(2*denominator_scale)
palate_formula: dmmd_test/(dmmd_test + dmmd_train)
m_palate_formula_hash: 3266612f4115
palate_formula_hash: c656cc8e53a0

==================================================
```

---

### 4. `arguments.txt`

Exact arguments used for reproducibility:

```
model: dinov2
nsample: 10000
sigma: 0.01
batch_size: 50
num_workers: None
device: None
path: ['/home/mubuntu/datasets/CIFAR10/CIFAR10/train', '/home/mubuntu/datasets/CIFAR10/CIFAR10/test', '/home/mubuntu/datasets/CIFAR10/CIFAR10-ACGAN-Mod']
output_dir: fresh
exp_dir: test
dino_ckpt: /shared/results/gmdziarm/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
seed: 13579
clean_resize: False
depth: 0
repr_dir: new
save: True
load: True

==================================================
```

---

### 5. Saved representations (optional)

If `--save` is enabled:

```
<--repr_dir>/
└── dinov2_vitl14_CIFAR10_train_10000.npz
└── dinov2_vitl14_CIFAR10_test_10000.npz
└── dinov2_vitl14_CIFAR10_CIFAR10-ACGAN-Mod_10000.npz. 
```
