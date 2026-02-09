# PALATE: Peculiar Application of the Law of Total Expectation to Enhance the Evaluation of Deep Generative Models

Abstract: *Deep generative models (DGMs) have caused a paradigm shift in the field of machine learning, yielding noteworthy advancements in domains such as image synthesis, natural language processing, and other related areas. However, a comprehensive evaluation of these models that effectively captures the interplay among fidelity, diversity, and novelty in generated samples remains a persistent challenge. A recently proposed approach, the Feature Likelihood Divergence (FLD), offers a theoretically grounded and practical tool in this regard but faces notable computational limitations. In this paper, we propose PALATE, a novel framework designed to enhance the evaluation of DGMs by addressing the efficiency constraints of FLD. Our approach is based on a peculiar application of the law of total expectation to random variables representing accessible data. When integrated with the MMD baseline metric and the recently introduced DINOv3 feature extractor, PALATE provides a comprehensive evaluation framework that matches or exceeds state-of-the-art methods while offering superior computational efficiency and scalability to large-scale datasets. Through a series of experiments, we demonstrate that the PALATE enhancement constitutes a practical and computationally efficient framework for holistic DGM assessment, particularly in detecting sample memorization and evaluating generalization capabilities.*

## Instructions to Run

### 1. Setting things up

#### a. Install conda

Install one of Anaconda Distributions (for example [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)).

#### b. Create conda environment:

```bash
conda env create -f environment.yml
```

#### c. Activate environment:

```bash
conda activate palate
```

### 2. Example call

```bash
python3 main.py ./path/to/train ./path/to/test ./path/to/gen_1 ./path/to/gen_2 --batch_size 256 --nsample 1000 --save --load
```

- The **first path** should point to the **training data**.

- The **second path** should point to the **test data**.

- Subsequent paths should point to folders containing **generated samples** (one folder per model).

Each run generates a unique folder in the specified output directory. The folder contains metrics summary in `.txt` and `.csv` format.

### Detailed Information
```text
usage: main.py [-h] [--model {dinov2,dinov3}] [--nsample NSAMPLE] [--sigma SIGMA] [-bs BATCH_SIZE] [--num-workers NUM_WORKERS] [--device DEVICE] [--output_dir OUTPUT_DIR] [--tau TAU]
               [--exp_dir EXP_DIR] [--dino_ckpt DINO_CKPT] [--seed SEED] [--clean_resize] [--load_npz] [--depth DEPTH] [--kde_percentile KDE_PERCENTILE [KDE_PERCENTILE ...]] [--repr_dir REPR_DIR]
               [--save] [--load]
               path [path ...]

positional arguments:
  path                  Paths to image datasets in order: train test gen_1 gen_2 ... gen_n. At least 3 paths are required.

options:
  -h, --help            show this help message and exit
  --model {dinov2,dinov3}
                        Encoder model used to generate representations. (default: dinov2)
  --nsample NSAMPLE     Maximum number of images used per dataset. (default: 10000)
  --sigma SIGMA         Sigma to use in blockwise_kernel_mean in dmmd.py (default: 10.5)
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size to use. If needed, equals to min(batch_size, nsample). (default: 50)
  --num-workers NUM_WORKERS
                        Number of processes to use for data loading. Defaults to `min(8, num_cpus)` (default: None)
  --device DEVICE       Device to use. Like cuda, cuda:0 or cpu (default: None)
  --output_dir OUTPUT_DIR
                        Root directory for all experiment outputs. (default: ./output)
  --tau TAU             Explicit global KDE log-density threshold. Overrides --kde_percentile. (default: -300.0)
  --exp_dir EXP_DIR     Name of the experiment directory. If not provided, a unique ID is generated. Parent is --output_dir (default: None)
  --dino_ckpt DINO_CKPT
                        Path to dinov3 weights (used only if --model dinov3). (default: None)
  --seed SEED           Random seed (default: 13579)
  --clean_resize        Use clean resizing (from pillow) (default: False)
  --load_npz            Run in image-free mode. Paths must be .npz files containing representations. (default: False)
  --depth DEPTH         Negative depth for internal layers, positive 1 for after projection head. (default: 0)
  --kde_percentile KDE_PERCENTILE [KDE_PERCENTILE ...]
                        List of percentiles for global KDE threshold grid search (in %). (default: 5.0)
  --repr_dir REPR_DIR   Directory for cached representations. (default: ./saved_representations)
  --save                Save computed representations to repr_dir. (default: False)
  --load                Load representations from repr_dir instead of recomputing. (default: False)
```