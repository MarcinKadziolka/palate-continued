# PALATE: Peculiar Application of the Law of Total Expectation to Enhance the Evaluation of Deep Generative Models
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