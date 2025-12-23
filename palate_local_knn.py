import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

EPS = 1e-8


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def gaussian_kernel(dist2: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-dist2 / (2.0 * sigma ** 2))


def compute_local_palate_knn(
    train_representations: np.ndarray,
    test_representations: np.ndarray,
    gen_representations: np.ndarray,
    k: int = 50,
    sigma: float | None = None,
    estimate_sigma_samples: int = 1000,
) -> np.ndarray:
    """
    Compute per-sample local PALATE scores using kNN KDE.
    """

    # ---- Normalize (local only) ----
    train = l2_normalize(train_representations.astype(np.float32))
    test = l2_normalize(test_representations.astype(np.float32))
    gen = l2_normalize(gen_representations.astype(np.float32))

    # ---- Build kNN indices ----
    nn_train = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nn_test = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)

    nn_train.fit(train)
    nn_test.fit(test)

    # ---- Estimate sigma if needed ----
    if sigma is None:
        dists, _ = nn_train.kneighbors(
            train[:estimate_sigma_samples], n_neighbors=k
        )
        sigma = np.median(dists[:, -1])

    # ---- Local PALATE ----
    r_values = np.zeros(len(gen), dtype=np.float32)

    for i, y in enumerate(tqdm(gen, desc="Local PALATE (kNN)")):
        d_tr, _ = nn_train.kneighbors(y[None], n_neighbors=k)
        p_tr = gaussian_kernel(d_tr[0] ** 2, sigma).mean()

        d_te, _ = nn_test.kneighbors(y[None], n_neighbors=k)
        p_te = gaussian_kernel(d_te[0] ** 2, sigma).mean()

        r_values[i] = p_tr / (p_tr + p_te + EPS)

    return r_values
