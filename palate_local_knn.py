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

    return r_values, sigma

import numpy as np
from tqdm import tqdm

def compute_global_palate_batched(train, test, gen, sigma, batch_size=500):

    train = train.astype(np.float32)
    test  = test.astype(np.float32)
    gen   = gen.astype(np.float32)

    N = len(gen)
    r_values = np.zeros(N, dtype=np.float32)

    for i in tqdm(range(0, N, batch_size), desc="Global PALATE batched"):
        batch = gen[i:i+batch_size]      # [B, D]

        # squared distances in bulk using matrix operations
        d_tr = np.sum((train[None,:,:] - batch[:,None,:])**2, axis=2)  # [B, N_train]
        d_te = np.sum((test[None,:,:]  - batch[:,None,:])**2, axis=2)  # [B, N_test]

        p_tr = np.exp(-d_tr / (2*sigma**2)).mean(axis=1)  # density estimate per sample
        p_te = np.exp(-d_te / (2*sigma**2)).mean(axis=1)

        r_values[i:i+batch_size] = p_tr / (p_tr + p_te + 1e-8)

    return r_values

import numpy as np
from tqdm import tqdm

def estimate_sigma(train, samples=10000):
    """
    Sigma = sqrt(median(squared_distances) / 2)
    """

    idx = np.random.choice(len(train), min(samples, len(train)), replace=False)
    sub = train[idx]

    # squared distances from cosine similarity
    sim = sub @ sub.T                         # cosine similarity matrix
    dist2 = 2 - 2 * sim                       # squared L2 distances

    dist2 = dist2[dist2 > 1e-8]               # remove diagonal self-distances

    median_dist2 = np.median(dist2)           # median of squared distances
    sigma = np.sqrt(median_dist2 / 2)         # ← correct formula

    return sigma



def compute_global_palate_fast(train, test, gen, sigma=None, batch_size=500):

    # ---- normalize first ----
    train = train.astype(np.float32)
    test  = test.astype(np.float32)
    gen   = gen.astype(np.float32)

    train /= np.linalg.norm(train, axis=1, keepdims=True) + 1e-8
    test  /= np.linalg.norm(test, axis=1, keepdims=True) + 1e-8
    gen   /= np.linalg.norm(gen, axis=1, keepdims=True) + 1e-8

    # ---- auto sigma ----
    if sigma is None:
        print("Estimating sigma...")
        sigma = estimate_sigma(train)
        print(f"Sigma = {sigma:.4f}")

    N = len(gen)
    r_values = np.zeros(N, dtype=np.float32)

    # ---- main loop ----
    for i in tqdm(range(0, N, batch_size), desc="Global PALATE FAST"):
        batch = gen[i:i+batch_size]                # (B, D)

        sim_tr = batch @ train.T                   # (B, N_train)
        sim_te = batch @ test.T                    # (B, N_test)

        d_tr = 2 - 2*sim_tr
        d_te = 2 - 2*sim_te

        p_tr = np.exp(-d_tr / (2*sigma**2)).mean(axis=1)
        p_te = np.exp(-d_te / (2*sigma**2)).mean(axis=1)

        r_values[i:i+batch_size] = p_tr / (p_tr + p_te + 1e-8)

    return r_values, sigma

'''
def compute_global_palate_fast_unnormalized(train, test, gen, sigma=1, batch_size=200):

    train = train.astype(np.float32)
    test  = test.astype(np.float32)
    gen   = gen.astype(np.float32)

    if sigma is None:
        print("Estimating sigma...")
        sigma = estimate_sigma(train)   # works even without norm
        print(f"Sigma = {sigma:.4f}")

    N = len(gen)
    r_values = np.zeros(N, dtype=np.float32)

    for i in tqdm(range(0, N, batch_size), desc="Global PALATE (no norm)"):

        batch = gen[i:i+batch_size]                  # (B, D)

        # true squared Euclidean distance
        d_tr = np.sum((batch[:,None,:] - train[None,:,:])**2, axis=2)
        d_te = np.sum((batch[:,None,:]  - test[None,:,:])**2, axis=2)

        p_tr = np.exp(-d_tr / (2*sigma**2)).mean(axis=1)
        p_te = np.exp(-d_te / (2*sigma**2)).mean(axis=1)

        r_values[i:i+batch_size] = p_tr / (p_tr + p_te + 1e-8)

    return r_values, sigma
'''

def compute_global_palate_fast_normalized(train, test, gen, sigma=1, batch_size=200):

    train = train.astype(np.float32)
    test  = test.astype(np.float32)
    gen   = gen.astype(np.float32)

    #train /= np.linalg.norm(train, axis=1, keepdims=True)
    #test /= np.linalg.norm(test, axis=1, keepdims=True)
    #gen /= np.linalg.norm(gen, axis=1, keepdims=True)

    if sigma is None:
        sigma = estimate_sigma(train)

    train_norm = np.sum(train**2, axis=1)
    test_norm  = np.sum(test**2, axis=1)

    N = len(gen)
    r_values = np.zeros(N, dtype=np.float32)
    p_trs = np.zeros(N, dtype=np.float32)
    p_tes = np.zeros(N, dtype=np.float32)


    for i in tqdm(range(0, N, batch_size), desc="Global PALATE (no norm)"):
        batch = gen[i:i+batch_size]
        batch_norm = np.sum(batch**2, axis=1)[:, None]

        d_tr = batch_norm + train_norm[None, :] - 2.0 * batch @ train.T
        d_te = batch_norm + test_norm[None, :]  - 2.0 * batch @ test.T

        p_tr = np.exp(-d_tr / (2*sigma**2)).mean(axis=1)
        p_te = np.exp(-d_te / (2*sigma**2)).mean(axis=1)

        r_values[i:i+batch_size] = p_tr / (p_tr + p_te)
        p_trs[i:i + batch_size] = p_tr
        p_tes[i:i + batch_size] = p_te

    return p_trs, p_tes, r_values, sigma
