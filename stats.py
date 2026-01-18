import os
import argparse
import numpy as np
from scipy.special import logsumexp


# ============================================================
# KDE
# ============================================================

def log_kde_anisotropic_batched(query, data, sigma, batch_size=512):
    inv_sigma2 = 1.0 / (sigma ** 2)
    data_norm = np.sum(data ** 2 * inv_sigma2, axis=1)

    out = np.empty(len(query), dtype=np.float32)

    for i in range(0, len(query), batch_size):
        q = query[i:i + batch_size]
        q_norm = np.sum(q ** 2 * inv_sigma2, axis=1)[:, None]
        cross = (q * inv_sigma2) @ data.T
        d = q_norm + data_norm[None, :] - 2.0 * cross
        out[i:i + batch_size] = logsumexp(-0.5 * d, axis=1) - np.log(len(data))

    return out


# ============================================================
# Pairwise distance (median of squared distances)
# ============================================================

def median_pairwise_sqdist(X, block_size=512):
    n = X.shape[0]
    dists = []

    for i in range(0, n, block_size):
        Xi = X[i:i + block_size]
        Xi_norm = np.sum(Xi**2, axis=1)[:, None]

        for j in range(i, n, block_size):
            Xj = X[j:j + block_size]
            Xj_norm = np.sum(Xj**2, axis=1)[None, :]

            sq = Xi_norm + Xj_norm - 2 * Xi @ Xj.T

            if i == j:
                sq = sq[np.triu_indices_from(sq, k=1)]

            dists.append(sq.ravel())

    return float(np.median(np.concatenate(dists)))


# ============================================================
# Main computation
# ============================================================

def compute_stats(train, test, tau):
    D = np.vstack([train, test])
    sigma_D = np.std(D, axis=0)

    # KDE likelihoods
    logp_train = log_kde_anisotropic_batched(train, D, sigma_D)
    logp_test  = log_kde_anisotropic_batched(test,  D, sigma_D)

    stats = {
        # KDE means
        "logp_train_mean": float(np.mean(logp_train)),
        "logp_test_mean":  float(np.mean(logp_test)),

        # KDE medians
        "logp_train_median": float(np.median(logp_train)),
        "logp_test_median":  float(np.median(logp_test)),

        # KDE minima (WHAT YOU ASKED FOR)
        "logp_train_min": float(np.min(logp_train)),
        "logp_test_min":  float(np.min(logp_test)),

        # Fractions below tau
        "train_frac_below_tau": float((logp_train < tau).mean()),
        "test_frac_below_tau":  float((logp_test  < tau).mean()),

        # Pairwise distances
        "median_sqdist_train": median_pairwise_sqdist(train),
        "median_sqdist_test":  median_pairwise_sqdist(test),
        "median_sqdist_all":   median_pairwise_sqdist(D),

        # Meta
        "tau": float(tau),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }

    return stats


# ============================================================
# I/O
# ============================================================

def load_npz(path):
    data = np.load(path)
    if "reps" not in data:
        raise ValueError(f"{path} does not contain 'reps'")
    return data["reps"]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=str)
    parser.add_argument("test", type=str)
    parser.add_argument("--tau", type=float, default=-9.0)
    parser.add_argument("--out", type=str, default="kde_stats.txt")
    args = parser.parse_args()

    train = load_npz(args.train)
    test  = load_npz(args.test)

    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")

    stats = compute_stats(train, test, args.tau)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

    print(f"Saved stats to {args.out}")


if __name__ == "__main__":
    main()
