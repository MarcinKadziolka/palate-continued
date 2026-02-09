import numpy as np
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    print("Loading representations...")
    X = np.load(args.path)["reps"].astype(np.float32)

    n, d = X.shape
    print(n, d)

    print("Computing median squared distance...")
    t0 = time.time()

    norms = np.sum(X**2, axis=1)
    num_pairs = n * (n - 1) // 2
    dists = np.empty(num_pairs, dtype=np.float32)

    k = 0
    for i in range(n):
        dots = X[i] @ X[i + 1 :].T
        d = norms[i] + norms[i + 1 :] - 2 * dots
        dists[k : k + len(d)] = d
        k += len(d)

    median_dist = np.median(dists)

    print(f"Median squared distance: {median_dist:.6f}")
    print(f"Time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
