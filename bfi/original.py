"""Reference (baseline) implementation of BFI as described in Zhang et al., IEEE TFS 33(8), 2025.

This version follows the paper literally:
  * full N x N distance matrix, every row sorted (Sec. III-A: "sort them from small to large")
  * Bel_i(C) over the K1 nearest neighbours (Eq. 10), theta_i from the 0.9-quantile of ALL distances (Eq. 9)
  * delta_i = distance to the nearest object with larger Bel (Eq. 11), rho_f = exp(delta) (Eq. 12)
  * evidential convergence in decreasing-Bel order with K2 nearest *allocated* neighbours (Eqs. 13-15)
Complexity O(N^2 log N) time, O(N^2) memory -> this is the cost we want to reduce.
"""
import time
import numpy as np
from scipy.spatial.distance import cdist
from .evidence import fuse_neighbours
from .result import BFIResult
from .validation import check_inputs


def select_centers(delta, bel, c, rule="bel_delta"):
    """Pick the c objects with the largest gamma on the decision graph.

    rule="paper":     gamma = rho_f * delta = exp(delta) * delta   (Eq. 12; monotone in delta, so an
                      isolated outlier with a large delta is selected as a 1-object "cluster")
    rule="bel_delta": gamma = Bel * delta  (DEFAULT here; Bel suppresses low-belief outliers).
                      This is a documented deviation from the paper, see README.
    """
    if rule == "paper":
        with np.errstate(divide="ignore"):
            score = delta + np.log(delta)
    else:
        score = bel * delta
    return np.argsort(-score, kind="stable")[:c]


def bfi_original(X, n_clusters, K1, K2=9, zeta=0.1, q=0.9, center_rule="bel_delta"):
    """Baseline BFI.  The algorithm is unchanged; it additionally records which neighbours it used
    (result.diag) so that the accelerated variants can be checked against it."""
    if center_rule not in ("bel_delta", "paper"):
        raise ValueError(f"center_rule must be 'bel_delta' or 'paper', got {center_rule!r}")
    X, c, K1, K2 = check_inputs(X, n_clusters, K1, K2, zeta, q)
    N = len(X)
    info = {}
    t0 = time.perf_counter()

    D = cdist(X, X)                                   # O(N^2) memory
    Ds = np.sort(D, axis=1)                           # column 0 is the object itself
    info["t_dist_sort"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    theta = 1.0 / np.maximum(np.quantile(Ds[:, 1:], q, axis=1), 1e-12)          # Eq. (9)
    dk = Ds[:, 1:K1 + 1]
    log_comp = np.log1p(-np.exp(-theta[:, None] * dk) / K1).sum(1)              # Eq. (10)
    bel = 1.0 - np.exp(log_comp)
    del Ds
    order = np.argsort(-bel, kind="stable")           # decreasing Bel (ties broken by index)
    delta = np.empty(N)
    delta_nn = -np.ones(N, dtype=np.int64)
    delta[order[0]] = D[order[0]].max()               # Eq. (11), highest-belief object
    for r in range(1, N):
        i = order[r]
        j = D[i, order[:r]].argmin()
        delta[i] = D[i, order[j]]
        delta_nn[i] = order[j]
    centers = select_centers(delta, bel, c, center_rule)
    info["t_bel_delta"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    lab = -np.ones(N, dtype=int)
    M = np.zeros((N, c))
    conv_nn = -np.ones((N, K2), dtype=np.int64)
    for k, ci in enumerate(centers):
        lab[ci] = k
        M[ci, k] = 1.0
    for t in order:
        if lab[t] >= 0:
            continue
        alloc = np.flatnonzero(lab >= 0)
        k = min(K2, len(alloc))
        dd = D[t, alloc]
        nn = np.arange(len(alloc)) if k == len(alloc) else np.argpartition(dd, k - 1)[:k]
        conv_nn[t, :len(nn)] = alloc[nn]
        mv = fuse_neighbours(lab[alloc[nn]], dd[nn], theta[t], zeta, c)
        M[t] = mv
        lab[t] = mv.argmax()
    info["t_convergence"] = time.perf_counter() - t2
    info["t_total"] = time.perf_counter() - t0
    diag = {"delta_nn": delta_nn, "conv_nn": conv_nn, "bel_exact": np.ones(N, bool)}
    return BFIResult(M=M, bel=bel, delta=delta, centers=centers, theta=theta, info=info, diag=diag)
