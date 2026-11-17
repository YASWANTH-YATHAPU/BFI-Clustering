"""Solution 2 - Exact Distance-Based Refinement.

Solution 1 answers neighbour queries from candidate lists, but two kinds of quantities would
otherwise rest on *estimates*:
  (a) Bel_i (Eq. 10) needs the K1 nearest distances; a candidate list of k_cand < K1 entries
      covers only the head, the tail (ranks k_cand+1..K1) would be extrapolated by a model;
  (b) objects whose candidate budget is exhausted without a certificate (Solution 1 returned
      "unresolved") would get an estimated delta / an incomplete set of convergence neighbours.

This module replaces those estimates by exact computations, organised so that exact distances
are computed only where they are needed:
  * exact_bel_streamed  - exact K1-NN distance profile for a set of objects, streamed through the
                          spatial index in memory-bounded chunks (O(chunk*K1) memory instead of the
                          O(N^2) matrix).  The same pass also returns the first `keep` neighbours of
                          every object, so the Solution-1 base lists come for free.
  * exact_nearest_higher - exact delta for the few objects left unresolved by Solution 1, by a scan
                          restricted to the higher-Bel subset.
  * exact_k_nearest      - exact K2 nearest allocated objects for objects left unresolved by
                          Solution 1, by a scan restricted to the allocated subset.
  * exact_theta          - exact theta_i of Eq. (9) by chunked partial selection (no N x N matrix).

Note.  Computing exact Euclidean distances is not new - the original BFI computes all of them.
What is specific to this design is *which* distances are computed: the refinement is triggered
only by a failed sufficiency certificate (b) or by the Bel policy (a), and it reuses the index /
restricted subsets instead of the full matrix.
"""
import numpy as np
from scipy.spatial.distance import cdist

_ENTRY_BUDGET = 2 ** 19      # max neighbour entries per streamed chunk (~4 MB float64 + 4 MB int64)


def bel_terms(dist, theta, K1):
    """Sum over the given neighbour distances of log(1 - exp(-theta d)/K1)  (log of Eq. 10 product)."""
    return np.log1p(-np.exp(-theta[:, None] * dist) / K1).sum(1)


def exact_bel_streamed(searcher, theta, K1, keep, rows=None):
    """Exact log-product of Eq. (10) for `rows` (default: all objects) from exact K1-NN queries.

    Returns (logprod, keep_idx, keep_dist) where keep_idx/keep_dist hold the first `keep`
    neighbours of each row (sorted).  Memory is bounded by _ENTRY_BUDGET entries per chunk."""
    N = searcher.N
    rows = np.arange(N) if rows is None else np.asarray(rows)
    kq = int(min(max(K1, keep), N - 1))
    keep = int(min(keep, kq))
    logprod = np.empty(len(rows))
    kidx = np.empty((len(rows), keep), dtype=np.int64)
    kdist = np.empty((len(rows), keep))
    step = max(1, _ENTRY_BUDGET // kq)
    for s in range(0, len(rows), step):
        r = rows[s:s + step]
        idx, dist = searcher.query(r, kq)
        logprod[s:s + step] = bel_terms(dist[:, :K1], theta[r], K1)
        kidx[s:s + step], kdist[s:s + step] = idx[:, :keep], dist[:, :keep]
    return logprod, kidx, kdist


def exact_theta(X, q, chunk_entries=2 ** 19):
    """Exact theta_i = 1 / quantile_q({d_ij : j != i}) (Eq. 9) without materialising N x N.

    Equals np.quantile(sorted_row[1:], q) (linear interpolation) as used in bfi_original: the
    object's own zero distance is the smallest entry, so the quantile of the other N-1 distances
    is found at positions 1+lo and 2+lo of the full sorted row, lo = floor(q (N-2))."""
    N = len(X)
    pos = q * (N - 2)
    lo = int(np.floor(pos))
    frac = pos - lo
    kth = [1 + lo] if 2 + lo > N - 1 else [1 + lo, 2 + lo]
    out = np.empty(N)
    step = max(1, chunk_entries // N)
    for s in range(0, N, step):
        Dc = cdist(X[s:s + step], X)
        P = np.partition(Dc, kth, axis=1)
        a = P[:, kth[0]]
        b = P[:, kth[-1]]
        # same lerp as numpy's 'linear' quantile (numerically symmetric form for frac >= 0.5)
        out[s:s + step] = a + (b - a) * frac if frac < 0.5 else b - (b - a) * (1.0 - frac)
    return 1.0 / np.maximum(out, 1e-12)


def sampled_theta(X, q, theta_sample, rng, chunk_entries=2 ** 19):
    """Approximate theta: quantile of the distances to a random reference sample of size s."""
    N = len(X)
    ref = X[rng.choice(N, size=theta_sample, replace=False)]
    out = np.empty(N)
    step = max(1, chunk_entries // theta_sample)
    for s in range(0, N, step):
        out[s:s + step] = np.quantile(cdist(X[s:s + step], ref), q, axis=1)
    return 1.0 / np.maximum(out, 1e-12)


def exact_nearest_higher(X, u, order, rank):
    """Exact delta for object u: nearest object among the rank[u] objects with larger Bel."""
    if rank[u] == 0:
        return -1, float(np.sqrt(((X - X[u]) ** 2).sum(1)).max())
    hi = order[:rank[u]]
    d = np.sqrt(((X[hi] - X[u]) ** 2).sum(1))
    j = int(d.argmin())
    return int(hi[j]), float(d[j])


def exact_k_nearest(X, t, cand, k):
    """Exact k nearest objects to X[t] among the index set `cand` (sorted by distance)."""
    d = np.sqrt(((X[cand] - X[t]) ** 2).sum(1))
    k = min(k, len(cand))
    sel = np.arange(len(cand)) if k == len(cand) else np.argpartition(d, k - 1)[:k]
    sel = sel[np.argsort(d[sel], kind="stable")]
    return cand[sel], d[sel]
