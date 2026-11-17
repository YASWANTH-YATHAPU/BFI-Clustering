"""Spatial-index back-ends used by the accelerated BFI.

All back-ends return the k nearest neighbours of the query rows (the query object itself
excluded) sorted by increasing distance.  The k-d tree and scikit-learn back-ends are *exact*
k-NN searches: the distances they return are true Euclidean distances, so a prefix of a returned
list is always the exact nearest-neighbour prefix.  `pynndescent` is approximate (optional).

`n_jobs` controls parallelism.  The experiments use n_jobs=1 so that the comparison with the
single-threaded numpy baseline is fair (see README, "Measurement protocol").
"""
import numpy as np

_BACKENDS = ("auto", "kdtree", "sklearn", "pynndescent")


class Searcher:
    def __init__(self, X, backend="auto", n_jobs=1):
        if backend not in _BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; choose one of {_BACKENDS}")
        self.X = X
        self.N, d = X.shape
        if backend == "auto":
            # k-d tree for every dimension: scikit-learn's per-call overhead makes the many
            # single-object queries of Solution 1 up to ~9x slower (measured on digits10, d=16..30)
            backend = "kdtree"
        self.backend = backend
        self.n_jobs = int(n_jobs)
        self.n_queries = 0           # number of query *rows* answered (diagnostics)
        self.n_entries = 0           # number of neighbour entries returned  (sum of k)
        if backend == "kdtree":
            from scipy.spatial import cKDTree
            self.index = cKDTree(X)
        elif backend == "sklearn":
            from sklearn.neighbors import NearestNeighbors
            self.index = NearestNeighbors(algorithm="auto", n_jobs=self.n_jobs).fit(X)
        else:                                     # approximate, optional dependency
            from pynndescent import NNDescent
            self.index = NNDescent(X, random_state=0, n_jobs=self.n_jobs)
            self.index.prepare()

    def query(self, rows, k):
        """k nearest neighbours (self excluded) of X[rows]; returns (idx, dist), each (len(rows), k)."""
        rows = np.atleast_1d(np.asarray(rows))
        k = int(min(k, self.N - 1))
        if k < 1:
            raise ValueError("k must be >= 1 and N >= 2")
        Q = self.X[rows]
        if self.backend == "kdtree":
            dist, idx = self.index.query(Q, k=k + 1, workers=self.n_jobs)
        elif self.backend == "sklearn":
            dist, idx = self.index.kneighbors(Q, n_neighbors=k + 1)
        else:
            idx, dist = self.index.query(Q, k=k + 1)
        idx = np.asarray(idx).reshape(len(rows), k + 1)
        dist = np.asarray(dist, dtype=float).reshape(len(rows), k + 1)
        # drop self (normally column 0; with duplicate points it may be elsewhere)
        selfpos = idx == rows[:, None]
        none = ~selfpos.any(1)
        selfpos[none, -1] = True
        first = selfpos.argmax(1)                 # keep exactly one self column per row
        selfpos[:] = False
        selfpos[np.arange(len(rows)), first] = True
        keep = ~selfpos
        self.n_queries += len(rows)
        self.n_entries += len(rows) * k
        return idx[keep].reshape(len(rows), k), dist[keep].reshape(len(rows), k)
