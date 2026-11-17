"""Benchmark datasets.  All are min-max scaled to [0,1]^d (BFI's e^delta / theta assume a bounded scale).

Synthetic sets keep their *structure* fixed (cluster centres are drawn once from a fixed seed)
and use `seed` only for the sampling noise, so repeated runs with different seeds measure the
sampling variability of one benchmark, not different benchmarks.  The real sets ship with
scikit-learn (no download); for them `seed` has no effect (they are fixed data).

Coverage:  shapes (moons, circles, anisotropic), densities (varied_density), overlap
(blobs_overlap), dimensionality (blobs_hd8, blobs_hd32, digits PCA), real data (iris, wine,
breast_cancer, digits), sample size (`scalable`, used by the scalability study).
"""
import numpy as np
from sklearn import datasets as skd
from sklearn.decomposition import PCA


def _norm(X):
    X = np.asarray(X, float)
    rng = X.max(0) - X.min(0)
    rng[rng == 0] = 1
    return (X - X.min(0)) / rng


def _centres(n_centres, dim, structure_seed):
    """The centres sklearn.make_blobs would draw for this structure seed (box (-10, 10))."""
    return np.random.RandomState(structure_seed).uniform(-10, 10, size=(n_centres, dim))


def synthetic(n=3000, seed=0):
    """2-D benchmark shapes.  `seed` changes only the sampled points."""
    out = {}
    X, y = skd.make_moons(n, noise=0.08, random_state=seed); out["moons"] = (X, y)
    X, y = skd.make_circles(n, noise=0.06, factor=0.5, random_state=seed); out["circles"] = (X, y)
    C0 = _centres(3, 2, 0)
    X, y = skd.make_blobs(n, centers=C0, cluster_std=0.6, random_state=seed); out["blobs_separated"] = (X, y)
    X, y = skd.make_blobs(n, centers=C0, cluster_std=1.2, random_state=seed); out["blobs_overlap"] = (X, y)
    C170 = _centres(3, 2, 170)
    X, y = skd.make_blobs(n, centers=C170, cluster_std=[1.0, 2.5, 0.5], random_state=seed)
    out["varied_density"] = (X, y)
    X, y = skd.make_blobs(n, centers=C170, random_state=seed)
    X = X @ np.array([[0.6, -0.6], [-0.4, 0.8]]); out["anisotropic"] = (X, y)
    return {k: (_norm(X), y) for k, (X, y) in out.items()}


def high_dim(n=3000, seed=0):
    """Gaussian mixtures in 8 and 32 dimensions (5 clusters, moderately overlapping)."""
    out = {}
    for d, std in ((8, 4.5), (32, 8.0)):
        X, y = skd.make_blobs(n, centers=_centres(5, d, 7), cluster_std=std, random_state=seed)
        out[f"blobs_hd{d}"] = (X, y)
    return {k: (_norm(X), y) for k, (X, y) in out.items()}


def real():
    out = {}
    d = skd.load_iris(); out["iris"] = (d.data, d.target)
    d = skd.load_wine(); out["wine"] = (d.data, d.target)
    d = skd.load_breast_cancer(); out["breast_cancer"] = (d.data, d.target)
    d = skd.load_digits(n_class=6); out["digits6_pca10"] = (PCA(10, random_state=0).fit_transform(d.data), d.target)
    d = skd.load_digits(); out["digits10_pca16"] = (PCA(16, random_state=0).fit_transform(d.data), d.target)
    return {k: (_norm(X), y) for k, (X, y) in out.items()}


def all_datasets(seed=0, n=3000):
    return {**synthetic(n, seed), **high_dim(n, seed), **real()}


def scalable(n, seed=0, centers=4, dim=2):
    """Gaussian mixture used by the scalability study (fixed centres, seed = sampling noise)."""
    X, y = skd.make_blobs(n, centers=_centres(centers, dim, 0), n_features=dim, cluster_std=0.8,
                          random_state=seed)
    return _norm(X), y
