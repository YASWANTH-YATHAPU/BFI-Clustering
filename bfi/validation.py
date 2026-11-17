"""Input validation shared by the original and the accelerated BFI implementations."""
import numbers
import numpy as np


def _int_param(name, v, lo):
    if isinstance(v, bool) or not isinstance(v, numbers.Integral) and not (
            isinstance(v, numbers.Real) and float(v).is_integer()):
        raise ValueError(f"{name} must be an integer, got {v!r}")
    v = int(v)
    if v < lo:
        raise ValueError(f"{name} must be >= {lo}, got {v}")
    return v


def check_inputs(X, n_clusters, K1, K2, zeta, q):
    """Validate the common BFI inputs and return (X as float array, c, K1 clipped to [1, N-1], K2)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D array (N, d), got shape {X.shape}")
    N = X.shape[0]
    if N < 3:
        raise ValueError(f"BFI needs at least 3 objects, got {N}")
    if X.shape[1] < 1:
        raise ValueError("X must have at least one feature")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values")
    c = _int_param("n_clusters", n_clusters, 1)
    if c > N:
        raise ValueError(f"n_clusters={c} exceeds the number of objects N={N}")
    K1 = _int_param("K1", K1, 1)
    K2 = _int_param("K2", K2, 1)
    if not (0.0 <= float(zeta) <= 1.0):
        raise ValueError(f"zeta must be in [0, 1], got {zeta}")
    if not (0.0 < float(q) < 1.0):
        raise ValueError(f"q must be in (0, 1), got {q}")
    return X, c, int(min(K1, N - 1)), K2
