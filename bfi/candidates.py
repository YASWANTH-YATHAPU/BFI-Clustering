"""Solution 1 - Adaptive Candidate Neighbour Retrieval (ACR).

The original BFI answers every neighbour query from the full N x N distance matrix.  The
accelerated BFI instead keeps, for every object, a short *base* candidate list (its k_base exact
nearest neighbours, sorted, obtained from a spatial index), and answers the two "search"
queries of BFI from that list:

  Q_delta(i):  the nearest object with a larger Bel                       (Eq. 11)
  Q_conv(t):   the K2 nearest objects that are already allocated           (Eqs. 13-15)

Both queries have an *exact sufficiency certificate* on a sorted exact candidate prefix:
  * the first candidate with larger Bel IS the nearest larger-Bel object, and
  * the first K2 allocated candidates ARE the K2 nearest allocated objects,
because every object closer than the returned one is in the prefix and was checked.

Adaptivity.  How many candidates an object needs depends on its local data characteristics:
inside a dense cluster core the nearest higher-Bel / allocated objects are among the first few
neighbours, whereas for local density maxima, cluster borders, sparse regions and the early,
sparsely-allocated phase of the convergence they can be far away.  ACR therefore starts every
object with the same small list and grows *only the lists that fail the certificate*, by
geometric doubling (k -> 2k) through the spatial index, up to a budget k_max.  Objects whose
list still fails at k_max are returned as unresolved; what happens to them is decided by the
caller (exact refinement = Solution 2, or an estimate when Solution 2 is disabled).

Every answer returned as "resolved" by this module is exact.
"""
import numpy as np


def _k_schedule(k_start, k_max, N):
    """Geometric sizes k_start*2, k_start*4, ... capped at min(k_max, N-1) (last size = the cap)."""
    cap = int(min(k_max, N - 1))
    ks, k = [], int(k_start)
    while k < cap:
        k = min(2 * k, cap)
        ks.append(k)
    return ks


def first_higher_in_lists(idx, dist, rank, rows=None):
    """Certificate test for Q_delta on (rows of) sorted candidate lists.

    Returns (found, nn, d): found[i] is True iff the list of row i contains an object of higher
    Bel (smaller rank); then nn[i] / d[i] are that object and its distance (exact answer)."""
    r = rank[np.arange(len(idx))] if rows is None else rank[rows]
    higher = rank[idx] < r[:, None]
    found = higher.any(1)
    first = higher.argmax(1)
    ar = np.arange(len(idx))
    return found, idx[ar, first], dist[ar, first]


def adaptive_nearest_higher(searcher, rows, rank, k_start, k_max):
    """Grow the candidate lists of `rows` geometrically until each contains a higher-Bel object.

    Returns (found, nn, d, k_used, d_last) aligned with `rows`.  Unfound rows keep nn=-1, d=nan,
    k_used = the largest list size tried and d_last = the distance of their k_used-th neighbour:
    their true delta is then known to be >= d_last (certified lower bound).  d_last is nan for
    rows that were never expanded (empty schedule)."""
    rows = np.asarray(rows)
    n = len(rows)
    found = np.zeros(n, bool)
    nn = -np.ones(n, dtype=np.int64)
    d = np.full(n, np.nan)
    k_used = np.full(n, int(k_start))
    d_last = np.full(n, np.nan)
    pending = np.arange(n)
    for k in _k_schedule(k_start, k_max, searcher.N):
        if len(pending) == 0:
            break
        idx, dist = searcher.query(rows[pending], k)
        f, a, b = first_higher_in_lists(idx, dist, rank, rows[pending])
        k_used[pending] = k
        d_last[pending] = dist[:, -1]
        hit = pending[f]
        found[hit], nn[hit], d[hit] = True, a[f], b[f]
        pending = pending[~f]
    return found, nn, d, k_used, d_last


def adaptive_allocated(searcher, t, base_idx, base_dist, lab, K2, k_max):
    """Q_conv for one object t: K2 nearest allocated objects from adaptively grown candidates.

    Returns (status, nbr, nd, k_used) where status is
      'base'      - certified from the base list,
      'expanded'  - certified after growing the list through the index,
      'partial'   - list budget exhausted; nbr/nd hold the allocated objects that were found
                    (fewer than K2, possibly none) - NOT a certified answer."""
    ok = np.flatnonzero(lab[base_idx] >= 0)
    if len(ok) >= K2:
        s = ok[:K2]
        return "base", base_idx[s], base_dist[s], len(base_idx)
    idx, dist, k_used = base_idx, base_dist, len(base_idx)
    for k in _k_schedule(len(base_idx), k_max, searcher.N):
        idx, dist = searcher.query(np.array([t]), k)
        idx, dist, k_used = idx[0], dist[0], k
        ok = np.flatnonzero(lab[idx] >= 0)
        if len(ok) >= K2:
            s = ok[:K2]
            return "expanded", idx[s], dist[s], k_used
    ok = np.flatnonzero(lab[idx] >= 0)
    return "partial", idx[ok], dist[ok], k_used
