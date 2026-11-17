"""Evidence-theory helpers for BFI (Zhang et al., IEEE TFS 2025).

Convergence stage, per remaining object x_t:
  * every allocated neighbour k with label phi gives a simple support function
        m_k({w_phi}) = exp(-theta_t d_tk),   m_k(not w_phi) = 1 - exp(-...)      (Eq. 13)
  * evidence from the same cluster is fused by Dempster's rule (Eq. 6); since the
    "not w_phi" mass is neutral (ignorance) this gives  m^phi = 1 - prod_k (1 - s_k)
  * clusters whose m^phi is within zeta of the maximum form M_t               (Eq. 14)
  * BBA over 2^{M_t}  (Eq. 15):   m(A) ∝ prod_{phi in A} m^phi * prod_{phi in M_t\\A} (1 - m^phi)
    -> the maximiser has the closed form  A* = {phi : m^phi > 0.5}  (or the arg-max singleton
       if empty), so we never have to enumerate 2^|M_t| subsets for the decision.
"""
import numpy as np


def fuse_neighbours(labels, dists, theta, zeta, c):
    """Return the cluster-level support vector m^phi restricted to M_t (zeros elsewhere)."""
    s = np.clip(np.exp(-theta * dists), 0.0, 1.0 - 1e-12)
    log_comp = np.bincount(labels, weights=np.log1p(-s), minlength=c)
    m = 1.0 - np.exp(log_comp)                      # 0 for clusters with no neighbour
    keep = (m.max() - m <= zeta) & (m > 0)          # Eq. (14)
    return np.where(keep, m, 0.0)


def decision_sets(M):
    """Focal element with highest BBA (empty set excluded) for each row of M."""
    S = M > 0.5
    none = ~S.any(1)
    S[np.flatnonzero(none), M[none].argmax(1)] = True
    return S


def pignistic(M):
    """Pignistic probability BetP (Eq. 16) of every object over the c singleton clusters.

    BBA of a subset A of M_t is prod_{A} m * prod_{M_t \\ A}(1-m); the sum over subsets that
    contain phi of m(A)/|A| is evaluated with a Poisson-binomial recursion (O(c^2)).
    The empty set (noise) is excluded and the result renormalised.
    """
    N, c = M.shape
    out = np.zeros_like(M)
    for i in range(N):
        m = M[i]
        nz = np.flatnonzero(m > 0)
        if len(nz) == 1:
            out[i, nz[0]] = 1.0
            continue
        for p in nz:
            pb = np.array([1.0])
            for o in nz[nz != p]:
                pb = np.convolve(pb, [1.0 - m[o], m[o]])
            out[i, p] = m[p] * np.sum(pb / (np.arange(len(pb)) + 1.0))
        out[i] /= out[i].sum()
    return out


def full_bba(M):
    """Full BBA over all 2^c subsets (columns indexed by bitmask, column 0 = empty set)."""
    N, c = M.shape
    masks = np.arange(2 ** c)
    sub = ((masks[:, None] >> np.arange(c)) & 1).astype(float)          # (2^c, c)
    lm = np.log(np.clip(M, 1e-300, 1.0))
    l1m = np.log(np.clip(1.0 - M, 1e-300, 1.0))
    L = lm @ sub.T + l1m @ (1.0 - sub).T
    B = np.exp(L)
    return B / B.sum(1, keepdims=True), masks
