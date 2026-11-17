"""Clustering metrics (all computed against the ground-truth labels y).

  ARI, NMI, AMI   sklearn adjusted Rand index / normalised & adjusted mutual information of the
                  crisp labels (arg-max of the pignistic probability, Eq. 16).
  P, R, RI        pair-counting precision, recall, Rand index of the crisp labels.
  Re              error rate: the true class (Hungarian-matched to a cluster) is NOT contained in
                  the focal set chosen for the object (singleton or meta-cluster).
  Ri              imprecision rate: the true class IS covered, but by a meta-cluster (|A| > 1).
  EP, ER, ERI     pairwise-credal precision / recall / Rand index on a 600-object subsample.  For a
                  pair (i, j) the BBAs of Eq. 15 give m(same) = sum over singletons A of
                  m_i(A) m_j(A), m(different) = sum over disjoint focal-set pairs, the rest is
                  ignorance; P_same = m(same) + m(ignorance)/2.  EP = sum_{same class} P_same /
                  sum P_same,  ER = sum_{same class} P_same / #same-class pairs,
                  ERI = mean over pairs of P_same (same class) or 1 - P_same (different class).
                  This is a SIMPLIFIED variant of the evidential Rand-index family
                  (Denoeux & Kanjanatarakul), not the exact published definition.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics as skm
from .evidence import full_bba


def _match(y, pred, c):
    """Hungarian matching of predicted cluster ids to ground-truth classes."""
    classes = np.unique(y)
    yi = np.searchsorted(classes, y)
    C = np.zeros((c, len(classes)))
    np.add.at(C, (pred, yi), 1)
    r, col = linear_sum_assignment(-C)
    pred_of_class = {int(col[i]): int(r[i]) for i in range(len(r))}
    return yi, pred_of_class


def error_imprecision(res, y):
    """Re: true cluster NOT in the assigned focal set.  Ri: true cluster in a meta-cluster (|A|>1)."""
    c = res.M.shape[1]
    yi, mp = _match(y, res.labels, c)
    tp = np.array([mp.get(int(k), -1) for k in yi])
    S = res.sets
    covered = np.where(tp >= 0, S[np.arange(len(y)), np.clip(tp, 0, c - 1)], False)
    imprecise = res.imprecise
    return float((~covered).mean()), float((covered & imprecise).mean())


def pair_prf(y, pred):
    (tn, fp), (fn, tp) = skm.cluster.pair_confusion_matrix(y, pred)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); RI = (tp + tn) / max(tp + tn + fp + fn, 1)
    return float(P), float(R), float(RI)


def pairwise_credal(res, y, n_sub=600, seed=0):
    """Pairwise-credal EP / ER / ERI on a random subsample (O(n_sub^2)).
    For a pair (i,j): mass(same)=sum_A m_i(A)m_j(A) over singletons, mass(different)=sum over
    disjoint focal sets, rest = ignorance; BetP_same = m_same + m_ign/2.  This is a simplified
    variant of the evidential Rand-index family (Denoeux & Kanjanatarakul) - see README."""
    c = res.M.shape[1]
    if c > 10:                      # 2^c focal sets; 2^10 = 1024 columns is still cheap
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(y), size=min(n_sub, len(y)), replace=False)
    B, masks = full_bba(res.M[ids])
    B = B[:, 1:] / np.maximum(B[:, 1:].sum(1, keepdims=True), 1e-300)   # drop empty set
    m = masks[1:]
    sing = (m & (m - 1)) == 0
    Ms = B[:, sing] @ B[:, sing].T
    Dis = ((m[:, None] & m[None, :]) == 0).astype(float)
    Md = B @ Dis @ B.T
    Mi = np.clip(1 - Ms - Md, 0, 1)
    P = Ms + Mi / 2
    yy = y[ids]
    T = yy[:, None] == yy[None, :]
    iu = np.triu_indices(len(ids), 1)
    P, T = P[iu], T[iu]
    ERI = np.mean(np.where(T, P, 1 - P))
    EP = P[T].sum() / max(P.sum(), 1e-12)
    ER = P[T].sum() / max(T.sum(), 1)
    return float(EP), float(ER), float(ERI)


def evaluate(res, y):
    lab = res.labels
    Re, Ri = error_imprecision(res, y)
    P, R, RI = pair_prf(y, lab)
    EP, ER, ERI = pairwise_credal(res, y)
    return {"Re": Re, "Ri": Ri, "ARI": skm.adjusted_rand_score(y, lab),
            "NMI": skm.normalized_mutual_info_score(y, lab), "AMI": skm.adjusted_mutual_info_score(y, lab),
            "P": P, "R": R, "RI": RI, "EP": EP, "ER": ER, "ERI": ERI}
