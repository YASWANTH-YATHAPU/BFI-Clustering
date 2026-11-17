"""Evidence-preservation metrics: how close is an accelerated result to the reference BFI?"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score


def align_clusters(ref_labels, new_labels, c):
    """Permutation p such that new cluster j corresponds to reference cluster p[j]."""
    C = np.zeros((c, c))
    np.add.at(C, (new_labels, ref_labels), 1)
    r, col = linear_sum_assignment(-C)
    perm = np.empty(c, dtype=int)
    perm[r] = col
    return perm


def compare_results(ref, new, X=None):
    c = ref.M.shape[1]
    perm = align_clusters(ref.labels, new.labels, c)
    Mn = np.zeros_like(new.M)
    Mn[:, perm] = new.M                       # columns re-indexed to the reference cluster ids
    Sn = np.zeros_like(new.sets)
    Sn[:, perm] = new.sets
    Bn = np.zeros_like(new.betp)
    Bn[:, perm] = new.betp
    shift = np.nan
    if X is not None:      # mean distance between Hungarian-matched centres (data is min-max scaled)
        from scipy.spatial.distance import cdist
        Dc = cdist(X[ref.centers], X[new.centers])
        r, col = linear_sum_assignment(Dc)
        shift = float(Dc[r, col].mean())
    return {
        "center_shift": shift,
        "bel_spearman": float(spearmanr(ref.bel, new.bel)[0]),
        "label_ari_vs_ref": float(adjusted_rand_score(ref.labels, new.labels)),
        "label_agree": float((Bn.argmax(1) == ref.labels).mean()),
        "betp_tv": float(0.5 * np.abs(ref.betp - Bn).sum(1).mean()),
        "meta_set_agree": float((Sn == ref.sets).all(1).mean()),
        "imprecision_ref": float(ref.imprecise.mean()),
        "imprecision_new": float(new.imprecise.mean()),
    }
