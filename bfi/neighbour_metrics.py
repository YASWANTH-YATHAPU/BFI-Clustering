"""Neighbour-search quality of an (accelerated) BFI run.

Every metric checks the neighbours a method actually used against the EXACT answer to the same
query *in the method's own state* (its own theta, Bel ordering, centres and allocation order).
This isolates the error of the neighbour search itself from upstream differences (e.g. theta
sampling), which are measured separately by the agreement metrics in bfi.compare.

Definitions (on a random subsample of at most n_eval query objects, fixed seed):
  bel_exact_frac      fraction of objects whose Bel used exact K1-NN distances (all objects).
  bel_err             mean |Bel_method - Bel_exact| / (P95 - P5 of Bel_exact), where Bel_exact uses
                      the exact K1 nearest distances and the method's own theta.
  bel_rho             Spearman correlation of Bel_method and Bel_exact (ordering quality).
  delta_cand_recall   fraction of objects whose exact delta-neighbour (nearest object with higher
                      Bel) lies inside the INITIAL candidate list (first k_base neighbours).
  delta_acc           final selection accuracy: the delta used equals the exact delta (tie-safe,
                      relative tolerance 1e-9).
  delta_relerr        mean |delta_method - delta_exact| / delta_exact.
  conv_cand_recall    mean over objects of |E_t  ∩ initial list| / |E_t|, E_t = exact K2 nearest
                      objects among those allocated before t.
  conv_acc            final selection accuracy: mean of |{used neighbours s : d(t,s) <= d_K2(t)}| /
                      |E_t| (tie-safe; fewer than K2 used neighbours count as misses).
End-to-end agreement with the ORIGINAL BFI's choices (needs a reference result, all objects):
  delta_nn_agree_ref  fraction of objects whose delta-neighbour equals the original's.
  conv_jaccard_ref    mean Jaccard similarity of the convergence neighbour sets.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr


def _exact_bel(X, theta, K1, rows):
    d, _ = cKDTree(X).query(X[rows], k=K1 + 1)
    d = d[:, 1:]
    return 1.0 - np.exp(np.log1p(-np.exp(-theta[rows, None] * d) / K1).sum(1))


def neighbour_quality(X, res, K1, K2, k_base=None, n_eval=1500, seed=0, ref=None, chunk=256):
    N = len(X)
    kb = N - 1 if k_base is None else int(k_base)
    rng = np.random.default_rng(seed)
    ev = np.sort(rng.choice(N, size=min(N, n_eval), replace=False))
    order = np.argsort(-res.bel, kind="stable")
    rank = np.empty(N, dtype=np.int64)
    rank[order] = np.arange(N)
    is_centre = np.zeros(N, bool)
    is_centre[res.centers] = True
    out = {"bel_exact_frac": float(res.diag["bel_exact"].mean())}

    be = _exact_bel(X, res.theta, K1, ev)
    rngb = max(np.percentile(be, 95) - np.percentile(be, 5), 1e-15)
    out["bel_err"] = float(np.abs(res.bel[ev] - be).mean() / rngb)
    out["bel_rho"] = float(spearmanr(res.bel[ev], be)[0])

    d_acc, d_rel, d_rec, c_acc, c_rec = [], [], [], [], []
    conv_nn = res.diag["conv_nn"]
    for s in range(0, len(ev), chunk):
        rows = ev[s:s + chunk]
        D = cdist(X[rows], X)
        D[np.arange(len(rows)), rows] = np.inf                       # exclude self
        for a, i in enumerate(rows):
            Di = D[a]
            if rank[i] > 0:                                          # ---- delta
                hm = rank < rank[i]
                de = Di[hm].min()
                d_acc.append(abs(res.delta[i] - de) <= 1e-9 * max(1.0, de))
                d_rel.append(abs(res.delta[i] - de) / max(de, 1e-15))
                d_rec.append((Di < de).sum() < kb)
            if not is_centre[i]:                                     # ---- convergence
                am = is_centre | (rank < rank[i])
                am[i] = False
                da = np.sort(Di[am])
                k = min(K2, len(da))
                rK = da[k - 1]
                used = conv_nn[i][conv_nn[i] >= 0]
                good = (Di[used] <= rK * (1 + 1e-12)).sum() if len(used) else 0
                c_acc.append(min(good, k) / k)
                ex = da[:k]
                pos = np.searchsorted(np.sort(Di), ex, side="left")  # rank of each exact neighbour
                c_rec.append((pos < kb).mean())
    out.update(delta_acc=float(np.mean(d_acc)), delta_relerr=float(np.mean(d_rel)),
               delta_cand_recall=float(np.mean(d_rec)), conv_acc=float(np.mean(c_acc)),
               conv_cand_recall=float(np.mean(c_rec)))
    if ref is not None:
        out["delta_nn_agree_ref"] = float((res.diag["delta_nn"] == ref.diag["delta_nn"]).mean())
        both = ~(is_centre | np.isin(np.arange(N), ref.centers))
        A, B = res.diag["conv_nn"][both], ref.diag["conv_nn"][both]
        jac = []
        for a, b in zip(A, B):
            sa, sb = set(a[a >= 0].tolist()), set(b[b >= 0].tolist())
            jac.append(len(sa & sb) / max(len(sa | sb), 1))
        out["conv_jaccard_ref"] = float(np.mean(jac))
    return out
