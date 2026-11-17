"""Accelerated, evidence-preserving BFI: Solution 1 (adaptive candidate retrieval) + Solution 2
(exact distance-based refinement), each switchable for the ablation study.

Pipeline (the evidential part - Eqs. 13-16, `bfi.evidence` - is shared with bfi_original):

  theta (Eq. 9)   exact by chunked selection (theta_sample=None) or from a random reference
                  sample of size theta_sample (approximation, independent of S1/S2).
  index           one spatial index (k-d tree / ball tree) replaces the N x N matrix.
  Bel (Eq. 10)    refine=True, bel_refine='all': exact K1 distances, streamed in chunks (S2a).
                  otherwise: exact head over k_cand neighbours + tail ranks k_cand+1..K1 from a
                  local power-law model d_k ~ d_kc (k/kc)^(1/h)  (estimate).
                  bel_refine='flagged' / 'centres' refine only a subset exactly (ablation).
  delta (Eq. 11)  first higher-Bel object in the sorted base list (exact certificate);
                  adaptive=True (S1): failing lists grow k -> 2k via the index up to k_max;
                  still unresolved: refine=True (S2b) exact scan of the higher-Bel subset,
                  refine=False -> estimate (nearest higher-Bel landmark).
  convergence     K2 nearest *allocated* objects: first K2 allocated entries of the sorted list
  (Eqs. 13-15)    (exact certificate); S1 grows failing lists; still unresolved: S2 exact scan of
                  the allocated subset, otherwise the allocated objects found so far (or the
                  nearest centre if none) are used (estimate).

Variants used in the experiments (see bfi.variants):
  control  adaptive=False, refine=False   fixed k-NN graph, everything else estimated
  S1       adaptive=True,  refine=False
  S2       adaptive=False, refine=True
  S1+S2    adaptive=True,  refine=True    (proposed)
With refine=True, bel_refine='all' and theta_sample=None the result is identical to bfi_original
(up to floating-point rounding of the distances), whatever adaptive/k_min/k_max are.
"""
import time
import numpy as np
from scipy.spatial.distance import cdist
from .candidates import first_higher_in_lists, adaptive_nearest_higher, adaptive_allocated
from .evidence import fuse_neighbours
from .neighbors import Searcher
from .original import select_centers
from .refine import (bel_terms, exact_bel_streamed, exact_theta, sampled_theta,
                     exact_nearest_higher, exact_k_nearest, _ENTRY_BUDGET)
from .result import BFIResult
from .validation import check_inputs, _int_param

_trapz = getattr(np, "trapezoid", None) or np.trapz

# source codes recorded per object in result.diag (delta_src / conv_src)
SRC = {"centre": -1, "base": 0, "expanded": 1, "exact_scan": 2, "estimate": 3, "trivial": 4}
BEL_POLICIES = ("all", "flagged", "centres", "none")


# ----------------------------------------------------------------------------------------------
# estimated Bel (used when Solution 2 does not refine Bel)
# ----------------------------------------------------------------------------------------------
def _tail_model(dist, theta, K1, kc, tol):
    """Tail of the Eq.-10 log-product over ranks kc+1..K1 from a local power-law model.

    The local intrinsic dimension h is estimated from the head at two scales (ranks kc/4->kc/2 and
    kc/2->kc).  Returns (tail, flagged) where flagged marks objects whose two estimates disagree
    by more than `tol` (relative) - used only by the 'flagged' refinement policy."""
    q1, q2 = max(kc // 4, 1), max(kc // 2, 2)
    d1, d2, d3 = dist[:, q1 - 1], dist[:, q2 - 1], dist[:, kc - 1]
    valid = (d1 > 0) & (d2 > d1) & (d3 > d2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ha = np.log(q2 / q1) / np.log(d2 / d1)
        hb = np.log(kc / q2) / np.log(d3 / d2)
        rel = np.abs(ha - hb) / np.maximum(ha, hb)
    flagged = ~valid | ~(rel <= tol)                         # NaN -> flagged
    h = np.clip(np.where(valid, hb, 1.0), 0.5, 30.0)
    nodes = np.geomspace(kc + 0.5, K1 + 0.5, 32)
    dn = d3[:, None] * (nodes[None, :] / kc) ** (1.0 / h[:, None])
    f = np.log1p(-np.exp(-theta[:, None] * dn) / K1)
    return _trapz(f, nodes, axis=1), flagged


def _estimated_bel_streamed(searcher, theta, K1, kc, keep, tail, tol):
    """Exact head over kc neighbours + modelled tail; also returns the first `keep` neighbours."""
    N = searcher.N
    kq = int(min(max(kc, keep), N - 1))
    logprod = np.empty(N)
    flagged = np.zeros(N, bool)
    kidx = np.empty((N, keep), dtype=np.int64)
    kdist = np.empty((N, keep))
    step = max(1, _ENTRY_BUDGET // kq)
    for s in range(0, N, step):
        r = np.arange(s, min(s + step, N))
        idx, dist = searcher.query(r, kq)
        lp = bel_terms(dist[:, :kc], theta[r], K1)
        if tail == "extrapolate":
            t_, fl = _tail_model(dist[:, :kc], theta[r], K1, kc, tol)
            lp, flagged[r] = lp + t_, fl
        else:                                                   # 'truncate' (ablation)
            flagged[r] = True
        logprod[r] = lp
        kidx[r], kdist[r] = idx[:, :keep], dist[:, :keep]
    return logprod, kidx, kdist, flagged


# ----------------------------------------------------------------------------------------------
# delta (Eq. 11)
# ----------------------------------------------------------------------------------------------
def _compute_delta(X, sr, bidx, bdist, order, rank, adaptive, refine, kb, kmax, landmarks, cnt,
                   delta_estimate="landmark"):
    N = len(X)
    delta = np.full(N, np.nan)
    nn = -np.ones(N, dtype=np.int64)
    src = np.full(N, SRC["base"], dtype=np.int8)
    k_used = np.full(N, bidx.shape[1], dtype=np.int64)
    d_last = bdist[:, -1].copy()        # distance of the last candidate examined (lower bound)
    found, a, b = first_higher_in_lists(bidx, bdist, rank)
    delta[found], nn[found] = b[found], a[found]
    top = order[0]
    nn[top], delta[top] = exact_nearest_higher(X, top, order, rank)   # max distance, O(N)
    found[top], src[top] = True, SRC["trivial"]
    unres = np.flatnonzero(~found)
    cnt["delta_base"] = int(found.sum()) - 1
    if adaptive and len(unres):
        f2, a2, b2, k2, dl2 = adaptive_nearest_higher(sr, unres, rank, kb, kmax)
        hit = unres[f2]
        delta[hit], nn[hit], src[hit] = b2[f2], a2[f2], SRC["expanded"]
        k_used[unres], d_last[unres] = k2, np.where(np.isnan(dl2), d_last[unres], dl2)
        unres = unres[~f2]
        cnt["delta_expanded"] = int(f2.sum())
    else:
        cnt["delta_expanded"] = 0
    if len(unres):
        if refine:                                   # S2: exact scan of the higher-Bel subset
            for u in unres:
                nn[u], delta[u] = exact_nearest_higher(X, u, order, rank)
            src[unres] = SRC["exact_scan"]
            k_used[unres] = rank[unres]
        else:                                        # estimate (no exact refinement)
            # lower bound: delta_u > distance to the last candidate examined (certified);
            # upper bound: distance to the nearest higher-Bel landmark (a real object).
            L = np.union1d(landmarks, [top])
            Dl = cdist(X[unres], X[L])
            Dl[rank[L][None, :] >= rank[unres][:, None]] = np.inf
            j = Dl.argmin(1)
            ub = Dl[np.arange(len(unres)), j]
            lb = d_last[unres]
            if delta_estimate == "landmark":
                est = ub
            elif delta_estimate == "lower_bound":
                est = lb
            else:                                    # 'geometric': geometric mean of the bounds
                est = np.sqrt(lb * np.maximum(ub, lb))
            delta[unres], nn[unres] = est, L[j]
            src[unres] = SRC["estimate"]
    cnt["delta_scan"] = int(len(unres)) if refine else 0
    cnt["delta_est"] = 0 if refine else int(len(unres))
    return delta, nn, src, k_used


# ----------------------------------------------------------------------------------------------
def bfi_fast(X, n_clusters, K1, K2=9, zeta=0.1, q=0.9, k_cand=64, tol=0.5, theta_sample=1000,
             backend="auto", tail="extrapolate", seed=0, center_rule="bel_delta",
             adaptive=True, refine=True, bel_refine="all", k_min=16, k_max=512,
             refine_top=None, n_landmarks=256, n_jobs=1, delta_estimate="landmark"):
    """Accelerated BFI.

    Parameters (beyond the BFI parameters K1, K2, zeta, q, center_rule)
    adaptive      Solution 1 on/off - adaptive candidate retrieval for delta and convergence.
    refine        Solution 2 on/off - exact refinement of Bel and of unresolved queries.
    bel_refine    with refine=True: 'all' (exact Bel for every object, default), 'flagged'
                  (objects failing the two-scale model test, tolerance `tol`), 'centres' (the
                  `refine_top` objects with the largest gamma), 'none'.
    k_cand        neighbours used for the Bel head when Bel is estimated, and the fixed list size
                  when adaptive=False.
    k_min, k_max  initial and maximal candidate-list size of Solution 1.
    theta_sample  None -> exact theta (Eq. 9); int s -> theta from a random sample of s objects.
    tail          'extrapolate' (power-law tail) | 'truncate' (ignore ranks > k_cand), only used
                  when Bel is estimated.
    seed          random seed (theta sample, landmarks).  n_jobs: index threads (default 1).
    """
    X, c, K1, K2 = check_inputs(X, n_clusters, K1, K2, zeta, q)
    N = len(X)
    k_cand = _int_param("k_cand", k_cand, 1)
    k_min = _int_param("k_min", k_min, 1)
    k_max = _int_param("k_max", k_max, 1)
    if k_max < k_min:
        raise ValueError(f"k_max ({k_max}) must be >= k_min ({k_min})")
    if bel_refine not in BEL_POLICIES:
        raise ValueError(f"bel_refine must be one of {BEL_POLICIES}, got {bel_refine!r}")
    if tail not in ("extrapolate", "truncate"):
        raise ValueError(f"tail must be 'extrapolate' or 'truncate', got {tail!r}")
    if center_rule not in ("bel_delta", "paper"):
        raise ValueError(f"center_rule must be 'bel_delta' or 'paper', got {center_rule!r}")
    if theta_sample is not None:
        theta_sample = _int_param("theta_sample", theta_sample, 2)
    if delta_estimate not in ("landmark", "lower_bound", "geometric"):
        raise ValueError(f"delta_estimate must be 'landmark', 'lower_bound' or 'geometric', got {delta_estimate!r}")
    if not (tol >= 0):
        raise ValueError(f"tol must be >= 0, got {tol}")

    kc = int(min(max(k_cand, K2, 8), K1, N - 1))          # Bel head size (estimated Bel)
    kc = max(kc, min(K2, N - 1))
    kb = int(min(max(k_min, K2), N - 1)) if adaptive else kc   # base candidate-list size
    kmax = int(min(max(k_max, kb), N - 1))
    bel_policy = bel_refine if refine else "none"
    rng = np.random.default_rng(seed)
    info, cnt = {}, {}
    T0 = time.perf_counter()

    # ---- theta (Eq. 9) -----------------------------------------------------------------------
    t = time.perf_counter()
    theta_exact = theta_sample is None or theta_sample >= N
    theta = exact_theta(X, q) if theta_exact else sampled_theta(X, q, theta_sample, rng)
    info["t_theta"] = time.perf_counter() - t

    # ---- spatial index -------------------------------------------------------------------------
    t = time.perf_counter()
    sr = Searcher(X, backend, n_jobs)
    info["t_index"] = time.perf_counter() - t

    # ---- Bel (Eq. 10) + base candidate lists ------------------------------------------------------
    t = time.perf_counter()
    if bel_policy == "all" or kc >= K1:
        logprod, bidx, bdist = exact_bel_streamed(sr, theta, K1, keep=kb)
        bel_exact = np.ones(N, bool)
    else:
        logprod, bidx, bdist, flagged = _estimated_bel_streamed(sr, theta, K1, kc, kb, tail, tol)
        bel_exact = np.zeros(N, bool)
        if bel_policy == "flagged" and flagged.any():
            er = np.flatnonzero(flagged)
            logprod[er] = exact_bel_streamed(sr, theta, K1, keep=1, rows=er)[0]
            bel_exact[er] = True
    bel = 1.0 - np.exp(logprod)
    order = np.argsort(-bel, kind="stable")
    rank = np.empty(N, dtype=np.int64)
    rank[order] = np.arange(N)
    info["t_bel"] = time.perf_counter() - t

    # ---- delta (Eq. 11) ----------------------------------------------------------------------
    t = time.perf_counter()
    landmarks = rng.choice(N, size=min(N, n_landmarks), replace=False)
    delta, delta_nn, delta_src, delta_k = _compute_delta(
        X, sr, bidx, bdist, order, rank, adaptive, refine, kb, kmax, landmarks, cnt,
                delta_estimate)
    centers = select_centers(delta, bel, c, center_rule)
    if bel_policy == "centres":
        # S2 (evidence-aware variant): exact Bel for the decision-graph contenders, then redo delta
        m = int(refine_top) if refine_top is not None else max(5 * c, 20)
        cand = np.argsort(-(bel * delta), kind="stable")[:min(m, N)]
        cand = cand[~bel_exact[cand]]
        if len(cand):
            logprod[cand] = exact_bel_streamed(sr, theta, K1, keep=1, rows=cand)[0]
            bel_exact[cand] = True
            bel = 1.0 - np.exp(logprod)
            order = np.argsort(-bel, kind="stable")
            rank[order] = np.arange(N)
            delta, delta_nn, delta_src, delta_k = _compute_delta(
                X, sr, bidx, bdist, order, rank, adaptive, refine, kb, kmax, landmarks, cnt,
                delta_estimate)
            centers = select_centers(delta, bel, c, center_rule)
    info["t_delta"] = time.perf_counter() - t

    # ---- evidential convergence (Eqs. 13-15) ---------------------------------------------------
    t = time.perf_counter()
    lab = -np.ones(N, dtype=int)
    M = np.zeros((N, c))
    conv_nn = -np.ones((N, K2), dtype=np.int64)
    conv_src = np.full(N, SRC["centre"], dtype=np.int8)
    conv_k = np.zeros(N, dtype=np.int64)
    alloc_ids = np.empty(N, dtype=np.int64)
    n_alloc = 0
    for k, ci in enumerate(centers):
        lab[ci] = k
        M[ci, k] = 1.0
        alloc_ids[n_alloc] = ci
        n_alloc += 1
    for tt in order:
        if lab[tt] >= 0:
            continue
        if n_alloc <= K2:                                    # paper footnote 3: use all allocated
            nb = alloc_ids[:n_alloc].copy()
            nd = np.sqrt(((X[nb] - X[tt]) ** 2).sum(1))
            status, ku = "trivial", 0
        elif adaptive:                                       # S1
            status, nb, nd, ku = adaptive_allocated(sr, tt, bidx[tt], bdist[tt], lab, K2, kmax)
        else:                                                # fixed list
            ok = np.flatnonzero(lab[bidx[tt]] >= 0)
            sel = ok[:K2]
            nb, nd, ku = bidx[tt, sel], bdist[tt, sel], kb
            status = "base" if len(ok) >= K2 else "partial"
        if status == "partial":
            if refine:                                       # S2: exact scan of the allocated subset
                nb, nd = exact_k_nearest(X, tt, alloc_ids[:n_alloc], K2)
                status, ku = "exact_scan", n_alloc
            else:                                            # estimate
                if len(nb) == 0:                             # nothing allocated in the list
                    dc = np.sqrt(((X[centers] - X[tt]) ** 2).sum(1))
                    j = int(dc.argmin())
                    nb, nd = centers[[j]], dc[[j]]
                status = "estimate"
        conv_src[tt], conv_k[tt] = SRC[status], ku
        conv_nn[tt, :len(nb)] = nb
        mv = fuse_neighbours(lab[nb], nd, theta[tt], zeta, c)
        M[tt] = mv
        lab[tt] = mv.argmax()
        alloc_ids[n_alloc] = tt
        n_alloc += 1
    info["t_convergence"] = time.perf_counter() - t
    info["t_total"] = time.perf_counter() - T0

    nc = conv_src[conv_src >= 0]
    info.update(
        variant=("S1+S2" if adaptive and refine else "S1" if adaptive else "S2" if refine else "control"),
        K1=K1, K2=K2, k_cand=kc, k_base=kb, k_max=kmax, theta_exact=bool(theta_exact),
        bel_policy=bel_policy, n_bel_exact=int(bel_exact.sum()),
        n_delta_base=cnt["delta_base"], n_delta_expanded=cnt["delta_expanded"],
        n_delta_scan=cnt["delta_scan"], n_delta_est=cnt["delta_est"],
        n_conv_trivial=int((nc == SRC["trivial"]).sum()), n_conv_base=int((nc == SRC["base"]).sum()),
        n_conv_expanded=int((nc == SRC["expanded"]).sum()),
        n_conv_scan=int((nc == SRC["exact_scan"]).sum()), n_conv_est=int((nc == SRC["estimate"]).sum()),
        mean_k_delta=float(delta_k.mean()),
        mean_k_conv=float(conv_k[conv_src >= 0].mean()) if len(nc) else 0.0,
        index_entries=int(sr.n_entries), base_list_entries=int(bidx.size),
        # backward-compatible counters
        n_escalated=int(bel_exact.sum()) if bel_policy == "flagged" else 0,
        n_conv_fallback=int(((nc == SRC["exact_scan"]) | (nc == SRC["estimate"])).sum()),
    )
    diag = {"delta_nn": delta_nn, "delta_src": delta_src, "delta_k": delta_k,
            "conv_nn": conv_nn, "conv_src": conv_src, "conv_k": conv_k, "bel_exact": bel_exact}
    return BFIResult(M=M, bel=bel, delta=delta, centers=centers, theta=theta, info=info, diag=diag)
