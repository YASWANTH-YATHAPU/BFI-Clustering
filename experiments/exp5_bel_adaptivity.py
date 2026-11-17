"""Exp 5 - can the candidate-set size for Bel be chosen adaptively per object from local data?

This experiment justifies a design decision: Solution 1 adapts candidate lists for delta and the
convergence (where an exact sufficiency certificate exists), but NOT for the Bel tail.

For an estimated Bel (exact head over k neighbours + power-law tail to K1) we compare, at equal
mean candidate count:
  fixed      every object uses the same k                        (k = 16, 32, 64, 128)
  two-scale  object i grows k (16 -> 32 -> ... -> 256) until its local-dimension estimates at two
             scales agree within tol (the test used by the earlier 'escalation' design)
  validation object i grows k until a power law fitted on its first k/2 neighbours predicts the
             Bel terms of neighbours k/2..k within tol (a direct out-of-sample check)
Error = |Bel_est - Bel_exact| / (P95 - P5 of Bel_exact) and Spearman rank correlation with the exact
Bel; exact theta; N = 4000; 3 seeds.
"""
import os, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from common import RES, PLOTS, TABLES, k1_of, mean_std, write_markdown
from bfi import datasets as D
from bfi.refine import exact_theta

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _tail(dk, k, K1, th, h):
    nodes = np.geomspace(k + 0.5, K1 + 0.5, 32)
    dn = dk[:, None] * (nodes[None, :] / k) ** (1.0 / h[:, None])
    return _trapz(np.log1p(-np.exp(-th[:, None] * dn) / K1), nodes, axis=1)


def _h(dd, k):
    lr = np.log(dd[:, k - 1:k] / np.maximum(dd[:, :k - 1], 1e-300))
    return np.clip((k - 2) / np.maximum(lr.sum(1), 1e-12), 0.5, 30)


def _est(d, th, K1, k, rows):
    dd = d[rows, :k]
    head = np.log1p(-np.exp(-th[rows, None] * dd) / K1).sum(1)
    return head if k >= K1 else head + _tail(dd[:, k - 1], k, K1, th[rows], _h(dd, k))


def _crit(d, th, K1, k, rows, kind):
    dd = d[rows, :k]
    if kind == "two-scale":
        q1, q2 = max(k // 4, 1), max(k // 2, 2)
        d1, d2, d3 = dd[:, q1 - 1], dd[:, q2 - 1], dd[:, k - 1]
        with np.errstate(all="ignore"):
            ha, hb = np.log(q2 / q1) / np.log(d2 / d1), np.log(k / q2) / np.log(d3 / d2)
            return np.nan_to_num(np.abs(ha - hb) / np.maximum(ha, hb), nan=9.0)
    kh = k // 2
    dh, h = dd[:, :kh], _h(dd[:, :kh], kh)
    nodes = np.geomspace(kh + 0.5, k + 0.5, 16)
    pred = _trapz(np.log1p(-np.exp(-th[rows, None] * (dh[:, -1:] * (nodes[None] / kh) ** (1 / h[:, None]))) / K1),
                  nodes, axis=1)
    act = np.log1p(-np.exp(-th[rows, None] * dd[:, kh:]) / K1).sum(1)
    return np.abs(pred - act) / np.abs(act)


def run(quick=False):
    N = 2000 if quick else 4000
    seeds = (0,) if quick else (0, 1, 2)
    rows = []
    for seed in seeds:
        S, H = D.synthetic(N, seed), D.high_dim(N, seed)
        data = {"varied_density": S["varied_density"], "blobs_overlap": S["blobs_overlap"],
                "moons": S["moons"], "blobs_hd8": H["blobs_hd8"]}
        for name, (X, y) in data.items():
            K1 = k1_of(N)
            th = exact_theta(X, 0.9)
            d = cKDTree(X).query(X, K1 + 1)[0][:, 1:]
            bx = 1 - np.exp(np.log1p(-np.exp(-th[:, None] * d) / K1).sum(1))
            rg = np.percentile(bx, 95) - np.percentile(bx, 5)

            def rec(rule, param, L, ks):
                b = 1 - np.exp(L)
                rows.append({"dataset": name, "seed": seed, "rule": rule, "param": param,
                             "mean_k": ks.mean(), "err": np.abs(b - bx).mean() / rg,
                             "err_p99": np.percentile(np.abs(b - bx), 99) / rg,
                             "rho": spearmanr(b, bx)[0]})
            for k in (16, 32, 64, 128):
                rec("fixed", k, _est(d, th, K1, k, np.arange(N)), np.full(N, k))
            for kind, tols in (("two-scale", (0.05, 0.1, 0.2, 0.35)), ("validation", (0.01, 0.025, 0.05, 0.1))):
                for tol in tols:
                    ks, L, act, k = np.zeros(N), np.zeros(N), np.arange(N), 16
                    while len(act):
                        kk = min(k, K1)
                        ok = (_crit(d, th, K1, kk, act, kind) <= tol) | (kk >= min(256, K1))
                        fin = act[ok]
                        L[fin], ks[fin] = _est(d, th, K1, kk, fin), kk
                        act, k = act[~ok], 2 * k
                    rec(kind, tol, L, ks)
            print(f"seed={seed} {name} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "exp5_raw.csv"), index=False)
    summ = mean_std(df, ["dataset", "rule", "param"], ["mean_k", "err", "err_p99", "rho"])
    summ.to_csv(os.path.join(RES, "exp5_summary.csv"), index=False)
    write_markdown(summ[["dataset", "rule", "param", "mean_k_mean", "err_mean", "err_std", "rho_mean"]],
                   os.path.join(TABLES, "exp5_bel_adaptivity.md"), 4)
    names = list(dict.fromkeys(summ.dataset))
    fig, ax = plt.subplots(1, len(names), figsize=(4 * len(names), 3.6))
    for a, n in zip(ax, names):
        for rule, mk in (("fixed", "ko-"), ("two-scale", "s--"), ("validation", "^:")):
            d = summ[(summ.dataset == n) & (summ.rule == rule)].sort_values("mean_k_mean")
            a.errorbar(d.mean_k_mean, d.err_mean, yerr=d.err_std, fmt=mk, ms=4, capsize=2, label=rule)
        a.set(xscale="log", yscale="log", xlabel="mean candidates per object", title=n)
    ax[0].set_ylabel("mean |Bel_est - Bel| / Bel range")
    ax[0].legend(fontsize=7)
    plt.suptitle("Estimated Bel: per-object adaptive candidate size vs fixed size (lower-left is better)", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "exp5_bel_adaptivity.png"), dpi=140); plt.close()
    return df


if __name__ == "__main__":
    t = time.time(); run(); print(f"done in {time.time() - t:.0f}s")
