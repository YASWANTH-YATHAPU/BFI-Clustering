"""Exp 2 - scalability in N (4-component 2-D Gaussian mixture, K1 = 0.06 N, K2 = 3, zeta = 0.1).

Original BFI up to N = 6000 (its N x N matrices need ~0.8 GB there); accelerated variants up to
N = 50000.  Accelerated variants use sampled theta (s = 1000, the practical setting); S1+S2 is
additionally run with exact theta up to N = 20000 to show the cost of exact Eq. (9).
3 seeds (data sampling + theta sample); time = median of 3 runs (1 run for N > 6000);
memory = one fresh-process run (seed 0).
"""
import os, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common import (RES, PLOTS, TABLES, K2_DEFAULT, ZETA_DEFAULT, THETA_SAMPLE, COLORS, k1_of,
                    mean_std, write_markdown)
from bfi.datasets import scalable
from bfi.bench import time_variant, measure_memory
from bfi.metrics import evaluate
from bfi.compare import compare_results
from bfi.neighbour_metrics import neighbour_quality

FAST = ["control", "S1", "S2", "S1+S2"]
PARTS = ["t_theta", "t_index", "t_bel", "t_delta", "t_convergence"]


def configs(N, both, exact_max):
    out = [("original", "exact")] if N in both else []
    out += [(v, "sampled") for v in FAST]
    if N <= exact_max:
        out.append(("S1+S2", "exact"))
    return out


def run(quick=False):
    both = [1000, 2000] if quick else [1000, 2000, 4000, 6000]
    # the original needs ~ 3 N x N float64 arrays (D, sorted copy, temporaries); only run it where
    # that fits comfortably in the RAM that is free right now, otherwise its timing would be
    # distorted by swapping.  The cut-off is printed and stored in the results.
    import psutil
    free = psutil.virtual_memory().available
    skipped = [N for N in both if 3 * 8 * N * N > 0.8 * free]
    if skipped:
        print(f"free RAM {free / 2**30:.2f} GB -> original BFI skipped for N = {skipped}", flush=True)
    both_orig = [N for N in both if N not in skipped]
    fast_only = [5000] if quick else [10000, 20000, 50000]
    exact_max = 5000 if quick else 20000
    seeds = (0, 1) if quick else (0, 1, 2)
    rows = []
    for N in both + fast_only:
        for seed in seeds:
            X, y = scalable(N, seed=seed)
            c, K1 = 4, k1_of(N)
            ref, t_ref = None, None
            for v, th in configs(N, both_orig, exact_max):
                kw = dict(K1=K1, K2=K2_DEFAULT, zeta=ZETA_DEFAULT, seed=seed,
                          theta_sample=None if th == "exact" else THETA_SAMPLE)
                t, r = time_variant(v, X, c, repeats=3 if N <= 6000 else 1, **kw)
                row = {"N": N, "seed": seed, "method": v, "theta": th, "time_s": t, **evaluate(r, y)}
                if v == "original":
                    ref, t_ref = r, t
                else:
                    row.update({k: r.info[k] for k in PARTS + ["n_delta_expanded", "n_delta_scan",
                                "n_delta_est", "n_conv_expanded", "n_conv_scan", "n_conv_est",
                                "mean_k_delta", "mean_k_conv", "index_entries"]})
                    if ref is not None:
                        row["speedup"] = t_ref / t
                        row.update({f"vs_ref_{k}": val for k, val in compare_results(ref, r, X).items()})
                row.update(neighbour_quality(X, r, K1, K2_DEFAULT, k_base=r.info.get("k_base"),
                                             ref=ref, n_eval=1000, seed=seed))
                if seed == 0:
                    row["peak_mem_mb"] = measure_memory(v, X, c, **kw)["peak_mb"]
                rows.append(row)
                print(f"N={N:6d} seed={seed} {v:8s} {th:7s} t={t:7.3f}s mem={row.get('peak_mem_mb', np.nan):7.1f}MB "
                      f"ARI={row['ARI']:.3f} agree={row.get('vs_ref_label_agree', np.nan):.3f}", flush=True)
    df = pd.DataFrame(rows)
    df["free_ram_gb_at_start"] = free / 2 ** 30
    df.to_csv(os.path.join(RES, "exp2_raw.csv"), index=False)
    cols = ["time_s", "speedup", "peak_mem_mb", "ARI", "NMI", "vs_ref_label_agree", "vs_ref_betp_tv",
            "delta_acc", "conv_acc", "bel_rho"] + PARTS + ["n_delta_scan", "n_conv_scan",
            "n_conv_expanded", "n_conv_est", "n_delta_est"]
    summ = mean_std(df, ["N", "method", "theta"], [c for c in cols if c in df])
    summ.to_csv(os.path.join(RES, "exp2_summary.csv"), index=False)
    t = summ[["N", "method", "theta", "time_s_mean", "time_s_std", "speedup_mean", "peak_mem_mb_mean",
              "ARI_mean", "vs_ref_label_agree_mean", "n_conv_scan_mean", "n_conv_est_mean"]]
    write_markdown(t, os.path.join(TABLES, "exp2_scalability.md"))
    plot(df, summ)
    return df


def plot(df, summ):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    lines = [("original", "exact", "o-", "Original BFI"), ("control", "sampled", "v--", "control"),
             ("S1", "sampled", "^-", "S1"), ("S2", "sampled", "s-", "S2"),
             ("S1+S2", "sampled", "D-", "S1+S2 (sampled θ)"), ("S1+S2", "exact", "D:", "S1+S2 (exact θ)")]
    for m, th, mk, lab in lines:
        d = summ[(summ.method == m) & (summ.theta == th)]
        if not len(d):
            continue
        ax[0, 0].errorbar(d.N, d.time_s_mean, yerr=d.time_s_std, fmt=mk, color=COLORS[m], label=lab, ms=4, capsize=2)
        dm = d.dropna(subset=["peak_mem_mb_mean"])
        ax[0, 1].plot(dm.N, dm.peak_mem_mb_mean.clip(lower=0.1), mk, color=COLORS[m], label=lab, ms=4)
        if m != "original" and "speedup_mean" in d:
            ds = d.dropna(subset=["speedup_mean"])
            ax[1, 0].errorbar(ds.N, ds.speedup_mean, yerr=ds.speedup_std, fmt=mk, color=COLORS[m], label=lab, ms=4, capsize=2)
    ax[0, 0].set(xscale="log", yscale="log", xlabel="N", ylabel="time (s)", title="Runtime (mean ± std, 3 seeds)")
    ax[0, 1].set(xscale="log", yscale="log", xlabel="N", ylabel="peak additional memory (MB)", title="Peak memory (fresh process)")
    ax[1, 0].axhline(1, color="k", lw=0.6)
    ax[1, 0].set(xscale="log", xlabel="N", ylabel="speed-up over original", title="Speed-up (N where the original fits in memory)")
    d = summ[(summ.method == "S1+S2") & (summ.theta == "sampled")]
    bottom = np.zeros(len(d))
    for p in PARTS:
        ax[1, 1].bar(d.N.astype(str), d[f"{p}_mean"], bottom=bottom, label=p.replace("t_", ""))
        bottom += d[f"{p}_mean"].values
    ax[1, 1].set(xlabel="N", ylabel="time (s)", title="S1+S2 (sampled θ): time breakdown")
    for a in ax.flat:
        a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "exp2_scalability.png"), dpi=140); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for m in ("S2", "S1+S2"):
        d = summ[(summ.method == m) & (summ.theta == "sampled")]
        ax[0].plot(d.N, d.n_conv_scan_mean + d.n_delta_scan_mean, "o-", color=COLORS[m], label=f"{m}: exact scans")
        if m == "S1+S2":
            ax[0].plot(d.N, d.n_conv_expanded_mean, "^--", color=COLORS[m], label=f"{m}: index expansions")
    for m in ("control", "S1"):
        d = summ[(summ.method == m) & (summ.theta == "sampled")]
        ax[1].plot(d.N, (d.n_conv_est_mean + d.n_delta_est_mean) / d.N, "o-", color=COLORS[m], label=m)
    ax[0].set(xscale="log", yscale="log", xlabel="N", ylabel="count", title="Exact O(N) scans needed (refinement work)")
    ax[1].set(xscale="log", xlabel="N", ylabel="fraction of objects", title="Queries answered by an estimate (no refinement)")
    for a in ax:
        a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "exp2_refinement_work.png"), dpi=140); plt.close()


if __name__ == "__main__":
    t = time.time(); run(); print(f"done in {time.time() - t:.0f}s")
