"""Exp 3 - ablation of Solution 1 / Solution 2 and of their parameters.

Datasets: varied_density, blobs_overlap, moons (2-D) and blobs_hd8 (8-D), N = 4000, 3 seeds.
theta is EXACT in every run except the theta sweep, so that the only source of deviation from the
original BFI is the neighbour search.  Reference = original BFI on the same data.

Groups (variant x parameter):
  factorial        control / S1 / S2 / S1+S2 at the defaults
  k_cand           fixed list size of control and S2 (and Bel head size of control)   16..128
  bel_head         Bel head size k_cand of S1 (estimated Bel)                           16..128
  k_min            initial adaptive list size of S1 and S1+S2                            4..64
  k_max            adaptive list budget of S1 and S1+S2                                32..1024
  bel_refine       Bel refinement policy of S1+S2   none / flagged / centres / all
  delta_estimate   estimator used without refinement (control, S1)
  tail             tail model of the estimated Bel (control, S1)  extrapolate / truncate
  theta_sample     S1+S2 with theta from s = 100..3000 samples, or exact
"""
import os, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common import RES, PLOTS, TABLES, K2_DEFAULT, ZETA_DEFAULT, COLORS, k1_of, mean_std, write_markdown
from bfi import datasets as D
from bfi.bench import time_variant
from bfi.metrics import evaluate
from bfi.compare import compare_results
from bfi.neighbour_metrics import neighbour_quality

GROUPS = {
    "factorial": [(v, None, {}) for v in ("control", "S1", "S2", "S1+S2")],
    "k_cand": [(v, k, {"k_cand": k}) for v in ("control", "S2") for k in (16, 32, 64, 128)],
    "bel_head": [("S1", k, {"k_cand": k}) for k in (16, 32, 64, 128)],
    "k_min": [(v, k, {"k_min": k}) for v in ("S1", "S1+S2") for k in (4, 8, 16, 32, 64)],
    "k_max": [(v, k, {"k_max": k}) for v in ("S1", "S1+S2") for k in (32, 64, 128, 256, 512, 1024)],
    "bel_refine": [("S1+S2", p, {"bel_refine": p}) for p in ("none", "flagged", "centres", "all")],
    "delta_estimate": [(v, e, {"delta_estimate": e}) for v in ("control", "S1")
                       for e in ("landmark", "lower_bound", "geometric")],
    "tail": [(v, e, {"tail": e}) for v in ("control", "S1") for e in ("extrapolate", "truncate")],
    "theta_sample": [("S1+S2", s, {"theta_sample": s}) for s in (100, 300, 1000, 3000, "exact")],
}
METRICS = ["time_s", "speedup", "ARI", "NMI", "vs_ref_label_agree", "vs_ref_betp_tv",
           "vs_ref_meta_set_agree", "bel_rho", "delta_acc", "conv_acc", "delta_cand_recall",
           "conv_cand_recall", "n_delta_scan", "n_conv_scan", "n_delta_est", "n_conv_est",
           "n_conv_expanded", "mean_k_conv", "mean_k_delta", "n_bel_exact"]


def data(N, seed):
    S, H = D.synthetic(N, seed), D.high_dim(N, seed)
    return {"varied_density": S["varied_density"], "blobs_overlap": S["blobs_overlap"],
            "moons": S["moons"], "blobs_hd8": H["blobs_hd8"]}


def run(quick=False):
    N = 2000 if quick else 4000
    seeds = (0,) if quick else (0, 1, 2)
    rows = []
    for seed in seeds:
        for name, (X, y) in data(N, seed).items():
            c, K1 = len(set(y)), k1_of(N)
            base = dict(K1=K1, K2=K2_DEFAULT, zeta=ZETA_DEFAULT, seed=seed)
            t_ref, ref = time_variant("original", X, c, repeats=3, **base, theta_sample=None)
            rows.append({"dataset": name, "seed": seed, "group": "reference", "method": "original",
                         "value": "", "time_s": t_ref, **evaluate(ref, y)})
            for grp, items in GROUPS.items():
                for v, val, kw in items:
                    kw = dict(kw)
                    ts = kw.pop("theta_sample", None)
                    ts = None if ts == "exact" else ts
                    t, r = time_variant(v, X, c, repeats=2, **base, theta_sample=ts, **kw)
                    row = {"dataset": name, "seed": seed, "group": grp, "method": v,
                           "value": "default" if val is None else str(val), "time_s": t,
                           "speedup": t_ref / t, **evaluate(r, y),
                           **{f"vs_ref_{k}": x for k, x in compare_results(ref, r, X).items()},
                           **{k: r.info[k] for k in ("n_delta_scan", "n_conv_scan", "n_delta_est",
                              "n_conv_est", "n_conv_expanded", "mean_k_conv", "mean_k_delta", "n_bel_exact")}}
                    row.update(neighbour_quality(X, r, K1, K2_DEFAULT, k_base=r.info["k_base"],
                                                 ref=ref, n_eval=1000, seed=seed))
                    rows.append(row)
                    print(f"seed={seed} {name:15s} {grp:14s} {v:6s} {row['value']:>11s} t={t:.3f}s "
                          f"x{row['speedup']:.2f} agree={row['vs_ref_label_agree']:.3f} "
                          f"tv={row['vs_ref_betp_tv']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "exp3_raw.csv"), index=False)
    summ = mean_std(df[df.group != "reference"], ["group", "method", "value"],
                    [m for m in METRICS if m in df])
    summ.to_csv(os.path.join(RES, "exp3_summary.csv"), index=False)
    per_ds = mean_std(df[df.group != "reference"], ["group", "dataset", "method", "value"],
                      ["speedup", "vs_ref_label_agree", "vs_ref_betp_tv", "ARI"])
    per_ds.to_csv(os.path.join(RES, "exp3_summary_per_dataset.csv"), index=False)
    t = summ[["group", "method", "value", "speedup_mean", "speedup_std", "vs_ref_label_agree_mean",
              "vs_ref_label_agree_std", "vs_ref_betp_tv_mean", "delta_acc_mean", "conv_acc_mean",
              "bel_rho_mean", "n_conv_scan_mean", "n_conv_est_mean", "mean_k_conv_mean"]]
    write_markdown(t, os.path.join(TABLES, "exp3_ablation.md"))
    plot(summ)
    return df


def plot(summ):
    show = ["speedup", "vs_ref_label_agree", "vs_ref_betp_tv", "conv_acc", "mean_k_conv"]
    titles = {"speedup": "speed-up vs original", "vs_ref_label_agree": "label agreement",
              "vs_ref_betp_tv": "BetP total variation", "conv_acc": "conv. neighbour accuracy",
              "mean_k_conv": "mean conv. list size"}
    groups = [g for g in GROUPS if g != "factorial"]
    fig, ax = plt.subplots(len(groups), len(show), figsize=(3.2 * len(show), 2.5 * len(groups)))
    for i, g in enumerate(groups):
        d = summ[summ.group == g]
        for m in dict.fromkeys(d.method):
            dm = d[d.method == m]
            x = np.arange(len(dm))
            for j, met in enumerate(show):
                ax[i, j].errorbar(x, dm[f"{met}_mean"], yerr=dm[f"{met}_std"], fmt="o-", ms=3,
                                  color=COLORS[m], label=m, capsize=2)
                ax[i, j].set_xticks(x)
                ax[i, j].set_xticklabels(dm.value, fontsize=6, rotation=30)
        ax[i, 0].set_ylabel(g, fontsize=9, fontweight="bold")
        ax[i, 0].legend(fontsize=6)
    for j, met in enumerate(show):
        ax[0, j].set_title(titles[met], fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "exp3_ablation_parameters.png"), dpi=130); plt.close()

    d = summ[summ.group == "factorial"].set_index("method").reindex(["control", "S1", "S2", "S1+S2"])
    mets = ["speedup", "vs_ref_label_agree", "vs_ref_meta_set_agree", "bel_rho", "delta_acc", "conv_acc"]
    fig, ax = plt.subplots(1, len(mets), figsize=(3 * len(mets), 3.2))
    for a, met in zip(ax, mets):
        a.bar(d.index, d[f"{met}_mean"], yerr=d[f"{met}_std"], color=[COLORS[m] for m in d.index], capsize=3)
        a.set_title(met, fontsize=9); a.tick_params(axis="x", labelsize=7)
    plt.suptitle("2x2 factorial (exact θ, 4 datasets x 3 seeds, N = 4000)", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "exp3_factorial.png"), dpi=140); plt.close()


if __name__ == "__main__":
    t = time.time(); run(); print(f"done in {time.time() - t:.0f}s")
