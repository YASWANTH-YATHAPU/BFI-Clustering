"""Exp 1 - main comparison: Original BFI vs control vs S1 vs S2 vs S1+S2 on 13 datasets, 5 seeds.

Every accelerated variant is run twice:
  theta = 'exact'    exact theta (Eq. 9)  -> differences to the original are caused ONLY by the
                     neighbour search (isolation of Solutions 1 and 2);
  theta = 'sampled'  theta from a 1000-object reference sample (the practical configuration).
Seeds change the sampling noise of the synthetic data and the theta sample; the real datasets
are fixed, so for them only the theta-sampled variants vary across seeds.
Outputs: results/exp1_raw.csv, results/exp1_summary.csv, results/tables/exp1_*.md, results/plots/exp1_*.png
"""
import os, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common import (RES, PLOTS, TABLES, K2_DEFAULT, ZETA_DEFAULT, THETA_SAMPLE, SEEDS, COLORS,
                    k1_of, mean_std, write_markdown, pm)
from bfi.datasets import all_datasets
from bfi.bench import time_variant, measure_memory
from bfi.metrics import evaluate
from bfi.compare import compare_results
from bfi.neighbour_metrics import neighbour_quality
from bfi.variants import LABELS

FAST = ["control", "S1", "S2", "S1+S2"]
CONFIGS = [("original", "exact")] + [(v, "exact") for v in FAST] + [(v, "sampled") for v in FAST]
KIND = {"moons": "shape", "circles": "shape", "anisotropic": "shape", "blobs_separated": "separated",
        "blobs_overlap": "overlap", "varied_density": "density", "blobs_hd8": "high-dim",
        "blobs_hd32": "high-dim", "iris": "real", "wine": "real", "breast_cancer": "real",
        "digits6_pca10": "real/high-dim", "digits10_pca16": "real/high-dim"}
QUAL = ["ARI", "NMI", "AMI", "Re", "Ri", "EP", "ER", "ERI"]
CMP = ["vs_ref_label_agree", "vs_ref_label_ari_vs_ref", "vs_ref_betp_tv", "vs_ref_meta_set_agree",
       "vs_ref_bel_spearman", "vs_ref_center_shift"]
NBR = ["bel_exact_frac", "bel_err", "bel_rho", "delta_cand_recall", "delta_acc", "delta_relerr",
       "conv_cand_recall", "conv_acc", "delta_nn_agree_ref", "conv_jaccard_ref"]
CNT = ["n_delta_expanded", "n_delta_scan", "n_delta_est", "n_conv_expanded", "n_conv_scan",
       "n_conv_est", "mean_k_delta", "mean_k_conv", "k_base"]


def run(quick=False):
    seeds = SEEDS[:2] if quick else SEEDS
    n = 1500 if quick else 3000
    rows, mem_rows, store = [], [], {}
    for seed in seeds:
        for name, (X, y) in all_datasets(seed, n).items():
            c, N = len(set(y)), len(X)
            K1 = k1_of(N)
            base = dict(K1=K1, K2=K2_DEFAULT, zeta=ZETA_DEFAULT, seed=seed)
            ref, t_ref = None, None
            for v, th in CONFIGS:
                kw = dict(base, theta_sample=None if th == "exact" else THETA_SAMPLE)
                t, r = time_variant(v, X, c, repeats=3, **kw)
                row = {"dataset": name, "kind": KIND[name], "N": N, "dim": X.shape[1], "c": c,
                       "K1": K1, "seed": seed, "method": v, "theta": th, "time_s": t,
                       **evaluate(r, y)}
                if v == "original":
                    ref, t_ref = r, t
                else:
                    row["speedup"] = t_ref / t
                    row.update({f"vs_ref_{k}": val for k, val in compare_results(ref, r, X).items()})
                    row.update({k: r.info[k] for k in CNT})
                row.update(neighbour_quality(X, r, K1, K2_DEFAULT, k_base=r.info.get("k_base"),
                                             ref=ref, seed=seed))
                rows.append(row)
                if seed == 0:
                    store[(name, v, th)] = r
                    m = measure_memory(v, X, c, **kw)
                    mem_rows.append({"dataset": name, "method": v, "theta": th, "peak_mem_mb": m["peak_mb"]})
                print(f"seed={seed} {name:15s} {v:8s} {th:7s} t={t:.3f}s ARI={row['ARI']:.3f} "
                      f"agree={row.get('vs_ref_label_agree', 1.0):.3f}", flush=True)
    df = pd.DataFrame(rows)
    mem = pd.DataFrame(mem_rows)
    df = df.merge(mem, on=["dataset", "method", "theta"], how="left")
    df.to_csv(os.path.join(RES, "exp1_raw.csv"), index=False)
    cols = ["time_s", "speedup", "peak_mem_mb"] + QUAL + CMP + NBR + CNT
    summ = mean_std(df, ["dataset", "kind", "N", "dim", "c", "method", "theta"], cols)
    summ.to_csv(os.path.join(RES, "exp1_summary.csv"), index=False)
    tables(df, summ)
    plots(df, summ, store)
    return df


# ------------------------------------------------------------------------------------------------
def tables(df, summ):
    key = lambda r: f"{r['method']} ({r['theta']} θ)" if r["method"] != "original" else "original"
    summ = summ.assign(config=summ.apply(key, axis=1))
    # 1. ARI / NMI per dataset
    for met in ("ARI", "NMI"):
        t = summ.pivot_table(index="dataset", columns="config", values=f"{met}_mean", sort=False)
        s = summ.pivot_table(index="dataset", columns="config", values=f"{met}_std", sort=False)
        out = pd.DataFrame({c: [pm(t.loc[d, c], s.loc[d, c]) for d in t.index] for c in t.columns}, index=t.index)
        write_markdown(out.reset_index(), os.path.join(TABLES, f"exp1_{met.lower()}.md"))
    # 2. efficiency per dataset (time mean ± std, speed-up, memory)
    eff = summ[["dataset", "N", "dim", "config", "time_s_mean", "time_s_std", "speedup_mean",
                "peak_mem_mb_mean"]].copy()
    eff["time_s"] = [pm(a, b) for a, b in zip(eff.time_s_mean, eff.time_s_std)]
    eff = eff[["dataset", "N", "dim", "config", "time_s", "speedup_mean", "peak_mem_mb_mean"]]
    eff.columns = ["dataset", "N", "dim", "config", "time (s)", "speed-up", "peak mem (MB)"]
    write_markdown(eff, os.path.join(TABLES, "exp1_efficiency.md"), 2)
    # 3. averaged over datasets: agreement, neighbour quality, evidence metrics
    per = df.assign(config=df.apply(key, axis=1))
    agg_cols = ["ARI", "NMI", "Re", "Ri", "EP", "ER", "ERI", "speedup", "vs_ref_label_agree",
                "vs_ref_betp_tv", "vs_ref_meta_set_agree"] + NBR
    g = per.groupby(["dataset", "config"], sort=False)[agg_cols].mean().reset_index()   # seed-mean
    avg = g.groupby("config", sort=False)[agg_cols].agg(["mean", "std"])
    out = pd.DataFrame({c: [pm(avg.loc[i, (c, "mean")], avg.loc[i, (c, "std")]) for i in avg.index]
                        for c in agg_cols}, index=avg.index)
    write_markdown(out.reset_index(), os.path.join(TABLES, "exp1_overall.md"))
    # 4. failures / quality drops: any dataset where an accelerated config loses > 0.01 ARI
    o = summ[summ.method == "original"].set_index("dataset")["ARI_mean"]
    f = summ[summ.method != "original"].copy()
    f["ARI_drop_vs_original"] = o.loc[f.dataset].values - f["ARI_mean"].values
    f = f[f.ARI_drop_vs_original > 0.01][["dataset", "config", "ARI_mean", "ARI_std",
                                          "ARI_drop_vs_original", "vs_ref_label_agree_mean"]]
    write_markdown(f.sort_values(["config", "ARI_drop_vs_original"], ascending=[True, False]),
                   os.path.join(TABLES, "exp1_quality_drops.md"))


def _bars(ax, summ, metric, methods, theta, log=False):
    ds = list(dict.fromkeys(summ.dataset))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        th = "exact" if m == "original" else theta
        d = summ[(summ.method == m) & (summ.theta == th)].set_index("dataset").reindex(ds)
        ax.bar(np.arange(len(ds)) + (j - (len(methods) - 1) / 2) * w, d[f"{metric}_mean"],
               w, yerr=d.get(f"{metric}_std"), color=COLORS[m], label=LABELS[m], capsize=1.5,
               error_kw={"lw": 0.6})
    ax.set_xticks(np.arange(len(ds)))
    ax.set_xticklabels(ds, rotation=40, ha="right", fontsize=7)
    if log:
        ax.set_yscale("log")


def plots(df, summ, store):
    methods = ["original"] + FAST
    for theta in ("exact", "sampled"):
        fig, ax = plt.subplots(2, 1, figsize=(13, 7.5))
        for a, met in zip(ax, ("ARI", "NMI")):
            _bars(a, summ, met, methods, theta)
            a.set_ylabel(f"{met} (mean ± std, 5 seeds)")
        ax[0].set_title(f"Clustering quality vs ground truth  (accelerated variants with {theta} θ)")
        ax[0].legend(fontsize=7, ncol=5, loc="lower left")
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS, f"exp1_ari_nmi_theta_{theta}.png"), dpi=140); plt.close()

    fig, ax = plt.subplots(2, 1, figsize=(13, 7.5))
    _bars(ax[0], summ, "time_s", methods, "sampled", log=True)
    ax[0].set_ylabel("time (s), log"); ax[0].set_title("Runtime per dataset (single thread; accelerated variants: sampled θ)")
    _bars(ax[1], summ, "time_s", methods, "exact", log=True)
    ax[1].set_ylabel("time (s), log"); ax[1].set_title("Runtime per dataset (accelerated variants: exact θ)")
    ax[0].legend(fontsize=7, ncol=5)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "exp1_runtime.png"), dpi=140); plt.close()

    fig, ax = plt.subplots(figsize=(13, 4))
    _bars(ax, summ.assign(peak_mem_mb_std=0.0), "peak_mem_mb", methods, "sampled", log=True)
    ax.set_ylabel("peak additional memory (MB), log"); ax.set_title("Peak memory per dataset (fresh process, seed 0; accelerated: sampled θ)")
    ax.legend(fontsize=7, ncol=5)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "exp1_memory.png"), dpi=140); plt.close()

    # neighbour-search quality, averaged over datasets (theta exact isolates the neighbour search)
    mets = ["bel_rho", "delta_cand_recall", "delta_acc", "conv_cand_recall", "conv_acc",
            "delta_nn_agree_ref", "conv_jaccard_ref"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    for a, theta in zip(ax, ("exact", "sampled")):
        d = df[(df.method != "original") & (df.theta == theta)]
        g = d.groupby(["method", "dataset"])[mets].mean().groupby("method")[mets]
        mu, sd = g.mean().reindex(FAST), g.std().reindex(FAST)
        w = 0.8 / len(FAST)
        for j, m in enumerate(FAST):
            a.bar(np.arange(len(mets)) + (j - 1.5) * w, mu.loc[m], w, yerr=sd.loc[m], color=COLORS[m],
                  label=LABELS[m], capsize=1.5, error_kw={"lw": 0.6})
        a.set_xticks(np.arange(len(mets))); a.set_xticklabels(mets, rotation=30, ha="right", fontsize=7)
        a.set_ylim(0, 1.05); a.set_title(f"Neighbour-search quality ({theta} θ), mean ± std over 13 datasets", fontsize=9)
    ax[0].legend(fontsize=7, loc="lower left")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "exp1_neighbour_accuracy.png"), dpi=140); plt.close()

    fig, ax = plt.subplots(2, 1, figsize=(13, 7))
    for a, (met, ttl) in zip(ax, (("vs_ref_label_agree", "label agreement with original BFI"),
                                  ("vs_ref_betp_tv", "mean total-variation distance of BetP to original"))):
        _bars(a, summ[summ.method != "original"], met, FAST, "sampled")
        a.set_ylabel(ttl, fontsize=8)
    ax[0].set_title("Evidence preservation (accelerated variants with sampled θ)"); ax[0].legend(fontsize=7, ncol=4)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "exp1_agreement.png"), dpi=140); plt.close()

    names = [n for n in dict.fromkeys(df.dataset) if (df[df.dataset == n].dim.iloc[0] == 2)]
    Xs = all_datasets(0, int(df.N[df.dataset == names[0]].iloc[0]))
    fig, ax = plt.subplots(len(names), 2, figsize=(7, 3.1 * len(names)))
    for i, n in enumerate(names):
        for j, (m, th, ttl) in enumerate((("original", "exact", "Original BFI"),
                                          ("S1+S2", "sampled", "S1+S2 (sampled θ)"))):
            r, X, a = store[(n, m, th)], Xs[n][0], ax[i, j]
            p = ~r.imprecise
            a.scatter(X[p, 0], X[p, 1], c=r.labels[p], s=2, cmap="tab10", vmin=0, vmax=9)
            a.scatter(X[r.imprecise, 0], X[r.imprecise, 1], c="k", s=7, marker="x", label="meta-cluster")
            a.scatter(X[r.centers, 0], X[r.centers, 1], c="r", s=60, marker="*", edgecolor="w", label="centre")
            a.set_title(f"{n} - {ttl}", fontsize=8); a.set_xticks([]); a.set_yticks([])
    ax[0, 0].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "exp1_clusters.png"), dpi=120); plt.close()


if __name__ == "__main__":
    t = time.time(); run(); print(f"done in {time.time() - t:.0f}s")
