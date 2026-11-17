"""Exp 4 - sensitivity to the BFI parameters (K1 fraction, K2, zeta): does the proposed method
(S1+S2, sampled theta) track the original BFI across the parameter space?  The control variant is
included to show how an un-refined acceleration behaves.  3 datasets, 3 seeds, N = 3000."""
import os, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common import (RES, PLOTS, TABLES, K2_DEFAULT, ZETA_DEFAULT, THETA_SAMPLE, COLORS, k1_of,
                    mean_std, write_markdown)
from bfi import datasets as D
from bfi.variants import run_variant
from bfi.metrics import evaluate
from bfi.compare import compare_results

GRIDS = {"K1_frac": [0.02, 0.04, 0.06, 0.10, 0.15], "K2": [1, 3, 5, 9, 15],
         "zeta": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]}
METHODS = [("original", None), ("control", THETA_SAMPLE), ("S1+S2", THETA_SAMPLE)]


def run(quick=False):
    N = 1500 if quick else 3000
    seeds = (0,) if quick else (0, 1, 2)
    rows = []
    for seed in seeds:
        S, H = D.synthetic(N, seed), D.high_dim(N, seed)
        data = {"blobs_overlap": S["blobs_overlap"], "varied_density": S["varied_density"],
                "blobs_hd8": H["blobs_hd8"], "iris": D.real()["iris"]}
        for name, (X, y) in data.items():
            c = len(set(y))
            for par, vals in GRIDS.items():
                for v in vals:
                    kw = dict(K1=k1_of(len(X), v if par == "K1_frac" else 0.06),
                              K2=int(v) if par == "K2" else K2_DEFAULT,
                              zeta=v if par == "zeta" else ZETA_DEFAULT, seed=seed)
                    ref = None
                    for m, ts in METHODS:
                        r = run_variant(m, X, c, theta_sample=ts, **kw)
                        row = {"dataset": name, "seed": seed, "param": par, "value": v, "method": m,
                               **evaluate(r, y)}
                        if m == "original":
                            ref = r
                        else:
                            row["vs_ref_label_agree"] = compare_results(ref, r, X)["label_agree"]
                        rows.append(row)
            print(f"seed={seed} {name} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "exp4_raw.csv"), index=False)
    summ = mean_std(df, ["param", "value", "method"], ["ARI", "NMI", "Re", "Ri", "vs_ref_label_agree"])
    summ.to_csv(os.path.join(RES, "exp4_summary.csv"), index=False)
    write_markdown(summ[["param", "value", "method", "ARI_mean", "ARI_std", "Re_mean", "Ri_mean",
                         "vs_ref_label_agree_mean"]], os.path.join(TABLES, "exp4_sensitivity.md"))
    per = mean_std(df, ["dataset", "param", "value", "method"], ["ARI", "Re", "Ri", "vs_ref_label_agree"])
    fig, ax = plt.subplots(4, 3, figsize=(13, 12))
    for j, par in enumerate(GRIDS):
        for i, met in enumerate(["ARI", "Re", "Ri", "vs_ref_label_agree"]):
            for m, _ in METHODS:
                if met == "vs_ref_label_agree" and m == "original":
                    continue
                for k, name in enumerate(dict.fromkeys(per.dataset)):
                    d = per[(per.param == par) & (per.method == m) & (per.dataset == name)]
                    ax[i, j].plot(d.value, d[f"{met}_mean"], ["o-", "s--", "^:", "d-."][k], ms=3,
                                  color=COLORS[m], alpha=0.85, label=f"{m} / {name}")
            ax[i, j].set(xlabel=par, ylabel=met)
    ax[0, 0].legend(fontsize=5, ncol=2)
    plt.suptitle("Parameter sensitivity (mean over 3 seeds); colour = method, marker = dataset", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "exp4_sensitivity.png"), dpi=130); plt.close()
    print(summ.pivot_table(index=["param", "value"], columns="method", values="ARI_mean").round(3))
    return df


if __name__ == "__main__":
    t = time.time(); run(); print(f"done in {time.time() - t:.0f}s")
