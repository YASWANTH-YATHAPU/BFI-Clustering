"""Generate the LaTeX result tables of the paper from results/exp1_raw.csv and exp1_summary.csv.

    python paper/make_tables.py        (run from the project root)

Writes paper/tables/tab_synthetic.tex, tab_real.tex and tab_efficiency.tex.  Every number is a
mean over the seeds recorded in the CSV files; nothing is typed by hand.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "tables")
os.makedirs(OUT, exist_ok=True)
raw = pd.read_csv(os.path.join(ROOT, "results", "exp1_raw.csv"))
summ = pd.read_csv(os.path.join(ROOT, "results", "exp1_summary.csv"))

METRICS = ["Re", "Ri", "P", "R", "RI", "EP", "ER", "ERI", "ARI", "AMI", "NMI"]
HEAD = ["$R_e$", "$R_i$", "$P$", "$R$", "$RI$", "$EP$", "$ER$", "$ERI$", "ARI", "AMI", "NMI"]
ROWS = [("original", "exact", "Original BFI"), ("control", "exact", "Control"),
        ("S1", "exact", "S1"), ("S2", "exact", "S2"),
        ("S1+S2", "exact", "S1+S2"), ("S1+S2", "sampled", r"S1+S2$^{\dagger}$")]
NAMES = {"moons": "moons", "circles": "circles", "anisotropic": "aniso.",
         "blobs_separated": "blobs\\_sep.", "blobs_overlap": "blobs\\_ovl.",
         "varied_density": "varied", "blobs_hd8": "hd8", "blobs_hd32": "hd32",
         "iris": "iris", "wine": "wine", "breast_cancer": "breast",
         "digits6_pca10": "digits6", "digits10_pca16": "digits10"}

means = raw.groupby(["dataset", "method", "theta"])[METRICS].mean()


def metric_table(datasets, caption, label):
    L = [r"\begin{table*}[!t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
         r"\scriptsize", r"\setlength{\tabcolsep}{4.2pt}",
         r"\begin{tabular}{ll" + "c" * len(METRICS) + "}", r"\toprule",
         "Datasets & Methods & " + " & ".join(HEAD) + r" \\", r"\midrule"]
    for di, ds in enumerate(datasets):
        block = np.array([[means.loc[(ds, m, t), k] for k in METRICS] for m, t, _ in ROWS])
        r4 = np.round(block, 4)
        best = np.where(np.array([k == "Re" for k in METRICS]), r4.min(0), r4.max(0))
        for ri, (m, t, lab) in enumerate(ROWS):
            cells = []
            for ci, k in enumerate(METRICS):
                s = f"{r4[ri, ci]:.4f}"
                if k != "Ri" and r4[ri, ci] == best[ci]:
                    s = r"\textbf{" + s + "}"
                cells.append(s)
            first = (r"\multirow{" + str(len(ROWS)) + "}{*}{" + NAMES[ds] + "}") if ri == 0 else ""
            L.append(f"{first} & {lab} & " + " & ".join(cells) + r" \\")
        if di < len(datasets) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\vspace{2pt}",
          r"\parbox{\textwidth}{\footnotesize Mean over seeds (5 for synthetic data; real data are deterministic). "
          r"All variants with exact $\theta$ except S1+S2$^{\dagger}$ ($\theta$ sampled from 1000 objects). "
          r"Bold: best value per dataset (lowest for $R_e$); $R_i$ is not ranked.}",
          r"\end{table*}"]
    return "\n".join(L)


syn = ["moons", "circles", "anisotropic", "blobs_separated", "blobs_overlap", "varied_density",
       "blobs_hd8", "blobs_hd32"]
real = ["iris", "wine", "breast_cancer", "digits6_pca10", "digits10_pca16"]
open(os.path.join(OUT, "tab_synthetic.tex"), "w").write(
    metric_table(syn, "Clustering results of the original BFI and the accelerated variants on the eight synthetic datasets",
                 "tab:synthetic"))
open(os.path.join(OUT, "tab_real.tex"), "w").write(
    metric_table(real, "Clustering results of the original BFI and the accelerated variants on the five real-world datasets",
                 "tab:real"))

# ---- efficiency per dataset ------------------------------------------------------------------
S = summ.set_index(["dataset", "method", "theta"])
L = [r"\begin{table}[!t]", r"\centering",
     r"\caption{Runtime, speed-up and peak memory per dataset (mean over seeds)}", r"\label{tab:efficiency}",
     r"\scriptsize", r"\setlength{\tabcolsep}{2.6pt}",
     r"\begin{tabular}{lrrcccccc}", r"\toprule",
     r" & & & \multicolumn{2}{c}{Time (s)} & \multicolumn{2}{c}{Speed-up} & \multicolumn{2}{c}{Memory (MB)} \\",
     r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
     r"Dataset & $N$ & $d$ & Orig. & S1+S2$^{\dagger}$ & exact & $\dagger$ & Orig. & S1+S2$^{\dagger}$ \\",
     r"\midrule"]
for ds in syn + real:
    o = S.loc[(ds, "original", "exact")]
    e = S.loc[(ds, "S1+S2", "exact")]
    p = S.loc[(ds, "S1+S2", "sampled")]
    L.append(f"{NAMES[ds]} & {int(o.N)} & {int(o.dim)} & {o.time_s_mean:.3f} & {p.time_s_mean:.3f} & "
             f"{e.speedup_mean:.2f} & {p.speedup_mean:.2f} & {o.peak_mem_mb_mean:.0f} & {p.peak_mem_mb_mean:.0f} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\vspace{2pt}",
      r"\parbox{\columnwidth}{\footnotesize $\dagger$: $\theta$ sampled from 1000 objects; ``exact'': S1+S2 with exact $\theta$. "
      r"Speed-up $=t_{\mathrm{orig}}/t$; memory = additional peak resident memory in a fresh process.}",
      r"\end{table}"]
open(os.path.join(OUT, "tab_efficiency.tex"), "w").write("\n".join(L))
print("written to", OUT)
