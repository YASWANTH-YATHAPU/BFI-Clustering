import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(RES, "plots")
TABLES = os.path.join(RES, "tables")
for _d in (RES, PLOTS, TABLES):
    os.makedirs(_d, exist_ok=True)

K2_DEFAULT = 3      # see README: K2=9 breaks several datasets with the literal DS fusion
ZETA_DEFAULT = 0.1
THETA_SAMPLE = 1000  # reference-sample size of the practical (sampled-theta) configuration
SEEDS = (0, 1, 2, 3, 4)

# colours fixed per variant so every plot reads the same way
COLORS = {"original": "#4d4d4d", "control": "#b07aa1", "S1": "#59a14f", "S2": "#4e79a7",
          "S1+S2": "#e15759"}


def k1_of(n, frac=0.06):
    return max(10, int(frac * n))


def mean_std(df, keys, cols):
    """Mean and standard deviation over seeds; columns '<col>_mean', '<col>_std', plus 'n_runs'."""
    g = df.groupby(keys, sort=False)
    m = g[cols].mean().add_suffix("_mean")
    s = g[cols].std(ddof=1).fillna(0.0).add_suffix("_std")
    n = g.size().rename("n_runs")
    out = m.join(s).join(n)
    inter = [x for c in cols for x in (f"{c}_mean", f"{c}_std")]
    return out[inter + ["n_runs"]].reset_index()


def pm(m, s, nd=3):
    return "-" if (m is None or np.isnan(m)) else f"{m:.{nd}f} ± {s:.{nd}f}"


def write_markdown(df, path, floatfmt=3):
    """Small dependency-free markdown table writer."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w", encoding="utf8") as f:
        f.write("\n".join(lines) + "\n")
