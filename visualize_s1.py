"""Live, step-by-step visualisation of Solution 1 - Adaptive Candidate Neighbour Retrieval.

    python visualize_s1.py                      # animated window (default: S1, varied_density, N=1500)
    python visualize_s1.py --save               # no window; writes results/plots/s1_live.gif
    python visualize_s1.py --dataset moons --n 2000 --speed 2
    python visualize_s1.py --variant S1+S2      # same retrieval + Solution 2's exact fallback

Everything shown is computed by the project code (bfi.neighbors.Searcher, the certificate of
bfi.candidates and the per-object counters recorded by bfi.fast) - nothing is simulated.

Stages
  1. the data                          4. delta for ALL objects: how big each list had to grow
  2. Bel of every object (Eq. 10)      5. cluster centres
  3. delta search for 3 example        6. evidential convergence in Bel order: K2 nearest allocated
     objects: list 16 -> 32 -> ...        objects found from the short lists (red ring = list grown)
     until the certificate holds       7. summary vs the original BFI (full N x N matrix)
"""
import argparse
import os
import time
import numpy as np
import matplotlib

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dataset", default="varied_density",
                choices=["varied_density", "moons", "blobs_separated", "blobs_overlap", "anisotropic", "circles",
                         "iris", "wine", "breast_cancer", "digits6_pca10", "digits10_pca16"])
ap.add_argument("--n", type=int, default=1500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--variant", default="S1", choices=["S1", "S1+S2"])
ap.add_argument("--speed", type=float, default=1.0, help="2 = twice as fast")
ap.add_argument("--save", action="store_true", help="write results/plots/s1_live.gif instead of showing a window")
args = ap.parse_args()
if args.save:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrowPatch

from bfi.datasets import synthetic, real
from bfi.variants import run_variant, DEFAULTS
from bfi.neighbors import Searcher
from bfi.candidates import _k_schedule, first_higher_in_lists
from bfi.compare import compare_results
from bfi.fast import SRC

# ----------------------------------------------------------------------------- run the algorithms
REAL_SETS = {"iris", "wine", "breast_cancer", "digits6_pca10", "digits10_pca16"}
if args.dataset in REAL_SETS:
    from sklearn.decomposition import PCA
    XA, y = real()[args.dataset]                       # the algorithm uses ALL features
    X = PCA(2, random_state=0).fit_transform(XA)       # display only: 2-D PCA projection
    X = (X - X.min(0)) / np.ptp(X, 0)
else:
    XA, y = synthetic(args.n, seed=args.seed)[args.dataset]
    X = XA
PROJ = XA.shape[1] > 2                                  # circles are not drawn in a projection
PNOTE = (f"\n{args.dataset}: {XA.shape[1]} features, shown as a 2-D PCA projection "
         f"(all distances computed in {XA.shape[1]}-D)") if PROJ else ""
N, c = len(X), len(np.unique(y))
K1, K2, zeta = max(10, int(0.06 * N)), 3, 0.1
print(f"{args.dataset}: N={N}, c={c}, K1={K1}, K2={K2}  - running original BFI and {args.variant} ...")
t = time.perf_counter(); ref = run_variant("original", XA, c, K1, K2, zeta); t_orig = time.perf_counter() - t
t = time.perf_counter(); res = run_variant(args.variant, XA, c, K1, K2, zeta, theta_sample=None)
t_fast = time.perf_counter() - t
agree = compare_results(ref, res, XA)["label_agree"]
print(f"original {t_orig:.2f}s   {args.variant} {t_fast:.2f}s   label agreement {agree:.3f}")

D = res.diag
kb = int(res.info.get("k_base", DEFAULTS["k_min"]))
kmax = DEFAULTS["k_max"]
ks_all = [kb] + _k_schedule(kb, kmax, N)
order = np.argsort(-res.bel, kind="stable")
rank = np.empty(N, dtype=np.int64); rank[order] = np.arange(N)
sr = Searcher(XA)
labels = res.labels
CMAP = plt.get_cmap("tab10")

# example objects for the delta demo: a core object, a grown list, the hardest grown list
dsrc, dk = D["delta_src"], D["delta_k"]
cand_core = np.flatnonzero(dsrc == SRC["base"])
core = cand_core[np.argsort(res.bel[cand_core])[int(0.8 * len(cand_core))]]
grown = np.flatnonzero(dsrc == SRC["expanded"])
demo = [core]
if len(grown):
    hardest = grown[np.argmax(dk[grown])]
    mid = grown[dk[grown] == dk[grown].min()]
    if len(mid) and mid[0] != hardest:
        demo.append(mid[len(mid) // 2])
    demo.append(hardest)
hard_left = np.flatnonzero(np.isin(dsrc, [SRC["exact_scan"], SRC["estimate"]]))
if len(hard_left) and hard_left[0] != order[0]:
    demo.append(hard_left[0])

# ----------------------------------------------------------------------------- figure
fig = plt.figure(figsize=(14, 7.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1.1, 1], wspace=0.12, hspace=0.3)
ax = fig.add_subplot(gs[:, 0]); axt = fig.add_subplot(gs[0, 1]); axh = fig.add_subplot(gs[1, 1])
axt.axis("off")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
pad = 0.04
ax.set_xlim(X[:, 0].min() - pad, X[:, 0].max() + pad); ax.set_ylim(X[:, 1].min() - pad, X[:, 1].max() + pad)
sc = ax.scatter(X[:, 0], X[:, 1], s=9, c="lightgrey", linewidths=0)
suptitle = fig.suptitle("", fontsize=14, fontweight="bold")
head = axt.text(0, 1, "", va="top", fontsize=12.5, fontweight="bold", color="#1f3b73")
body = axt.text(0, 0.86, "", va="top", fontsize=10.2, family="monospace", linespacing=1.45)
overlay = []


def clear():
    while overlay:
        overlay.pop().remove()


def say(title, text):
    head.set_text(title); body.set_text(text)


def base_points(colors=None, sizes=9, edge=None):
    sc.set_facecolors("lightgrey" if colors is None else colors)
    sc.set_sizes(np.broadcast_to(sizes, N) if np.ndim(sizes) == 0 else sizes)
    sc.set_edgecolors("none" if edge is None else edge)
    sc.set_linewidths(0 if edge is None else 1.1)


def bel_colors():
    b = (res.bel - res.bel.min()) / (np.ptp(res.bel) + 1e-12)
    return plt.get_cmap("viridis")(b)


K_COL = {}
for j, k in enumerate(ks_all):
    K_COL[k] = plt.get_cmap("RdYlGn_r")(j / max(1, len(ks_all) - 1))


def k_colors(karr, src):
    col = np.array([K_COL.get(int(k), (0.5, 0.5, 0.5, 1)) for k in karr])
    col[np.isin(src, [SRC["exact_scan"], SRC["estimate"]])] = (0.75, 0, 0.75, 1)
    col[src == SRC["centre"]] = (0, 0, 0, 1)
    return col


def histogram(karr, src, title, upto=None):
    axh.clear()
    m = src >= 0 if upto is None else (src >= 0) & upto
    counts = [int(np.sum((karr == k) & m & (src <= SRC["expanded"]))) for k in ks_all]
    other = int(np.sum(m & np.isin(src, [SRC["exact_scan"], SRC["estimate"]])))
    xs = [str(k) for k in ks_all] + ["fallback"]
    bars = axh.bar(xs, counts + [other], color=[K_COL[k] for k in ks_all] + [(0.75, 0, 0.75, 1)])
    for b_, v in zip(bars, counts + [other]):
        if v:
            axh.text(b_.get_x() + b_.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=8)
    axh.set_yscale("symlog"); axh.set_ylim(0, N * 3)
    axh.set_xlabel("candidate-list size needed (original BFI: N-1 = %d)" % (N - 1), fontsize=9)
    axh.set_ylabel("objects"); axh.set_title(title, fontsize=10)
    axh.tick_params(labelsize=8)


# ----------------------------------------------------------------------------- the frames
steps = []


def add(fn, hold=1):
    steps.extend([fn] * max(1, int(hold)))


def s_data():
    clear(); base_points()
    suptitle.set_text("Solution 1 - Adaptive Candidate Neighbour Retrieval   (live, real algorithm run)" + PNOTE)
    mb = 8 * N * N / 2 ** 20
    say("Step 1 - the data",
        f"dataset   : {args.dataset}\nobjects N : {N}\nclusters  : {c}\n\n"
        f"Original BFI builds and sorts the full\nN x N distance matrix:\n"
        f"  {N * N:,} distances  (~{mb:.0f} MB)\n  -> O(N^2 log N) time, O(N^2) memory\n\n"
        f"Solution 1 instead gives every object a\nSHORT, SORTED, EXACT list of its {kb}\n"
        f"nearest neighbours from a k-d tree.")
    axh.clear(); axh.axis("off")


def s_bel():
    clear(); base_points(bel_colors(), 11)
    say("Step 2 - Bel: belief of being a centre (Eq. 10)",
        "colour = Bel  (yellow = high, purple = low)\n\n"
        "Bel uses each object's K1 nearest distances.\n"
        "High Bel = dense region = candidate centre.\n\n"
        "Next BFI needs, for EVERY object,\n"
        "  delta = distance to the nearest object\n"
        "          with HIGHER Bel   (Eq. 11)")
    axh.axis("on"); axh.clear(); axh.axis("off")


def delta_frame(i, k, final, idx_found=None):
    def fn():
        clear(); base_points(bel_colors(), 9)
        idx, dist = sr.query(np.array([i]), k)
        idx, dist = idx[0], dist[0]
        higher = rank[idx] < rank[i]
        overlay.append(ax.scatter(X[idx, 0], X[idx, 1], s=26, facecolors="none", edgecolors="orange", lw=1.2,
                                  label=f"candidate list (k={k})"))
        if higher.any():
            overlay.append(ax.scatter(X[idx[higher], 0], X[idx[higher], 1], s=34, c="red", marker="^",
                                      label="candidate with higher Bel"))
        PROJ or overlay.append(ax.add_patch(Circle(X[i], dist[-1], fill=False, ls="--", lw=1.4, color="orange")))
        overlay.append(ax.scatter(*X[i], s=170, c="k", marker="o", edgecolors="w", lw=1.5, zorder=5,
                                  label="query object"))
        found, nn, dd = first_higher_in_lists(idx[None], dist[None], rank, np.array([i]))
        lines = [f"object #{i}   rank by Bel: {rank[i] + 1} of {N}", "",
                 "candidate sizes tried: " + " -> ".join(str(kk) for kk in ks_all if kk <= k), ""]
        if found[0]:
            j = int(nn[0])
            overlay.append(ax.add_patch(FancyArrowPatch(X[i], X[j], arrowstyle="-|>", mutation_scale=18,
                                                        color="red", lw=2, zorder=6)))
            same = ref.diag["delta_nn"][i] == j
            lines += [f"k = {k}: CERTIFICATE HOLDS",
                      "  the list is sorted and exact, so the first",
                      "  higher-Bel candidate IS the true nearest",
                      "  higher-Bel object - nothing closer exists.",
                      f"  delta = {dd[0]:.4f}",
                      f"  original BFI (full row of {N - 1}):",
                      "    " + ("SAME neighbour  [OK]" if same else "different: S1 alone uses estimated Bel,"),
                      *([] if same else ["    so the Bel ranking differs (S1+S2 = exact)"]),
                      "", f"  work: {k} candidates instead of {N - 1}"]
        elif final:
            lines += [f"k = {k} = k_max: still no higher-Bel candidate",
                      "  -> cannot certify from the index.",
                      "  S1 alone: estimate delta",
                      "  S1+S2   : exact scan of higher-Bel objects"]
        else:
            lines += [f"k = {k}: no higher-Bel candidate in the list",
                      "  -> certificate FAILS (answer may lie outside)",
                      f"  -> grow the list: {k} -> {min(2 * k, kmax, N - 1)}"]
        say(f"Step 3 - adaptive delta search (example {demo.index(i) + 1}/{len(demo)})", "\n".join(lines))
        overlay.append(ax.legend(loc="lower right", fontsize=8))
    return fn


def s_all_delta():
    clear()
    base_points(k_colors(dk, dsrc), np.where(dk > kb, 22, 9))
    histogram(dk, dsrc, "delta: list size each object needed")
    nb = int(np.sum(dsrc == SRC["base"])); ne = int(np.sum(dsrc == SRC["expanded"]))
    nf = int(np.sum(np.isin(dsrc, [SRC["exact_scan"], SRC["estimate"]])))
    say("Step 4 - delta for ALL objects",
        f"colour = list size needed (green {kb} ... red {ks_all[-1]})\n\n"
        f"answered from the {kb}-list : {nb:5d}  ({100 * nb / N:.1f}%)\n"
        f"list had to grow          : {ne:5d}  ({100 * ne / N:.1f}%)\n"
        f"not certifiable (fallback): {nf:5d}\n\n"
        f"mean list size: {dk.mean():.1f}   vs   {N - 1} in original\n\n"
        "Lists grow only where the data needs it:\n"
        "density peaks, cluster borders, sparse areas.\n"
        "-> the candidate size ADAPTS to the data.")


def s_centres():
    clear(); base_points(bel_colors(), 9)
    overlay.append(ax.scatter(X[res.centers, 0], X[res.centers, 1], s=420, c="red", marker="*",
                              edgecolors="w", lw=1.5, zorder=6, label="cluster centre"))
    overlay.append(ax.legend(loc="lower right", fontsize=8))
    same = set(res.centers.tolist()) == set(ref.centers.tolist())
    say("Step 5 - cluster centres",
        f"centres = the {c} objects with the largest\n  gamma = Bel x delta\n"
        "(high belief AND far from any higher-Bel object)\n\n"
        f"same centres as original BFI: {'YES' if same else 'NO'}")


csrc, ck = D["conv_src"], D["conv_k"]
conv_order = [t for t in order]
n_batches = 30
batches = np.array_split(np.arange(N), n_batches)


def conv_frame(b):
    def fn():
        clear()
        done = np.zeros(N, bool)
        upto = batches[b][-1] + 1
        done[order[:upto]] = True
        col = np.tile(np.array([[0.85, 0.85, 0.85, 1.0]]), (N, 1))
        col[done] = CMAP(labels[done] % 10)
        edge = np.tile(np.array([[0, 0, 0, 0.0]]), (N, 1))
        grownm = done & (csrc == SRC["expanded"])
        fb = done & np.isin(csrc, [SRC["exact_scan"], SRC["estimate"]])
        edge[grownm] = (1, 0, 0, 1); edge[fb] = (0.75, 0, 0.75, 1)
        sizes = np.where(grownm | fb, 26, 9)
        sc.set_facecolors(col); sc.set_sizes(sizes); sc.set_edgecolors(edge); sc.set_linewidths(1.1)
        overlay.append(ax.scatter(X[res.centers, 0], X[res.centers, 1], s=300, c="red", marker="*",
                                  edgecolors="w", lw=1.2, zorder=6))
        cur = order[batches[b]]
        t_last = cur[-1]
        nb_ = conv_nn_row = D["conv_nn"][t_last]
        nb_ = nb_[nb_ >= 0]
        if len(nb_):
            for j in nb_:
                overlay.append(ax.plot(*zip(X[t_last], X[j]), color="k", lw=1.2, zorder=7)[0])
            overlay.append(ax.scatter(*X[t_last], s=120, c="k", zorder=8))
        histogram(ck, csrc, "convergence: list size needed so far", upto=done)
        m = done & (csrc >= 0)
        nbase = int(np.sum(m & (csrc == SRC["base"]))); ngr = int(grownm.sum()); nfb = int(fb.sum())
        ntr = int(np.sum(done & (csrc == SRC["trivial"])))
        say("Step 6 - evidential convergence (Eqs. 13-15)",
            f"objects are allocated in decreasing Bel order.\n"
            f"Each needs its K2={K2} nearest ALLOCATED objects\n"
            f"-> evidence fused with Dempster's rule.\n\n"
            f"allocated : {upto:5d} / {N}\n"
            f"from {kb}-list          : {nbase:5d}\n"
            f"list grown (red ring) : {ngr:5d}\n"
            f"fallback (magenta)    : {nfb:5d}\n"
            f"first <= K2 (trivial) : {ntr:5d}\n\n"
            "Early on few objects are allocated, so lists\n"
            "must grow; later the 16 neighbours suffice.\n"
            "black lines = neighbours used by the last object")
    return fn


def s_final():
    clear()
    col = CMAP(labels % 10)
    base_points(col, 9)
    imp = res.imprecise
    overlay.append(ax.scatter(X[imp, 0], X[imp, 1], s=22, c="k", marker="x", lw=0.9, label="meta-cluster (imprecise)"))
    overlay.append(ax.scatter(X[res.centers, 0], X[res.centers, 1], s=420, c="red", marker="*",
                              edgecolors="w", lw=1.5, zorder=6, label="centre"))
    overlay.append(ax.legend(loc="lower right", fontsize=8))
    say("Step 7 - result vs original BFI",
        f"                     original     {args.variant}\n"
        f"time               : {t_orig:7.3f}s   {t_fast:7.3f}s\n"
        f"speed-up           :              {t_orig / t_fast:6.2f}x\n"
        f"neighbours / query : {N - 1:7d}    {dk.mean():7.1f} (delta)\n"
        f"                                  {ck[csrc >= 0].mean():7.1f} (conv.)\n"
        f"label agreement    :              {agree:7.3f}\n"
        f"meta-cluster objs  : {int(ref.imprecise.sum()):7d}    {int(imp.sum()):7d}\n\n"
        "Every list-based answer is certified exact.\n"
        + ("S1 alone estimates Bel and the few\nuncertifiable queries -> small differences.\n"
           "Run --variant S1+S2 for an exact match.\n" if args.variant == "S1" else
           "Solution 2 scans the uncertifiable queries\nexactly -> identical to the original.\n")
        + "Speed-up grows with N - see REPORT.md, Exp 2.")


add(s_data, 5)
add(s_bel, 5)
for i in demo:
    ks_i = [k for k in ks_all if k <= max(int(dk[i]), kb)]
    for n_, k in enumerate(ks_i):
        last = n_ == len(ks_i) - 1
        add(delta_frame(i, k, final=last and dsrc[i] != SRC["expanded"] and dsrc[i] != SRC["base"]),
            4 if last else 2)
add(s_all_delta, 7)
add(s_centres, 4)
for b in range(n_batches):
    add(conv_frame(b), 3 if b == n_batches - 1 else 1)
add(s_final, 10)

interval = 650 / args.speed
anim = FuncAnimation(fig, lambda f: steps[f](), frames=len(steps), interval=interval, repeat=False)
if args.save:
    os.makedirs("results/plots", exist_ok=True)
    out = "results/plots/s1_live.gif"
    anim.save(out, writer=PillowWriter(fps=1000 / interval), dpi=80)
    print("saved", out)
else:
    plt.show()
