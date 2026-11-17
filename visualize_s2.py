"""Live, step-by-step visualisation of Solution 2 - Exact Distance-Based Refinement.

    python visualize_s2.py                      # animated window (default: S2, varied_density, N=1500)
    python visualize_s2.py --save               # no window; writes results/plots/s2_live.gif
    python visualize_s2.py --variant S1+S2      # Solution 2 as the exact fallback of Solution 1
    python visualize_s2.py --dataset moons --n 2000 --speed 2

Solution 2 never builds the N x N distance matrix, yet every quantity it produces is EXACT:
  1. exact theta (Eq. 9)  - chunked partial selection: only a (chunk x N) block in memory at a time
  2. exact Bel   (Eq. 10) - exact K1 nearest distances streamed from the k-d tree in chunks
  3. exact delta (Eq. 11) - from the candidate list when its certificate holds, otherwise an exact
                            scan restricted to the HIGHER-Bel subset only
  4. exact convergence    - K2 nearest allocated objects from the list, otherwise an exact scan
     (Eqs. 13-15)           restricted to the ALLOCATED subset only
Each step is checked live against the original BFI (full matrix).  Nothing is simulated: the
frames call bfi.refine / bfi.neighbors / bfi.candidates and read the counters of bfi.fast.
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
ap.add_argument("--variant", default="S2", choices=["S2", "S1+S2"])
ap.add_argument("--chunks", type=int, default=6, help="number of chunks shown for the streamed theta / Bel passes")
ap.add_argument("--speed", type=float, default=1.0, help="2 = twice as fast")
ap.add_argument("--save", action="store_true", help="write results/plots/s2_live.gif instead of showing a window")
args = ap.parse_args()
if args.save:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy.spatial.distance import cdist

from bfi.datasets import synthetic, real
from bfi.variants import run_variant, DEFAULTS
from bfi.neighbors import Searcher
from bfi.refine import exact_bel_streamed, exact_nearest_higher, exact_k_nearest
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
K1, K2, zeta, q = max(10, int(0.06 * N)), 3, 0.1, 0.9
print(f"{args.dataset}: N={N}, c={c}, K1={K1}, K2={K2}  - running original BFI and {args.variant} (exact theta) ...")
t = time.perf_counter(); ref = run_variant("original", XA, c, K1, K2, zeta); t_orig = time.perf_counter() - t
t = time.perf_counter(); res = run_variant(args.variant, XA, c, K1, K2, zeta, theta_sample=None)
t_fast = time.perf_counter() - t
cmp = compare_results(ref, res, XA)
print(f"original {t_orig:.2f}s   {args.variant} {t_fast:.2f}s   label agreement {cmp['label_agree']:.3f}   "
      f"max |BBA diff| {np.abs(res.M - ref.M).max():.1e}")

D = res.diag
kb = int(res.info.get("k_base", DEFAULTS["k_cand"]))
kmax = DEFAULTS["k_max"]
list_k = kb if args.variant == "S2" else min(kmax, N - 1)     # list size examined before a scan
order = np.argsort(-res.bel, kind="stable")
rank = np.empty(N, dtype=np.int64); rank[order] = np.arange(N)
sr = Searcher(XA)
labels = res.labels
CMAP = plt.get_cmap("tab10")
dsrc, csrc, ck = D["delta_src"], D["conv_src"], D["conv_k"]
SCAN = SRC["exact_scan"]

# ---- recompute theta and Bel chunk by chunk with the real code (to animate the streaming) ------
row_chunks = np.array_split(np.arange(N), args.chunks)
pos = q * (N - 2); lo = int(np.floor(pos)); frac = pos - lo
kth = [1 + lo, 2 + lo]
theta_live = np.empty(N)
for r in row_chunks:                                    # identical lines to bfi.refine.exact_theta
    P = np.partition(cdist(XA[r], XA), kth, axis=1)
    a, b = P[:, kth[0]], P[:, kth[1]]
    theta_live[r] = 1.0 / np.maximum(a + (b - a) * frac if frac < 0.5 else b - (b - a) * (1 - frac), 1e-12)
logprod = np.empty(N)
for r in row_chunks:                                    # the real streamed Bel function, per chunk
    logprod[r] = exact_bel_streamed(sr, theta_live, K1, keep=1, rows=r)[0]
bel_live = 1.0 - np.exp(logprod)
th_err = np.abs(theta_live - ref.theta).max() / ref.theta.max()
bel_err = np.abs(bel_live - ref.bel).max()

# example objects
ex_theta = order[int(0.3 * N)]
ex_bel = order[int(0.15 * N)]
d_scan = [int(u) for u in np.flatnonzero(dsrc == SCAN)]
d_scan = sorted(d_scan, key=lambda u: rank[u])
demo_delta = ([d_scan[0]] if d_scan else []) + ([d_scan[len(d_scan) // 2]] if len(d_scan) > 2 else [])
c_scan = [int(t) for t in order if csrc[t] == SCAN]
demo_conv = c_scan[:1] + ([c_scan[len(c_scan) // 2]] if len(c_scan) > 2 else [])

# ----------------------------------------------------------------------------- figure
fig = plt.figure(figsize=(14, 7.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1.1, 1], wspace=0.12, hspace=0.3)
ax = fig.add_subplot(gs[:, 0]); axt = fig.add_subplot(gs[0, 1]); axh = fig.add_subplot(gs[1, 1])
axt.axis("off")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
pad = 0.04
ax.set_xlim(X[:, 0].min() - pad, X[:, 0].max() + pad); ax.set_ylim(X[:, 1].min() - pad, X[:, 1].max() + pad)
sc = ax.scatter(X[:, 0], X[:, 1], s=9, c="lightgrey", linewidths=0)
fig.suptitle(f"Solution 2 - Exact Distance-Based Refinement   ({args.variant}, live, real algorithm run)" + PNOTE,
             fontsize=14, fontweight="bold")
head = axt.text(0, 1, "", va="top", fontsize=12.5, fontweight="bold", color="#73241f")
body = axt.text(0, 0.86, "", va="top", fontsize=10.2, family="monospace", linespacing=1.45)
overlay = []
GREY = np.array([0.85, 0.85, 0.85, 1.0])


def clear():
    while overlay:
        overlay.pop().remove()


def say(title, text):
    head.set_text(title); body.set_text(text)


def points(colors=None, sizes=9, edge=None):
    sc.set_facecolors("lightgrey" if colors is None else colors)
    sc.set_sizes(np.broadcast_to(sizes, N) if np.ndim(sizes) == 0 else sizes)
    sc.set_edgecolors("none" if edge is None else edge)
    sc.set_linewidths(0 if edge is None else 1.2)


def viridis(v, mask=None):
    v = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-12)
    col = plt.get_cmap("viridis")(np.nan_to_num(v))
    if mask is not None:
        col[~mask] = GREY
    return col


def matrix_panel(rows_done, chunk, width, title, note):
    """Memory picture: the full N x N matrix of the original vs the block Solution 2 holds now."""
    axh.clear(); axh.axis("on")
    axh.add_patch(Rectangle((0, 0), N, N, fill=False, ls="--", lw=1.2, color="grey"))
    axh.text(N / 2, N * 1.02, f"original: full N x N = {N}x{N} = {8 * N * N / 2 ** 20:.0f} MB",
             ha="center", va="bottom", fontsize=8, color="grey")
    if rows_done:
        axh.add_patch(Rectangle((0, N - rows_done), width, rows_done, color="#9ecae1", alpha=0.6, lw=0))
    if chunk is not None:
        axh.add_patch(Rectangle((0, N - chunk[1]), width, chunk[1] - chunk[0], color="#d62728", lw=0))
    axh.set_xlim(-N * 0.02, N * 1.02); axh.set_ylim(-N * 0.02, N * 1.12)
    axh.set_xticks([]); axh.set_yticks([])
    axh.set_title(title, fontsize=10)
    axh.set_xlabel(note, fontsize=8.5)


def counts_panel(groups, title):
    axh.clear(); axh.axis("on")
    names = [g[0] for g in groups]; vals = [g[1] for g in groups]; cols = [g[2] for g in groups]
    bars = axh.barh(names, vals, color=cols)
    for b_, v in zip(bars, vals):
        axh.text(b_.get_width(), b_.get_y() + b_.get_height() / 2, f" {v:,}", va="center", fontsize=8.5)
    axh.set_xscale("symlog"); axh.set_xlim(0, max(vals) * 8)
    axh.set_title(title, fontsize=10); axh.tick_params(labelsize=8.5)


# ----------------------------------------------------------------------------- frames
steps = []


def add(fn, hold=1):
    steps.extend([fn] * max(1, int(hold)))


def s_intro():
    clear(); points()
    say("Step 1 - the problem",
        f"dataset {args.dataset}, N = {N}, {c} clusters\n\n"
        "Original BFI computes ALL N x N distances,\n"
        f"keeps them in one matrix ({8 * N * N / 2 ** 20:.0f} MB here,\n"
        f"{8 * 50000 ** 2 / 2 ** 30:.0f} GB at N = 50 000) and sorts every row.\n\n"
        "Solution 2 computes EXACT distances too -\n"
        "but only the ones each formula really needs,\n"
        "a small block at a time, so the result is\n"
        "identical to the original without the matrix.")
    matrix_panel(0, None, N, "memory picture", "")


def theta_frame(ci):
    def fn():
        clear()
        done = np.zeros(N, bool)
        for r in row_chunks[:ci + 1]:
            done[r] = True
        cur = row_chunks[ci]
        col = viridis(theta_live, done)
        col[cur] = (0.84, 0.15, 0.16, 1)
        points(col, np.where(np.isin(np.arange(N), cur), 14, 9))
        i = ex_theta
        if done[i]:
            r = 1 / theta_live[i]
            PROJ or overlay.append(ax.add_patch(Circle(X[i], r, fill=False, ls="--", lw=1.6, color="k")))
            overlay.append(ax.scatter(*X[i], s=150, c="k", edgecolors="w", lw=1.5, zorder=6))
        a, b = cur[0], cur[-1] + 1
        say("Step 2 - exact theta, chunk by chunk (Eq. 9)",
            f"theta_i = 1 / (90% quantile of distances from i)\n\n"
            f"chunk {ci + 1}/{len(row_chunks)}: rows {a}..{b - 1} (red)\n"
            f"  distances to all N objects: {len(cur)} x {N}\n"
            f"  np.partition picks the 90% quantile\n"
            f"  WITHOUT sorting the whole row\n"
            f"  then the block is thrown away.\n\n"
            "dashed circle = example object: 90% of all\n"
            "other objects lie inside 1/theta.\n"
            + (f"\ncheck vs original: max rel. error {th_err:.1e}  [EXACT]" if ci == len(row_chunks) - 1 else ""))
        matrix_panel(b, (a, b), N, "memory: one chunk x N at a time (red)",
                     f"held now: {len(cur)} x {N} = {8 * len(cur) * N / 2 ** 20:.1f} MB   (real code: <= 4 MB per block)")
    return fn


def bel_frame(ci):
    def fn():
        clear()
        done = np.zeros(N, bool)
        for r in row_chunks[:ci + 1]:
            done[r] = True
        cur = row_chunks[ci]
        col = viridis(bel_live, done)
        col[cur] = (0.84, 0.15, 0.16, 1)
        points(col, np.where(np.isin(np.arange(N), cur), 14, 9))
        i = ex_bel
        idx, dist = sr.query(np.array([i]), K1)
        overlay.append(ax.scatter(X[idx[0], 0], X[idx[0], 1], s=24, facecolors="none", edgecolors="orange", lw=1))
        PROJ or overlay.append(ax.add_patch(Circle(X[i], dist[0, -1], fill=False, ls="--", lw=1.5, color="orange")))
        overlay.append(ax.scatter(*X[i], s=150, c="k", edgecolors="w", lw=1.5, zorder=6))
        a, b = cur[0], cur[-1] + 1
        say("Step 3 - exact Bel, streamed (Eq. 10)",
            f"Bel_i needs only the K1 = {K1} NEAREST distances.\n\n"
            f"chunk {ci + 1}/{len(row_chunks)}: rows {a}..{b - 1} (red)\n"
            f"  k-d tree returns the exact {K1} nearest\n"
            f"  neighbours of each row -> {len(cur)} x {K1} block\n"
            f"  -> Bel = 1 - prod(1 - exp(-theta d)/K1)\n\n"
            f"orange = the {K1} neighbours of the example.\n"
            "colour = Bel (yellow = likely centre)\n"
            + (f"\ncheck vs original: max |Bel diff| {bel_err:.1e}  [EXACT]" if ci == len(row_chunks) - 1 else ""))
        matrix_panel(b, (a, b), K1, f"memory: chunk x K1 (red) - not N x N",
                     f"held now: {len(cur)} x {K1} = {8 * len(cur) * K1 / 2 ** 20:.2f} MB  vs  {8 * N * N / 2 ** 20:.0f} MB")
    return fn


def delta_demo(u, phase):
    def fn():
        clear(); points(viridis(res.bel), 9)
        idx, dist = sr.query(np.array([u]), list_k)
        idx, dist = idx[0], dist[0]
        overlay.append(ax.scatter(X[idx, 0], X[idx, 1], s=26, facecolors="none", edgecolors="orange", lw=1.2,
                                  label=f"candidate list ({list_k})"))
        PROJ or overlay.append(ax.add_patch(Circle(X[u], dist[-1], fill=False, ls="--", lw=1.4, color="orange")))
        hi = order[:rank[u]]
        lines = [f"object #{u}   Bel rank {rank[u] + 1} of {N}", ""]
        if phase == 0:
            lines += [f"its {list_k} nearest neighbours contain NO",
                      "object with higher Bel -> the list cannot",
                      "certify delta (true answer lies outside).", "",
                      "S1 alone would ESTIMATE delta here."]
        else:
            overlay.append(ax.scatter(X[hi, 0], X[hi, 1], s=30, c="red", marker="^",
                                      label=f"higher-Bel subset ({len(hi)})"))
            j, dd = exact_nearest_higher(XA, u, order, rank)
            overlay.append(ax.add_patch(FancyArrowPatch(X[u], X[j], arrowstyle="-|>", mutation_scale=18,
                                                        color="k", lw=2.2, zorder=7)))
            same = ref.diag["delta_nn"][u] == j and np.isclose(ref.delta[u], dd)
            lines += ["Solution 2: EXACT SCAN, but only of the",
                      f"objects with higher Bel: {len(hi)} distances",
                      f"(original: full row of {N - 1}, then sort)", "",
                      f"exact delta = {dd:.4f}",
                      "original BFI: " + ("SAME neighbour & delta  [EXACT]" if same else "differs")]
        overlay.append(ax.scatter(*X[u], s=170, c="k", edgecolors="w", lw=1.5, zorder=8, label="query object"))
        overlay.append(ax.legend(loc="lower right", fontsize=8))
        say(f"Step 4 - exact refinement of delta (Eq. 11)", "\n".join(lines))
        counts_panel([("from list", int(np.sum(dsrc <= SRC["expanded"]) - np.sum(dsrc == SRC["centre"])),
                       "#2ca02c"), ("exact scan", int(np.sum(dsrc == SCAN)), "#bf00bf")],
                     "delta: how each object was answered")
    return fn


def s_delta_all():
    clear()
    col = np.tile(np.array([[0.17, 0.63, 0.17, 1]]), (N, 1))
    m = dsrc == SCAN
    col[m] = (0.75, 0, 0.75, 1)
    points(col, np.where(m, 40, 8))
    ns = int(m.sum())
    work = int(rank[m].sum())
    same = np.array_equal(res.diag["delta_nn"], ref.diag["delta_nn"])
    say("Step 4 - delta for ALL objects",
        f"green   : certified from the candidate list\n"
        f"magenta : exact scan of higher-Bel subset\n\n"
        f"objects needing a scan : {ns} of {N}\n"
        f"distances in all scans : {work:,}\n"
        f"original (all rows)    : {N * (N - 1):,}\n\n"
        f"delta neighbours identical to original: {'YES' if same else 'NO'}\n"
        f"centres identical: {'YES' if set(res.centers) == set(ref.centers) else 'NO'}")
    counts_panel([("from list", N - ns, "#2ca02c"), ("exact scan", ns, "#bf00bf")],
                 "delta: how each object was answered")


def conv_demo(t, phase):
    def fn():
        clear()
        before = set(res.centers.tolist()) | set(order[:rank[t]].tolist())
        alloc = np.array(sorted(before))
        col = np.tile(GREY, (N, 1))
        col[alloc] = CMAP(labels[alloc] % 10)
        points(col, 9)
        idx, dist = sr.query(np.array([t]), list_k)
        overlay.append(ax.scatter(X[idx[0], 0], X[idx[0], 1], s=26, facecolors="none", edgecolors="orange", lw=1.2,
                                  label=f"candidate list ({list_k})"))
        PROJ or overlay.append(ax.add_patch(Circle(X[t], dist[0, -1], fill=False, ls="--", lw=1.4, color="orange")))
        n_in = int(np.isin(idx[0], alloc).sum())
        lines = [f"object #{t}, allocated so far: {len(alloc)} of {N}", "",
                 f"needs its K2 = {K2} nearest ALLOCATED objects.",
                 f"allocated objects in its list: {n_in} < {K2}", ""]
        if phase == 0:
            lines += ["-> the list cannot certify the answer",
                      "   (early in the convergence few objects",
                      "    are allocated, so they are far away)."]
        else:
            nb, nd = exact_k_nearest(XA, t, alloc, K2)
            for j in nb:
                overlay.append(ax.plot(*zip(X[t], X[j]), color="k", lw=2, zorder=7)[0])
            overlay.append(ax.scatter(X[alloc, 0], X[alloc, 1], s=22, facecolors="none", edgecolors="m", lw=0.8,
                                      label=f"allocated subset scanned ({len(alloc)})"))
            same = set(nb.tolist()) == set(ref.diag["conv_nn"][t][ref.diag["conv_nn"][t] >= 0].tolist())
            lines += ["Solution 2: exact scan of the ALLOCATED",
                      f"subset only: {len(alloc)} distances (not {N - 1})",
                      f"-> exact {K2} nearest (black lines) -> Dempster",
                      "original BFI: " + ("SAME neighbours  [EXACT]" if same else "differs")]
        overlay.append(ax.scatter(*X[t], s=170, c="k", edgecolors="w", lw=1.5, zorder=8, label="query object"))
        overlay.append(ax.legend(loc="lower right", fontsize=8))
        say("Step 5 - exact refinement of the convergence", "\n".join(lines))
        counts_panel([("list examined", list_k, "#ff7f0e"), ("allocated subset (S2 scan)", len(alloc), "#bf00bf"),
                      ("original: full row", N - 1, "#7f7f7f")], "distances computed for this object")
    return fn


n_batches = 25
batches = np.array_split(np.arange(N), n_batches)


def conv_frame(b):
    def fn():
        clear()
        upto = batches[b][-1] + 1
        done = np.zeros(N, bool); done[order[:upto]] = True; done[res.centers] = True
        col = np.tile(GREY, (N, 1)); col[done] = CMAP(labels[done] % 10)
        scan = done & (csrc == SCAN)
        edge = np.tile(np.array([[0, 0, 0, 0.0]]), (N, 1)); edge[scan] = (0.75, 0, 0.75, 1)
        points(col, np.where(scan, 34, 9), edge)
        overlay.append(ax.scatter(X[res.centers, 0], X[res.centers, 1], s=300, c="red", marker="*",
                                  edgecolors="w", lw=1.2, zorder=6))
        listed = int(np.sum(done & np.isin(csrc, [SRC["base"], SRC["expanded"]])))
        ns = int(scan.sum()); tr = int(np.sum(done & (csrc == SRC["trivial"])))
        say("Step 5 - evidential convergence (Eqs. 13-15)",
            f"allocation in decreasing Bel order\n\n"
            f"allocated            : {int(done.sum()):5d} / {N}\n"
            f"answered from list   : {listed:5d}\n"
            f"exact scan (magenta) : {ns:5d}\n"
            f"first <= K2 (trivial): {tr:5d}\n\n"
            "magenta rings = objects whose K2 nearest\n"
            "allocated neighbours were found by an exact\n"
            "scan of the allocated subset.")
        counts_panel([("from list", listed, "#2ca02c"), ("exact scan", ns, "#bf00bf"), ("trivial", tr, "#7f7f7f")],
                     "convergence: how each object was answered")
    return fn


def s_final():
    clear()
    points(CMAP(labels % 10), 9)
    imp = res.imprecise
    overlay.append(ax.scatter(X[imp, 0], X[imp, 1], s=22, c="k", marker="x", lw=0.9, label="meta-cluster (imprecise)"))
    overlay.append(ax.scatter(X[res.centers, 0], X[res.centers, 1], s=420, c="red", marker="*",
                              edgecolors="w", lw=1.5, zorder=6, label="centre"))
    overlay.append(ax.legend(loc="lower right", fontsize=8))
    say("Step 6 - result vs original BFI",
        f"                       original   {args.variant}\n"
        f"time                 : {t_orig:7.3f}s  {t_fast:7.3f}s\n"
        f"largest distance blk : {N}x{N:<5d}  chunk x N\n"
        f"label agreement      :            {cmp['label_agree']:7.3f}\n"
        f"max |theta| rel diff :            {th_err:7.1e}\n"
        f"max |Bel| diff       :            {bel_err:7.1e}\n"
        f"max |BBA| diff       :            {np.abs(res.M - ref.M).max():7.1e}\n"
        f"meta-cluster objects : {int(ref.imprecise.sum()):7d}   {int(imp.sum()):7d}\n\n"
        "Same theta, Bel, delta, centres, BBAs and\n"
        "meta-clusters as the original (up to float\n"
        "rounding) - evidential reasoning preserved.\n"
        "Speed/memory gains grow with N: REPORT.md Exp 2.")
    counts_panel([("theta (exact)", N, "#1f77b4"), ("Bel (exact)", N, "#1f77b4"),
                  ("delta scans", int(np.sum(dsrc == SCAN)), "#bf00bf"),
                  ("conv. scans", int(np.sum(csrc == SCAN)), "#bf00bf")],
                 "Solution 2: exact work done")


add(s_intro, 6)
for ci in range(len(row_chunks)):
    add(theta_frame(ci), 4 if ci == len(row_chunks) - 1 else 2)
for ci in range(len(row_chunks)):
    add(bel_frame(ci), 4 if ci == len(row_chunks) - 1 else 2)
for u in demo_delta:
    add(delta_demo(u, 0), 4); add(delta_demo(u, 1), 6)
add(s_delta_all, 6)
for t in demo_conv:
    add(conv_demo(t, 0), 4); add(conv_demo(t, 1), 6)
for b in range(n_batches):
    add(conv_frame(b), 3 if b == n_batches - 1 else 1)
add(s_final, 12)

interval = 650 / args.speed
anim = FuncAnimation(fig, lambda f: steps[f](), frames=len(steps), interval=interval, repeat=False)
if args.save:
    os.makedirs("results/plots", exist_ok=True)
    out = "results/plots/s2_live.gif"
    anim.save(out, writer=PillowWriter(fps=1000 / interval), dpi=80)
    print("saved", out)
else:
    plt.show()
