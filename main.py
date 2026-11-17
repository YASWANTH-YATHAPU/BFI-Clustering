"""Minimal demo:  python main.py
Compares the original BFI with the four accelerated variants on one dataset and plots
original vs the proposed S1+S2 (results/plots/demo.png)."""
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bfi.datasets import synthetic
from bfi.metrics import evaluate
from bfi.compare import compare_results
from bfi.variants import run_variant, ORDER, LABELS

X, y = synthetic(4000)["varied_density"]
c, K1, K2, zeta = 3, 240, 3, 0.1
res, times = {}, {}
for name in ORDER:
    t = time.perf_counter()
    res[name] = run_variant(name, X, c, K1, K2, zeta, theta_sample=1000)
    times[name] = time.perf_counter() - t

ref = res["original"]
print(f"{'variant':45s} {'time':>7s} {'speed-up':>8s} {'ARI':>6s} {'NMI':>6s} {'agree':>6s} {'BetP TV':>8s}")
for name in ORDER:
    e = evaluate(res[name], y)
    cm = compare_results(ref, res[name], X)
    print(f"{LABELS[name]:45s} {times[name]:6.3f}s {times['original'] / times[name]:7.2f}x "
          f"{e['ARI']:6.3f} {e['NMI']:6.3f} {cm['label_agree']:6.3f} {cm['betp_tv']:8.4f}")
info = res["S1+S2"].info
print("\nS1+S2 work counters:", {k: info[k] for k in ("n_delta_base", "n_delta_expanded", "n_delta_scan",
                                                       "n_conv_base", "n_conv_expanded", "n_conv_scan")})

os.makedirs("results/plots", exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
for a, name in zip(ax, ("original", "S1+S2")):
    r = res[name]
    p = ~r.imprecise
    a.scatter(X[p, 0], X[p, 1], c=r.labels[p], s=4, cmap="tab10", vmin=0, vmax=9)
    a.scatter(X[r.imprecise, 0], X[r.imprecise, 1], c="k", marker="x", s=12, label="meta-cluster")
    a.scatter(X[r.centers, 0], X[r.centers, 1], c="r", marker="*", s=150, edgecolor="w", label="centre")
    a.set_title(f"{LABELS[name]}  ({times[name]:.2f}s)", fontsize=9)
    a.legend(fontsize=7)
plt.tight_layout()
plt.savefig("results/plots/demo.png", dpi=140)
print("saved results/plots/demo.png")
