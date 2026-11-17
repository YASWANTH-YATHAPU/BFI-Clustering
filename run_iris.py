"""Run the original BFI and the four accelerated variants on the iris dataset.

    python run_iris.py
"""
import collections
import time
import numpy as np
from bfi.datasets import real
from bfi.variants import run_variant, ORDER, LABELS
from bfi.metrics import evaluate
from bfi.compare import compare_results

X, y = real()["iris"]
c, K1, K2, zeta = 3, 10, 3, 0.1
print(f"IRIS: N={len(X)}, features={X.shape[1]}, clusters={c}, K1={K1}, K2={K2}, zeta={zeta}\n")
print(f"{'method':42s} {'time':>7s} {'ARI':>6s} {'NMI':>6s} {'Re':>6s} {'Ri':>6s} {'agree':>6s} {'meta':>5s}")
res = {}
for m in ORDER:
    t = time.perf_counter()
    r = run_variant(m, X, c, K1, K2, zeta, theta_sample=None)
    t = time.perf_counter() - t
    res[m] = r
    e = evaluate(r, y)
    a = compare_results(res["original"], r, X)["label_agree"]
    print(f"{LABELS[m]:42s} {t * 1000:5.1f}ms {e['ARI']:6.3f} {e['NMI']:6.3f} {e['Re']:6.3f} {e['Ri']:6.3f} "
          f"{a:6.3f} {int(r.imprecise.sum()):5d}")

r, o = res["S1+S2"], res["original"]
print("\nS1+S2 vs original: identical labels =", np.array_equal(r.labels, o.labels),
      "| max BBA difference =", float(np.abs(r.M - o.M).max()))
names = ["setosa", "versicolor", "virginica"]
for k in range(c):
    m = r.labels == k
    print(f"cluster {k}: {m.sum():3d} flowers ->", dict(collections.Counter(names[t] for t in y[m])))
