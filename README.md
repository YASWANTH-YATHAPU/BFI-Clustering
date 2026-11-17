# Efficient and Evidence-Preserving BFI Clustering

Accelerating **Belief-Based Fuzzy and Imprecise Clustering (BFI)** (Z.-w. Zhang et al., *IEEE Trans. Fuzzy Syst.* 33(8), 2755–2767, 2025) without changing its evidential reasoning.
The authors identify long execution time as BFI's limitation: the reference algorithm builds and sorts the full N × N distance matrix — **O(N² log N) time and O(N²) memory**.

This project adds two neighbour-search solutions to BFI, evaluates them in a 2 × 2 factorial design against the original, and reports every result — including the cases where the acceleration does not help.

> **Full technical report with all measured results: [REPORT.md](REPORT.md)**

---

## 1. The two proposed solutions

BFI needs neighbours for three things: Bel (Eq. 10, the K1 nearest distances), δ (Eq. 11, the nearest object with higher Bel) and the evidential convergence (Eqs. 13–15, the K2 nearest *allocated* objects). Only the neighbour search is changed; θ, Bel, δ, the mass functions, Dempster fusion, the ζ-set M_t, the BBA of Eq. 15 and the pignistic decision of Eq. 16 are computed by the same formulas (and the same shared code, [bfi/evidence.py](bfi/evidence.py)) as in the original.

### Solution 1 — Adaptive Candidate Neighbour Retrieval ([bfi/candidates.py](bfi/candidates.py))
* Every object gets a short, **exact, sorted** candidate list (k_min = 16 nearest neighbours) from a k-d tree instead of a row of the N × N matrix.
* **Sufficiency certificate.** On a sorted exact list, *the first candidate with higher Bel is the true δ-neighbour* and *the first K2 allocated candidates are the true K2 nearest allocated objects* — every closer object is in the list and was checked. A query answered from the list is therefore provably exact.
* **Adaptivity.** Only lists that fail the certificate are grown, geometrically (16 → 32 → … → k_max = 512) through the index. Cluster cores stop at 16; density peaks, cluster borders, sparse regions and the sparsely-allocated start of the convergence search further. The candidate size thus follows the local data structure.

### Solution 2 — Exact Distance-Based Refinement ([bfi/refine.py](bfi/refine.py))
* **Exact Bel** from the exact K1 nearest distances, **streamed** through the index in memory-bounded chunks (the same pass yields Solution 1's base lists).
* **Exact fallback** for the few queries Solution 1 cannot certify within k_max: a scan restricted to the higher-Bel subset (δ) or to the allocated subset (convergence).
* **Exact θ** (Eq. 9) by chunked partial selection — no N × N matrix.

Exact distance computation itself is not new (the original computes all of them). What is new is *which* distances are computed: refinement is triggered only by a failed certificate, runs on restricted subsets, and is rarely needed because Solution 1 resolves almost every query through the index.

### Integration ([bfi/fast.py](bfi/fast.py)) and the compared variants ([bfi/variants.py](bfi/variants.py))

| Variant | Solution 1 | Solution 2 | Role |
|---|---|---|---|
| `original` | – | – | reference BFI, full N × N matrix ([bfi/original.py](bfi/original.py)) |
| `control` | off | off | naive fixed-k k-NN BFI (shows what happens without the two solutions) |
| `S1` | **on** | off | adaptive retrieval only; uncertified queries use estimates |
| `S2` | off | **on** | fixed candidate list + exact refinement |
| `S1+S2` | **on** | **on** | **proposed method** |

**Exactness guarantee.** With exact θ, `S2` and `S1+S2` return *the same result as the original BFI* (Bel, δ, centres and every BBA, up to floating-point rounding) for any candidate budget — verified by unit tests. θ may optionally be estimated from a random sample (`theta_sample=1000`), the only approximation in the practical configuration.

---

## 2. Setup and usage

**Working on this repository together (GitHub).**
```bash
git clone https://github.com/YASWANTH-YATHAPU/BFI-Clustering.git
cd BFI-Clustering
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest -q tests         # must say "39 passed" before you change anything

git checkout -b my-change         # work on your own branch, never directly on main
# ... edit, run the tests again ...
git add -A
git commit -m "Describe what you changed"
git push -u origin my-change      # then open a Pull Request on GitHub
git pull origin main              # get your friends' latest changes
```

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

```bash
python main.py                    # demo: all 5 variants on one dataset + results/plots/demo.png
python visualize_s1.py            # LIVE animation of how Solution 1 works (--save -> results/plots/s1_live.gif)
python visualize_s2.py            # LIVE animation of how Solution 2 works (--save -> results/plots/s2_live.gif)
python -m pytest -q tests         # 39 unit tests (exactness, certificates, metrics, validation)
python run_all.py --quick         # every experiment at small sizes (~10 min)
python run_all.py                 # full experiments (~1-1.5 h, single-threaded)
python run_all.py --only 1 3      # selected experiments
```

```python
from bfi import bfi_original, bfi_fast
from bfi.variants import run_variant

res = bfi_fast(X, n_clusters=3, K1=180, K2=3, zeta=0.1)       # proposed S1+S2, sampled theta
res = bfi_fast(X, 3, 180, K2=3, theta_sample=None)             # S1+S2, exact (== original)
res.labels, res.sets, res.imprecise, res.betp, res.info        # crisp labels, focal sets, meta-clusters
```

VS Code: `.vscode/launch.json` contains run configurations for the demo and the experiments.

---

## 3. Project layout

```
bfi/original.py           reference BFI (paper, Eqs. 9-16)        bfi/fast.py        integrated accelerated BFI
bfi/candidates.py         Solution 1: adaptive retrieval           bfi/refine.py      Solution 2: exact refinement
bfi/neighbors.py          spatial-index back-ends                  bfi/evidence.py    mass fusion, BBA, pignistic (shared)
bfi/variants.py           named configurations + defaults          bfi/validation.py  input validation
bfi/metrics.py            ARI, NMI, AMI, Re, Ri, P/R/RI, EP/ER/ERI bfi/compare.py     agreement with the original
bfi/neighbour_metrics.py  neighbour recall / accuracy              bfi/bench.py       time + fresh-process peak memory
bfi/datasets.py           13 benchmark datasets + scalability data
experiments/exp1..5       quality | scalability | ablation | sensitivity | Bel-adaptivity study
results/*.csv             raw and summary tables  results/tables/*.md  markdown tables  results/plots/*.png  figures
REPORT.md                 technical report        tests/test_bfi.py    unit tests
```

## 4. Experiments

| Script | Question |
|---|---|
| `exp1_quality.py` | Original vs control / S1 / S2 / S1+S2 on 13 datasets × 5 seeds, exact and sampled θ: quality, time, memory, neighbour accuracy, evidence preservation |
| `exp2_scalability.py` | Time, memory, speed-up and refinement work for N = 1 000 … 50 000 (3 seeds) |
| `exp3_ablation.py` | 2 × 2 factorial and every parameter: k_cand, k_min, k_max, Bel policy, estimators, tail model, θ sample (4 datasets × 3 seeds) |
| `exp4_sensitivity.py` | Does the proposed method track the original across K1, K2, ζ? (4 datasets × 3 seeds) |
| `exp5_bel_adaptivity.py` | Can the Bel candidate size be adapted per object from local data? (negative result) |

**Measurement protocol.** Time = median of repeated in-process runs, no sampler running. Peak memory = additional peak resident memory of one run in a fresh process (OS peak working set on Windows / 1 ms RSS sampling). All variants are single-threaded so they compete fairly with the numpy baseline. Hardware: the 16-core Windows 11 laptop the project was developed on; absolute times will differ on other machines, ratios are what matter.

**Datasets.** moons, circles, anisotropic, blobs_separated, blobs_overlap, varied_density (2-D, N = 3000); blobs_hd8, blobs_hd32 (8-D / 32-D, N = 3000); iris, wine, breast_cancer, digits6 (PCA-10), digits10 (PCA-16) from scikit-learn. All min-max scaled. Seeds change the sampling noise only, not the cluster layout. K1 = 0.06 N (min 10), K2 = 3, ζ = 0.1, q = 0.9.

The headline results, limitations and failure cases are in [REPORT.md](REPORT.md).
