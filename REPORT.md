# Efficient and Evidence-Preserving BFI: Technical Implementation and Results Report

**Base method.** Z.-w. Zhang, Z.-g. Liu, L.-b. Ning, H.-p. Tian, B.-l. Wang, "Belief-based fuzzy and imprecise clustering for arbitrary data distributions," *IEEE Trans. Fuzzy Syst.*, 33(8), 2755–2767, 2025.
**Code.** this repository; every number below is produced by `python run_all.py` and stored in `results/` (raw CSV per run, summary CSV with mean ± std, markdown tables in `results/tables/`, figures in `results/plots/`).

---

## 1. Summary

BFI identifies cluster centres from belief-based density peaks and assigns the remaining objects by evidential convergence, representing ambiguous objects through meta-clusters. Its reference implementation computes and sorts the full N × N distance matrix (O(N² log N) time, O(N²) memory); the authors name execution time as its limitation.

We replace the neighbour search of BFI by two components and leave the evidential reasoning untouched:

* **Solution 1 – Adaptive Candidate Neighbour Retrieval.** Short exact candidate lists from a k-d tree, an exact *sufficiency certificate* for the δ and convergence queries, and geometric growth of only the lists that fail it.
* **Solution 2 – Exact Distance-Based Refinement.** Exact Bel streamed in bounded memory, and exact scans on restricted subsets for the queries Solution 1 cannot certify.

Main findings (single-threaded, measured):

| Finding | Evidence |
|---|---|
| **S1+S2 is exactly equivalent to the original BFI** (identical Bel, δ, centres, BBAs, labels) when θ is exact, for every candidate budget. | 39 unit tests; exp1: label agreement 1.000 ± 0.000, BetP distance 0.000 on all 13 datasets × 5 seeds |
| **Memory falls from quadratic to near-linear.** | N = 4000: 366 MB → 26 MB (14×); N = 3000: 206 MB → 18 MB (11×). Original grows ×4 per doubling of N (25 → 92 → 366 MB). S1+S2 needs 38 MB at N = 50 000, where the matrix alone would need ≈ 20 GB. |
| **Runtime is lower**, increasingly with N where the original can run. | N = 4000: 0.40 s → 0.17 s (**2.3 ×**, sampled θ) or 0.27 s (1.5 ×, exact θ); N = 3000, 2-D datasets: 2.0–2.3 × |
| **Quality is preserved** in the practical configuration (θ sampled from 1000 objects). | Label agreement with the original 0.999 at N = 2000–4000; ARI 0.917–0.924 for N up to 50 000; overall ARI 0.605 ± 0.331 vs 0.630 ± 0.328 (difference concentrated in 4 high-dimensional datasets, §6.4) |
| **Both solutions are necessary.** Without refinement, quality collapses as N grows. | ARI at N = 50 000: control 0.265, S1 0.396, S2 / S1+S2 0.924 |
| **Solution 1's contribution** is fewer exact scans and smaller candidate memory, not raw speed. | N = 50 000: candidate memory 38 MB (S1+S2) vs 83 MB (S2); ablation: 7.6 vs 12.3 convergence scans per run |

Limitations (§7): for N ≥ ~20 000 the exact Bel dominates the runtime (K1 = 0.06 N makes it O(N·K1) = O(N²) work, though with O(N) memory); exact θ is inherently O(N²) time; in 32-D and on small real datasets (N ≤ 1800) there is no speed-up; θ sampling changes the results on the digits datasets.

---

## 2. The original BFI and where its cost lies

For N objects with K1 = 0.06 N, K2, threshold ζ and quantile q = 0.9 the algorithm computes:

| Step | Equation | Neighbour information needed | Original cost |
|---|---|---|---|
| θ_i | (9) 1 / 0.9-quantile of all distances from x_i | all N − 1 distances | O(N²) |
| Bel_i(C) | (10) 1 − ∏_{k≤K1} (1 − e^{−θ_i d_ik}/K1) | K1 nearest distances | sort: O(N² log N) |
| δ_i | (11) distance to the nearest object with higher Bel | nearest higher-Bel object | O(N²) |
| centres | (12) γ = Bel·δ (see §5.4) | – | O(N log N) |
| convergence | (13)–(15) masses from the K2 nearest *allocated* objects, Dempster fusion per cluster, ζ-set M_t, BBA on 2^{M_t} | K2 nearest allocated | O(N²) |
| decision | (16) pignistic probability | – | O(N c²) |

The reference implementation (`bfi/original.py`) materialises the N × N matrix D and its row-sorted copy; this is the O(N²) memory bottleneck that limits it to N ≈ 6000 on a laptop.

---

## 3. What was implemented

| File | Content |
|---|---|
| `bfi/original.py` | Reference BFI (unchanged algorithm; now also validates inputs and records which neighbours it used). |
| `bfi/candidates.py` | **Solution 1**: certificate tests and adaptive list growth for δ and the convergence. |
| `bfi/refine.py` | **Solution 2**: streamed exact Bel, exact θ by chunked selection, restricted exact scans. |
| `bfi/fast.py` | Integration; `adaptive` (S1) and `refine` (S2) switches; work counters and per-object diagnostics. |
| `bfi/evidence.py` | Mass fusion, BBA, pignistic probability — **shared by all variants**. |
| `bfi/variants.py` | The five compared configurations and the defaults chosen in the ablation. |
| `bfi/neighbour_metrics.py` | Candidate recall, selection accuracy, Bel error, agreement with the original's neighbours. |
| `bfi/bench.py` | Timing and fresh-process peak-memory measurement. |
| `bfi/validation.py` | Input validation (shape, finiteness, parameter ranges). |
| `experiments/exp1..5` | The five experiments. `tests/test_bfi.py`: 39 unit tests. |

---

## 4. Solution 1 – Adaptive Candidate Neighbour Retrieval

**Candidate lists.** A k-d tree is built once (O(N log N)). Every object receives its k_base = 16 nearest neighbours, sorted, with exact Euclidean distances (they are produced by the same streamed pass that computes Bel, §5).

**Sufficiency certificate.** Let L_i be the exact k nearest neighbours of x_i in increasing distance.
* *δ query.* If L_i contains an object with larger Bel, the first such object is the nearest larger-Bel object overall: every object closer to x_i than it is in L_i and has been checked.
* *Convergence query.* At the time x_t is processed, if L_t contains at least K2 allocated objects, its first K2 allocated entries are the K2 nearest allocated objects overall, for the same reason.
Both statements are exact, so a query answered from a list is identical to the original's answer.

**Adaptive growth.** A list that fails its certificate is regrown through the index to 2k, 4k, … up to the budget k_max = 512 (for δ: in one vectorised batch; for the convergence: on demand during the sequential pass). The number of candidates each object ends up with therefore depends on its local situation — cluster cores are resolved by the base list, while density peaks (no higher-Bel neighbour nearby), sparse regions, cluster borders, and objects allocated early (when few objects are allocated) search further. Queries still uncertified at k_max are handed to Solution 2 (or, when it is disabled, answered by an estimate: nearest higher-Bel landmark for δ, the allocated objects found so far / the nearest centre for the convergence).

**What adaptivity is *not* used for — and why (negative result, exp5).** We also tried to choose the candidate count *for Bel* per object from local data (two-scale consistency of the local intrinsic dimension; out-of-sample validation of a power-law model), so that the K1-neighbour tail could be extrapolated from a short list. At equal mean candidate count, both adaptive rules were **worse than a fixed list size** (mean |Bel error| / Bel range over 4 datasets × 3 seeds: two-scale rule 0.106 at 66 candidates vs fixed 0.054 at 64; validation rule 0.198 at 19 candidates vs fixed 0.224 at 16 and 0.111 at 32 — `results/tables/exp5_bel_adaptivity.md`, `plots/exp5_bel_adaptivity.png`). The reason is fundamental: the head of a neighbour list carries no information about density changes beyond its radius. Solution 1 is therefore applied only where an exact certificate exists, and Bel is handled by Solution 2.

---

## 5. Solution 2 – Exact Distance-Based Refinement

**Exact Bel, streamed.** The exact K1 nearest distances are obtained from the index for chunks of objects (≤ 2^19 neighbour entries per chunk), reduced immediately to the log-product of Eq. (10), and discarded; the first k_base columns are kept as Solution 1's lists. Memory is O(N·k_base + chunk) instead of O(N²).

**Restricted exact scans.** For a δ query not certified within k_max, the distance is computed only to the rank(i) objects with larger Bel; for a convergence query, only to the currently allocated objects.

**Exact θ.** Eq. (9) needs the 0.9-quantile of all N − 1 distances; it is computed by chunked `cdist` + `np.partition` with numpy's linear interpolation reproduced bit-for-bit (unit-tested), in O(N) memory. Optionally θ is estimated from a random reference sample of s = 1000 objects (`theta_sample`), which reduces this step from O(N²) to O(N s).

**Bel policies (ablation).** `all` (exact Bel for every object, default), `flagged` (only objects failing a two-scale model test – the previous project version's heuristic), `centres` (only the decision-graph contenders), `none` (head over k_cand = 64 + power-law tail extrapolation).

**What is and is not new.** Exact Euclidean distances and k-d-tree k-NN search are standard, and the original BFI already computes every distance; kNN-graph acceleration of density-peaks clustering also exists in the literature. The contribution here is the combination for BFI: (i) the sufficiency certificates for the two evidential neighbour queries, which make it possible to *know* when a candidate-derived answer is exact; (ii) refinement triggered only by a failed certificate and restricted to the relevant subset; (iii) a streamed exact Bel that reuses the same pass for the candidate lists; and (iv) the resulting exactness guarantee for the whole evidential output.

### 5.4 Documented deviations from the paper (identical in all variants)
* Centre rule γ = Bel·δ instead of the literal e^δ·δ (which is monotone in δ and selects isolated outliers). `center_rule="paper"` reproduces the literal rule.
* K2 = 3. With the literal Dempster fusion, K2 = 9 makes several far neighbours outweigh a near one; exp4 confirms the collapse (mean ARI 0.30 for the original at K2 = 9).
* Objects in a meta-cluster take the arg-max singleton as crisp label; data are min-max scaled.
* EP / ER / ERI are a simplified pairwise-credal variant (defined in `bfi/metrics.py`), not Denoeux's exact definition.

---

## 6. Experimental evaluation

### 6.1 Setup

**Variants.** `original`; `control` (fixed k = 64 list, no adaptivity, no refinement); `S1`; `S2` (fixed list + refinement); `S1+S2` (proposed). Accelerated variants run with **exact θ** (isolates the neighbour search) and with **sampled θ, s = 1000** (practical configuration).

**Datasets** (all min-max scaled; K1 = max(10, 0.06 N), K2 = 3, ζ = 0.1, q = 0.9):

| Group | Datasets | N | d | c |
|---|---|---|---|---|
| shapes | moons, circles, anisotropic | 3000 | 2 | 2/2/3 |
| separation / overlap | blobs_separated, blobs_overlap | 3000 | 2 | 3 |
| density | varied_density (std 1.0 / 2.5 / 0.5) | 3000 | 2 | 3 |
| high-dimensional | blobs_hd8, blobs_hd32 | 3000 | 8 / 32 | 5 |
| real | iris, wine, breast_cancer | 150 / 178 / 569 | 4 / 13 / 30 | 3/3/2 |
| real, high-dim | digits6 (PCA-10), digits10 (PCA-16) | 1083 / 1797 | 10 / 16 | 6/10 |
| scalability | 4-component Gaussian mixture | 1000 – 50 000 | 2 | 4 |

**Seeds.** 5 seeds in exp1, 3 in exp2–exp5. A seed changes the sampled points of a synthetic dataset (cluster layout fixed) and the θ sample. Real datasets are fixed, so their exact-θ results have zero variance by construction.

**Metrics.** Quality: ARI, NMI (also AMI, Re, Ri, pair P/R/RI). Efficiency: wall time (median of 3 in-process runs; 1 run for N > 6000), peak additional memory in a fresh process, speed-up = t_original / t_variant on the same data and seed. Evidence: EP, ER, ERI, Re (error), Ri (imprecision), and agreement with the original — label agreement after Hungarian alignment, mean total-variation distance of the pignistic probabilities (BetP TV), focal-set agreement. Neighbour search (each checked against the exact answer *in the method's own state*, `bfi/neighbour_metrics.py`): Bel rank correlation, candidate recall (does the initial list contain the true neighbour?), final selection accuracy (are the neighbours used the true ones?), and agreement of δ-neighbours / convergence-neighbour sets with the original run.

**Hardware / fairness.** Windows 11 laptop, Python 3.11, numpy 2.4, scipy 1.17, scikit-learn 1.9. All variants single-threaded (`n_jobs = 1`), because the original's numpy sort is single-threaded. Free RAM during exp2 was 0.47 GB, so the original was run up to N = 4000 (larger sizes would have swapped and distorted its timing; recorded in `exp2_raw.csv`).

### 6.2 Exactness and evidence preservation (exp1, 13 datasets × 5 seeds)

Averages over datasets of the per-dataset seed means (`results/tables/exp1_overall.md`):

| Configuration | ARI | NMI | label agreement | BetP TV | focal-set agreement | EP | ERI |
|---|---|---|---|---|---|---|---|
| original | 0.630 ± 0.328 | 0.674 ± 0.327 | 1 | 0 | 1 | 0.723 | 0.835 |
| control (exact θ) | 0.611 ± 0.274 | 0.639 ± 0.296 | 0.833 ± 0.087 | 0.167 | 0.826 | 0.730 | 0.826 |
| S1 (exact θ) | 0.636 ± 0.322 | 0.667 ± 0.320 | 0.959 ± 0.065 | 0.041 | 0.955 | 0.740 | 0.839 |
| S2 (exact θ) | **0.630 ± 0.328** | **0.674 ± 0.327** | **1.000 ± 0.000** | **0.000** | **1.000** | 0.723 | 0.835 |
| **S1+S2 (exact θ)** | **0.630 ± 0.328** | **0.674 ± 0.327** | **1.000 ± 0.000** | **0.000** | **1.000** | 0.723 | 0.835 |
| control (sampled θ) | 0.623 ± 0.280 | 0.645 ± 0.298 | 0.833 ± 0.090 | 0.167 | 0.826 | 0.744 | 0.832 |
| S1 (sampled θ) | 0.619 ± 0.323 | 0.655 ± 0.316 | 0.931 ± 0.097 | 0.069 | 0.927 | 0.724 | 0.830 |
| S2 (sampled θ) | 0.605 ± 0.331 | 0.661 ± 0.325 | 0.956 ± 0.085 | 0.044 | 0.955 | 0.701 | 0.822 |
| **S1+S2 (sampled θ)** | 0.605 ± 0.331 | 0.661 ± 0.325 | 0.956 ± 0.085 | 0.044 | 0.955 | 0.701 | 0.822 |

(± = standard deviation across the 13 datasets.) With exact θ the proposed method reproduces the original's evidential output on every dataset and seed — including the meta-cluster (imprecise) assignments, whose BBAs are identical. The 4.4 % disagreement of the sampled-θ configuration is caused entirely by θ sampling (it is identical for S2 and S1+S2, and zero with exact θ) and is concentrated in four high-dimensional datasets (§6.4).

ARI per dataset (mean ± std over 5 seeds; `results/tables/exp1_ari.md`, `plots/exp1_ari_nmi_theta_*.png`):

| dataset | original | control (exact θ) | S1 (exact θ) | **S1+S2 (exact θ)** | **S1+S2 (sampled θ)** |
|---|---|---|---|---|---|
| moons | 0.998 ± 0.003 | 0.530 ± 0.113 | 0.822 ± 0.115 | **0.998 ± 0.003** | **0.998 ± 0.003** |
| circles | 0.002 ± 0.003 | 0.003 ± 0.006 | 0.002 ± 0.001 | 0.002 ± 0.003 | 0.002 ± 0.003 |
| blobs_separated | 0.984 ± 0.005 | 0.776 ± 0.285 | 0.983 ± 0.006 | **0.984 ± 0.005** | **0.984 ± 0.006** |
| blobs_overlap | 0.629 ± 0.012 | 0.459 ± 0.116 | 0.624 ± 0.023 | **0.629 ± 0.012** | **0.632 ± 0.009** |
| varied_density | 0.866 ± 0.016 | 0.631 ± 0.214 | 0.872 ± 0.017 | **0.866 ± 0.016** | **0.865 ± 0.016** |
| anisotropic | 0.989 ± 0.005 | 0.709 ± 0.256 | 0.989 ± 0.006 | **0.989 ± 0.005** | **0.988 ± 0.006** |
| blobs_hd8 | 0.642 ± 0.075 | 0.708 ± 0.019 | 0.700 ± 0.004 | **0.642 ± 0.075** | 0.581 ± 0.120 |
| blobs_hd32 | 0.645 ± 0.171 | 0.801 ± 0.104 | 0.849 ± 0.087 | **0.645 ± 0.171** | 0.587 ± 0.168 |
| iris | 0.694 | 0.886 | 0.694 | **0.694** | **0.694** |
| wine | 0.443 | 0.835 | 0.443 | **0.443** | **0.443** |
| breast_cancer | 0.002 | 0.105 ± 0.230 | 0.002 | 0.002 | 0.002 |
| digits6_pca10 | 0.546 | 0.757 | 0.546 | **0.546** | 0.413 ± 0.132 |
| digits10_pca16 | 0.746 | 0.745 | 0.745 | **0.746** | 0.680 ± 0.090 |

### 6.3 Neighbour-search quality (exp1, `plots/exp1_neighbour_accuracy.png`)

Mean over 13 datasets, exact θ:

| | Bel rank corr. | δ candidate recall | δ selection acc. | conv. candidate recall | conv. selection acc. | δ-neighbour = original's | conv. set Jaccard vs original |
|---|---|---|---|---|---|---|---|
| control | 0.979 | 0.992 | 0.997 | 0.984 | 0.985 | 0.763 | 0.728 |
| S1 | 0.979 | 0.983 | **1.000** | 0.962 | 0.999 | 0.763 | 0.737 |
| S2 | **1.000** | 0.993 | **1.000** | 0.987 | **1.000** | **1.000** | **1.000** |
| **S1+S2** | **1.000** | 0.992 | **1.000** | 0.982 | **1.000** | 0.999 | 0.999 |

The candidate recall shows that the short initial lists (16 entries for S1 / S1+S2) already contain 98–99 % of the true δ- and convergence-neighbours; Solution 1's growth and Solution 2's scans take care of the rest, so the *final* selection accuracy is 1.000. Without refinement the Bel estimate (rank correlation 0.979) changes the ordering of objects, and 24 % of the δ-neighbours differ from the original's.

### 6.4 Efficiency (exp1 N = 3000; exp2 N = 1000–50 000)

**exp2 – scalability** (4-component mixture, mean of 3 seeds; `results/tables/exp2_scalability.md`, `plots/exp2_scalability.png`):

| N | original time / memory | S1+S2 sampled θ: time / memory | speed-up | S1+S2 exact θ: time / speed-up | ARI orig → S1+S2 | label agreement |
|---|---|---|---|---|---|---|
| 1 000 | 0.037 s / 25 MB | 0.031 s / 12 MB | 1.18 × | 0.031 s / 1.20 × | 0.917 → 0.917 | 1.000 |
| 2 000 | 0.116 s / 92 MB | 0.070 s / 9 MB | 1.67 × | 0.085 s / 1.36 × | 0.918 → 0.917 | 0.999 |
| 4 000 | 0.401 s / 366 MB | 0.173 s / 26 MB | **2.31 ×** | 0.274 s / 1.46 × | 0.917 → 0.917 | 0.999 |
| 6 000 | (not run, §6.1) | 0.343 s / 27 MB | – | 0.553 s | – → 0.919 | – |
| 10 000 | – | 0.742 s / 28 MB | – | 1.413 s | – → 0.919 | – |
| 20 000 | – | 2.53 s / 30 MB | – | 5.36 s | – → 0.923 | – |
| 50 000 | (≈ 20 GB for D alone) | 14.6 s / 38 MB | – | – | – → 0.924 | – |

The original's memory grows ×3.7–4.0 per doubling of N (quadratic); S1+S2's grows from 12 MB to 38 MB over a 50-fold increase of N. Speed-up grows with N while the original fits in memory.

**exp1 – N = 3000, 2-D datasets**: S1+S2 is 1.98–2.29 × faster (sampled θ) and 1.44–1.67 × (exact θ); peak memory 206 MB → 18 MB.

**Where the time goes** (S1+S2, sampled θ): at N = 4000 θ 0.04 s, Bel 0.08 s, δ 0.002 s, convergence 0.06 s; at N = 50 000 θ 0.46 s, **Bel 13.5 s**, δ 0.01 s, convergence 0.71 s. δ and the convergence — the steps Solution 1 accelerates — stay cheap at every size.

**Refinement work** (`plots/exp2_refinement_work.png`): S1+S2 needs about 8 exact convergence scans per run at every N from 2000 to 50 000 (S2: 8–12); without refinement the number of estimated queries grows with N (control: 12 → 1951; S1: 3 → 142).

### 6.5 Ablation (exp3: varied_density, blobs_overlap, moons, blobs_hd8; N = 4000; 3 seeds; exact θ)

`results/tables/exp3_ablation.md`, `plots/exp3_factorial.png`, `plots/exp3_ablation_parameters.png`.

**2 × 2 factorial** (mean ± std over 12 dataset-seed pairs):

| | speed-up | label agreement | BetP TV | conv. scans | estimated conv. queries | mean conv. list |
|---|---|---|---|---|---|---|
| control | 1.71 ± 0.25 | 0.835 | 0.165 | 0 | 56.3 | 64.0 |
| S1 | 1.61 ± 0.25 | 0.914 | 0.087 | 0 | 13.8 | 20.2 |
| S2 | 1.34 ± 0.25 | **1.000** | **0.000** | 12.3 | 0 | 66.5 |
| **S1+S2** | 1.37 ± 0.21 | **1.000** | **0.000** | **7.6** | 0 | **17.9** |

* **Effect of S1** (control → S1, S2 → S1+S2): quarter-size candidate lists (64 → 18), 4 × fewer estimated queries without refinement (56 → 14), 38 % fewer exact scans with refinement (12.3 → 7.6), agreement +0.08 without refinement; equal speed within noise.
* **Effect of S2** (control → S2, S1 → S1+S2): exactness (agreement 0.84 / 0.91 → 1.000), at a cost of about 20 % runtime at this N.

**Parameters** (decisions taken from these results):

| Parameter | Tested | Result | Chosen |
|---|---|---|---|
| k_min (S1+S2) | 4, 8, 16, 32, 64 | speed-up 1.11 / 1.35 / 1.43 / 1.43 / 1.46; always exact | **16** (fastest group, smallest lists) |
| k_max (S1+S2) | 32 … 1024 | speed-up 1.44–1.48 (flat); scans 19.3 → 4.2 | **512** |
| k_max (S1 alone) | 32 … 1024 | agreement 0.79 / 0.84 / 0.89 / 0.91 / **0.98** / 0.96 | 512 |
| fixed list k_cand (control) | 16 … 128 | agreement 0.69 → 0.93 | 64 (control only) |
| Bel policy | none / flagged / centres / all | agreement 0.943 / 0.949 / 0.893 / **1.000**; speed-up 1.73 / 1.66 / 1.69 / 1.46 | **all** — partial refinement never gives exactness, and refining only the centre contenders is worse than none |
| θ sample s | 100, 300, 1000, 3000, exact | speed-up 2.54 / 2.39 / 2.15 / 1.55 / 1.43; agreement 0.969 / 0.952 / **0.997** / 0.983 / 1.000 | **1000** (agreement is not monotone in s — see §7) |
| δ estimator (no refinement) | landmark / lower bound / geometric | control 0.835 / 0.712 / 0.859; S1 0.914 / 0.817 / 0.919 | landmark kept; landmark vs geometric is within seed std (≈ 0.1) |
| tail model (no refinement) | extrapolate / truncate | control 0.835 / 0.884; S1 0.914 / 0.917 | inconclusive: truncation wins only on varied_density |

### 6.6 Parameter sensitivity (exp4: blobs_overlap, varied_density, blobs_hd8, iris; 3 seeds; sampled θ)

| | K1 = 0.02 N … 0.15 N | K2 = 1 / 3 / 5 | K2 = 9 / 15 | ζ = 0 … 0.5 |
|---|---|---|---|---|
| original ARI | 0.730 → 0.592 | 0.717 / 0.712 / 0.610 | 0.302 / 0.081 | 0.712 |
| S1+S2 ARI | 0.730 → 0.573 | 0.716 / 0.699 / 0.611 | 0.249 / 0.028 | 0.699 |
| S1+S2 agreement with original | 0.977 – 0.999 | 0.994 / 0.982 / 0.999 | 0.899 / 0.900 | 0.982 |
| control agreement | 0.76 – 0.92 | 0.83 / 0.86 / 0.79 | 0.62 / 0.45 | 0.856 |

S1+S2 follows the original across the whole parameter space, including its failure at large K2. ζ changes only the evidential output (imprecision rate Ri 0 → 0.042 for the original, 0 → 0.040 for S1+S2; error rate Re 0.150 → 0.137 and 0.165 → 0.153), as the paper describes. The control's higher ARI at K2 ≥ 9 (≈ 0.65) is *not* preservation: it agrees with the original on only 45–62 % of the objects.

---

## 7. Limitations, failures and cases where the method is worse

1. **Exact Bel dominates at large N.** With K1 = 0.06 N the exact K1-NN distances are O(N·K1) = O(N²) work (13.5 s of 14.6 s at N = 50 000). The method removes the O(N²) *memory* and the O(N² log N) sort, not the quadratic number of Bel terms. A faster Bel needs an approximation, and our tail-extrapolation alternative loses quality as N grows (control / S1 ARI 0.27 / 0.40 at N = 50 000) — exactness is bought with time.
2. **Exact θ is inherently O(N²).** Eq. (9) is a quantile over all distances; no index can avoid that. The exact-θ configuration is only 1.2–1.5 × faster than the original at N ≤ 4000. Sampling θ removes this cost but is the only remaining approximation.
3. **θ sampling changes results on some data.** digits6: ARI 0.546 → 0.413 ± 0.132 (agreement 0.70); digits10: 0.746 → 0.680 ± 0.090; blobs_hd8 / hd32: −0.061 / −0.058. Agreement is not monotone in the sample size (s = 3000 < s = 1000 in exp3), indicating that BFI's sequential convergence amplifies small θ perturbations near decision boundaries, i.e. these datasets are fragile for BFI itself. Mitigation: exact θ (100 % agreement) when N is moderate.
4. **No speed-up in high dimension and for small N.** blobs_hd32 (d = 32): 0.61–0.75 × (slower); real datasets with N ≤ 1800: 0.78–1.03 ×. k-d trees degrade towards brute force in high dimension, and for small N the numpy baseline has no overhead to remove. The memory saving remains (digits10: 74 MB → 13 MB).
5. **Without Solution 2 quality is not reliable.** S1 alone drops ARI on moons (0.998 → 0.822) and at large N (0.396 at N = 50 000); the control is worse everywhere (§6.2). On iris / wine / blobs_hd32 the control *raises* ARI (e.g. wine 0.443 → 0.835), but because it agrees with the original on fewer objects this is a different clustering produced by approximation, not an improvement of BFI.
6. **Base-method failures are unchanged.** circles and breast_cancer have ARI ≈ 0 for every variant, including the original — a limitation of density-peak centres, not of the acceleration.
7. **Scope.** Single machine, single thread; the original could only be run to N = 4000 in exp2 because of the RAM available at that time (it was run at N = 6000 in exp1-style tests during development, 824 MB peak, but that measurement is not part of the recorded experiments). The Olivetti-faces and Berkeley-segmentation datasets of the paper were not used (they need downloads). EP/ER/ERI are a simplified variant.

---

## 8. Conclusion

The proposed combination of adaptive candidate retrieval with exactness certificates (Solution 1) and certificate-triggered exact refinement (Solution 2) makes BFI **memory-scalable** (quadratic → near-linear: 366 MB → 26 MB at N = 4000, 38 MB at N = 50 000) and **faster** (up to 2.3 × at the largest size where the original fits in memory) while **preserving its evidential output exactly** when θ is exact, and to 99.9 % label agreement on the scalability data with sampled θ. The factorial ablation shows that refinement is what guarantees exactness and that adaptive retrieval makes refinement cheap (38 % fewer scans, ¼ of the candidate memory). The study also identifies where the approach does not help: high-dimensional data, small data sets, and very large N where the K1 = 0.06 N exact Bel terms dominate.

**Future work.** A certified bound on the Bel tail (refine only objects whose bound interval could change the decision graph); parallel / compiled convergence loop; approximate-NN back-ends with recall guarantees for high dimension; the paper's image datasets; a principled choice of the θ sample size.

---

## Appendix A – Reproducing every number

```
python -m pytest -q tests          # 39 passed
python run_all.py                  # exp1-exp5, writes results/, results/tables/, results/plots/
```
| Experiment | Raw | Summary | Tables | Figures |
|---|---|---|---|---|
| exp1 | exp1_raw.csv | exp1_summary.csv | exp1_ari.md, exp1_nmi.md, exp1_efficiency.md, exp1_overall.md, exp1_quality_drops.md | exp1_ari_nmi_theta_exact/sampled.png, exp1_runtime.png, exp1_memory.png, exp1_neighbour_accuracy.png, exp1_agreement.png, exp1_clusters.png |
| exp2 | exp2_raw.csv | exp2_summary.csv | exp2_scalability.md | exp2_scalability.png, exp2_refinement_work.png |
| exp3 | exp3_raw.csv | exp3_summary.csv, exp3_summary_per_dataset.csv | exp3_ablation.md | exp3_factorial.png, exp3_ablation_parameters.png |
| exp4 | exp4_raw.csv | exp4_summary.csv | exp4_sensitivity.md | exp4_sensitivity.png |
| exp5 | exp5_raw.csv | exp5_summary.csv | exp5_bel_adaptivity.md | exp5_bel_adaptivity.png |

Logs of the runs: `results/log_exp1.txt` … `log_exp4_exp5.txt`. Results of the previous project version are archived in `results/_previous_version/`.

## Appendix B – Metric definitions
* **ARI / NMI / AMI** – scikit-learn, crisp labels = arg-max of BetP (Eq. 16).
* **Re** – fraction of objects whose (Hungarian-matched) true class is not in the chosen focal set; **Ri** – fraction whose true class is covered by a meta-cluster.
* **EP / ER / ERI** – pairwise-credal precision / recall / Rand index from the BBAs on a 600-object subsample (definition in `bfi/metrics.py`).
* **Label agreement** – fraction of objects with the same label as the original after Hungarian alignment of cluster ids; **BetP TV** – mean ½‖BetP − BetP_orig‖₁; **focal-set agreement** – fraction of objects with the identical focal set.
* **Candidate recall** – fraction of true δ-neighbours (resp. of the true K2 nearest allocated objects) contained in the initial candidate list; **selection accuracy** – fraction of queries whose final neighbours equal the exact ones (tie-safe), both checked in the method's own Bel order and allocation state; **Bel error / rank corr.** – vs. the exact-K1 Bel with the method's θ.
* **Speed-up** – t_original / t_variant on the same data and seed; **peak memory** – additional peak resident memory of one run in a fresh process.
