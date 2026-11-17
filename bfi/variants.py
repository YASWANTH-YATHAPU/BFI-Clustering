"""Named configurations compared in the experiments (single source of truth).

  original        bfi_original - full N x N matrix (reference)
  control         fixed k-NN candidate graph, no adaptivity, no refinement (naive kNN-BFI)
  S1              Solution 1 only  - adaptive candidate retrieval, estimates where uncertified
  S2              Solution 2 only  - fixed candidate list + exact refinement
  S1+S2           both solutions   - proposed method
All accelerated variants share the same theta setting within an experiment (passed separately),
so differences between them are caused by the neighbour search only.
"""
from .fast import bfi_fast
from .original import bfi_original

# Defaults selected from the ablation study (experiments/exp3_ablation.py, see README / REPORT).
DEFAULTS = dict(k_cand=64, k_min=16, k_max=512, bel_refine="all", n_jobs=1)

VARIANTS = {
    "original": None,
    "control": dict(adaptive=False, refine=False),
    "S1": dict(adaptive=True, refine=False),
    "S2": dict(adaptive=False, refine=True),
    "S1+S2": dict(adaptive=True, refine=True),
}
LABELS = {"original": "Original BFI", "control": "kNN-BFI control (fixed k, no refinement)",
          "S1": "BFI + S1 (adaptive retrieval)", "S2": "BFI + S2 (exact refinement)",
          "S1+S2": "BFI + S1 + S2 (proposed)"}
ORDER = ["original", "control", "S1", "S2", "S1+S2"]


def run_variant(name, X, c, K1, K2, zeta, theta_sample=None, seed=0, **overrides):
    """Run a named variant.  `overrides` change accelerated-variant parameters (ablation)."""
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; choose one of {list(VARIANTS)}")
    if name == "original":
        return bfi_original(X, c, K1, K2=K2, zeta=zeta)
    p = dict(DEFAULTS, **VARIANTS[name])
    p.update(overrides)
    return bfi_fast(X, c, K1, K2=K2, zeta=zeta, theta_sample=theta_sample, seed=seed, **p)
