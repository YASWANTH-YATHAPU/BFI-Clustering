"""Result container shared by the original and the fast BFI implementations."""
from dataclasses import dataclass, field
import numpy as np
from .evidence import pignistic, decision_sets


@dataclass
class BFIResult:
    M: np.ndarray          # (N, c) cluster-level support m^phi for every object (0 = not in M_t)
    bel: np.ndarray        # (N,) Bel_i(C): belief that x_i is a cluster centre
    delta: np.ndarray      # (N,) delta_i
    centers: np.ndarray    # (c,) indices of cluster centres (ordered by gamma)
    theta: np.ndarray      # (N,) distance factor of Eq. (9)
    info: dict = field(default_factory=dict)   # timings / counters (scalars)
    # per-object neighbour diagnostics (arrays), used by bfi.neighbour_metrics:
    #   delta_nn (N,)      object used as the delta-neighbour (-1 for the top-Bel object)
    #   conv_nn  (N, K2)   allocated neighbours used in the convergence (-1 padding / centres)
    #   delta_src, conv_src (N,) int8 source codes, see bfi.fast.SRC
    #   bel_exact (N,) bool  Bel computed from exact K1 distances
    diag: dict = field(default_factory=dict)

    @property
    def betp(self):
        if "_betp" not in self.__dict__:
            self.__dict__["_betp"] = pignistic(self.M)
        return self.__dict__["_betp"]

    @property
    def labels(self):
        """Crisp labels from the pignistic probability (Eq. 16)."""
        return self.betp.argmax(1)

    @property
    def sets(self):
        """(N, c) boolean matrix: focal element (singleton or meta-cluster) chosen for each object."""
        if "_sets" not in self.__dict__:
            self.__dict__["_sets"] = decision_sets(self.M)
        return self.__dict__["_sets"]

    @property
    def imprecise(self):
        return self.sets.sum(1) > 1
