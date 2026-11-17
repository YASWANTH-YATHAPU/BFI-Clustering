import numpy as np
import pytest
from scipy.spatial.distance import cdist
from bfi import bfi_original, bfi_fast
from bfi.candidates import adaptive_nearest_higher, adaptive_allocated, first_higher_in_lists
from bfi.compare import compare_results
from bfi.datasets import synthetic, high_dim, all_datasets
from bfi.evidence import fuse_neighbours, pignistic, decision_sets, full_bba
from bfi.metrics import evaluate
from bfi.neighbors import Searcher
from bfi.neighbour_metrics import neighbour_quality
from bfi.refine import exact_theta, exact_bel_streamed, exact_nearest_higher, exact_k_nearest
from bfi.variants import run_variant, VARIANTS


@pytest.fixture(scope="module")
def data():
    return synthetic(800)["varied_density"]


@pytest.fixture(scope="module")
def ref(data):
    X, y = data
    return bfi_original(X, 3, 48, K2=3)


def _same(a, b):
    assert np.allclose(a.bel, b.bel, rtol=0, atol=1e-12)
    assert np.allclose(a.delta, b.delta, rtol=0, atol=1e-12)
    assert np.array_equal(a.centers, b.centers)
    assert np.allclose(a.M, b.M, rtol=0, atol=1e-12)


# ---------------------------------------------------------------- exactness of the integration
def test_exact_mode_reproduces_original(data, ref):
    """With k_cand = K1 and exact theta the fast algorithm must equal the original."""
    X, y = data
    _same(ref, bfi_fast(X, 3, 48, K2=3, k_cand=48, theta_sample=None))


@pytest.mark.parametrize("adaptive", [True, False])
@pytest.mark.parametrize("k_min,k_max", [(3, 3), (4, 8), (16, 256), (32, 799)])
def test_refined_variants_are_exact(data, ref, adaptive, k_min, k_max):
    """S2 and S1+S2 (exact Bel, exact theta) reproduce the original for ANY candidate sizes,
    including tiny budgets that force many exact-scan refinements."""
    X, y = data
    r = bfi_fast(X, 3, 48, K2=3, theta_sample=None, adaptive=adaptive, refine=True,
                 bel_refine="all", k_cand=8, k_min=k_min, k_max=k_max)
    _same(ref, r)
    assert np.array_equal(r.diag["delta_nn"], ref.diag["delta_nn"])


def test_s1_unbounded_budget_is_exact_without_refinement(data):
    """Solution 1 alone: if the budget covers the whole data set every delta / convergence query is
    certified by the index, so with an exact Bel the result equals the original without any scan."""
    X, y = data
    ref = bfi_original(X, 3, 30, K2=3)
    r = bfi_fast(X, 3, 30, K2=3, theta_sample=None, adaptive=True, refine=False,
                 k_cand=30, k_min=4, k_max=10 ** 6)
    _same(ref, r)
    assert r.info["n_delta_est"] == 0 and r.info["n_conv_est"] == 0
    assert r.info["n_delta_scan"] == 0 and r.info["n_conv_scan"] == 0


def test_s1_needs_fewer_scans_than_s2(data):
    X, y = data
    s2 = bfi_fast(X, 3, 48, K2=3, adaptive=False, refine=True, k_cand=16, theta_sample=None)
    both = bfi_fast(X, 3, 48, K2=3, adaptive=True, refine=True, k_min=16, theta_sample=None)
    n = lambda r: r.info["n_conv_scan"] + r.info["n_delta_scan"]
    assert n(both) < n(s2)
    _same(s2, both)


def test_approximation_stays_close(data, ref):
    X, y = data
    b = bfi_fast(X, 3, 48, K2=3)
    cmp = compare_results(ref, b, X)
    assert cmp["bel_spearman"] > 0.99 and cmp["label_agree"] > 0.9


@pytest.mark.parametrize("name", list(VARIANTS))
def test_all_variants_run_and_are_reproducible(data, name):
    X, y = data
    a = run_variant(name, X, 3, 48, 3, 0.1, theta_sample=200, seed=7)
    b = run_variant(name, X, 3, 48, 3, 0.1, theta_sample=200, seed=7)
    assert np.array_equal(a.labels, b.labels) and np.allclose(a.M, b.M)
    assert a.M.shape == (len(X), 3) and np.all(a.labels >= 0)


# ---------------------------------------------------------------- Solution 1 components
def _brute_nearest_higher(X, rank, i):
    hi = np.flatnonzero(rank < rank[i])
    d = np.linalg.norm(X[hi] - X[i], axis=1)
    return hi[d.argmin()], d.min()


def test_adaptive_nearest_higher_is_exact():
    rng = np.random.default_rng(0)
    X = rng.random((500, 3))
    rank = rng.permutation(500)
    sr = Searcher(X)
    rows = np.flatnonzero(rank > 0)
    found, nn, d, k_used, d_last = adaptive_nearest_higher(sr, rows, rank, 2, 499)
    assert found.all()
    for r, j, dj in zip(rows, nn, d):
        bj, bd = _brute_nearest_higher(X, rank, r)
        assert j == bj and np.isclose(dj, bd)
    # with a tiny budget unresolved rows report a valid lower bound
    found, nn, d, k_used, d_last = adaptive_nearest_higher(sr, rows, rank, 2, 4)
    for r, f, lb in zip(rows, found, d_last):
        if not f:
            assert _brute_nearest_higher(X, rank, r)[1] >= lb - 1e-12


def test_first_higher_certificate_on_sorted_lists():
    rng = np.random.default_rng(1)
    X = rng.random((300, 2))
    rank = rng.permutation(300)
    idx, dist = Searcher(X).query(np.arange(300), 20)
    found, nn, d = first_higher_in_lists(idx, dist, rank)
    for i in np.flatnonzero(found):
        assert nn[i] == _brute_nearest_higher(X, rank, i)[0]


def test_adaptive_allocated_is_exact():
    rng = np.random.default_rng(2)
    X = rng.random((400, 2))
    lab = np.where(rng.random(400) < 0.05, 0, -1)      # 5 % allocated -> lists must grow
    sr = Searcher(X)
    bidx, bdist = sr.query(np.arange(400), 4)
    for t in np.flatnonzero(lab < 0)[:60]:
        status, nb, nd, k = adaptive_allocated(sr, t, bidx[t], bdist[t], lab, 3, 399)
        assert status in ("base", "expanded")
        alloc = np.flatnonzero(lab >= 0)
        ex, exd = exact_k_nearest(X, t, alloc, 3)
        assert np.allclose(np.sort(nd), exd)


# ---------------------------------------------------------------- Solution 2 components
def test_exact_theta_matches_definition():
    X = synthetic(300)["moons"][0]
    Ds = np.sort(cdist(X, X), axis=1)
    ref = 1.0 / np.maximum(np.quantile(Ds[:, 1:], 0.9, axis=1), 1e-12)
    assert np.array_equal(exact_theta(X, 0.9), ref)
    assert np.array_equal(exact_theta(X, 0.9, chunk_entries=1000), ref)   # chunking invariant


def test_exact_bel_streamed_matches_matrix():
    X = synthetic(400)["blobs_overlap"][0]
    th = exact_theta(X, 0.9)
    K1 = 30
    Ds = np.sort(cdist(X, X), axis=1)[:, 1:K1 + 1]
    ref = np.log1p(-np.exp(-th[:, None] * Ds) / K1).sum(1)
    lp, kidx, kdist = exact_bel_streamed(Searcher(X), th, K1, keep=5)
    assert np.allclose(lp, ref, atol=1e-12) and kidx.shape == (400, 5)
    assert np.allclose(kdist, Ds[:, :5])


def test_exact_scans():
    rng = np.random.default_rng(3)
    X = rng.random((200, 2))
    order = rng.permutation(200)
    rank = np.empty(200, int); rank[order] = np.arange(200)
    for u in order[1:20]:
        assert exact_nearest_higher(X, u, order, rank)[0] == _brute_nearest_higher(X, rank, u)[0]
    nb, nd = exact_k_nearest(X, 0, np.arange(1, 200), 4)
    d = np.linalg.norm(X[1:] - X[0], axis=1)
    assert np.allclose(nd, np.sort(d)[:4])


# ---------------------------------------------------------------- metrics
def test_neighbour_metrics_of_original_are_perfect(data, ref):
    X, y = data
    q = neighbour_quality(X, ref, 48, 3, ref=ref, n_eval=300)
    for k in ("bel_exact_frac", "bel_rho", "delta_acc", "delta_cand_recall", "conv_acc",
              "conv_cand_recall", "delta_nn_agree_ref", "conv_jaccard_ref"):
        assert q[k] == pytest.approx(1.0), k
    assert q["bel_err"] == pytest.approx(0.0) and q["delta_relerr"] == pytest.approx(0.0)


def test_neighbour_metrics_detect_estimates(data, ref):
    X, y = data
    r = bfi_fast(X, 3, 48, K2=3, theta_sample=None, adaptive=False, refine=False, k_cand=8)
    q = neighbour_quality(X, r, 48, 3, k_base=r.info["k_base"], ref=ref, n_eval=400)
    assert q["bel_exact_frac"] == 0.0 and q["bel_err"] > 0
    assert q["delta_cand_recall"] < 1.0


def test_evaluate_perfect_labels():
    X, y = synthetic(600)["blobs_separated"]
    r = bfi_original(X, 3, 36, K2=3, zeta=0.0)
    e = evaluate(r, y)
    for k in ("ARI", "NMI", "AMI", "EP", "ER", "ERI", "Re", "Ri"):
        assert np.isfinite(e[k]), k
    assert 0 <= e["Re"] <= 1 and e["Ri"] == 0.0           # zeta = 0 -> no meta-clusters


# ---------------------------------------------------------------- validation / data
@pytest.mark.parametrize("kw", [dict(K1=0), dict(K2=0), dict(zeta=1.5), dict(q=1.0),
                                dict(n_clusters=0), dict(n_clusters=10 ** 6), dict(K1=2.5)])
def test_input_validation(data, kw):
    X, y = data
    args = dict(n_clusters=3, K1=10, K2=3, zeta=0.1, q=0.9)
    args.update(kw)
    for fn in (bfi_original, bfi_fast):
        with pytest.raises(ValueError):
            fn(X, args["n_clusters"], args["K1"], K2=args["K2"], zeta=args["zeta"], q=args["q"])


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        bfi_fast(np.array([[0.0, np.nan], [1, 1], [2, 2]]), 2, 1)
    with pytest.raises(ValueError):
        bfi_fast(np.zeros(10), 2, 1)
    X = np.random.default_rng(0).random((50, 2))
    for kw in (dict(bel_refine="x"), dict(k_min=10, k_max=5), dict(backend="x"),
               dict(tail="x"), dict(delta_estimate="x")):
        with pytest.raises(ValueError):
            bfi_fast(X, 2, 5, **kw)


def test_duplicate_points_are_handled():
    X = np.repeat(np.random.default_rng(0).random((40, 2)), 3, axis=0)
    r = bfi_fast(X, 2, 10, K2=3, theta_sample=None)
    assert np.all(np.isfinite(r.bel)) and np.all(np.isfinite(r.M))


def test_seeds_change_sampling_not_structure():
    a, b = synthetic(500, 0)["varied_density"], synthetic(500, 1)["varied_density"]
    assert not np.allclose(a[0], b[0])
    assert set(all_datasets(0, 300)) == set(all_datasets(1, 300))
    X, y = high_dim(300)["blobs_hd32"]
    assert X.shape == (300, 32) and X.min() >= 0 and X.max() <= 1


# ---------------------------------------------------------------- evidence (unchanged logic)
def test_zeta_zero_has_no_metaclusters(data):
    X, y = data
    assert not bfi_original(X, 3, 48, K2=3, zeta=0.0).imprecise.any()


def test_bba_properties():
    M = np.array([[0.93, 0.94, 0.0], [0.9, 0.0, 0.0], [0.4, 0.3, 0.0]])
    B, masks = full_bba(M)
    assert np.allclose(B.sum(1), 1)
    S = decision_sets(M)
    assert S[0].tolist() == [True, True, False]       # two strong, conflicting neighbours -> meta-cluster
    assert S[1].tolist() == [True, False, False]
    assert S[2].tolist() == [True, False, False]      # weak evidence -> arg-max singleton
    P = pignistic(M)
    assert np.allclose(P.sum(1), 1) and P[0, 0] < P[0, 1] + 1e-9 + 0.05


def test_fusion_same_cluster_increases_support():
    one = fuse_neighbours(np.array([0]), np.array([0.2]), 2.0, 0.1, 2)[0]
    two = fuse_neighbours(np.array([0, 0]), np.array([0.2, 0.2]), 2.0, 0.1, 2)[0]
    assert two > one
