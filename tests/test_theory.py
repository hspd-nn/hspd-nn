import numpy as np
from hspd.svec import svec, smat, svec_features
from hspd.hspd import kappa, power_iterate, tangent_operator, hspd_recover, score
from hspd.pnn import random_tanh_net, tanh_hessian_matrix, tanh_forward, recover_tanh_parameters
from hspd.finite_difference import finite_difference_hessians, queries_per_probe
from hspd.metrics import direction_summary, relative_rmse


def _basis(W):
    return np.linalg.qr(svec_features(W))[0]


def test_kappa_positive_up_to_codim_plus_one_and_zero_beyond():
    rng = np.random.default_rng(0)
    d = 6
    codim = d * (d - 1) // 2
    for r, positive in ((codim, True), (codim + 1, True), (codim + 2, False)):
        W = rng.normal(size=(d, r)); W /= np.linalg.norm(W, axis=0)
        k = min(kappa(W[:, j], _basis(W)) for j in range(r))
        assert (k > 1e-6) == positive


def test_local_rate_matches_kappa_formula():
    rng = np.random.default_rng(1)
    d, r = 6, 14  # close to codim = 15, so convergence is slow enough to measure
    W = rng.normal(size=(d, r)); W /= np.linalg.norm(W, axis=0)
    U = _basis(W); Q = smat(U.T, d)
    w = W[:, 0]
    k = kappa(w, U)
    assert 1 - k ** 2 / 2 > 0.5
    v = w + 1e-7 * rng.normal(size=d); v /= np.linalg.norm(v)
    errs = []
    for _ in range(600):
        V, *_ = power_iterate(Q, v[None, :], gamma=0.0, max_iter=1, tol=0.0, fixed_iter=True)
        v = V[0]
        errs.append(np.linalg.norm(v - np.sign(v @ w) * w))
    rate = (errs[-1] / errs[-101]) ** 0.01  # slowest mode dominates late iterations
    assert abs(rate - (1 - k ** 2 / 2)) < 1e-3


def test_shift_one_half_is_monotone():
    rng = np.random.default_rng(2)
    for _ in range(50):
        d = int(rng.integers(3, 8)); r = int(rng.integers(2, d * (d + 1) // 2))
        U = np.linalg.qr(rng.normal(size=(d * (d + 1) // 2, r)))[0]
        V0 = rng.normal(size=(10, d)); V0 /= np.linalg.norm(V0, axis=1, keepdims=True)
        *_, nonmono = power_iterate(smat(U.T, d), V0, gamma=0.5, max_iter=100, tol=0.0, fixed_iter=True)
        assert nonmono.sum() == 0


def test_tanh_single_output_recovery_with_enough_probes():
    d, r, m, N = 6, 10, 1, 16
    W, A, b, c = random_tanh_net(d, r, m, seed=3)
    X = np.random.default_rng(4).normal(size=(N, d))
    M, C = tanh_hessian_matrix(X, W, A, b)
    assert np.linalg.matrix_rank(C) == r
    res = hspd_recover(M, d, r, seed=5, polish='newton')
    summ, *_ = direction_summary(W, res['W'], res['certified'])
    assert summ['success'] and summ['n_false_certified'] == 0


def test_tanh_reconstruction_from_exact_directions():
    d, r, m, N = 6, 8, 3, 30
    W, A, b, c = random_tanh_net(d, r, m, seed=6)
    X = np.random.default_rng(7).normal(size=(N, d))
    M, _ = tanh_hessian_matrix(X, W, A, b)
    Y = tanh_forward(X, W, A, b, c)
    Ws, A2, b2, c2, _ = recover_tanh_parameters(M, X, W, Y)
    Xte = np.random.default_rng(8).normal(size=(100, d))
    assert relative_rmse(np.tanh(Xte @ Ws + b2) @ A2.T + c2, tanh_forward(Xte, W, A, b, c)) < 1e-6


def test_noisy_queries_count_and_error_scaling():
    d, r, m, N = 4, 3, 2, 5
    W, A, b, c = random_tanh_net(d, r, m, seed=9)
    X = np.random.default_rng(10).normal(size=(N, d))
    Mex, _ = tanh_hessian_matrix(X, W, A, b)
    f = lambda Z: tanh_forward(Z, W, A, b, c)
    M, nq = finite_difference_hessians(f, X, m, h=1e-3)
    assert nq == N * queries_per_probe(d) == N * (2 * d * d + 1)
    assert np.linalg.norm(M - Mex) / np.linalg.norm(Mex) < 1e-5
    Mn, _ = finite_difference_hessians(f, X, m, h=1e-3, noise_std=1e-6, seed=1)
    assert np.linalg.norm(Mn - Mex) / np.linalg.norm(Mex) > 1e-3  # sigma/h^2 amplification
