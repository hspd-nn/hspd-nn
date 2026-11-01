import numpy as np
from hspd.pnn import random_cubic_pnn, pnn_forward, pnn_hessian_matrix
from hspd.finite_difference import finite_difference_hessians


def test_cubic_finite_difference_hessians_are_exact_up_to_roundoff():
    d, r, m, N = 4, 5, 2, 3
    net = random_cubic_pnn(d, r, m, seed=11)
    X = np.random.default_rng(12).normal(size=(N, d))
    exact, _ = pnn_hessian_matrix(X, net.W, net.A, net.b)
    func = lambda Z: pnn_forward(Z, net.W, net.A, net.b, net.c)
    approx, _ = finite_difference_hessians(func, X, m, h=1e-3)
    rel = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    assert rel < 1e-7
