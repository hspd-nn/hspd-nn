import numpy as np
from hspd.pnn import random_cubic_pnn, pnn_hessian_matrix, pnn_forward, recover_cubic_parameters
from hspd.metrics import relative_rmse


def test_parameter_recovery_given_directions():
    d, r, m, N = 5, 7, 3, 30
    net = random_cubic_pnn(d, r, m, seed=7)
    rng = np.random.default_rng(8)
    X = rng.normal(size=(N, d))
    M, _ = pnn_hessian_matrix(X, net.W, net.A, net.b)
    Y = pnn_forward(X, net.W, net.A, net.b, net.c)
    est = recover_cubic_parameters(M, X, net.W, Y)
    Xte = rng.normal(size=(50, d))
    Yte = pnn_forward(Xte, net.W, net.A, net.b, net.c)
    Yhat = pnn_forward(Xte, est.W, est.A, est.b, est.c)
    assert relative_rmse(Yhat, Yte) < 1e-10
