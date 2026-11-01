import numpy as np
from hspd.pnn import random_cubic_pnn, pnn_hessian_matrix
from hspd.hspd import hspd_recover
from hspd.metrics import match_directions


def test_noiseless_overcomplete_recovery():
    d, r, m, N = 6, 10, 3, 24  # r>d but below d(d-1)/2=15
    net = random_cubic_pnn(d, r, m, seed=123)
    X = np.random.default_rng(456).normal(size=(N, d))
    M, C = pnn_hessian_matrix(X, net.W, net.A, net.b)
    assert np.linalg.matrix_rank(C) == r
    rec = hspd_recover(M, d, r, seed=789, n_starts=500, max_iter=220)
    mean_cos, min_cos, *_ = match_directions(net.W, rec['W'])
    assert rec['W'].shape[1] == r
    assert mean_cos > 0.999
    assert min_cos > 0.995
