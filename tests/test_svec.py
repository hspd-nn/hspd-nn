import numpy as np
from hspd.svec import svec, smat, svec_features


def test_svec_isometry_and_inverse():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5)); A = (A + A.T) / 2
    v = svec(A)
    assert np.allclose(smat(v, 5), A)
    assert np.allclose(np.linalg.norm(v), np.linalg.norm(A, 'fro'))


def test_svec_features():
    rng = np.random.default_rng(1)
    W = rng.normal(size=(4, 3))
    B = svec_features(W)
    for j in range(W.shape[1]):
        assert np.allclose(B[:, j], svec(np.outer(W[:, j], W[:, j])))
