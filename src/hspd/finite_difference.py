"""Black-box Hessians by central finite differences of function queries."""
import numpy as np
from .svec import svec


def fd_stencil(d, h):
    """Query offsets for one probe: center, +-h e_i, and +-h e_i +-h e_j (i<j)."""
    E = np.eye(d) * h
    offs = [np.zeros(d)]
    offs += [E[i] for i in range(d)] + [-E[i] for i in range(d)]
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    for i, j in pairs:
        offs += [E[i] + E[j], E[i] - E[j], -E[i] + E[j], -E[i] - E[j]]
    return np.asarray(offs), pairs


def queries_per_probe(d):
    return 1 + 2 * d + 4 * (d * (d - 1) // 2)  # = 2 d^2 + 1


def finite_difference_hessians(func, X, output_dim, h=1e-2, noise_std=0.0, seed=0,
                               max_batch=2 ** 22):
    """Central finite-difference Hessians of every output at every probe.

    ``func`` maps an (n, d) array of inputs to an (n, m) array of outputs; one
    query returns all outputs.  ``noise_std`` adds i.i.d. Gaussian noise to
    every returned output value before differencing.  Returns the p x (m N)
    matrix of symmetric-vectorized Hessians (columns ordered by output, then
    probe) and the number of queries used.
    """
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    p = d * (d + 1) // 2
    offs, pairs = fd_stencil(d, h)
    q = offs.shape[0]
    rng = np.random.default_rng(seed)
    M = np.empty((p, output_dim * N))
    per_chunk = max(1, int(max_batch // (q * d)))
    ii = np.array([i for i, _ in pairs], dtype=int)
    jj = np.array([j for _, j in pairs], dtype=int)
    for lo in range(0, N, per_chunk):
        Xc = X[lo:lo + per_chunk]
        Q = (Xc[:, None, :] + offs[None, :, :]).reshape(-1, d)
        Fq = np.asarray(func(Q), dtype=float).reshape(len(Xc), q, output_dim)
        if noise_std > 0:
            Fq = Fq + noise_std * rng.normal(size=Fq.shape)
        f0 = Fq[:, 0]
        fp = Fq[:, 1:1 + d]
        fm = Fq[:, 1 + d:1 + 2 * d]
        H = np.zeros((len(Xc), output_dim, d, d))
        diag = (fp - 2.0 * f0[:, None, :] + fm) / (h * h)       # n x d x m
        H[:, :, np.arange(d), np.arange(d)] = np.transpose(diag, (0, 2, 1))
        if pairs:
            mix = Fq[:, 1 + 2 * d:].reshape(len(Xc), len(pairs), 4, output_dim)
            val = (mix[:, :, 0] - mix[:, :, 1] - mix[:, :, 2] + mix[:, :, 3]) / (4.0 * h * h)
            val = np.transpose(val, (0, 2, 1))                    # n x m x pairs
            H[:, :, ii, jj] = val
            H[:, :, jj, ii] = val
        sv = svec(H)                                              # n x m x p
        for o in range(output_dim):
            M[:, o * N + lo:o * N + lo + len(Xc)] = sv[:, o, :].T
    return M, N * q
