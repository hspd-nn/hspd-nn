from dataclasses import dataclass
import numpy as np
from .svec import svec_features


@dataclass
class CubicPNN:
    W: np.ndarray  # d x r, unit columns
    A: np.ndarray  # m x r
    b: np.ndarray  # r
    c: np.ndarray  # m


def normalize_cubic_parameters(W, A, b):
    """Normalize hidden directions while preserving A*(W^T x+b)^3."""
    norms = np.linalg.norm(W, axis=0)
    Wn = W / norms
    bn = b / norms
    An = A * (norms[None, :] ** 3)
    return Wn, An, bn


def random_cubic_pnn(d, r, m, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(d, r))
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    A = rng.normal(scale=0.7, size=(m, r)) / np.sqrt(max(r, 1))
    b = rng.normal(scale=0.35, size=r)
    c = rng.normal(scale=0.2, size=m)
    return CubicPNN(W, A, b, c)


def pnn_forward(X, W, A, b, c):
    Z = X @ W + b
    return Z ** 3 @ A.T + c


def tanh_forward(X, W, A, b, c):
    """Evaluate a shallow vector-valued tanh ridge network."""
    return np.tanh(X @ W + b) @ A.T + c


def pnn_hessian_matrix(X, W, A, b, noise=0.0, seed=0):
    """Stack exact Hessians for f_o(x)=sum_j A_oj(w_j^T x+b_j)^3+c_o."""
    N, d = X.shape
    m, r = A.shape
    Z = X @ W + b
    B = svec_features(W)
    C = np.concatenate([6.0 * Z * A[o][None, :] for o in range(m)], axis=0)
    M = B @ C.T
    if noise > 0:
        rng = np.random.default_rng(seed)
        E = rng.normal(size=M.shape)
        E *= noise * np.linalg.norm(M) / max(np.linalg.norm(E), 1e-15)
        M = M + E
    return M, C


def tanh_hessian_matrix(X, W, A, b, noise=0.0, seed=0):
    N, d = X.shape
    m, r = A.shape
    Z = X @ W + b
    T = np.tanh(Z)
    phi2 = -2.0 * T * (1.0 - T * T)
    B = svec_features(W)
    C = np.concatenate([phi2 * A[o][None, :] for o in range(m)], axis=0)
    M = B @ C.T
    if noise > 0:
        rng = np.random.default_rng(seed)
        E = rng.normal(size=M.shape)
        E *= noise * np.linalg.norm(M) / max(np.linalg.norm(E), 1e-15)
        M = M + E
    return M, C


def recover_cubic_parameters(M, X, W_est, Y_probe):
    """Analytic parameter reconstruction after hidden directions are recovered."""
    d, r = W_est.shape
    N = X.shape[0]
    m = Y_probe.shape[1]
    Bhat = svec_features(W_est)
    Ct, *_ = np.linalg.lstsq(Bhat, M, rcond=None)  # r x (mN)
    Ahat = np.zeros((m, r))
    bhat_per_output = np.full((m, r), np.nan)
    for o in range(m):
        block = Ct[:, o * N:(o + 1) * N].T  # N x r
        for j in range(r):
            z = X @ W_est[:, j]
            D = np.column_stack([z, np.ones(N)])
            slope, intercept = np.linalg.lstsq(D, block[:, j], rcond=None)[0]
            Ahat[o, j] = slope / 6.0
            if abs(slope) > 1e-10:
                bhat_per_output[o, j] = intercept / slope
    bhat = np.zeros(r)
    for j in range(r):
        valid = np.isfinite(bhat_per_output[:, j])
        if np.any(valid):
            weights = Ahat[valid, j] ** 2 + 1e-12
            bhat[j] = np.average(bhat_per_output[valid, j], weights=weights)
    raw = (X @ W_est + bhat) ** 3 @ Ahat.T
    chat = np.mean(Y_probe - raw, axis=0)
    return CubicPNN(W_est, Ahat, bhat, chat)


# ---------------------------------------------------------------------------
# Smooth (tanh) networks: reconstruction from the Hessian coefficient matrix
# ---------------------------------------------------------------------------

def tanh_d2(z):
    t = np.tanh(z)
    return -2.0 * t * (1.0 - t * t)


def coefficient_blocks(M, W_est, N, m):
    """Least-squares Hessian coefficients C^T = B^+ M, reshaped per unit to N x m blocks."""
    Bhat = svec_features(W_est)
    Ct, *_ = np.linalg.lstsq(Bhat, M, rcond=None)  # r x (m N), columns ordered (o, s)
    return [Ct[j].reshape(m, N).T for j in range(W_est.shape[1])]


def _fit_profile(z, u, scales, biases):
    """Best (s, b) such that tanh''(s z + b) is proportional to the unit vector u."""
    best = (-1.0, 1.0, 0.0)
    for s_ in scales:
        G = tanh_d2(s_ * z[:, None] + biases[None, :])           # N x nb
        nrm = np.linalg.norm(G, axis=0)
        cosv = np.abs(u @ G) / np.maximum(nrm, 1e-300)
        k = int(np.argmax(cosv))
        if cosv[k] > best[0]:
            best = (float(cosv[k]), float(s_), float(biases[k]))
    return best


def recover_tanh_parameters(M, X, W_est, Y_probe, scales=None, refine=True):
    """Reconstruct a shallow tanh network from Hessians once directions are known.

    For each unit j the coefficient block C_j (probes x outputs) equals
    tanh''(s_j X w_j + b_j) (s_j^2 a_j)^T, a rank-one matrix.  Its leading left
    singular vector gives the curvature profile, from which the scale s_j and
    bias b_j are fitted by a two-parameter search; output weights follow by
    least squares and the offset from function values at the probes.
    Returns (W_scaled, A, b, c) for f(x) = tanh(x W_scaled + b) A^T + c.
    """
    d, r = W_est.shape
    N = X.shape[0]
    m = Y_probe.shape[1]
    blocks = coefficient_blocks(M, W_est, N, m)
    if scales is None:
        scales = np.geomspace(0.05, 8.0, 41)
    biases = np.linspace(-8.0, 8.0, 321)
    s_hat = np.ones(r); b_hat = np.zeros(r); A_hat = np.zeros((m, r)); fit_cos = np.zeros(r)
    from scipy.optimize import minimize as _min
    for j in range(r):
        Cj = blocks[j]
        U_, S_, Vt_ = np.linalg.svd(Cj, full_matrices=False)
        u = U_[:, 0]
        z = X @ W_est[:, j]
        c0, s0, b0 = _fit_profile(z, u, scales, biases)

        def neg(p):
            g = tanh_d2(p[0] * z + p[1])
            n = np.linalg.norm(g)
            return -abs(u @ g) / max(n, 1e-300)
        if refine:
            res = _min(neg, np.array([s0, b0]), method='Nelder-Mead',
                       options={'xatol': 1e-10, 'fatol': 1e-14, 'maxiter': 2000})
            s0, b0 = float(res.x[0]), float(res.x[1])
            c0 = -float(res.fun)
        if s0 < 0:  # tanh is odd: (s, b, a) and (-s, -b, -a) give the same unit
            s0, b0 = -s0, -b0
        g = tanh_d2(s0 * z + b0)
        # C_j ~ g (s^2 a_j)^T  ->  a_j = C_j^T g / (s^2 ||g||^2)
        A_hat[:, j] = Cj.T @ g / max(s0 * s0 * (g @ g), 1e-300)
        s_hat[j], b_hat[j], fit_cos[j] = s0, b0, c0
    Ws = W_est * s_hat[None, :]
    c_hat = np.mean(Y_probe - np.tanh(X @ Ws + b_hat) @ A_hat.T, axis=0)
    return Ws, A_hat, b_hat, c_hat, fit_cos


def refine_tanh_student(X, Y, Ws, A, b, c, fix_directions=True, steps=200):
    """Polish (scales, biases, output weights, offset) by L-BFGS on function values.

    With ``fix_directions`` the unit directions stay at their HSPD estimates and
    only per-unit scales are adapted.
    """
    import torch
    Xt = torch.tensor(X, dtype=torch.float64)
    Yt = torch.tensor(Y, dtype=torch.float64)
    norms = np.linalg.norm(Ws, axis=0)
    U = torch.tensor(Ws / norms, dtype=torch.float64)
    s = torch.tensor(norms, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    At = torch.tensor(A, dtype=torch.float64, requires_grad=True)
    ct = torch.tensor(c, dtype=torch.float64, requires_grad=True)
    params = [s, bt, At, ct]
    if not fix_directions:
        U.requires_grad_(True)
        params.append(U)
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=steps, tolerance_grad=1e-12,
                            tolerance_change=1e-15, history_size=50, line_search_fn='strong_wolfe')
    scale = float(torch.mean(Yt ** 2))

    def closure():
        opt.zero_grad()
        pred = torch.tanh(Xt @ (U * s) + bt) @ At.T + ct
        loss = torch.mean((pred - Yt) ** 2) / scale
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        Wn = (U * s).numpy().copy()
        return Wn, At.numpy().copy(), bt.numpy().copy(), ct.numpy().copy()


def random_tanh_net(d, r, m, seed=0, bias_scale=0.4):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(d, r)); W /= np.linalg.norm(W, axis=0, keepdims=True)
    A = rng.normal(size=(m, r)) / np.sqrt(r)
    b = rng.normal(scale=bias_scale, size=r)
    c = rng.normal(scale=0.2, size=m)
    return W, A, b, c
