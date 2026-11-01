"""Trained teachers on real data and gradient-based distillation students."""
from __future__ import annotations

import numpy as np


def load_dataset(name, seed, test_size=0.30):
    """Standardized stratified split of a scikit-learn dataset (no download needed)."""
    from sklearn.datasets import load_wine, load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    data = {'wine': load_wine, 'digits': load_digits}[name]()
    X0 = data.data.astype(np.float64)
    y0 = data.target.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X0, y0, test_size=test_size, random_state=seed, stratify=y0)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte, int(len(np.unique(y0)))


def _act(name):
    import torch
    return {'tanh': torch.tanh, 'cubic': lambda z: z ** 3}[name]


def train_teacher(Xtr, ytr, r, m, activation='tanh', seed=0, epochs=1500, lr=0.01,
                  weight_decay=1e-4):
    """Full-batch Adam training of a shallow network f(x) = phi(x W + b) A^T + c.

    For the cubic teacher the hidden directions are kept at unit norm during
    training (as in the original experiment); the tanh teacher is unconstrained.
    Returns numpy arrays (W, A, b, c) with W of shape d x r.
    """
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    d = Xtr.shape[1]
    dt = torch.float64
    X = torch.tensor(Xtr, dtype=dt)
    y = torch.tensor(ytr, dtype=torch.long)
    gen = torch.Generator().manual_seed(50000 + seed)
    if activation == 'cubic':
        Wraw = torch.randn((d, r), generator=gen, dtype=dt).requires_grad_()
        b = (0.05 * torch.randn((r,), generator=gen, dtype=dt)).requires_grad_()
        A = (0.05 * torch.randn((m, r), generator=gen, dtype=dt)).requires_grad_()
    else:
        Wraw = (torch.randn((d, r), generator=gen, dtype=dt) / np.sqrt(d)).requires_grad_()
        b = (0.1 * torch.randn((r,), generator=gen, dtype=dt)).requires_grad_()
        A = (torch.randn((m, r), generator=gen, dtype=dt) / np.sqrt(r)).requires_grad_()
    c = torch.zeros((m,), dtype=dt, requires_grad=True)
    opt = torch.optim.Adam([Wraw, b, A, c], lr=lr, weight_decay=weight_decay)
    phi = _act(activation)

    def weights():
        if activation == 'cubic':
            return Wraw / torch.linalg.norm(Wraw, dim=0, keepdim=True).clamp_min(1e-12)
        return Wraw
    for ep in range(epochs):
        opt.zero_grad()
        logits = phi(X @ weights() + b) @ A.T + c
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        if ep in (int(0.5 * epochs), int(0.8 * epochs)):
            for g in opt.param_groups:
                g['lr'] *= 0.4
    with torch.no_grad():
        return (weights().numpy().copy(), A.numpy().copy(), b.numpy().copy(), c.numpy().copy())


def forward(X, W, A, b, c, activation='tanh'):
    Z = X @ W + b
    H = np.tanh(Z) if activation == 'tanh' else Z ** 3
    return H @ A.T + c


def hessian_matrix(X, W, A, b, activation='tanh'):
    """Exact stacked Hessians (p x m N) of f(x) = phi(x W + b) A^T + c, W unnormalized.

    Uses the factorization with unit directions u_j = w_j/||w_j|| and coefficients
    ||w_j||^2 a_oj phi''(w_j^T x + b_j).
    """
    from .svec import svec_features
    norms = np.linalg.norm(W, axis=0)
    U = W / norms
    Z = X @ W + b
    if activation == 'tanh':
        T = np.tanh(Z)
        d2 = -2.0 * T * (1.0 - T * T)
    else:
        d2 = 6.0 * Z
    m = A.shape[0]
    C = np.concatenate([d2 * (A[o] * norms ** 2)[None, :] for o in range(m)], axis=0)
    return svec_features(U) @ C.T, C, U


def distill_student(Xq, Yq, r, activation='tanh', seed=0, epochs=3000, lr=0.01):
    """Gradient-based distillation: a same-width student fitted to teacher outputs (MSE)."""
    import torch
    torch.manual_seed(1000 + seed)
    dt = torch.float64
    X = torch.tensor(Xq, dtype=dt)
    Y = torch.tensor(Yq, dtype=dt)
    d, m = Xq.shape[1], Yq.shape[1]
    gen = torch.Generator().manual_seed(70000 + seed)
    W = (torch.randn((d, r), generator=gen, dtype=dt) / np.sqrt(d)).requires_grad_()
    b = (0.1 * torch.randn((r,), generator=gen, dtype=dt)).requires_grad_()
    A = (torch.randn((m, r), generator=gen, dtype=dt) / np.sqrt(r)).requires_grad_()
    c = torch.zeros((m,), dtype=dt, requires_grad=True)
    opt = torch.optim.Adam([W, b, A, c], lr=lr)
    phi = _act(activation)
    scale = float(torch.mean(Y ** 2))
    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.mean((phi(X @ W + b) @ A.T + c - Y) ** 2) / scale
        loss.backward()
        opt.step()
        if ep in (int(0.5 * epochs), int(0.8 * epochs)):
            for g in opt.param_groups:
                g['lr'] *= 0.4
    # Final full-batch L-BFGS polish so the baseline is not limited by Adam.
    lb = torch.optim.LBFGS([W, b, A, c], lr=1.0, max_iter=500, tolerance_grad=1e-12,
                           tolerance_change=1e-15, history_size=50, line_search_fn='strong_wolfe')

    def closure():
        lb.zero_grad()
        loss = torch.mean((phi(X @ W + b) @ A.T + c - Y) ** 2) / scale
        loss.backward()
        return loss
    lb.step(closure)
    with torch.no_grad():
        return W.numpy().copy(), A.numpy().copy(), b.numpy().copy(), c.numpy().copy()


def genericity_diagnostics(W, A, X, activation='tanh', b=None):
    """Checks of the genericity assumptions on a trained teacher.

    Near-parallel units (largest absolute cosine between directions), weak or
    inactive units (Hessian signal ||w_j||^2 ||a_j|| relative to the median),
    and saturation (fraction of probes with |w_j^T x + b_j| > 3 for tanh).
    """
    norms = np.linalg.norm(W, axis=0)
    U = W / norms
    G = np.abs(U.T @ U)
    np.fill_diagonal(G, 0.0)
    signal = norms ** 2 * np.linalg.norm(A, axis=0)
    rel = signal / max(np.median(signal), 1e-300)
    out = {
        'max_abs_cos_pair': float(G.max()) if G.size else 0.0,
        'n_pairs_cos_gt_0.95': int(np.sum(np.triu(G > 0.95, 1))),
        'n_pairs_cos_gt_0.99': int(np.sum(np.triu(G > 0.99, 1))),
        'min_rel_signal': float(rel.min()),
        'n_weak_units_1e-2': int(np.sum(rel < 1e-2)),
        'n_weak_units_1e-1': int(np.sum(rel < 1e-1)),
    }
    if activation == 'tanh' and b is not None:
        Z = X @ W + b
        sat = np.mean(np.abs(Z) > 3.0, axis=0)
        out['median_saturation'] = float(np.median(sat))
        out['n_units_saturated_90pct'] = int(np.sum(sat > 0.9))
    return out, rel
