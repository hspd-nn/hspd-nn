import time
import numpy as np
from .svec import smat, svec_features


def jennrich_symmetric(M, d, rank, seed=0, trials=8):
    """Symmetric Jennrich baseline; only appropriate for rank <= d."""
    tic = time.perf_counter()
    if rank > d:
        return {'W': np.empty((d, 0)), 'runtime_sec': time.perf_counter()-tic, 'applicable': False}
    rng = np.random.default_rng(seed)
    K = M.shape[1]
    H = np.stack([smat(M[:, k], d) for k in range(K)], axis=0)
    # Estimate input span robustly.
    G = np.einsum('kij,klj->il', H, H, optimize=True)
    evals, evecs = np.linalg.eigh((G + G.T) / 2)
    U = evecs[:, -rank:]
    best = None
    for _ in range(trials):
        a = rng.normal(size=K); b = rng.normal(size=K)
        Ha = np.tensordot(a, H, axes=(0,0))
        Hb = np.tensordot(b, H, axes=(0,0))
        Ar = U.T @ Ha @ U
        Br = U.T @ Hb @ U
        try:
            Cmat = np.linalg.solve(Br.T, Ar.T).T
            vals, V = np.linalg.eig(Cmat)
        except np.linalg.LinAlgError:
            continue
        if np.max(np.abs(V.imag)) > 1e-6:
            continue
        V = V.real
        W = U @ V
        W /= np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-15)
        Bfeat = svec_features(W)
        coef = np.linalg.lstsq(Bfeat, M, rcond=None)[0]
        err = np.linalg.norm(Bfeat @ coef - M) / max(np.linalg.norm(M), 1e-15)
        if best is None or err < best[0]:
            best = (err, W)
    W = best[1] if best else np.empty((d,0))
    return {'W': W, 'runtime_sec': time.perf_counter()-tic, 'applicable': True}


def cp_adam(M, d, rank, seed=0, steps=500, lr=0.035, restarts=1):
    """Unstructured symmetric CP fit with Adam, included as a generic optimization baseline."""
    import torch
    tic = time.perf_counter()
    torch.set_num_threads(1)
    dtype = torch.float64
    Mt = torch.tensor(M, dtype=dtype)
    scale = torch.mean(Mt * Mt).detach().clamp_min(1e-15)
    best_loss = float('inf'); best_W = None
    # upper-triangular feature indices in svec order
    pairs = []
    for i in range(d):
        pairs.append((i, i, 1.0))
        for j in range(i+1, d):
            pairs.append((i, j, np.sqrt(2.0)))
    for rr in range(restarts):
        gen = torch.Generator().manual_seed(seed * 1009 + rr)
        Wraw = torch.randn((d, rank), generator=gen, dtype=dtype, requires_grad=True)
        C = (0.1 * torch.randn((M.shape[1], rank), generator=gen, dtype=dtype)).requires_grad_()
        opt = torch.optim.Adam([Wraw, C], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            W = Wraw / torch.linalg.norm(Wraw, dim=0, keepdim=True).clamp_min(1e-12)
            feats = [fac * W[i] * W[j] for i,j,fac in pairs]
            B = torch.stack(feats, dim=0)
            pred = B @ C.T
            loss = torch.mean((pred - Mt) ** 2) / scale
            loss.backward(); opt.step()
        with torch.no_grad():
            W = Wraw / torch.linalg.norm(Wraw, dim=0, keepdim=True).clamp_min(1e-12)
            feats = [fac * W[i] * W[j] for i,j,fac in pairs]
            B = torch.stack(feats, dim=0)
            Copt = torch.linalg.lstsq(B, Mt).solution
            loss = float(torch.mean((B @ Copt - Mt) ** 2) / scale)
            if loss < best_loss:
                best_loss = loss
                best_W = W.detach().cpu().numpy().copy()
    return {'W': best_W, 'runtime_sec': time.perf_counter()-tic, 'loss': best_loss, 'applicable': True}
