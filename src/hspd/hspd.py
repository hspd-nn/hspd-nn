"""Hessian Subspace Power Distillation (HSPD) and subspace baselines.

All solvers work on the r-dimensional estimated Hessian subspace spanned by
the leading left singular vectors U of the stacked, symmetric-vectorized
Hessian matrix M (shape p x K, p = d(d+1)/2).  For a unit vector v,

    F(v) = ||U^T svec(v v^T)||^2 = sum_k (v^T Q_k v)^2,   Q_k = smat(U[:, k]),

and the shifted power map is v <- (H_v v + gamma v) / ||H_v v + gamma v||
with H_v = sum_k (v^T Q_k v) Q_k = smat(P_A svec(v v^T)).
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize

from .svec import smat, svec, svec_features


# ---------------------------------------------------------------------------
# Subspace and objective
# ---------------------------------------------------------------------------

def subspace_basis(M, rank, d):
    """Leading rank-``rank`` left singular subspace of M, reshaped to matrices."""
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    q = min(rank, U.shape[1])
    Ur = np.ascontiguousarray(U[:, :q])
    Q = smat(Ur.T, d)  # (q, d, d)
    return Ur, Q, s


def subspace_perturbation_estimate(singular_values, rank):
    """Data-driven proxy sigma_{r+1}/sigma_r for the subspace perturbation."""
    s = np.asarray(singular_values)
    if len(s) <= rank or s[rank - 1] <= 0:
        return 0.0
    return float(s[rank] / s[rank - 1])


def _apply(Qf, V, r, d):
    """Return alpha[s,k] = v_s^T Q_k v_s and Y = H_v v via Z[s,k,:] = Q_k v_s."""
    Z = (V @ Qf.T).reshape(V.shape[0], r, d)
    alpha = np.matmul(Z, V[:, :, None])[:, :, 0]
    Y = np.matmul(alpha[:, None, :], Z)[:, 0, :]
    return alpha, Y


class SvecKernel:
    """F(v) and H_v v from the p x r basis U without forming Q_k v.

    alpha = U^T svec(v v^T) and h = U alpha are two dense products of width p;
    H_v = smat(h) is then applied to v.  This is the arithmetic-intensive form
    of the subspace power iteration used for large d.
    """

    def __init__(self, U, d):
        self.d = d
        self.i0, self.i1 = np.triu_indices(d)
        self.scale = np.where(self.i0 == self.i1, 1.0, np.sqrt(2.0))
        # Scale factors folded into the basis: q = V[:, i0] V[:, i1] (unscaled).
        self.Us = np.ascontiguousarray(self.scale[:, None] * U)
        self.UTh = np.ascontiguousarray((0.5 * self.scale)[None, :] * U.T)
        p = len(self.i0)
        # 0/1 incidence matrices of the row and column index of each svec entry
        self.E0 = np.zeros((p, d)); self.E0[np.arange(p), self.i0] = 1.0
        self.E1 = np.zeros((p, d)); self.E1[np.arange(p), self.i1] = 1.0

    def apply(self, V):
        # q = svec(v v^T); alpha = U^T q; H_v v = (1/2) J_q(v)^T (U alpha).
        V0, V1 = V[:, self.i0], V[:, self.i1]
        alpha = (V0 * V1) @ self.Us
        hs = alpha @ self.UTh
        Y = (hs * V1) @ self.E0 + (hs * V0) @ self.E1
        return alpha, Y


def score(V, Q):
    """Subspace score F(v) for the rows of V (Q: r x d x d basis matrices)."""
    V = np.atleast_2d(V)
    r, d, _ = Q.shape
    alpha, _ = _apply(Q.reshape(r * d, d), V, r, d)
    return np.sum(alpha * alpha, axis=1)


def random_starts(n_starts, d, seed):
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n_starts, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V


def default_starts(rank):
    """Common start budget: the coupon-collector count 5 r ln r, at least 100."""
    return int(max(100, np.ceil(5 * rank * np.log(max(rank, 2)))))


# ---------------------------------------------------------------------------
# Batched shifted power iteration
# ---------------------------------------------------------------------------

def power_iterate(Q, V0, gamma=0.0, max_iter=1000, tol=1e-6, chunk_bytes=2**28,
                  seed=0, fixed_iter=False):
    """Run the shifted power map from every row of V0.

    A start stops once its Riemannian gradient satisfies ||H_v v - F(v) v|| <= tol
    (unless ``fixed_iter``).  Returns final vectors, scores, iteration counts, a
    convergence mask and the number of steps at which F decreased.
    """
    r, d, _ = Q.shape
    from .svec import svec as _svec
    kern = SvecKernel(_svec(Q).T, d)   # p x r orthonormal basis
    S = V0.shape[0]
    V = V0.copy()
    F = np.zeros(S)
    iters = np.zeros(S, dtype=int)
    converged = np.zeros(S, dtype=bool)
    nonmono = np.zeros(S, dtype=int)
    rng = np.random.default_rng(seed + 7919)
    p = d * (d + 1) // 2
    chunk = max(1, int(chunk_bytes // (8 * (2 * p + r + d * d) + 1)))
    for lo in range(0, S, chunk):
        idx = np.arange(lo, min(S, lo + chunk))
        Vc = V[idx]
        Fprev = np.full(len(idx), -np.inf)
        active = np.ones(len(idx), dtype=bool)
        it_c = np.zeros(len(idx), dtype=int)
        nm_c = np.zeros(len(idx), dtype=int)
        Fc = np.zeros(len(idx))
        conv_c = np.zeros(len(idx), dtype=bool)
        for it in range(max_iter + 1):
            a = np.flatnonzero(active)
            if a.size == 0:
                break
            alpha, Y = kern.apply(Vc[a])
            Fa = np.sum(alpha * alpha, axis=1)
            Fc[a] = Fa
            nm_c[a] += (Fa < Fprev[a] - 1e-12)
            Fprev[a] = Fa
            res = np.linalg.norm(Y - Fa[:, None] * Vc[a], axis=1)
            done = (res <= tol) & (not fixed_iter)
            if it == max_iter:
                done = np.ones_like(done)
            if np.any(done):
                fin = a[done]
                conv_c[fin] = res[done] <= tol
                it_c[fin] = it
                active[fin] = False
            keep = ~done
            if not np.any(keep):
                break
            a = a[keep]
            Yk = Y[keep] + gamma * Vc[a]
            nrm = np.linalg.norm(Yk, axis=1)
            dead = nrm < 1e-14
            if np.any(dead):
                Yk[dead] = rng.normal(size=(int(dead.sum()), d))
                nrm[dead] = np.linalg.norm(Yk[dead], axis=1)
            Vc[a] = Yk / nrm[:, None]
        V[idx] = Vc
        F[idx] = Fc
        iters[idx] = it_c
        converged[idx] = conv_c
        nonmono[idx] = nm_c
    return V, F, iters, converged, nonmono


def newton_polish(Q, V0, max_steps=30, tol=1e-10, chunk_bytes=2**28):
    """Safeguarded Riemannian Newton refinement of the subspace score on the sphere.

    For each start, the Newton step solves Hess F(v) eta = -grad F(v) on the
    tangent space; it is taken only where the Riemannian Hessian is negative
    definite and the step increases F, otherwise an unshifted power step is
    taken.  Near a component with kappa > 0 the convergence is quadratic.
    Returns refined vectors, scores and the number of steps used.
    """
    r, d, _ = Q.shape
    Qf = np.ascontiguousarray(Q.reshape(r * d, d))
    Qd = np.ascontiguousarray(Q.reshape(r, d * d))
    V = V0.copy()
    S = V.shape[0]
    steps = np.zeros(S, dtype=int)
    F = np.zeros(S)
    chunk = max(1, int(chunk_bytes // (8 * r * d + 8 * d * d + 1)))
    eye = np.eye(d)
    for lo in range(0, S, chunk):
        idx = np.arange(lo, min(S, lo + chunk))
        Vc = V[idx]
        active = np.ones(len(idx), dtype=bool)
        Fc = np.zeros(len(idx))
        st = np.zeros(len(idx), dtype=int)
        for it in range(max_steps + 1):
            a = np.flatnonzero(active)
            if a.size == 0:
                break
            v = Vc[a]
            Z = (v @ Qf.T).reshape(a.size, r, d)
            alpha = np.matmul(Z, v[:, :, None])[:, :, 0]
            Y = np.matmul(alpha[:, None, :], Z)[:, 0, :]
            Fa = np.sum(alpha * alpha, axis=1)
            Fc[a] = Fa
            g = Y - Fa[:, None] * v                      # grad / 4
            res = np.linalg.norm(g, axis=1)
            done = res <= tol
            if it == max_steps:
                done[:] = True
            st[a[done]] = it
            active[a[done]] = False
            keep = ~done
            if not np.any(keep):
                break
            a, v, Z, alpha, Y, Fa, g = a[keep], v[keep], Z[keep], alpha[keep], Y[keep], Fa[keep], g[keep]
            Hv = (alpha @ Qd).reshape(a.size, d, d)
            ZZ = np.matmul(Z.transpose(0, 2, 1), Z)
            vv = v[:, :, None] * v[:, None, :]
            P = eye[None] - vv
            Hr = np.matmul(np.matmul(P, Hv + 2.0 * ZZ), P) - Fa[:, None, None] * P  # Hess / 4
            Hmod = Hr - vv
            evals = np.linalg.eigvalsh(Hr - 1e3 * vv)
            negdef = evals[:, -1] < -1e-12
            eta = -np.linalg.solve(Hmod, g[:, :, None])[:, :, 0]
            pw = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-300)
            newv = pw.copy()
            pending = negdef.copy()
            step = 1.0
            for _ in range(6):  # backtracking on the Newton direction
                if not np.any(pending):
                    break
                vn = v + step * eta
                vn /= np.linalg.norm(vn, axis=1, keepdims=True)
                Zn = (vn @ Qf.T).reshape(vn.shape[0], r, d)
                Fn = np.sum(np.matmul(Zn, vn[:, :, None])[:, :, 0] ** 2, axis=1)
                ok = pending & (Fn >= Fa - 1e-15)
                newv[ok] = vn[ok]
                pending &= ~ok
                step *= 0.5
            Vc[a] = newv
        V[idx] = Vc
        F[idx] = Fc
        steps[idx] = st
    return V, F, steps


# ---------------------------------------------------------------------------
# Candidate selection, certification and conditioning
# ---------------------------------------------------------------------------

def stationarity_residual(V, Q):
    """||H_v v - F(v) v|| (a quarter of the Riemannian gradient norm) for rows of V."""
    r, d, _ = Q.shape
    kern = SvecKernel(svec(Q).T, d)
    out_r, out_F = [], []
    for lo in range(0, V.shape[0], 4096):
        alpha, Y = kern.apply(V[lo:lo + 4096])
        F = np.sum(alpha * alpha, axis=1)
        out_r.append(np.linalg.norm(Y - F[:, None] * V[lo:lo + 4096], axis=1))
        out_F.append(F)
    return np.concatenate(out_r), np.concatenate(out_F)


def cluster_candidates(V, F, max_k, threshold=0.995, stationary=None):
    """Greedy sign-invariant clustering in decreasing score order.

    Stationary candidates (if flags are given) are ranked before the others,
    then by score.  A candidate opens a new cluster if its absolute cosine with
    every chosen representative is below ``threshold``; at most ``max_k`` are
    returned.  No relaxation is applied: fewer than ``max_k`` distinct
    candidates may be returned, and the evaluation penalizes missing ones.
    """
    if stationary is None:
        order = np.argsort(-F)
    else:
        order = np.lexsort((-F, ~np.asarray(stationary, dtype=bool)))
    chosen = []
    for i in order:
        if not chosen or np.max(np.abs(V[chosen] @ V[i])) < threshold:
            chosen.append(i)
            if len(chosen) == max_k:
                break
    if not chosen:
        return np.empty((0, V.shape[1])), np.empty(0), np.empty(0, dtype=int)
    chosen = np.asarray(chosen)
    return V[chosen].copy(), F[chosen].copy(), chosen


def acceptance_threshold(singular_values, rank, c=3.0, floor=1e-4):
    """Certification tolerance tau on the residual 1 - F (see manuscript)."""
    eps = subspace_perturbation_estimate(singular_values, rank)
    return float(max(floor, (c * eps) ** 2)), eps


def tangent_operator(w):
    """p x (d-1) matrix with columns svec(w u^T + u w^T), u an orthonormal basis of w^perp."""
    d = w.shape[0]
    Qm, _ = np.linalg.qr(np.column_stack([w, np.eye(d)]))
    N = Qm[:, 1:d]
    T = np.einsum('i,jk->kij', w, N) + np.einsum('ik,j->kij', N, w)
    return svec(T).T


def kappa(w, Ur):
    """Tangent transversality constant kappa(w) = sigma_min(P_A^perp D_w)."""
    w = w / np.linalg.norm(w)
    D = tangent_operator(w)
    R = D - Ur @ (Ur.T @ D)
    sv = np.linalg.svd(R, compute_uv=False)
    return float(sv[-1]) if sv.size else 0.0


def kappas(W, Ur):
    return np.array([kappa(W[:, j], Ur) for j in range(W.shape[1])])


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def _package(W, scores, s, rank, accept_c, accept_floor, extra):
    tau, eps = acceptance_threshold(s, rank, accept_c, accept_floor)
    certified = (1.0 - np.asarray(scores)) <= tau if len(scores) else np.zeros(0, bool)
    out = {
        'W': W,
        'scores': np.asarray(scores),
        'certified': certified,
        'tau': tau,
        'eps_hat': eps,
        'singular_values': s,
    }
    out.update(extra)
    return out


def hspd_recover(M, d, rank, n_starts=None, max_iter=1000, shift=0.0, tol=1e-6,
                 seed=0, cluster_threshold=0.995, accept_c=3.0, accept_floor=1e-4,
                 V0=None, fixed_iter=False, return_candidates=False, polish=None,
                 polish_steps=30, stationary_tol=1e-5):
    """HSPD: batched shifted power iteration on the Hessian subspace.

    Returns the (at most ``rank``) highest-scoring distinct candidates together
    with certification flags (score test) and convergence diagnostics.
    """
    tic = time.perf_counter()
    Ur, Q, s = subspace_basis(M, rank, d)
    t_svd = time.perf_counter() - tic
    if V0 is None:
        V0 = random_starts(n_starts or default_starts(rank), d, seed)
    V, F, iters, conv, nonmono = power_iterate(Q, V0, gamma=shift, max_iter=max_iter,
                                              tol=tol, seed=seed, fixed_iter=fixed_iter)
    t_power = time.perf_counter() - tic
    n_polish = 0
    if polish == 'newton':
        V, F, pst = newton_polish(Q, V, max_steps=polish_steps)
        n_polish = int(np.median(pst))
    res, F = stationarity_residual(V, Q)
    stat = res <= stationary_tol
    C, sc, _ = cluster_candidates(V, F, rank, cluster_threshold, stat)
    W = C.T if len(C) else np.empty((d, 0))
    extra = {
        'runtime_sec': time.perf_counter() - tic,
        'svd_sec': t_svd,
        'power_sec': t_power,
        'polish_steps_median': n_polish,
        'n_starts': V0.shape[0],
        'iterations_median': float(np.median(iters)),
        'iterations_max': int(np.max(iters)),
        'frac_converged': float(np.mean(conv)),
        'nonmonotone_steps': int(np.sum(nonmono)),
        'frac_nonmonotone_starts': float(np.mean(nonmono > 0)),
        'Ur': Ur,
    }
    extra['frac_stationary'] = float(np.mean(stat))
    if return_candidates:
        extra.update({'V_final': V, 'F_final': F, 'res_final': res, 'iters': iters,
                      'converged': conv})
    return _package(W, sc, s, rank, accept_c, accept_floor, extra)


def lbfgs_subspace_recover(M, d, rank, n_starts=None, max_iter=1000, tol=1e-6, seed=0,
                           cluster_threshold=0.995, accept_c=3.0, accept_floor=1e-4,
                           V0=None, stationary_tol=1e-5):
    """Generic multi-start L-BFGS maximization of the same subspace score.

    Uses the same starts, iteration cap and (Riemannian-gradient) tolerance as
    HSPD so that only the optimizer differs.
    """
    tic = time.perf_counter()
    Ur, Q, s = subspace_basis(M, rank, d)
    r = Q.shape[0]
    Qf = np.ascontiguousarray(Q.reshape(r * d, d))
    if V0 is None:
        V0 = random_starts(n_starts or default_starts(rank), d, seed)

    def fun_grad(x):
        n = np.linalg.norm(x)
        if n < 1e-14:
            x = np.ones_like(x); n = np.linalg.norm(x)
        v = x / n
        Z = (Qf @ v).reshape(r, d)
        a = Z @ v
        f = float(a @ a)
        y = a @ Z
        g_riem = 4.0 * (y - f * v)
        return -f, -g_riem / n

    gtol = 4.0 * tol / np.sqrt(d)
    V = np.empty_like(V0)
    nit = np.zeros(V0.shape[0], dtype=int)
    for i, x0 in enumerate(V0):
        res = minimize(fun_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': gtol})
        V[i] = res.x / max(np.linalg.norm(res.x), 1e-15)
        nit[i] = res.nit
    res, F = stationarity_residual(V, Q)
    stat = res <= stationary_tol
    C, sc, _ = cluster_candidates(V, F, rank, cluster_threshold, stat)
    W = C.T if len(C) else np.empty((d, 0))
    extra = {'runtime_sec': time.perf_counter() - tic, 'n_starts': V0.shape[0],
             'iterations_median': float(np.median(nit)), 'iterations_max': int(nit.max()),
             'frac_stationary': float(np.mean(stat)), 'Ur': Ur,
             'V_final': V, 'F_final': F, 'res_final': res}
    return _package(W, sc, s, rank, accept_c, accept_floor, extra)


def fvd_recover(M, d, rank, n_starts=None, max_iter=1000, tol=1e-6, seed=0, gamma=2.0,
                cluster_threshold=0.995, accept_c=3.0, accept_floor=1e-4, stationary_tol=1e-5):
    """Projected subgradient ascent of Fornasier, Vybiral and Daubechies (2021).

    Iterates X <- P_A Pi_gamma(X) in the matrix subspace, where Pi_gamma
    multiplies the largest singular value by gamma and renormalizes; the
    direction estimate is the leading eigenvector of the final X.
    """
    tic = time.perf_counter()
    Ur, Q, s = subspace_basis(M, rank, d)
    r = Q.shape[0]
    S = n_starts or default_starts(rank)
    rng = np.random.default_rng(seed)
    Aco = rng.normal(size=(S, r))
    Aco /= np.linalg.norm(Aco, axis=1, keepdims=True)
    iters = np.zeros(S, dtype=int)
    active = np.ones(S, dtype=bool)
    for it in range(max_iter):
        a = np.flatnonzero(active)
        if a.size == 0:
            break
        X = np.einsum('sk,kij->sij', Aco[a], Q, optimize=True)
        lam, vec = np.linalg.eigh(X)
        top = np.argmax(np.abs(lam), axis=1)
        l1 = lam[np.arange(a.size), top]
        u1 = vec[np.arange(a.size), :, top]
        fro2 = np.sum(Aco[a] ** 2, axis=1)
        Pi = X + ((gamma - 1.0) * l1)[:, None, None] * np.einsum('si,sj->sij', u1, u1)
        Pi /= np.sqrt(fro2 + (gamma ** 2 - 1.0) * l1 ** 2)[:, None, None]
        # Projection onto the subspace in svec coordinates.
        new = np.einsum('kij,sij->sk', Q, Pi, optimize=True)
        step = np.linalg.norm(new - Aco[a], axis=1)
        Aco[a] = new
        iters[a] = it + 1
        active[a[step <= tol]] = False
    X = np.einsum('sk,kij->sij', Aco, Q, optimize=True)
    lam, vec = np.linalg.eigh(X)
    top = np.argmax(np.abs(lam), axis=1)
    V = vec[np.arange(S), :, top]
    res, F = stationarity_residual(V, Q)
    stat = res <= stationary_tol
    C, sc, _ = cluster_candidates(V, F, rank, cluster_threshold, stat)
    W = C.T if len(C) else np.empty((d, 0))
    extra = {'runtime_sec': time.perf_counter() - tic, 'n_starts': S,
             'iterations_median': float(np.median(iters)), 'iterations_max': int(iters.max()),
             'frac_stationary': float(np.mean(stat)), 'Ur': Ur,
             'V_final': V, 'F_final': F, 'res_final': res}
    return _package(W, sc, s, rank, accept_c, accept_floor, extra)


def spm_deflation_recover(M, d, rank, shift=0.75, max_iter=1000, tol=1e-6, seed=0,
                          zeta=0.99, repetitions=3, accept_c=3.0, accept_floor=1e-4):
    """Subspace power method with deflation (Kileel and Pereira).

    One component at a time: a shifted power iteration from a random start
    (repeated up to ``repetitions`` times if the score stays below ``zeta``),
    followed by SPM's symmetric deflation applied to the Gram flattening
    M M^T, whose column space is the Hessian subspace.  The deflation is exact
    only when the context Gram matrix C^T C is diagonal.
    """
    tic = time.perf_counter()
    U_full, s, _ = np.linalg.svd(M, full_matrices=False)
    r = min(rank, U_full.shape[1])
    U = U_full[:, :r].copy()
    Cm = np.diag(1.0 / np.maximum(s[:r] ** 2, 1e-300))
    rng = np.random.default_rng(seed)
    found, scores, total_iters = [], [], 0
    for i in range(r):
        Q = smat(U.T, d)
        best = None
        for _ in range(repetitions):
            v0 = rng.normal(size=(1, d)); v0 /= np.linalg.norm(v0)
            V, F, it, _, _ = power_iterate(Q, v0, gamma=shift, max_iter=max_iter, tol=tol,
                                           seed=seed + i)
            total_iters += int(it[0])
            if best is None or F[0] > best[1]:
                best = (V[0], F[0])
            if F[0] > zeta:
                break
        x = best[0]
        found.append(x)
        q = svec_features(x[:, None])[:, 0]
        if i == r - 1:
            break
        alpha = U.T @ q
        y = Cm @ alpha
        y /= np.linalg.norm(y)
        # Orthonormal basis of span{y}^perp via a Householder reflection.
        H = np.eye(len(y)) - 2.0 * np.outer(_hh(y), _hh(y))
        O = H[:, 1:]
        U = U @ O
        Cm = O.T @ Cm @ O
    W = np.asarray(found).T if found else np.empty((d, 0))
    # Scores are reported on the full (undeflated) subspace.
    _, Qfull, _ = subspace_basis(M, rank, d)
    sc = score(W.T, Qfull) if W.shape[1] else np.empty(0)
    order = np.argsort(-sc)
    W, sc = W[:, order], sc[order]
    extra = {'runtime_sec': time.perf_counter() - tic, 'n_starts': None,
             'iterations_total': total_iters}
    return _package(W, sc, s, rank, accept_c, accept_floor, extra)


def _hh(y):
    """Householder vector z with (I - 2 z z^T) e_1 = +-y."""
    e = np.zeros_like(y); e[0] = 1.0
    sgn = 1.0 if y[0] >= 0 else -1.0
    z = y + sgn * e
    n = np.linalg.norm(z)
    return z / n if n > 0 else e
