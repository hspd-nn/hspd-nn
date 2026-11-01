#!/usr/bin/env python3
"""Run one task of an experiment campaign and write its records.

    python scripts/run_campaign.py --campaign width8 --n-tasks 40 [--task-id 3]

Units of a campaign are listed deterministically and assigned round-robin to
tasks; under a Slurm array the task id defaults to $SLURM_ARRAY_TASK_ID.  Each
task writes out/runs/<campaign>/task-XXX/{config,env,metrics}.json and rewrites
metrics.json after every unit, so a resubmitted task skips finished units.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--campaign', required=True)
ap.add_argument('--n-tasks', type=int, default=1)
ap.add_argument('--task-id', type=int, default=None)
ap.add_argument('--run-root', default='out/runs')
ap.add_argument('--threads', type=int, default=None)
ap.add_argument('--list', action='store_true', help='print the number of units and exit')
ap.add_argument('--limit', type=int, default=None, help='run at most this many units (testing)')
args = ap.parse_args()

threads = args.threads or int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[var] = str(threads)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np  # noqa: E402

from hspd.hspd import (hspd_recover, lbfgs_subspace_recover, fvd_recover,  # noqa: E402
                       spm_deflation_recover, default_starts, random_starts, kappas,
                       cluster_candidates, acceptance_threshold, stationarity_residual,
                       subspace_basis)
from hspd.baselines import jennrich_symmetric  # noqa: E402
from hspd.metrics import direction_summary, relative_rmse  # noqa: E402
from hspd.pnn import (random_cubic_pnn, pnn_hessian_matrix, pnn_forward,  # noqa: E402
                      recover_cubic_parameters, random_tanh_net, recover_tanh_parameters,
                      refine_tanh_student)
from hspd.svec import svec_features  # noqa: E402
from hspd.finite_difference import finite_difference_hessians, queries_per_probe  # noqa: E402

# ---------------------------------------------------------------------------
# Method configurations (identical starts, tolerance and selection for all
# multi-start solvers)
# ---------------------------------------------------------------------------
METHODS = {
    'HSPD': dict(fn='hspd', shift=0.0, max_iter=300, polish='newton'),
    'HSPD-power': dict(fn='hspd', shift=0.0, max_iter=1000, polish=None),
    'HSPD-shift0.5': dict(fn='hspd', shift=0.5, max_iter=1000, polish=None),
    'FKMR': dict(fn='hspd', shift=0.25, max_iter=1000, polish=None, fixed_iter=True),
    'SPM': dict(fn='spm', shift=0.75),
    'LBFGS': dict(fn='lbfgs', max_iter=1000),
    'FVD': dict(fn='fvd', max_iter=1000),
    'Jennrich': dict(fn='jennrich'),
}
TOL = 1e-6
CLUSTER = 0.995


def run_method(name, M, d, r, V0, seed, cfg=None):
    cfg = dict(METHODS[name] if cfg is None else cfg)
    fn = cfg.pop('fn')
    if fn == 'hspd':
        return hspd_recover(M, d, r, V0=V0, tol=TOL, seed=seed, cluster_threshold=CLUSTER,
                            return_candidates=True, **cfg)
    if fn == 'lbfgs':
        return lbfgs_subspace_recover(M, d, r, V0=V0, tol=TOL, seed=seed,
                                      cluster_threshold=CLUSTER, **cfg)
    if fn == 'fvd':
        return fvd_recover(M, d, r, n_starts=V0.shape[0], tol=TOL, seed=seed,
                           cluster_threshold=CLUSTER, **cfg)
    if fn == 'spm':
        return spm_deflation_recover(M, d, r, tol=TOL, seed=seed, **cfg)
    if fn == 'jennrich':
        res = jennrich_symmetric(M, d, r, seed=seed)
        res['certified'] = None
        return res
    raise ValueError(fn)


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def summarize(name, res, W_true, extra_diag=False, Ut=None):
    """Metrics for one solver run; extra diagnostics for the HSPD family."""
    summ, cos, est = direction_summary(W_true, res['W'], res.get('certified'))
    rec = {'method': name, **summ,
           'runtime_sec': res.get('runtime_sec'),
           'iterations_median': res.get('iterations_median'),
           'n_starts': res.get('n_starts'),
           'frac_stationary': res.get('frac_stationary'),
           'tau': res.get('tau'), 'eps_hat': res.get('eps_hat'),
           'nonmonotone_steps': res.get('nonmonotone_steps'),
           'frac_nonmonotone_starts': res.get('frac_nonmonotone_starts'),
           'frac_converged': res.get('frac_converged'),
           'power_sec': res.get('power_sec'), 'svd_sec': res.get('svd_sec')}
    if 'scores' in res and res['W'].shape[1]:
        sc = np.asarray(res['scores'])
        good = np.zeros(len(sc), dtype=bool)
        for j, e in enumerate(est):
            if e >= 0 and cos[j] >= 0.99:
                good[e] = True
        rec['resid_true'] = jsonable(np.sort(1.0 - sc[good])) if good.any() else []
        rec['resid_false'] = jsonable(np.sort(1.0 - sc[~good])) if (~good).any() else []
    if extra_diag and 'V_final' in res:
        V, F, R = res['V_final'], res['F_final'], res['res_final']
        Wn = W_true / np.linalg.norm(W_true, axis=0)
        near = np.abs(V @ Wn)
        hits = (near >= 0.99).sum(axis=0)
        spurious = near.max(axis=1) < 0.99
        stat = R <= 1e-5
        rec['basin_hits'] = jsonable(hits)
        rec['frac_starts_true'] = float(np.mean(~spurious))
        rec['frac_starts_spurious_stationary'] = float(np.mean(spurious & stat))
        Fs = F[spurious & stat]
        rec['spurious_F_quantiles'] = jsonable(np.quantile(Fs, [0.5, 0.9, 0.99, 1.0])) if Fs.size else None
        rec['comp_cos'] = jsonable(cos)
        # Threshold sensitivity (clustering) on the same final candidates.
        sens = {}
        for thr in (0.9, 0.95, 0.98, 0.99, 0.995, 0.999):
            C, sc_, _ = cluster_candidates(V, F, W_true.shape[1], thr, stat)
            s2, _, _ = direction_summary(W_true, C.T if len(C) else np.empty((W_true.shape[0], 0)))
            sens[str(thr)] = {'success': s2['success'], 'frac_recovered': s2['frac_recovered'],
                              'n_returned': s2['n_returned']}
        rec['cluster_sensitivity'] = sens
        if Ut is not None and res['W'].shape[1]:
            # Estimated kappa of the returned candidates on the estimated subspace.
            rec['kappa_hat'] = jsonable(kappas(res['W'], res['Ur']))
    return rec


def cubic_instance(d, r, m, N, noise, seed_base):
    net = random_cubic_pnn(d, r, m, seed=seed_base + 1)
    X = np.random.default_rng(seed_base + 2).normal(size=(N, d))
    M, C = pnn_hessian_matrix(X, net.W, net.A, net.b, noise=noise, seed=seed_base + 3)
    return net, X, M, C


def true_kappas(W):
    Ut = np.linalg.qr(svec_features(W))[0]
    return kappas(W, Ut), Ut


def env_info():
    info = {'python': sys.version.split()[0], 'platform': platform.platform(),
            'machine': platform.machine(), 'host': socket.gethostname(),
            'threads': threads, 'numpy': np.__version__}
    try:
        import scipy, sklearn, torch
        info.update({'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
                     'torch': torch.__version__})
    except Exception:
        pass
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    info['cpu'] = line.split(':', 1)[1].strip()
                    break
    except OSError:
        pass
    git = ROOT / '.git_commit'
    if git.exists():
        info['git'] = dict(l.split('=', 1) for l in git.read_text().split() if '=' in l)
    return info


# ---------------------------------------------------------------------------
# Campaigns: each returns (units, run_unit)
# ---------------------------------------------------------------------------

def campaign_width8():
    units = [dict(r=r, noise=eta, seed=s)
             for r in (4, 8, 12, 16, 20, 24, 28, 29, 30, 32)
             for eta in (0.0, 0.01) for s in range(50)]

    def run(u):
        d, m, N, r = 8, 4, 32, u['r']
        base = 100000 * (1 if u['noise'] else 0) + 1000 * r + u['seed'] * 10
        net, X, M, C = cubic_instance(d, r, m, N, u['noise'], base)
        kt, Ut = true_kappas(net.W)
        S = default_starts(r)
        V0 = random_starts(S, d, base + 4)
        out = {'d': d, 'm': m, 'N': N, 'context_rank': int(np.linalg.matrix_rank(C)),
               'codim': d * (d - 1) // 2, 'kappa_min': float(kt.min()),
               'kappa_median': float(np.median(kt)), 'methods': []}
        for name in ('HSPD', 'HSPD-power', 'HSPD-shift0.5', 'FKMR', 'SPM', 'LBFGS', 'FVD', 'Jennrich'):
            if name == 'Jennrich' and r > d:
                continue
            res = run_method(name, M, d, r, V0, base + 5)
            out['methods'].append(summarize(name, res, net.W, extra_diag=(name == 'HSPD'), Ut=Ut))
        out['kappa_true'] = jsonable(kt)
        return out
    return units, run


def multid_setup(d):
    codim = d * (d - 1) // 2
    rhos = (0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0)
    rs = sorted(set([max(2, int(round(x * codim))) for x in rhos] + [codim + 1]))
    rmax = max(rs)
    m = int(np.ceil(1.2 * rmax / (d + 1)))
    N = 2 * (d + 1)
    return codim, rs, m, N


def campaign_multid():
    units = []
    for d in (6, 10, 16, 24):
        codim, rs, m, N = multid_setup(d)
        for r in rs:
            for eta in (0.0, 0.01):
                for s in range(20):
                    units.append(dict(d=d, r=r, noise=eta, seed=s))
    # Most expensive units first, so the array finishes evenly.
    units.sort(key=lambda u: -(u['d'] ** 2 * u['r']))

    def run(u):
        d, r = u['d'], u['r']
        codim, _, m, N = multid_setup(d)
        base = 10 ** 6 * d + 1000 * r + 100 * (1 if u['noise'] else 0) + u['seed']
        net, X, M, C = cubic_instance(d, r, m, N, u['noise'], base)
        kt, Ut = true_kappas(net.W)
        S = default_starts(r)
        V0 = random_starts(S, d, base + 4)
        out = {'m': m, 'N': N, 'codim': codim, 'rho': r / codim,
               'context_rank': int(np.linalg.matrix_rank(C)),
               'kappa_min': float(kt.min()), 'kappa_median': float(np.median(kt)), 'methods': []}
        names = ['HSPD', 'HSPD-power', 'FKMR', 'SPM', 'LBFGS']
        if d <= 16:
            names.append('FVD')
        if r <= d:
            names.append('Jennrich')
        for name in names:
            res = run_method(name, M, d, r, V0, base + 5)
            out['methods'].append(summarize(name, res, net.W, extra_diag=(name == 'HSPD'), Ut=Ut))
        out['kappa_true'] = jsonable(kt)
        return out
    return units, run


def campaign_shift():
    units = [dict(d=d, rho=rho, noise=eta, seed=s)
             for d in (8, 16) for rho in (0.5, 0.8, 0.9, 1.0)
             for eta in (0.0, 0.01) for s in range(20)]

    def run(u):
        d = u['d']
        codim = d * (d - 1) // 2
        r = int(round(u['rho'] * codim))
        m = int(np.ceil(1.2 * r / (d + 1))) + 1
        N = 2 * (d + 1)
        base = 7 * 10 ** 6 + 10 ** 5 * d + 1000 * r + 100 * (1 if u['noise'] else 0) + u['seed']
        net, X, M, C = cubic_instance(d, r, m, N, u['noise'], base)
        S = default_starts(r)
        V0 = random_starts(S, d, base + 4)
        out = {'r': r, 'm': m, 'N': N, 'codim': codim, 'methods': []}
        for g in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0):
            for variant, cfg in (('power', dict(fn='hspd', shift=g, max_iter=1000, polish=None)),
                                 ('polish', dict(fn='hspd', shift=g, max_iter=300, polish='newton'))):
                res = run_method(None, M, d, r, V0, base + 5, cfg=cfg)
                rec = summarize(f'gamma={g}/{variant}', res, net.W)
                rec.update({'gamma': g, 'variant': variant})
                out['methods'].append(rec)
        return out
    return units, run


def campaign_noise():
    units = []
    for d, r, m, N in ((8, 20, 4, 40), (16, 60, 5, 34)):
        for eta in (0.0, 0.001, 0.01, 0.03, 0.05, 0.1):
            for s in range(20):
                units.append(dict(d=d, r=r, m=m, N=N, noise=eta, seed=s))

    def run(u):
        d, r, m, N = u['d'], u['r'], u['m'], u['N']
        base = 12 * 10 ** 6 + 10 ** 5 * d + u['seed'] * 10
        net, X, M, C = cubic_instance(d, r, m, N, u['noise'], base)
        kt, Ut = true_kappas(net.W)
        S = default_starts(r)
        V0 = random_starts(S, d, base + 4)
        Mex, _ = pnn_hessian_matrix(X, net.W, net.A, net.b)
        Ue = np.linalg.svd(Mex, full_matrices=False)[0][:, :r]
        Un = np.linalg.svd(M, full_matrices=False)[0][:, :r]
        sin_theta = float(np.linalg.norm(Un - Ue @ (Ue.T @ Un), 2))
        s = np.linalg.svd(M, compute_uv=False)
        out = {'spectral_gap': float(s[r - 1] - (s[r] if len(s) > r else 0.0)),
               'sigma_r': float(s[r - 1]), 'sin_theta': sin_theta,
               'kappa_min': float(kt.min()), 'kappa_median': float(np.median(kt)), 'methods': []}
        for name in ('HSPD', 'HSPD-power'):
            res = run_method(name, M, d, r, V0, base + 5)
            rec = summarize(name, res, net.W, extra_diag=(name == 'HSPD'), Ut=Ut)
            out['methods'].append(rec)
        out['kappa_true'] = jsonable(kt)
        return out
    return units, run


def campaign_context():
    units = []
    for m in (1, 2, 4):
        for N in (4, 8, 16, 32):
            for s in range(20):
                units.append(dict(act='cubic', d=8, r=16, m=m, N=N, seed=s))
    for m in (1, 2):
        for N in (4, 8, 12, 16, 20, 24, 32, 64):
            for s in range(20):
                units.append(dict(act='tanh', d=8, r=20, m=m, N=N, seed=s))

    def run(u):
        d, r, m, N = u['d'], u['r'], u['m'], u['N']
        base = 20 * 10 ** 6 + 10 ** 4 * m + 100 * N + u['seed']
        X = np.random.default_rng(base + 2).normal(size=(N, d))
        if u['act'] == 'cubic':
            net = random_cubic_pnn(d, r, m, seed=base + 1)
            W = net.W
            M, C = pnn_hessian_matrix(X, net.W, net.A, net.b, noise=1e-6, seed=base + 3)
            capacity = min(r, m * min(N, d + 1))
        else:
            from hspd.pnn import tanh_hessian_matrix
            W, A, b, c = random_tanh_net(d, r, m, seed=base + 1)
            M, C = tanh_hessian_matrix(X, W, A, b, noise=1e-6, seed=base + 3)
            capacity = min(r, m * N)
        sv = np.linalg.svd(C, compute_uv=False)
        rank = int(np.sum(sv > sv[0] * 1e-9))
        S = default_starts(r)
        V0 = random_starts(S, d, base + 4)
        res = run_method('HSPD', M, d, r, V0, base + 5)
        rec = summarize('HSPD', res, W)
        return {'context_rank': rank, 'capacity': capacity,
                'sigma_r_rel': float(sv[r - 1] / sv[0]) if len(sv) >= r else 0.0,
                'methods': [rec]}
    return units, run


def campaign_cubic_full():
    units = [dict(kind='full', r=r, rho_w=1.0, seed=s) for r in (8, 16, 24) for s in range(20)]
    units += [dict(kind='weak', r=16, rho_w=rw, seed=s)
              for rw in (1.0, 0.3, 0.1, 0.03, 0.01, 0.003) for s in range(20)]

    def run(u):
        d, m, N, r = 8, 4, 48, u['r']
        base = 30 * 10 ** 6 + 1000 * r + u['seed'] * 10 + (0 if u['kind'] == 'full' else 500)
        net = random_cubic_pnn(d, r, m, seed=base + 1)
        net.A[:, 0] *= u['rho_w']
        rng = np.random.default_rng(base + 2)
        X = rng.normal(size=(N, d))
        Xte = rng.normal(size=(300, d))
        M, _ = pnn_hessian_matrix(X, net.W, net.A, net.b, noise=0.01, seed=base + 3)
        Yp = pnn_forward(X, net.W, net.A, net.b, net.c)
        Yte = pnn_forward(Xte, net.W, net.A, net.b, net.c)
        S = default_starts(r)
        V0 = random_starts(S, d, base + 4)
        res = run_method('HSPD', M, d, r, V0, base + 5)
        rec = summarize('HSPD', res, net.W)
        out = {'methods': [rec]}
        if res['W'].shape[1] == r:
            est = recover_cubic_parameters(M, X, res['W'], Yp)
            Yhat = pnn_forward(Xte, est.W, est.A, est.b, est.c)
            out['relative_rmse'] = relative_rmse(Yhat, Yte)
            from hspd.metrics import matched_cosines
            cos, idx = matched_cosines(net.W, est.W)
            # Align signs: (w, b, a) ~ (-w, -b, -a).
            j0 = idx[0]
            sgn = np.sign(est.W[:, j0] @ net.W[:, 0])
            out['weak_cos'] = float(cos[0])
            out['weak_bias_err'] = float(abs(sgn * est.b[j0] - net.b[0]))
            out['weak_a_relerr'] = float(np.linalg.norm(sgn * est.A[:, j0] - net.A[:, 0]) /
                                         max(np.linalg.norm(net.A[:, 0]), 1e-300))
            contrib_true = (Xte @ net.W[:, 0] + net.b[0])[:, None] ** 3 * net.A[:, 0][None, :]
            contrib_est = (Xte @ est.W[:, j0] + est.b[j0])[:, None] ** 3 * est.A[:, j0][None, :]
            out['weak_contrib_rmse_rel_to_output'] = float(
                np.sqrt(np.mean((contrib_est - contrib_true) ** 2)) / np.sqrt(np.mean(Yte ** 2)))
            out['bias_err_median'] = float(np.median(np.abs(
                np.array([np.sign(est.W[:, idx[j]] @ net.W[:, j]) * est.b[idx[j]] - net.b[j]
                          for j in range(r)]))))
        return out
    return units, run


def campaign_tanh():
    units = [dict(noise=eta, seed=s) for eta in (0.0, 0.01, 0.05) for s in range(20)]

    def run(u):
        from hspd.pnn import tanh_hessian_matrix, tanh_forward
        d, m, N, r = 8, 4, 64, 20
        base = 40 * 10 ** 6 + int(1000 * u['noise'] * 100) + u['seed'] * 10
        W, A, b, c = random_tanh_net(d, r, m, seed=base + 1)
        rng = np.random.default_rng(base + 2)
        X = rng.normal(size=(N, d))
        Xte = rng.normal(size=(500, d))
        M, C = tanh_hessian_matrix(X, W, A, b, noise=u['noise'], seed=base + 3)
        S = default_starts(r)
        V0 = random_starts(S, d, base + 4)
        res = run_method('HSPD', M, d, r, V0, base + 5)
        rec = summarize('HSPD', res, W)
        out = {'context_rank': int(np.linalg.matrix_rank(C)), 'methods': [rec]}
        if res['W'].shape[1] == r:
            Yp = tanh_forward(X, W, A, b, c)
            Ws, A2, b2, c2, fc = recover_tanh_parameters(M, X, res['W'], Yp)
            Yte = tanh_forward(Xte, W, A, b, c)
            out['relative_rmse_coef'] = relative_rmse(np.tanh(Xte @ Ws + b2) @ A2.T + c2, Yte)
            Wr, Ar, br, cr = refine_tanh_student(X, Yp, Ws, A2, b2, c2)
            out['relative_rmse_refined'] = relative_rmse(np.tanh(Xte @ Wr + br) @ Ar.T + cr, Yte)
            out['profile_fit_cos_min'] = float(fc.min())
        return out
    return units, run


def campaign_blackbox():
    units = []
    for d in (5, 8, 12, 16):
        for sig in (0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2):
            for h in (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3):
                for s in range(10):
                    units.append(dict(d=d, sigma=sig, h=h, seed=s))

    def run(u):
        from hspd.pnn import tanh_hessian_matrix, tanh_forward
        d, m = u['d'], 2
        codim = d * (d - 1) // 2
        r = max(2, int(round(0.5 * codim)))
        N = r + 4
        base = 50 * 10 ** 6 + 10 ** 5 * d + u['seed'] * 10
        W, A, b, c = random_tanh_net(d, r, m, seed=base + 1)
        X = np.random.default_rng(base + 2).normal(size=(N, d))
        Mex, _ = tanh_hessian_matrix(X, W, A, b)
        f = lambda Z: tanh_forward(Z, W, A, b, c)
        scale = float(np.sqrt(np.mean(f(X) ** 2)))
        t0 = time.perf_counter()
        M, nq = finite_difference_hessians(f, X, m, h=u['h'], noise_std=u['sigma'] * scale,
                                           seed=base + 7 + int(1e6 * u['h']))
        t_fd = time.perf_counter() - t0
        S = default_starts(r)
        V0 = random_starts(S, d, base + 4)
        res = run_method('HSPD', M, d, r, V0, base + 5)
        rec = summarize('HSPD', res, W)
        return {'r': r, 'm': m, 'N': N, 'queries': int(nq), 'queries_per_probe': queries_per_probe(d),
                'fd_sec': t_fd, 'output_rms': scale,
                'hessian_relative_error': float(np.linalg.norm(M - Mex) / np.linalg.norm(Mex)),
                'methods': [rec]}
    return units, run


def campaign_scaling():
    units = [dict(d=d, seed=s) for d in (8, 16, 24, 32, 48, 64) for s in range(3)]
    units.sort(key=lambda u: -u['d'])

    def run(u):
        import tracemalloc
        from hspd.teachers import hessian_matrix
        from hspd.pnn import tanh_forward
        d = u['d']
        codim = d * (d - 1) // 2
        r = int(round(0.5 * codim))
        m = 4
        N = r + 8
        base = 60 * 10 ** 6 + 10 ** 4 * d + u['seed']
        W, A, b, c = random_tanh_net(d, r, m, seed=base + 1)
        X = np.random.default_rng(base + 2).normal(size=(N, d))
        t0 = time.perf_counter()
        M, C, U = hessian_matrix(X, W, A, b, 'tanh')
        t_hess = time.perf_counter() - t0
        S = default_starts(r)
        V0 = random_starts(S, d, base + 4)
        tracemalloc.start()
        res = run_method('HSPD', M, d, r, V0, base + 5)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rec = summarize('HSPD', res, W)
        rec['peak_mem_bytes'] = int(peak)
        out = {'r': r, 'm': m, 'N': N, 'p': d * (d + 1) // 2, 'K': m * N,
               'hessian_sec': t_hess, 'methods': [rec]}
        if d <= 32:
            res2 = run_method('LBFGS', M, d, r, V0, base + 5)
            out['methods'].append(summarize('LBFGS', res2, W))
        # Black-box cost: finite-difference Hessians for the same probes.
        f = lambda Z: tanh_forward(Z, W, A, b, c)
        Np = min(N, 64)
        t1 = time.perf_counter()
        _, nq = finite_difference_hessians(f, X[:Np], m, h=1e-3)
        out['fd_sec_per_probe'] = (time.perf_counter() - t1) / Np
        out['queries_per_probe'] = queries_per_probe(d)
        return out
    return units, run


def campaign_realdata():
    units = []
    for s in range(5):
        units.append(dict(data='wine', act='cubic', r=20, mode='hessian1pct', seed=s))
        for r in (20, 60):
            for mode in ('hessian', 'hessian1pct', 'fd'):
                units.append(dict(data='wine', act='tanh', r=r, mode=mode, seed=s))
        for r in (32, 128, 512):
            units.append(dict(data='digits', act='tanh', r=r, mode='hessian', seed=s))
        for r in (32, 128):
            units.append(dict(data='digits', act='tanh', r=r, mode='fd', seed=s))
    units.sort(key=lambda u: -u['r'])

    def run(u):
        from hspd.teachers import (load_dataset, train_teacher, forward, hessian_matrix,
                                   distill_student, genericity_diagnostics)
        from hspd.metrics import matched_cosines
        Xtr, Xte, ytr, yte, m = load_dataset(u['data'], u['seed'])
        d, r, act = Xtr.shape[1], u['r'], u['act']
        epochs = 800 if act == 'cubic' else 1500
        lr = 0.012 if act == 'cubic' else 0.01
        W, A, b, c = train_teacher(Xtr, ytr, r, m, act, seed=u['seed'], epochs=epochs, lr=lr)
        Yte = forward(Xte, W, A, b, c, act)
        teacher_acc = float(np.mean(Yte.argmax(1) == yte))
        N = min(len(Xtr), max(2 * r, 300)) if u['data'] == 'digits' else len(Xtr)
        Xp = Xtr[:N]
        Yp = forward(Xp, W, A, b, c, act)
        Mex, C, U = hessian_matrix(Xp, W, A, b, act)
        sv = np.linalg.svd(C, compute_uv=False)
        gen, rel = genericity_diagnostics(W, A, Xp, act, b)
        kt = kappas(U, np.linalg.qr(svec_features(U))[0])
        t0 = time.perf_counter()
        nq = 0
        if u['mode'] == 'hessian':
            M = Mex
        elif u['mode'] == 'hessian1pct':
            rng = np.random.default_rng(51000 + u['seed'])
            E = rng.normal(size=Mex.shape)
            M = Mex + 0.01 * np.linalg.norm(Mex) / np.linalg.norm(E) * E
        else:
            f = lambda Z: forward(Z, W, A, b, c, act)
            M, nq = finite_difference_hessians(f, Xp, m, h=1e-3)
        t_obs = time.perf_counter() - t0
        S = default_starts(r)
        V0 = random_starts(S, d, 52000 + u['seed'])
        res = run_method('HSPD', M, d, r, V0, 52000 + u['seed'])
        rec = summarize('HSPD', res, U)
        cos, est = matched_cosines(U, res['W'])
        out = {'d': d, 'm': m, 'N': N, 'teacher_test_accuracy': teacher_acc,
               'context_rank': int(np.sum(sv > sv[0] * 1e-10)),
               'sigma_r_rel_C': float(sv[r - 1] / sv[0]) if len(sv) >= r else 0.0,
               'kappa_min': float(kt.min()), 'kappa_median': float(np.median(kt)),
               'genericity': gen, 'queries': int(nq), 'observation_sec': t_obs,
               'unit_rel_signal': jsonable(rel), 'unit_cos': jsonable(cos),
               'methods': [rec]}
        rng = np.random.default_rng(99 + u['seed'])
        Xood = rng.normal(size=(2000, d))
        Yood = forward(Xood, W, A, b, c, act)

        def evaluate(P, tag):
            Wq, Aq, bq, cq = P
            Ht = forward(Xte, Wq, Aq, bq, cq, act)
            Ho = forward(Xood, Wq, Aq, bq, cq, act)
            Un = Wq / np.linalg.norm(Wq, axis=0)
            cs, _ = matched_cosines(U, Un)
            out[tag] = {'test_accuracy': float(np.mean(Ht.argmax(1) == yte)),
                        'fidelity': float(np.mean(Ht.argmax(1) == Yte.argmax(1))),
                        'rel_logit_rmse_test': relative_rmse(Ht, Yte),
                        'rel_logit_rmse_ood': relative_rmse(Ho, Yood),
                        'direction_mean_cos': float(cs.mean()),
                        'direction_frac_099': float(np.mean(cs >= 0.99))}
        if res['W'].shape[1] == r:
            t1 = time.perf_counter()
            if act == 'cubic':
                est = recover_cubic_parameters(M, Xp, res['W'], Yp)
                evaluate((est.W, est.A, est.b, est.c), 'student_hspd')
            else:
                Ws, A2, b2, c2, fc = recover_tanh_parameters(M, Xp, res['W'], Yp)
                evaluate((Ws, A2, b2, c2), 'student_hspd')
                P = refine_tanh_student(Xp, Yp, Ws, A2, b2, c2)
                evaluate(P, 'student_hspd_refined')
            out['reconstruction_sec'] = time.perf_counter() - t1
        t2 = time.perf_counter()
        Pd = distill_student(Xtr, forward(Xtr, W, A, b, c, act), r, act, seed=u['seed'],
                             epochs=1500)
        out['distill_sec'] = time.perf_counter() - t2
        evaluate(Pd, 'student_distill')
        return out
    return units, run


def campaign_spm_check():
    """SPM deflation on the same Hessian geometry with orthogonal versus generic coefficients."""
    units = [dict(r=r, coeff=c, seed=s) for r in (4, 8, 12, 16, 20, 24)
             for c in ('orthogonal', 'gaussian', 'cubic') for s in range(20)]

    def run(u):
        d, K, r = 8, 128, u['r']
        base = 80 * 10 ** 6 + 1000 * r + u['seed']
        rng = np.random.default_rng(base)
        if u['coeff'] == 'cubic':
            net, X, M, C = cubic_instance(d, r, 4, 32, 0.0, base)
            W = net.W
        else:
            W = rng.normal(size=(d, r)); W /= np.linalg.norm(W, axis=0)
            if u['coeff'] == 'orthogonal':
                Cq, _ = np.linalg.qr(rng.normal(size=(K, r)))
                C = Cq * np.exp(rng.uniform(-0.5, 0.5, size=r))
            else:
                C = rng.normal(size=(K, r))
            M = svec_features(W) @ C.T
        G = C.T @ C
        off = float(np.max(np.abs(G - np.diag(np.diag(G)))) / np.max(np.diag(G)))
        res = run_method('SPM', M, d, r, None, base + 5)
        rec = summarize('SPM', res, W)
        return {'gram_offdiag_rel': off, 'methods': [rec]}
    return units, run


CAMPAIGNS = {
    'width8': campaign_width8, 'multid': campaign_multid, 'shift': campaign_shift,
    'noise': campaign_noise, 'context': campaign_context, 'cubic_full': campaign_cubic_full,
    'tanh': campaign_tanh, 'blackbox': campaign_blackbox, 'scaling': campaign_scaling,
    'realdata': campaign_realdata, 'spm_check': campaign_spm_check,
}


def main():
    units, run = CAMPAIGNS[args.campaign]()
    if args.list:
        print(len(units))
        return
    tid = args.task_id
    if tid is None:
        tid = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    mine = [u for i, u in enumerate(units) if i % args.n_tasks == tid]
    if args.limit:
        mine = mine[:args.limit]
    out_dir = ROOT / args.run_root / args.campaign / f'task-{tid:03d}'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'config.json').write_text(json.dumps(
        {'campaign': args.campaign, 'task_id': tid, 'n_tasks': args.n_tasks,
         'n_units': len(mine), 'argv': sys.argv}, indent=1))
    (out_dir / 'env.json').write_text(json.dumps(env_info(), indent=1))
    mpath = out_dir / 'metrics.json'
    records = json.loads(mpath.read_text())['records'] if mpath.exists() else []
    done = {json.dumps(r['unit'], sort_keys=True) for r in records}
    for k, u in enumerate(mine):
        key = json.dumps(u, sort_keys=True)
        if key in done:
            continue
        t0 = time.perf_counter()
        rec = run(u)
        rec = jsonable({'unit': u, **rec, 'unit_sec': time.perf_counter() - t0})
        records.append(rec)
        tmp = mpath.with_suffix('.tmp')
        tmp.write_text(json.dumps({'campaign': args.campaign, 'task_id': tid, 'records': records}))
        tmp.replace(mpath)
        print(f'[{args.campaign} task {tid}] unit {k + 1}/{len(mine)} {u} '
              f'{time.perf_counter() - t0:.1f}s', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
