#!/usr/bin/env python3
"""Diagnose direction recovery for a wide Digits tanh teacher (r=512).

Separates the two possible causes of failure: the global search (basins of
attraction, number of starts) and the local conditioning of the hidden
directions. Each stage is printed with flush and appended to a JSON file, so a
wall-time kill keeps the finished stages.

    python scripts/diagnostics/digits_wide.py <seed> [--out out/diag/digits_wide.json]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np  # noqa: E402
from hspd.teachers import load_dataset, train_teacher, hessian_matrix  # noqa: E402
from hspd.hspd import (hspd_recover, random_starts, kappas, subspace_basis,  # noqa: E402
                       newton_polish, stationarity_residual)
from hspd.svec import svec_features  # noqa: E402
from hspd.metrics import matched_cosines  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('seed', type=int, nargs='?', default=1)
ap.add_argument('--r', type=int, default=512)
ap.add_argument('--out', default=str(ROOT / 'out' / 'diag' / 'digits_wide.json'))
args = ap.parse_args()
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
log = {'seed': args.seed, 'r': args.r, 'stages': []}


def record(stage, **kw):
    kw = {k: (float(v) if isinstance(v, (np.floating,)) else int(v) if isinstance(v, (np.integer,)) else v)
          for k, v in kw.items()}
    log['stages'].append({'stage': stage, **kw})
    out_path.write_text(json.dumps(log, indent=1))
    print(stage, kw, flush=True)


seed, r = args.seed, args.r
t0 = time.perf_counter()
Xtr, Xte, ytr, yte, m = load_dataset('digits', seed)
d = Xtr.shape[1]
W, A, b, c = train_teacher(Xtr, ytr, r, m, 'tanh', seed=seed, epochs=1500, lr=0.01)
N = min(len(Xtr), max(2 * r, 300))
Xp = Xtr[:N]
M, C, U = hessian_matrix(Xp, W, A, b, 'tanh')
Bq = np.linalg.qr(svec_features(U))[0]
kt = kappas(U, Bq)
G = np.abs(U.T @ U)
np.fill_diagonal(G, 0)
nn_cos = G.max(axis=1)
Ur, Q, s = subspace_basis(M, r, d)
sin_theta = float(np.linalg.norm(Ur - Bq @ (Bq.T @ Ur), 2))
record('teacher', sec=time.perf_counter() - t0, sigma_ratio_M=s[r - 1] / s[0], gap_ratio=s[r] / s[r - 1],
       sin_theta=sin_theta, kappa_q=list(np.quantile(kt, [0, 0.05, 0.25, 0.5])),
       nn_cos_q=list(np.quantile(nn_cos, [0.5, 0.95, 1.0])))

# Local stage: starts near each true direction (angle about delta), Newton polish only.
rng = np.random.default_rng(5)
for delta in (0.05, 0.2):
    t1 = time.perf_counter()
    E = rng.normal(size=U.T.shape)
    E -= np.sum(E * U.T, axis=1, keepdims=True) * U.T
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    V0 = np.cos(delta) * U.T + np.sin(delta) * E
    V, F, st = newton_polish(Q, V0, max_steps=60)
    cosj = np.abs(np.sum(V * U.T, axis=1))
    res, _ = stationarity_residual(V, Q)
    record('oracle_starts', delta=delta, frac_back_to_own=float(np.mean(cosj >= 0.99)),
           frac_stationary=float(np.mean(res <= 1e-5)), F_min=float(F.min()), sec=time.perf_counter() - t1)

# Global stage: random starts with the default budget and with four times more.
for mult in (1, 4):
    t1 = time.perf_counter()
    S = int(np.ceil(5 * r * np.log(r))) * mult
    V0 = random_starts(S, d, 777 + mult)
    res = hspd_recover(M, d, r, V0=V0, max_iter=300, polish='newton', polish_steps=30, return_candidates=True)
    cos, _ = matched_cosines(U, res['W'])
    near = np.abs(res['V_final'] @ U)
    hits = (near >= 0.99).sum(axis=0)
    good = near.max(axis=1) >= 0.99
    stat = res['res_final'] <= 1e-5
    miss = cos < 0.99
    record('random_starts', S=S, frac_recovered=float(np.mean(~miss)), never_hit=int(np.sum(hits == 0)),
           frac_starts_true=float(good.mean()), frac_spurious_stationary=float(np.mean(stat & ~good)),
           spurious_F_max=float(res['F_final'][stat & ~good].max()) if np.any(stat & ~good) else None,
           max_hits_share=float(hits.max() / S), median_kappa_missed=float(np.median(kt[miss])) if miss.any() else None,
           median_kappa_found=float(np.median(kt[~miss])) if (~miss).any() else None,
           median_nn_cos_missed=float(np.median(nn_cos[miss])) if miss.any() else None,
           median_nn_cos_found=float(np.median(nn_cos[~miss])) if (~miss).any() else None,
           sec=time.perf_counter() - t1)
print('done', flush=True)
