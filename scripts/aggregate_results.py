#!/usr/bin/env python3
"""Collect the pulled campaign records (results/runs/<campaign>/task-*/metrics.json)
into tidy CSV files under results/csv/ (one row per unit and method), plus the
per-component and per-unit side tables used by the figures.

    python scripts/aggregate_results.py [--runs results/runs] [--out results/csv]

Set HSPD_RUNS=out/runs to aggregate the records of a rerun instead of the shipped ones.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--runs', default=os.environ.get('HSPD_RUNS', str(ROOT / 'results' / 'runs')))
ap.add_argument('--out', default=str(ROOT / 'results' / 'csv'))
args = ap.parse_args()
RUNS, OUT = Path(args.runs), Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

SCALAR_METHOD_KEYS = (
    'method', 'mean_cosine', 'min_cosine', 'frac_recovered', 'success', 'n_returned',
    'n_certified', 'n_false_certified', 'n_missed_certification', 'runtime_sec',
    'iterations_median', 'n_starts', 'frac_stationary', 'tau', 'eps_hat', 'nonmonotone_steps',
    'frac_nonmonotone_starts', 'frac_converged', 'power_sec', 'svd_sec', 'gamma', 'variant',
    'frac_starts_true', 'frac_starts_spurious_stationary', 'peak_mem_bytes')


def records(campaign):
    recs = []
    for f in sorted((RUNS / campaign).glob('task-*/metrics.json')):
        env = json.loads((f.parent / 'env.json').read_text()) if (f.parent / 'env.json').exists() else {}
        for r in json.loads(f.read_text())['records']:
            r['_env'] = env
            recs.append(r)
    return recs


def flat(campaign, keep=()):
    rows, comps, sens = [], [], []
    for r in records(campaign):
        unit = dict(r['unit'])
        base = {**{f'u_{k}': v for k, v in unit.items()}}
        for k in keep:
            if k in r:
                base[k] = r[k]
        base['cpu'] = r['_env'].get('cpu')
        base['threads'] = r['_env'].get('threads')
        for m in r.get('methods', []):
            row = dict(base)
            row.update({k: m.get(k) for k in SCALAR_METHOD_KEYS if k in m})
            if m.get('spurious_F_quantiles'):
                row['spurious_F_max'] = m['spurious_F_quantiles'][-1]
                row['spurious_F_q90'] = m['spurious_F_quantiles'][1]
            rows.append(row)
            if 'comp_cos' in m and 'kappa_true' in r:
                hits = m.get('basin_hits') or [None] * len(m['comp_cos'])
                for j, (c, k, h) in enumerate(zip(m['comp_cos'], r['kappa_true'], hits)):
                    comps.append({**base, 'method': m['method'], 'j': j, 'cos': c, 'kappa': k,
                                  'basin_hits': h, 'n_starts': m.get('n_starts')})
            for thr, v in (m.get('cluster_sensitivity') or {}).items():
                sens.append({**base, 'method': m['method'], 'threshold': float(thr), **v})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f'{campaign}.csv', index=False)
    if comps:
        pd.DataFrame(comps).to_csv(OUT / f'{campaign}_components.csv', index=False)
    if sens:
        pd.DataFrame(sens).to_csv(OUT / f'{campaign}_cluster_sensitivity.csv', index=False)
    print(f'{campaign}: {len(df)} rows, {len(comps)} component rows')
    return df


keep_common = ('d', 'm', 'N', 'codim', 'rho', 'r', 'context_rank', 'kappa_min', 'kappa_median',
               'capacity', 'sigma_r_rel', 'spectral_gap', 'sigma_r', 'sin_theta', 'queries',
               'queries_per_probe', 'fd_sec', 'output_rms', 'hessian_relative_error',
               'relative_rmse', 'weak_cos', 'weak_bias_err', 'weak_a_relerr',
               'weak_contrib_rmse_rel_to_output', 'bias_err_median', 'relative_rmse_coef',
               'relative_rmse_refined', 'profile_fit_cos_min', 'p', 'K', 'hessian_sec',
               'fd_sec_per_probe', 'unit_sec', 'gram_offdiag_rel')
for c in ('width8', 'multid', 'shift', 'noise', 'context', 'cubic_full', 'tanh', 'blackbox', 'scaling', 'spm_check'):
    if (RUNS / c).exists():
        flat(c, keep_common)

# Real data: one row per unit, with nested student metrics flattened.
if (RUNS / 'realdata').exists():
    rows, units = [], []
    for r in records('realdata'):
        row = {f'u_{k}': v for k, v in r['unit'].items()}
        for k in ('d', 'm', 'N', 'teacher_test_accuracy', 'context_rank', 'sigma_r_rel_C', 'kappa_min',
                  'kappa_median', 'queries', 'observation_sec', 'reconstruction_sec', 'distill_sec',
                  'unit_sec'):
            row[k] = r.get(k)
        for k, v in (r.get('genericity') or {}).items():
            row[f'gen_{k}'] = v
        m = r['methods'][0]
        for k in ('mean_cosine', 'min_cosine', 'frac_recovered', 'success', 'n_certified',
                  'n_false_certified', 'runtime_sec'):
            row[f'hspd_{k}'] = m.get(k)
        for tag in ('student_hspd', 'student_hspd_refined', 'student_distill'):
            for k, v in (r.get(tag) or {}).items():
                row[f'{tag}_{k}'] = v
        row['cpu'] = r['_env'].get('cpu')
        rows.append(row)
        for j, (sig, c) in enumerate(zip(r.get('unit_rel_signal', []), r.get('unit_cos', []))):
            units.append({**{f'u_{k}': v for k, v in r['unit'].items()}, 'j': j, 'rel_signal': sig,
                          'cos': c})
    pd.DataFrame(rows).to_csv(OUT / 'realdata.csv', index=False)
    pd.DataFrame(units).to_csv(OUT / 'realdata_units.csv', index=False)
    print(f'realdata: {len(rows)} rows')
