#!/usr/bin/env python3
"""Build every figure and LaTeX table of the paper from results/csv/*.csv into paper/.

    python scripts/aggregate_results.py && python scripts/make_artifacts.py

No number in the tables is typed by hand: each table below is
written from the aggregated campaign records.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'results' / 'csv'
FIG = ROOT / 'paper' / 'figures'
TAB = ROOT / 'paper' / 'tables'
SUM = ROOT / 'results' / 'summaries'
for p in (FIG, TAB, SUM):
    p.mkdir(parents=True, exist_ok=True)

# --- Style: reference categorical palette in its validated slot order --------
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
METHOD_STYLE = {  # fixed entity -> color/marker/linestyle (never by rank)
    'HSPD':          dict(color='#2a78d6', marker='o', ls='-',  label='HSPD (power + Newton)'),
    'LBFGS':         dict(color='#eb6834', marker='s', ls='--', label='Subspace L-BFGS'),
    'FKMR':          dict(color='#1baf7a', marker='^', ls='-.', label='FKMR iteration ($\\gamma=1/4$)'),
    'FVD':           dict(color='#eda100', marker='D', ls=':',  label='FVD projected ascent'),
    'SPM':           dict(color='#e87ba4', marker='v', ls='--', label='SPM with deflation'),
    'HSPD-power':    dict(color='#008300', marker='P', ls='-',  label='HSPD, power only'),
    'HSPD-shift0.5': dict(color='#6250d6', marker='X', ls=':',  label='HSPD, power only, $\\gamma=1/2$'),
    'Jennrich':      dict(color='#e34948', marker='*', ls='',   label='Jennrich ($r\\leq d$)'),
}
BLUE_RAMP = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']  # ordinal steps 250/400/550/700
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9, 'legend.fontsize': 7.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.linewidth': 0.6,
    'lines.linewidth': 1.5, 'lines.markersize': 4.5, 'legend.frameon': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'pdf.fonttype': 42, 'savefig.dpi': 300, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False, 'mathtext.fontset': 'dejavusans',
})


def savefig(fig, name):
    # Fixed metadata so that a rebuild from the same records is byte-identical.
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', metadata={'CreationDate': None})
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


def label_floor(fig):
    """Figure y-coordinate of the lowest tick or axis label (after tight_layout)."""
    rend = fig.canvas.get_renderer()
    return min(a.get_tightbbox(rend).transformed(fig.transFigure.inverted()).y0 for a in fig.axes)


def legend_below(fig, axs, handles, labels, y, ncol, **kw):
    """Legend centred under a group of panels, its top at y, for panels whose data
    leave no free corner. Entries read row by row (matplotlib fills columns first)."""
    idx = [i for c in range(ncol) for i in range(c, len(handles), ncol)]
    x = (axs[0].get_position().x0 + axs[-1].get_position().x1) / 2
    fig.legend([handles[i] for i in idx], [labels[i] for i in idx], loc='upper center',
               bbox_to_anchor=(x, y), ncol=ncol, fontsize=7, handlelength=1.8,
               columnspacing=1.2, borderaxespad=0.3, **kw)


def has(name):
    f = CSV / f'{name}.csv'
    return f.exists() and f.stat().st_size > 1 and len(pd.read_csv(f)) > 0


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def pct(x):
    return f'{100 * x:.0f}'


def fmt_sec(x):
    if x < 0.1:
        return f'{x:.3f}'
    if x < 10:
        return f'{x:.2f}'
    return f'{x:.0f}'


summary = {}

# =============================================================================
# Width sweep, d = 8
# =============================================================================
if has('width8'):
    w = pd.read_csv(CSV / 'width8.csv')
    g = w.groupby(['u_noise', 'u_r', 'method']).agg(
        n=('success', 'size'), succ=('success', 'mean'), frac=('frac_recovered', 'mean'),
        frac_sem=('frac_recovered', 'sem'), mean=('mean_cosine', 'mean'), t=('runtime_sec', 'median'),
        cert=('n_certified', 'mean'), fcert=('n_false_certified', 'mean')).reset_index()
    order = ['HSPD', 'LBFGS', 'FKMR', 'FVD', 'SPM', 'HSPD-power', 'HSPD-shift0.5']
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    for ax, eta, title in zip(axes, (0.0, 0.01), ('(a) exact Hessians', '(b) 1% Hessian perturbation')):
        for mth in order + ['Jennrich']:
            z = g[(g.u_noise == eta) & (g.method == mth)].sort_values('u_r')
            if z.empty:
                continue
            st = METHOD_STYLE[mth]
            ax.errorbar(z.u_r, z.frac, yerr=z.frac_sem, color=st['color'], marker=st['marker'],
                        ls=st['ls'], label=st['label'], capsize=1.5, elinewidth=0.8,
                        markersize=6 if mth == 'Jennrich' else 4.5)
        ax.axvline(28, color=INK2, lw=0.8, ls=(0, (4, 3)))
        ax.text(27.6, 0.04, '$d(d-1)/2=28$', rotation=90, ha='right', va='bottom', fontsize=7, color=INK2)
        ax.set_xlabel('Hidden width $r$')
        ax.set_title(title, loc='left')
        ax.set_xticks([4, 8, 12, 16, 20, 24, 28, 32])
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel('Fraction of directions recovered\n(matched |cos| $\\geq$ 0.99)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    savefig(fig, 'width_recovery')

    # Table: success rate (%) and median runtime per method.
    rs = sorted(g.u_r.unique())
    cols = ['HSPD', 'LBFGS', 'FKMR', 'FVD', 'SPM', 'HSPD-power']
    lines = ['\\begin{tabular}{r ' + ' '.join(['rr'] * len(cols)) + ' r}', '\\toprule',
             ' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{c}}}' for c in
                                ['HSPD', 'L-BFGS', 'FKMR', 'FVD', 'SPM-defl.', 'HSPD-power']) + ' & \\\\',
             ' '.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(cols))),
             '$r$ & ' + ' & '.join(['exact & 1\\%'] * len(cols)) + ' & $\\kappa_{\\min}$\\\\', '\\midrule']
    kmin = w.groupby('u_r').kappa_min.median()
    for r in rs:
        cells = [str(r)]
        for c in cols:
            for eta in (0.0, 0.01):
                z = g[(g.u_noise == eta) & (g.u_r == r) & (g.method == c)]
                cells.append(pct(z.succ.iloc[0]) if len(z) else 'n/a')
        km = kmin.loc[r]
        cells.append('0' if km < 1e-10 else f'{km:.3f}')
        lines.append(' & '.join(cells) + ' \\\\')
    lines += ['\\midrule', 'time (s) & ' + ' & '.join(
        f'\\multicolumn{{2}}{{c}}{{{fmt_sec(g[(g.method == c) & (g.u_r == 24)].t.median())}}}' for c in cols) + ' & \\\\',
        '\\bottomrule', '\\end{tabular}']
    (TAB / 'width.tex').write_text('\n'.join(lines) + '\n')
    summary['width8'] = g.to_dict(orient='records')
    summary['width8_n_min'] = int(g.n.min())

    # Threshold sensitivity (HSPD candidates, re-clustered) and certification.
    if has('width8_cluster_sensitivity'):
        cs = pd.read_csv(CSV / 'width8_cluster_sensitivity.csv')
        summary['width8_cluster'] = cs.groupby(['u_noise', 'u_r', 'threshold']).agg(
            succ=('success', 'mean'), frac=('frac_recovered', 'mean')).reset_index().to_dict(orient='records')

# =============================================================================
# Dimension sweep
# =============================================================================
if has('multid'):
    md = pd.read_csv(CSV / 'multid.csv')
    md['rho_r'] = md['u_r'] / md['codim']
    g = md.groupby(['u_d', 'u_noise', 'u_r', 'method']).agg(
        n=('success', 'size'), succ=('success', 'mean'), frac=('frac_recovered', 'mean'),
        t=('runtime_sec', 'median'), codim=('codim', 'first'), kmin=('kappa_min', 'median')).reset_index()
    g['rho'] = g.u_r / g.codim
    dims = sorted(g.u_d.unique())
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for ax, eta, title in zip(axes[:2], (0.0, 0.01), ('(a) exact Hessians', '(b) 1% perturbation')):
        for col, dd in zip(BLUE_RAMP, dims):
            z = g[(g.u_d == dd) & (g.u_noise == eta) & (g.method == 'HSPD')].sort_values('rho')
            ax.plot(z.rho, z.succ, color=col, marker='o', label=f'$d={dd}$')
            zl = g[(g.u_d == dd) & (g.u_noise == eta) & (g.method == 'LBFGS')].sort_values('rho')
            ax.plot(zl.rho, zl.succ, color=col, ls=':', lw=1.0)
        ax.axvline(1.0, color=INK2, lw=0.8, ls=(0, (4, 3)))
        ax.set_xlabel('$r\\,/\\,[d(d-1)/2]$')
        ax.set_title(title, loc='left')
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel('Success rate (all $r$ recovered)')
    ax = axes[2]
    for mth in ('HSPD', 'LBFGS', 'FKMR', 'SPM', 'FVD'):
        z = g[(g.method == mth) & (g.u_noise == 0.0)]
        z = z.loc[(z.rho - 0.5).abs().groupby(z.u_d).idxmin()] if len(z) else z
        z = z.sort_values('u_d')
        st = METHOD_STYLE[mth]
        # Short names, as in the table headers.
        ax.plot(z.u_d, z.t, color=st['color'], marker=st['marker'], ls=st['ls'],
                label={'LBFGS': 'L-BFGS', 'SPM': 'SPM-defl.'}.get(mth, mth))
    ax.set_yscale('log')
    ax.set_xlabel('Input dimension $d$')
    ax.set_ylabel('Median wall-clock (s)')
    ax.set_title('(c) runtime at $r\\approx d(d-1)/4$', loc='left')
    fig.tight_layout()
    y = label_floor(fig)
    legend_below(fig, axes[:2], *axes[0].get_legend_handles_labels(), y, ncol=4)
    legend_below(fig, axes[2:], *axes[2].get_legend_handles_labels(), y, ncol=2)
    savefig(fig, 'multid_recovery')
    summary['multid'] = g.to_dict(orient='records')

    # Table: success per d and rho for HSPD and L-BFGS (exact and 1%).
    lines = ['\\begin{tabular}{r r r rr rr rr r}', '\\toprule',
             '& & & \\multicolumn{2}{c}{HSPD} & \\multicolumn{2}{c}{L-BFGS} & \\multicolumn{2}{c}{FKMR} & \\\\',
             '\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}',
             '$d$ & $r$ & $r/\\mathrm{codim}$ & exact & 1\\% & exact & 1\\% & exact & 1\\% & $\\kappa_{\\min}$\\\\', '\\midrule']
    for dd in dims:
        for r in sorted(g[g.u_d == dd].u_r.unique()):
            row = [str(dd), str(r), f'{r / g[g.u_d == dd].codim.iloc[0]:.2f}']
            for mth in ('HSPD', 'LBFGS', 'FKMR'):
                for eta in (0.0, 0.01):
                    z = g[(g.u_d == dd) & (g.u_r == r) & (g.method == mth) & (g.u_noise == eta)]
                    row.append(pct(z.succ.iloc[0]) if len(z) else 'n/a')
            km = g[(g.u_d == dd) & (g.u_r == r)].kmin.median()
            row.append('0' if km < 1e-10 else f'{km:.1e}')
            lines.append(' & '.join(row) + ' \\\\')
        lines.append('\\midrule' if dd != dims[-1] else '')
    lines += ['\\bottomrule', '\\end{tabular}']
    (TAB / 'multid.tex').write_text('\n'.join(l for l in lines if l) + '\n')

# =============================================================================
# Shift study
# =============================================================================
if has('shift'):
    sh = pd.read_csv(CSV / 'shift.csv')
    sh['rho'] = sh['u_rho']
    g = sh.groupby(['u_d', 'u_rho', 'u_noise', 'gamma', 'variant']).agg(
        succ=('success', 'mean'), frac=('frac_recovered', 'mean'), it=('iterations_median', 'median'),
        nonmono=('frac_nonmonotone_starts', 'mean'), t=('runtime_sec', 'median')).reset_index()
    summary['shift'] = g.to_dict(orient='records')
    gammas = sorted(g.gamma.unique())
    lines = ['\\begin{tabular}{r ' + 'r' * len(gammas) + '}', '\\toprule',
             'Quantity & ' + ' & '.join(f'$\\gamma={x:g}$' for x in gammas) + '\\\\', '\\midrule']

    def row(label, q, **sel):
        z = g
        for k, v in sel.items():
            z = z[z[k] == v]
        vals = [z[z.gamma == x][q].mean() for x in gammas]
        return label + ' & ' + ' & '.join(vals_fmt(q, v) for v in vals) + ' \\\\'

    def vals_fmt(q, v):
        if q in ('succ', 'nonmono'):
            return pct(v)
        if q == 'it':
            return f'{v:.0f}'
        return f'{v:.3f}'
    for dd in sorted(g.u_d.unique()):
        lines.append(f'\\multicolumn{{{len(gammas) + 1}}}{{l}}{{\\emph{{$d={dd}$, exact Hessians}}}}\\\\')
        for rho in (0.8, 0.9, 1.0):
            lines.append(row(f'success \\%, power, $\\rho={rho:g}$', 'succ', u_d=dd, u_rho=rho, u_noise=0.0, variant='power'))
            lines.append(row(f'success \\%, polished, $\\rho={rho:g}$', 'succ', u_d=dd, u_rho=rho, u_noise=0.0, variant='polish'))
        lines.append(row('median iterations, power, $\\rho=0.9$', 'it', u_d=dd, u_rho=0.9, u_noise=0.0, variant='power'))
        lines.append(row('starts with an $F$ decrease (\\%)', 'nonmono', u_d=dd, u_noise=0.0, variant='power'))
    lines += ['\\bottomrule', '\\end{tabular}']
    (TAB / 'shift.tex').write_text('\n'.join(lines) + '\n')

# =============================================================================
# Conditioning: kappa vs rho, perturbation law, basins
# =============================================================================
if has('noise_components'):
    nc = pd.read_csv(CSV / 'noise_components.csv')
    nc = nc[nc.method == 'HSPD']
    nc = nc[(nc.u_noise > 0) & (nc.cos > 0)]
    nc['angle'] = np.arccos(np.clip(nc['cos'], -1, 1))
    nc['pred'] = nc['sin_theta'] / nc['kappa']
    nc['ratio'] = nc['angle'] / nc['pred']
    nc['regime'] = nc['sin_theta'] < 0.5 * nc['kappa'] ** 2
    inr = nc[nc.regime]
    summary['perturbation_law'] = {
        'n': int(len(nc)), 'n_regime': int(len(inr)), 'max_ratio_regime': float(inr.ratio.max()),
        'median_ratio_regime': float(inr.ratio.median()), 'max_ratio_all': float(nc.ratio.max())}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45))
    ax = axes[0]
    if has('multid'):
        md = pd.read_csv(CSV / 'multid.csv')
        mk = md[md.method == 'HSPD'].groupby(['u_d', 'u_r']).agg(k=('kappa_min', 'median'), codim=('codim', 'first')).reset_index()
        for col, dd in zip(BLUE_RAMP, sorted(mk.u_d.unique())):
            z = mk[(mk.u_d == dd) & (mk.u_r <= mk.codim + 1)].sort_values('u_r')
            ax.plot(z.u_r / z.codim, z.k, color=col, marker='o', label=f'$d={dd}$')
    ax.set_yscale('log')
    ax.axvline(1.0, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.set_xlabel('$r\\,/\\,[d(d-1)/2]$')
    ax.set_ylabel('Median $\\min_j \\kappa_j$')
    ax.set_title('(a) conditioning near the bound', loc='left')
    ax = axes[1]
    out = nc[~nc.regime]
    ax.scatter(out.pred, out.angle, s=4, color='#c3c2b7', lw=0, label='$\\sin\\Theta\\geq\\kappa_j^2/2$')
    for col, (dd, eta) in zip(['#2a78d6', '#eb6834', '#1baf7a', '#6250d6'], [(8, 0.001), (8, 0.01), (16, 0.001), (16, 0.01)]):
        z = inr[(inr.u_d == dd) & (np.isclose(inr.u_noise, eta))]
        ax.scatter(z.pred, z.angle, s=5, color=col, alpha=0.6, lw=0, label=f'$d={dd}$, {100 * eta:g}%')
    xx = np.geomspace(1e-5, 10, 50)
    ax.plot(xx, xx, color=INK, lw=0.9, ls='--', label='$y=x$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-4, 10); ax.set_ylim(1e-6, 3)
    ax.set_xlabel('$\\sin\\Theta\\,/\\,\\kappa_j$')
    ax.set_ylabel('Angle error of $\\hat w_j$ (rad)')
    ax.set_title('(b) first-order sensitivity', loc='left')
    ax = axes[2]
    if has('width8_components'):
        wc = pd.read_csv(CSV / 'width8_components.csv')
        wc = wc[(wc.method == 'HSPD') & (wc.u_noise == 0.0)]
        wc['p'] = wc.basin_hits / wc.n_starts
        for col, r in zip(['#2a78d6', '#eb6834', '#1baf7a', '#eda100'], (12, 20, 24, 28)):
            z = wc[wc.u_r == r]
            ax.scatter(z.kappa, z.p * r, s=5, color=col, alpha=0.5, lw=0, label=f'$r={r}$')
        ax.set_xscale('log')
        ax.set_xlabel('$\\kappa_j$')
        ax.set_ylabel('Basin share $\\times\\, r$')
        ax.set_title('(c) basins of attraction, $d=8$', loc='left')
    fig.tight_layout()
    y = label_floor(fig)
    if has('multid'):
        legend_below(fig, axes[:1], *axes[0].get_legend_handles_labels(), y, ncol=2)
    h, l = axes[1].get_legend_handles_labels()
    k = [1, 2, 3, 4, 0, 5]  # the four groups first, then the grey points and y = x
    legend_below(fig, axes[1:2], [h[i] for i in k], [l[i] for i in k], y, ncol=2, markerscale=2)
    if has('width8_components'):
        legend_below(fig, axes[2:], *axes[2].get_legend_handles_labels(), y, ncol=2, markerscale=2)
    savefig(fig, 'conditioning')

# =============================================================================
# Noise robustness
# =============================================================================
if has('noise'):
    nz = pd.read_csv(CSV / 'noise.csv')
    nz = nz[nz.method == 'HSPD']
    g = nz.groupby(['u_d', 'u_noise']).agg(
        succ=('success', 'mean'), frac=('frac_recovered', 'mean'), mean=('mean_cosine', 'mean'),
        worst=('min_cosine', 'mean'), sin=('sin_theta', 'mean'), eps=('eps_hat', 'mean'),
        gap=('spectral_gap', 'mean'), cert=('n_certified', 'mean'), fcert=('n_false_certified', 'mean'),
        r=('n_returned', 'max')).reset_index()
    lines = ['\\begin{tabular}{r r r r r r r r r}', '\\toprule',
             '$d$ & Noise (\\%) & Success (\\%) & Recovered & Worst cosine & $\\sin\\Theta$ & $\\hat\\varepsilon$ & Certified & False cert.\\\\',
             '\\midrule']
    for _, x in g.iterrows():
        lines.append(f"{int(x.u_d)} & {100 * x.u_noise:g} & {pct(x.succ)} & {x.frac:.3f} & {x.worst:.4f} & {x.sin:.3f} & {x.eps:.3f} & {x.cert:.1f} & {x.fcert:.2f} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    (TAB / 'noise.tex').write_text('\n'.join(lines) + '\n')
    summary['noise'] = g.to_dict(orient='records')
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharex=True)
    for col, mk, dd in zip(['#2a78d6', '#eb6834'], ['o', 's'], sorted(g.u_d.unique())):
        z = g[g.u_d == dd]
        lab = f'$d={dd}$, $r={int(z.r.max())}$'
        axes[0].plot(100 * z.u_noise, z.frac, color=col, marker=mk, label=lab)
        axes[1].plot(100 * z.u_noise, z.worst.clip(lower=0), color=col, marker=mk, label=lab)
    axes[0].set_ylabel('Fraction of directions recovered')
    axes[1].set_ylabel('Mean worst-component cosine')
    axes[0].set_title('(a) recovered fraction', loc='left')
    axes[1].set_title('(b) worst component', loc='left')
    for ax in axes:
        ax.set_xlabel('Relative Hessian perturbation (%)')
        ax.set_ylim(-0.03, 1.05)
        ax.legend(loc='lower left', fontsize=7)
    fig.tight_layout()
    savefig(fig, 'noise_robustness')

# =============================================================================
# Context excitation
# =============================================================================
if has('context'):
    c = pd.read_csv(CSV / 'context.csv')
    g = c.groupby(['u_act', 'u_m', 'u_N']).agg(rank=('context_rank', 'mean'), rmin=('context_rank', 'min'),
                                                rmax=('context_rank', 'max'), cap=('capacity', 'mean'),
                                                succ=('success', 'mean'), frac=('frac_recovered', 'mean')).reset_index()
    summary['context'] = g.to_dict(orient='records')
    summary['context_rank_matches_capacity'] = bool(((g.rmin == g.cap) & (g.rmax == g.cap)).all())
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
    for ax, act, title, rr in zip(axes, ('cubic', 'tanh'), ('(a) cubic, $d=8$, $r=16$', '(b) tanh, $d=8$, $r=20$'), (16, 20)):
        cols = ['#2a78d6', '#eb6834', '#1baf7a']
        for col, m in zip(cols, sorted(g[g.u_act == act].u_m.unique())):
            z = g[(g.u_act == act) & (g.u_m == m)].sort_values('u_N')
            ax.plot(z.u_N, z.frac, color=col, marker='o', label=f'$m={m}$ output' + ('s' if m > 1 else ''))
        ax.set_xscale('log', base=2)
        ax.set_xticks([4, 8, 16, 32, 64]); ax.set_xticklabels(['4', '8', '16', '32', '64'])
        ax.set_xlabel('Number of probes $N$')
        ax.set_title(title, loc='left')
        ax.set_ylim(-0.03, 1.05)
        ax.legend(loc='center right', fontsize=7)
    axes[0].set_ylabel('Fraction recovered')
    fig.tight_layout()
    savefig(fig, 'context_ablation')
    lines = ['\\begin{tabular}{l r r r r r}', '\\toprule',
             'Activation & $m$ & $N$ & Capacity & rank$(C)$ & Success (\\%)\\\\', '\\midrule']
    for _, x in g.iterrows():
        if x.u_act == 'tanh' and x.u_N not in (8, 12, 16, 20, 24):
            continue
        lines.append(f"{x.u_act} & {int(x.u_m)} & {int(x.u_N)} & {x.cap:.0f} & {x['rank']:.0f} & {pct(x.succ)} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    (TAB / 'context.tex').write_text('\n'.join(lines) + '\n')

# =============================================================================
# Cubic reconstruction and weak units
# =============================================================================
if has('cubic_full'):
    cf = pd.read_csv(CSV / 'cubic_full.csv')
    full = cf[cf.u_kind == 'full'].groupby('u_r').agg(succ=('success', 'mean'), mean=('mean_cosine', 'mean'),
                                                       worst=('min_cosine', 'mean'), rmse=('relative_rmse', 'median'),
                                                       rmse_mean=('relative_rmse', 'mean'), bias=('bias_err_median', 'median')).reset_index()
    weak = cf[cf.u_kind == 'weak'].groupby('u_rho_w').agg(cos=('weak_cos', 'median'), bias=('weak_bias_err', 'median'),
                                                          a=('weak_a_relerr', 'median'), contrib=('weak_contrib_rmse_rel_to_output', 'median'),
                                                          rmse=('relative_rmse', 'median'), succ=('success', 'mean')).reset_index().sort_values('u_rho_w', ascending=False)
    summary['cubic_full'] = full.to_dict(orient='records')
    summary['cubic_weak'] = weak.to_dict(orient='records')
    lines = ['\\begin{tabular}{r r r r r}', '\\toprule',
             '$r$ & Success (\\%) & Worst cosine & Median bias error & Median rel.\\ output RMSE\\\\', '\\midrule']
    for _, x in full.iterrows():
        lines.append(f"{int(x.u_r)} & {pct(x.succ)} & {x.worst:.5f} & {x.bias:.1e} & {x.rmse:.2e} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    (TAB / 'recovery.tex').write_text('\n'.join(lines) + '\n')
    lines = ['\\begin{tabular}{r r r r r r}', '\\toprule',
             '$\\rho_a$ & Cosine of $\\hat w_1$ & $|\\hat b_1-b_1|$ & $\\|\\hat a_1-a_1\\|/\\|a_1\\|$ & Unit-1 output error & Rel.\\ output RMSE\\\\', '\\midrule']
    for _, x in weak.iterrows():
        lines.append(f"{x.u_rho_w:g} & {x.cos:.5f} & {x.bias:.1e} & {x.a:.1e} & {x.contrib:.1e} & {x.rmse:.2e} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    (TAB / 'weak_unit.tex').write_text('\n'.join(lines) + '\n')

# =============================================================================
# Synthetic tanh networks
# =============================================================================
if has('tanh'):
    t = pd.read_csv(CSV / 'tanh.csv')
    g = t.groupby('u_noise').agg(succ=('success', 'mean'), mean=('mean_cosine', 'mean'), worst=('min_cosine', 'mean'),
                                 coef=('relative_rmse_coef', 'median'), ref=('relative_rmse_refined', 'median')).reset_index()
    summary['tanh'] = g.to_dict(orient='records')
    lines = ['\\begin{tabular}{r r r r r r}', '\\toprule',
             'Noise (\\%) & Success (\\%) & Mean cosine & Worst cosine & Rel.\\ RMSE (coefficients) & Rel.\\ RMSE (refined)\\\\', '\\midrule']
    for _, x in g.iterrows():
        lines.append(f"{100 * x.u_noise:g} & {pct(x.succ)} & {x['mean']:.6f} & {x.worst:.5f} & {x.coef:.1e} & {x.ref:.1e} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    (TAB / 'tanh.tex').write_text('\n'.join(lines) + '\n')

# =============================================================================
# Black-box finite differences with query noise
# =============================================================================
if has('blackbox'):
    b = pd.read_csv(CSV / 'blackbox.csv')
    sig_levels = sorted(b.u_sigma.unique())
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax = axes[0]
    z12 = b[b.u_d == 12]
    cols = ['#0b0b0b', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b']
    for col, sg in zip(cols, sig_levels):
        z = z12[z12.u_sigma == sg].groupby('u_h').hessian_relative_error.median().reset_index()
        ax.plot(z.u_h, z.hessian_relative_error, color=col, marker='o',
                label='$\\sigma=0$' if sg == 0 else f'$\\sigma=10^{{{int(np.log10(sg))}}}$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Finite-difference step $h$')
    ax.set_ylabel('Relative Hessian error')
    ax.set_title('(a) error vs step, $d=12$', loc='left')
    ax = axes[1]
    piv = b[b.u_d == 16].pivot_table(index='u_sigma', columns='u_h', values='success', aggfunc='mean')
    im = ax.imshow(piv.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        'blue', ['#f4f8fd', '#9ec5f4', '#3987e5', '#184f95', '#0d366b']), vmin=0, vmax=1, aspect='auto', origin='lower')
    ax.grid(False)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f'{x:g}' for x in piv.columns], fontsize=7)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(['0' if x == 0 else f'$10^{{{int(np.log10(x))}}}$' for x in piv.index], fontsize=7)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f'{100 * v:.0f}', ha='center', va='center', fontsize=6.5,
                    color='white' if v > 0.55 else INK)
    ax.set_xlabel('Finite-difference step $h$')
    ax.set_ylabel('Query noise $\\sigma$ (relative)')
    ax.set_title('(b) success rate (%), $d=16$, $r=60$', loc='left')
    fig.tight_layout()
    legend_below(fig, axes[:1], *axes[0].get_legend_handles_labels(), label_floor(fig), ncol=3)
    savefig(fig, 'blackbox_noise')
    g = b.groupby(['u_d']).agg(r=('r', 'first'), N=('N', 'first'), q=('queries', 'first'),
                               qp=('queries_per_probe', 'first'), fd=('fd_sec', 'median')).reset_index()
    best = b.groupby(['u_d', 'u_sigma', 'u_h']).success.mean().reset_index()
    besth = best.loc[best.groupby(['u_d', 'u_sigma']).success.idxmax()]
    summary['blackbox'] = {'sizes': g.to_dict(orient='records'), 'best': besth.to_dict(orient='records')}
    lines = ['\\begin{tabular}{r r r r r r ' + 'r' * len(sig_levels) + '}', '\\toprule',
             '& & & & & & \\multicolumn{%d}{c}{Best success (\\%%) over $h$ at query noise $\\sigma$}\\\\' % len(sig_levels),
             '\\cmidrule(lr){7-%d}' % (6 + len(sig_levels)),
             '$d$ & $r$ & $m$ & $N$ & Queries/probe & Total queries & ' + ' & '.join(
                 '0' if s == 0 else f'$10^{{{int(np.log10(s))}}}$' for s in sig_levels) + '\\\\', '\\midrule']
    for _, x in g.iterrows():
        bs = [best[(best.u_d == x.u_d) & (best.u_sigma == s)].success.max() for s in sig_levels]
        lines.append(f"{int(x.u_d)} & {int(x.r)} & 2 & {int(x.N)} & {int(x.qp):,} & {int(x.q):,} & " + ' & '.join(pct(v) for v in bs) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (TAB / 'blackbox.tex').write_text('\n'.join(lines).replace(',', '{,}') + '\n')

# =============================================================================
# Runtime and memory scaling
# =============================================================================
if has('scaling'):
    sc = pd.read_csv(CSV / 'scaling.csv')
    h = sc[sc.method == 'HSPD'].groupby('u_d').agg(r=('r', 'first'), p=('p', 'first'), K=('K', 'first'),
                                                  S=('n_starts', 'first'), t=('runtime_sec', 'median'),
                                                  svd=('svd_sec', 'median'), power=('power_sec', 'median'),
                                                  hess=('hessian_sec', 'median'), mem=('peak_mem_bytes', 'median'),
                                                  succ=('success', 'mean'), fdp=('fd_sec_per_probe', 'median'),
                                                  qp=('queries_per_probe', 'first')).reset_index()
    lb = sc[sc.method == 'LBFGS'].groupby('u_d').runtime_sec.median()
    summary['scaling'] = h.to_dict(orient='records')
    summary['scaling_lbfgs'] = lb.to_dict()
    lines = ['\\begin{tabular}{r r r r r r r r r r}', '\\toprule',
             '$d$ & $r$ & $p$ & Starts & SVD (s) & Total (s) & L-BFGS (s) & Peak memory (GB) & Success (\\%) & FD (s/probe)\\\\', '\\midrule']
    for _, x in h.iterrows():
        lbv = fmt_sec(lb.loc[x.u_d]) if x.u_d in lb.index else 'not run'
        lines.append(f"{int(x.u_d)} & {int(x.r)} & {int(x.p)} & {int(x.S):,} & {fmt_sec(x.svd)} & {fmt_sec(x.t)} & {lbv} & {x.mem / 1e9:.2f} & {pct(x.succ)} & {x.fdp:.3f} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    (TAB / 'scaling.tex').write_text('\n'.join(lines).replace(',', '{,}') + '\n')

# =============================================================================
# Real data: Wine and Digits teachers, distillation comparison
# =============================================================================
if has('realdata'):
    rd = pd.read_csv(CSV / 'realdata.csv')
    keys = ['u_data', 'u_act', 'u_r', 'u_mode']
    agg = {'teacher_test_accuracy': 'mean', 'hspd_frac_recovered': 'mean', 'hspd_success': 'mean',
           'hspd_min_cosine': 'mean', 'hspd_runtime_sec': 'median', 'queries': 'first',
           'gen_max_abs_cos_pair': 'max', 'gen_min_rel_signal': 'min', 'kappa_min': 'median',
           'sigma_r_rel_C': 'median'}
    for tag in ('student_hspd', 'student_hspd_refined', 'student_distill'):
        for k in ('fidelity', 'rel_logit_rmse_test', 'rel_logit_rmse_ood', 'direction_mean_cos', 'test_accuracy'):
            col = f'{tag}_{k}'
            if col in rd:
                agg[col] = 'median' if 'rmse' in k else 'mean'
    agg = {k: v for k, v in agg.items() if k in rd}
    g = rd.groupby(keys).agg(agg).reset_index()
    summary['realdata'] = g.to_dict(orient='records')
    mode_name = {'hessian': 'exact', 'hessian1pct': '1\\% pert.', 'fd': 'finite diff.'}
    full = rd.assign(full=rd.hspd_success.astype(float)).groupby(keys).agg(k=('full', 'sum'), n=('full', 'size')).reset_index()
    g = g.merge(full, on=keys)
    summary['realdata'] = g.to_dict(orient='records')
    order = [('wine', 'cubic', 20, 'hessian1pct')] + [('wine', 'tanh', r, m) for r in (20, 60) for m in ('hessian', 'hessian1pct', 'fd')] + \
            [('digits', 'tanh', r, 'hessian') for r in (32, 128, 512)] + [('digits', 'tanh', r, 'fd') for r in (32, 128)]
    lines = ['\\begin{tabular}{l l r l r r r r r r r}', '\\toprule',
             '& & & & & \\multicolumn{2}{c}{Directions} & \\multicolumn{2}{c}{HSPD student} & \\multicolumn{2}{c}{Distilled student}\\\\',
             '\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}',
             'Data & $\\phi$ & $r$ & Hessians & Teacher acc. & Recovered & Teachers & Test & Off-dist. & Test & Off-dist.\\\\', '\\midrule']
    for key in order:
        z = g[(g.u_data == key[0]) & (g.u_act == key[1]) & (g.u_r == key[2]) & (g.u_mode == key[3])]
        if z.empty:
            continue
        x = z.iloc[0]
        hs = 'student_hspd_refined' if key[1] == 'tanh' else 'student_hspd'

        def e(v):
            return 'n/a' if pd.isna(v) else (f'{v:.1e}' if v >= 1e-12 else '$<10^{-12}$')
        if x.k == 0 and x.hspd_frac_recovered < 0.5:
            st_t = st_o = 'fail'
        else:
            st_t, st_o = e(x[f'{hs}_rel_logit_rmse_test']), e(x[f'{hs}_rel_logit_rmse_ood'])
        lines.append(f"{key[0].capitalize()} & {key[1]} & {key[2]} & {mode_name[key[3]]} & {x.teacher_test_accuracy:.3f} & "
                     f"{x.hspd_frac_recovered:.3f} & {int(x.k)}/{int(x.n)} & {st_t} & {st_o} & "
                     f"{e(x['student_distill_rel_logit_rmse_test'])} & {e(x['student_distill_rel_logit_rmse_ood'])} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    (TAB / 'realdata.tex').write_text('\n'.join(lines) + '\n')
    gen = rd.groupby(['u_data', 'u_act', 'u_r']).agg(
        acc=('teacher_test_accuracy', 'mean'), maxcos=('gen_max_abs_cos_pair', 'median'),
        maxcos_max=('gen_max_abs_cos_pair', 'max'), weak=('gen_n_weak_units_1e-1', 'mean'),
        minsig=('gen_min_rel_signal', 'median'), sat=('gen_median_saturation', 'median'),
        kmin=('kappa_min', 'median'), sr=('sigma_r_rel_C', 'median')).reset_index()
    summary['realdata_genericity'] = gen.to_dict(orient='records')
    lines = ['\\begin{tabular}{l l r r r r r r r}', '\\toprule',
             'Data & $\\phi$ & $r$ & Teacher acc. & Max.\\ $|\\cos(w_i,w_j)|$ & Min.\\ rel.\\ signal & Median saturation & $\\sigma_r(C)/\\sigma_1(C)$ & $\\min_j\\kappa_j$\\\\', '\\midrule']
    for _, x in gen.iterrows():
        sat = 'n/a' if pd.isna(x.sat) else f'{x.sat:.2f}'
        lines.append(f"{x.u_data.capitalize()} & {x.u_act} & {int(x.u_r)} & {x.acc:.3f} & {x.maxcos:.3f} & {x.minsig:.2f} & {sat} & {x.sr:.1e} & {x.kmin:.3f} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    (TAB / 'genericity.tex').write_text('\n'.join(lines) + '\n')
    # Figure: fidelity of HSPD vs distilled students on Digits.
    dg = g[(g.u_data == 'digits') & (g.u_mode == 'hessian')].sort_values('u_r').reset_index(drop=True)
    if len(dg):
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        xs = np.arange(len(dg))
        for tag, lab, col, mk, offs in (('student_hspd_refined', 'HSPD student', '#2a78d6', 'o', (-0.24, -0.09)),
                                        ('student_distill', 'Distilled student', '#eb6834', 's', (0.09, 0.24))):
            ok = (dg.k > 0).values if tag == 'student_hspd_refined' else np.ones(len(dg), bool)
            ax.scatter(xs[ok] + offs[0], dg.loc[ok, f'{tag}_rel_logit_rmse_test'].clip(lower=1e-12), color=col, marker=mk, s=30,
                       label=f'{lab}, test inputs')
            ax.scatter(xs[ok] + offs[1], dg.loc[ok, f'{tag}_rel_logit_rmse_ood'].clip(lower=1e-12), facecolors='white',
                       edgecolors=col, marker=mk, s=30, label=f'{lab}, off-distribution')
        for i, x in dg.iterrows():
            if x.k == 0:
                ax.annotate(f'HSPD: {100 * x.hspd_frac_recovered:.0f}% of\ndirections\nrecovered', (i - 0.16, 1e-6),
                            ha='center', fontsize=6.5, color=INK2)
        ax.set_yscale('log')
        ax.set_ylim(1e-10, 3)
        ax.set_xlim(-0.6, len(dg) - 0.4)
        ax.set_xticks(xs); ax.set_xticklabels([f'$r={int(r)}$' for r in dg.u_r])
        ax.set_ylabel('Relative logit RMSE')
        ax.set_title('Digits, tanh teachers ($d=64$, $m=10$)', loc='left')
        ax.legend(loc='upper center', fontsize=6.3, ncol=2, bbox_to_anchor=(0.45, -0.12))
        fig.tight_layout()
        savefig(fig, 'digits_distillation')

(SUM / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
print('artifacts written:', sorted(p.name for p in TAB.glob('*.tex')), sorted(p.name for p in FIG.glob('*.pdf')))
