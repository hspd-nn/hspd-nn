# HSPD: Hessian Subspace Power Distillation

HSPD recovers the hidden units of a trained shallow neural network from its input Hessians.
For a network `f(x) = c + sum_j a_j phi_j(w_j^T x + b_j)` with `m` outputs, every Hessian of every
output is a linear combination of the rank-one matrices `w_j w_j^T`. HSPD

1. stacks the symmetric-vectorized Hessians at `N` probes into a matrix `M` and extracts its rank-`r`
   column space (the Hessian subspace);
2. searches that subspace for rank-one matrices with a batched subspace power iteration from
   `ceil(5 r ln r)` random starts, refines the candidates by safeguarded Riemannian Newton steps, and
   selects and certifies the distinct candidates by their scores;
3. reconstructs the remaining parameters from the Hessian coefficients: by least squares for cubic
   networks, and by fitting the curvature profile of each unit for tanh networks.

The directions are generically identifiable up to `r = d(d-1)/2` hidden units (`d` inputs) when the
outputs and probes excite every unit. The package also computes, for each recovered direction, the
conditioning number `kappa_j` that governs the local convergence rate of the search and its
sensitivity to Hessian errors. Hessians can be computed exactly (white box) or by central finite
differences from function queries (black box, `2 d^2 + 1` queries per probe).

## Installation

Python 3.10 to 3.13.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The package depends on NumPy, SciPy, pandas, matplotlib, scikit-learn and PyTorch (CPU is enough;
PyTorch is used only to train the teachers of the real-data experiments).

## Example

```python
import numpy as np
from hspd.pnn import random_tanh_net, tanh_hessian_matrix, tanh_forward, recover_tanh_parameters
from hspd.hspd import hspd_recover
from hspd.metrics import direction_summary, relative_rmse

d, r, m, N = 8, 20, 4, 64                          # 20 hidden units > d: overcomplete
W, A, b, c = random_tanh_net(d, r, m, seed=0)      # unit hidden directions in the columns of W
X = np.random.default_rng(1).normal(size=(N, d))   # probes
M, C = tanh_hessian_matrix(X, W, A, b)             # stacked Hessians, p x (m N)

res = hspd_recover(M, d, r, seed=2, max_iter=300, polish='newton')
summary, cos, _ = direction_summary(W, res['W'], res['certified'])
print(summary['success'], summary['n_certified'], round(cos.min(), 6))   # True 20 1.0

Ws, A2, b2, c2, _ = recover_tanh_parameters(M, X, res['W'], tanh_forward(X, W, A, b, c))
Xt = np.random.default_rng(3).normal(size=(500, d))
print(relative_rmse(np.tanh(Xt @ Ws + b2) @ A2.T + c2, tanh_forward(Xt, W, A, b, c)))  # about 1e-8
```

Black-box Hessians: `hspd.finite_difference.finite_difference_hessians(f, X, m, h, noise_std)`, where
`f` maps an `(n, d)` array of inputs to an `(n, m)` array of outputs.

## Repository layout

| path | contents |
|---|---|
| `src/hspd/` | the method (`hspd.py`), baselines, networks and reconstruction (`pnn.py`), finite differences, metrics, trained teachers (`teachers.py`) |
| `tests/` | unit tests, including the curvature, convergence-rate and monotone-shift results |
| `scripts/run_campaign.py` | every experiment, as named campaigns of independent units |
| `scripts/aggregate_results.py` | collects run records into `results/csv/` |
| `scripts/make_artifacts.py` | builds all tables and figures into `paper/tables/` and `paper/figures/` |
| `scripts/diagnostics/digits_wide.py` | diagnostic of the widest Digits teacher: local versus global failure, budget study |
| `results/runs/` | the records of every run used in the paper (`config.json`, `env.json`, `metrics.json`) |
| `results/csv/` | the aggregated records |
| `results/diagnostics/` | output of the diagnostic (`python scripts/diagnostics/digits_wide.py 1 --out results/diagnostics/digits_wide_seed1.json`, about one hour on 32 cores) |
| `paper/` | the generated tables (LaTeX) and figures (PDF and PNG) |

## Rebuilding the tables and figures

From the shipped run records:

```bash
python scripts/aggregate_results.py      # results/runs -> results/csv
python scripts/make_artifacts.py         # results/csv  -> paper/tables, paper/figures
```

| campaign | tables and figures it feeds |
|---|---|
| `width8` | width sweep for `d = 8`: success of all solvers, recovered fraction, runtimes, clustering-threshold sensitivity, basins of attraction |
| `multid` | dimension sweep (`d` = 6, 10, 16, 24) against the relative width, conditioning near the bound, runtimes |
| `shift` | effect of the shift `gamma` near the bound |
| `noise` | robustness to Hessian perturbations; first-order sensitivity of each direction |
| `context` | rank of the output/probe coefficient matrix for cubic and tanh networks |
| `cubic_full` | full cubic reconstruction; weak units with small output weights |
| `tanh` | direction recovery and reconstruction of synthetic tanh networks |
| `blackbox` | finite-difference Hessians with noisy queries: error versus step and success |
| `scaling` | runtime and memory up to `d = 64` |
| `realdata` | trained teachers on Wine and Digits, genericity diagnostics, comparison with gradient-based distillation |
| `spm_check` | control experiment for the deflation of the subspace power method |

## Rerunning the experiments

Each campaign is a list of independent units, assigned round-robin to `--n-tasks` tasks; a task writes
`out/runs/<campaign>/task-XXX/metrics.json` after every unit and skips finished units when restarted.

Locally (one task):

```bash
python scripts/run_campaign.py --campaign width8 --n-tasks 1 --task-id 0 --threads 4
```

On a Slurm cluster, as an array job (the task id is read from `SLURM_ARRAY_TASK_ID`, the thread count
from `SLURM_CPUS_PER_TASK`):

```bash
sbatch --array=0-63 -c 8 --mem=14G --time=16:00:00 \
  --wrap "python scripts/run_campaign.py --campaign multid --n-tasks 64"
```

Then point the scripts at the new records instead of the shipped ones:

```bash
HSPD_RUNS=out/runs python scripts/aggregate_results.py
python scripts/make_artifacts.py
```

The paper's runs used compute nodes with two 96-core AMD EPYC 9655 processors and took about 120
core-hours in total, 85 of them for the dimension sweep (`multid`); every other campaign takes less
than 20 core-hours, most of them minutes. Thread counts per campaign: 4, except `multid` and `shift`
(8), `realdata` (16) and `scaling` (32).

## Data

The real-data experiments use the Wine and the optical recognition of handwritten Digits datasets as
distributed with scikit-learn (`sklearn.datasets.load_wine`, `load_digits`); no download is needed.
Both originate from the UCI Machine Learning Repository and are licensed under CC BY 4.0. All other
experiments use synthetic networks generated from the seeds recorded in the run records.

## License

MIT; see `LICENSE`.
