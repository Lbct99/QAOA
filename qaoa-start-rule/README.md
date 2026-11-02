# qaoa-start-rule

Constant starting parameters for p=2 QAOA on small portfolio problems, plus the budgeted optimization
procedure that goes with them. **A rule, not a trained network.** There are no learned weights: the whole
artifact is four constant vectors and one scale formula.

## The rule

Given the problem's Ising coefficients `H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j` (no constant):

```
scale = sqrt( sum_i h_i^2 + sum_{i<j} J_ij^2 )
start = [ gamma_1, gamma_2, beta_1, beta_2 ] = [ c_1/scale, c_2/scale, c_3, c_4 ]
```

| start | c_1 (gamma_1·scale) | c_2 (gamma_2·scale) | c_3 (beta_1) | c_4 (beta_2) |
|---|---|---|---|---|
| 1 (the single-constant rule) | -0.742 | -1.122 | 0.659 | 0.396 |
| 2 | -1.206 | 2.532 | 0.614 | -0.311 |
| 3 | 2.113 | -2.117 | 1.569 | 0.785 |
| 4 | -3.507 | 3.507 | 1.571 | -0.785 |

Parameter order is `[gamma_1..gamma_p, beta_1..beta_p]` (Qiskit orders its own parameters beta-first, so
always bind by name: `bind_parameters`). Procedure (`optimize`): run POWELL for at most 25 evaluations from
each of the k starts, keep the best end point, polish it. Deterministic.

## Evidence (from the study; 600 six-asset problems, 120 later-window test problems)

Share of test problems ending within 1% of the energy span of the best-known energy, and median total
energy evaluations:

| procedure | within 1% | median evaluations |
|---|---|---|
| pipeline: ramp + 3 random restarts, each to convergence | 85% | 1234 |
| pipeline: ramp + 3 scale-aware random restarts | 86% | 1110 |
| this rule, k=1 | 83% | 157 |
| this rule, k=2 | 88% | 182 |
| this rule, k=3 | 92% | 206 |
| this rule, k=4 | 96% | 230 |

k=3 and k=4 beat the pipeline's multi-start (k=4: +10 to +11 points, 95% intervals excluding 0, with about 5x
fewer evaluations); k=1 is statistically indistinguishable from it at about 7 to 8x fewer evaluations. A learned
model (ridge regression, nearest neighbours) did not beat the single constant.

## Limits: what is NOT established

- Validated only for **p=2, 6 assets, budget 3** (the code warns for other sizes). Other settings need new constants.
- One universe (ten large-cap stocks) and one price history; the test problems overlap heavily in time.
- Validated with **exact, noise-free energies**. With a noisy objective (shot noise, hardware in the loop)
  POWELL behaves differently; nothing here has been tested for that.
- The saving is in energy evaluations. It changed nothing about the circuit sent to hardware for the one job
  checked: the pipeline's parameters were already optimal there.
- Evaluations were counted in a numpy simulator; labels are best-known optima, not proven global.
- Starts 3 and 4 sit at special angles (beta_1 ≈ pi/2, beta_2 = ±pi/4); what those problems have in common is
  not understood.

## Running it (Windows PowerShell)

Python only loads the package's files correctly as a MODULE, so run from the folder that contains the
`qaoa_start_rule` folder and never `cd` into `qaoa_start_rule` itself:

```
cd qaoa-start-rule                                   # the project folder (it holds pyproject.toml)
pip install -e ".[test,onnx,qiskit]"                 # once; afterwards everything works from any folder
python -m qaoa_start_rule.onnx_export --out qaoa_start_rule.onnx       # or: qaoa-start-rule-onnx --out ...
python tests\test_start_rule.py
python scripts\validate_on_dataset.py --data ..\qaoa_instances.npz
```

Without installing, `python scripts\export_onnx.py`, `python tests\test_start_rule.py` and
`python scripts\validate_on_dataset.py` also work from the project folder. Do not keep a second copy of the
`qaoa_start_rule` folder elsewhere (for example directly in a parent folder): the copy in the current folder
wins over the installed one and you can end up running stale code.

## Troubleshooting

- `ModuleNotFoundError: No module named 'qaoa_start_rule'` after `pip install -e`: the install ran in a folder that
  does not contain the `qaoa_start_rule` folder. Install with an absolute path to the project folder (the one with
  `pyproject.toml` AND the `qaoa_start_rule` folder), for example
  `pip install -e "C:\Users\you\QAOA\qaoa-start-rule[test,onnx,qiskit]"`, and check with
  `python -c "import qaoa_start_rule; print(qaoa_start_rule.__file__)"`.
- `Expand-Archive ... does not exist`: the zip is not where the command looks. Find it with
  `Get-ChildItem "$HOME\Downloads" -Filter "qaoa-start-rule*.zip"`.
- `python onnx_export.py` fails with an import error: run it as a module instead
  (`python -m qaoa_start_rule.onnx_export`), from the folder that contains `qaoa_start_rule`.

## Install and use

```
pip install -e .                 # from this folder;  add [onnx], [qiskit] or [test] extras as needed
```

```python
import numpy as np
from qaoa_start_rule import suggest_starts, optimize, QAOASim

starts = suggest_starts(h, J)                    # (4, 4) raw [g1, g2, b1, b2]; h: (n,), J: (n, n) upper-triangular
sim = QAOASim(h, J, 2)                           # exact simulator, fine up to roughly 14 qubits
result = optimize(sim.energy, h, J)              # result.params, .energy, .evaluations, .chosen_start
```

With Qiskit (tested with Aer's exact estimator, see `tests/`):

```python
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer.primitives import EstimatorV2
from qaoa_start_rule import bind_parameters

ansatz = QAOAAnsatz(cost_operator=H, reps=2).decompose(reps=2)      # H = SparsePauliOp of h and J, no constant
est = EstimatorV2()
energy = lambda p: float(est.run([(bind_parameters(ansatz, p), H)]).result()[0].data.evs)
result = optimize(energy, h, J)
```

Capping the polish (`polish_max_evals=...`) saves more evaluations (POWELL spends roughly 80 evaluations just
confirming convergence) but was not part of the validated procedure.

## ONNX (optional, and honest about it)

```
python -m qaoa_start_rule.onnx_export --out qaoa_start_rule.onnx
```

The graph maps `h [n]`, `J [n,n]` to the `[k, 4]` starts (float32, dynamic `n`) and is verified against the
numpy rule with onnxruntime. It covers **only the arithmetic**. The optimization loop calls the quantum backend
and cannot be part of a graph. For a function this small, plain Python is simpler; use ONNX only if you want
the same artifact format and runtime as other served models.

## Deployment sketch (not deployed or tested on Azure)

The starts function is static arithmetic and needs no GPU or model registry. The part with real cost is the
loop that calls the energy evaluator, so the natural home for `optimize` is the service that already owns the
Qiskit Runtime or simulator calls (for example an Azure Function or container app). A managed ML endpoint
hosting the ONNX file would add infrastructure for four constants. Ship the package version
(`qaoa_start_rule.__version__`) and `RULE_VERSION` with every result so a regenerated set of constants stays traceable.

## Tests and validation

```
python tests/test_start_rule.py                                  # or: pytest
python scripts/validate_on_dataset.py --data qaoa_instances.npz  # reproduces the study's k=4 numbers
```

## Regenerating the constants

```
python qaoa_instance_generator.py --n 600 --random-starts 20     # dataset (took 73 minutes on the original machine)
python qaoa_start_portfolio.py                                   # greedy selection; prints the starts
```

Paste the printed starts into `CONSTANT_STARTS` in `qaoa_start_rule/rule.py`, bump `RULE_VERSION`, re-run the
tests and the validation script, and update the table above.

## Provenance

Constants derived October 2026 from the 600-instance dataset (seed 0, 20 random starts per instance, p=2),
selection on the training and validation windows only (windows ending at least 180 days before 2024-08-19), greedy cover with
25 evaluations per start and a 1% tolerance. Rule version 0.1.0 (the constants); package version 0.1.2.
