"""The constant-start rule: from the Ising coefficients of a problem to starting QAOA parameters.

A RULE, not a trained network. The only "model" is four constant vectors in normalized units and a scale:

    scale  = sqrt( sum_i h_i^2  +  sum_{i<j} J_ij^2 )          (energy scale of the problem)
    start  = [ gamma_1, gamma_2, beta_1, beta_2 ]
           = [ c_1 / scale, c_2 / scale, c_3, c_4 ]            (c = one row of CONSTANT_STARTS)

Parameter order everywhere: [gamma_1 .. gamma_p, beta_1 .. beta_p], for the QAOA circuit
U = prod_k exp(-i beta_k sum_i X_i) exp(-i gamma_k H_C) applied to |+>^n, with H_C = sum_i h_i Z_i +
sum_{i<j} J_ij Z_i Z_j (no constant), exactly as in the pipeline's QAOAAnsatz / cost_hamiltonian.

Provenance of the constants: greedy selection of 4 starts on the training+validation windows of the
600-instance dataset (qaoa_instances.npz: 6-asset portfolio problems, budget 3, p=2, generated with seed 0
and 20 random starts), where each candidate gets 25 POWELL evaluations per instance and a start is
"covering" an instance if it ends within 1% of the energy span of the best-known energy. See README.md.
"""

from __future__ import annotations

import warnings

import numpy as np

RULE_VERSION = "0.1.0"
LAYERS = 2

# Rows = starts in order of greedy selection (start 1 alone is the single-constant rule).
# Columns = (gamma_1 * scale, gamma_2 * scale, beta_1, beta_2).
CONSTANT_STARTS = np.array([
    [-0.742, -1.122, 0.659, 0.396],
    [-1.206, 2.532, 0.614, -0.311],
    [2.113, -2.117, 1.569, 0.785],
    [-3.507, 3.507, 1.571, -0.785],
])

# What the rule was validated for. Anything else is untested.
VALIDATED_ASSETS = 6
VALIDATED_BUDGET = 3

_warned = set()


def _check(h, J):
    h = np.asarray(h, dtype=float)
    J = np.asarray(J, dtype=float)
    if h.ndim != 1:
        raise ValueError("h must be a 1-D array of single-qubit coefficients")
    n = len(h)
    if J.shape != (n, n):
        raise ValueError(f"J must be an ({n}, {n}) array; only the strictly upper triangle (i < j) is used")
    if n != VALIDATED_ASSETS and "n" not in _warned:
        _warned.add("n")
        warnings.warn(f"The rule was validated for {VALIDATED_ASSETS}-asset problems (budget {VALIDATED_BUDGET}, "
                      f"p={LAYERS}); n={n} is untested.", stacklevel=3)
    return h, J


def energy_scale(h, J) -> float:
    """sqrt(sum h_i^2 + sum_{i<j} J_ij^2): the standard deviation of H_C over the uniform superposition."""
    h, J = _check(h, J)
    s = float(np.sqrt(np.sum(h**2) + np.sum(np.triu(J, 1) ** 2)))
    if not s > 0:
        raise ValueError("the problem has zero energy scale (all coefficients are zero)")
    return s


def suggest_starts(h, J, k: int = 4) -> np.ndarray:
    """Return the first `k` starting parameter vectors, shape (k, 4), in raw units [g1, g2, b1, b2]."""
    if not 1 <= k <= len(CONSTANT_STARTS):
        raise ValueError(f"k must be between 1 and {len(CONSTANT_STARTS)}")
    scale = energy_scale(h, J)
    c = CONSTANT_STARTS[:k]
    return np.concatenate([c[:, :LAYERS] / scale, c[:, LAYERS:]], axis=1)


def bind_parameters(ansatz, params):
    """Bind [gamma_1..gamma_p, beta_1..beta_p] into a Qiskit QAOAAnsatz BY NAME (never by position:
    Qiskit orders its parameters as beta[0], beta[1], gamma[0], gamma[1])."""
    params = np.asarray(params, dtype=float)
    p = len(params) // 2
    binding = {}
    for prm in ansatz.parameters:
        digits = "".join(ch for ch in prm.name if ch.isdigit())
        if not digits:
            raise ValueError(f"cannot read the layer index from parameter name {prm.name!r}")
        is_gamma = "\u03b3" in prm.name or "gamma" in prm.name.lower()
        binding[prm] = params[int(digits)] if is_gamma else params[p + int(digits)]
    return ansatz.assign_parameters(binding)
