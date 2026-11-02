"""The budgeted optimization procedure that goes with the constant starts.

    1. run POWELL for at most `budget_per_start` evaluations from each of the k starts
    2. keep the best end point
    3. polish it with POWELL (to convergence, or up to `polish_max_evals`)

`energy` is ANY callable params -> float (the numpy simulator, a Qiskit Estimator wrapper, ...), so the
package never touches a quantum backend itself. Deterministic: no randomness anywhere.

Validated only with exact, noise-free energies. With a noisy objective (sampling noise, hardware in the
loop) POWELL behaves differently and nothing here has been tested for that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .rule import suggest_starts


class _Stop(Exception):
    pass


def _run(energy, x0, max_evals=None):
    """POWELL from x0, stopping after `max_evals` evaluations if given. Returns (best x, best energy, evaluations)."""
    st = {"n": 0, "e": np.inf, "x": np.array(x0, dtype=float)}

    def f(x):
        if max_evals is not None and st["n"] >= max_evals:
            raise _Stop
        v = float(energy(np.array(x, dtype=float)))
        st["n"] += 1
        if v < st["e"]:
            st["e"], st["x"] = v, np.array(x, dtype=float)
        return v

    try:
        minimize(f, x0, method="Powell")
    except _Stop:
        pass
    return st["x"], st["e"], st["n"]


@dataclass
class StartRuleResult:
    params: np.ndarray                 # best parameters [gamma_1, gamma_2, beta_1, beta_2]
    energy: float                      # energy at `params`
    evaluations: int                   # total energy evaluations (first stage + polish)
    evaluations_first_stage: int
    evaluations_polish: int
    chosen_start: int                  # index (0-based) of the start whose end point was polished
    first_stage_energies: list = field(default_factory=list)


def optimize(energy, h, J, k: int = 4, budget_per_start: int = 25, polish: bool = True,
             polish_max_evals: int | None = None) -> StartRuleResult:
    """Run the rule. k=4 is the validated default; k=1 is the single-constant rule."""
    starts = suggest_starts(h, J, k)
    shorts = [_run(energy, x0, budget_per_start) for x0 in starts]
    first = sum(s[2] for s in shorts)
    pick = int(np.argmin([s[1] for s in shorts]))
    x, e = shorts[pick][0], shorts[pick][1]
    n_pol = 0
    if polish:
        x_p, e_p, n_pol = _run(energy, x, polish_max_evals)
        if e_p < e:
            x, e = x_p, e_p
    return StartRuleResult(params=x, energy=float(e), evaluations=first + n_pol, evaluations_first_stage=first,
                           evaluations_polish=n_pol, chosen_start=pick,
                           first_stage_energies=[float(s[1]) for s in shorts])
