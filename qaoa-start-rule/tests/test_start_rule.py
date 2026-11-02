"""Tests for qaoa_start_rule. Run with pytest, or directly:  python tests/test_start_rule.py

Fast, no dataset needed. Tests that need qiskit, qiskit-aer or onnx/onnxruntime are skipped if those
packages are missing.
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # works without `pip install`

from qaoa_start_rule import (CONSTANT_STARTS, QAOASim, bind_parameters, energy_scale, optimize, suggest_starts)


class Skip(Exception):
    pass


def _skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        raise Skip(msg)


def _instance(seed=0, n=6):
    rng = np.random.default_rng(seed)
    return rng.normal(-0.1, 0.06, n), np.triu(rng.normal(0.03, 0.02, (n, n)), 1)


def test_scale_formula():
    h, J = _instance()
    expected = np.sqrt(np.sum(h**2) + sum(J[i, j] ** 2 for i in range(6) for j in range(i + 1, 6)))
    assert abs(energy_scale(h, J) - expected) < 1e-12
    # a full symmetric matrix gives the same scale (only the strict upper triangle is read)
    assert abs(energy_scale(h, J + J.T) - expected) < 1e-12


def test_starts_shape_and_values():
    h, J = _instance()
    s = suggest_starts(h, J)
    assert s.shape == (4, 4)
    sc = energy_scale(h, J)
    assert np.allclose(s[:, :2] * sc, CONSTANT_STARTS[:, :2])
    assert np.allclose(s[:, 2:], CONSTANT_STARTS[:, 2:])
    assert suggest_starts(h, J, k=1).shape == (1, 4)


def test_scale_invariance():
    h, J = _instance()
    a, b = suggest_starts(h, J), suggest_starts(3.0 * h, 3.0 * J)
    assert np.allclose(b[:, :2], a[:, :2] / 3.0)       # gamma shrinks with the energy scale
    assert np.allclose(b[:, 2:], a[:, 2:])             # beta does not


def test_input_validation():
    h, J = _instance()
    for bad in (lambda: suggest_starts(h, J[:5, :5]), lambda: suggest_starts(h[None, :], J),
                lambda: suggest_starts(np.zeros(6), np.zeros((6, 6))), lambda: suggest_starts(h, J, k=5)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected a ValueError")


def test_unvalidated_size_warns():
    import qaoa_start_rule.rule as rule
    rule._warned.discard("n")
    h, J = _instance(n=4)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        suggest_starts(h, J)
    assert any("untested" in str(x.message) for x in w)


def test_permutation_invariance():
    h, J = _instance(seed=3)
    perm = np.random.default_rng(1).permutation(6)
    Jfull = J + J.T
    hp, Jp = h[perm], np.triu(Jfull[np.ix_(perm, perm)], 1)
    assert np.allclose(suggest_starts(h, J), suggest_starts(hp, Jp))
    r1 = optimize(QAOASim(h, J, 2).energy, h, J)
    r2 = optimize(QAOASim(hp, Jp, 2).energy, hp, Jp)
    assert abs(r1.energy - r2.energy) < 1e-6           # relabelling the assets cannot change the optimum found


def test_optimize_behaviour():
    h, J = _instance(seed=5)
    sim = QAOASim(h, J, 2)
    a, b = optimize(sim.energy, h, J), optimize(sim.energy, h, J)
    assert a.evaluations == b.evaluations and np.array_equal(a.params, b.params)   # deterministic
    assert a.evaluations == a.evaluations_first_stage + a.evaluations_polish
    assert a.evaluations_first_stage <= 4 * 25
    assert abs(sim.energy(a.params) - a.energy) < 1e-12
    assert a.energy <= min(a.first_stage_energies) + 1e-12                          # polish never makes it worse
    assert a.energy < 0                                                              # beats the uniform state (energy 0)


def test_polish_cap_and_single_start():
    h, J = _instance(seed=6)
    sim = QAOASim(h, J, 2)
    capped = optimize(sim.energy, h, J, polish_max_evals=10)
    assert capped.evaluations_polish <= 10
    nopolish = optimize(sim.energy, h, J, polish=False)
    assert nopolish.evaluations_polish == 0
    one = optimize(sim.energy, h, J, k=1)
    assert one.evaluations_first_stage <= 25 and len(one.first_stage_energies) == 1


def test_qiskit_crosscheck_and_end_to_end():
    try:
        from qiskit.circuit.library import QAOAAnsatz
        from qiskit.quantum_info import SparsePauliOp, Statevector
        from qiskit_aer.primitives import EstimatorV2
    except ImportError:
        _skip("qiskit / qiskit-aer not installed")
    h, J = _instance(seed=8)
    n = len(h)
    terms, coeffs = [], []
    for i in range(n):
        lab = ["I"] * n; lab[n - 1 - i] = "Z"; terms.append("".join(lab)); coeffs.append(h[i])
        for j in range(i + 1, n):
            lab = ["I"] * n; lab[n - 1 - i] = "Z"; lab[n - 1 - j] = "Z"; terms.append("".join(lab)); coeffs.append(J[i, j])
    H = SparsePauliOp(terms, coeffs)
    ansatz = QAOAAnsatz(cost_operator=H, reps=2).decompose(reps=2)
    sim = QAOASim(h, J, 2)
    x = suggest_starts(h, J)[2]
    e_q = float(Statevector(bind_parameters(ansatz, x)).expectation_value(H).real)
    assert abs(e_q - sim.energy(x)) < 1e-9
    # the whole procedure through a Qiskit estimator (exact expectation values) matches the numpy simulator
    est = EstimatorV2()
    qiskit_energy = lambda p: float(est.run([(bind_parameters(ansatz, p), H)]).result()[0].data.evs)
    r_np = optimize(sim.energy, h, J, k=2)
    r_q = optimize(qiskit_energy, h, J, k=2)
    assert abs(r_np.energy - r_q.energy) < 1e-6 and abs(r_np.evaluations - r_q.evaluations) <= 3


def test_onnx_matches_numpy(tmp_path=None):
    try:
        import onnx, onnxruntime  # noqa: F401
    except ImportError:
        _skip("onnx / onnxruntime not installed")
    import os
    import tempfile
    from qaoa_start_rule.onnx_export import export, verify
    path = os.path.join(str(tmp_path) if tmp_path else tempfile.mkdtemp(), "rule.onnx")
    export(path)
    assert verify(path) < 1e-5


if __name__ == "__main__":
    passed = skipped = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
                passed += 1
            except Skip as s:
                print(f"SKIP  {name}: {s}")
                skipped += 1
    print(f"\n{passed} passed, {skipped} skipped")
