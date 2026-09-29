"""
Week 2 -- QUBO -> Ising Hamiltonian mapping.

Substituting x_i = (1 - Z_i)/2 into x^T Q x and collecting terms gives:

    x^T Q x  =  C0  +  sum_i h_i Z_i  +  sum_{i<j} J_ij Z_i Z_j

with (Q assumed symmetric, as build_qubo produces):

    h_i  = -0.5 * sum_j Q[i, j]            (half the i-th row sum of Q, negated)
    J_ij =  0.5 * Q[i, j]                   (i < j)
    C0   =  0.5 * trace(Q)  +  0.5 * sum_{i<j} Q[i, j]

This script derives h, J, C0 from qubo_week1.npz, verifies the mapping is
exact by comparing every one of the 2^n classical assignments against the
QUBO objective, then builds the Ising Hamiltonian as a Qiskit SparsePauliOp
ready for QAOAAnsatz in Week 2-3.

Run:
    python week2_ising_mapping.py
"""

import itertools

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def qubo_to_ising(Q):
    """Return h (n,), J (n,n upper-triangular, zero elsewhere), and constant C0."""
    n = Q.shape[0]
    h = -0.5 * Q.sum(axis=1)
    J = np.triu(0.5 * Q, k=1)
    C0 = 0.5 * np.trace(Q) + J.sum()
    return h, J, C0


def ising_energy(z, h, J, C0):
    """z is a +-1 vector. Direct evaluation of C0 + h.z + sum_{i<j} J_ij z_i z_j."""
    return C0 + h @ z + z @ J @ z


def verify_mapping(Q, h, J, C0, n):
    """Check C0 + h.z + z^T J z == x^T Q x for every one of the 2^n bitstrings."""
    max_err = 0.0
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        z = 1 - 2 * x  # x=0 -> z=+1, x=1 -> z=-1
        qubo_val = x @ Q @ x
        ising_val = ising_energy(z, h, J, C0)
        max_err = max(max_err, abs(qubo_val - ising_val))
    return max_err


def build_hamiltonian(h, J, n):
    """SparsePauliOp for C0 + sum h_i Z_i + sum_{i<j} J_ij Z_i Z_j (C0 added separately)."""
    terms, coeffs = [], []
    for i in range(n):
        if h[i] != 0:
            label = ["I"] * n
            label[n - 1 - i] = "Z"  # Qiskit qubit 0 is the rightmost character
            terms.append("".join(label))
            coeffs.append(h[i])
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0:
                label = ["I"] * n
                label[n - 1 - i] = "Z"
                label[n - 1 - j] = "Z"
                terms.append("".join(label))
                coeffs.append(J[i, j])
    return SparsePauliOp(terms, coeffs)


def main():
    d = np.load("qubo_week1.npz")
    Q, tickers = d["Q"], [str(t) for t in d["tickers"]]
    n = len(tickers)

    h, J, C0 = qubo_to_ising(Q)

    max_err = verify_mapping(Q, h, J, C0, n)
    print(f"Mapping verified over all {2**n} bitstrings. Max error: {max_err:.2e}")
    assert max_err < 1e-9, "QUBO -> Ising mapping does not match -- do not proceed"

    print(f"\nLinear terms h_i (single-qubit Z rotations):")
    for t, hi in zip(tickers, h):
        print(f"  {t:>6s}: {hi:+.4f}")

    off_diag = J[np.triu_indices(n, k=1)]
    print(f"\nCoupling terms J_ij (two-qubit ZZ interactions): {len(off_diag)} pairs "
          f"(fully connected -- every asset pair is coupled)")
    print(f"  range: [{off_diag.min():.4f}, {off_diag.max():.4f}], "
          f"mean magnitude: {np.abs(off_diag).mean():.4f}")
    print(f"\nConstant offset C0 = {C0:.4f}  (shifts total energy, irrelevant to the argmin)")

    H = build_hamiltonian(h, J, n)
    print(f"\nSparsePauliOp built: {len(H)} Pauli terms, {n} qubits")

    # Cross-check: ground state of H (via brute-force over Z assignments) should
    # reproduce the same optimal portfolio found in Week 1.
    best_z, best_e = None, np.inf
    for bits in itertools.product([0, 1], repeat=n):
        z = 1 - 2 * np.array(bits, dtype=float)
        e = ising_energy(z, h, J, C0)
        if e < best_e:
            best_e, best_z = e, z
    selected = [t for t, zi in zip(tickers, best_z) if zi < 0]  # z=-1 -> x=1 -> selected
    print(f"\nIsing ground state selects: {selected}  (energy {best_e:.4f})")
    print("This should match the Week 1 classical optimum -- eyeball it against that output.")

    np.savez("ising_week2.npz", h=h, J=J, C0=C0, tickers=tickers)
    print("\nSaved h, J, C0 to ising_week2.npz for the QAOA circuit build.")


if __name__ == "__main__":
    main()
