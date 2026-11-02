"""Exact numpy QAOA simulator for small problems (used for validation and offline optimization).

<H_C> after p QAOA layers from |+>^n. H_C is diagonal; the mixer exp(-i beta sum X) is applied as
W diag(exp(-i beta sum z)) W with W = Hadamard on every qubit. Qubit k is bit k of the basis index, which
matches the pipeline's Qiskit labels (asset i <-> qubit i). Checked against Qiskit's QAOAAnsatz to 1e-9
(see tests/test_start_rule.py). Practical up to roughly 14 qubits (the Hadamard matrix is dense).
"""

from __future__ import annotations

import numpy as np


def all_bits(n):
    return ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(float)


class QAOASim:
    def __init__(self, h, J, layers):
        h, J = np.asarray(h, float), np.asarray(J, float)
        n = len(h)
        self.n, self.p = n, layers
        z = 1.0 - 2.0 * all_bits(n)
        Ju = np.triu(J, 1)
        self.diag = z @ h + 0.5 * np.einsum("ki,ij,kj->k", z, Ju + Ju.T, z)
        self.zsum = z.sum(axis=1)
        H1 = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
        W = np.array([[1.0]])
        for _ in range(n):
            W = np.kron(W, H1)
        self.W = W
        self.psi0 = np.full(2**n, 1.0 / np.sqrt(2.0**n), dtype=complex)

    def probabilities(self, params):
        p = self.p
        psi = self.psi0
        for g, b in zip(params[:p], params[p:]):
            psi = psi * np.exp(-1j * g * self.diag)
            psi = self.W @ (np.exp(-1j * b * self.zsum) * (self.W @ psi))
        return psi.real**2 + psi.imag**2

    def energy(self, params):
        return float(np.sum(self.diag * self.probabilities(params)))
