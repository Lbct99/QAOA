"""
Week 2-3 -- Building the QAOA circuit.

Two parts:

  PART A -- p=1, built by hand, so the circuit is not a black box.
      U_C(gamma) = exp(-i*gamma*H_C):
          Z_i term (h_i)      -> Rz(2*gamma*h_i) on qubit i
          Z_i Z_j term (J_ij) -> CNOT(i,j), Rz(2*gamma*J_ij) on j, CNOT(i,j)
      U_M(beta) = exp(-i*beta*sum_i X_i):
          Rx(2*beta) on every qubit
      Circuit: H^n -> U_C(gamma) -> U_M(beta) -> measure
      We scan (gamma, beta) on a grid and plot <H_C> exactly (statevector,
      no shot noise) -- this is the "landscape" the optimizer has to search.

  PART B -- general p (REPS layers), using Qiskit's QAOAAnsatz so the same
      cost Hamiltonian scales to more layers without hand-building each one.
      Optimized with POWELL, using a linear-ramp initialization for
      (gamma_1..gamma_p, beta_1..beta_p) -- an adiabatic-inspired schedule
      that tends to work better than random initialization.

Run:
    python week3_qaoa_circuit.py
Produces:
    qaoa_p1_landscape.png
    qaoa_week3_result.npz
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2

REPS = 2          # number of QAOA layers (p) for the full optimization in Part B
SHOTS = 4096       # shots for the sampled final distribution
LANDSCAPE_POINTS = 40   # grid resolution per axis for the p=1 scan (40x40 = 1600 evals)


# ---------------------------------------------------------------------------
# Shared: rebuild the cost Hamiltonian from Week 2's saved h, J, C0
# ---------------------------------------------------------------------------
def load_ising():
    d = np.load("ising_week2.npz")
    return d["h"], d["J"], float(d["C0"]), [str(t) for t in d["tickers"]]


def cost_hamiltonian(h, J, n):
    """SparsePauliOp for sum h_i Z_i + sum_{i<j} J_ij Z_i Z_j (no constant -- Estimator adds it back)."""
    terms, coeffs = [], []
    for i in range(n):
        if h[i] != 0:
            label = ["I"] * n
            label[n - 1 - i] = "Z"
            terms.append("".join(label)); coeffs.append(h[i])
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0:
                label = ["I"] * n
                label[n - 1 - i] = "Z"; label[n - 1 - j] = "Z"
                terms.append("".join(label)); coeffs.append(J[i, j])
    return SparsePauliOp(terms, coeffs)


# ---------------------------------------------------------------------------
# PART A -- hand-built p=1 circuit and landscape
# ---------------------------------------------------------------------------
def qaoa_p1_circuit(gamma, beta, h, J, n):
    qc = QuantumCircuit(n)
    qc.h(range(n))                                    # uniform superposition

    for i in range(n):                                 # cost unitary: Z_i terms
        if h[i] != 0:
            qc.rz(2 * gamma * h[i], i)
    for i in range(n):                                  # cost unitary: Z_i Z_j terms
        for j in range(i + 1, n):
            if J[i, j] != 0:
                qc.cx(i, j)
                qc.rz(2 * gamma * J[i, j], j)
                qc.cx(i, j)

    for i in range(n):                                  # mixer unitary
        qc.rx(2 * beta, i)
    return qc


def precompute_basis_energies(h, J, C0, n):
    """H_C(bitstring) for every computational basis state, vectorized (used every grid point)."""
    idx = np.arange(2**n)[:, None]
    bits = (idx >> np.arange(n)) & 1                 # (2^n, n), bits[:, k] = qubit k
    z = 1 - 2 * bits.astype(float)
    vals = z @ h + np.einsum("ki,ij,kj->k", z, np.triu(J, k=1) + np.triu(J, k=1).T, z) / 2
    return C0 + vals


def landscape_scan(h, J, C0, n, points):
    basis_energy = precompute_basis_energies(h, J, C0, n)
    gammas = np.linspace(0, 2 * np.pi, points)
    betas = np.linspace(0, np.pi, points)
    energy = np.zeros((points, points))
    for ig, g in enumerate(gammas):
        for ib, b in enumerate(betas):
            qc = qaoa_p1_circuit(g, b, h, J, n)
            probs = Statevector(qc).probabilities()
            energy[ib, ig] = probs @ basis_energy
    return gammas, betas, energy


def plot_landscape(gammas, betas, energy, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(gammas, betas, energy, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="<H_C> (energy)")
    ax.set_xlabel("gamma")
    ax.set_ylabel("beta")
    ax.set_title("p=1 QAOA cost landscape")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved landscape plot to {path}")


# ---------------------------------------------------------------------------
# PART B -- general p (REPS), QAOAAnsatz + EstimatorV2, POWELL optimizer
# ---------------------------------------------------------------------------
def linear_ramp_init(reps):
    """Adiabatic-inspired: gamma ramps up, beta ramps down across layers."""
    gammas = np.linspace(1.0 / reps, 1.0, reps) * 0.5
    betas = np.linspace(1.0, 1.0 / reps, reps) * 0.5
    # QAOAAnsatz parameter order is [beta_0..beta_{p-1}, gamma_0..gamma_{p-1}] or vice versa
    # depending on version -- we set by name below instead of relying on order.
    return gammas, betas


def run_optimization(H, n, reps):
    ansatz = QAOAAnsatz(cost_operator=H, reps=reps)
    ansatz = ansatz.decompose(reps=2)  # expand to basis gates so the estimator can run it

    estimator = EstimatorV2()

    gammas0, betas0 = linear_ramp_init(reps)
    x0 = np.zeros(ansatz.num_parameters)
    param_names = [p.name for p in ansatz.parameters]
    for k, name in enumerate(param_names):
        if "gamma" in name.lower() or name.startswith("γ"):
            layer = int("".join(ch for ch in name if ch.isdigit()) or 0)
            x0[k] = gammas0[min(layer, reps - 1)]
        else:
            layer = int("".join(ch for ch in name if ch.isdigit()) or 0)
            x0[k] = betas0[min(layer, reps - 1)]

    history = []

    def objective(params):
        job = estimator.run([(ansatz, H, params)])
        val = job.result()[0].data.evs
        history.append(float(val))
        return float(val)

    res = minimize(objective, x0, method="POWELL")
    print(f"POWELL finished: {len(history)} evaluations, final energy {res.fun:.4f}")
    return ansatz, res, history


def sample_final_circuit(ansatz, params, n, shots, seed=42):
    """seed fixes the shot sampling so counts are reproducible run to run.
    The optimized energy above is already deterministic (exact statevector
    estimator, fixed initialization) -- this only pins the sampling noise."""
    qc = ansatz.assign_parameters(params)
    qc.measure_all()
    backend = AerSimulator(seed_simulator=seed)
    job = backend.run(qc, shots=shots, seed_simulator=seed)
    counts = job.result().get_counts()
    return counts


def main():
    h, J, C0, tickers = load_ising()
    n = len(tickers)
    H = cost_hamiltonian(h, J, n)

    print("=== Part A: p=1 landscape ===")
    gammas, betas, energy = landscape_scan(h, J, C0, n, LANDSCAPE_POINTS)
    plot_landscape(gammas, betas, energy, "qaoa_p1_landscape.png")
    ib, ig = np.unravel_index(np.argmin(energy), energy.shape)
    print(f"Best on grid: gamma={gammas[ig]:.3f}, beta={betas[ib]:.3f}, "
          f"energy={energy[ib, ig]:.4f}  (grid resolution only -- not a real optimum)")

    print(f"\n=== Part B: p={REPS} optimization (QAOAAnsatz + EstimatorV2 + POWELL) ===")
    ansatz, res, history = run_optimization(H, n, REPS)
    print(f"Optimized energy (no constant): {res.fun:.4f}  "
          f"| with C0: {res.fun + C0:.4f}")

    counts = sample_final_circuit(ansatz, res.x, n, SHOTS)
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    print(f"\nTop 5 measured bitstrings out of {SHOTS} shots:")
    for bitstring, count in top:
        # Qiskit bitstrings are ordered qubit n-1 ... qubit 0 (left to right)
        x = [int(b) for b in bitstring[::-1]]
        selected = [t for t, xi in zip(tickers, x) if xi == 1]
        feasible = len(selected) == 3
        print(f"  {bitstring}  count={count:4d} ({100*count/SHOTS:.1f}%)  "
              f"selected={selected}  feasible={feasible}")

    np.savez("qaoa_week3_result.npz", opt_params=res.x, energy=res.fun,
              history=history, tickers=tickers)
    print("\nSaved optimized parameters and history to qaoa_week3_result.npz")


if __name__ == "__main__":
    main()
