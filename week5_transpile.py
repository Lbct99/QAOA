"""
Week 4a -- Transpile the 6-stock QAOA circuit for a real IBM device.

Warm-starts from the simulator-optimized parameters (no re-optimization on
hardware -- see the module docstring in week5_submit_job.py for why) and
transpiles for a realistic IBM device layout. Since this sandbox has no
internet access to IBM Cloud, this uses FakeSherbrooke, a local, offline
snapshot of a real 127-qubit IBM device's connectivity and basis gates --
the exact same transpilation code works against a live backend, just point
BACKEND at a real one once you're running this with your own IBM account
(see week5_submit_job.py).

Run:
    python week5_transpile.py
"""

import numpy as np

from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

from week3_qaoa_circuit import cost_hamiltonian
from week3b_reps_sweep import best_over_restarts

FINAL_REPS = 2     # deliberately lower than the "best" simulator REPS: depth scales ~linearly with
                    # REPS (see the transpiled-depth check) while P(optimal) gains were small past
                    # reps=2 given the near-degenerate landscape -- not worth the extra noise exposure
N_RESTARTS_FINAL = 3


def main():
    d = np.load("hw6_pipeline.npz")
    h, J, tickers = d["h"], d["J"], [str(t) for t in d["tickers"]]
    n = len(tickers)
    H = cost_hamiltonian(h, J, n)

    print(f"Re-optimizing at reps={FINAL_REPS} with {N_RESTARTS_FINAL} restarts "
          f"to get the final warm-start parameters (this is the LAST simulator "
          f"optimization -- hardware will only sample, not re-optimize)...")
    rng = np.random.default_rng(0)
    orig_n_restarts = None
    import week3b_reps_sweep as sw
    orig_n_restarts, sw.N_RESTARTS = sw.N_RESTARTS, N_RESTARTS_FINAL
    ansatz, res, label = best_over_restarts(H, n, FINAL_REPS, rng)
    sw.N_RESTARTS = orig_n_restarts
    print(f"Final energy: {res.fun:.4f} (best start: {label})")

    qc = ansatz.assign_parameters(res.x)
    qc.measure_all()
    print(f"\nLogical circuit: {qc.depth()} depth, {qc.count_ops()}")

    backend = FakeSherbrooke()
    print(f"\nTranspiling for {backend.name} ({backend.num_qubits} qubits, "
          f"{len(list(backend.coupling_map.get_edges()))} coupling edges)...")
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=42)
    transpiled = pm.run(qc)

    ops = transpiled.count_ops()
    two_q = ops.get("cx", 0) + ops.get("cz", 0) + ops.get("ecr", 0)
    print(f"\nTranspiled circuit: depth {transpiled.depth()}, {ops}")
    print(f"Two-qubit gate count: {two_q}  "
          f"(started from {n*(n-1)//2} logical ZZ interactions -- "
          f"the excess above that is SWAP-routing overhead from limited hardware connectivity)")
    print(f"Physical qubits used: {transpiled.num_qubits} "
          f"(only {n} are 'yours' -- the rest exist on the chip but are unused)")

    transpiled.qpy_file = "hw6_transpiled_circuit.qpy"
    from qiskit import qpy
    with open("hw6_transpiled_circuit.qpy", "wb") as f:
        qpy.dump(transpiled, f)
    np.save("hw6_final_params.npy", res.x)
    print(f"\nSaved transpiled circuit to hw6_transpiled_circuit.qpy "
          f"and optimized parameters to hw6_final_params.npy")
    print("Next: week5_submit_job.py runs this exact circuit on real IBM hardware.")


if __name__ == "__main__":
    main()
