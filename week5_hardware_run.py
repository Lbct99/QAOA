"""
Week 4b (fixed) -- Transpile AND submit against the SAME backend.

The previous bug: week5_transpile.py transpiled for FakeSherbrooke (gate set:
ecr, on a specific 127-qubit layout), but week5_submit_job.py then picked
whatever backend was least busy (ibm_marrakesh) and submitted the
FakeSherbrooke-shaped circuit there. Different chips can have different
native two-qubit gates and connectivity, so the circuit was rejected.

Fix: pick the backend FIRST, then transpile specifically for it, then submit
-- all in one script, so there's no window for the two to drift apart.

Still submits exactly ONE job (see the reasoning in the old week5_submit_job.py
docstring -- not repeated here, same logic: simulator-optimized parameters,
hardware only samples).

Adds mthree readout error mitigation: hardware measurement itself has a real
error rate (a qubit prepared as |1> is sometimes read as |0> and vice versa).
mthree calibrates this per-qubit and corrects the measured distribution --
distinct from (and on top of) the circuit/gate noise you're already
investigating by comparing hardware vs. simulator distributions.

REQUIRES, on your machine: a saved IBM Quantum account (see the old
week5_submit_job.py docstring for the one-time save_account call).

Run:
    python week5_hardware_run.py
"""

import numpy as np
import mthree
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from week3_qaoa_circuit import cost_hamiltonian
from week3b_reps_sweep import best_over_restarts
import week3b_reps_sweep as sw

FINAL_REPS = 2
N_RESTARTS_FINAL = 3
SHOTS = 4096


def main():
    d = np.load("hw6_pipeline.npz")
    h, J, tickers = d["h"], d["J"], [str(t) for t in d["tickers"]]
    budget = int(d["budget"])
    n = len(tickers)
    H = cost_hamiltonian(h, J, n)

    # --- 1. Pick the backend FIRST ---
    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=n)
    print(f"Selected backend: {backend.name} ({backend.num_qubits} qubits)")

    # --- 2. Optimize on the simulator (unchanged -- this part is fine) ---
    print(f"\nOptimizing at reps={FINAL_REPS} with {N_RESTARTS_FINAL} restarts...")
    rng = np.random.default_rng(0)
    orig_n_restarts, sw.N_RESTARTS = sw.N_RESTARTS, N_RESTARTS_FINAL
    ansatz, res, label = best_over_restarts(H, n, FINAL_REPS, rng)
    sw.N_RESTARTS = orig_n_restarts
    print(f"Final simulator energy: {res.fun:.4f} (best start: {label})")

    qc = ansatz.assign_parameters(res.x)
    qc.measure_all()

    # --- 3. Transpile for THIS SPECIFIC backend (the actual fix) ---
    print(f"\nTranspiling for {backend.name} specifically...")
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=42)
    transpiled = pm.run(qc)
    ops = transpiled.count_ops()
    two_q = sum(v for k, v in ops.items() if k in ("cx", "cz", "ecr"))
    print(f"Transpiled: depth {transpiled.depth()}, two-qubit gates {two_q}, ops {ops}")

    physical_qubits = transpiled.layout.final_index_layout()[:n] if transpiled.layout else list(range(n))

    # --- 4. Calibrate readout error mitigation for the qubits we're using ---
    print(f"\nCalibrating readout error mitigation (mthree) for qubits {physical_qubits}...")
    mit = mthree.M3Mitigation(backend)
    mit.cals_from_system(physical_qubits)

    # --- 5. Submit ONE job ---
    sampler = SamplerV2(mode=backend)
    print(f"\nSubmitting ONE job, {SHOTS} shots, to {backend.name}...")
    job = sampler.run([transpiled], shots=SHOTS)
    print(f"Job ID: {job.job_id()} -- check status at https://quantum.ibm.com/jobs")

    result = job.result()
    raw_counts = result[0].data.meas.get_counts()

    # --- 6. Apply error mitigation ---
    quasi_probs = mit.apply_correction(raw_counts, physical_qubits)

    def report(dist, label, is_quasi):
        print(f"\nTop 5 ({label}):")
        items = sorted(dist.items(), key=lambda kv: -kv[1])[:5]
        for bitstring, val in items:
            x = [int(b) for b in bitstring[::-1]]
            selected = [t for t, xi in zip(tickers, x) if xi == 1]
            feasible = len(selected) == budget
            shown = f"{val:.4f} (quasi-prob)" if is_quasi else f"{val} counts ({100*val/SHOTS:.1f}%)"
            print(f"  {bitstring}  {shown}  selected={selected}  feasible={feasible}")

    report(raw_counts, "raw hardware counts", is_quasi=False)
    report(dict(quasi_probs), "readout-error-mitigated", is_quasi=True)

    np.savez("hw6_hardware_result.npz", raw_counts=raw_counts,
             mitigated=dict(quasi_probs), backend=backend.name, job_id=job.job_id())
    print(f"\nSaved to hw6_hardware_result.npz")
    print("Compare 'raw hardware counts' against the simulator distribution from "
          "week4_six_stock_pipeline.py (reps=2) for the full noise picture: "
          "simulator -> raw hardware is gate/decoherence noise, "
          "raw -> mitigated isolates readout error specifically.")


if __name__ == "__main__":
    main()
