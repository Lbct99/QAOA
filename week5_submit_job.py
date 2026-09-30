"""
Week 4b -- Run the QAOA circuit on real IBM quantum hardware.

DESIGN CHOICE, stated explicitly: this submits exactly ONE hardware job,
using parameters already optimized on the simulator (week5_transpile.py).
It does NOT run an optimization loop on hardware.

Why: a VQE/QAOA-style optimization loop calls the cost function dozens to
hundreds of times (your simulator run in week3b took 50-250 evaluations per
depth). On real hardware, each evaluation is a separate job with real queue
time, and the IBM Open (free) Plan has limited monthly runtime. Optimizing
directly on hardware would burn that budget in one run for very little gain,
since simulator-optimized parameters already transfer reasonably well when
the circuit is this shallow (reps=2) and the mapping is exact. One sampling
job answers the real remaining question: "how much does real hardware noise
degrade the simulator's distribution?" -- which is what actually matters for
your write-up.

REQUIRES, on your machine (not this sandbox, which has no internet access to
IBM Cloud):
    1. A free IBM Quantum account: https://quantum.ibm.com
    2. Your API token, saved once via:
         python -c "from qiskit_ibm_runtime import QiskitRuntimeService; \
                    QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token='YOUR_TOKEN')"
       (channel name has changed across qiskit-ibm-runtime versions -- if this
       errors, check https://quantum.ibm.com/ for the current account-setup snippet)

Run:
    python week5_submit_job.py
"""

import numpy as np
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

SHOTS = 4096


def main():
    with open("hw6_transpiled_circuit.qpy", "rb") as f:
        circuits = qpy.load(f)
    qc = circuits[0]

    d = np.load("hw6_pipeline.npz")
    tickers = [str(t) for t in d["tickers"]]
    budget = int(d["budget"])
    n = len(tickers)

    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=qc.num_qubits)
    print(f"Selected backend: {backend.name} ({backend.num_qubits} qubits)")
    print("NOTE: qc was transpiled for FakeSherbrooke's layout in week5_transpile.py. "
          "If your least-busy backend is a DIFFERENT chip, re-run week5_transpile.py with "
          "BACKEND = service.least_busy(...) substituted for FakeSherbrooke() first -- "
          "a circuit transpiled for the wrong topology will fail or need re-routing.")

    sampler = SamplerV2(mode=backend)
    print(f"Submitting ONE job, {SHOTS} shots, to {backend.name}...")
    job = sampler.run([qc], shots=SHOTS)
    print(f"Job ID: {job.job_id()}  -- check status at https://quantum.ibm.com/jobs")

    result = job.result()
    counts = result[0].data.meas.get_counts()

    total = sum(counts.values())
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    print(f"\nTop 5 measured bitstrings out of {total} shots (REAL HARDWARE):")
    for bitstring, count in top:
        x = [int(b) for b in bitstring[::-1]]
        selected = [t for t, xi in zip(tickers, x) if xi == 1]
        feasible = len(selected) == budget
        print(f"  {bitstring}  count={count:4d} ({100*count/total:.1f}%)  "
              f"selected={selected}  feasible={feasible}")

    np.savez("hw6_hardware_result.npz", counts=counts, backend=backend.name, job_id=job.job_id())
    print(f"\nSaved to hw6_hardware_result.npz")
    print("Compare this distribution against the simulator's (week4_six_stock_pipeline.py, "
          "reps=2 run) -- the difference between the two IS your hardware-noise finding.")


if __name__ == "__main__":
    main()
