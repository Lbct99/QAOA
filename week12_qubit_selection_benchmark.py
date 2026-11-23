"""
week12_qubit_selection_benchmark.py -- a safe, heavily scaled-down version of IBM's
"Real-time benchmarking for qubit selection" tutorial, sized for the Open plan.

WHY THE ORIGINAL TUTORIAL IS EXPENSIVE (not a bug, just scaled for a bigger plan):
it forces min_num_qubits=127 (the whole chip), runs single-qubit RB out to length 500
(500 Clifford layers -- very deep circuits), and compares Bell-state fidelity on chains
up to 127 qubits long with dynamical decoupling enabled. Width x depth x shots is what
actually costs QPU time -- the circuit COUNT (~210 across two jobs) was never huge.

WHAT IS DIFFERENT HERE:
- A small, hand-picked qubit subset (default 6, hard-capped at 12) instead of the full
  backend. Picking is FREE (reads backend.properties()/coupling_map, no QPU time) and
  greedily favours low readout error and long T1/T2 in TODAY's calibration -- so it
  actively steers away from whatever the worst qubits are right now, the same way you'd
  want to avoid a qubit like the one you found earlier (marrakesh qubit 13).
- Transpilation is pinned to exactly that qubit subset (initial_layout=qubits), so width
  never silently grows back to the full chip.
- Two-qubit RB is OFF by default (--include-2q-rb to enable) -- the most circuit-heavy
  part of the suite and the least essential for a quick check.
- RB lengths default to [10, 30] (not [10, 100, 500]); Bell-chain lengths default to
  [2, 4, 6] (not up to 127); n_trials is still 3 by default but hard-capped at 3.
- Nothing runs on hardware without BOTH --submit on the command line AND typing RUN at
  an interactive prompt that shows the exact circuit count first. Metadata-only calls
  (--list-backends, and the dry-run plan itself) never need --submit: they query free
  backend/calibration data, not the QPU, and are always safe to run -- including right
  now, before your runtime resets.
- Every submitted job is logged to week12_runs.jsonl (time, backend, kind, qubits,
  n_circuits, shots, job id(s)), the same pattern as week11_backend_benchmark.py, so
  usage can be audited afterwards.

IMPORTANT REFRAMING: for picking good qubits before an ordinary run (week5/week11), you
do NOT need to pay for new characterization at all -- reading today's already-published
calibration (free) and picking qubits from it, which is exactly what --list-backends /
the dry-run plan below already does, gets you most of the benefit for nothing. The paid
real-time re-characterization (T1/T2/RB re-measurement) below is worth its QPU cost
mainly when you suspect the published calibration is stale or wrong (like the marrakesh
qubit-13 drift), not as a routine warm-up before every single run.

This closely follows the original tutorial's logic for the characterization-and-update
step, but has NOT been run against the real IBM API or qiskit-experiments in this
environment (no network access here) -- only the dry-run arithmetic (qubit picking,
circuit-count plan) has been tested. Run the dry run first (no --submit, free, works
right now) and check the printed plan carefully before ever adding --submit.

Requires: qiskit-ibm-runtime, qiskit-experiments, rustworkx (same as the tutorial).

Typical use:
    python week12_qubit_selection_benchmark.py --list-backends
    python week12_qubit_selection_benchmark.py --backend ibm_marrakesh --n-qubits 6
        (dry run: shows the picked qubits and the exact circuit/shot counts --
         submits NOTHING, costs no QPU time, safe to run anytime)
    python week12_qubit_selection_benchmark.py --backend ibm_marrakesh --n-qubits 6 --submit
        (same plan, but asks for typed confirmation before each real submission)
"""
import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone

LOG_FILE = "week12_runs.jsonl"
HARD_MAX_QUBITS = 12
HARD_MAX_TRIALS = 3
HARD_MAX_CIRCUITS_PER_JOB = 80  # refuse anything bigger without an explicit override


def log(entry):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def pick_qubits(backend, n):
    """FREE (no QPU time): greedily pick a connected subgraph of n qubits favouring low
    readout error and long T1/T2 in TODAY's calibration. Good enough for a warm-up
    check and for keeping the paid characterization below away from bad qubits -- not a
    substitute for a thorough search."""
    import rustworkx as rx  # noqa: F401  (imported for the user's own extensions)
    props = backend.properties()
    graph = backend.coupling_map.graph.to_undirected(multigraph=False)

    def score(q):
        try:
            ro = props.readout_error(q)
        except Exception:
            ro = 0.05
        try:
            t1, t2 = props.t1(q), props.t2(q)
        except Exception:
            t1, t2 = 100e-6, 100e-6
        return ro - 1e-7 * (t1 + t2)  # lower is better

    nodes = list(graph.node_indices())
    start = min(nodes, key=score)
    chosen = [start]
    frontier = set(graph.neighbors(start))
    while len(chosen) < n and frontier:
        nxt = min(frontier, key=score)
        chosen.append(nxt)
        frontier.discard(nxt)
        frontier |= set(graph.neighbors(nxt)) - set(chosen)
    if len(chosen) < n:
        raise RuntimeError(f"could only find {len(chosen)} connected qubits on this "
                           f"backend, wanted {n}")
    return chosen


def edges_within(backend, qubits):
    qs = set(qubits)
    graph = backend.coupling_map.graph.to_undirected(multigraph=False)
    return [tuple(sorted(e)) for e in graph.edge_list() if e[0] in qs and e[1] in qs]


def build_plan(args, backend):
    """Pure arithmetic plus free metadata calls -- no QPU time, always safe to run."""
    qubits = args.qubits if args.qubits else pick_qubits(backend, args.n_qubits)
    edges = edges_within(backend, qubits)

    n_t1 = len(args.t1_delays)
    n_t2 = len(args.t1_delays)
    n_ro = 2  # LocalReadoutError: one all-0 and one all-1 calibration circuit
    n_1q_rb = len(args.rb_lengths) * args.rb_samples
    n_2q_rb = (len(args.rb_lengths) * args.rb_samples) if (args.include_2q_rb and edges) else 0
    n_char = n_t1 + n_t2 + n_ro + n_1q_rb + n_2q_rb

    chain_lengths = [c for c in args.chain_lengths if c <= len(qubits)]
    dropped = [c for c in args.chain_lengths if c > len(qubits)]
    n_bell = 2 * len(chain_lengths)  # original + updated, per length
    n_main = n_bell * args.n_trials

    return dict(qubits=qubits, edges=edges, chain_lengths=chain_lengths, dropped=dropped,
                n_char=n_char, n_main=n_main, shots=args.shots,
                breakdown=dict(t1=n_t1, t2=n_t2, readout=n_ro, rb1q=n_1q_rb, rb2q=n_2q_rb,
                               bell_per_trial=n_bell, trials=args.n_trials))


def print_plan(plan):
    b = plan["breakdown"]
    print(f"Qubits ({len(plan['qubits'])}): {plan['qubits']}")
    print(f"Couplers within this subset: {plan['edges']}")
    if plan["dropped"]:
        print(f"Dropped chain length(s) longer than the qubit subset: {plan['dropped']}")
    print(f"Bell-chain lengths tested: {plan['chain_lengths']}")
    print("\nCharacterization circuits (job 1):")
    print(f"  T1 {b['t1']}  T2 {b['t2']}  readout-cal {b['readout']}  "
          f"1Q-RB {b['rb1q']}  2Q-RB {b['rb2q']}  ->  total {plan['n_char']}")
    print(f"Main comparison circuits (job 2, x{b['trials']} trials): "
          f"{b['bell_per_trial']} per trial  ->  total {plan['n_main']}")
    print(f"Shots per circuit: {plan['shots']}")
    total = plan["n_char"] + plan["n_main"]
    print(f"\nGRAND TOTAL circuits this would submit: {total} "
          f"({plan['n_char']} characterization + {plan['n_main']} comparison)")
    print("Reference point: a 6-qubit, 4096-shot, depth-256 QAOA job you ran earlier cost "
          "3 seconds of QPU charge time. This plan's circuits are far shallower and "
          "narrower than the original tutorial's -- but this is a circuit COUNT, not a "
          "time estimate. Check the actual qpu_charge_time after each job (job.metrics()).")
    return total


def confirm(total_circuits):
    if total_circuits > HARD_MAX_CIRCUITS_PER_JOB and "--i-have-checked-this" not in sys.argv:
        print(f"\nREFUSING: {total_circuits} circuits exceeds the safety cap of "
              f"{HARD_MAX_CIRCUITS_PER_JOB}. Lower --n-qubits / --rb-lengths / "
              "--rb-samples / --n-trials, or re-run with --i-have-checked-this if this "
              "is deliberate.")
        return False
    ans = input("\nType RUN to submit this to real hardware, anything else to cancel: ")
    return ans.strip() == "RUN"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--backend")
    ap.add_argument("--list-backends", action="store_true",
                    help="free: list operational backends and their queue length")
    ap.add_argument("--n-qubits", type=int, default=6)
    ap.add_argument("--qubits", type=int, nargs="+",
                    help="use these exact physical qubits instead of auto-picking")
    ap.add_argument("--chain-lengths", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--rb-lengths", type=int, nargs="+", default=[10, 30])
    ap.add_argument("--rb-samples", type=int, default=5)
    ap.add_argument("--t1-delays", type=float, nargs="+", default=[1e-6, 40e-6, 200e-6])
    ap.add_argument("--include-2q-rb", action="store_true",
                    help="off by default: the most circuit-heavy part of the suite")
    ap.add_argument("--n-trials", type=int, default=1)
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--submit", action="store_true",
                    help="without this, only prints the plan -- no QPU time used")
    args = ap.parse_args()

    if args.n_qubits > HARD_MAX_QUBITS:
        raise SystemExit(f"--n-qubits capped at {HARD_MAX_QUBITS} for the Open plan.")
    if args.n_trials > HARD_MAX_TRIALS:
        raise SystemExit(f"--n-trials capped at {HARD_MAX_TRIALS} for the Open plan.")

    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()

    if args.list_backends:
        print(f"{'backend':18s} {'qubits':>6s} {'pending jobs':>13s}")
        for b in service.backends(simulator=False, operational=True):
            print(f"{b.name:18s} {b.num_qubits:6d} {b.status().pending_jobs:13d}")
        return

    backend = service.backend(args.backend) if args.backend else \
        service.least_busy(operational=True, simulator=False)
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits total, "
          f"{backend.status().pending_jobs} pending jobs) -- metadata only, no QPU time.\n")

    plan = build_plan(args, backend)
    total = print_plan(plan)

    if not args.submit:
        print("\nDry run only (pass --submit to go further). Nothing was sent to hardware.")
        return

    if not confirm(total):
        print("Cancelled.")
        return

    # ---- real submission, in two separately-logged jobs -------------------------------
    from qiskit import QuantumCircuit
    from qiskit.transpiler import generate_preset_pass_manager, InstructionProperties
    from qiskit.quantum_info import hellinger_fidelity
    from qiskit_experiments.library import T1, T2Hahn, LocalReadoutError, StandardRB
    from qiskit_experiments.framework import BatchExperiment, ParallelExperiment
    from qiskit_ibm_runtime import SamplerV2
    import numpy as np

    qubits, edges = plan["qubits"], plan["edges"]
    sampler = SamplerV2(mode=backend)
    sampler.options.environment.job_tags = ["week12_qubit_selection"]

    # --- characterization job ---
    batches = [
        ParallelExperiment([T1(physical_qubits=[q], delays=args.t1_delays) for q in qubits],
                           backend, analysis=None),
        ParallelExperiment([T2Hahn(physical_qubits=[q], delays=args.t1_delays) for q in qubits],
                           backend, analysis=None),
        LocalReadoutError(qubits),
        ParallelExperiment([StandardRB(physical_qubits=[q], lengths=args.rb_lengths,
                                       num_samples=args.rb_samples) for q in qubits],
                           backend, analysis=None),
    ]
    if args.include_2q_rb and edges:
        batches.append(ParallelExperiment(
            [StandardRB(physical_qubits=list(e), lengths=args.rb_lengths,
                        num_samples=args.rb_samples) for e in edges],
            backend, analysis=None))
    batches_exp = BatchExperiment(batches, backend, flatten_results=True, analysis=None)

    t0 = datetime.now(timezone.utc)
    data = batches_exp.run(sampler=sampler, shots=args.shots).block_for_results()
    log(dict(time=t0.isoformat(), backend=backend.name, kind="characterization",
             qubits=qubits, n_circuits=plan["n_char"], shots=args.shots))
    print(f"\nCharacterization done ({t0.isoformat()}).")

    target = copy.deepcopy(backend.target)
    for r in data.analysis_results("EPG_sx"):
        target.update_instruction_properties(
            "sx", (r.device_components[0].index,),
            InstructionProperties(error=r.value.nominal_value))
    for r in data.analysis_results("EPG_x"):
        target.update_instruction_properties(
            "x", (r.device_components[0].index,),
            InstructionProperties(error=r.value.nominal_value))
    readout_data = data.analysis_results("Local Readout Mitigator")[0]
    for q in qubits:
        m = readout_data.value.assignment_matrix(q)
        target.update_instruction_properties(
            "measure", (q,), InstructionProperties(error=(m[0, 1] + m[1, 0]) / 2))
    for r in data.analysis_results("T1"):
        target.qubit_properties[r.device_components[0].index].t1 = r.value.nominal_value
    for r in data.analysis_results("T2"):
        target.qubit_properties[r.device_components[0].index].t2 = r.value.nominal_value
    if args.include_2q_rb and edges:
        for r in data.analysis_results("EPG_cz"):
            qs = tuple(c.index for c in r.device_components)
            try:
                target.update_instruction_properties(
                    "cz", qs, InstructionProperties(error=r.value.nominal_value))
            except (KeyError, ValueError):
                target.update_instruction_properties(
                    "cz", qs[::-1], InstructionProperties(error=r.value.nominal_value))

    # --- Bell-chain comparison: original (published calibration) vs updated (just now) ---
    ideal_dist = {"00": 0.5, "11": 0.5}
    interleaved = []
    for n in plan["chain_lengths"]:
        qc = QuantumCircuit(n, 2)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.barrier(); qc.measure(0, 0); qc.measure(n - 1, 1)
        layout = qubits[:n]
        pm_orig = generate_preset_pass_manager(target=backend.target, optimization_level=3,
                                               initial_layout=layout)
        pm_upd = generate_preset_pass_manager(target=target, optimization_level=3,
                                              initial_layout=layout)
        interleaved.append(pm_orig.run(qc))
        interleaved.append(pm_upd.run(qc))

    all_orig, all_upd = [], []
    for trial in range(args.n_trials):
        t0 = datetime.now(timezone.utc)
        job = sampler.run(interleaved, shots=args.shots)
        res = job.result()
        log(dict(time=t0.isoformat(), backend=backend.name, kind="bell_trial", trial=trial,
                 qubits=qubits, n_circuits=len(interleaved), shots=args.shots,
                 job_id=job.job_id()))
        orig, upd = [], []
        for i in range(len(plan["chain_lengths"])):
            orig.append(hellinger_fidelity(ideal_dist, res[2 * i].data.c.get_counts()))
            upd.append(hellinger_fidelity(ideal_dist, res[2 * i + 1].data.c.get_counts()))
        all_orig.append(orig); all_upd.append(upd)
        print(f"  trial {trial}: job {job.job_id()}  original={orig}  updated={upd}")

    print("\nMean Hellinger fidelity per chain length:")
    for n, o, u in zip(plan["chain_lengths"], np.mean(all_orig, axis=0), np.mean(all_upd, axis=0)):
        print(f"  length {n:3d}: original {o:.3f}  updated {u:.3f}  "
              f"(updated - original {u - o:+.3f})")


if __name__ == "__main__":
    main()
