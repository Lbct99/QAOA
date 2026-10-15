"""
Week 11 -- Backend benchmark: how much does the result depend on the device and the day?

Your pipeline picks a backend with `least_busy`, which only makes sense if devices and days behave
alike. This script runs the SAME p=2 circuit (the constant-start rule's parameters, for the 6-stock
problem you already ran) as ONE JOB PER BACKEND, one circuit and 4096 shots each, and compares:

  * a repeat on ibm_marrakesh, the backend of your first job, on a different day:
    how much does the retention drift from day to day?
  * other backends: is any clearly better or worse than that day-to-day spread?

Metric: measured energy <H_C> over all counts. Noise pulls it toward 0 (the uniform mixture has
energy exactly 0), so retention = measured energy / ideal energy; its standard error is about 1%.
The first job (hw6_hardware_result.npz) is the baseline. Differences between backends only mean
something if they are larger than the day-to-day difference on the same backend.

Each backend gets its own transpilation (its own topology and layout). Readout error is not mitigated,
so it counts as part of device quality. Expect no change in what the best portfolio is: this is about
variability and device choice, not about better answers.

SAFETY. The default is a free OFFLINE DRY RUN on noisy simulators of the backends. Nothing is sent to
IBM unless you pass --submit. Open Plan time is counted in seconds of quantum time: look at what your
first job used on the IBM jobs page; this benchmark uses roughly that much per backend (without the
mitigation calibration). Keep (number of backends) x (first job's seconds) well below what you have left.

Modes:
    python week11_backend_benchmark.py --list-backends
    python week11_backend_benchmark.py                                   # free offline dry run
    python week11_backend_benchmark.py --submit --backends ibm_marrakesh ibm_marrakesh ibm_fez ibm_kingston
    python week11_backend_benchmark.py --retrieve                        # fetch finished jobs and analyze
Options: --shots 4096, --data hw6_pipeline.npz, --baseline hw6_hardware_result.npz, --max-jobs 4
"""

import os

# Tiny problems: single-threaded BLAS is fastest. Must be set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import warnings

import numpy as np

from qaoa_instance_generator import QAOASim
from week10_depth_experiment import rule_params, numpy_probs, count_stats

META_FILE = "week11_job_meta.npz"
DEFAULT_BACKENDS = ["ibm_marrakesh", "ibm_fez", "ibm_kingston"]   # what --list-backends showed for your account


def fake_for(name):
    from qiskit_ibm_runtime import fake_provider as fp
    return getattr(fp, "Fake" + name.replace("ibm_", "").capitalize(), None)


def label_list(names):
    seen, out = {}, []
    for nm in names:
        seen[nm] = seen.get(nm, 0) + 1
        out.append(nm if seen[nm] == 1 else f"{nm} #{seen[nm]}")
    return out


def report(rows, base, ideal_E, ideal_Popt, ideal_Pfeas, dry=False):
    """rows: dicts with name, depth, twoq, stats. base: dict(stats, name) or None."""
    if dry:
        print("\n*** DRY RUN: the rows below are SIMULATED (noisy models of the backends). Their numbers, and any")
        print("*** comparison with your real first job, say nothing about the real devices; this only tests the mechanics.")
    print(f"\nIdeal (no noise): energy {ideal_E:.4f}, P(optimal) {ideal_Popt:.3f}, P(feasible) {ideal_Pfeas:.3f}\n")
    head = (f"{'run':28s} {'depth':>6s} {'2Q gates':>9s} {'measured E (+- SE)':>20s} {'retention':>10s} "
            f"{'P(opt)':>8s} {'P(feas)':>8s} {'z vs first job':>15s}")
    print(head)
    if base:
        s = base["stats"]
        print(f"{base['name']:28s} {'':>6s} {'':>9s} {s['E']:10.4f} +- {s['seE']:.4f} {s['E'] / ideal_E:10.2f} "
              f"{s['Popt']:8.3f} {s['Pfeas']:8.3f} {'(baseline)':>15s}")
    for r in rows:
        s = r["stats"]
        z = ""
        if base:
            z = "n/a (simulated)" if dry else f"{(s['E'] - base['stats']['E']) / np.hypot(s['seE'], base['stats']['seE']):+.1f}"
        print(f"{r['name']:28s} {r['depth']:6d} {r['twoq']:9d} {s['E']:10.4f} +- {s['seE']:.4f} {s['E'] / ideal_E:10.2f} "
              f"{s['Popt']:8.3f} {s['Pfeas']:8.3f} {z:>15s}")

    if not rows:
        return
    best = max(rows, key=lambda r: r["stats"]["E"] / ideal_E)      # retention: higher is better
    print(f"\nHighest retention among the new runs: {best['name']} ({best['stats']['E'] / ideal_E:.2f}).")
    # same backend, several runs: how far apart are two runs submitted in the same session?
    by_backend = {}
    for r in rows:
        by_backend.setdefault(r["name"].split(" #")[0], []).append(r)
    for nm, rs in by_backend.items():
        for a, b in zip(rs[:-1], rs[1:]):
            d = a["stats"]["E"] - b["stats"]["E"]
            se = np.hypot(a["stats"]["seE"], b["stats"]["seE"])
            print(f"Same-session repeat on {nm}: {a['name']} - {b['name']} = {d:+.4f} +- {se:.4f} (z = {d / se:+.1f}) -> "
                  + ("consistent with shot noise" if abs(d / se) < 2 else "more than shot noise"))
    if base and not dry:
        same = [r for r in rows if r["name"].split(" #")[0] == base["name"].split(" (")[0]]
        for r in same:
            s, b = r["stats"], base["stats"]
            d = s["E"] - b["E"]
            se = np.hypot(s["seE"], b["seE"])
            print(f"Day-to-day on {r['name']}: E(new) - E(first job) = {d:+.4f} +- {se:.4f} (z = {d / se:+.1f}) -> "
                  + ("consistent with shot noise" if abs(d / se) < 2 else "more than shot noise: retention drifts from day to day"))
    print("\nHow to read this: a backend only counts as better or worse if its retention differs from the other runs by")
    print("more than the day-to-day spread on ibm_marrakesh; with a single repeat that spread is itself a rough estimate.")
    print("Caveats: one job per backend on one day; each backend has its own layout; readout error is unmitigated;")
    print("P(optimal) is noisy by construction, so lean on the energy.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default="hw6_pipeline.npz")
    ap.add_argument("--baseline", default="hw6_hardware_result.npz")
    ap.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--max-jobs", type=int, default=4)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--retrieve", action="store_true")
    ap.add_argument("--list-backends", action="store_true")
    args = ap.parse_args()

    if args.list_backends:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        print(f"{'backend':18s} {'qubits':>6s} {'pending jobs':>13s}")
        for b in service.backends(simulator=False, operational=True):
            print(f"{b.name:18s} {b.num_qubits:6d} {b.status().pending_jobs:13d}")
        return

    d = np.load(args.data)
    h, J = d["h"], d["J"]
    n = len(h)
    budget = int(d["budget"])
    best_x = np.array(d["best_x"], dtype=int)
    opt_idx = int(sum(int(best_x[i]) << i for i in range(n)))
    feasible = np.array([bin(k).count("1") == budget for k in range(2**n)])
    scale = float(np.sqrt(np.sum(h**2) + np.sum(J**2)))
    sim = QAOASim(h, J, 2)
    diag = sim.diag
    params, e2 = rule_params(sim, scale)
    pr = numpy_probs(sim, params)
    ideal_E, ideal_Popt, ideal_Pfeas = float(pr @ diag), float(pr[opt_idx]), float(pr[feasible].sum())
    print(f"Instance: budget {budget}; p=2 rule parameters, ideal energy {ideal_E:.4f}")

    base = None
    if os.path.exists(args.baseline):
        hw = np.load(args.baseline, allow_pickle=True)
        base = dict(name=f"{str(hw['backend'])} (first job)",
                    stats=count_stats(hw["raw_counts"].item(), diag, opt_idx, feasible))

    # ---- retrieve finished jobs ----
    if args.retrieve:
        from qiskit_ibm_runtime import QiskitRuntimeService
        meta = np.load(META_FILE, allow_pickle=True)
        service = QiskitRuntimeService()
        rows = []
        for i, (nm, jid) in enumerate(zip(meta["names"], meta["job_ids"])):
            job = service.job(str(jid))
            status = str(job.status())
            if "DONE" not in status.upper():
                print(f"  {nm}: job {jid} not finished yet (status {status}); run --retrieve again later.")
                continue
            counts = job.result()[0].data.meas.get_counts()
            rows.append(dict(name=str(nm), depth=int(meta["depth"][i]), twoq=int(meta["twoq"][i]),
                             stats=count_stats(counts, diag, opt_idx, feasible)))
        report(rows, base, ideal_E, ideal_Popt, ideal_Pfeas)
        return

    # ---- build, transpile, run ----
    if len(args.backends) > args.max_jobs:
        raise SystemExit(f"{len(args.backends)} backends requested; --max-jobs is {args.max_jobs}. "
                         f"Raise it deliberately if you really want that many jobs.")
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import Statevector
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from week3_qaoa_circuit import cost_hamiltonian
    try:
        from scipy.sparse import SparseEfficiencyWarning
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
    except ImportError:
        pass
    H = cost_hamiltonian(h, J, n)
    ansatz = QAOAAnsatz(cost_operator=H, reps=2)
    binding = {}
    for q in ansatz.parameters:
        k = int("".join(ch for ch in q.name if ch.isdigit()))
        is_gamma = "\u03b3" in q.name or "gamma" in q.name.lower()
        binding[q] = params[k] if is_gamma else params[2 + k]
    e_q = float(Statevector(ansatz.assign_parameters(binding)).expectation_value(H).real)
    assert abs(e_q - sim.energy(params)) < 1e-9, "numpy simulator and Qiskit disagree"
    print("Qiskit cross-check passed (tolerance 1e-9).")
    qc = ansatz.decompose(reps=2).assign_parameters(binding)
    qc.measure_all()

    names = label_list(args.backends)
    rows, meta_names, meta_ids, meta_depth, meta_twoq = [], [], [], [], []
    if args.submit:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        service = QiskitRuntimeService()
        objs = {}                                     # pre-flight: validate EVERY backend before sending anything
        for nm in dict.fromkeys(args.backends):
            try:
                objs[nm] = service.backend(nm)
            except Exception as e:
                avail = ", ".join(b.name for b in service.backends(simulator=False, operational=True))
                raise SystemExit(f"Backend {nm} is not available to your account ({e}).\n"
                                 f"Available now: {avail}\nNothing was submitted.")
            if not objs[nm].status().operational:
                raise SystemExit(f"Backend {nm} is not operational right now. Nothing was submitted.")
        print(f"\nMode: REAL HARDWARE. Submitting {len(names)} job(s): {', '.join(names)} "
              f"({args.shots} shots, 1 circuit each).")
    else:
        from qiskit_aer import AerSimulator
        print("\nMode: OFFLINE DRY RUN (noisy simulators of the backends; nothing is sent to IBM).")

    for i, nm in enumerate(names):
        real_name = args.backends[i]
        if args.submit:
            backend = objs[real_name]
        else:
            cls = fake_for(real_name)
            if cls is None:
                print(f"  {nm}: no offline model for this backend; skipped in the dry run.")
                continue
            backend = cls()
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=42)
        t = pm.run(qc)
        depth = t.depth()
        twoq = sum(v for k, v in t.count_ops().items() if k in ("cx", "cz", "ecr", "rzz"))
        print(f"  {nm:22s} transpiled: depth {depth}, two-qubit gates {twoq}")
        if args.submit:
            sampler = SamplerV2(mode=backend)
            job = sampler.run([t], shots=args.shots)
            meta_names.append(nm); meta_ids.append(job.job_id())
            meta_depth.append(depth); meta_twoq.append(twoq)
            np.savez(META_FILE, names=np.array(meta_names), job_ids=np.array(meta_ids),
                     depth=np.array(meta_depth), twoq=np.array(meta_twoq), shots=args.shots)
            print(f"    submitted, job ID {job.job_id()} (saved to {META_FILE})")
        else:
            counts = AerSimulator.from_backend(backend).run(t, shots=args.shots, seed_simulator=100 + i).result().get_counts()
            rows.append(dict(name=nm, depth=depth, twoq=twoq, stats=count_stats(counts, diag, opt_idx, feasible)))

    if args.submit:
        print(f"\nAll jobs submitted. They queue independently and may finish at different times.\n"
              f"Fetch results with:  python week11_backend_benchmark.py --retrieve")
        return
    report(rows, base, ideal_E, ideal_Popt, ideal_Pfeas, dry=True)


if __name__ == "__main__":
    main()
