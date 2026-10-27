"""
Week 11 -- Backend benchmark and day-on-day monitor.

Your pipeline picks a backend with `least_busy`. That only makes sense if devices and days behave alike.
This script runs the SAME p=2 circuit (the constant-start rule's parameters, for the 6-stock problem you
already ran) as ONE JOB PER BACKEND (one circuit, 4096 shots) and keeps a permanent record, so you can repeat
it day after day and watch the retention over time.

Metric: measured energy <H_C> over all counts. Noise pulls it toward 0 (the uniform mixture has energy
exactly 0), so retention = measured energy / ideal energy; one run's standard error is about 1%.
Your first job (hw6_hardware_result.npz) is the first point of the series.

What accumulates (keep these files; commit them to git):
    week11_runs.jsonl    one line per submitted job: time, backend, job id, depth, gates, note
    week11_counts.json   the measured counts of every finished job, cached on first retrieval
                         (IBM keeps job results for a limited time, so retrieve soon after jobs finish)

The analysis separates shot noise from real run-to-run drift: for each backend it compares the observed
scatter of the energies with what shot noise alone would give (a chi-square test), then ranks backends
with honest uncertainties. With only a few runs these statements are rough; they sharpen as days add up.

Suggested routine (check Open Plan usage after the first day to see what one job costs):
    day 1:  --submit --backends ibm_marrakesh ibm_marrakesh ibm_fez ibm_kingston     (the repeat gives the
            same-session spread); later days: whichever backends your budget allows, same command style.
    after the jobs finish:  --retrieve

SAFETY. The default is a free OFFLINE DRY RUN on noisy simulators of the backends. Nothing is sent to IBM
unless you pass --submit. Open Plan time is counted in seconds of quantum time: each job costs about what
your first job used (without its mitigation calibration). Each backend gets its own transpilation.
Readout error is not mitigated, so it counts as part of device quality.

Modes:
    python week11_backend_benchmark.py --list-backends
    python week11_backend_benchmark.py --backends ibm_marrakesh ibm_marrakesh ibm_fez ibm_kingston   # free dry run
    python week11_backend_benchmark.py --submit --backends ibm_marrakesh ibm_marrakesh ibm_fez ibm_kingston
    python week11_backend_benchmark.py --retrieve                        # fetch finished jobs, analyze the series
Options: --note "text" (stored with each submitted job), --shots 4096, --max-jobs 4,
         --data hw6_pipeline.npz, --baseline hw6_hardware_result.npz
"""

import os

# Tiny problems: single-threaded BLAS is fastest. Must be set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import warnings
from datetime import datetime

import numpy as np
from scipy.stats import chi2 as chi2_dist

from qaoa_instance_generator import QAOASim
from week10_depth_experiment import rule_params, numpy_probs, count_stats

LOG_FILE = "week11_runs.jsonl"
COUNTS_FILE = "week11_counts.json"
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


def read_log():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_log(entry):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_counts():
    if os.path.exists(COUNTS_FILE):
        with open(COUNTS_FILE) as f:
            return json.load(f)
    return {}


def save_counts(cache):
    with open(COUNTS_FILE, "w") as f:
        json.dump(cache, f)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def dry_report(rows, ideal_E, ideal_Popt, ideal_Pfeas):
    print("\n*** DRY RUN: the rows below are SIMULATED (noisy models of the backends). Their numbers say nothing")
    print("*** about the real devices (the offline models are more optimistic than your real job); this only")
    print("*** tests the mechanics.")
    print(f"\nIdeal (no noise): energy {ideal_E:.4f}, P(optimal) {ideal_Popt:.3f}, P(feasible) {ideal_Pfeas:.3f}\n")
    print(f"{'run':22s} {'depth':>6s} {'2Q gates':>9s} {'measured E (+- SE)':>20s} {'retention':>10s} {'P(opt)':>8s} {'P(feas)':>8s}")
    for r in rows:
        s = r["stats"]
        print(f"{r['name']:22s} {r['depth']:6d} {r['twoq']:9d} {s['E']:10.4f} +- {s['seE']:.4f} {s['E'] / ideal_E:10.2f} "
              f"{s['Popt']:8.3f} {s['Pfeas']:8.3f}")
    by = {}
    for r in rows:
        by.setdefault(r["name"].split(" #")[0], []).append(r)
    for nm, rs in by.items():
        for a, b in zip(rs[:-1], rs[1:]):
            d = a["stats"]["E"] - b["stats"]["E"]
            se = np.hypot(a["stats"]["seE"], b["stats"]["seE"])
            print(f"\nSame-session repeat on {nm}: {a['name']} - {b['name']} = {d:+.4f} +- {se:.4f} (z = {d / se:+.1f}) -> "
                  + ("consistent with shot noise" if abs(d / se) < 2 else "more than shot noise"))


def series_report(rows, ideal_E, ideal_Popt, ideal_Pfeas):
    """rows: dicts with time, backend, depth, twoq (or None), stats; first row may be the earlier first job."""
    print(f"\nIdeal (no noise): energy {ideal_E:.4f}, P(optimal) {ideal_Popt:.3f}, P(feasible) {ideal_Pfeas:.3f}\n")
    print(f"{'time':17s} {'backend':15s} {'depth':>6s} {'2Q':>5s} {'measured E (+- SE)':>20s} {'retention':>10s} "
          f"{'P(opt)':>8s} {'P(feas)':>8s}")
    for r in rows:
        s = r["stats"]
        dep = "" if r["depth"] is None else str(r["depth"])
        tq = "" if r["twoq"] is None else str(r["twoq"])
        print(f"{r['time']:17s} {r['backend']:15s} {dep:>6s} {tq:>5s} {s['E']:10.4f} +- {s['seE']:.4f} "
              f"{s['E'] / ideal_E:10.2f} {s['Popt']:8.3f} {s['Pfeas']:8.3f}")

    by = {}
    for r in rows:
        by.setdefault(r["backend"], []).append(r)
    print("\nPer backend (retention = measured energy / ideal energy):")
    summary = {}
    for b, rs in by.items():
        E = np.array([r["stats"]["E"] for r in rs])
        se = np.array([r["stats"]["seE"] for r in rs])
        n = len(rs)
        w = 1.0 / se**2
        mean = float((w * E).sum() / w.sum())
        unc = float(np.sqrt(1.0 / w.sum()))
        line = f"  {b:15s} n={n}  mean retention {mean / ideal_E:.3f}"
        if n >= 2:
            var_obs = float(E.var(ddof=1))
            exp_var = float((se**2).mean())
            stat = float((w * (E - mean) ** 2).sum())
            pval = float(chi2_dist.sf(stat, n - 1))
            drift = float(np.sqrt(max(var_obs - exp_var, 0.0)))
            unc = max(unc, float(np.sqrt(var_obs / n)))
            verdict = ("consistent with shot noise" if pval >= 0.05 else
                       f"drift beyond shot noise, about {drift / abs(ideal_E):.3f} retention (1 sigma)")
            line += (f" +- {unc / abs(ideal_E):.3f}; run-to-run scatter {np.sqrt(var_obs) / abs(ideal_E):.3f} "
                     f"(shot noise alone: {np.sqrt(exp_var) / abs(ideal_E):.3f}); chi-square p = {pval:.2f} -> {verdict}")
        else:
            line += f" +- {unc / abs(ideal_E):.3f} (a single run: shot noise only)"
        summary[b] = (mean, unc)
        print(line)

    if len(summary) >= 2:
        ranked = sorted(summary.items(), key=lambda kv: -kv[1][0] / ideal_E)
        print("\nRanking by mean retention:")
        for b, (m, u) in ranked:
            print(f"  {b:15s} {m / ideal_E:.3f} +- {u / abs(ideal_E):.3f}")
        (b1, (m1, u1)), (b2, (m2, u2)) = ranked[0], ranked[-1]
        d, ud = (m1 - m2) / ideal_E, np.hypot(u1, u2) / abs(ideal_E)
        print(f"Best minus worst ({b1} - {b2}): {d:+.3f} +- {ud:.3f} (z = {d / ud:+.1f}) -> "
              + ("not distinguishable yet" if d / ud < 2 else "a real difference at this precision"))
    print("\nHow to read this: a backend only counts as better or worse if its retention differs by more than the")
    print("run-to-run scatter on a single backend. A drift verdict needs at least 2 runs on a backend and becomes")
    print("reliable with 5 or more. Caveats: each backend has its own layout and calibration; readout error is")
    print("unmitigated; P(optimal) is noisy by construction, so lean on the energy.")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default="hw6_pipeline.npz")
    ap.add_argument("--baseline", default="hw6_hardware_result.npz")
    ap.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--max-jobs", type=int, default=4)
    ap.add_argument("--note", default="")
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
    params, _ = rule_params(sim, scale)
    pr = numpy_probs(sim, params)
    ideal_E, ideal_Popt, ideal_Pfeas = float(pr @ diag), float(pr[opt_idx]), float(pr[feasible].sum())
    print(f"Instance: budget {budget}; p=2 rule parameters, ideal energy {ideal_E:.4f}")

    # ---- retrieve: every logged job, cached locally ----
    if args.retrieve:
        entries, cache = read_log(), load_counts()
        if not entries and not os.path.exists(args.baseline):
            raise SystemExit(f"No runs logged yet ({LOG_FILE} missing). Submit first.")
        service = None
        rows = []
        if os.path.exists(args.baseline):
            hw = np.load(args.baseline, allow_pickle=True)
            rows.append(dict(time="first job", backend=str(hw["backend"]), depth=None, twoq=None,
                             stats=count_stats(hw["raw_counts"].item(), diag, opt_idx, feasible)))
        for e in entries:
            jid = e["job_id"]
            if jid not in cache:
                if service is None:
                    from qiskit_ibm_runtime import QiskitRuntimeService
                    service = QiskitRuntimeService()
                job = service.job(jid)
                status = str(job.status())
                if "DONE" not in status.upper():
                    print(f"  {e['backend']} job {jid} ({e['time']}): status {status}; run --retrieve again later.")
                    continue
                counts = job.result()[0].data.meas.get_counts()
                cache[jid] = {str(k): int(v) for k, v in counts.items()}
                save_counts(cache)
            rows.append(dict(time=e["time"].replace("T", " ")[:16], backend=e["backend"], depth=e["depth"],
                             twoq=e["twoq"], stats=count_stats(cache[jid], diag, opt_idx, feasible)))
        series_report(rows, ideal_E, ideal_Popt, ideal_Pfeas)
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
    rows = []
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
            append_log(dict(time=datetime.now().isoformat(timespec="seconds"), backend=real_name,
                            job_id=job.job_id(), shots=args.shots, depth=depth, twoq=twoq, note=args.note))
            print(f"    submitted, job ID {job.job_id()} (logged in {LOG_FILE})")
        else:
            counts = AerSimulator.from_backend(backend).run(t, shots=args.shots, seed_simulator=100 + i).result().get_counts()
            rows.append(dict(name=nm, depth=depth, twoq=twoq, stats=count_stats(counts, diag, opt_idx, feasible)))

    if args.submit:
        print("\nAll jobs submitted and logged. They queue independently and may finish at different times.\n"
              "Fetch and analyze with:  python week11_backend_benchmark.py --retrieve\n"
              "Repeat the --submit command on other days to build the series.")
        return
    dry_report(rows, ideal_E, ideal_Popt, ideal_Pfeas)


if __name__ == "__main__":
    main()
