"""
Week 10 -- Hardware depth experiment: does a shallower circuit survive noise better?

Your first hardware job ran p=2 (about 109 two-qubit gates after routing). On the simulator, p=1
reaches almost the same ideal quality with far fewer gates, so on a noisy device it may lose less.
This script tests that on the problem you actually ran, with ONE job containing THREE circuits that
run back to back on the same backend and the same physical qubits:

    1. p=2 with the constant-start rule's parameters
    2. p=1 with its best parameters
    3. p=2 again (the identical circuit as 1), which measures short-term repeatability

Main metric: the measured energy <H_C>, computed from ALL counts. It is far more precise than the
probability of the single optimal portfolio (about 0.1 +- 0.005). Noise pulls the energy toward 0,
because the uniform mixture has energy exactly 0, so  retention = measured energy / ideal energy.
p=1 beats p=2 on hardware only if its retention is at least E_ideal(p=2) / E_ideal(p=1) times that of p=2.

SAFETY. The default is a free OFFLINE DRY RUN on a noisy simulator of ibm_marrakesh (Aer + the fake
backend). Nothing is sent to IBM unless you pass --submit. Open Plan time is measured in seconds of
quantum time: look at what your first job used on the IBM jobs page and expect about three times that
for this job (it has three circuits).

Modes:
    python week10_depth_experiment.py --analyze-existing hw6_hardware_result.npz   # free: retention of your first job
    python week10_depth_experiment.py                                              # free: offline dry run
    python week10_depth_experiment.py --submit                                     # real hardware, one job
    python week10_depth_experiment.py --retrieve JOB_ID                            # fetch and analyze a submitted job

Options: --backend ibm_marrakesh (or least_busy), --shots 4096, --data hw6_pipeline.npz, --seed 0
"""

import os

# Tiny problems: single-threaded BLAS is fastest. Must be set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import warnings

import numpy as np
from scipy.optimize import minimize

from qaoa_instance_generator import QAOASim, ramp_start
from qaoa_pipeline_comparison import run_budgeted, run_full, DEFAULT_STARTS

META_FILE = "week10_job_meta.npz"
NAMES = ["p=2 (first)", "p=1", "p=2 (repeat)"]


# ---------------------------------------------------------------------------
# Parameters and ideal numbers (numpy only)
# ---------------------------------------------------------------------------
def numpy_probs(sim, params):
    p = sim.p
    psi = sim.psi0
    for g, b in zip(params[:p], params[p:]):
        psi = psi * np.exp(-1j * g * sim.diag)
        psi = sim.W @ (np.exp(-1j * b * sim.zsum) * (sim.W @ psi))
    return psi.real**2 + psi.imag**2


def rule_params(sim, scale, budget=25):
    """The constant-start rule (p=2 only): 4 short runs from the portfolio starts, polish the best."""
    p = sim.p
    if p != 2:
        raise SystemExit("The built-in constant starts exist for p=2 only.")
    starts = np.array(DEFAULT_STARTS, dtype=float).reshape(-1, 2 * p)
    shorts = [run_budgeted(sim, np.concatenate([s[:p] / scale, s[p:]]), budget) for s in starts]
    pick = int(np.argmin([t[1] for t in shorts]))
    x, e, _ = run_full(sim, shorts[pick][0])
    if shorts[pick][1] < e:
        x, e = shorts[pick][0], shorts[pick][1]
    return x, float(e)


def wide_search(sim, scale, rng, n_random=40):
    p = sim.p
    cands = [np.concatenate(ramp_start(p))]
    for _ in range(n_random):
        cands.append(np.concatenate([rng.uniform(-3, 3, p) / scale, rng.uniform(-np.pi / 2, np.pi / 2, p)]))
    best_e, best_x = np.inf, None
    for x0 in cands:
        x, e, _ = run_full(sim, x0)
        if e < best_e:
            best_e, best_x = e, x
    res = minimize(sim.energy, best_x, method="Powell", options={"xtol": 1e-8, "ftol": 1e-12})
    if res.fun < best_e:
        best_e, best_x = float(res.fun), res.x
    return best_x, float(best_e)


# ---------------------------------------------------------------------------
# Analysis of measured counts
# ---------------------------------------------------------------------------
def count_stats(counts, diag, opt_idx, feasible):
    total = float(sum(counts.values()))
    probs = np.zeros(len(diag))
    for b, c in counts.items():
        probs[int(str(b).replace(" ", ""), 2)] += c / total
    e = float(probs @ diag)
    se_e = float(np.sqrt(max(probs @ diag**2 - e**2, 0.0) / total))
    po, pf = float(probs[opt_idx]), float(probs[feasible].sum())
    return dict(E=e, seE=se_e, Popt=po, sePopt=float(np.sqrt(po * (1 - po) / total)),
                Pfeas=pf, sePfeas=float(np.sqrt(pf * (1 - pf) / total)))


def report(meta, counts_list, diag, opt_idx, feasible):
    stats = [count_stats(c, diag, opt_idx, feasible) for c in counts_list]
    print(f"\nBackend {meta['backend']}, job {meta['job_id']}, {int(meta['shots'])} shots per circuit.\n")
    print(f"{'circuit':14s} {'depth':>6s} {'2Q gates':>9s} {'ideal E':>9s} {'measured E (+- SE)':>20s} {'retention':>10s}")
    for i, s in enumerate(stats):
        ret = s["E"] / meta["ideal_E"][i]
        print(f"{NAMES[i]:14s} {int(meta['depth'][i]):6d} {int(meta['twoq'][i]):9d} {meta['ideal_E'][i]:9.4f} "
              f"{s['E']:10.4f} +- {s['seE']:.4f} {ret:10.2f}")
    print(f"\n{'circuit':14s} {'ideal P(opt)':>13s} {'measured':>16s} {'ideal P(feas)':>14s} {'measured':>16s}")
    for i, s in enumerate(stats):
        print(f"{NAMES[i]:14s} {meta['ideal_Popt'][i]:13.3f} {s['Popt']:8.3f} +- {s['sePopt']:.3f} "
              f"{meta['ideal_Pfeas'][i]:14.3f} {s['Pfeas']:8.3f} +- {s['sePfeas']:.3f}")

    a, b, c = stats[0], stats[1], stats[2]
    d = a["E"] - c["E"]
    se = np.hypot(a["seE"], c["seE"])
    print(f"\nRepeatability (the identical p=2 circuit twice, same job): E(first) - E(repeat) = {d:+.4f} +- {se:.4f} "
          f"(z = {d / se:+.1f}) -> " + ("consistent with shot noise" if abs(d / se) < 2 else "more than shot noise: the device drifted within the job"))
    p2 = 0.5 * (a["E"] + c["E"])
    se2 = 0.5 * np.hypot(a["seE"], c["seE"])
    dd = b["E"] - p2
    sed = np.hypot(b["seE"], se2)
    better = "p=1 delivered the LOWER (better) energy" if dd < 0 else "p=2 delivered the lower (better) energy"
    print(f"Depth effect: E(p=1) - mean E(p=2) = {dd:+.4f} +- {sed:.4f} (z = {dd / sed:+.1f}) -> "
          + (better if abs(dd / sed) >= 2 else "difference within shot noise") )
    r1 = b["E"] / meta["ideal_E"][1]
    r2 = p2 / meta["ideal_E"][0]
    need = meta["ideal_E"][0] / meta["ideal_E"][1]
    print(f"Retention ratio p=1 / p=2: observed {r1 / r2:.2f}; needed to break even: {need:.2f}.")
    print("\nCaveats: one job on one day; the repeat shows only short-term repeatability, not day-to-day drift;")
    print("readout error is not mitigated here (it affects all three circuits alike); P(optimal) is noisy by")
    print("construction, so lean on the energy.")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default="hw6_pipeline.npz")
    ap.add_argument("--backend", default="ibm_marrakesh", help="backend name, or 'least_busy'")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--submit", action="store_true", help="send ONE job to real hardware (uses Open Plan time)")
    ap.add_argument("--retrieve", metavar="JOB_ID", help="fetch and analyze a job submitted earlier")
    ap.add_argument("--analyze-existing", metavar="NPZ", help="analyze counts saved by week5_hardware_run.py (free)")
    args = ap.parse_args()

    d = np.load(args.data)
    h, J = d["h"], d["J"]
    n = len(h)
    budget = int(d["budget"])
    best_x = np.array(d["best_x"], dtype=int)
    opt_idx = int(sum(int(best_x[i]) << i for i in range(n)))
    feasible = np.array([bin(k).count("1") == budget for k in range(2**n)])
    scale = float(np.sqrt(np.sum(h**2) + np.sum(J**2)))
    rng = np.random.default_rng(args.seed)
    sim2, sim1 = QAOASim(h, J, 2), QAOASim(h, J, 1)
    diag = sim2.diag
    span = float(diag.max() - diag.min())

    print("Finding parameters (numpy simulator)...")
    params2, e2 = rule_params(sim2, scale)
    params1, e1 = wide_search(sim1, scale, rng)
    ideal = []
    for sim, prm in ((sim2, params2), (sim1, params1), (sim2, params2)):
        pr = numpy_probs(sim, prm)
        ideal.append((float(pr @ diag), float(pr[opt_idx]), float(pr[feasible].sum())))
    print(f"  p=2 (constant-start rule): energy {e2:.4f}, P(optimal) {ideal[0][1]:.3f}, P(feasible) {ideal[0][2]:.3f}")
    print(f"  p=1 (wide search)        : energy {e1:.4f}, P(optimal) {ideal[1][1]:.3f}, P(feasible) {ideal[1][2]:.3f}")
    print(f"  instance energy span {span:.4f}; p=2 reaches {e2 / (diag.min()):.0%} and p=1 {e1 / (diag.min()):.0%} "
          f"of the ground-state energy ({diag.min():.4f}).")
    need = e2 / e1
    print(f"  Break-even: p=1 beats p=2 on hardware only if it retains at least {need:.2f}x as much of its ideal energy.")

    # ---- free analysis of an existing job ----
    if args.analyze_existing:
        hw = np.load(args.analyze_existing, allow_pickle=True)
        raw, mit = hw["raw_counts"].item(), hw["mitigated"].item()
        s_raw = count_stats(raw, diag, opt_idx, feasible)
        tot = sum(mit.values())
        pm = np.zeros(len(diag))
        for b, v in mit.items():
            pm[int(str(b).replace(" ", ""), 2)] += float(v) / tot
        e_m = float(pm @ diag)
        print(f"\nYour existing job ({str(hw['backend'])}, p=2): ideal energy {ideal[0][0]:.4f}")
        print(f"  raw counts      : energy {s_raw['E']:.4f} +- {s_raw['seE']:.4f} -> retention {s_raw['E'] / ideal[0][0]:.2f}; "
              f"P(optimal) {s_raw['Popt']:.3f}, P(feasible) {s_raw['Pfeas']:.3f}")
        print(f"  readout-mitigated: energy {e_m:.4f} -> retention {e_m / ideal[0][0]:.2f}; "
              f"P(optimal) {pm[opt_idx]:.3f}, P(feasible) {pm[feasible].sum():.3f}")
        lam = s_raw["E"] / ideal[0][0]
        print(f"\nWhat this implies for p=1: it needs a retention of at least {lam * need:.2f} (raw) to match your p=2 job;")
        print("a retention above 1.00 is impossible, so p=1 can only win if that number is below 1.00.")
        return

    # ---- Qiskit side ----
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

    def bind(ansatz, prm, p):
        binding = {}
        for q in ansatz.parameters:
            k = int("".join(ch for ch in q.name if ch.isdigit()))
            is_gamma = "\u03b3" in q.name or "gamma" in q.name.lower()
            binding[q] = prm[k] if is_gamma else prm[p + k]
        return ansatz.assign_parameters(binding)

    if args.retrieve:
        from qiskit_ibm_runtime import QiskitRuntimeService
        meta = dict(np.load(META_FILE, allow_pickle=True))
        meta = {k: (v.item() if v.shape == () else v) for k, v in meta.items()}
        service = QiskitRuntimeService()
        job = service.job(args.retrieve)
        print(f"Job {args.retrieve} status: {job.status()}")
        res = job.result()
        counts = [res[i].data.meas.get_counts() for i in range(3)]
        meta["job_id"] = args.retrieve
        report(meta, counts, diag, opt_idx, feasible)
        return

    circuits, p_list, prm_list = [], [2, 1, 2], [params2, params1, params2]
    ansatz = {1: QAOAAnsatz(cost_operator=H, reps=1), 2: QAOAAnsatz(cost_operator=H, reps=2)}
    for p, prm, sim in ((2, params2, sim2), (1, params1, sim1)):
        bound = bind(ansatz[p], prm, p)
        e_q = float(Statevector(bound).expectation_value(H).real)
        assert abs(e_q - sim.energy(prm)) < 1e-9, f"p={p}: numpy simulator and Qiskit disagree"
    print("Qiskit cross-check passed for both parameter sets (tolerance 1e-9).")

    if args.submit:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        service = QiskitRuntimeService()
        backend = (service.least_busy(operational=True, simulator=False, min_num_qubits=n)
                   if args.backend == "least_busy" else service.backend(args.backend))
        mode = f"REAL HARDWARE ({backend.name})"
    else:
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        backend = FakeMarrakesh()
        mode = "OFFLINE DRY RUN (noisy simulator of ibm_marrakesh; nothing is sent to IBM)"
    print(f"\nMode: {mode}")

    def build(p, prm):
        qc = bind(ansatz[p].decompose(reps=2), prm, p)
        qc.measure_all()
        return qc

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=42)
    t2 = pm.run(build(2, params2))
    layout = t2.layout.initial_index_layout(filter_ancillas=True)
    pm_same = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=42,
                                           initial_layout=layout)
    t1 = pm_same.run(build(1, params1))
    transpiled = [t2, t1, t2]
    depth = [t.depth() for t in transpiled]
    twoq = [sum(v for k, v in t.count_ops().items() if k in ("cx", "cz", "ecr", "rzz")) for t in transpiled]
    print(f"Same physical qubits for all circuits: {layout}")
    for i, t in enumerate(transpiled):
        print(f"  {NAMES[i]:14s} depth {depth[i]:4d}, two-qubit gates {twoq[i]:4d}")
    meta = dict(backend=backend.name, job_id="dry-run", shots=args.shots, depth=np.array(depth), twoq=np.array(twoq),
                ideal_E=np.array([x[0] for x in ideal]), ideal_Popt=np.array([x[1] for x in ideal]),
                ideal_Pfeas=np.array([x[2] for x in ideal]))

    if args.submit:
        sampler = SamplerV2(mode=backend)
        job = sampler.run(transpiled, shots=args.shots)
        meta["job_id"] = job.job_id()
        np.savez(META_FILE, **meta)
        print(f"\nSubmitted ONE job with 3 circuits x {args.shots} shots. Job ID: {meta['job_id']}")
        print(f"(Saved to {META_FILE}. If this window closes, fetch the results later with "
              f"--retrieve {meta['job_id']}.)")
        res = job.result()
        counts = [res[i].data.meas.get_counts() for i in range(3)]
        np.savez("week10_counts.npz", counts=np.array(counts, dtype=object), **{k: v for k, v in meta.items()})
    else:
        from qiskit_aer import AerSimulator
        simulator = AerSimulator.from_backend(backend)
        counts = [simulator.run(t, shots=args.shots, seed_simulator=args.seed + i).result().get_counts()
                  for i, t in enumerate(transpiled)]
    report(meta, counts, diag, opt_idx, feasible)


if __name__ == "__main__":
    main()
