"""
export_dashboard_data.py (v2) -- writes dashboard_data.json for the QAOA backend-benchmark dashboard.

Run from C:/Users/loure/QAOA (same folder as week11_backend_benchmark.py):
    python export_dashboard_data.py

Sources: week11_runs.jsonl + week11_counts.json (week 11 jobs), first_job_original.json (the original
week5 job, retrieved by job ID) and runs/*.json (week5_hardware_run.py runs with provenance).
It reuses functions of week11_backend_benchmark.py, so numbers match that script's own report.
Convention: index k = int(bitstring, 2); qubit 0 is the rightmost character. All times are local.
"""
import datetime
import glob
import json
import os

import numpy as np

import week11_backend_benchmark as w

FIRST_JOB_QUBITS = [31, 10, 11, 18, 13, 12]   # measured qubits of the original first job (from its run output)
FIRST_JOB_DEPTH, FIRST_JOB_TWOQ = 265, 109    # from the same run output

d = np.load("hw6_pipeline.npz")
h, J = d["h"], d["J"]
n = len(h)
budget = int(d["budget"])
best_x = np.array(d["best_x"], dtype=int)
opt_idx = int(sum(int(best_x[i]) << i for i in range(n)))
feasible = np.array([bin(k).count("1") == budget for k in range(2 ** n)])
scale = float(np.sqrt(np.sum(h ** 2) + np.sum(J ** 2)))
sim = w.QAOASim(h, J, 2)
diag = np.asarray(sim.diag, dtype=float)
params, _ = w.rule_params(sim, scale)
pr = np.asarray(w.numpy_probs(sim, params), dtype=float)
ideal_E = float(pr @ diag)


def own_energy(counts):
    """Independent energy from counts, to cross-check count_stats and the bit-order convention."""
    tot = sum(counts.values())
    p = np.zeros(2 ** n)
    for b, c in counts.items():
        p[int(b, 2)] += c / tot
    return float(p @ diag)


runs = []


def add(time, backend, job_id, depth, twoq, counts, note="", mitigated=None, source="week11", prov=None):
    if any(r["job_id"] == job_id for r in runs):
        return
    counts = {str(k): int(v) for k, v in counts.items()}
    s = w.count_stats(counts, diag, opt_idx, feasible)
    runs.append(dict(
        time=time, backend=backend, job_id=job_id, depth=depth, twoq=twoq, source=source,
        shots=int(sum(counts.values())), note=note, counts=counts, mitigated=mitigated, prov=prov,
        stats={k: float(v) for k, v in s.items()}, own_E=own_energy(counts)))


# 1. the original first job (retrieved by job ID)
if os.path.exists("first_job_original.json"):
    fj = json.load(open("first_job_original.json"))
    add(fj["created"][:16], "ibm_marrakesh", fj["job_id"], FIRST_JOB_DEPTH, FIRST_JOB_TWOQ, fj["counts"],
        note="week5 first job (original)", source="week5", prov=dict(measured_qubits=FIRST_JOB_QUBITS))

# 2. week5_hardware_run.py runs with provenance (time = when the record was written, i.e. job finished)
for fn in sorted(glob.glob(os.path.join("runs", "*.json"))):
    r = json.load(open(fn))
    t = datetime.datetime.fromisoformat(r["submitted_utc"]).astimezone().strftime("%Y-%m-%d %H:%M")
    add(t, r["backend"], r["job_id"], r.get("depth"), r.get("two_qubit_gates"), r["raw_counts"],
        note="week5_hardware_run", mitigated=r.get("mitigated"), source="week5",
        prov=dict(calibration=r.get("calibration_last_update"), measured_qubits=r.get("measured_qubits"),
                  active_2q_qubits=r.get("active_2q_qubits")))

# 3. fallback: the .npz, only if that job is not already in runs/
if os.path.exists("hw6_hardware_result.npz"):
    hw = np.load("hw6_hardware_result.npz", allow_pickle=True)
    t = datetime.datetime.fromtimestamp(os.path.getmtime("hw6_hardware_result.npz")).strftime("%Y-%m-%d %H:%M")
    add(t, str(hw["backend"]), str(hw["job_id"]), None, None, hw["raw_counts"].item(),
        note="week5_hardware_run (from .npz)", source="week5",
        mitigated={str(k): float(v) for k, v in hw["mitigated"].item().items()})

# 4. week 11 jobs
cache = w.load_counts()
for e in w.read_log():
    c = cache.get(e["job_id"])
    if c is None:
        print(f"not retrieved yet, skipped: {e['backend']} {e['job_id']} "
              f"(run: python week11_backend_benchmark.py --retrieve)")
        continue
    add(e["time"].replace("T", " ")[:16], e["backend"], e["job_id"], e.get("depth"), e.get("twoq"),
        c, e.get("note", ""))

runs.sort(key=lambda r: r["time"])

out = dict(
    generated=datetime.datetime.now().isoformat(timespec="seconds"),
    instance=dict(
        tickers=[str(t) for t in d["tickers"]], n=n, budget=budget,
        bit_order="index k = int(bitstring, 2); qubit 0 is the rightmost character",
        energies=diag.tolist(), feasible=feasible.tolist(), opt_idx=opt_idx,
        e_uniform=float(diag.mean()), e_min=float(diag.min()),
        e_min_feasible=float(diag[feasible].min())),
    ideal=dict(E=ideal_E, p_opt=float(pr[opt_idx]), p_feas=float(pr[feasible].sum()), probs=pr.tolist()),
    runs=runs)
with open("dashboard_data.json", "w") as f:
    json.dump(out, f, separators=(",", ":"))

# ---- check table: compare with the numbers of your own report ----
print(f"\nideal E {ideal_E:.4f} | E_uniform {diag.mean():.4f} | E_min {diag.min():.4f} "
      f"| E_min feasible {diag[feasible].min():.4f}")
print(f"classical optimum {opt_idx:0{n}b} has E {diag[opt_idx]:.4f}; lowest of all {2 ** n}: "
      f"{int(diag.argmin()):0{n}b}")
print(f"\n{'time':17s} {'backend':15s} {'depth':>5s} {'E (script)':>11s} {'E (own)':>9s} {'retention':>10s} source")
for r in runs:
    flag = "" if abs(r["stats"]["E"] - r["own_E"]) < 1e-9 else "  <-- MISMATCH"
    dep = "" if r["depth"] is None else str(r["depth"])
    print(f"{r['time']:17s} {r['backend']:15s} {dep:>5s} {r['stats']['E']:11.4f} {r['own_E']:9.4f} "
          f"{r['stats']['E'] / ideal_E:10.2f} {r['source']}{flag}")
print("\nProvenance (week5 runs):")
for r in runs:
    if r["prov"]:
        print(f"  {r['time']} {r['backend']} {r['job_id']} depth {r['depth']}: {r['prov']}")
print(f"\nSaved dashboard_data.json ({os.path.getsize('dashboard_data.json') / 1024:.1f} KB, {len(runs)} runs)")
