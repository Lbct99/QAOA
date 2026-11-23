"""
simulate_circuits.py -- simulate the circuits that were ACTUALLY submitted to IBM (not the idealised ones).

For every run in dashboard_data_enriched.json (or dashboard_data.json): fetch the submitted circuit from the IBM
job (and cache it in circuits/<job_id>.qpy, because IBM only keeps jobs for a limited time), simulate it noise-free
on the qubits it uses, and compare:
  * its ideal energy / P(opt) / P(feasible) with the shared reference ideal (E = -0.1477),
  * runs that used the same qubits (e.g. week5 vs week11 circuits), via total variation distance,
  * the measured distribution with the simulated one of the SAME circuit (TVD, Hellinger fidelity).
Writes dashboard_data_enriched.json (adds prov["sim"] and prov["sim_probs"]). Read-only: no QPU time.
Run from C:/Users/loure/QAOA:   python simulate_circuits.py
"""
import collections
import itertools
import json
import os

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector

SRC = "dashboard_data_enriched.json" if os.path.exists("dashboard_data_enriched.json") else "dashboard_data.json"
data = json.load(open(SRC))
I = data["instance"]
E = np.array(I["energies"], dtype=float)
feas = np.array(I["feasible"], dtype=bool)
opt, n, N = I["opt_idx"], I["n"], 2 ** I["n"]
ref = np.array(data["ideal"]["probs"], dtype=float)
ideal_E = data["ideal"]["E"]
os.makedirs("circuits", exist_ok=True)
service = None


def get_circuit(jid):
    global service
    path = os.path.join("circuits", f"{jid}.qpy")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return qpy.load(f)[0]
    if service is None:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
    pub = service.job(jid).inputs["pubs"][0]
    circ = pub[0]
    if circ.parameters:                       # bind parameter values if the circuit was submitted unbound
        circ = circ.assign_parameters(np.asarray(pub[1], dtype=float).reshape(-1))
    with open(path, "wb") as f:
        qpy.dump(circ, f)
    return circ


def simulate(circ):
    """Noise-free distribution over the measured bits. Index k = int(bitstring, 2): bit i = clbit i."""
    meas = {}
    for inst in circ.data:
        if inst.operation.name == "measure":
            meas[circ.find_bit(inst.clbits[0]).index] = circ.find_bit(inst.qubits[0]).index
    measured = [meas[k] for k in sorted(meas)]
    skip = ("measure", "barrier", "delay")
    used = sorted({circ.find_bit(q).index for inst in circ.data if inst.operation.name not in skip
                   for q in inst.qubits} | set(measured))
    if len(used) > 16:
        raise ValueError(f"{len(used)} qubits used; too many for a statevector")
    idx = {q: i for i, q in enumerate(used)}
    sub = QuantumCircuit(len(used))
    for inst in circ.data:
        if inst.operation.name in skip:
            continue
        sub.append(inst.operation, [idx[circ.find_bit(q).index] for q in inst.qubits])
    p = np.asarray(Statevector(sub).probabilities(qargs=[idx[q] for q in measured]), dtype=float)
    return p, measured, len(used)


def tvd(a, b):
    return 0.5 * float(np.abs(a - b).sum())


def hfid(a, b):
    return float(np.sum(np.sqrt(a * b)) ** 2)


rows = []
for r in data["runs"]:
    try:
        circ = get_circuit(r["job_id"])
        p, measured, nq = simulate(circ)
        if len(p) != N:
            raise ValueError(f"{len(measured)} measured bits, expected {n}")
        pm = np.zeros(N)
        tot = sum(r["counts"].values())
        for b, c in r["counts"].items():
            pm[int(b, 2)] += c / tot
        Em, Es = float(pm @ E), float(p @ E)
        sim = dict(E=Es, p_opt=float(p[opt]), p_feas=float(p[feas].sum()), tvd_ref=tvd(p, ref),
                   tvd_meas=tvd(pm, p), hellinger=hfid(pm, p), ret_own=Em / Es)
        prov = dict(r.get("prov") or {})
        prov["sim"], prov["sim_probs"] = sim, [round(float(x), 6) for x in p]
        prov.setdefault("measured_qubits", measured)
        r["prov"] = prov
        rows.append(dict(r=r, p=p, sim=sim, measured=tuple(measured), nq=nq, ret_ref=Em / ideal_E))
    except Exception as e:
        print(f"FAILED {r['backend']} {r['job_id']}: {type(e).__name__}: {e}")

print(f"\nReference ideal: E {ideal_E:.4f}, P(opt) {ref[opt]:.3f}, P(feas) {ref[feas].sum():.3f}\n")
print(f"{'time':17s} {'backend':14s} {'nq':>2s} {'E_sim':>8s} {'P(opt)':>6s} {'P(feas)':>7s} {'TVD ref':>7s} | "
      f"{'ret(ref)':>8s} {'ret(own)':>8s} {'TVD m-s':>7s} {'HellFid':>7s}")
for x in sorted(rows, key=lambda x: x["r"]["time"]):
    s = x["sim"]
    print(f"{x['r']['time']:17s} {x['r']['backend']:14s} {x['nq']:2d} {s['E']:8.4f} {s['p_opt']:6.3f} {s['p_feas']:7.3f} "
          f"{s['tvd_ref']:7.4f} | {x['ret_ref']:8.2f} {s['ret_own']:8.2f} {s['tvd_meas']:7.3f} {s['hellinger']:7.3f}")
if rows:
    print(f"\nMax |E_sim - reference E| = {max(abs(x['sim']['E'] - ideal_E) for x in rows):.4f}; "
          f"max TVD to reference = {max(x['sim']['tvd_ref'] for x in rows):.4f}")

print("\nRuns on the same backend AND the same measured qubits (do the circuits make the same state?):")
groups = collections.defaultdict(list)
for x in rows:
    groups[(x["r"]["backend"], x["measured"])].append(x)
for (b, q), g in groups.items():
    for a, c in itertools.combinations(g, 2):
        print(f"  {b} {list(q)}: {a['r']['time']} vs {c['r']['time']}: dE_sim {a['sim']['E'] - c['sim']['E']:+.4f}, "
              f"TVD(sim,sim) {tvd(a['p'], c['p']):.4f} | measured retention {a['ret_ref']:.2f} vs {c['ret_ref']:.2f}, "
              f"Hellinger {a['sim']['hellinger']:.3f} vs {c['sim']['hellinger']:.3f}")

ok = [x for x in rows if x["r"].get("prov", {}).get("esp_all") is not None]
if len(ok) >= 4:
    try:
        from scipy.stats import pearsonr, spearmanr
        print("\nDoes the calibration-based ESP track a noise measure that does not depend on the energy?")
        for tgt, name in (("hellinger", "Hellinger fidelity"), ("ret_own", "retention (own circuit)")):
            for k in ("esp_all", "esp", "esp_2q"):
                xs, ys = [x["r"]["prov"][k] for x in ok], [x["sim"][tgt] for x in ok]
                print(f"  {k:8s} vs {name:24s} Spearman {spearmanr(xs, ys)[0]:+.2f}   Pearson {pearsonr(xs, ys)[0]:+.2f}")
    except Exception as e:
        print("Could not compute correlations:", type(e).__name__, e)

with open("dashboard_data_enriched.json", "w") as f:
    json.dump(data, f, separators=(",", ":"))
print("\nSaved dashboard_data_enriched.json (circuits cached in circuits/)")
