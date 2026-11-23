"""
analyse_qubits.py -- per-qubit diagnostics: is the retention drop localised to one qubit/pair, or global?

Reads dashboard_data_enriched.json (falls back to dashboard_data.json) -- no IBM calls, no QPU time.
For every run: measured vs ideal marginal P(bit_i=1) per logical qubit (bit i <-> tickers[i] <-> the physical
qubit in prov["measured_qubits"][i]), and the measured vs ideal correlation on every CZ-coupled pair (does the
pair's joint land off in a way lone marginals don't explain -- a sign of a correlated/two-qubit error).
Then, for runs on the SAME backend with the SAME physical qubits, it compares bit-by-bit and pair-by-pair to
localise a drop (e.g. the 27 Oct marrakesh dip) to specific qubits/pairs rather than "everything is worse".
Needs prov["measured_qubits"], prov["cz_edges"] and prov["sim_probs"] (from simulate_circuits.py) per run.
v2: also shows the ABSOLUTE measured vs ideal P(bit=1) per qubit (not just the delta between two runs), so you
can see whether a flagged qubit went to near-1 (stuck-at) or just got noisier -- and, where recover_layouts.py
(v4+) logged prov["qubit_calib"], prints the calibrated readout error / T1 / T2 for that qubit next to it: a high
calibrated readout error means the official calibration already knew about it (a known-bad qubit); a normal
calibrated value next to an anomalous measured marginal means the problem happened BETWEEN calibrations
(invisible drift).
Run from C:/Users/loure/QAOA:   python analyse_qubits.py
v3: the calibration section no longer needs prov['sim_probs'] -- it only needs measured_qubits + qubit_calib
(both written by recover_layouts.py alone), so it shows even if simulate_circuits.py has not been (re-)run.
v4: next to each pair's measured/ideal correlation, also prints the coupler's own calibrated CZ gate error (prov['cz_gate_errors'], from recover_layouts.py v5+), so a persistently-biased qubit's coupler can be checked on its own terms rather than inferred only from the aggregate 2Q-ESP.
"""
import itertools
import json
import os

import numpy as np

SRC = "dashboard_data_enriched.json" if os.path.exists("dashboard_data_enriched.json") else "dashboard_data.json"
data = json.load(open(SRC))
I = data["instance"]
tickers, n, N = I["tickers"], I["n"], 1 << I["n"]
bits = [np.array([(k >> i) & 1 for k in range(N)]) for i in range(n)]  # bit i of index k


def dist_from_counts(counts):
    tot = sum(counts.values())
    p = np.zeros(N)
    for b, c in counts.items():
        p[int(b, 2)] += c / tot
    return p


def marginals(p):
    return np.array([float(p @ bits[i]) for i in range(n)])


def pair_joint(p, i, j):
    return float(p @ (bits[i] * bits[j]))


rows = []
cal_only = []
skipped = []
for r in data["runs"]:
    prov = r.get("prov") or {}
    mq, sp = prov.get("measured_qubits"), prov.get("sim_probs")
    edges = prov.get("cz_edges")
    if mq and prov.get("qubit_calib"):
        cal_only.append(dict(r=r, mq=mq, calib=prov["qubit_calib"],
                             ret=r["stats"]["E"] / data["ideal"]["E"]))
    if not mq or sp is None:
        skipped.append(r["job_id"])
        continue
    pm = dist_from_counts(r["counts"])
    ps = np.array(sp, dtype=float)
    mm, ms = marginals(pm), marginals(ps)
    phys2bit = {q: i for i, q in enumerate(mq)}
    pairs = {}
    if edges:
        for a, b, _c in edges:
            if a in phys2bit and b in phys2bit:
                i, j = phys2bit[a], phys2bit[b]
                pairs[(i, j)] = dict(meas=pair_joint(pm, i, j), ideal=pair_joint(ps, i, j),
                                     meas_indep=mm[i] * mm[j], ideal_indep=ms[i] * ms[j])
    rows.append(dict(r=r, mq=mq, mm=mm, ms=ms, pairs=pairs, ret=r["stats"]["E"] / data["ideal"]["E"]))

if skipped:
    print(f"Skipped (no prov/measured_qubits/sim_probs -- run recover_layouts.py and simulate_circuits.py first): "
          f"{skipped}")

print(f"\nPer-qubit ABSOLUTE P(bit=1): measured / ideal, ticker order {tickers}:\n")
print(f"{'time':17s} {'backend':14s} {'ret':>5s}  " + "  ".join(f"{t:>13s}" for t in tickers))
for x in sorted(rows, key=lambda x: x["r"]["time"]):
    print(f"{x['r']['time']:17s} {x['r']['backend']:14s} {x['ret']:5.2f}  "
          + "  ".join(f"{m:5.3f}/{s:5.3f}" for m, s in zip(x["mm"], x["ms"])))

print(f"\nPer-qubit marginal error (measured - ideal P(bit=1)), ticker order {tickers}:\n")
print(f"{'time':17s} {'backend':14s} {'ret':>5s}  " + "  ".join(f"{t:>6s}" for t in tickers))
for x in sorted(rows, key=lambda x: x["r"]["time"]):
    d = x["mm"] - x["ms"]
    print(f"{x['r']['time']:17s} {x['r']['backend']:14s} {x['ret']:5.2f}  " + "  ".join(f"{v:+6.3f}" for v in d))

if cal_only:
    print(f"\nCalibrated readout error / T1 / T2 per qubit (from recover_layouts.py), ticker order {tickers}:\n")
    for x in sorted(cal_only, key=lambda x: x["r"]["time"]):
        print(f"  {x['r']['time']} {x['r']['backend']} (ret {x['ret']:.2f}):")
        for t, c in zip(tickers, x["calib"]):
            ro = "-" if c["readout_error"] is None else f"{c['readout_error']:.3f}"
            t1 = "-" if c["t1_us"] is None else f"{c['t1_us']:.0f}"
            t2 = "-" if c["t2_us"] is None else f"{c['t2_us']:.0f}"
            print(f"    {t:6s} phys {c['phys']:3d}: readout_error {ro:>5s}  T1 {t1:>4s}us  T2 {t2:>4s}us")
else:
    print("\n(no prov['qubit_calib'] found -- run the updated recover_layouts.py first for the calibration comparison)")

print("\nSame backend, same physical qubits -- bit-by-bit comparison (biggest-mover qubit highlighted):")
groups = {}
for x in rows:
    groups.setdefault((x["r"]["backend"], tuple(x["mq"])), []).append(x)
for (b, q), g in groups.items():
    if len(g) < 2:
        continue
    print(f"\n  {b}  qubits {list(q)} (ticker order {tickers})")
    for a, c in itertools.combinations(sorted(g, key=lambda x: x["r"]["time"]), 2):
        d = a["mm"] - c["mm"]  # same ideal circuit assumed (checked by simulate_circuits.py TVD), so compare measured directly
        worst = int(np.argmax(np.abs(d)))
        tag = "localised to one qubit" if np.abs(d[worst]) > 2 * np.median(np.abs(d)) else "spread across qubits"
        print(f"    {a['r']['time']} ({a['ret']:.2f}) vs {c['r']['time']} ({c['ret']:.2f}): "
              f"delta P(1) = " + ", ".join(f"{tickers[i]} {d[i]:+.3f}" for i in range(n))
              + f"  -> largest: {tickers[worst]} ({d[worst]:+.3f}), {tag}")

print("\nPairwise (CZ-coupled) correlation: measured P(both=1) minus what independence would predict,")
print("ideal column shows the same for the noise-free circuit (near 0 only if the ideal state itself is product-like):")
for x in sorted(rows, key=lambda x: x["r"]["time"]):
    if not x["pairs"]:
        continue
    print(f"\n  {x['r']['time']} {x['r']['backend']} (ret {x['ret']:.2f}):")
    cz = (x["r"].get("prov") or {}).get("cz_gate_errors") or {}
    for (i, j), v in x["pairs"].items():
        edge = next((cz[k] for k in (f"{x['mq'][i]}-{x['mq'][j]}", f"{x['mq'][j]}-{x['mq'][i]}") if k in cz), None)
        ge = "-" if edge is None else f"{edge:.4f}"
        print(f"    {tickers[i]}-{tickers[j]}: measured corr {v['meas'] - v['meas_indep']:+.4f}  "
              f"ideal corr {v['ideal'] - v['ideal_indep']:+.4f}  cz_gate_error {ge:>6s}")

with open("dashboard_data_enriched.json", "w") as f:
    json.dump(data, f, separators=(",", ":"))
