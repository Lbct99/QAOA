"""
recover_layouts.py (v5) -- for every run in dashboard_data.json: fetch the submitted circuit from IBM, recover the
physical qubits, CZ edges, depth and gate counts, fetch the calibration at job time, and compute estimated success
probabilities (ESP) from (a) gate + readout errors and (b) NEW: decoherence while qubits sit idle (T1/T2 and the
scheduled circuit duration). Compare each with the measured retention.
v3: the calibration is looked up at the moment the job actually RAN (job.metrics() timestamps), not when it was created
(jobs can wait hours in the queue).
v4: adds prov['qubit_calib'], the calibrated readout error / T1 / T2 / 1Q-gate error for each measured qubit
individually (not just the geometric-mean ESP), so a single bad qubit shows up on its own instead of being
averaged away -- and lets you check whether a qubit found bad by analyse_qubits.py was ALSO flagged bad by the
official calibration (a known-bad qubit) or looked normal in the calibration (invisible, between-calibration drift).
v5: also logs prov['cz_gate_errors'], the calibrated CZ gate error for EVERY used coupler individually (not just
the geometric-mean esp_2q), so a persistently-biased qubit's own couplers can be checked on their own terms.
Writes dashboard_data_enriched.json (same format, richer "prov") and prints a table.
Read-only: uses no QPU time.   Run from C:/Users/loure/QAOA:   python recover_layouts.py
"""
import collections
import datetime
import json
import math
import os
import statistics

from qiskit_ibm_runtime import QiskitRuntimeService

SRC = "dashboard_data_enriched.json" if os.path.exists("dashboard_data_enriched.json") else "dashboard_data.json"
data = json.load(open(SRC))
ideal_E = data["ideal"]["E"]
service = QiskitRuntimeService()
backends, props_cache = {}, {}


def get_props(name, when):
    """Calibration at job time if IBM still serves it, else the latest one (flagged as such)."""
    key = (name, when.isoformat())
    if key in props_cache:
        return props_cache[key]
    if name not in backends:
        backends[name] = service.backend(name)
    b = backends[name]
    try:
        p = b.properties(datetime=when)
        if p is None:
            raise ValueError("no properties returned")
        res = (p, "historical")
    except Exception as e1:
        try:
            res = (b.properties(refresh=True), "current (not at job time)")
        except Exception as e2:
            res = (None, f"none ({type(e1).__name__}/{type(e2).__name__})")
    props_cache[key] = res
    return res


def analyse(circ):
    meas, edges, one_q = {}, collections.Counter(), collections.Counter()
    for inst in circ.data:
        op, qs = inst.operation, [circ.find_bit(q).index for q in inst.qubits]
        if op.name == "measure":
            meas[circ.find_bit(inst.clbits[0]).index] = qs[0]
        elif op.name in ("barrier", "delay", "rz"):
            continue
        elif len(qs) == 2:
            edges[tuple(sorted(qs))] += 1
        elif len(qs) == 1:
            one_q[(op.name, qs[0])] += 1
    return [meas[k] for k in sorted(meas)], edges, one_q


def gerr(props, name, qs):
    for order in (qs, qs[::-1]):
        try:
            return props.gate_error(name, list(order))
        except Exception:
            pass
    return None


def glen(props, name, qs):
    """Gate duration in seconds."""
    for order in (qs, qs[::-1]):
        try:
            v = props.gate_length(name, list(order))
            if v is not None:
                return float(v)
        except Exception:
            pass
    return None


def esp(props, measured, edges, one_q):
    l2 = l1 = lr = 0.0
    miss = 0
    for (a, b), c in edges.items():
        e = gerr(props, "cz", [a, b])
        if e is None or e >= 1:
            miss += 1
        else:
            l2 += c * math.log1p(-e)
    for (nm, q), c in one_q.items():
        e = gerr(props, nm, [q])
        if e is None or e >= 1:
            miss += 1
        else:
            l1 += c * math.log1p(-e)
    for q in measured:
        try:
            lr += math.log1p(-props.readout_error(q))
        except Exception:
            miss += 1
    return dict(esp_2q=math.exp(l2), esp_1q=math.exp(l1), esp_ro=math.exp(lr),
                esp=math.exp(l2 + l1 + lr), esp_missing=miss)


def per_edge_calibration(props, edges):
    """Calibrated CZ gate error for every used coupler individually, keyed 'a-b' (a<b, physical qubit ids)."""
    return {f"{a}-{b}": gerr(props, "cz", [a, b]) for (a, b) in edges}


def per_qubit_calibration(props, measured, one_q):
    """Calibrated readout error / T1 / T2 / 1Q-gate error for EACH measured qubit individually, in the same
    order as `measured` (which matches the instance's ticker order). A qubit that the official calibration
    already flagged as bad will stand out here on its own; a qubit that looks normal here but still produced
    an anomalous measured marginal (see analyse_qubits.py) points to drift the calibration snapshot missed."""
    out = []
    for q in measured:
        rec = dict(phys=int(q))
        try:
            rec["readout_error"] = float(props.readout_error(q))
        except Exception:
            rec["readout_error"] = None
        try:
            rec["t1_us"] = 1e6 * float(props.t1(q))
        except Exception:
            rec["t1_us"] = None
        try:
            rec["t2_us"] = 1e6 * float(props.t2(q))
        except Exception:
            rec["t2_us"] = None
        name1q = next((nm for (nm, qq) in one_q if qq == q), None)
        rec["gate1q_error"] = gerr(props, name1q, [q]) if name1q else None
        out.append(rec)
    return out


def decoherence(circ, props, measured, edges):
    """ASAP-schedule the circuit with the calibration's gate durations; every used qubit waits (idles) from its
    first gate until measurement. Idle error per qubit = idle * (1/(6*T1) + 1/(3*T2)) (first-order average
    infidelity of an idle qubit). Time spent inside gates is NOT counted again (gate errors include it)."""
    avail, busy, first = collections.defaultdict(float), collections.defaultdict(float), {}
    miss = 0
    for inst in circ.data:
        op, qs = inst.operation, [circ.find_bit(q).index for q in inst.qubits]
        if op.name == "measure":
            continue
        if op.name == "barrier":
            t = max(avail[q] for q in qs)
            for q in qs:
                avail[q] = t
            continue
        dur = 0.0 if op.name in ("rz", "delay") else glen(props, op.name, qs)
        if dur is None:
            miss += 1
            dur = 0.0
        start = max(avail[q] for q in qs)
        for q in qs:
            avail[q] = start + dur
            busy[q] += dur
            if dur > 0:
                first.setdefault(q, start)
    used = sorted(set(measured) | {q for e in edges for q in e})
    D = max(avail[q] for q in used)
    tot, idles, t1s, t2s = 0.0, [], [], []
    for q in used:
        idle = max((D - first[q]) - busy[q], 0.0) if q in first else 0.0
        idles.append(idle)
        try:
            t1, t2 = props.t1(q), props.t2(q)
            tot += idle * (1.0 / (6 * t1) + 1.0 / (3 * t2))
            t1s.append(t1)
            t2s.append(t2)
        except Exception:
            miss += 1
    return dict(esp_dec=math.exp(-tot), duration_us=D * 1e6, idle_us_mean=1e6 * sum(idles) / len(idles),
                t1_us_median=1e6 * statistics.median(t1s) if t1s else None,
                t2_us_median=1e6 * statistics.median(t2s) if t2s else None, dec_missing=miss)


rows = []
for r in data["runs"]:
    try:
        job = service.job(r["job_id"])
        circ = job.inputs["pubs"][0][0]
        try:
            stamps = job.metrics().get("timestamps", {})
        except Exception:
            stamps = {}

        def ts(k):
            v = stamps.get(k)
            return datetime.datetime.fromisoformat(v.replace("Z", "+00:00")) if v else None

        created = ts("created") or job.creation_date
        running = ts("running")
        when = running or created          # calibration at EXECUTION time
        queue_min = (running - created).total_seconds() / 60 if running else None
        measured, edges, one_q = analyse(circ)
        props, kind = get_props(r["backend"], when)
        ops = {k: int(v) for k, v in circ.count_ops().items()}
        prov = dict(r.get("prov") or {})
        prov.update(measured_qubits=measured,
                    active_2q_qubits=sorted({q for e in edges for q in e}),
                    cz_edges=[[a, b, c] for (a, b), c in sorted(edges.items())],
                    depth_recovered=circ.depth(), twoq_recovered=int(sum(edges.values())), ops=ops,
                    created_local=created.astimezone().strftime("%Y-%m-%d %H:%M"), calibration_kind=kind,
                    run_utc=running.strftime("%Y-%m-%d %H:%M:%S") if running else None, queue_min=queue_min)
        est = None
        if props is not None:
            if kind == "historical":
                prov["calibration"] = str(props.last_update_date)
            est = esp(props, measured, edges, one_q)
            est.update(decoherence(circ, props, measured, edges))
            est["esp_all"] = est["esp"] * est["esp_dec"]
            prov.update(est)
            prov["qubit_calib"] = per_qubit_calibration(props, measured, one_q)
            prov["cz_gate_errors"] = per_edge_calibration(props, edges)
        r["prov"] = prov
        rows.append(dict(r=r, est=est, kind=kind, ret=r["stats"]["E"] / ideal_E, ops=ops, when=when, queue=queue_min))
    except Exception as e:
        print(f"FAILED {r['backend']} {r['job_id']}: {type(e).__name__}: {e}")

print(f"\n{'time':17s} {'backend':14s} {'depth':>5s} {'dur us':>6s} {'T1med':>6s} {'T2med':>6s} {'ESP':>5s} {'dec':>5s} "
      f"{'ESPall':>6s} {'ret':>5s} {'ran UTC':>11s} {'queue':>6s}   sx/x/rz/cz   calib")
for x in sorted(rows, key=lambda x: x["when"]):
    e, o = x["est"] or {}, x["ops"]
    f = lambda k, d=3: f"{e[k]:.{d}f}" if e.get(k) is not None else "-"
    qs = "-" if x["queue"] is None else "%.1fm" % x["queue"]
    print(f"{x['r']['time']:17s} {x['r']['backend']:14s} {x['r']['prov']['depth_recovered']:5d} {f('duration_us', 1):>6s} "
          f"{f('t1_us_median', 0):>6s} {f('t2_us_median', 0):>6s} {f('esp'):>5s} {f('esp_dec'):>5s} {f('esp_all'):>6s} "
          f"{x['ret']:5.2f} {x['when'].strftime('%d %H:%M'):>11s} "
          f"{qs:>6s}   {o.get('sx', 0)}/{o.get('x', 0)}/{o.get('rz', 0)}/{o.get('cz', 0)}   {x['kind'][:10]}")

ok = [x for x in rows if x["est"] and "esp_all" in x["est"]]
print(f"\nRuns with ESP: {len(ok)} of {len(rows)}; calibration used: {dict(collections.Counter(x['kind'] for x in ok))}")
if ok:
    print("Missing calibration values per run (should be 0): "
          + str([(x['est'].get('esp_missing', 0), x['est'].get('dec_missing', 0)) for x in ok]))
if len(ok) >= 4:
    ys = [x["ret"] for x in ok]
    print(f"\nSpread: retention mean {statistics.mean(ys):.3f} SD {statistics.stdev(ys):.3f}")
    for k in ("esp", "esp_dec", "esp_all"):
        v = [x["est"][k] for x in ok]
        print(f"        {k:8s} mean {statistics.mean(v):.3f} SD {statistics.stdev(v):.3f}")
    try:
        from scipy.stats import pearsonr, spearmanr
        print("\nCorrelation with measured retention (across runs; replicate runs share ESP, so p-values are optimistic):")
        for k in ("esp_all", "esp_dec", "esp", "esp_2q", "esp_ro", "esp_1q"):
            xs = [x["est"][k] for x in ok]
            rho, p1 = spearmanr(xs, ys)
            pr, p2 = pearsonr(xs, ys)
            print(f"  {k:8s} Spearman {rho:+.2f} (p={p1:.2f})   Pearson {pr:+.2f} (p={p2:.2f})")
    except Exception as e:
        print("Could not compute correlations:", type(e).__name__, e)

with open("dashboard_data_enriched.json", "w") as f:
    json.dump(data, f, separators=(",", ":"))
print("\nSaved dashboard_data_enriched.json")
