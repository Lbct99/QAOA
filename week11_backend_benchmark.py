"""
Week 11 -- Backend benchmark and day-on-day monitor (version 2).

Your pipeline picks a backend with `least_busy`. That only makes sense if devices and days behave alike.
This script runs the SAME p=2 circuit (the constant-start rule's parameters, for the 6-stock problem you
already ran) as ONE JOB PER BACKEND (one circuit, 4096 shots) and keeps a permanent record, so you can repeat
it day after day.

Metric: measured energy <H_C> over all counts. Noise pulls it toward 0 (the uniform mixture has energy
exactly 0), so retention = measured energy / ideal energy; one run's standard error is about 1%.

Version 3 also logs, at submission and at no quantum cost, the physical qubits each circuit lands on and the
backend's own reported error rates (two-qubit, single-qubit, readout, T1/T2), turns them into a predicted fidelity,
and tests in --retrieve whether that free number predicts the day's measured retention.

Version 2 analyses by SESSION (one backend on one day), because runs submitted together share that day's
calibration and layout and are NOT independent evidence about the device:
    * same-session repeats        -> are two back-to-back runs within shot noise of each other?
    * between-session spread      -> how much does retention move from day to day on one backend?
    * a day-by-day table          -> which backend was best on which day, and does that persist?
    * paired comparisons by day   -> backend A minus backend B on the days both ran
It also checks where the "first job" baseline comes from, and (if your runtime version provides it) records
the quantum seconds each job used and the day it actually ran.

Files (keep them; commit them to git):
    week11_runs.jsonl    one line per submitted job: time, backend, job id, depth, gates, note
    week11_counts.json   measured counts of every finished job, cached on first retrieval
                         (IBM keeps job results only for a limited time)
    week11_info.json     quantum seconds and run times per job, when the API provides them

SAFETY. The default is a free OFFLINE DRY RUN on noisy simulators. Nothing is sent to IBM unless you pass
--submit. Open Plan time is counted in seconds of quantum time (--retrieve reports what your jobs used).

Modes:
    python week11_backend_benchmark.py --list-backends
    python week11_backend_benchmark.py --backends ibm_marrakesh ibm_marrakesh ibm_fez ibm_kingston   # free dry run
    python week11_backend_benchmark.py --submit --backends ibm_marrakesh ibm_marrakesh ibm_fez ibm_kingston
    python week11_backend_benchmark.py --retrieve                        # fetch finished jobs, analyze everything
Options: --note "text", --shots 4096, --max-jobs 4, --data hw6_pipeline.npz, --baseline hw6_hardware_result.npz,
         --no-info (skip the usage/run-time lookups), --no-manual-first-job
"""

import os

# Tiny problems: single-threaded BLAS is fastest. Must be set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import itertools
import json
import re
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
from scipy.stats import chi2 as chi2_dist

from qaoa_instance_generator import QAOASim
from week10_depth_experiment import rule_params, numpy_probs, count_stats

LOG_FILE = "week11_runs.jsonl"
COUNTS_FILE = "week11_counts.json"
INFO_FILE = "week11_info.json"
DEFAULT_BACKENDS = ["ibm_marrakesh", "ibm_fez", "ibm_kingston"]   # what --list-backends showed for your account

# Your original first job (job ID from the day you ran week5_hardware_run.py) and its numbers as analyzed on the
# day. hw6_hardware_result.npz is overwritten by every run of week5_hardware_run.py, so it may hold a later job.
KNOWN_FIRST_JOB_ID = "dav4pvbojkfs738t6dhg"
MANUAL_FIRST_JOB = dict(E=-0.1178, seE=0.0014, Popt=0.100, sePopt=0.0047, Pfeas=0.485, sePfeas=0.0078)

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
XID_ALPHABET = "0123456789abcdefghijklmnopqrstuv"


def job_id_time(job_id):
    """IBM job IDs are xid identifiers (20 characters, base32hex): the first 4 bytes are the creation time.
    Validated on your jobs: the queued Kingston job decodes to exactly its logged submission time, and the original
    first job to 2026-10-01 12:07 UTC. Returns a local datetime, or None if the ID does not look like an xid."""
    jid = str(job_id).lower()
    if len(jid) != 20 or any(c not in XID_ALPHABET for c in jid):
        return None
    n = 0
    for c in jid:
        n = n * 32 + XID_ALPHABET.index(c)
    ts = int.from_bytes((n >> 4).to_bytes(12, "big")[:4], "big")      # the encoding pads 96 bits to 100
    if not 1.5e9 < ts < 2.2e9:
        return None
    return datetime.fromtimestamp(ts).astimezone()


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


def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def circuit_usage(t):
    """Physical qubits and operations used by a transpiled circuit."""
    one_q, two_q, measured = Counter(), Counter(), {}
    for inst in t.data:
        op = inst.operation.name
        if op == "barrier":
            continue
        qs = tuple(t.find_bit(q).index for q in inst.qubits)
        if op == "measure":
            measured[t.find_bit(inst.clbits[0]).index] = qs[0]
        elif len(qs) == 1:
            one_q[(op, qs[0])] += 1
        elif len(qs) == 2:
            two_q[(op, qs)] += 1
    return one_q, two_q, measured


def op_error(target, op, qargs):
    try:
        props = target[op][qargs]
        return None if props is None or props.error is None else float(props.error)
    except Exception:
        return None


def predict_fidelity(t, backend):
    """The backend's own reported error rates, applied to THIS transpiled circuit. Costs no quantum time.
    pred_fid = (product over two-qubit gates of 1 - error) x (same for single-qubit gates) x (product over the six
    measured qubits of 1 - readout error). A crude no-idle-decoherence estimate; T1/T2 are logged for later."""
    target = backend.target
    one_q, two_q, measured = circuit_usage(t)
    missing, f2, f1, fr = 0, 1.0, 1.0, 1.0
    edges, ro = [], []
    for (op, qs), cnt in two_q.items():
        e = op_error(target, op, qs)
        if e is None:
            e = op_error(target, op, qs[::-1])
        if e is None:
            missing += cnt
            continue
        f2 *= (1.0 - e) ** cnt
        edges.append([qs[0], qs[1], op, cnt, e])
    for (op, q), cnt in one_q.items():
        e = op_error(target, op, (q,))
        if e is None:
            missing += cnt
            continue
        f1 *= (1.0 - e) ** cnt
    for _, q in sorted(measured.items()):
        e = op_error(target, "measure", (q,))
        ro.append(e)
        if e is not None:
            fr *= 1.0 - e
    used = sorted(set(measured.values()) | {q for _, q in one_q} | {q for _, qs in two_q for q in qs})
    t1, t2 = [], []
    for q in used:
        try:
            qp = target.qubit_properties[q]
            if qp.t1:
                t1.append(float(qp.t1) * 1e6)
            if qp.t2:
                t2.append(float(qp.t2) * 1e6)
        except Exception:
            pass
    cal_time = None
    try:
        cal_time = str(backend.properties().last_update_date)
    except Exception:
        pass
    med = lambda v: float(np.median(v)) if len(v) else None
    return dict(physical_qubits=[q for _, q in sorted(measured.items())], used_qubits=used, edges=edges,
                two_q_fid=float(f2), one_q_fid=float(f1), readout_fid=float(fr), pred_fid=float(f2 * f1 * fr),
                median_2q_error=med([e[4] for e in edges]), max_2q_error=max([e[4] for e in edges], default=None),
                median_readout_error=med([e for e in ro if e is not None]), median_t1_us=med(t1), median_t2_us=med(t2),
                missing_errors=int(missing), calibration_time=cal_time)


def job_info(job):
    """Quantum seconds and run times, if this runtime version provides them. Never raises."""
    info = {"tried": True}
    try:
        info["quantum_seconds"] = float(job.usage())
    except Exception:
        pass
    try:
        m = job.metrics()
        if isinstance(m, dict):
            ts = m.get("timestamps") or {}
            if ts.get("running"):
                info["started"] = str(ts["running"])
            if ts.get("finished"):
                info["finished"] = str(ts["finished"])
            qs = (m.get("usage") or {}).get("quantum_seconds")
            if qs is not None and "quantum_seconds" not in info:
                info["quantum_seconds"] = float(qs)
    except Exception:
        pass
    return info


def get_service(state):
    if state["service"] is None:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            state["service"] = QiskitRuntimeService()
        except Exception as e:
            state["service"] = False
            print(f"  (could not connect to IBM: {type(e).__name__}; using cached data only)")
    return state["service"]


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
    print("\nCalibration predictor on the offline models (free; the simulated noise comes from the same numbers, so this")
    print("is the BEST case for the predictor: real devices add drift and effects the error rates do not capture):")
    print(f"  {'run':22s} {'qubits':26s} {'2Q fid':>7s} {'1Q fid':>7s} {'readout':>8s} {'predicted':>10s} {'simulated retention':>20s}")
    for r in rows:
        c = r["calib"]
        print(f"  {r['name']:22s} {str(c['physical_qubits']):26s} {c['two_q_fid']:7.3f} {c['one_q_fid']:7.3f} "
              f"{c['readout_fid']:8.3f} {c['pred_fid']:10.3f} {r['stats']['E'] / ideal_E:20.2f}")
    by = {}
    for r in rows:
        by.setdefault(r["name"].split(" #")[0], []).append(r)
    for nm, rs in by.items():
        for a, b in zip(rs[:-1], rs[1:]):
            d = a["stats"]["E"] - b["stats"]["E"]
            se = np.hypot(a["stats"]["seE"], b["stats"]["seE"])
            print(f"\nSame-session repeat on {nm}: {a['name']} - {b['name']} = {d:+.4f} +- {se:.4f} (z = {d / se:+.1f}) -> "
                  + ("consistent with shot noise" if abs(d / se) < 2 else "more than shot noise"))


def _sessions(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["backend"], r["session"], r.get("variant", "rule")), []).append(r)
    out = []
    for (b, s, v), rs in groups.items():
        E = np.array([r["stats"]["E"] for r in rs])
        se = np.array([r["stats"]["seE"] for r in rs])
        w = 1.0 / se**2
        out.append(dict(backend=b, session=s, variant=v, n=len(rs), E=float((w * E).sum() / w.sum()),
                        se=float(np.sqrt(1.0 / w.sum())), runs=rs))
    out.sort(key=lambda s: (DATE_RE.match(s["session"]) is not None, s["session"]))
    return out


def series_report(rows, ideal_E, ideal_Popt, ideal_Pfeas):
    sign = abs(ideal_E)
    print(f"\nIdeal (no noise): energy {ideal_E:.4f}, P(optimal) {ideal_Popt:.3f}, P(feasible) {ideal_Pfeas:.3f}\n")
    print(f"{'submitted':25s} {'ran':10s} {'backend':14s} {'circuit':13s} {'depth':>5s} {'2Q':>4s} {'measured E (+- SE)':>19s} "
          f"{'ret.':>5s} {'P(opt)':>7s} {'P(feas)':>8s} {'q-sec':>6s}")
    for r in rows:
        st = r["stats"]
        ran = r["session"] if DATE_RE.match(r["session"]) else ""
        dep = "" if r["depth"] is None else str(r["depth"])
        tq = "" if r["twoq"] is None else str(r["twoq"])
        qs = "" if r.get("qsec") is None else f"{r['qsec']:.1f}"
        print(f"{r['submitted']:25s} {ran:10s} {r['backend']:14s} {r.get('variant', 'rule'):13s} {dep:>5s} {tq:>4s} "
              f"{st['E']:10.4f} +- {st['seE']:.4f} {st['E'] / ideal_E:5.2f} {st['Popt']:7.3f} {st['Pfeas']:8.3f} {qs:>6s}")

    sess = _sessions(rows)
    ret = lambda x: x["E"] / ideal_E
    rule_sess = [x for x in sess if x["variant"] == "rule"]

    # --- same-session repeats (same circuit, same backend, same day) ---
    zs = []
    for x in sess:
        rs = x["runs"]
        for a_, b_ in zip(rs[:-1], rs[1:]):
            zs.append((x["backend"], x["session"], (a_["stats"]["E"] - b_["stats"]["E"]) /
                       np.hypot(a_["stats"]["seE"], b_["stats"]["seE"])))
    if zs:
        worst = max(abs(z) for _, _, z in zs)
        print(f"\nSame-session repeats (back-to-back runs of one circuit on one backend): {len(zs)} pair(s), z = "
              + ", ".join(f"{z:+.1f}" for _, _, z in zs)
              + (" -> all within shot noise: a session is internally consistent." if worst < 2
                 else " -> at least one pair differs by more than shot noise."))

    # --- per backend, over benchmark sessions ---
    by_backend = {}
    for x in rule_sess:
        by_backend.setdefault(x["backend"], []).append(x)
    print("\nPer backend, over benchmark sessions (a session = one backend on one day; each counts once):")
    summary = {}
    for b, ss in by_backend.items():
        E = np.array([x["E"] for x in ss])
        se = np.array([x["se"] for x in ss])
        n = len(ss)
        r = E / ideal_E
        line = f"  {b:14s} {n} session(s): mean retention {r.mean():.3f}"
        if n >= 2:
            w = 1.0 / se**2
            mean_w = (w * E).sum() / w.sum()
            pval = float(chi2_dist.sf(float((w * (E - mean_w) ** 2).sum()), n - 1))
            sd = float(r.std(ddof=1))
            shot = float(np.sqrt((se**2).mean()) / sign)
            line += (f", between-session SD {sd:.3f} (shot noise alone: {shot:.3f}), chi-square p = {pval:.3f} -> "
                     + (f"retention drifts between sessions by about {sd:.2f} (1 sigma)" if pval < 0.05
                        else "consistent with shot noise (weak test with so few sessions)"))
            summary[b] = (float(r.mean()), sd / np.sqrt(n))
        else:
            summary[b] = (float(r.mean()), float(se[0] / sign))
            line += " (a single session)"
        print(line)

    # --- day-by-day table (benchmark circuit only) ---
    dates = sorted({x["session"] for x in rule_sess if DATE_RE.match(x["session"])})
    backends = list(by_backend)
    if dates:
        print("\nRetention by day (benchmark circuit; mean of that day's runs; '-' = no run that day):")
        print(f"  {'day':12s}" + "".join(f"{b:>15s}" for b in backends))
        table = {(x["backend"], x["session"]): x for x in rule_sess}
        for d in dates:
            cells = []
            for b in backends:
                x = table.get((b, d))
                cells.append("-" if x is None else f"{ret(x):.2f}" + (f" (x{x['n']})" if x["n"] > 1 else ""))
            print(f"  {d:12s}" + "".join(f"{c:>15s}" for c in cells))
        best = []
        for d in dates:
            cand = [(ret(table[(b, d)]), b) for b in backends if (b, d) in table]
            if len(cand) >= 2:
                best.append((d, max(cand)[1]))
        if best:
            print("  best backend by day: " + "; ".join(f"{d} {b}" for d, b in best))
            if len(best) >= 3:
                changes = sum(1 for (_, a_), (_, c_) in zip(best[:-1], best[1:]) if a_ != c_)
                print(f"  the best backend changed on {changes} of {len(best) - 1} day-to-day transitions"
                      + (" -> yesterday's best is not a reliable guide to today's." if changes >= (len(best) - 1) / 2 else "."))

        print("\nPaired comparisons on shared days (retention A - retention B):")
        shown = False
        for A, B in itertools.combinations(backends, 2):
            shared = [d for d in dates if (A, d) in table and (B, d) in table]
            if len(shared) < 2:
                continue
            diffs = np.array([ret(table[(A, d)]) - ret(table[(B, d)]) for d in shared])
            tail = f", SE {diffs.std(ddof=1) / np.sqrt(len(diffs)):.3f}" if len(diffs) >= 3 else ""
            print(f"  {A} - {B}: mean {diffs.mean():+.3f} over {len(shared)} shared days{tail}; "
                  f"{A} higher on {int((diffs > 0).sum())} of {len(shared)}")
            shown = True
        if not shown:
            print("  (no two backends share two days yet)")
        print("  With fewer than about 5 shared days these are indications, not findings.")

    # --- other circuits (the week5 job files), kept out of the controlled series ---
    others = [x for x in sess if x["variant"] != "rule"]
    if others:
        print("\nOther circuits (from week5_hardware_run.py; not part of the controlled series):")
        zmax = 0.0
        for x in others:
            line = f"  {x['backend']} {x['session']}: retention {ret(x):.2f}"
            ref = next((r_ for r_ in rule_sess if r_["backend"] == x["backend"] and r_["session"] == x["session"]), None)
            if ref is not None:
                d_ = (x["E"] - ref["E"]) / ideal_E
                z = d_ / (np.hypot(x["se"], ref["se"]) / sign)        # same sign as the retention difference
                zmax = max(zmax, abs(z))
                line += f"; benchmark circuit on that backend and day: {ret(ref):.2f} -> difference {d_:+.3f} (z = {z:+.1f})"
            print(line)
        if zmax >= 3:
            print("  Two different circuits run minutes apart on one device differ by more than shot noise: retention")
            print("  depends on the compiled circuit (layout, parameters), not only on the device and the day.")

    # --- ranking and usage ---
    if len(summary) >= 2:
        print("\nRanking by mean session retention (means cover DIFFERENT days; prefer the paired comparisons):")
        for b, (m, u) in sorted(summary.items(), key=lambda kv: -kv[1][0]):
            print(f"  {b:14s} {m:.3f} +- {u:.3f}")
    qs = [r["qsec"] for r in rows if r.get("qsec") is not None]
    if qs:
        print(f"\nQuantum time used by the {len(qs)} jobs with usage data: {sum(qs):.1f} s ({sum(qs) / 60:.2f} min), "
              f"median {np.median(qs):.1f} s per job.")
    else:
        print("\nQuantum seconds per job: not available from this runtime version; read them on the IBM jobs page.")
    print("\nCaveats: each backend has its own layout and calibration; readout error is unmitigated; 'ran' is the day")
    print("the job started when the API provides it, else the submission day (week5 jobs: decoded from the job ID);")
    print("P(optimal) does not track the energy (see Fez on a low-retention day), so keep both in view.")


def calibration_report(rows, ideal_E):
    """Does the predicted fidelity (free, from the backend's reported errors at submission) track the measured
    retention? Only runs submitted with this version carry the data."""
    from scipy.stats import pearsonr, spearmanr
    cal = [r for r in rows if r.get("variant", "rule") == "rule" and r.get("calib")]
    if not cal:
        print("\nCalibration predictor: no logged run carries calibration data yet (runs submitted before version 3 do not).")
        return
    print(f"\nCalibration predictor: {len(cal)} run(s) carry the backend's reported errors from submission time.")
    print(f"  {'submitted':17s} {'backend':14s} {'qubits':26s} {'2Q fid':>7s} {'readout':>8s} {'predicted':>10s} {'retention':>10s} {'ret/pred':>9s}")
    x = np.array([r["calib"]["pred_fid"] for r in cal])
    y = np.array([r["stats"]["E"] / ideal_E for r in cal])
    for r, xi, yi in zip(cal, x, y):
        c = r["calib"]
        print(f"  {r['submitted']:17s} {r['backend']:14s} {str(c['physical_qubits']):26s} {c['two_q_fid']:7.3f} "
              f"{c['readout_fid']:8.3f} {xi:10.3f} {yi:10.3f} {yi / xi:9.2f}")
    if len(cal) < 4 or np.ptp(x) < 1e-9:
        print("  Too few runs (or no spread in the prediction) for statistics; the test needs about 10 runs over 4+ days.")
        return
    pr, sr = pearsonr(x, y), spearmanr(x, y)
    b, a = np.polyfit(x, y, 1)
    loo = []
    for k in range(len(x)):
        m = np.arange(len(x)) != k
        bk, ak = np.polyfit(x[m], y[m], 1)
        loo.append(y[k] - (ak + bk * x[k]))
    ratio = y / x
    print(f"  correlation predicted vs measured: Pearson {pr[0]:+.2f} (p = {pr[1]:.3f}), Spearman {sr[0]:+.2f} (p = {sr[1]:.3f})")
    print(f"  linear fit: retention = {a:.2f} + {b:.2f} x predicted; leave-one-out error {np.sqrt(np.mean(np.square(loo))):.3f} "
          f"vs SD of retention {y.std(ddof=1):.3f}")
    print(f"  ret/pred ratio: mean {ratio.mean():.2f}, SD {ratio.std(ddof=1):.2f} "
          "(a stable ratio means the predictor is useful up to a constant)")
    by_day = {}
    for r, xi, yi in zip(cal, x, y):
        by_day.setdefault(r["session"], {}).setdefault(r["backend"], []).append((xi, yi))
    hits = total = 0
    for d, bk in sorted(by_day.items()):
        if len(bk) >= 2:
            pred_best = max(bk, key=lambda k: np.mean([v[0] for v in bk[k]]))
            meas_best = max(bk, key=lambda k: np.mean([v[1] for v in bk[k]]))
            total += 1
            hits += pred_best == meas_best
    if total:
        print(f"  best backend of the day: the predictor picked the measured best on {hits} of {total} day(s).")
    print("  Decision rule: if the predictor beats `least_busy` on held-out days, choose the backend (and layout) by")
    print("  predicted fidelity; otherwise it adds nothing. Few runs: treat every number here as provisional.")


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
    ap.add_argument("--no-info", action="store_true", help="skip the usage and run-time lookups")
    ap.add_argument("--no-manual-first-job", action="store_true")
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
        entries = read_log()
        cache, info = load_json(COUNTS_FILE, {}), load_json(INFO_FILE, {})
        state = {"service": None}
        rows = []
        if os.path.exists(args.baseline):
            hw = np.load(args.baseline, allow_pickle=True)
            jid_b = str(hw["job_id"]) if "job_id" in hw.files else "unknown"
            backend_b = str(hw["backend"]) if "backend" in hw.files else "unknown"
            stats_b = count_stats(hw["raw_counts"].item(), diag, opt_idx, feasible)
            t_b = job_id_time(jid_b)
            when_b = (t_b.strftime("%Y-%m-%d") if t_b else
                      datetime.fromtimestamp(os.path.getmtime(args.baseline)).strftime("%Y-%m-%d"))
            sub_b = (t_b.strftime("%Y-%m-%d %H:%M") + " (job ID)") if t_b else f"file {when_b}"
            t_known = job_id_time(KNOWN_FIRST_JOB_ID)
            when_known = t_known.strftime("%Y-%m-%d") if t_known else "first job"
            if jid_b == KNOWN_FIRST_JOB_ID:
                rows.append(dict(submitted=sub_b, session=when_b, backend=backend_b, depth=None, twoq=None, qsec=None,
                                 stats=stats_b, variant="week5 circuit"))
            else:
                print(f"\nWARNING: {args.baseline} holds job {jid_b} ({sub_b.replace(' (job ID)', '')} according to its job ID),")
                print(f"  not your original first job {KNOWN_FIRST_JOB_ID}. Running week5_hardware_run.py again overwrites")
                print("  that file. The original counts are no longer stored here, so the original first job enters the")
                print(f"  series from the numbers analyzed on the day (energy {MANUAL_FIRST_JOB['E']:.4f} +- {MANUAL_FIRST_JOB['seE']:.4f},")
                print("  dated from its job ID); the file's job is listed separately. --no-manual-first-job drops the manual row.\n")
                if not args.no_manual_first_job:
                    rows.append(dict(submitted=f"{when_known} (job ID)", session=when_known, backend="ibm_marrakesh",
                                     depth=None, twoq=None, qsec=None, stats=dict(MANUAL_FIRST_JOB),
                                     variant="week5 circuit"))
                rows.append(dict(submitted=sub_b, session=when_b, backend=backend_b, depth=None, twoq=None, qsec=None,
                                 stats=stats_b, variant="week5 circuit"))
        pending = []
        for e in entries:
            jid = e["job_id"]
            if jid not in cache:
                svc = get_service(state)
                if not svc:
                    pending.append(e)
                    continue
                job = svc.job(jid)
                status = str(job.status())
                if "DONE" not in status.upper():
                    print(f"  {e['backend']} job {jid} (submitted {e['time']}): status {status}")
                    pending.append(e)
                    continue
                counts = job.result()[0].data.meas.get_counts()
                cache[jid] = {str(k): int(v) for k, v in counts.items()}
                save_json(COUNTS_FILE, cache)
                if not args.no_info:
                    info[jid] = job_info(job)
                    save_json(INFO_FILE, info)
            elif not args.no_info and jid not in info:
                svc = get_service(state)
                if svc:
                    try:
                        info[jid] = job_info(svc.job(jid))
                    except Exception:
                        info[jid] = {"tried": True}
                    save_json(INFO_FILE, info)
            inf = info.get(jid, {})
            started = inf.get("started")
            rows.append(dict(submitted=e["time"].replace("T", " ")[:16],
                             session=started[:10] if started else e["time"][:10], backend=e["backend"],
                             depth=e["depth"], twoq=e["twoq"], qsec=inf.get("quantum_seconds"),
                             stats=count_stats(cache[jid], diag, opt_idx, feasible), variant="rule", calib=e.get("calib")))
        if pending:
            print(f"\n{len(pending)} job(s) not retrieved yet (queued, or not reachable); they are left out below. "
                  "Run --retrieve again later.")
        series_report(rows, ideal_E, ideal_Popt, ideal_Pfeas)
        calibration_report(rows, ideal_E)
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
        calib = predict_fidelity(t, backend)
        print(f"    qubits {calib['physical_qubits']}; predicted fidelity {calib['pred_fid']:.3f} "
              f"(2Q {calib['two_q_fid']:.3f}, 1Q {calib['one_q_fid']:.3f}, readout {calib['readout_fid']:.3f}); "
              f"median 2Q error {calib['median_2q_error']}")
        if args.submit:
            sampler = SamplerV2(mode=backend)
            job = sampler.run([t], shots=args.shots)
            append_log(dict(time=datetime.now().isoformat(timespec="seconds"), backend=real_name,
                            job_id=job.job_id(), shots=args.shots, depth=depth, twoq=twoq, note=args.note, calib=calib))
            print(f"    submitted, job ID {job.job_id()} (logged in {LOG_FILE})")
        else:
            counts = AerSimulator.from_backend(backend).run(t, shots=args.shots, seed_simulator=100 + i).result().get_counts()
            rows.append(dict(name=nm, depth=depth, twoq=twoq, stats=count_stats(counts, diag, opt_idx, feasible), calib=calib))

    if args.submit:
        print("\nAll jobs submitted and logged. They queue independently and may finish at different times.\n"
              "Fetch and analyze with:  python week11_backend_benchmark.py --retrieve\n"
              "Repeat the --submit command on other days to build the series.")
        return
    dry_report(rows, ideal_E, ideal_Popt, ideal_Pfeas)


if __name__ == "__main__":
    main()
