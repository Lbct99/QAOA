# Portfolio Optimization with QAOA on Real IBM Hardware — Write-up Outline

*A working outline — numbers are filled in from your actual runs where you have them; bracketed notes mark what still needs your input or a final decision. Updated: Section 8 (continuous-weight extension and classical backtest) is new; Section 8.5 now holds the backtest and budget-sweep results; placeholders in Sections 2, 6 and 7 are filled.*

---

## 1. Motivation and scope

- Why QAOA for portfolio selection: Markowitz selection as a combinatorial (QUBO) problem, a natural fit for quantum optimization demonstrations.
- **State the goal honestly up front** (this framing matters for how a reader judges your results): this is a learning/architecture exercise — building a complete, correct pipeline from financial formulation to real quantum hardware — not a claim of quantum advantage over classical solvers.
- One-paragraph summary of the end-to-end pipeline, for readers who stop here.

## 2. Problem formulation

- Markowitz portfolio selection: minimize risk, maximize return, subject to a fixed-size budget constraint (equal-weight selection of B of N assets).
- The QUBO: `x^T Σ x − q·(μ^T x) + P·(Σx − B)²`, and what each term means.
- Data: 10 stocks for the simulator study; a 6-stock subset (AAPL, MSFT, JPM, JNJ, KO, NVDA) for the hardware run. 499 trading days, 2024-09-30 → 2026-09-25 (pinned dates, cached prices), risk-aversion `q = 0.5`, budget `B = 3`.
- **Convention note (state it explicitly)**: the objective is built in raw selection units (`x`, not `x/B`), exactly as in the code. Normalizing by the budget would change the effective risk aversion: raw `q = 0.5` at `B = 3` equals `q = 0.5/3 ≈ 0.167` on actual portfolio weights. Nothing downstream is invalid, but any comparison across budgets must account for it.
- **Penalty tuning as its own finding, not a footnote**: your auto-heuristic initially set `P` \~31x higher than needed; `week1b_diagnostics.py`'s penalty sweep found the true minimum and you settled on a \~2x safety margin. Worth a short subsection — it's a concrete, demonstrable engineering improvement with before/after numbers.

## 3. Classical benchmark

- Brute-force optimum over all `2^N` bitstrings: your ground truth throughout.
- 10-asset universe: optimal portfolio **JNJ, KO, NVDA**.
- 6-asset hardware-target universe: same optimal portfolio, **JNJ, KO, NVDA**.
- **The near-degeneracy finding** — this is a genuinely important result, give it real space: the 2nd-best portfolio (JPM, JNJ, KO) sits only `0.0067` away from optimal out of a `0.4249` full spread — a **1.6% gap-to-spread ratio**. This single number explains most of what follows (why QAOA doesn't converge sharply, why hardware's top result was the 2nd-best, not the best).

## 4. QUBO → Ising mapping

- The substitution `x_i = (1−Z_i)/2`, and the resulting closed forms: `h_i = -0.5·rowsum(Q)_i`, `J_ij = 0.5·Q_ij`.
- Verification: exact match across all `2^N` bitstrings (max error \~1e-15 — worth stating precisely, it's a strong correctness claim).
- Resulting structure: fully connected (every asset pair coupled) — 45 terms for 10 assets, 15 for 6 — and why that matters for hardware later (connects forward to Section 6).

## 5. QAOA circuit and simulator results

- p=1 by hand: `Rz`/`CNOT-Rz-CNOT`/`Rx` construction, and the `(γ,β)` landscape plot (`qaoa_p1_landscape.png`).
- General-p via `QAOAAnsatz` + `EstimatorV2`, `POWELL`, linear-ramp initialization.
- **REPS sweep with restarts** (`qaoa_reps_sweep.png`) — your key simulator finding: on the 10-asset problem, energy and `P(optimal)` kept improving through `reps=6` with no plateau (energy: −0.441 → −0.590; `P(optimal)`: 0.5% → 5.6%); on the 6-asset problem, `P(optimal)` stayed low and non-monotonic (8.6% → 13.1%, dipping at `reps=3`) — **explained by the near-degeneracy in Section 3, not by insufficient depth**. This contrast is worth stating explicitly: it shows you distinguished "landscape is flat" from "circuit is too shallow," which is not a distinction most QAOA demos bother making.

## 6. Hardware transpilation

- Fully-connected 15-coupling problem onto a sparse heavy-hex IBM device: logical depth 21 → transpiled depth **265** on `ibm_marrakesh` with **109** two-qubit (CZ) gates — about **7.3x** the logical 15 interactions, all SWAP-routing overhead. For comparison, the offline model of a different chip (`FakeSherbrooke`) gave depth 403 / 111 ECR gates: routing cost is backend-specific.
- **The depth-vs-REPS tradeoff decision**: chose `reps=2` for the hardware run specifically because depth scales \~linearly with REPS (\~200/layer) while simulator `P(optimal)` gains past `reps=2` were small given the near-degenerate landscape — not worth the added noise exposure. State this as a deliberate engineering tradeoff, not an afterthought.
- The backend-mismatch bug (transpiled for one chip, submitted to another → `ecr` not supported) is worth a short, honest paragraph — real debugging is part of the engineering story, and the fix (always transpile for the backend you'll actually submit to) is a genuinely reusable lesson.

## 7. Hardware results

- Job: single job, `reps=2`, `ibm_marrakesh` (156 qubits), 4096 shots — and *why* one job (job-count-aware design given Open Plan limits; simulator-optimized warm start, hardware only samples — no on-hardware optimization loop).
- **Headline result, with the actual numbers**:

  |  | P(optimal) | P(feasible) |
  | --- | --- | --- |
  | Simulator (ideal) | 0.098 | 0.575 |
  | Raw hardware | 0.100 | 0.485 |
  | Readout-mitigated | 0.114 | 0.522 |
- `P(optimal)` was statistically unaffected by hardware noise (sim vs. raw within \~1 shot-noise standard error). `P(feasible)` dropped by a real \~9 points — noise pulls probability mass toward the uniform 31% feasible baseline. Mitigation recovered \~40% of that gap, meaning **\~40% of the feasibility loss was readout error, \~60% was gate/decoherence noise** accumulated over the depth-265 circuit. This split is your most specific, citable finding — lead with it in any summary/abstract.
- Top hardware outcomes were the top-2 *classical* near-tied portfolios (JPM/JNJ/KO at 16.1% and JNJ/KO/NVDA at 10.0%) — direct evidence the full pipeline is correct end to end, not just "it ran."
- Reproducibility: the comparison script was re-run in later sessions and reproduced every number exactly (deterministic optimization, seeded sampling).
- Figure: `sim_vs_hardware.png` (grouped bar chart, all three distributions).

## 8. Extension: continuous weights and out-of-sample validation

*New section. The honest arc: the extension works mechanically and improves the in-sample objective (as it must). Out of sample, neither it nor the K=1 model beats plain 1/N, and at the tested settings K=2 almost never uses its extra freedom. It also costs far too many gates for current hardware.*

**8.1 Formulation**

- Each asset gets K bits: `y_i = Σ_k 2^k·b_ik`, `y_i ∈ {0..2^K−1}`, budget constraint `Σ y_i = N_total`. The objective stays in raw units, same as K=1.
- Correctness check: with K=1 the new `Q` reproduces the original exactly (max difference `0.00e+00`, on real data). An earlier draft that normalized weights inside the objective *failed* this check — which exposed the convention issue in Section 2. Worth a sentence: the check did its job.
- Cost: `n × K` qubits (6 stocks, K=2 → 12 qubits). The feasible share of the search space falls from 31% (20/64) to 1.4% (56/4096), so the budget constraint becomes much harder for any sampler to satisfy.

**8.2 In-sample result (6 stocks, K=2, budget 3 units)**

- Optimum: **JNJ 2/3, NVDA 1/3** (KO dropped), versus JNJ/KO/NVDA at 1/3 each.
- True Markowitz value `−0.2534` vs `−0.2178` for the best equal-weight allocation (which matches the Week 1 value — a second consistency check). Improvement `0.0356`; gap from best to 2nd-best `0.0202` (vs `0.0067` in the K=1 problem, so the top is less tied; the full spread was not compared).
- In finance terms (derived from the objective values and the Week 1 return estimates, rounded): return ≈ 28.8% → 34.1%, volatility ≈ 15.4% → 17.0%, return/vol ≈ 1.87 → 2.01. This is a point on the risk/return trade-off at `q = 0.5`, not a free lunch.
- Penalty: minimum feasible `0.0216`; the auto-heuristic used `0.8018` (\~37x).
- Caveat to state plainly: K=2 contains every K=1 allocation, so an in-sample gain is guaranteed and proves nothing by itself.

**8.3 Out-of-sample sanity check (4 expanding splits, test windows nested)**

| Split (train fraction) | 1/N | K=1 | K=2 | K=2 − K=1 |
| --- | --- | --- | --- | --- |
| 50% | 2.20 | 1.50 | 1.41 | −0.09 |
| 60% | 2.21 | 1.56 | 2.06 | +0.50 |
| 70% | 2.63 | 1.59 | 1.66 | +0.07 |
| 80% | 3.95 | 2.80 | 2.87 | +0.07 |
| Mean | 2.75 | 1.86 | 2.00 | +0.14 |

(test return/vol per split)

- K=2 vs K=1: **inconclusive**. Three of four "wins" are ≤ 0.07; the mean gap is driven by one split. K=2 raised both return and volatility in every split, consistent with its objective.
- **Plain 1/N beat both** on return/vol in every split. K=1 was dominated (lower return and higher volatility) in all four, K=2 in two. Consistent with the literature on naive diversification vs. optimizers fed noisy historical means (DeMiguel, Garlappi & Uppal, 2009).
- Caveats: nested/overlapping windows, a strongly rising market, a universe picked with hindsight, only \~2 years of data.
- **Instability link to Section 7**: the K=1 allocation chosen from the training window changed across splits — MSFT/JPM/JNJ (50%, 60%) then JPM/JNJ/KO (70%, 80%). JPM/JNJ/KO is the portfolio hardware ranked first. The near-tie that made QAOA's job hard also makes the "optimal" portfolio unstable over time, so landing in the near-tied top group is the more meaningful target than hitting the exact bitstring.
- Cross-reference: in the 10-stock backtest (8.5), K=2 matched K=1's weights at 18 of 19 rebalances, so the unequal weights it chose in all four 6-stock splits above did not carry over to the larger universe.

**8.4 Hardware cost of K=2 (measured on an offline model of `ibm_marrakesh`)**

| Problem | reps | Depth | Two-qubit gates |
| --- | --- | --- | --- |
| 6 qubits (15 couplings) | 2 | 257 | 109 |
| 12 qubits (66 couplings) | 1 | 343 | 234 |
| 12 qubits (66 couplings) | 2 | 580 | 471 |

- The 6-qubit row matches the real run (265 / 109), which validates the model. Coefficients were random; routing depends on structure, not values.
- Even one K=2 layer has \~2x the gates of the run that lost \~9 points of feasibility; two layers \~4.3x. K=2 stays **simulator-only**.
- Not done: a K=2 QAOA run on the simulator. Expectation (unmeasured): lower `P(feasible)` than K=1 at equal depth, given the 1.4% feasible share.

**8.5 Classical walk-forward backtest (`week8_backtest.py`)**

- Setup: 10 stocks, daily data 2014-01-03 → 2026-09-25 (3,201 returns). Train on the trailing 3 years, rebalance every 126 trading days, 19 non-overlapping test periods covering 2017-01-04 → 2026-07-15. Target weights drift between rebalances; 10 bps per unit of traded notional; first allocation free; 95% confidence intervals from a paired block bootstrap.
- Built-in checks: the harness picks the same allocations as the pipeline's QUBO objective, and its drift-aware daily returns match explicit position tracking.
- **Methods note worth keeping**: the first version computed daily returns with constant weights (an implicit daily rebalance at no cost) while charging trading costs as if weights drifted. Fixing it moved the pipeline's gap to 1/N from −0.30 \[−0.72, +0.03\] to −0.39 \[−0.82, −0.05\]. The corrected script agreed with a second drift-aware variant (`week8b_drift_backtest.py`) on every printed number; the two share the bootstrap code, so the check covers the return and cost accounting.

  | Strategy | CAGR | Vol | Mean/vol | Max DD | Traded per rebalance |
  | --- | --- | --- | --- | --- | --- |
  | 1/N (rebalanced) | 21.9% | 18.3% | 1.17 | −33.6% | 13.0% |
  | 1/N buy-and-hold | 30.5% | 26.2% | 1.14 | −34.5% | 0% |
  | K=1, B=3 (pipeline) | 15.3% | 20.9% | 0.79 | −28.8% | 99.8% |
  | K=2, B=3 | 15.5% | 20.8% | 0.79 | −28.8% | 99.6% |
  | K=1, B=5 | 19.5% | 18.8% | 1.04 | −30.7% | 46.6% |
  | K=1, B=3 min-variance | 9.4% | 16.2% | 0.63 | −36.1% | 29.3% |
  | K=1, B=3 shrunk μ | 10.5% | 17.9% | 0.65 | −31.9% | 67.4% |
  | K=2, B=3 shrunk μ | 10.1% | 17.9% | 0.63 | −31.9% | 70.7% |
- Difference in mean/vol versus 1/N (95% CI): K=1 B=3 −0.39 \[−0.82, −0.05\]; K=2 B=3 −0.38 \[−0.83, −0.03\]; K=1 B=5 −0.14 \[−0.44, +0.11\]; min-variance −0.54 \[−1.13, −0.06\]; shrunk μ −0.52 \[−1.01, −0.10\] (K=1) and −0.55 \[−1.02, −0.13\] (K=2); buy-and-hold −0.03 \[−0.40, +0.28\].
- Reading: no strategy beat 1/N. The five rows whose intervals exclude zero are all 3-stock selections with overlapping holdings, so they are one finding rather than five; the budget-5 row, holding half the universe, is within noise. A budget sweep (below) tests whether concentration explains the gap. With 9 comparisons the chance that at least one flags by luck is about 37%, and the three nearest interval edges sit within 0.03–0.06 of zero.
- K=2 vs K=1: +0.01 \[−0.03, +0.05\]. The tight interval has a simple cause: K=2 chose the same weights as K=1 at 18 of 19 rebalances and used unequal weights once. At this budget, risk weight and universe the optimizer almost never uses its extra freedom, so the extension is moot here; the value of unequal weights is untested, not refuted.
- The chosen K=1 and K=2 allocations changed at all 18 rebalances; about half the portfolio is replaced each time.
- 1/N's lead is not an artifact of rebalancing: buy-and-hold has a similar mean/vol (1.14 vs 1.17) despite 26.2% volatility.
- Robustness (mean/vol for 2y / 3y / 5y training windows): 1/N 1.24 / 1.17 / 1.22; K=1 B=3 0.85 / 0.79 / 0.86; K=1 B=5 1.10 / 1.04 / 1.17; min-variance 0.71 / 0.63 / 0.68; shrunk μ (K=1) 1.00 / 0.65 / 0.55. The pipeline gap to 1/N is −0.36 to −0.39 in all three; shrinkage results swing with window length, so don't lean on them.
- **Budget sweep with a random-selection baseline (`week8c_budget_sweep.py`)**: the K=1 optimizer at budgets 2–9, same accounting, each compared with 500 random selections of the same size (re-drawn at every rebalance, same costs). Mean/vol, 3-year training window; 1/N = 1.17.

  | B | Optimized | Random median \[5%, 95%\] | Optimizer's rank | Gap to 1/N \[95% CI\] | Concentration | Selection |
  | --- | --- | --- | --- | --- | --- | --- |
  | 2 | 0.79 | 0.89 \[0.51, 1.30\] | 38% | −0.38 \[−0.89, +0.10\] | −0.28 | −0.10 |
  | 3 | 0.79 | 0.96 \[0.65, 1.31\] | 19% | −0.39 \[−0.86, −0.03\] | −0.21 | −0.17 |
  | 4 | 0.99 | 1.01 \[0.72, 1.29\] | 44% | −0.19 \[−0.55, +0.15\] | −0.17 | −0.02 |
  | 5 | 1.04 | 1.07 \[0.83, 1.31\] | 41% | −0.14 \[−0.44, +0.12\] | −0.10 | −0.03 |
  | 6 | 1.09 | 1.09 \[0.90, 1.29\] | 48% | −0.09 \[−0.30, +0.12\] | −0.08 | −0.01 |
  | 7 | 1.09 | 1.13 \[0.95, 1.28\] | 34% | −0.09 \[−0.30, +0.10\] | −0.05 | −0.04 |
  | 8 | 1.08 | 1.15 \[1.01, 1.25\] | 20% | −0.09 \[−0.27, +0.07\] | −0.03 | −0.07 |
  | 9 | 1.23 | 1.17 \[1.06, 1.24\] | 91% | +0.06 \[−0.05, +0.16\] | −0.01 | +0.06 |

  (Concentration = typical random B-stock portfolio minus 1/N; selection = optimizer minus typical random B-stock portfolio.)
- Sweep reading: the concentration cost is clean and monotone, from −0.28 at B=2 to −0.01 at B=9, and accounts for about half of the B=3 gap (−0.21 of −0.39). The rest is selection: the optimizer's picks were no better than random at any budget from 2 to 8 (selection effect below zero each time, rank below 50% in seven of eight), and at B=3 it beat only 19% of random strategies. That is weak evidence of worse-than-random (a rank near 20% occurs about one time in five by chance), but there is no evidence of positive selection skill. The one positive result, B=9 (rank 91%), is not robust: B=9 falls below 1/N with the 2-year and 5-year training windows (1.20 vs 1.24, 1.16 vs 1.22). Only the B=3 interval excludes zero, and with 8 intervals one flag is not surprising. The B=3 shortfall is the most stable pattern across windows (−0.39, −0.38, −0.36). The random B=3 strategies range from 0.65 to 1.31, so luck in which stocks are held swings mean/vol more than the optimizer's effect does.
- Turnover falls steeply with budget (97.5% at B=2, 99.8% at B=3, 16.1% at B=9): small budgets reshuffle nearly the whole portfolio at every rebalance.
- Caveats to carry into the text: hindsight/survivorship universe (so absolute levels, about 22% CAGR for 1/N, are not expectations), overlapping training windows, one market regime, flat costs.

## 9. Limitations and honest caveats

- Continuous weights were explored only on the 6-stock universe and only classically; no QAOA run exists for K=2 (Section 8.4). In the 10-stock backtest K=2 almost never chose unequal weights, so their value is untested there rather than refuted.
- Small universe (6–10 well-known large caps), chosen with hindsight; scaling behavior untested.
- The 6-stock out-of-sample check (Section 8.3) is short (\~2 years, nested windows); the 10-stock backtest (Section 8.5) is longer but uses a hindsight universe and a single market regime.
- Single hardware job — no run-to-run hardware variance measured (noise profiles drift over hours/days on real devices).
- No claim of quantum advantage — classical brute force solved all instances here instantly; the value demonstrated is architecture and pipeline correctness, stated as such in Section 1.

## 10. Next steps (brief — full detail in a follow-up, not this write-up)

- On the evidence so far, no classical variant (K=2, min-variance, shrinkage) justifies more quantum work: none beat 1/N, K=2 rarely differs from K=1, and the budget sweep shows the optimizer's stock choices were no better than random ones. Optional follow-ups, each cheap with the existing harness: settings where K=2 actually uses unequal weights (e.g. a higher return weight q) if its value is to be tested at all; and different asset classes (a hypothesis worth testing: optimizers may matter more when assets differ widely in risk).
- Universe expansion only after that, since more assets under the current objective mostly give the optimizer more room to fit noise. Practical issues to plan for: crypto trades every day while equities do not (calendar alignment), much higher volatility, and metals/energy usually enter as ETF or futures proxies with their own data quirks.
- Production direction: not ONNX-exporting QAOA itself (no neural network in the loop) — instead, a learned `(γ,β)`-initialization network (replacing the linear-ramp heuristic) as the actual ONNX/Azure-servable artifact, amortizing the simulator-optimization cost across problem instances.

---

## Appendix: artifacts to reference

- Scripts: `week1_qubo_setup.py` → `week8_backtest.py` (the full pipeline, in order; `week7*` are the continuous-weight extension, `week8` the backtest; `week8b_drift_backtest.py` is a cross-check variant; `week8c_budget_sweep.py` is the budget sweep).
- Figures: `qaoa_p1_landscape.png`, `qaoa_reps_sweep.png`, `sim_vs_hardware.png`.
- Data: `qubo_week1.npz`, `ising_week2.npz`, `hw6_pipeline.npz`, `hw6_hardware_result.npz`, `continuous_weights_week5.npz`, `prices_long_cache.csv` (created by the first backtest run).
- Repo: `github.com/Lbct99/QAOA`.

---

*\[Your notes: anything here that's wrong, missing, or where you want more/less emphasis — mark it up and we'll turn this into full prose next.\]*