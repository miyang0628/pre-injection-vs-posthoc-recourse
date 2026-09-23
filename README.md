# LLM-guided counterfactual recourse for comorbid chronic-disease risk

Code and analysis for a study on **where admissibility constraints should enter a
counterfactual search** when generating health-improvement recourse for
policyholders in a high-risk (comorbid hypertension + diabetes) tier, and on
**what is gained by delegating the discretionary constraint layer to an LLM**.

The pipeline classifies KNHANES respondents into risk tiers with a
gender-stratified gradient-boosting model, then searches for counterfactual
recourse under a three-layer guardrail. Its central finding is that the point of
constraint injection — not the "violation rate" — determines whether usable
recourse survives: post-hoc filtering collapses to near-zero feasibility while
pre-injection preserves it, on the same admissible region.

> **Framing (revised).** The headline contribution is now stated as a *metric*
> claim: the "violation rate" universally reported for constrained recourse is a
> **design tautology** — any scheme that applies the bounds at all reports zero
> — so it certifies nothing and conceals the only quantity that discriminates,
> **feasibility**. The pre-injection-vs-filtering gap is the *evidence* for that
> reframing, not the claim itself. The LLM section is framed as an **ablation**
> of the discretionary layer: delegation neither self-enforces the safety floors
> nor dominates the rule on recourse quality, which is exactly what makes the
> code-enforced layers load-bearing.

## Repository layout

```
data/                     raw survey files + generated artefacts (not committed; see data/README.md)
notebooks/
  01_data_preprocessing.ipynb        KNHANES 2020-2024 -> modelling frame
  02_xgboost_modeling.ipynb          gender-stratified XGBoost + diagnostics
  03_representative_cases.ipynb      illustrative per-patient recourse
  04_condition_comparison.ipynb      core injection experiment: C0 / C1 / C2 + delegation ablation
  05_llm_vs_rule_guardrails.ipynb    LLM-tier vs rule guardrails (IoU divergence, safety breaches)
  06_actuarial_projection.ipynb      illustrative expected-cost sizing
  07_robustness_and_delegation_value.ipynb   robustness of the injection result + value of LLM delegation
  08_admissibility_baselines.ipynb   NEW: independent admissibility-aware baselines (B1 soft-penalty, B2 Monte-Carlo)

  guardrail_core.py                  three-layer guardrail + interval metrics
  experiment_core.py                 DiCE wrappers and C0/C1/C2 evaluators
  baseline_core.py                   NEW: B1 (soft-penalty) and B2 (Monte-Carlo) recourse generators
  run_baselines.py                   NEW: run B1/B2 on the 155-patient cohort -> baselines_cohort.csv
  run_b1_lambda_sweep.py             NEW: sweep the B1 penalty weight -> b1_lambda_sweep.csv
  cost_scenarios.py                  NEW: conditional exposure-reclassification sensitivity table
  robustness_optimizer_independence.py  non-DiCE random-search reproduction
  robustness_violation_breakdown.py     per-candidate violation-type rates
  robustness_admissibility_levels.py    cumulative admissibility decomposition
  robustness_box_tightness.py           box-widening sweep
  delegation_value_stats.py             Holm-corrected tier stats + LLM-vs-rule value
  make_robustness_figures.py            builds the two summary figures
results/
  figures/                grayscale figures (PNG + PDF)
  tables/                 result tables (.tex for the paper) and per-patient CSVs
requirements.txt
.env.example              template for the LLM keys used by notebook 05
```

## What each stage produces

| Notebook | Key outputs |
|---|---|
| 01 | `data/df_final.pkl`, `data/agent_config.pkl` |
| 02 | `data/model_{male,female}.pkl`, `data/val_{male,female}.pkl`, CM/ROC/calibration figures, `table1_model_performance.tex` |
| 03 | `results/tables/representative_cases.json` |
| 04 | `results/tables/_cohort_{male,female}.csv`, `experiment_cohort.csv`, `table_condition_comparison.tex`, `table_delegation.tex`, condition figures |
| 05 | `results/tables/llm_tier_divergence.csv`, `table_tier_divergence.tex`, tier figures |
| 06 | `results/tables/actuarial_projection.json`, `table12_cost_gradient.tex`, `table13_cost_offset.tex` |
| 07 | `results/tables/*` robustness CSVs + `summary_*.{csv,tex}`, `fig_optimizer_independence`, `fig_llm_vs_rule_value` |
| 08 | `results/tables/baselines_cohort.csv`, `b1_lambda_sweep.csv`, `table_baseline_comparison.tex`, `table_b1_lambda.tex` |
| — | `cost_scenarios.py` -> `results/tables/cost_scenarios.csv`, `table_cost_scenarios.tex` |

## Notebook 07 at a glance

The intervention cohort is the 155 model-confirmed Class-3 patients (89 male,
66 female).

1. **Optimiser-independence** — a non-DiCE random search reproduces the collapse:
   post-hoc 5.2% vs pre-injection 98.1% feasibility (pooled).
2. **Mechanism** — among unconstrained DiCE candidates, 100% leave the range box,
   92% increase a reduce-only feature, 48% break anthropometric coupling, 36%
   cross a Layer-1 floor; widening the box to 5x still yields 0% post-hoc.
3. **Box-tightness sweep** — feasibility does not recover as the box is widened,
   confirming the obstacle is directional, not a matter of box width.
4. **Delegation value** — LLM ranges diverge from the rule (IoU ~ 0.4,
   Holm-corrected p << 0.001) but do not improve recourse: feasibility <= rule
   (98.7%) and sparsity is worse at every tier (p < 1e-4). Every tier breaches
   the floors for ~1/4 of patients, so the code-enforced safety layer is
   required regardless of model capability.

## Notebook 08 at a glance (new, revision)

Added to answer the "weak comparison" review point: the original C0/C1/C2 are
three handlings of a single DiCE pipeline, so pre-injection beating post-hoc
filtering can look unsurprising. Notebook 08 tests the injection-point result
against two **independent** admissibility-aware recourse generators, on the same
155-patient cohort and the same admissible region:

- **B1 (soft-penalty)** — constraints in the *objective* as a soft cost
  (Wachter / actionable-recourse style), by random-restart local search.
- **B2 (Monte-Carlo)** — optimiser-free rejection sampling *inside* the region,
  sharing no code with DiCE (pre-injection without a genetic optimiser).

Headline results (pooled, n=155, Wilson 95% CI):

| Generator | Constraint handling | Feasibility | Reaches target tier |
|---|---|---|---|
| C0 Pure DiCE | none | 100.0% | 100.0% |
| C1 Post-hoc filter | after search | 0.0% [0.0, 2.4] | 100.0% |
| C2 Pre-injection (DiCE) | before search (hard box) | 98.7% [95.4, 99.6] | 98.7% |
| **B1 Soft-penalty (non-DiCE)** | in objective (soft) | **11.6% [7.5, 17.6]** | **99.4%** |
| **B2 Monte-Carlo (non-DiCE)** | before search (hard box) | **98.7% [95.4, 99.6]** | 98.7% |

Two conclusions the C0–C2 comparison alone could not give: (i) the injection
effect is **not a DiCE artefact** — the DiCE-free pre-injection search (B2)
reproduces C2 exactly; (ii) placing the identical constraints in the
**objective** (B1) is not enough — B1 reaches the target tier for 99.4% of
patients but 87.7% of its optima land **outside** the admissible region, so
feasibility collapses. A penalty-weight sweep (`run_b1_lambda_sweep.py`) shows
B1 feasibility saturates near 20% no matter how hard the penalty is pushed, so
the failure is structural, not an under-weighted penalty.

## Conditional cost scenarios (new, revision)

`cost_scenarios.py` produces a **bounded sensitivity** table (paper Table 9),
not a pricing claim. It maps the demonstrated reachability to aggregate exposure
under externally cited comorbidity multipliers (French et al. 2005: 2.56x;
Gilmer et al. 2005: 10–50% per comorbidity), holding fixed the two conditionals
stated in the paper. Across a conservative-to-upper band the enacted-recourse
exposure of the predicted Class-3 book falls ~44%–69%; the point is the
first-order *shape* under any cited multiplier, not the percentage.

## Reproduce

```bash
pip install -r requirements.txt          # pinned stack, Python 3.10+
cp .env.example .env                      # add OPENAI_API_KEY + tier model names (notebook 05 only)
cd notebooks
jupyter notebook                          # run 01 -> 08 in order

# or, from the command line, the revision experiments only:
python run_baselines.py                   # B1/B2 cohort -> results/tables/baselines_cohort.csv
python run_b1_lambda_sweep.py             # penalty sweep -> results/tables/b1_lambda_sweep.csv
python cost_scenarios.py                  # conditional scenarios -> results/tables/cost_scenarios.csv
```

Notebooks 04/05/07/08 cache their per-patient cohort loops to
`results/tables/`; delete the cache CSV (or set `FORCE_RERUN = True` in 04) to
recompute. The standalone `run_*.py`, `cost_scenarios.py`, `robustness_*.py`,
and `delegation_value_stats.py` scripts reproduce the individual analyses from
the command line. `baseline_core.py` fixes all random seeds, so B1/B2 results
are deterministic.

## Data availability

Raw KNHANES 2020-2024 files are obtained from the Korea Disease Control and
Prevention Agency (https://knhanes.kdca.go.kr) and are **not redistributed**
here; see `data/README.md`. `results/tables/llm_tier_divergence.csv` contains
only validation-split indices and derived metrics (IoU, breach counts,
feasibility) — no raw personal data — so notebook 07 runs without re-querying any
model. The baseline experiments (notebook 08) need only the trained models and
validation splits, and the cost scenarios need no data at all.
