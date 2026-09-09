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

  guardrail_core.py                  three-layer guardrail + interval metrics
  experiment_core.py                 DiCE wrappers and C0/C1/C2 evaluators
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

## Reproduce

```bash
pip install -r requirements.txt          # pinned stack, Python 3.10+
cp .env.example .env                      # add OPENAI_API_KEY + tier model names (notebook 05 only)
cd notebooks
jupyter notebook                          # run 01 -> 07 in order
```

Notebooks 04/05/07 cache their per-patient cohort loops to `results/tables/`;
set `FORCE_RERUN = True` (04) or delete the cache CSV to recompute. The
standalone `robustness_*.py` / `delegation_value_stats.py` scripts reproduce the
individual notebook-07 analyses from the command line.

## Data availability

Raw KNHANES 2020-2024 files are obtained from the Korea Disease Control and
Prevention Agency (https://knhanes.kdca.go.kr) and are **not redistributed**
here; see `data/README.md`. `results/tables/llm_tier_divergence.csv` contains
only validation-split indices and derived metrics (IoU, breach counts,
feasibility) — no raw personal data — so notebook 07 runs without re-querying any
model.
