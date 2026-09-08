# Guardrail-Constrained Counterfactual Recourse for Comorbid Chronic Disease Risk

> **Anonymous submission — Anonymous authors.**
> Code and experiments are released for anonymous peer review. Author, affiliation,
> and venue details are withheld.

A reproducible pipeline that turns a comorbidity **risk score** into
**guideline-constrained, actionable recourse candidates** for patients with both
hypertension and diabetes, evaluated on KNHANES 2020–2024.

Three stages: **XGBoost** multi-class risk model → **clinical guardrail** that
constrains the search space (a three-layer structure: absolute safety floors in
code, discretionary ranges from an LLM agent, fixed non-modifiable variables) →
**DiCE** counterfactual generation with a stepwise fallback (Class 0 → 1 → 2).

---

## Suggested repository names

Pick whichever reads best for the anonymized repo:

- `guardrail-counterfactual-recourse`
- `llm-guardrail-dice-comorbid`
- `pre-injection-vs-posthoc-recourse`
- `comorbid-risk-recourse` (shortest)

For a double-blind release, avoid names containing author initials, institution,
or the target journal.

---

## What this repository shows

Two claims, backed by the held-out validation cohort (all 155 model-confirmed
Class 3 patients).

### (b) How you inject constraints decides feasibility — the main result

A four-condition comparison:

| Condition | Constraint handling |
|---|---|
| **C0 — Pure DiCE** | unconstrained search, Class-0 target, no fallback |
| **C1 — Post-hoc filter** | unconstrained search, then drop candidates that violate the safety guardrail |
| **C2 — Pre-injection** | guardrail injected *before* search + stepwise fallback |

| Condition | Feasibility | Valid candidates | Violation rate |
|---|---|---|---|
| C0 Pure DiCE | 100% | 3.94 | 91% |
| C1 Post-hoc filter | **0%** | 0.00 | — |
| C2 Pre-injection | 98.7% | 3.41 | 0% |

A zero violation rate is trivial for any post-hoc or pre-injection scheme — true
*by construction*. So violation rate is a **sanity check**; the operative outcome
is **feasibility**. **Post-hoc filtering collapses to 0% feasibility** (the
unconstrained candidates violate the safety guardrail so consistently that
filtering removes all of them), while **pre-injection preserves recourse**. Same
safety guarantee, opposite usability.

### (a) The LLM guardrail is not the rule baseline

With only the absolute safety floors enforced in code and the "how far to move"
ranges delegated, the LLM proposes **patient-specific** ranges that diverge from
the fixed-ratio rule ranges. Measured across three model tiers (n=155, paired):

| Tier | Mean IoU vs rule | Raw safety breaches (mean) |
|---|---|---|
| weak  | 0.429 | 0.516 |
| mid   | 0.376 | 0.297 |
| strong| 0.402 | 0.374 |

- **LLM ≠ rule**: every tier's IoU differs from 1.0 (Wilcoxon p ≈ 3.5×10⁻²⁷).
- **Tiers differ but not monotonically** in model size (Friedman p ≈ 2.4×10⁻⁵);
  the mid tier diverges from the rule most.
- **No tier eliminates safety-floor breaches** (21–28% of patients; tier
  differences not significant, Friedman p = 0.079), which is why the code-level
  safety layer (Layer 1) is retained regardless of model capability.

Figures (`results/figures/`, grayscale, dpi 600, PNG+PDF): condition feasibility,
diversity, sparsity; tier IoU; tier safety breaches.

---

## Three-layer guardrail

- **Layer 1 — Safety.** Absolute physiological floors (energy, sodium, protein,
  potassium, carbohydrate, fiber) + anthropometric direction coupling
  (BMI/waist/weight move together). Always enforced in code; neither the LLM nor
  the rule baseline may relax these.
- **Layer 2 — Discretion.** The "how far to move" ranges. Filled by the LLM agent
  or by a fixed-ratio rule. This is where LLM-vs-rule divergence lives.
- **Layer 3 — Fixed.** Non-modifiable variables, held at baseline.

A **delegation** setting (aggressive / conservative) controls how much of Layer 2
is opened to the LLM vs. pinned by code.

---

## Repository structure

```
.
├── PROJECT_PLAN.md            # design rationale + open tasks
├── ANALYSIS_RESULTS.md        # results + test statistics
├── data/                      # raw SAS files (not committed) + artefacts
│   └── README.md
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_xgboost_modeling.ipynb
│   ├── 03_representative_cases.ipynb
│   ├── 04_condition_comparison.ipynb    # * four-condition cohort (b)
│   ├── 05_llm_vs_rule_guardrails.ipynb  # * model-tier comparison (a)
│   ├── 06_actuarial_projection.ipynb    # illustrative cost sizing
│   ├── guardrail_core.py                 # 3-layer guardrail + IoU metrics
│   └── experiment_core.py                # C0-C2 evaluators, DiCE wrappers
├── results/
│   ├── figures/               # PNG + PDF, grayscale, dpi 600, no captions
│   └── tables/                # LaTeX tables + CSV/JSON records
├── requirements.txt
└── README.md
```

Each notebook holds its entire code in a single first cell, with the notebook
name as the top comment.

---

## Reproducing

### 1. Environment
```bash
pip install -r requirements.txt
```
Python 3.10+. Key pins: `xgboost==2.0.3`, `dice-ml==0.11`.

### 2. Data
KNHANES is a repeated **cross-sectional** survey from the Korea Disease Control and
Prevention Agency (<https://knhanes.kdca.go.kr>); raw files are **not
redistributed**. Place five cycles under `data/`:
```
data/hn20_all.sas7bdat ... data/hn24_all.sas7bdat
```

### 3. Run order
```
01 -> 02 -> 03 -> 04 -> 06     (05 additionally needs an OpenAI API key)
```
Each notebook loads artefacts written by the previous stage. Notebook 04 runs the
cohort experiment itself and caches per-gender CSVs. Notebook 05 runs the full
cohort across three model tiers with **incremental save + resume**; provide the
key and model tiers via a local `.env`:
```
OPENAI_API_KEY=sk-...
LLM_MODEL_WEAK=gpt-4o-mini
LLM_MODEL_MID=gpt-5.4-mini
LLM_MODEL_STRONG=gpt-5.4
```
```python
from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv())
```
GPT-5.x tiers are routed through the Responses API and billed per token. **Never
commit `.env` or your key** — `.gitignore` excludes it.

### Reproduced reference numbers
Notebooks 01–02 reproduce the descriptive/model results exactly: risk-class
distribution 6,381 / 1,370 / 649 / 1,338 (total 9,738); model-confirmed Class 3 =
89 male, 66 female. Small model deviations arise only from Optuna's stochastic
search.

---

## Figure and artefact conventions

Figures are saved as **PNG and PDF** at **dpi 600**, without captions, in
**grayscale** (classes/conditions distinguished by grayscale level + line style +
marker + hatch, never color). Intermediate `.pkl`/`.csv` artefacts are **not
committed** — joblib pickles are tied to the pandas/numpy versions that wrote
them, so regenerate them locally by running the notebooks in order.

---

## Important limitation — cross-sectional data

KNHANES is a repeated cross-sectional survey with **no personal key linking a
respondent across years**. A model-predicted risk-tier transition is a
within-classifier, cross-sectional statement, **not** an observed change over
time, and cannot be validated against claims. The actuarial projection is an
**illustrative order-of-magnitude sizing exercise** from externally published
cost parameters — not a validated pricing model. Generated recourse candidates
are decision-support hypotheses, not clinical prescriptions, and require
domain-expert review before any real-world use.

---

## Status

Proof-of-concept research code released for anonymous peer review.
