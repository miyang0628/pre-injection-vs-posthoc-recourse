"""
cost_scenarios.py
Conditional exposure-reclassification scenarios (revision).

This is NOT a pricing claim. It is a bounded sensitivity arithmetic: given the
paper's demonstrated reachability (feasibility on the 363 predicted Class-3
patients), and holding fixed the two conditionals stated in the paper --- (i) an
admissible recourse is enacted, (ii) the classifier tier tracks expected cost
--- it maps the reachability to aggregate exposure under a band of EXTERNALLY
SOURCED comorbidity cost multipliers. The multipliers are taken from the claims
literature and are not estimated here:

  * French, Rachlin & Sindelar (2005), J Intern Med: comorbid HTN+DM excess
    expenditure = 2.56x a no-condition reference.
  * Gilmer et al. (2005), Diabetes Care: 10-50% incremental cost per comorbidity
    -> single-condition tier treated as a 1.2-1.6x band.

Produces results/tables/cost_scenarios.csv and the numbers in Table 9.
"""
import os
import pandas as pd

# Reachability facts from the injection experiment (363 predicted Class-3).
N = 363
TO_CLASS0 = 347      # reach reference tier under stepwise fallback
TO_CLASS1 = 13       # reach single-condition tier
INFEASIBLE = N - TO_CLASS0 - TO_CLASS1  # remain at comorbid tier (=3)

# Exposure in multiples of the Class-0 (reference) per-capita expected cost = 1.0.
SCENARIOS = {
    "Low (conservative)":    {"c3": 1.80, "c1": 1.20},
    "Mid (French-anchored)": {"c3": 2.56, "c1": 1.40},
    "High (upper band)":     {"c3": 3.30, "c1": 1.60},
}


def compute():
    rows = []
    for name, s in SCENARIOS.items():
        c3, c1 = s["c3"], s["c1"]
        pre = N * c3
        post = TO_CLASS0 * 1.0 + TO_CLASS1 * c1 + INFEASIBLE * c3
        red = (pre - post) / pre * 100
        rows.append({"scenario": name, "c3_mult": c3, "c1_mult": c1,
                     "pre_recourse": round(pre), "post_recourse": round(post),
                     "exposure_reduction_pct": round(red, 1)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = compute()
    print(df.to_string(index=False))
    print("\nUnits: multiples of Class-0 per-capita expected cost (reference=1.0);"
          f" book of {N}.")
    out = "../results/tables/cost_scenarios.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"saved {out}")
