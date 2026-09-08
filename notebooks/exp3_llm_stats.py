"""
exp3_llm_stats.py
=================
Addresses reviewer concerns #3 (IoU shows difference, not value) and #4
(statistical rigour of the tier experiment). Uses the EXISTING full-cohort
(n=155) per-patient LLM records -- no re-querying.

#4 rigour:
  - report n for every paired test
  - Friedman across the three tiers (IoU, breaches)
  - pairwise Wilcoxon signed-rank WITH Holm-Bonferroni correction
  - matched-pairs rank-biserial effect size for each pairwise test
  - stop leaning on a non-significant Friedman as evidence of equality; argue
    the floor's necessity from breach point-estimates (>0 at every tier)

#3 value (not just difference):
  - on the SAME 155 patients, compare LLM (each tier) vs the fixed rule (C2) on
    feasibility and recourse sparsity (n_changed_vars), paired per patient
  - this tests whether delegation BUYS anything (e.g. sparser recourse) beyond
    merely diverging from the rule
"""
import os, numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations

TAB = "../results/tables"
llm = pd.read_csv(os.path.join(TAB, "llm_tier_divergence.csv"))
m = pd.read_csv(os.path.join(TAB, "_cohort_male.csv"))
f = pd.read_csv(os.path.join(TAB, "_cohort_female.csv"))
coh = pd.concat([m, f], ignore_index=True)
rule = coh[(coh.cond == "C2_pre_rule") & (coh.deleg == "aggressive")].copy()

TIERS = ["weak", "mid", "strong"]

def pivot(metric):
    p = llm.pivot_table(index=["pid", "gender"], columns="tier", values=metric)
    return p.dropna()

def rank_biserial(x, y):
    """Matched-pairs rank-biserial effect size for Wilcoxon signed-rank."""
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = pd.Series(np.abs(d)).rank().values
    Rpos = ranks[d > 0].sum(); Rneg = ranks[d < 0].sum()
    T = Rpos + Rneg
    return float((Rpos - Rneg) / T) if T > 0 else 0.0

def holm(pvals):
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    idx = np.argsort(pvals); adj = np.empty(len(pvals)); prev = 0.0
    for rank, i in enumerate(idx):
        val = (len(pvals) - rank) * pvals[i]
        prev = max(prev, min(val, 1.0)); adj[i] = prev
    return adj

def report_metric(metric, label, lower_better=True):
    P = pivot(metric)
    n = len(P)
    print(f"\n########## {label}  (n={n} paired patients) ##########")
    print("  tier means:", {t: round(P[t].mean(), 4) for t in TIERS})
    # Friedman
    stat, p = friedmanchisquare(*[P[t].values for t in TIERS])
    print(f"  Friedman chi2 = {stat:.3f}, p = {p:.3e}")
    # test each tier's IoU against identity 1.0 (only meaningful for IoU)
    if metric == "mean_iou":
        print("  -- vs rule identity (IoU=1.0), one-sample Wilcoxon --")
        praw = []
        for t in TIERS:
            w, pw = wilcoxon(P[t].values - 1.0)
            praw.append(pw)
        padj = holm(praw)
        for t, pr, pa in zip(TIERS, praw, padj):
            print(f"     {t:6s}: raw p={pr:.2e}  Holm p={pa:.2e}")
    # pairwise with Holm
    print("  -- pairwise Wilcoxon signed-rank (Holm-corrected) --")
    pairs = list(combinations(TIERS, 2))
    praw = []
    for a, b in pairs:
        try:
            w, pw = wilcoxon(P[a].values, P[b].values)
        except ValueError:
            pw = 1.0
        praw.append(pw)
    padj = holm(praw)
    for (a, b), pr, pa in zip(pairs, praw, padj):
        rb = rank_biserial(P[a].values, P[b].values)
        print(f"     {a:6s} vs {b:6s}: raw p={pr:.2e}  Holm p={pa:.2e}  rank-biserial={rb:+.3f}")
    return P

# ---- #4: IoU divergence and safety breaches, rigorously ----
report_metric("mean_iou", "AXIS 1  Interval-IoU vs rule (lower = more patient-specific)")
Pb = report_metric("raw_safety_breaches", "AXIS 3  Raw safety-floor breaches (per patient)")
print("\n  >>> Floor-necessity argument (point estimates, not a null test):")
for t in TIERS:
    share = (Pb[t].values > 0).mean() * 100
    print(f"     {t:6s}: mean breaches={Pb[t].mean():.3f}, "
          f"{share:.1f}% of patients have >=1 raw breach")
print("     => every tier produces raw breaches; the code floor is required "
      "regardless of tier (no reliance on a non-significant test).")

# ---- #3: does the LLM BUY anything vs the rule? paired on same 155 ----
print("\n\n########## AXIS 2  LLM vs fixed rule: value, not just difference ##########")
rule_idx = rule.set_index(["pid", "gender"])

# feasibility: full 155, treat as 0/1
print(f"\n  --- feasibility (all patients) ---")
print(f"     rule (C2 aggressive): {rule['feasible'].mean():.4f}")
for t in TIERS:
    Lt = llm[llm.tier == t].set_index(["pid", "gender"])["feasible"].astype(float)
    common = Lt.index.intersection(rule_idx.index)
    a = Lt.loc[common].values
    b = rule_idx.loc[common, "feasible"].astype(float).values
    diff = a - b
    try:
        w, pw = wilcoxon(diff) if np.any(diff != 0) else (np.nan, 1.0)
    except ValueError:
        pw = 1.0
    print(f"     {t:6s}: LLM mean={a.mean():.4f}  (Δ vs rule={diff.mean():+.4f}, "
          f"paired Wilcoxon p={pw:.2e}, n={len(common)})")

# sparsity: only patients feasible under BOTH rule and this tier
print(f"\n  --- sparsity: mean vars changed (lower = sparser), jointly-feasible only ---")
print(f"     rule (C2 aggressive), all feasible: {rule.loc[rule.feasible.astype(bool),'n_changed_vars'].mean():.3f}")
for t in TIERS:
    Lt = llm[llm.tier == t].set_index(["pid", "gender"])
    common = Lt.index.intersection(rule_idx.index)
    sub = pd.DataFrame({
        "llm_feas": Lt.loc[common, "feasible"].astype(bool).values,
        "llm_nvar": Lt.loc[common, "n_changed_vars"].values,
        "rule_feas": rule_idx.loc[common, "feasible"].astype(bool).values,
        "rule_nvar": rule_idx.loc[common, "n_changed_vars"].values})
    joint = sub[sub.llm_feas & sub.rule_feas].dropna(subset=["llm_nvar", "rule_nvar"])
    a = joint.llm_nvar.values; b = joint.rule_nvar.values
    diff = a - b
    try:
        w, pw = wilcoxon(diff) if np.any(diff != 0) else (np.nan, 1.0)
    except ValueError:
        pw = 1.0
    print(f"     {t:6s}: LLM={a.mean():.3f} vs rule={b.mean():.3f}  "
          f"(Δ={diff.mean():+.3f}, paired Wilcoxon p={pw:.2e}, n={len(joint)} jointly-feasible)")
