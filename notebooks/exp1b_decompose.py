"""
exp1b_decompose.py
==================
Reviewer concern #1a, redesigned. Instead of only scaling the box symmetrically
(which does not isolate the mechanism), we DECOMPOSE why post-hoc filtering
collapses, by measuring what fraction of unconstrained DiCE candidates satisfies
each admissibility constraint TYPE, cumulatively:

  (0) reaches Class 0 at all              (DiCE's own success)
  (1) + within magnitude box (range only) : per-feature [lo,hi] ignoring sign
  (2) + no-increase on intake/anthro      : direction constraints
  (3) + anthropometric coupling           : BMI/waist/weight move together
  (4) = full admissible region G(x)        (== paper's C1 filter)

For each level we report the candidate-level survival rate and the
patient-level feasibility (>=1 candidate survives). This shows the collapse is
driven by DIRECTION/COUPLING constraints that no amount of box-widening can fix,
which is exactly why post-hoc filtering is structurally doomed while
pre-injection (which samples inside the coupled region) is not.
"""
import os, time, warnings, joblib
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import guardrail_core as gc
import experiment_core as ec

DATA, OUT = "../data", "../results/tables"
df_final = joblib.load(os.path.join(DATA, "df_final.pkl"))
ac = joblib.load(os.path.join(DATA, "agent_config.pkl"))
X, T, VARY = ac["X_features"], ac["target_col"], ac["vary_features"]

ANTHRO = ["BMI", "WaistCirc", "Weight"]
NOINC  = ["Energy_kcal", "Sodium_mg", "Carb_g", "Sugar_g"]  # must not increase

def levels_for_candidate(orig, row, cur):
    """Return booleans for cumulative admissibility levels L1..L4."""
    g = gc.build_rule_guardrails(cur, X, "aggressive")
    # L1: magnitude box only (within [lo,hi] per feature)
    L1 = all(g[f][0]-1e-6 <= float(row[f]) <= g[f][1]+1e-6 for f in X)
    # L2: no-increase on intake+anthro (<= current within tol)
    L2 = all(float(row[f]) <= cur[f] + 1e-6 for f in ANTHRO + NOINC)
    # L3: anthropometric coupling -- all three reduce by a similar ratio
    ratios = [(cur[f]-float(row[f]))/cur[f] if cur[f]>1e-9 else 0.0 for f in ANTHRO]
    # coupled if all reductions non-negative and spread small
    L3 = (min(ratios) >= -1e-6) and (max(ratios)-min(ratios) <= 0.15)
    return L1, (L1 and L2), (L1 and L2 and L3)

def run_gender(gc_code, name, mf, vf, limit=None):
    model = joblib.load(os.path.join(DATA, mf)); val = joblib.load(os.path.join(DATA, vf))
    df_stable = df_final[df_final["Sex"] == gc_code].copy().astype(float)
    idx = val["X_val"][(val["y_val"] == 3) & (model.predict(val["X_val"]) == 3)].index.tolist()
    if limit: idx = idx[:limit]
    exp = ec.make_dice(model, df_stable, X, T)
    print(f"{name}: {len(idx)} patients", flush=True)
    rows = []; t0 = time.time()
    for k, pid in enumerate(idx):
        q = val["X_val"].loc[[pid]][X]; orig = q.iloc[0]
        cur = {f: float(q.iloc[0][f]) for f in X}
        cand = None
        for _ in range(5):
            try:
                cf = exp.generate_counterfactuals(q, total_CFs=4, desired_class=0,
                    features_to_vary=VARY, proximity_weight=0.2, sparsity_weight=0.1)
                d = cf.cf_examples_list[0].final_cfs_df
                if d is not None and len(d) > 0: cand = d.copy(); break
            except Exception: continue
        rec = {"pid": pid, "gender": name, "reached_c0": cand is not None,
               "n_cand": 0 if cand is None else len(cand)}
        if cand is None:
            for lv in ["L1_range","L2_dir","L3_couple"]:
                rec[f"cand_{lv}"] = 0; rec[f"feas_{lv}"] = False
        else:
            n = len(cand); s1=s2=s3=0
            for _, r in cand.iterrows():
                l1,l2,l3 = levels_for_candidate(orig, r, cur)
                s1+=l1; s2+=l2; s3+=l3
            rec["cand_L1_range"]=s1/n; rec["cand_L2_dir"]=s2/n; rec["cand_L3_couple"]=s3/n
            rec["feas_L1_range"]=s1>0; rec["feas_L2_dir"]=s2>0; rec["feas_L3_couple"]=s3>0
        rows.append(rec)
        if (k+1)%20==0: print(f"  {name} {k+1}/{len(idx)} ({(time.time()-t0)/60:.1f}m)", flush=True)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv)>1 else None
    d = pd.concat([run_gender(1.0,"Male","model_male.pkl","val_male.pkl",limit),
                   run_gender(2.0,"Female","model_female.pkl","val_female.pkl",limit)],
                  ignore_index=True)
    d.to_csv(os.path.join(OUT,"exp1b_decompose.csv"), index=False)
    print("\n=== Cumulative admissibility of unconstrained DiCE candidates (n=%d) ===" % len(d))
    print("Level                         cand-survival   patient-feasible")
    print(f"reached Class 0 (DiCE)          {d.reached_c0.mean()*100:6.1f}%          {d.reached_c0.mean()*100:6.1f}%")
    for lv,lab in [("L1_range","+ magnitude box"),("L2_dir","+ no-increase dir"),("L3_couple","+ anthro coupling = G(x)")]:
        print(f"{lab:28s}   {d['cand_'+lv].mean()*100:6.1f}%          {d['feas_'+lv].mean()*100:6.1f}%")
    print("\n(last row 'patient-feasible' == paper C1 feasibility)")
