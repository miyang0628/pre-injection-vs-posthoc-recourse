"""
exp1c_violation_breakdown.py
============================
Reviewer #1a, final form. For every unconstrained DiCE candidate produced across
the whole cohort, record which admissibility constraint TYPES it violates,
INDEPENDENTLY (not cumulatively). This quantifies *why* post-hoc filtering
removes everything and shows the collapse is driven by direction/coupling
geometry, not by a tunable box width.

Constraint types (each evaluated on its own):
  range   : at least one feature outside its magnitude band [lo,hi]
  noinc   : at least one intake/anthro feature INCREASED above current
  couple  : BMI/waist/weight do not move together as a coupled reduction
  floor   : at least one Layer-1 absolute floor crossed (e.g. energy<500)
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
NOINC  = ["Energy_kcal", "Sodium_mg", "Carb_g", "Sugar_g"]
FLOORS = {"Energy_kcal":500.0, "Sodium_mg":800.0, "Protein_g":30.0,
          "Potassium_mg":500.0, "Carb_g":30.0, "Fiber_g":5.0}

def viol_types(row, cur):
    g = gc.build_rule_guardrails(cur, X, "aggressive")
    v_range = any(not (g[f][0]-1e-6 <= float(row[f]) <= g[f][1]+1e-6) for f in X)
    v_noinc = any(float(row[f]) > cur[f] + 1e-6 for f in ANTHRO + NOINC)
    ratios = [(cur[f]-float(row[f]))/cur[f] if cur[f]>1e-9 else 0.0 for f in ANTHRO]
    v_couple = (min(ratios) < -1e-6) or (max(ratios)-min(ratios) > 0.15)
    v_floor = any(float(row[f]) < fl - 1e-6 for f, fl in FLOORS.items())
    return v_range, v_noinc, v_couple, v_floor

def run_gender(gc_code, name, mf, vf, limit=None):
    model = joblib.load(os.path.join(DATA, mf)); val = joblib.load(os.path.join(DATA, vf))
    df_stable = df_final[df_final["Sex"] == gc_code].copy().astype(float)
    idx = val["X_val"][(val["y_val"] == 3) & (model.predict(val["X_val"]) == 3)].index.tolist()
    if limit: idx = idx[:limit]
    exp = ec.make_dice(model, df_stable, X, T)
    print(f"{name}: {len(idx)} patients", flush=True)
    cand_rows = []; t0 = time.time()
    for k, pid in enumerate(idx):
        q = val["X_val"].loc[[pid]][X]; cur = {f: float(q.iloc[0][f]) for f in X}
        cand = None
        for _ in range(5):
            try:
                cf = exp.generate_counterfactuals(q, total_CFs=4, desired_class=0,
                    features_to_vary=VARY, proximity_weight=0.2, sparsity_weight=0.1)
                d = cf.cf_examples_list[0].final_cfs_df
                if d is not None and len(d) > 0: cand = d.copy(); break
            except Exception: continue
        if cand is None: continue
        for _, r in cand.iterrows():
            vr, vn, vc, vf_ = viol_types(r, cur)
            cand_rows.append({"pid": pid, "gender": name, "v_range": vr,
                              "v_noinc": vn, "v_couple": vc, "v_floor": vf_,
                              "admissible": not (vr or vn or vc or vf_)})
        if (k+1)%20==0: print(f"  {name} {k+1}/{len(idx)} ({(time.time()-t0)/60:.1f}m)", flush=True)
    return pd.DataFrame(cand_rows)

if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv)>1 else None
    d = pd.concat([run_gender(1.0,"Male","model_male.pkl","val_male.pkl",limit),
                   run_gender(2.0,"Female","model_female.pkl","val_female.pkl",limit)],
                  ignore_index=True)
    d.to_csv(os.path.join(OUT,"exp1c_violation_breakdown.csv"), index=False)
    N = len(d)
    print(f"\n=== Independent violation rates among {N} unconstrained DiCE candidates ===")
    for c,lab in [("v_range","outside magnitude box"),("v_noinc","increased a reduce-only feature"),
                  ("v_couple","broke anthropometric coupling"),("v_floor","crossed a Layer-1 floor")]:
        print(f"  {lab:34s}: {d[c].mean()*100:5.1f}% of candidates")
    print(f"  {'fully admissible (none of above)':34s}: {d['admissible'].mean()*100:5.1f}% of candidates")
