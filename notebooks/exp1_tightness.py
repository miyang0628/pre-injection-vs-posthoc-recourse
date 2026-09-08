"""
exp1_tightness.py
=================
Reviewer concern #1 (part a): is the C1 (post-hoc) collapse a general property
of tightening the admissible region, or an artefact of one particular DiCE
configuration?

We run the SAME unconstrained DiCE search once per patient (exactly as C0/C1 in
the paper), then re-filter the identical candidate set against a family of
admissible boxes G_alpha(x) obtained by scaling the rule band's half-width by a
factor alpha about the patient's current value. alpha = 1.0 reproduces the
paper's post-hoc filter; alpha > 1 loosens the box, alpha < 1 tightens it.

Because the unconstrained candidate set is fixed per patient, the only thing
that varies across alpha is the filter width. This isolates "how tight must the
box be before post-hoc filtering fails" as a pure function of tightness, with
the optimiser held constant. Output: a feasibility-vs-tightness curve.
"""
import os, time, warnings, joblib
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import guardrail_core as gc
import experiment_core as ec

DATA = "../data"
OUT  = "../results/tables"
os.makedirs(OUT, exist_ok=True)

df_final = joblib.load(os.path.join(DATA, "df_final.pkl"))
ac = joblib.load(os.path.join(DATA, "agent_config.pkl"))
X, T, VARY = ac["X_features"], ac["target_col"], ac["vary_features"]

ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

def scaled_box(cur, alpha):
    """Scale the rule band half-width by alpha about the current value,
    then re-impose Layer-1 safety so every box remains admissible."""
    g_rule = gc.build_rule_guardrails(cur, X, "aggressive")
    g = {}
    for f in X:
        lo, hi = g_rule[f]
        c = cur[f]
        # widen/narrow symmetrically about current value
        g[f] = [c + (lo - c) * alpha, c + (hi - c) * alpha]
    return gc.apply_layer1_safety(g, cur)

def in_box(row, box):
    for f in X:
        lo, hi = box[f]
        v = float(row[f])
        if v < lo - 1e-6 or v > hi + 1e-6:
            return False
    return True

def run_gender(gender_code, name, model_file, val_file, limit=None):
    model = joblib.load(os.path.join(DATA, model_file))
    val = joblib.load(os.path.join(DATA, val_file))
    df_stable = df_final[df_final["Sex"] == gender_code].copy().astype(float)
    idx = val["X_val"][(val["y_val"] == 3) &
                       (model.predict(val["X_val"]) == 3)].index.tolist()
    if limit:
        idx = idx[:limit]
    exp = ec.make_dice(model, df_stable, X, T)
    print(f"{name}: {len(idx)} patients")
    rows = []
    t0 = time.time()
    for k, pid in enumerate(idx):
        q = val["X_val"].loc[[pid]][X]
        cur = {f: float(q.iloc[0][f]) for f in X}
        # one unconstrained search, reused for every alpha (as in C0/C1)
        cand = None
        for _ in range(5):
            try:
                cf = exp.generate_counterfactuals(
                    q, total_CFs=4, desired_class=0, features_to_vary=VARY,
                    proximity_weight=0.2, sparsity_weight=0.1)
                d = cf.cf_examples_list[0].final_cfs_df
                if d is not None and len(d) > 0:
                    cand = d.copy(); break
            except Exception:
                continue
        rec = {"pid": pid, "gender": name,
               "c0_feasible": cand is not None,
               "n_cand": 0 if cand is None else len(cand)}
        for a in ALPHAS:
            if cand is None:
                rec[f"posthoc_a{a}"] = False
            else:
                box = scaled_box(cur, a)
                surv = sum(in_box(r, box) for _, r in cand.iterrows())
                rec[f"posthoc_a{a}"] = surv > 0
        rows.append(rec)
        if (k + 1) % 20 == 0:
            print(f"  {name} {k+1}/{len(idx)} ({(time.time()-t0)/60:.1f} min)")
    return pd.DataFrame(rows)

if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    dfm = run_gender(1.0, "Male", "model_male.pkl", "val_male.pkl", limit)
    dff = run_gender(2.0, "Female", "model_female.pkl", "val_female.pkl", limit)
    df = pd.concat([dfm, dff], ignore_index=True)
    df.to_csv(os.path.join(OUT, "exp1_tightness.csv"), index=False)
    print("\n=== Post-hoc feasibility vs tightness (alpha) ===")
    print(f"{'alpha':>6} {'posthoc feasibility':>22}")
    for a in ALPHAS:
        print(f"{a:>6} {df[f'posthoc_a{a}'].mean()*100:>20.1f}%")
    print(f"\nalpha=1.0 reproduces paper's C1. n={len(df)}")
