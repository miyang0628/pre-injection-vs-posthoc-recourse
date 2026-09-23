"""
run_baselines.py
Run the two admissibility-aware baselines (B1 soft-penalty, B2 Monte-Carlo)
on the same 155 model-confirmed Class-3 patients as the injection experiment,
under the aggressive delegation guardrail (matching C2's headline setting).
Caches to results/tables/_baselines_{male,female}.csv.
"""
import warnings; warnings.filterwarnings("ignore")
import os, time, joblib, numpy as np, pandas as pd
import experiment_core as ec, guardrail_core as gc, baseline_core as bc

DATA = "../data"; TAB = "../results/tables"
os.makedirs(TAB, exist_ok=True)
ac = joblib.load(f"{DATA}/agent_config.pkl")
df_final = joblib.load(f"{DATA}/df_final.pkl")
X = ac["X_features"]; T = ac["target_col"]; VARY = ac["vary_features"]

KEEP = ["feasible", "achieved_class", "fallback_depth", "n_valid_cands",
        "n_changed_vars", "mean_l1_dist", "diversity",
        "anthro_viol", "energy_viol", "conflict_viol", "any_viol"]
B1_EXTRA = ["reached_target", "reached_but_outside"]


def run_gender(code, name, mfile, vfile):
    cache = f"{TAB}/_baselines_{name.lower()}.csv"
    if os.path.exists(cache):
        print(f"{name}: cached"); return pd.read_csv(cache)
    model = joblib.load(f"{DATA}/{mfile}")
    val = joblib.load(f"{DATA}/{vfile}")
    df_stable = df_final[df_final["Sex"] == code].copy().astype(float)
    idx = val["X_val"][(val["y_val"] == 3) &
                       (model.predict(val["X_val"]) == 3)].index.tolist()
    print(f"{name}: {len(idx)} patients")
    rows = []; t0 = time.time()
    for k, pid in enumerate(idx):
        q = val["X_val"].loc[[pid]][X]
        cur = {f: float(q.iloc[0][f]) for f in X}
        g = gc.build_rule_guardrails(cur, X, "aggressive")

        b2 = bc.eval_B2_mc_admissible(ec, model, q, df_stable, X, VARY, g, cur,
                                      n_draw=4000, seed=0)
        rows.append({"pid": pid, "gender": name, "cond": "B2_mc",
                     **{kk: b2[kk] for kk in KEEP}})

        b1 = bc.eval_B1_soft_penalty(ec, model, q, df_stable, X, VARY, g, cur,
                                     n_iter=4000, n_restart=4, seed=0)
        rec = {"pid": pid, "gender": name, "cond": "B1_soft",
               **{kk: b1[kk] for kk in KEEP}}
        for e in B1_EXTRA:
            rec[e] = b1.get(e, np.nan)
        rows.append(rec)

        if (k + 1) % 10 == 0:
            print(f"  {name} {k+1}/{len(idx)} ({(time.time()-t0)/60:.1f} min)")
    df_g = pd.DataFrame(rows)
    df_g.to_csv(cache, index=False)
    print(f"{name}: done {(time.time()-t0)/60:.1f} min")
    return df_g


if __name__ == "__main__":
    m = run_gender(1.0, "Male", "model_male.pkl", "val_male.pkl")
    f = run_gender(2.0, "Female", "model_female.pkl", "val_female.pkl")
    allb = pd.concat([m, f], ignore_index=True)
    allb.to_csv(f"{TAB}/baselines_cohort.csv", index=False)
    print("\n=== FEASIBILITY SUMMARY ===")
    for cond in ["B1_soft", "B2_mc"]:
        s = allb[allb["cond"] == cond]
        print(f"{cond}: feasible {s['feasible'].mean()*100:.1f}% (n={len(s)})")
    b1 = allb[allb["cond"] == "B1_soft"]
    if "reached_target" in b1:
        print(f"B1 reached target class: {b1['reached_target'].mean()*100:.1f}%")
        print(f"B1 reached-but-outside-region: {b1['reached_but_outside'].mean()*100:.1f}%")
