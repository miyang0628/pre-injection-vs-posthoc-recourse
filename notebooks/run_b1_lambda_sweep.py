"""
run_b1_lambda_sweep.py
Sweep the soft-penalty weight lambda for B1 to show its feasibility does NOT
climb to the pre-injection level no matter how hard the constraint is pushed
into the objective. Pre-empts the "you under-weighted the penalty" objection.
Runs on a fixed 40-patient subsample (male+female) for speed; seed fixed.
"""
import warnings; warnings.filterwarnings("ignore")
import os, time, joblib, numpy as np, pandas as pd
import experiment_core as ec, guardrail_core as gc, baseline_core as bc

DATA = "../data"; TAB = "../results/tables"
ac = joblib.load(f"{DATA}/agent_config.pkl")
df_final = joblib.load(f"{DATA}/df_final.pkl")
X = ac["X_features"]; T = ac["target_col"]; VARY = ac["vary_features"]

LAMBDAS = [2.0, 8.0, 32.0, 128.0, 512.0]


def patch_lambda(lam):
    """Return a B1 evaluator with a specific lambda by monkey-editing the
    module-level default is awkward; instead re-implement the call with lam."""
    def _run(model, query, df_stable, cur, g):
        # replicate eval_B1_soft_penalty but with injectable lambda
        rng = np.random.default_rng(0)
        res = ec._empty_result()
        orig = query.iloc[0]; feats = X
        vary = [f for f in VARY if f in feats]
        stds = df_stable[feats].std().replace(0, 1.0)
        x0 = np.array([float(orig[f]) for f in feats]); fidx = {f: i for i, f in enumerate(feats)}
        def region_penalty(x):
            pen = 0.0
            for f in feats:
                lo, hi = g[f]; v = x[fidx[f]]
                if v < lo: pen += ((lo - v) / (abs(stds[f]) + 1e-9)) ** 2
                elif v > hi: pen += ((v - hi) / (abs(stds[f]) + 1e-9)) ** 2
            return pen
        reached = outside = False
        for target_class in [0, 1, 2]:
            best_x, best_L = None, np.inf
            for r in range(4):
                x = x0.copy()
                for f in vary:
                    lo, hi = g[f]; x[fidx[f]] = np.clip(x[fidx[f]], lo, hi)
                curL = None
                for it in range(0, 4000, 64):
                    cand = np.repeat(x[None, :], 64, axis=0)
                    for f in vary:
                        j = fidx[f]; cand[:, j] += rng.normal(0, 0.15 * (abs(stds[f]) + 1e-9), 64)
                    pt = model.predict_proba(pd.DataFrame(cand, columns=feats))[:, target_class]
                    Ls = -pt + lam * np.array([region_penalty(c) for c in cand])
                    m = int(np.argmin(Ls))
                    if curL is None or Ls[m] < curL: curL = Ls[m]; x = cand[m].copy()
                if curL < best_L: best_L, best_x = curL, x.copy()
            row = {f: best_x[fidx[f]] for f in feats}
            pred = int(model.predict(pd.DataFrame([row], columns=feats))[0])
            inside = bc._in_region(row, g, feats)
            if pred == target_class:
                reached = True
                if not inside: outside = True
            if pred == target_class and inside:
                return {"feasible": True, "reached": True, "outside": False}
        return {"feasible": False, "reached": reached, "outside": outside}
    return _run


if __name__ == "__main__":
    # 20 male + 20 female
    subs = []
    for code, mfile, vfile in [(1.0, "model_male.pkl", "val_male.pkl"),
                               (2.0, "model_female.pkl", "val_female.pkl")]:
        model = joblib.load(f"{DATA}/{mfile}"); val = joblib.load(f"{DATA}/{vfile}")
        df_stable = df_final[df_final["Sex"] == code].copy().astype(float)
        idx = val["X_val"][(val["y_val"] == 3) &
                           (model.predict(val["X_val"]) == 3)].index.tolist()[:20]
        subs.append((model, val, df_stable, idx))

    rows = []
    for lam in LAMBDAS:
        run = patch_lambda(lam)
        feas = reach = outside = n = 0
        t0 = time.time()
        for model, val, df_stable, idx in subs:
            for pid in idx:
                q = val["X_val"].loc[[pid]][X]; cur = {f: float(q.iloc[0][f]) for f in X}
                g = gc.build_rule_guardrails(cur, X, "aggressive")
                r = run(model, q, df_stable, cur, g)
                n += 1; feas += r["feasible"]; reach += r["reached"]; outside += r["outside"]
        rows.append({"lambda": lam, "n": n, "feasible_pct": round(feas / n * 100, 1),
                     "reached_pct": round(reach / n * 100, 1),
                     "outside_pct": round(outside / n * 100, 1)})
        print(f"lambda={lam:6.1f}: feas {feas/n*100:5.1f}%  reached {reach/n*100:5.1f}%  "
              f"outside {outside/n*100:5.1f}%  ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(f"{TAB}/b1_lambda_sweep.csv", index=False)
    print("saved b1_lambda_sweep.csv")
