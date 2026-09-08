"""
exp2_optimizer.py
=================
Reviewer concern #1 (part b): is the injection-point result (post-hoc collapses,
pre-injection preserves) specific to DiCE's genetic search, or does it transfer
to a different counterfactual optimiser?

We implement a simple, fully independent random-search counterfactual generator
(no DiCE, no genetic algorithm) and reproduce the C1 vs C2 contrast with it.

  RS-C1 (post-hoc)     : sample perturbations from a broad, unconstrained box
                         (current +/- 30% per varying feature, matching the
                         rule's *default* width but WITHOUT Layer-1 / direction
                         coupling), keep those the model maps to Class 0, then
                         filter survivors against the admissible box G(x).
  RS-C2 (pre-injection): sample perturbations from INSIDE the admissible box
                         G(x) directly, keep those the model maps to a lower
                         class, with the same stepwise fallback 0->1->2.

Both draw the same number of samples with the same seed policy; the only
difference is whether G(x) constrains the sampling (C2) or only filters
afterwards (C1). If the paper's qualitative result is optimiser-independent,
RS-C1 should collapse and RS-C2 should preserve feasibility.
"""
import os, time, warnings, joblib
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import guardrail_core as gc

DATA = "../data"
OUT  = "../results/tables"
N_SAMPLES = 200          # candidates drawn per patient (per target class for C2)
SEED = 20260908

df_final = joblib.load(os.path.join(DATA, "df_final.pkl"))
ac = joblib.load(os.path.join(DATA, "agent_config.pkl"))
X, T, VARY = ac["X_features"], ac["target_col"], ac["vary_features"]

def sample_box(cur, box, rng, n):
    """Uniformly sample n candidate rows inside a per-feature box (dict f->[lo,hi]);
    non-varying features stay at current value."""
    cols = {}
    for f in X:
        if f in VARY and f in box:
            lo, hi = box[f]
            cols[f] = rng.uniform(lo, hi, size=n) if hi > lo else np.full(n, cur[f])
        else:
            cols[f] = np.full(n, cur[f])
    return pd.DataFrame(cols)[X]

def unconstrained_box(cur):
    """Broad box independent of Layer-1: current +/- 30% for varying features."""
    b = {}
    for f in VARY:
        c = cur[f]
        lo, hi = (min(c*0.7, c*1.3), max(c*0.7, c*1.3))
        b[f] = [lo, hi]
    return b

def in_admissible(row, box):
    for f in X:
        if f in box:
            lo, hi = box[f]
            v = float(row[f])
            if v < lo - 1e-6 or v > hi + 1e-6:
                return False
    return True

def run_gender(gender_code, name, model_file, val_file, limit=None):
    model = joblib.load(os.path.join(DATA, model_file))
    val = joblib.load(os.path.join(DATA, val_file))
    idx = val["X_val"][(val["y_val"] == 3) &
                       (model.predict(val["X_val"]) == 3)].index.tolist()
    if limit:
        idx = idx[:limit]
    print(f"{name}: {len(idx)} patients")
    rows = []
    t0 = time.time()
    for k, pid in enumerate(idx):
        rng = np.random.default_rng(SEED + hash(str(pid)) % 10000)
        q = val["X_val"].loc[[pid]][X]
        cur = {f: float(q.iloc[0][f]) for f in X}
        g_adm = gc.build_rule_guardrails(cur, X, "aggressive")   # admissible box G(x)

        # RS-C1: sample unconstrained -> predict Class0 -> filter by G(x)
        cand = sample_box(cur, unconstrained_box(cur), rng, N_SAMPLES)
        pred = model.predict(cand[X])
        c0_hits = cand[pred == 0]
        c1_feasible = False
        if len(c0_hits) > 0:
            c1_feasible = any(in_admissible(r, g_adm) for _, r in c0_hits.iterrows())
        rs_c1_unconstrained_feasible = len(c0_hits) > 0

        # RS-C2: sample inside G(x) -> predict lower class (fallback 0->1->2)
        c2_feasible, achieved = False, None
        for target in [0, 1, 2]:
            cand2 = sample_box(cur, g_adm, rng, N_SAMPLES)
            pred2 = model.predict(cand2[X])
            if (pred2 == target).sum() > 0:
                c2_feasible, achieved = True, target
                break
        rows.append({"pid": pid, "gender": name,
                     "rs_c0_feasible": rs_c1_unconstrained_feasible,
                     "rs_c1_posthoc_feasible": c1_feasible,
                     "rs_c2_preinj_feasible": c2_feasible,
                     "rs_c2_achieved": achieved})
        if (k + 1) % 20 == 0:
            print(f"  {name} {k+1}/{len(idx)} ({(time.time()-t0)/60:.1f} min)")
    return pd.DataFrame(rows)

if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    dfm = run_gender(1.0, "Male", "model_male.pkl", "val_male.pkl", limit)
    dff = run_gender(2.0, "Female", "model_female.pkl", "val_female.pkl", limit)
    df = pd.concat([dfm, dff], ignore_index=True)
    df.to_csv(os.path.join(OUT, "exp2_optimizer.csv"), index=False)
    print("\n=== Random-search optimiser (no DiCE): injection-point contrast ===")
    for g in ["Male", "Female"]:
        s = df[df.gender == g]
        print(f"{g:>7}: RS-C0 unconstrained feasible {s.rs_c0_feasible.mean()*100:5.1f}% | "
              f"RS-C1 post-hoc {s.rs_c1_posthoc_feasible.mean()*100:5.1f}% | "
              f"RS-C2 pre-injection {s.rs_c2_preinj_feasible.mean()*100:5.1f}%")
    print(f"pooled: RS-C1 {df.rs_c1_posthoc_feasible.mean()*100:.1f}% vs "
          f"RS-C2 {df.rs_c2_preinj_feasible.mean()*100:.1f}%  (n={len(df)})")
