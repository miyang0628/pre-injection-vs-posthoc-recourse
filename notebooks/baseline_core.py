"""
baseline_core.py
================
Admissibility-aware recourse baselines added in revision, to answer the
"weak comparison" review point. The original C0/C1/C2 conditions are three
handlings of ONE DiCE pipeline; a reviewer can reasonably say that
pre-injection beating post-hoc filtering is unsurprising. These baselines are
*independent* recourse generators that also try to respect admissibility, so
the injection-point result is tested against genuine alternatives rather than
against a straw man.

Baselines
---------
  B1  soft_penalty : gradient-free local search that treats the admissible
                     region as a SOFT penalty added to the class objective
                     (the Wachter / actionable-recourse style: constraints in
                     the loss, not as a hard box). This is the standard
                     alternative to hard pre-injection. Feasibility = did the
                     returned optimum land inside the admissible region.

  B2  mc_admissible : optimiser-free Monte-Carlo search that samples uniformly
                     INSIDE the admissible region (the same region C2 injects)
                     and keeps draws the classifier moves to a lower tier. This
                     promotes the paper's Monte-Carlo probe to a first-class
                     baseline and shows the feasibility result is not a DiCE
                     artefact.

Both reuse guardrail_core so the admissible region is identical to C1/C2;
only the SEARCH differs. All three share the same stepwise fallback
(Class 0 -> 1 -> 2) and the same per-call budget so the comparison is matched.
"""

import numpy as np
import pandas as pd
import guardrail_core as gc


# ---- shared helpers ---------------------------------------------------------

def _in_region(row, guardrails, feats):
    for f in feats:
        lo, hi = guardrails[f]
        v = float(row[f])
        if v < lo - 1e-6 or v > hi + 1e-6:
            return False
    return True


def _quality_from_experiment_core(ec, cf_df, orig, df_stable, X_FEATURES):
    return ec._quality(cf_df.copy(), orig, df_stable, X_FEATURES)


def _empty(ec):
    return ec._empty_result()


# ---- B1  soft-penalty local search -----------------------------------------

def eval_B1_soft_penalty(ec, model, query, df_stable, X_FEATURES, VARY,
                         guardrails, cur, n_iter=4000, n_restart=4, seed=0):
    """
    Gradient-free actionable-recourse baseline. Minimise
        L(x) = -P(target | x) + lambda * penalty_outside_region(x)
    by random-restart coordinate perturbation over the VARY features. The
    admissible region enters as a soft penalty (lambda), NOT as a hard box: the
    search may leave the region and pays for it, exactly the "constraints in the
    objective" alternative to pre-injection. Feasibility is judged by whether
    the returned optimum is actually inside the region and classified into a
    lower tier -- i.e. we grant B1 the same admissibility test as C2.

    Budget: n_iter*n_restart forward evaluations per target class; comparable to
    DiCE's genetic budget (population*generations) used by C0/C1/C2.
    """
    rng = np.random.default_rng(seed)
    res = _empty(ec)
    orig = query.iloc[0]
    feats = X_FEATURES
    vary = [f for f in VARY if f in feats]

    # search scale per feature: population std (fallback to 1) as step size
    stds = df_stable[feats].std().replace(0, 1.0)
    lam = 8.0  # penalty weight; large enough that leaving the region is costly

    x0 = np.array([float(orig[f]) for f in feats], dtype=float)
    fidx = {f: i for i, f in enumerate(feats)}

    def region_penalty(x):
        pen = 0.0
        for f in feats:
            lo, hi = guardrails[f]
            v = x[fidx[f]]
            if v < lo:
                pen += ((lo - v) / (abs(stds[f]) + 1e-9)) ** 2
            elif v > hi:
                pen += ((v - hi) / (abs(stds[f]) + 1e-9)) ** 2
        return pen

    def prob_target(xmat, target):
        p = model.predict_proba(pd.DataFrame(xmat, columns=feats))
        return p[:, target]

    reached_any = False       # did the optimum ever hit the target class?
    reached_but_outside = False  # hit the class but left the region?
    for target_class in [0, 1, 2]:
        best_x, best_L = None, np.inf
        for r in range(n_restart):
            x = x0.copy()
            # start with a small nudge toward the region interior
            for f in vary:
                lo, hi = guardrails[f]
                x[fidx[f]] = np.clip(x[fidx[f]], lo, hi)
            cur_L = None
            batch = 64
            it = 0
            while it < n_iter:
                cand = np.repeat(x[None, :], batch, axis=0)
                for f in vary:
                    j = fidx[f]
                    cand[:, j] = cand[:, j] + rng.normal(0, 0.15 * (abs(stds[f]) + 1e-9), batch)
                pt = prob_target(cand, target_class)
                Ls = -pt + lam * np.array([region_penalty(c) for c in cand])
                m = int(np.argmin(Ls))
                if cur_L is None or Ls[m] < cur_L:
                    cur_L = Ls[m]; x = cand[m].copy()
                it += batch
            if cur_L < best_L:
                best_L, best_x = cur_L, x.copy()

        # evaluate the returned optimum
        row = {f: best_x[fidx[f]] for f in feats}
        pred = int(model.predict(pd.DataFrame([row], columns=feats))[0])
        inside = _in_region(row, guardrails, feats)
        if pred == target_class:
            reached_any = True
            if not inside:
                reached_but_outside = True
        if pred == target_class and inside:
            cf_df = pd.DataFrame([row])
            q = _quality_from_experiment_core(ec, cf_df, orig, df_stable, feats)
            res.update({"feasible": True, "achieved_class": target_class,
                        "fallback_depth": target_class, "n_valid_cands": 1,
                        "reached_target": True, "reached_but_outside": False, **q})
            return res
    res.update({"reached_target": reached_any,
                "reached_but_outside": reached_but_outside})
    return res


# ---- B2  Monte-Carlo admissible search -------------------------------------

def eval_B2_mc_admissible(ec, model, query, df_stable, X_FEATURES, VARY,
                          guardrails, cur, n_draw=4000, seed=0):
    """
    Optimiser-free baseline: sample uniformly INSIDE the admissible region and
    keep draws the classifier moves to the target tier. Independent of DiCE
    entirely. Same region, same fallback, matched budget (n_draw forward
    evaluations per class). Returns up to 4 surviving candidates to mirror
    DiCE's total_CFs=4.
    """
    rng = np.random.default_rng(seed)
    res = _empty(ec)
    orig = query.iloc[0]
    feats = X_FEATURES
    vary = set(f for f in VARY if f in feats)

    los = np.array([guardrails[f][0] for f in feats])
    his = np.array([guardrails[f][1] for f in feats])
    x0 = np.array([float(orig[f]) for f in feats])
    fidx = {f: i for i, f in enumerate(feats)}
    vary_mask = np.array([f in vary for f in feats])

    for target_class in [0, 1, 2]:
        draws = rng.uniform(los, his, size=(n_draw, len(feats)))
        # non-vary features pinned to the patient's current value
        draws[:, ~vary_mask] = x0[~vary_mask]
        preds = model.predict(pd.DataFrame(draws, columns=feats))
        hit = draws[preds == target_class]
        if len(hit) > 0:
            # keep up to 4, closest to x0 in std-normalised L1 (proximity)
            stds = df_stable[feats].std().replace(0, 1.0).values
            d = np.abs((hit - x0) / stds).sum(axis=1)
            keep = hit[np.argsort(d)[:4]]
            cf_df = pd.DataFrame(keep, columns=feats)
            q = _quality_from_experiment_core(ec, cf_df, orig, df_stable, feats)
            res.update({"feasible": True, "achieved_class": target_class,
                        "fallback_depth": target_class,
                        "n_valid_cands": len(keep), **q})
            return res
    return res
