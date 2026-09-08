"""
make_review_figures.py
======================
Build the two grayscale figures answering reviewer concerns #1 and #3,
in the same visual style as the paper's existing figures.
"""
import os, warnings, joblib
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib as mpl
import matplotlib.pyplot as plt

FIG = "../results/figures"; TAB = "../results/tables"
os.makedirs(FIG, exist_ok=True)
mpl.rcParams.update({"savefig.dpi": 600, "font.family": "DejaVu Sans",
    "font.size": 10, "axes.edgecolor": "#000000", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False})
GREY = ["#BDBDBD", "#888888", "#565656", "#1A1A1A"]
HATCH = ["", "//", "..", "xx"]

def save(fig, name):
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"{name}.{ext}")
        fig.savefig(p, dpi=600, bbox_inches="tight", format=ext)
        print("saved", p)

# ---------- Figure A: optimiser-independence + violation mechanism ----------
opt = pd.read_csv(os.path.join(TAB, "exp2_optimizer.csv"))
vb  = pd.read_csv(os.path.join(TAB, "exp1c_violation_breakdown.csv"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))

# left: random-search optimiser C1 vs C2 (pooled + per gender)
labels = ["RS-C0\n(unconstr.)", "RS-C1\n(post-hoc)", "RS-C2\n(pre-inj.)"]
male = [opt[opt.gender=="Male"].rs_c0_feasible.mean(),
        opt[opt.gender=="Male"].rs_c1_posthoc_feasible.mean(),
        opt[opt.gender=="Male"].rs_c2_preinj_feasible.mean()]
female = [opt[opt.gender=="Female"].rs_c0_feasible.mean(),
          opt[opt.gender=="Female"].rs_c1_posthoc_feasible.mean(),
          opt[opt.gender=="Female"].rs_c2_preinj_feasible.mean()]
x = np.arange(3); w = 0.38
ax1.bar(x-w/2, [v*100 for v in male],   w, color=GREY[1], edgecolor="black", label="Male", linewidth=0.8)
ax1.bar(x+w/2, [v*100 for v in female], w, color=GREY[3], edgecolor="black", label="Female", linewidth=0.8)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8.5)
ax1.set_ylabel("Feasibility (%)"); ax1.set_ylim(0, 105)
ax1.set_title("(a) Non-DiCE random-search optimiser", fontsize=9.5)
ax1.legend(frameon=False, fontsize=8.5, loc="center right")
for xi, (mv, fv) in enumerate(zip(male, female)):
    ax1.text(xi-w/2, mv*100+2, f"{mv*100:.0f}", ha="center", fontsize=7.5)
    ax1.text(xi+w/2, fv*100+2, f"{fv*100:.0f}", ha="center", fontsize=7.5)

# right: independent violation rates among unconstrained DiCE candidates
vlabels = ["outside\nrange box", "increased\nreduce-only", "broke\ncoupling", "crossed\nfloor"]
vvals = [vb.v_range.mean()*100, vb.v_noinc.mean()*100, vb.v_couple.mean()*100, vb.v_floor.mean()*100]
xb = np.arange(4)
bars = ax2.bar(xb, vvals, color=GREY[2], edgecolor="black", linewidth=0.8)
for b, h in zip(bars, HATCH): b.set_hatch(h)
ax2.set_xticks(xb); ax2.set_xticklabels(vlabels, fontsize=8)
ax2.set_ylabel("Candidates violating (%)"); ax2.set_ylim(0, 108)
ax2.set_title(f"(b) Why post-hoc filtering fails (n={len(vb)} candidates)", fontsize=9.5)
for xi, v in enumerate(vvals):
    ax2.text(xi, v+2, f"{v:.0f}", ha="center", fontsize=8)
fig.tight_layout()
save(fig, "fig_optimizer_independence")
plt.close(fig)

# ---------- Figure B: LLM vs rule value (feasibility + sparsity) ----------
llm = pd.read_csv(os.path.join(TAB, "llm_tier_divergence.csv"))
m = pd.read_csv(os.path.join(TAB, "_cohort_male.csv"))
f = pd.read_csv(os.path.join(TAB, "_cohort_female.csv"))
coh = pd.concat([m, f], ignore_index=True)
rule = coh[(coh.cond=="C2_pre_rule") & (coh.deleg=="aggressive")]
TIERS = ["weak", "mid", "strong"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
# feasibility: rule vs each tier
feas = [rule.feasible.mean()*100] + [llm[llm.tier==t].feasible.mean()*100 for t in TIERS]
xf = np.arange(4); fl = ["rule\n(C2)", "weak", "mid", "strong"]
bars = ax1.bar(xf, feas, color=[GREY[0]]+[GREY[2]]*3, edgecolor="black", linewidth=0.8)
ax1.set_xticks(xf); ax1.set_xticklabels(fl, fontsize=8.5)
ax1.set_ylabel("Feasibility (%)"); ax1.set_ylim(0, 105)
ax1.set_title("(a) Recourse feasibility: rule vs LLM tiers", fontsize=9.5)
for xi, v in enumerate(feas): ax1.text(xi, v+2, f"{v:.1f}", ha="center", fontsize=8)

# sparsity: mean vars changed, rule vs each tier (jointly-feasible)
rule_sp = rule.loc[rule.feasible.astype(bool), "n_changed_vars"].mean()
sp = [rule_sp] + [llm[(llm.tier==t)&(llm.feasible.astype(bool))].n_changed_vars.mean() for t in TIERS]
bars = ax2.bar(xf, sp, color=[GREY[0]]+[GREY[2]]*3, edgecolor="black", linewidth=0.8)
ax2.set_xticks(xf); ax2.set_xticklabels(fl, fontsize=8.5)
ax2.set_ylabel("Mean variables changed"); ax2.set_ylim(0, max(sp)*1.25)
ax2.set_title("(b) Recourse sparsity (lower = sparser)", fontsize=9.5)
for xi, v in enumerate(sp): ax2.text(xi, v+0.3, f"{v:.1f}", ha="center", fontsize=8)
fig.tight_layout()
save(fig, "fig_llm_vs_rule_value")
plt.close(fig)
print("done")
