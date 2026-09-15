"""Full-partition CMI figure (2x3): panels by t, curves by p.

(a)-(c): I_J vs b for the full-chain tripartition (a+b+c = n = 12,
maximum over the position of B), one panel per t in {4, 6, 8},
curves p = 0.01, 0.02, 0.05 with tail fits over b >= t.
"""
import csv
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif",
    "font.size": 11, "axes.labelsize": 13, "legend.fontsize": 8.5,
})

fp = {}
for f in glob.glob("fullpart_results/*.csv"):
    for r in csv.DictReader(open(f)):
        fp.setdefault((int(r["depth"]), float(r["p"]), int(r["b"])), []).append(
            float(r["ivn_max"]))

PS = [0.01, 0.02, 0.05]
TS = [4, 6, 8]
colors = {0.01: "C0", 0.02: "C1", 0.05: "C2"}
marks = {0.01: "o", 0.02: "s", 0.05: "^"}

fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.8), dpi=200)

# ---- I vs b, panels by t, curves by p ----
for ax, t in zip(axes, TS):
    for p in PS:
        bs = sorted(b for (tt, pp, b) in fp if tt == t and pp == p)
        m = np.array([np.mean(fp[(t, p, b)]) for b in bs])
        ok = np.array(bs) >= t
        kappa = -np.polyfit(np.array(bs)[ok], np.log(m[ok]), 1)[0]
        lab = r"$p = %g\ (\xi = %.2f)$" % (p, 1.0 / kappa)
        ax.plot(bs, m, marks[p], ms=5, color=colors[p],
                markerfacecolor="none", markeredgewidth=1.3, label=lab)
        bfit = np.linspace(1, 10.3, 50)
        afit = np.max(m[ok] * np.exp(kappa * np.array(bs)[ok]))
        ax.plot(bfit, afit * np.exp(-kappa * bfit), "--", color=colors[p],
                lw=0.8, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel(r"$b$")
    ax.set_ylim(1e-8, 200)
    ax.set_xticks(range(1, 11))
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.set_title(rf"$t = {t}$", fontsize=12)
axes[0].set_ylabel(r"$I_J$")

for ax, tag in zip(axes, "abc"):
    ax.text(-0.16, 1.04, rf"({tag})", transform=ax.transAxes, fontsize=13)

fig.tight_layout()
fig.savefig("fullpart_cmi.pdf")
fig.savefig("fullpart_cmi.png")
print("wrote fullpart_cmi.pdf/png (panels by t, curves by p)")
