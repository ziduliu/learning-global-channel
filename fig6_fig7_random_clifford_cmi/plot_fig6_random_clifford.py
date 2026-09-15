"""Fig. 6 (random Clifford circuits with depolarizing noise), built from the data in this directory.

(a)      heatmaps I_J(t, b) for p = 0.01, 0.02, 0.05  [unchanged content]
(b)-(d)  I_J vs b at t = 4, 6, 8; curves p = 0.01, 0.02, 0.05,
         tail fits over b >= t (unweighted least squares in log space)
(e)-(g)  I_J vs a at fixed b = 2, t = 4, 6, 8; curves p (centered windows)

Data: final/*.csv (heatmap + cuts, 10^3 realizations),
      wg1k_d{4,6,8}_s{0,1,2}.csv (a-dependence, 1002 realizations).
"""
import csv
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif",
    "font.size": 11, "axes.labelsize": 13, "legend.fontsize": 8,
})

# ---- load cut/heatmap data (chunk-pooled means) ----
acc = {}
for f in glob.glob("final/*.csv"):
    for r in csv.DictReader(open(f)):
        acc.setdefault((int(r["depth"]), float(r["p"]), int(r["b"])), []).append(
            float(r["ivn_mean"]))
cut = {k: float(np.mean(v)) for k, v in acc.items()}

# ---- load a-dependence data ----
wg = {}
for f in glob.glob("wg1k_d[468]_s[012].csv"):
    for r in csv.DictReader(open(f)):
        wg.setdefault((int(r["depth"]), float(r["p"]), int(r["a"]), int(r["b"])),
                      []).append(float(r["ivn_mean"]))

PS = [0.01, 0.02, 0.05]
TS = [4, 6, 8]
pcol = {0.01: "C0", 0.02: "C1", 0.05: "C2"}
pmark = {0.01: "o", 0.02: "s", 0.05: "^"}
FLOOR = 1e-8

fig = plt.figure(figsize=(10.2, 8.6), dpi=200)
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.0],
                      hspace=0.52, wspace=0.28,
                      left=0.09, right=0.9, top=0.95, bottom=0.065)

# ---- row 0: heatmaps ----
hax = [fig.add_subplot(gs[0, i]) for i in range(3)]
for ax, p in zip(hax, PS):
    Z = np.full((10, 10), np.nan)
    for it, t in enumerate(range(1, 11)):
        for ib, b in enumerate(range(1, 11)):
            if (t, p, b) in cut:
                Z[it, ib] = max(cut[(t, p, b)], FLOOR)
    pc = ax.pcolormesh(np.arange(0.5, 11), np.arange(0.5, 11), Z,
                       norm=LogNorm(vmin=FLOOR, vmax=1.0), cmap="magma",
                       rasterized=True)
    ax.set_title(rf"$p = {p:g}$", fontsize=12)
    ax.set_xlabel(r"$b$")
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.set_yticks([2, 4, 6, 8, 10])
hax[0].set_ylabel(r"$t$")
cax = fig.add_axes([0.915, hax[2].get_position().y0, 0.015,
                    hax[2].get_position().height])
cb = fig.colorbar(pc, cax=cax)
cb.set_label(r"$I_J$")

# ---- row 1: cuts, panels by t, curves by p ----
bax = [fig.add_subplot(gs[1, i]) for i in range(3)]
for ax, t in zip(bax, TS):
    for p in PS:
        bs = [b for b in range(1, 11) if (t, p, b) in cut]
        m = np.array([cut[(t, p, b)] for b in bs])
        okb = np.array([b for b in bs if t <= b <= 10])
        okm = np.array([cut[(t, p, b)] for b in okb])
        kappa = -np.polyfit(okb, np.log(okm), 1)[0]
        lab = r"$p = %g\ (\xi = %.2f)$" % (p, 1.0 / kappa)
        ax.plot(bs, m, pmark[p], ms=4.5, color=pcol[p], markerfacecolor="none",
                markeredgewidth=1.2, label=lab)
        a0 = np.exp(np.polyfit(okb, np.log(okm), 1)[1])
        bb = np.linspace(1, 10.3, 50)
        ax.plot(bb, a0 * np.exp(-kappa * bb), "--", color=pcol[p], lw=0.8,
                alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel(r"$b$")
    ax.set_ylim(1e-6, 3e2)
    ax.set_xticks(range(1, 11))
    ax.set_title(rf"$t = {t}$", fontsize=12)
    ax.legend(loc="upper right", frameon=True)
bax[0].set_ylabel(r"$I_J$")

# ---- row 2: a-dependence at b = 2, panels by t, curves by p ----
aax = [fig.add_subplot(gs[2, i]) for i in range(3)]
B_FIX = 2
for ax, t in zip(aax, TS):
    for p in PS:
        avals = sorted({k[2] for k in wg if k[0] == t and k[1] == p
                        and k[3] == B_FIX})
        m = [np.mean(wg[(t, p, a, B_FIX)]) for a in avals]
        ax.plot(avals, m, pmark[p], ms=4.5, color=pcol[p],
                markerfacecolor="none", markeredgewidth=1.2,
                label=rf"$p = {p:g}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$a = c$")
    ax.set_xticks(range(1, 6))
    ax.set_title(rf"$t = {t}$", fontsize=12)
    ax.legend(loc="lower right", frameon=True)
aax[0].set_ylabel(r"$I_J$")

hax[0].text(-0.18, 1.06, r"(a)", transform=hax[0].transAxes, fontsize=13)
for ax, tag in zip(bax + aax, "bcdefg"):
    ax.text(-0.18, 1.06, rf"({tag})", transform=ax.transAxes, fontsize=13)

fig.savefig("random_clifford_cmi_v2.pdf")
fig.savefig("random_clifford_cmi_v2.png")
print("wrote random_clifford_cmi_v2.pdf/png")
