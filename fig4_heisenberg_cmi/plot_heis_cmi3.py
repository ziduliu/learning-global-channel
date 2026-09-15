"""Three-panel Heisenberg CMI figure, styled after the shallow-circuit Fig. 5.

(a) I_J vs b at t = 0.04, 0.06, 0.08, 0.10 with exponential envelopes;
(b) I_J vs t at b = 1, 2, 3;
(c) I_J vs a = c (1, 2, 3) at b = 1, at the four display times.

Usage: python plot_heis_cmi3.py [results.csv]  (default: production field-free CSV)
"""
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif",
    "font.size": 11, "axes.labelsize": 13, "legend.fontsize": 9.5,
})

CSV = sys.argv[1] if len(sys.argv) > 1 else "cmi_depol_n50_eps0_field0.csv"
rows = list(csv.DictReader(open(CSV)))
dat = {}   # (t, a, b) -> I_max
for r in rows:
    dat[(round(float(r["t"]), 3), int(r["a"]), int(r["b"]))] = float(r["cmi_max_bulk"])

T_SHOW = [0.04, 0.06, 0.08, 0.10]
ts_all = sorted({k[0] for k in dat})
colors = plt.cm.viridis(np.linspace(0.85, 0.15, len(T_SHOW)))
tcol = dict(zip(T_SHOW, colors))
tmark = dict(zip(T_SHOW, ["o", "s", "^", "D"]))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.2, 2.8), dpi=200)

# --- (a): I vs b at four times, with exponential envelopes ---
# display up to b = 4 only (b = 5 is at/near the numerical floor)
bs = sorted({k[2] for k in dat if k[1] == 1 and k[2] <= 4})
for t in T_SHOW:
    ys = np.array([dat[(t, 1, b)] for b in bs])
    # tightest exponential upper envelope on points above the numerical floor
    ok = ys > 1e-12
    bfit, yfit = np.array(bs)[ok], ys[ok]
    kappa = -np.polyfit(bfit, np.log(yfit), 1)[0]
    # lift the intercept over ALL plotted points (incl. floor-level ones,
    # which are displayed clipped at 1e-14), so the envelope is a true
    # pointwise upper bound on everything shown
    a0 = np.max(np.maximum(ys, 1e-14) * np.exp(kappa * np.array(bs)))
    lab = r"$t = %g\ (\xi = %.2f)$" % (t, 1.0 / kappa)
    ax1.plot(bs, np.maximum(ys, 1e-14), tmark[t], ms=5, color=tcol[t],
             markerfacecolor="none", markeredgewidth=1.3, label=lab)
    bb = np.linspace(bs[0] - 0.3, bs[-1] + 0.3, 100)
    ax1.plot(bb, a0 * np.exp(-kappa * bb), "--", color=tcol[t], lw=0.9)
ax1.set_yscale("log")
ax1.set_xticks(bs)
ax1.set_xlabel(r"$b$")
ax1.set_ylabel(r"$I_J$")
ax1.legend(loc="upper right", frameon=True, fontsize=8)

# --- (b): I vs t at b = 1, 2, 3 ---
for b, m, c in [(1, "o", "C0"), (2, "s", "C1"), (3, "^", "C2")]:
    ys = [max(dat[(t, 1, b)], 1e-14) for t in ts_all]
    ax2.plot(ts_all, ys, m, ms=5, color=c, markerfacecolor="none",
             markeredgewidth=1.3, label=rf"$b = {b}$")
ax2.set_yscale("log")
ax2.set_xlabel(r"$t$")
ax2.set_ylabel(r"$I_J$")
ax2.legend(loc="lower right", frameon=True, fontsize=8)

# --- (c): I vs a = c at b = 1 ---
avals = sorted({k[1] for k in dat if k[2] == 1})
have_c = len(avals) > 1
if have_c:
    for t in T_SHOW:
        ys = [dat[(t, a_, 1)] for a_ in avals]
        ax3.plot(avals, ys, tmark[t], ms=5, color=tcol[t],
                 markerfacecolor="none", markeredgewidth=1.3,
                 label=rf"$t = {t:g}$")
    ax3.set_yscale("log")
    ax3.set_xticks(avals)
    ax3.set_xlabel(r"$a = c$")
    ax3.set_ylabel(r"$I_J$")
    ax3.set_ylim(1e-5, 5e-3)
    ax3.legend(loc="upper left", frameon=True, fontsize=8, ncol=2)
else:
    ax3.text(0.5, 0.5, "a-scan pending", ha="center", va="center",
             transform=ax3.transAxes)

for ax, tag in zip((ax1, ax2, ax3), "abc"):
    ax.text(-0.22, 1.04, rf"({tag})", transform=ax.transAxes, fontsize=13)

fig.tight_layout()
fig.savefig("heisenberg_cmi3.pdf")
fig.savefig("heisenberg_cmi3.png")
print(f"avals={avals}, times={ts_all}, wrote heisenberg_cmi3.pdf/png")
