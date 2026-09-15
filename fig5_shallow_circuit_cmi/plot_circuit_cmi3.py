"""Three-panel shallow-circuit CMI figure: I vs b, I vs eps, I vs a.

Panels (a,b) use the complete dense eps-scan (10 eps x 10 reals).
Panel (c) uses the a-scan; falls back to the single completed realization
in run_ascan.log until the 10-real statistics are available.
"""
import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif",
    "font.size": 11, "axes.labelsize": 13, "legend.fontsize": 9.5,
})

res = json.load(open("circuit_cmi_dense_eps.json"))
eps_list = sorted({v["eps"] for v in res.values() if v["eps"] > 0})
bs = sorted({v["b"] for v in res.values()})

colors = plt.cm.viridis(np.linspace(0.85, 0.15, len(eps_list)))
markers = ["o", "s", "^", "D"]

# ---- a-scan data: prefer JSON (cluster/local complete run), else log ----
ascan = {}   # (eps, a, b) -> (mean, sem)
try:
    aj = json.load(open("circuit_cmi_ascan.json"))
    for v in aj.values():
        ascan[(v["eps"], v["a"], v["b"])] = (v["mean"], v["sem"])
    A_NREALS = max(v["nreals"] for v in aj.values())
except FileNotFoundError:
    rows = {}
    for line in open("run_ascan.log"):
        m = re.match(r"real=(\d+) eps=([\d.]+)\s+wall", line)
        if not m:
            continue
        eps = float(m.group(2))
        for am, bm, val in re.findall(r"a(\d)b(\d)=([-\d.e+]+)", line):
            rows.setdefault((eps, int(am), int(bm)), []).append(float(val))
    for k, v in rows.items():
        ascan[k] = (float(np.mean(v)),
                    float(np.std(v) / np.sqrt(len(v))) if len(v) > 1 else 0.0)
    A_NREALS = max(len(v) for v in rows.values()) if rows else 0

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.2, 2.8), dpi=200)

# --- (a): I_J vs b ---
SHOW_A = [e for e in [0.01, 0.02, 0.05, 0.1] if e in eps_list]
cmapA = {e: colors[np.argmin(np.abs(np.array(eps_list) - e))] for e in SHOW_A}
for (eps, m) in zip(SHOW_A, markers):
    c = cmapA[eps]
    ys = [res[f"g0.01_e{eps}_b{b}"]["mean"] for b in bs]
    es = [res[f"g0.01_e{eps}_b{b}"]["sem"] for b in bs]
    ax1.errorbar(bs, ys, yerr=es, marker=m, ms=5, color=c, ls="none",
                 capsize=2, markerfacecolor="none",
                 label=rf"$\epsilon = {eps}$")
ax1.set_yscale("log")
ax1.set_xticks(bs)
ax1.set_xlabel(r"$b$")
ax1.set_ylabel(r"$I_J$")
ax1.set_ylim(1e-12, 1.0)
ax1.legend(loc="upper right", frameon=True, ncol=2, fontsize=7)

# --- (b): I_J(b=1,2) vs eps, with power-law fits ---
from matplotlib.lines import Line2D
xg = np.array([eps_list[0], eps_list[-1]])
handles2 = []
for b, m, c in [(1, "o", "C0"), (2, "s", "C1")]:
    ys = np.array([res[f"g0.01_e{e}_b{b}"]["mean"] for e in eps_list])
    es = np.array([res[f"g0.01_e{e}_b{b}"]["sem"] for e in eps_list])
    ax2.errorbar(eps_list, ys, yerr=es, marker=m, ms=5, color=c, ls="none",
                 capsize=2, markerfacecolor="none")
    k, logC = np.polyfit(np.log10(eps_list), np.log10(ys), 1)
    ax2.plot(xg, 10 ** logC * xg ** k, "--", color=c, lw=0.9)
    handles2.append(Line2D([], [], color=c, marker=m, ls="--", lw=0.9, ms=5,
                           markerfacecolor="none",
                           label=rf"$b = {b}\ (\propto \epsilon^{{{k:.2f}}})$"))
    print(f"fit b={b}: I = {10 ** logC:.3g} * eps^{k:.3f}")
ax2.set_xscale("log"); ax2.set_yscale("log")
ticks = [0.01, 0.02, 0.05, 0.1]
ax2.set_xticks(ticks)
ax2.set_xticklabels([str(e) for e in ticks])
ax2.minorticks_off()
ax2.set_xlabel(r"$\epsilon$")
ax2.set_ylabel(r"$I_J$")
ax2.legend(handles=handles2, loc="upper left", frameon=True)

# --- (c): I_J vs a at b=1 ---
a_eps_all = sorted({k[0] for k in ascan})
SHOW = [e for e in [0.01, 0.02, 0.05, 0.1] if e in a_eps_all] or a_eps_all
cmap3 = {e: colors[np.argmin(np.abs(np.array(eps_list) - e))] for e in a_eps_all}
for e, m in zip(SHOW, ["o", "s", "^", "D"]):
    sel1 = sorted([(a, *ascan[(e, a, 1)]) for (ee, a, b) in ascan
                   if ee == e and b == 1])
    a1, y1, s1 = zip(*sel1)
    ax3.errorbar(a1, y1, yerr=s1, marker=m, ms=5, color=cmap3[e], ls="none",
                 capsize=2, markerfacecolor="none",
                 label=rf"$\epsilon = {e}$")
ax3.set_yscale("log")
ax3.set_xticks([1, 2, 3])
ax3.set_xlabel(r"$a = c$")
ax3.set_ylabel(r"$I_J$")
ax3.set_ylim(5e-4, 3.0)
ax3.legend(loc="upper left", frameon=True, fontsize=8, ncol=2)

for ax, tag in zip((ax1, ax2, ax3), "abc"):
    ax.text(-0.22, 1.04, rf"({tag})", transform=ax.transAxes, fontsize=13)

fig.tight_layout()
fig.savefig("circuit_coherent_cmi3.pdf")
fig.savefig("circuit_coherent_cmi3.png")
print(f"panel (c) realizations: {A_NREALS}")
