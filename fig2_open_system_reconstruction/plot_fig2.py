"""Fig. 2: reconstruction of the open Heisenberg chain (n = 50). Run inside this directory."""
import glob
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIMES = np.arange(1, 11) * 0.01
K_LABELS = [r"$k=0$", r"$k=1$", r"$k=2$"]
MARK = ["o", "s", "^"]


def cumulative(a):
    return np.cumsum(a, axis=-1)


def load_noisy(folder):
    z = np.load(f"{folder}/l1_reps.npz")
    l1, gexact = z["l1"], z["gexact"]                     # (9, 10, 3), (9, 3)
    delta = cumulative(l1) / cumulative(gexact)[:, None, :]  # (9, 10, 3)
    eta = np.zeros(l1.shape[0])
    for f in glob.glob(f"{folder}/eta_*_r1.npy"):
        i = int(re.search(r"eta_(\d+)_r1", f).group(1))
        eta[i] = np.load(f)[0]
    return eta, delta.mean(axis=1), delta.std(axis=1, ddof=1) / np.sqrt(delta.shape[1])


exact = cumulative(np.load("data/noiseless/out_list_result.npy"))   # G_k exact
recon = cumulative(np.load("data/noiseless/out_list.npy"))          # G_k reconstructed
delta_nf = cumulative(np.load("data/noiseless/l1_list.npy")) / exact  # Delta_R noise-free

fig, ax = plt.subplots(2, 2, figsize=(8, 6))
ax = ax.ravel()

for k in range(3):
    ax[0].semilogy(TIMES, exact[:, k], "-", color=f"C{k}", label=K_LABELS[k])
    ax[0].semilogy(TIMES, recon[:, k], MARK[k], color=f"C{k}", mfc="none")
ax[0].set_xlabel(r"$t$"); ax[0].set_ylabel(r"$G_k$"); ax[0].legend()
ax[0].set_title("(a)", loc="left")

for k in range(3):
    ax[1].semilogy(TIMES, delta_nf[:, k], MARK[k] + "-", color=f"C{k}", mfc="none", label=K_LABELS[k])
ax[1].set_xlabel(r"$t$"); ax[1].set_ylabel(r"$\Delta_R$"); ax[1].legend()
ax[1].set_title("(b)", loc="left")

for panel, folder, t in [(2, "data/noisy_t0.10", 0.10), (3, "data/noisy_t0.05", 0.05)]:
    eta, mean, sem = load_noisy(folder)
    for k in range(3):
        ax[panel].errorbar(eta, mean[:, k], yerr=sem[:, k], fmt=MARK[k] + "-", color=f"C{k}",
                           mfc="none", capsize=2, label=K_LABELS[k])
    ax[panel].set_xscale("log"); ax[panel].set_yscale("log")
    ax[panel].set_xlabel(r"$\eta$"); ax[panel].set_ylabel(r"$\Delta_R$"); ax[panel].legend()
    ax[panel].set_title(f"({'cd'[panel-2]}) $t = {t}$", loc="left")

fig.tight_layout()
fig.savefig("fig2_panels.pdf")
fig.savefig("fig2_panels.png", dpi=150)
