# Fig. 2 — reconstruction of an open Heisenberg chain (Sec. IV.A)

n = 50 qubits, H = ½ Σ (XX + YY + ZZ), jump operators √γ_α σ_α with γ_x = γ_y = 1/2 and
γ_z = 1. The Choi state J(Λ_t) is evolved on the doubled chain with OQuPy (first-order
Trotter, Δt = 1e-3, relative SVD threshold 1e-7) and reconstructed from three-site window
marginals (w = 1) with CVXPY/SCS.

## Files

| File | Role |
|---|---|
| `reconstruct_noiseless.py` | panels (a), (b). The SLURM array index 0…9 selects t = 0.01…0.10 |
| `reconstruct_noisy.py` | panels (c), (d). The array index 0…8 selects σ from `np.logspace(-5, -2, 9)`; the environment variable `REP` (0…9) selects the noise realization; `num_steps = 100` gives t = 0.10 and `num_steps = 50` gives t = 0.05 |
| `_functions_.py`, `dynamics.py`, `data_processing.py` | helper routines (MPO marginals, `mpo_grow`, CPTP constraint, Pauli-weight sums) |
| `noise_gauss.py` | tomography-noise model: Gaussian noise on the Pauli expectation values of a window, hermitization, clipping of negative eigenvalues, renormalization; returns η_i = ½‖Ĵ_i − J_i‖₁ |
| `noise_shadow.py` | shadow-noise variant (imported by `reconstruct_noisy.py`, not used for the figure) |
| `job_noiseless.sh`, `job_noisy.sh` | SLURM array jobs for the two scripts (array index = time index or noise-level index, see above; set `REP` in the environment for the noisy runs) |
| `plot_fig2.py` | plots the four panels from `data/` |

## Data

`data/noiseless/`: `out_list_result.npy` (exact) and `out_list.npy` (reconstruction) hold
the per-weight sums Σ_{|α|=k} |χ_αα| for k = 0, 1, 2 (columns) at t = 0.01…0.10 (rows);
`l1_list.npy` holds Σ_{|α|=k} |χ^R_αα − χ^E_αα|.

`data/noisy_t0.10/`, `data/noisy_t0.05/`: `l1_reps.npz` contains `l1` (9 σ × 10 realizations
× 3 weights) and `gexact` (9 × 3); `gk_reps.npz` contains the corresponding G sums;
`eta_{i}_r1.npy` = [η_max, η_mean, σ] for realization 1 at the i-th noise level.

G_k and Δ_R of the figure are cumulative in k and follow from these arrays by `np.cumsum`
along the weight axis; panels (c), (d) show the mean and standard error over the ten
realizations against η = max_i η_i.

## Run

```
python plot_fig2.py      # -> fig2_panels.pdf / fig2_panels.png
sbatch job_noiseless.sh
REP=0 sbatch --export=ALL,REP job_noisy.sh
```
