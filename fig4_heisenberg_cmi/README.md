# Fig. 4 — Choi-state CMI of the open Heisenberg chain (App. E.1)

Same model as Fig. 2 (n = 50, H = ½ Σ (XX + YY + ZZ), γ_x = γ_y = 1/2, γ_z = 1). The
doubled-chain Choi state is evolved with OQuPy (first-order Trotter, Δt = 1e-2, relative
SVD threshold 1e-12). The entropies of windows of up to seven sites are computed by exact
diagonalization (in bits), and the CMI is maximized over all window positions at least six
sites away from the chain ends.

## Files

| File | Role |
|---|---|
| `heis_sweep_prod.py` | data generation (window sweep) |
| `job_heis_sweep.sh` | SLURM job with the parameters of the figure: `--n 50 --eps 0.0 --dt 0.01 --epsrel 1e-12 --times 0.01 … 0.10 --ac-list 2 3` |
| `cmi_depol_n50_sweep.csv` | data of the figure (`t, a, b, cmi_max_bulk, cmi_center, s_max, …`) |
| `cmi_mpo_depol.py`, `job_heis_eps0.sh`, `cmi_depol_n50_eps0_field0.csv` | same computation for the central window only |
| `plot_heis_cmi3.py` | plotting script |

## Run

```
python plot_heis_cmi3.py cmi_depol_n50_sweep.csv    # -> heisenberg_cmi3.pdf
sbatch job_heis_sweep.sh
```
