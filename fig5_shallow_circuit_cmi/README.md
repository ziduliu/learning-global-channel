# Fig. 5 — Choi-state CMI of the noisy shallow circuit (App. E.2)

Same channel as Fig. 3 (n = 20, γ = 1e-2, ε = rγ with r = 0, 1, …, 10). The Choi state is
built as an MPS with a 16-dimensional physical leg, the entropies of the reduced Choi
states are obtained by exact diagonalization (in bits), and I_J^max is the maximum over the
window positions s₀ − 1, s₀, s₀ + 1 with s₀ = ⌊(n − w)/2⌋. Ten disorder realizations
(seeds 1234 + r) are used for every ε.

## Files

| File | Role |
|---|---|
| `circuit_cmi.py` | engine and driver for panels (a), (b): `python circuit_cmi.py --n 20 --eps 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 --gamma 0.01 --reals 10 --bmax 4` → `circuit_cmi_dense_eps.json`. `--selftest` compares with a dense simulation for n = 5 |
| `fix_b4.py` | recomputes the b = 4 entries of the JSON |
| `cluster/ascan_cluster.py`, `cluster/job_ascan.sh` | panel (c): a = c = 1, 2, 3 at b = 1; SLURM array task = 10·realization + ε index; place `circuit_cmi.py` in the same directory |
| `cluster_results/` | raw cluster outputs `ascan_r{r}_e{e}.json` |
| `merge_ascan.py` | merges `cluster_results/` into `circuit_cmi_ascan.json` |
| `circuit_cmi_dense_eps.json`, `circuit_cmi_ascan.json` | data of the figure |
| `plot_circuit_cmi3.py` | plotting script |
| `run_dense_eps.log` | log of the data-generation run |

## Run

```
python plot_circuit_cmi3.py     # -> circuit_coherent_cmi3.pdf
```
