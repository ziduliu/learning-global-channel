# Figs. 6 and 7 — Choi-state CMI of random Clifford circuits with depolarizing noise (App. E.3, F.4)

Brickwork circuits of depth t with two-qubit gates drawn uniformly from the two-qubit
Clifford group (`stim.Tableau.random(2)`), alternating bonds in consecutive layers with a
random parity of the first layer, and single-qubit depolarizing noise
(1−p)ρ + p/3 (XρX + YρY + ZρZ) on every qubit after every layer. The entropies of the reduced
Choi states are evaluated exactly with the stabilizer method of App. F.4, implemented in
`noisy_clifford_cmi.py` (stabilizer generators from the tableau, backward propagation to
obtain the space-time support K(g), expectation values (1 − 4p/3)^K, Walsh–Hadamard
inversion for the sign distribution, S = |R| − k_R + H in bits, factorization over connected
components).

## Files

| File | Role |
|---|---|
| `noisy_clifford_cmi.py` | engine; its `main` produced Fig. 6(a)–(d): n = 40, a = c = 1, sliding-window maximum with a margin of two sites, one CSV per chunk of realizations in `final/` (`full*_d{t}_p{p}_c{k}.csv`, 50 or 200 realizations per chunk, at least 1000 per (t, p)). `--selftest` compares with a dense simulation for n = 5 |
| `final/` | data of Fig. 6(a)–(d) |
| `wg_multip.py` | Fig. 6(e)–(g): centered tripartition, b = 2, a = 1…5: `python wg_multip.py --n 40 --depth {4,6,8} --ps 0.01 0.02 0.05 --reps 334 --seed {0,1,2} --csv wg1k_d{t}_s{s}.csv` |
| `wg1k_d{4,6,8}_s{0,1,2}.csv`, `.log` | data of Fig. 6(e)–(g) |
| `plot_fig6_random_clifford.py` | builds Fig. 6 from `final/` and `wg1k_*.csv` |
| `full_partition_cmi.py`, `run_fullpart.sh` | Fig. 7: n = 12, a + b + c = n, maximum over the position of B, 200 realizations: `python full_partition_cmi.py {t} {p} 0 200 12` |
| `fullpart_results/` | data of Fig. 7 |
| `plot_fig7_full_partition.py` | builds Fig. 7 |
| `clifford_cmi.py`, `noiseless/` | the same circuits without noise (n = 40, 100 realizations) |

## Run

```
python plot_fig6_random_clifford.py    # -> random_clifford_cmi_v2.pdf
python plot_fig7_full_partition.py     # -> fullpart_cmi.pdf
python noisy_clifford_cmi.py --selftest --p 0.1
```
