# learning-global-channel

Code and data for the numerical results in

> Z. Liu and D. S. Wild, *Efficiently Learning Global Quantum Channels with Local Tomography* (2026).

The repository is organized by figure. Each directory contains the scripts that generated
the data, the data themselves, the plotting script, and a short README with the commands.

| Figure | Content | Directory |
|---|---|---|
| 2 | Reconstruction of an open Heisenberg chain, n = 50 (Sec. IV.A) | `fig2_open_system_reconstruction/` |
| 3 | Reconstruction of a noisy shallow circuit, n = 20 (Sec. IV.B) | `fig3_shallow_circuit_reconstruction/` |
| 4 | Choi-state CMI of the open Heisenberg chain (App. E.1) | `fig4_heisenberg_cmi/` |
| 5 | Choi-state CMI of the noisy shallow circuit (App. E.2) | `fig5_shallow_circuit_cmi/` |
| 6, 7 | Choi-state CMI of random Clifford circuits with depolarizing noise (App. E.3, F.4) | `fig6_fig7_random_clifford_cmi/` |

Fig. 1 is a schematic and has no associated code.

## Requirements

Python 3.10 with the packages listed in `requirements.txt` (numpy, scipy, matplotlib,
quimb, OQuPy, CVXPY with the SCS solver, stim; TeNPy for the Fig. 3 script). The plotting
scripts use matplotlib with `text.usetex = True` and therefore need a LaTeX installation.
The data generation for Figs. 2–4 was run on a SLURM cluster; the job scripts are included.

## Usage

Run each plotting script inside its directory; it reads the data by relative path and
writes the figure next to it. Data generation commands are given in the per-directory
READMEs. Quantities that involve random circuits or random noise samples are reproduced
as ensemble averages within the quoted statistical errors.

## License

MIT, see `LICENSE`.
