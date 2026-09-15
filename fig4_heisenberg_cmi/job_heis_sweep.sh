#!/bin/bash
#SBATCH --job-name=heis_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=160:00:00
#SBATCH --output=heis_sweep.out
#SBATCH --error=heis_sweep.err
module purge; module load anaconda/3/2023.03
source activate $HOME/.conda/envs/quimb_env
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
python -u heis_sweep_prod.py --n 50 --eps 0.0 --dt 0.01 --epsrel 1e-12 \
  --times 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 \
  --ac-list 2 3 \
  --csv cmi_depol_n50_sweep.csv --no-plot
