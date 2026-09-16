#!/bin/bash
#SBATCH --job-name=fig2_noisy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-8
#SBATCH --output=out_file_%a
#SBATCH --error=err_file_%a
module purge; module load anaconda/3/2023.03
source activate $HOME/.conda/envs/quimb_env
export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
python -u reconstruct_noisy.py ${SLURM_ARRAY_TASK_ID}
