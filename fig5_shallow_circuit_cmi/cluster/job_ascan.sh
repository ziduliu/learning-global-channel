#!/bin/bash
#SBATCH --job-name=ascan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-99
#SBATCH --output=ascan_out_%a
#SBATCH --error=ascan_err_%a
module purge; module load anaconda/3/2023.03
source activate $HOME/.conda/envs/quimb_env
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python -u ascan_cluster.py ${SLURM_ARRAY_TASK_ID}
