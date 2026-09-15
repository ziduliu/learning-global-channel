#!/bin/bash
#SBATCH --job-name=mpo_glue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=32G
#SBATCH --time=168:00:00
#SBATCH --array=0-19              # <<< 20 jobs 0,1,...,19  (matches NOISE_LIST=linspace(0,0.2,20))
#SBATCH --output=out_file_%a  # %A=main job ID, %a=array ID
#SBATCH --error=err_file_%a

module purge
module load anaconda/3/2023.03
source activate $HOME/.conda/envs/quimb_env

python -u main.py ${SLURM_ARRAY_TASK_ID}
