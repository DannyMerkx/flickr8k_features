#!/bin/bash
#SBATCH --job-name=vgg
#SBATCH --array=0-9
#SBATCH -p RM-shared
#SBATCH -t 4:00:00
#SBATCH --mem=2G
#SBATCH --ntasks-per-node 10
#SBATCH --nodes 1
printf -v id "%02d" $SLURM_ARRAY_TASK_ID

module load python
module load anaconda

python main.py x$id
