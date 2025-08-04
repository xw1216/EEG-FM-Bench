#!/bin/bash
#SBATCH --job-name=preproc
#SBATCH --output=/path/to/your/code/%x_%j.log
#SBATCH --error=/path/to/your/code/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=your_partition
#SBATCH --mem=684G


set -eo pipefail
SCRIPT_ARGS="$@"


ROOT_PATH=/path/to/your/code
cd $ROOT_PATH


echo "======= SLURM ENVIRONMENT ======="
echo "Job ID            : $SLURM_JOB_ID"
echo "Job Name          : $SLURM_JOB_NAME"
echo "Num Tasks         : $SLURM_NTASKS"
echo "Node List         : $SLURM_JOB_NODELIST"
echo "CPUs per Task     : $SLURM_CPUS_PER_TASK"
echo "GPUs per Task     : $SLURM_GPUS_PER_TASK"


echo "======= SYSTEM SETUP ======="
# print variables
echo "Current working path: $ROOT_PATH"
echo "Script arguments: $SCRIPT_ARGS"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"


echo "======= CONDA ENV ======="

conda activate your_conda_env
echo "Python path: $(which python)"


echo "======= MAIN TASK ======="
# example submit cli
srun --export=ALL python preproc.py ${SCRIPT_ARGS}


EXIT_CODE=$?
echo "======= EXIT CODE: $EXIT_CODE ======="
exit $EXIT_CODE
