#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --output=/path/to/your/code/%x_%j.log
#SBATCH --error=/path/to/your/code/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=your_partition
#SBATCH --mem=64G


# set -eo pipefail
SCRIPT_ARGS="$@"


ROOT_PATH=/path/to/your/code
cd $ROOT_PATH

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1


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


conda activate your_conda_env
echo "Python path: $(which python)"


echo "======= MAIN TASK ======="

# example submit cli
srun --export=ALL python baseline_main.py ${SCRIPT_ARGS}


EXIT_CODE=$?
echo "======= EXIT CODE: $EXIT_CODE ======="
exit $EXIT_CODE
