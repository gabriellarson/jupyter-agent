#!/bin/bash
#SBATCH --job-name=finetuning-jupyter-agent
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --partition=hopper-prod
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --qos=high

module load cuda/12.4
set -x -e

source ~/.bashrc
source /fsx/$USER/venvs/finetuning/bin/activate

echo "Displaying GPU status with nvidia-smi:"
nvidia-smi

START_TIME=$(date +%s)
echo "START TIME: $(date)"

# Calculate distributed training parameters
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))                                                                                                                                                                                                                                                                           
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${NODELIST[0]}  # First node for main process
MASTER_PORT=6000
TRAIN_NODES=("${NODELIST[@]}")

# Triton compilation cache directories for performance
export TRITON_HOME=/scratch/train/triton/ 
export TRITON_CACHE_DIR=/scratch/train/.triton/

export NCCL_ASYNC_ERROR_HANDLING=1

export WANDB_PROJECT=jupyter-agent

export ACCELERATE_LOG_LEVEL=info
export TRANSFORMERS_VERBOSITY=info

# Training command with model-specific config and output paths
export CMD=" \
    train.py --config recipes/Qwen3-4B/sft.yaml \
    --run_name jupyter-agent-qwen3-4b_${SLURM_JOB_ID} \
    --output_dir /fsx/$USER/jupyter-agent/models/jupyter-agent-qwen3-4b-instruct-${SLURM_JOB_ID} \
    "

# Accelerate launcher for distributed training across multiple nodes/GPUs
export LAUNCHER="ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file recipes/accelerate_configs/zero2.yaml  \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend=c10d \
    "

NODELIST=$(IFS=,; echo "${TRAIN_NODES[*]}")

# srun arguments for distributed execution
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --nodes=$NUM_NODES \
    --ntasks=$NUM_NODES \
    --nodelist=$NODELIST
    "
# Execute training across all nodes - redirects stderr to stdout
clear; srun $SRUN_ARGS bash -c "$LAUNCHER $CMD" 2>&1
    
END_TIME=$(date +%s)
echo "END TIME: $(date)"
printf "TOTAL JOB TIME: %dh %dm %ds (%d seconds)\n" $(((END_TIME-START_TIME)/3600)) $(((END_TIME-START_TIME)%3600/60)) $(((END_TIME-START_TIME)%60)) $((END_TIME-START_TIME))
