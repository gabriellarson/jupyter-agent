#!/bin/bash
#SBATCH --job-name=merge-meta-kaggle
#SBATCH --partition=hopper-cpu
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --output=./logs/metadata-merge/%x_%j.out
#SBATCH --error=./logs/metadata-merge/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1

export KAGGLEHUB_CACHE="/fsx/$USER/venvs/data-env/.cache/kagglehub/"

source ~/.bashrc
source /fsx/$USER/venvs/data-env/bin/activate

python3 merge_meta_sources.py