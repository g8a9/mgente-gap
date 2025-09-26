#!/bin/bash
#SBATCH --job-name=gnt_eval
#SBATCH --output=./logs/%A.out
#SBATCH --time=04:00:00
#SBATCH --gpus=4
#SBATCH --qos=gpu-short
#SBATCH --partition=a6000
#SSBATCH --array=3ash

# outline_cache=~/myscratch/caches/outlines
# mkdir -p $outline_cache
# export OUTLINES_CACHE_DIR=$outline_cache
export VLLM_USE_V1=0
# vact gente
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
rm -rf /mnt/home/giuseppe/.cache/outlines/*
python local_evaluate.py \
    -u user \
    -m "google/gemma-3-27b-it"
echo "Done"
    # -m "meta-llama/Llama-3.1-8B-Instruct"
    # -m "Qwen/Qwen2.5-72B-Instruct"
