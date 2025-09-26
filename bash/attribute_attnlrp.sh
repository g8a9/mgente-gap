#!/bin/bash
#SBATCH --job-name=mgente
#SBATCH --output=./logs/%A.out
#SBATCH --time=08:00:00
#SBATCH --gpus=4
#SBATCH --qos=gpu-medium
#SBATCH --partition=a6000

source venv/bin/activate
echo `whereis python`

# source /etc/profile.d/02-lmod.sh
module unload cuda
module load cuda

model="Qwen/Qwen2.5-72B-Instruct"
output_dir="../results-interim-gente-xai/attributions/attnlrp/set-g"
mkdir -p $output_dir


python src/attribute.py \
    --input_files ../results-interim-gente-xai/attributions/data/set-g_it_* \
    --model_name $model \
    --output_dir $output_dir
