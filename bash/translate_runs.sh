#!/bin/bash
#SBATCH --job-name=tr_mgente
#SBATCH --output=./logs/%A-%a.out
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --qos=boost_qos_dbg
#SBATCH --partition=boost_usr_prod
#SBATCH --array=458,459,460,464,465,467
#SBATCH --account=iscrc_itallm_0
#SBATCH --mem=64G

set -e

module unload cuda
module load cuda/12.3

# source /etc/profile.d/02-lmod.sh
source $WORK/mgente/bin/activate

outline_cache=$WORK/caches/$SLURM_ARRAY_TASK_ID
mkdir -p $outline_cache
export OUTLINES_CACHE_DIR="$outline_cache"

# Leonardo-specific variables
unset LD_LIBRARY_PATH
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

if [ -n "$1" ]; then
    config_id=$1
else
    config_id=$SLURM_ARRAY_TASK_ID
fi
config_file="config/translation_runs.json"

# TODO: change this based on your local machine
output_dir="/leonardo_work/IscrC_ItaLLM_0/results-interim-gente-xai/translation_runs"

mkdir -p $output_dir

python src/translate.py \
    --output_dir $output_dir \
    --config_id $config_id \
    --config_file $config_file \
    --do_eval

rm -r $outline_cache