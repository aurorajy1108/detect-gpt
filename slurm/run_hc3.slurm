#!/bin/bash
#SBATCH --job-name=hc3_adaptive
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Usage:
#   sbatch slurm/run_hc3.slurm meta-llama/Llama-3.1-8B
#   sbatch slurm/run_hc3.slurm meta-llama/Meta-Llama-3-8B-Instruct
#   sbatch slurm/run_hc3.slurm Qwen/Qwen3-8B-Base
#
# Requires: HF_TOKEN env var set before sbatch (export HF_TOKEN=hf_xxx)

set -e

MODEL="${1:-meta-llama/Llama-3.1-8B}"
SAFE=$(echo "$MODEL" | tr '/' '_')
echo "===================================================="
echo "Model:        $MODEL"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Started at:   $(date)"
echo "===================================================="

# Stage scratch dir for HF cache (fast local SSD)
SCRATCH_BASE=/scratch/$USER
mkdir -p $SCRATCH_BASE
export HF_HOME=$SCRATCH_BASE/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
mkdir -p $HF_HOME

if [ -z "$HF_TOKEN" ] && [ -f $HOME/.cache/huggingface/token ]; then
  export HF_TOKEN=$(cat $HOME/.cache/huggingface/token)
fi
if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: HF_TOKEN not set and no cached token at ~/.cache/huggingface/token"
  echo "Run: huggingface-cli login   OR   export HF_TOKEN=hf_xxx"
  exit 1
fi

# Activate the python env created during setup (see slurm/SETUP.md)
source $HOME/detect-gpt/venv/bin/activate
cd $HOME/detect-gpt

nvidia-smi | head -20
echo "Python: $(which python3)  $(python3 --version)"

python3 run.py \
  --data_file dataset_manifests/hc3_en_mixed_200.json \
  --dataset hc3 \
  --n_samples 200 \
  --base_model_name "$MODEL" \
  --mask_filling_model_name t5-large \
  --n_perturbation_list 30 \
  --n_perturbation_rounds 1 \
  --pct_words_masked 0.3 \
  --span_length 2 \
  --adaptive_k \
  --adaptive_max_k 30 \
  --fast_detect_gpt \
  --cache_dir $HF_HOME

# Copy results out of tmp_results to a model-named folder for easy retrieval
RESULT_DIR=$HOME/detect-gpt/cluster_results/${SAFE}_${SLURM_JOB_ID}
mkdir -p "$RESULT_DIR"
cp -r tmp_results/* "$RESULT_DIR/" 2>/dev/null || true

echo "===================================================="
echo "Finished at:  $(date)"
echo "Results:      $RESULT_DIR"
echo "===================================================="
