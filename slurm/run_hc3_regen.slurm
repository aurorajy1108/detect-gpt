#!/bin/bash
#SBATCH --job-name=hc3_regen_adapt
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Generate model's own answers to HC3 prompts, then run adaptive-k on
# (human, regenerated) pairs. White-box matches team's "Regenerated" scenario.
#
# Usage:
#   sbatch slurm/run_hc3_regen.slurm meta-llama/Llama-3.1-8B
#   sbatch slurm/run_hc3_regen.slurm meta-llama/Meta-Llama-3-8B-Instruct
#   sbatch slurm/run_hc3_regen.slurm Qwen/Qwen3-8B-Base
#   sbatch slurm/run_hc3_regen.slurm Qwen/Qwen3-8B

set -e

MODEL="${1:-meta-llama/Llama-3.1-8B}"
SAFE=$(echo "$MODEL" | tr '/' '_')
echo "===================================================="
echo "Model:        $MODEL"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Started at:   $(date)"
echo "===================================================="

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
  echo "ERROR: HF_TOKEN not set"; exit 1
fi

source $HOME/detect-gpt/venv/bin/activate
cd $HOME/detect-gpt

REGEN_MANIFEST="dataset_manifests/hc3_en_regen_${SAFE}.json"

echo ""
echo "===== Step 1: regenerate answers with $MODEL ====="
python3 regenerate_manifest.py \
  --model "$MODEL" \
  --input_manifest dataset_manifests/hc3_en_mixed_200.json \
  --output_manifest "$REGEN_MANIFEST" \
  --n_samples 200 \
  --max_new_tokens 200 \
  --cache_dir $HF_HOME

echo ""
echo "===== Step 2: adaptive scoring on regenerated pairs ====="
python3 run.py \
  --data_file "$REGEN_MANIFEST" \
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

RESULT_DIR=$HOME/detect-gpt/cluster_results/${SAFE}_regen_${SLURM_JOB_ID}
mkdir -p "$RESULT_DIR"
cp -r tmp_results/* "$RESULT_DIR/" 2>/dev/null || true
cp "$REGEN_MANIFEST" "$RESULT_DIR/" 2>/dev/null || true

echo ""
echo "===================================================="
echo "Finished at:  $(date)"
echo "Results:      $RESULT_DIR"
echo "===================================================="
