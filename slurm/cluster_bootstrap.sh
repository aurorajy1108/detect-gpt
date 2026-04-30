#!/bin/bash
# Run this ONCE on neuronic after SSHing in.
#   ssh ly7998@neuronic.cs.princeton.edu
#   bash <(curl -sL https://raw.githubusercontent.com/tomliqianyin/detect-gpt/hc3-experiments/slurm/cluster_bootstrap.sh)
# Or paste the contents directly.

set -e
cd $HOME

echo "==== [1/4] Clone detect-gpt ===="
if [ -d detect-gpt ]; then
  cd detect-gpt && git fetch origin && git checkout hc3-experiments && git pull origin hc3-experiments
else
  git clone https://github.com/tomliqianyin/detect-gpt.git
  cd detect-gpt && git checkout hc3-experiments
fi

echo ""
echo "==== [2/4] Python env ===="
if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
pip install scikit-learn tqdm huggingface_hub --quiet

mkdir -p logs cluster_results

echo ""
echo "==== [3/4] HuggingFace login ===="
if [ ! -f $HOME/.cache/huggingface/token ]; then
  echo "Need to login to HuggingFace (for gated Llama models)."
  echo "Get a token at https://huggingface.co/settings/tokens"
  echo "Make sure you've accepted the license at https://huggingface.co/meta-llama/Llama-3.1-8B"
  huggingface-cli login
else
  echo "HF token already configured."
fi

echo ""
echo "==== [4/4] Submit jobs ===="
echo "About to submit 4 jobs:"
echo "  1. Llama-3.1-8B (base)"
echo "  2. Meta-Llama-3-8B-Instruct (post-trained)"
echo "  3. Qwen3-8B-Base"
echo "  4. Qwen3-8B (post-trained)"
echo ""
read -p "Submit all 4? [y/N] " yn
if [ "$yn" != "y" ] && [ "$yn" != "Y" ]; then
  echo "Skipped. Submit manually with: sbatch slurm/run_hc3.slurm <model_name>"
  exit 0
fi

export HF_TOKEN=$(cat $HOME/.cache/huggingface/token 2>/dev/null)
sbatch slurm/run_hc3.slurm meta-llama/Llama-3.1-8B
sbatch slurm/run_hc3.slurm meta-llama/Meta-Llama-3-8B-Instruct
sbatch slurm/run_hc3.slurm Qwen/Qwen3-8B-Base
sbatch slurm/run_hc3.slurm Qwen/Qwen3-8B

echo ""
echo "Done. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/hc3_adaptive_<JOBID>.log"
