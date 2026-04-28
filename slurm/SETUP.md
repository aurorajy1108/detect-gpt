# Neuronic cluster setup for detect-gpt

## One-time setup

```bash
ssh ly7998@neuronic.cs.princeton.edu

# Clone repo
cd $HOME
git clone git@github.com:tomliqianyin/detect-gpt.git
# (or: git clone https://github.com/tomliqianyin/detect-gpt.git)
cd detect-gpt
git checkout hc3-experiments

# Python env
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install scikit-learn tqdm  # in case missing

# Hugging Face token (for gated models like Llama)
# Get token at: https://huggingface.co/settings/tokens
# Accept license at: https://huggingface.co/meta-llama/Llama-3.1-8B
huggingface-cli login   # paste token

# Make logs dir
mkdir -p logs cluster_results
```

## Submitting jobs

```bash
# Set token in env (or source from ~/.bashrc)
export HF_TOKEN=hf_xxxxxxxxxxxx

# Llama 3.1 8B base
sbatch slurm/run_hc3.slurm meta-llama/Llama-3.1-8B

# Llama 3 8B Instruct (post-trained)
sbatch slurm/run_hc3.slurm meta-llama/Meta-Llama-3-8B-Instruct

# Qwen3-8B base
sbatch slurm/run_hc3.slurm Qwen/Qwen3-8B-Base

# Qwen3-8B post-trained
sbatch slurm/run_hc3.slurm Qwen/Qwen3-8B
```

Each job takes ~30-45 min on an L40 GPU.

## Monitoring

```bash
squeue -u $USER          # check queued/running jobs
tail -f logs/hc3_adaptive_<JOBID>.log
scancel <JOBID>          # cancel a job
```

## Retrieving results

Results land in `~/detect-gpt/cluster_results/<model>_<jobid>/`. To pull them
locally:

```bash
# from your laptop
scp -r ly7998@neuronic.cs.princeton.edu:~/detect-gpt/cluster_results/ ./cluster_results/
```

The metric files are:
- `*/perturbation_30_d_results.json` - fixed perturbation, d criterion
- `*/perturbation_30_z_results.json` - fixed perturbation, z criterion
- `*/adaptive_d_conf95_results.json` - adaptive, d criterion (max_k=30)
- `*/adaptive_z_conf95_results.json` - adaptive, z criterion
- `*/fast_detect_gpt_results.json` - Fast-DetectGPT
- `*/likelihood_threshold_results.json`, `rank_threshold_results.json`, etc.

Each JSON has `metrics.roc_auc` and `pr_metrics.pr_auc`.
