# DetectGPT Experiments on Modern LLMs

## Goal

We want to evaluate whether DetectGPT's perturbation-based signal still holds for modern language models, especially after post-training.

Our core research questions are:

1. How well does DetectGPT perform on modern base models?
2. How does performance change after post-training on the corresponding instruction / hybrid / reasoning variants?
3. Does the behavior differ between HC3 English and HC3 Chinese?
4. Does the result differ when we use HC3's original ChatGPT answers versus re-generating answers from the target model?

## Experimental Factors

### Factor A: Model Family and Training Stage

- Base model
- Post-trained counterpart

Recommended pairs:

- Qwen3-8B-Base vs Qwen3-8B
- Qwen3-30B-A3B-Base vs Qwen3-30B-A3B
- Llama-3.1-8B vs Llama-3.1-8B-Instruct
- Llama-3.1-70B vs Llama-3.3-70B-Instruct
- DeepSeek-V3.1-Base vs DeepSeek-V3.1

Notes:

- `GPT-OSS-20B` and `GPT-OSS-120B` are useful modern models, but they do not have a matching base model in the provided Tinker list, so they should be treated as additional post-trained / reasoning baselines rather than base-vs-post-trained pairs.
- OLMo is part of the broader research plan, but it is not present in the Tinker model list you provided. If we want to include OLMo, we will likely need a local HuggingFace workflow instead of Tinker.

### Factor B: Dataset Language

- HC3 English: `--dataset hc3`
- HC3 Chinese: `--dataset hc3_zh`

### Factor C: Source Domain

Recommended HC3 sources:

- English: `finance`, `medicine`, `open_qa`, `reddit_eli5`, `wiki_csai`
- Chinese: `finance`, `medicine`, `open_qa`, `law`, `psychology`, `baike`, `nlpcc_dbqa`

The exact source names depend on the HuggingFace schema exposed by HC3 / HC3-Chinese. If a source name differs slightly, use the name returned by the dataset loader.

### Factor D: AI Answers

- Use the original HC3 ChatGPT answers: add `--use_dataset_samples`
- Re-generate answers from the target model: do not add `--use_dataset_samples`

This factor is important because it separates:

- "Can DetectGPT identify legacy ChatGPT answers on this dataset?"
- "Can DetectGPT identify answers produced by the target modern model itself?"

## Default Hyperparameters

These are the recommended default settings for the main experiments:

```bash
--dataset_split train
--n_samples 200
--batch_size 10
--pct_words_masked 0.3
--span_length 2
--n_perturbation_list 10
--n_perturbation_rounds 1
--mask_filling_model_name t5-large
--do_top_p
--top_p 0.96
```

Additional notes:

- For HC3 Chinese, `t5-large` is not ideal because it is English-centric. A multilingual infilling model such as `google/mt5-large` is a better choice if it behaves correctly with the `<extra_id_*>` mask format in this pipeline.
- `n_samples=50` is suitable for quick smoke tests.
- `n_samples=200` is a good initial research setting.
- `n_samples=500+` is better for final plots and stronger conclusions.
- `n_perturbation_list=10` is a good default for the `z`-score version of DetectGPT because it needs a more stable perturbation mean and standard deviation.

## Recommended Formal Plan for Qwen3-8B

This is the recommended "first full experiment" before expanding to larger model families.

### Scope

- Model family: Qwen3-8B
- Training stages:
  - `Qwen/Qwen3-8B-Base`
  - `Qwen/Qwen3-8B`
- Languages:
  - HC3 English
  - HC3 Chinese
- Answer modes:
  - HC3 original ChatGPT answers
  - Re-generated answers from the target model

This gives a clean 2 x 2 x 2 design:

- base vs post-trained
- English vs Chinese
- dataset answers vs regenerated answers

Total conditions: 8

### Recommended Formal Hyperparameters

```bash
--n_samples 200
--batch_size 10
--pct_words_masked 0.3
--span_length 2
--n_perturbation_list 10
--n_perturbation_rounds 1
--do_top_p
--top_p 0.96
--max_sample_tries 10
```

Language-specific perturbation models:

- English: `--mask_filling_model_name t5-large`
- Chinese: `--mask_filling_model_name google/mt5-large`

Recommended domains for the first pass:

- English: `finance`
- Chinese: `finance`

After the first pass is stable, extend to additional sources.

### Recommended Run Strategy

Run the full Qwen3-8B experiment in this order:

1. English, dataset answers, base vs post-trained
2. English, regenerated answers, base vs post-trained
3. Chinese, dataset answers, base vs post-trained
4. Chinese, regenerated answers, base vs post-trained

This order is useful because:

- English is easier to debug than Chinese
- dataset answers are cheaper and more stable than regenerated answers
- if the base/post-trained trend is already clear in English, we can validate it before scaling further

### Concrete Commands

#### English, dataset answers

```bash
python3 run_modern_llm_batch.py \
  --families qwen3_8b \
  --languages en \
  --stages base post_trained \
  --answer_modes dataset_answers \
  --n_samples 200 \
  --batch_size 10 \
  --n_perturbation_list 10
```

#### English, regenerated answers

```bash
python3 run_modern_llm_batch.py \
  --families qwen3_8b \
  --languages en \
  --stages base post_trained \
  --answer_modes regenerated_answers \
  --n_samples 200 \
  --batch_size 10 \
  --n_perturbation_list 10 \
  --regenerated_min_sample_words 30 \
  --max_sample_tries 10
```

#### Chinese, dataset answers

```bash
python3 run_modern_llm_batch.py \
  --families qwen3_8b \
  --languages zh \
  --stages base post_trained \
  --answer_modes dataset_answers \
  --n_samples 200 \
  --batch_size 10 \
  --n_perturbation_list 10
```

#### Chinese, regenerated answers

```bash
python3 run_modern_llm_batch.py \
  --families qwen3_8b \
  --languages zh \
  --stages base post_trained \
  --answer_modes regenerated_answers \
  --n_samples 200 \
  --batch_size 10 \
  --n_perturbation_list 10 \
  --regenerated_min_sample_words 30 \
  --max_sample_tries 10
```

### Why This Is the Best First Formal Version

- It isolates the effect of post-training within one model family.
- It tests both older AI answers and modern self-generated answers.
- It checks whether the signal transfers across English and Chinese.
- It upgrades from smoke-test settings to a much more credible regime:
  - `n_samples=200`
  - `n_perturbations=10`
- It still keeps the total budget manageable enough to finish before scaling to larger models.

## Main Experiment Grid

### 1. Base Models on HC3 English, Re-generated Answers

This is the cleanest white-box DetectGPT setting.

#### Qwen3-8B-Base

```bash
export TINKER_API_KEY="YOUR_KEY"

python3 run.py \
  --dataset hc3 \
  --dataset_split train \
  --dataset_source finance \
  --n_samples 200 \
  --batch_size 10 \
  --pct_words_masked 0.3 \
  --span_length 2 \
  --n_perturbation_list 10 \
  --n_perturbation_rounds 1 \
  --mask_filling_model_name t5-large \
  --do_top_p \
  --top_p 0.96 \
  --tinker_model Qwen/Qwen3-8B-Base \
  --base_model_name Qwen/Qwen3-8B-Base \
  --output_name modern_llm_study
```

#### Qwen3-30B-A3B-Base

```bash
python3 run.py \
  --dataset hc3 \
  --dataset_split train \
  --dataset_source finance \
  --n_samples 200 \
  --batch_size 10 \
  --pct_words_masked 0.3 \
  --span_length 2 \
  --n_perturbation_list 10 \
  --n_perturbation_rounds 1 \
  --mask_filling_model_name t5-large \
  --do_top_p \
  --top_p 0.96 \
  --tinker_model Qwen/Qwen3-30B-A3B-Base \
  --base_model_name Qwen/Qwen3-30B-A3B-Base \
  --output_name modern_llm_study
```

#### Llama-3.1-8B

```bash
python3 run.py \
  --dataset hc3 \
  --dataset_split train \
  --dataset_source finance \
  --n_samples 200 \
  --batch_size 10 \
  --pct_words_masked 0.3 \
  --span_length 2 \
  --n_perturbation_list 10 \
  --n_perturbation_rounds 1 \
  --mask_filling_model_name t5-large \
  --do_top_p \
  --top_p 0.96 \
  --tinker_model meta-llama/Llama-3.1-8B \
  --base_model_name meta-llama/Llama-3.1-8B \
  --output_name modern_llm_study
```

### 2. Post-trained Counterparts on HC3 English, Re-generated Answers

These experiments test whether post-training weakens the DetectGPT signal.

#### Qwen3-8B

```bash
python3 run.py \
  --dataset hc3 \
  --dataset_split train \
  --dataset_source finance \
  --n_samples 200 \
  --batch_size 10 \
  --pct_words_masked 0.3 \
  --span_length 2 \
  --n_perturbation_list 10 \
  --n_perturbation_rounds 1 \
  --mask_filling_model_name t5-large \
  --do_top_p \
  --top_p 0.96 \
  --tinker_model Qwen/Qwen3-8B \
  --base_model_name Qwen/Qwen3-8B \
  --output_name modern_llm_study
```

#### Qwen3-30B-A3B

```bash
python3 run.py \
  --dataset hc3 \
  --dataset_split train \
  --dataset_source finance \
  --n_samples 200 \
  --batch_size 10 \
  --pct_words_masked 0.3 \
  --span_length 2 \
  --n_perturbation_list 10 \
  --n_perturbation_rounds 1 \
  --mask_filling_model_name t5-large \
  --do_top_p \
  --top_p 0.96 \
  --tinker_model Qwen/Qwen3-30B-A3B \
  --base_model_name Qwen/Qwen3-30B-A3B \
  --output_name modern_llm_study
```

#### Llama-3.1-8B-Instruct

```bash
python3 run.py \
  --dataset hc3 \
  --dataset_split train \
  --dataset_source finance \
  --n_samples 200 \
  --batch_size 10 \
  --pct_words_masked 0.3 \
  --span_length 2 \
  --n_perturbation_list 10 \
  --n_perturbation_rounds 1 \
  --mask_filling_model_name t5-large \
  --do_top_p \
  --top_p 0.96 \
  --tinker_model meta-llama/Llama-3.1-8B-Instruct \
  --base_model_name meta-llama/Llama-3.1-8B-Instruct \
  --output_name modern_llm_study
```

### 3. HC3 English vs HC3 Chinese

To compare across languages, keep the model family and perturbation settings fixed.

#### English

```bash
python3 run.py \
  --dataset hc3 \
  --dataset_split train \
  --dataset_source finance \
  --n_samples 200 \
  --batch_size 10 \
  --pct_words_masked 0.3 \
  --span_length 2 \
  --n_perturbation_list 10 \
  --n_perturbation_rounds 1 \
  --mask_filling_model_name t5-large \
  --do_top_p \
  --top_p 0.96 \
  --tinker_model Qwen/Qwen3-30B-A3B-Base \
  --base_model_name Qwen/Qwen3-30B-A3B-Base \
  --output_name modern_llm_study
```

#### Chinese

Recommended to try `google/mt5-large` for perturbation:

```bash
python3 run.py \
  --dataset hc3_zh \
  --dataset_split train \
  --dataset_source finance \
  --n_samples 200 \
  --batch_size 10 \
  --pct_words_masked 0.3 \
  --span_length 2 \
  --n_perturbation_list 10 \
  --n_perturbation_rounds 1 \
  --mask_filling_model_name google/mt5-large \
  --do_top_p \
  --top_p 0.96 \
  --tinker_model Qwen/Qwen3-30B-A3B-Base \
  --base_model_name Qwen/Qwen3-30B-A3B-Base \
  --output_name modern_llm_study
```

### 4. HC3 Original ChatGPT Answers vs Re-generated Answers

#### Use HC3's original ChatGPT answers

```bash
python3 run.py \
  --dataset hc3 \
  --dataset_split train \
  --dataset_source finance \
  --use_dataset_samples \
  --n_samples 200 \
  --batch_size 10 \
  --pct_words_masked 0.3 \
  --span_length 2 \
  --n_perturbation_list 10 \
  --n_perturbation_rounds 1 \
  --mask_filling_model_name t5-large \
  --tinker_model Qwen/Qwen3-30B-A3B-Base \
  --base_model_name Qwen/Qwen3-30B-A3B-Base \
  --output_name modern_llm_study
```

#### Re-generate answers from the target model

```bash
python3 run.py \
  --dataset hc3 \
  --dataset_split train \
  --dataset_source finance \
  --n_samples 200 \
  --batch_size 10 \
  --pct_words_masked 0.3 \
  --span_length 2 \
  --n_perturbation_list 10 \
  --n_perturbation_rounds 1 \
  --mask_filling_model_name t5-large \
  --tinker_model Qwen/Qwen3-30B-A3B-Base \
  --base_model_name Qwen/Qwen3-30B-A3B-Base \
  --output_name modern_llm_study
```

## Quick Smoke Tests

Before running the full experiments, use:

```bash
--n_samples 50
--batch_size 5
--n_perturbation_list 5
```

You can also use the batch runner:

```bash
python3 run_modern_llm_batch.py \
  --n_samples 20 \
  --batch_size 5
```

To preview commands without executing them:

```bash
python3 run_modern_llm_batch.py \
  --print_only \
  --n_samples 20 \
  --batch_size 5
```

The batch runner:

- sweeps base vs post-trained models
- sweeps HC3 English vs HC3 Chinese
- sweeps HC3 original answers vs regenerated answers
- writes completion markers so interrupted batches can resume
- reuses the shared API cache directory to avoid repeated Tinker calls

## Expected Outputs

Each run saves:

- raw paired data
- baseline likelihood results
- DetectGPT perturbation results for `d` and `z`
- ROC and histogram plots

The main numbers to compare are:

- ROC-AUC
- PR-AUC
- separation between original and perturbed likelihoods
- how much `z` degrades from base to post-trained models

## Interpretation Guide

Evidence supporting the original DetectGPT intuition:

- Base models show stronger DetectGPT separation than post-trained models
- The same family's base checkpoint consistently outperforms its instruction / hybrid counterpart
- Re-generated modern-model answers are harder to detect than HC3's older ChatGPT answers

Evidence against the original intuition:

- Post-trained models retain similar or better DetectGPT performance
- The curvature-based signal remains strong across both English and Chinese
- Re-generated answers from modern models are still clearly separated by DetectGPT
