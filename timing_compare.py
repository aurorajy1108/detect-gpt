"""
Compare wall-clock cost: k perturbations (DetectGPT) vs one forward pass (Fast-DetectGPT).

Models:
  Scorer  : GPT-2 Medium  (~355M params)
  Perturber: T5-Large     (~770M params)

Reported numbers:
  - Mean time for one GPT-2 forward pass (Fast-DetectGPT cost per text)
  - Mean time to generate one T5 perturbation
  - Total DetectGPT cost for k = {10, 20, 30} perturbations
  - Speedup factor Fast-DetectGPT vs DetectGPT-k

Usage:
    python3 timing_compare.py
"""

import os, time, torch
import numpy as np

CACHE = os.path.expanduser("~/.cache")
REPS  = 20   # number of timing repetitions to average
TEXT  = (
    "The quick brown fox jumps over the lazy dog. "
    "Natural language processing has advanced rapidly in recent years. "
    "Large language models can generate coherent text across many domains."
)

def mean_std(times):
    return np.mean(times) * 1000, np.std(times) * 1000  # → ms

# ── GPT-2 Medium forward pass ──────────────────────────────────────────────────
print("Loading GPT-2 Medium...")
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tok_gpt2  = GPT2Tokenizer.from_pretrained("gpt2-medium",  cache_dir=CACHE)
tok_gpt2.pad_token = tok_gpt2.eos_token
model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2-medium", cache_dir=CACHE)
model_gpt2.eval()
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model_gpt2 = model_gpt2.to(device)
print(f"  device: {device}")

inputs = tok_gpt2(TEXT, return_tensors="pt", padding=True).to(device)
inputs.pop("token_type_ids", None)

# Warm-up
with torch.no_grad():
    _ = model_gpt2(**inputs).logits

times_fwd = []
for _ in range(REPS):
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model_gpt2(**inputs).logits
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    times_fwd.append(time.perf_counter() - t0)

mu_fwd, sd_fwd = mean_std(times_fwd)
print(f"  GPT-2 Medium forward pass: {mu_fwd:.1f} ± {sd_fwd:.1f} ms  (n={REPS})")

n_tokens = inputs["input_ids"].shape[1]
print(f"  Input length: {n_tokens} tokens")

# ── T5-Large generation (one perturbation) ─────────────────────────────────────
print("\nLoading T5-Large...")
from transformers import T5Tokenizer, T5ForConditionalGeneration

tok_t5   = T5Tokenizer.from_pretrained("t5-large", cache_dir=CACHE)
model_t5 = T5ForConditionalGeneration.from_pretrained("t5-large", cache_dir=CACHE)
model_t5.eval()
model_t5 = model_t5.to(device)

# Mask ~15% of tokens (standard T5 mask ratio)
mask_pct = 0.15
words    = TEXT.split()
n_mask   = max(1, int(len(words) * mask_pct))
mask_idxs = np.random.choice(len(words), size=n_mask, replace=False)
masked_words = words.copy()
sentinel = 0
for idx in sorted(mask_idxs):
    masked_words[idx] = f"<extra_id_{sentinel}>"
    sentinel += 1
masked_text = " ".join(masked_words)

t5_inputs = tok_t5(masked_text, return_tensors="pt", padding=True).to(device)

# Warm-up
with torch.no_grad():
    _ = model_t5.generate(**t5_inputs, max_new_tokens=n_tokens, do_sample=True)

times_t5 = []
for _ in range(REPS):
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model_t5.generate(**t5_inputs, max_new_tokens=n_tokens, do_sample=True)
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    times_t5.append(time.perf_counter() - t0)

mu_t5, sd_t5 = mean_std(times_t5)
print(f"  T5-Large one perturbation: {mu_t5:.1f} ± {sd_t5:.1f} ms  (n={REPS})")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("COST COMPARISON SUMMARY")
print("="*60)
print(f"  Fast-DetectGPT (1 forward pass):  {mu_fwd:.1f} ms")
print()
for k in [10, 20, 30]:
    # DetectGPT cost = k × T5 generation + k × GPT-2 forward pass
    total = k * mu_t5 + k * mu_fwd
    speedup = total / mu_fwd
    print(f"  DetectGPT k={k:<2}:  {total:.0f} ms  (speedup {speedup:.0f}× slower than Fast-DetectGPT)")

print()
print(f"  T5 generation dominates: {mu_t5:.1f} ms vs {mu_fwd:.1f} ms per call")
print(f"  T5/GPT2 ratio per call:  {mu_t5/mu_fwd:.1f}×")
