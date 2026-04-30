"""
Fast-DetectGPT with Qwen2.5-7B as scorer on HC3.

Loads Qwen/Qwen2.5-7B locally, runs a single forward pass per text,
and computes the analytic sampling discrepancy score from Fast-DetectGPT
(Bao et al., 2023):

    d(x) = (log p(x) - mean_ref) / sqrt(var_ref)

where mean_ref and var_ref are computed from the full softmax at each position.
No T5, no sampling, no echo trick needed.

Usage:
    python3 qwen25_fastdetect.py [--model Qwen/Qwen2.5-7B] [--n_samples 100]
"""

import os, json, argparse
import numpy as np
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_curve, auc, precision_recall_curve

DATA_FILE = (
    "results/gpt2-medium-t5-large-temp/"
    "2026-04-23-10-19-55-879290-fp32-0.3-1-hc3-100/raw_data.json"
)

MAX_TOKENS = 512  # truncate to avoid OOM on very long texts


def get_fast_detect_score(text, model, tokenizer, device):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        return_token_type_ids=False,
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits          # (1, T, V)

    logits   = logits[:, :-1]                 # predict next token; drop last pos
    labels   = enc.input_ids[:, 1:].unsqueeze(-1)  # (1, T, 1)

    lprobs   = torch.log_softmax(logits, dim=-1)   # (1, T, V)
    probs    = torch.softmax(logits, dim=-1)        # (1, T, V)

    log_likelihood = lprobs.gather(dim=-1, index=labels).squeeze(-1)  # (1, T)
    mean_ref = (probs * lprobs).sum(dim=-1)                           # (1, T)
    var_ref  = (probs * lprobs.square()).sum(dim=-1) - mean_ref.square()

    discrepancy = (
        (log_likelihood.sum(-1) - mean_ref.sum(-1)) /
        var_ref.sum(-1).clamp(min=1e-8).sqrt()
    )
    return discrepancy.mean().item()


def get_roc_metrics(real, samples):
    labels = [0] * len(real) + [1] * len(samples)
    scores = real + samples
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr.tolist(), tpr.tolist(), float(auc(fpr, tpr))


def get_pr_metrics(real, samples):
    labels = [0] * len(real) + [1] * len(samples)
    scores = real + samples
    p, r, _ = precision_recall_curve(labels, scores)
    return p.tolist(), r.tolist(), float(auc(r, p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device:  {device}")
    print(f"Model:   {args.model}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print("Model loaded.\n")

    with open(DATA_FILE) as f:
        data = json.load(f)

    originals = data["original"][: args.n_samples]
    sampled   = data["sampled"][: args.n_samples]
    n = len(originals)
    print(f"Samples: {n}\nData:    {DATA_FILE}\n")

    results = []
    for orig, samp in tqdm.tqdm(zip(originals, sampled), total=n, desc="Scoring"):
        results.append({
            "original":      orig,
            "original_crit": get_fast_detect_score(orig, model, tokenizer, device),
            "sampled":       samp,
            "sampled_crit":  get_fast_detect_score(samp, model, tokenizer, device),
        })

    real_scores = [r["original_crit"] for r in results]
    samp_scores = [r["sampled_crit"]  for r in results]

    print(f"\nHuman   mean discrepancy: {np.mean(real_scores):.4f}")
    print(f"ChatGPT mean discrepancy: {np.mean(samp_scores):.4f}")

    fpr, tpr, roc_auc = get_roc_metrics(real_scores, samp_scores)
    p, r, pr_auc      = get_pr_metrics(real_scores, samp_scores)

    print(f"\n{args.model} Fast-DetectGPT  ROC AUC: {roc_auc:.4f}   PR AUC: {pr_auc:.4f}")

    print("\n--- Comparison (HC3, same 100 pairs) ---")
    prev = {
        "Pert-1 (z)":              0.7775,
        "Pert-10 (z)":             0.9429,
        "Adaptive-k (z)":          0.9545,
        "Fast-DetectGPT (GPT-2)":  0.9845,
    }
    for name, val in prev.items():
        print(f"  {name}: {val:.4f}")
    print(f"  Fast-DetectGPT (Qwen2.5-7B): {roc_auc:.4f}  ← new")

    safe_name = args.model.replace("/", "_").replace("-", "_").lower()
    output = {
        "name":        f"fast_detect_gpt_{safe_name}",
        "info":        {"n_samples": n, "model": args.model},
        "predictions": {"real": real_scores, "samples": samp_scores},
        "raw_results": results,
        "metrics":     {"roc_auc": roc_auc, "fpr": fpr, "tpr": tpr},
        "pr_metrics":  {"pr_auc": pr_auc, "precision": p, "recall": r},
        "loss":        1 - pr_auc,
    }

    out_path = f"results/fast_detect_{safe_name}_hc3.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
