"""
Echo detection: score texts using GPT-4o-mini via Chat API (echo/repeat trick).

The model is asked to repeat the text verbatim. Output logprobs for the repeated
tokens approximate log p(token | GPT model). We then compute the Fast-DetectGPT
analytic discrepancy from those logprobs + top-k alternatives.

Because we only have top-k logprobs (not the full vocabulary), we renormalize the
top-k probabilities to sum to 1 when estimating mean_ref and var_ref. This is an
approximation — the main thing being evaluated is whether the echo trick produces
a useful signal at all.

Usage:
    OPENAI_API_KEY="sk-..." python3 echo_detect.py
    OPENAI_API_KEY="sk-..." python3 echo_detect.py --model gpt-4o --n_samples 50
"""

import os
import json
import time
import argparse
import numpy as np
import tqdm
from openai import OpenAI
from sklearn.metrics import roc_curve, auc, precision_recall_curve

SYSTEM_PROMPT = (
    "You are a text copy machine. "
    "Repeat the following text EXACTLY as given, word for word, with absolutely "
    "no changes, additions, commentary, or omissions. Output only the repeated text."
)

DATA_FILE = (
    "results/gpt2-medium-t5-large-temp/"
    "2026-04-21-09-47-15-601497-fp32-0.3-1-xsum-100/raw_data.json"
)


def get_echo_score(client, text, model, top_logprobs=20, max_retries=3):
    """
    Call the Chat API with a repeat prompt, extract output logprobs,
    and compute the Fast-DetectGPT analytic discrepancy.

    Returns (score, error_string). score is None on failure.
    """
    word_count = len(text.split())
    max_tokens = word_count * 3  # generous: tokens > words

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                logprobs=True,
                top_logprobs=top_logprobs,
                temperature=0,
                max_tokens=max_tokens,
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                return None, f"API error: {e}"
            time.sleep(5 * (attempt + 1))

    choice = response.choices[0]
    output_text = choice.message.content or ""

    # Alignment check: the output should contain most of the input text.
    # We check whether the first 60 chars of input appear in the output.
    input_start = " ".join(text.split()[:10])
    if input_start.lower() not in output_text.lower():
        return None, f"misaligned — model did not copy faithfully"

    token_logprobs_data = choice.logprobs.content
    if not token_logprobs_data:
        return None, "no logprobs returned"

    log_likelihoods = []
    mean_refs = []
    var_refs = []

    for token_data in token_logprobs_data:
        actual_logprob = token_data.logprob
        top_lps = token_data.top_logprobs

        if not top_lps:
            continue

        # Renormalize top-k probs to sum to 1 (approximation of full distribution)
        raw_log_probs = np.array([lp.logprob for lp in top_lps], dtype=np.float64)
        raw_probs = np.exp(raw_log_probs)
        total = raw_probs.sum()
        if total <= 0:
            continue
        probs = raw_probs / total
        lprobs = np.log(probs + 1e-30)

        mean_ref = (probs * lprobs).sum()
        var_ref = (probs * np.square(lprobs)).sum() - mean_ref ** 2

        log_likelihoods.append(actual_logprob)
        mean_refs.append(mean_ref)
        var_refs.append(max(var_ref, 1e-10))

    if len(log_likelihoods) < 5:
        return None, f"too few valid tokens ({len(log_likelihoods)})"

    ll_sum   = sum(log_likelihoods)
    mean_sum = sum(mean_refs)
    var_sum  = sum(var_refs)

    discrepancy = (ll_sum - mean_sum) / np.sqrt(var_sum)
    return float(discrepancy), None


def get_roc_metrics(real_preds, sample_preds):
    labels = [0] * len(real_preds) + [1] * len(sample_preds)
    scores = real_preds + sample_preds
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr.tolist(), tpr.tolist(), float(auc(fpr, tpr))


def get_precision_recall_metrics(real_preds, sample_preds):
    labels = [0] * len(real_preds) + [1] * len(sample_preds)
    scores = real_preds + sample_preds
    p, r, _ = precision_recall_curve(labels, scores)
    return p.tolist(), r.tolist(), float(auc(r, p))


def main(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "Set OPENAI_API_KEY as an environment variable"
    client = OpenAI(api_key=api_key)

    # Load the same data used in the adaptive-k and Fast-DetectGPT experiments
    with open(args.data_file) as f:
        data = json.load(f)

    original_texts = data["original"][: args.n_samples]
    sampled_texts  = data["sampled"][: args.n_samples]
    n = len(original_texts)

    print(f"Model:    {args.model}")
    print(f"Samples:  {n}")
    print(f"Data:     {args.data_file}")
    print()

    results   = []
    n_skipped = 0

    for orig, samp in tqdm.tqdm(zip(original_texts, sampled_texts), total=n, desc="Echo scoring"):
        orig_score, orig_err = get_echo_score(client, orig, args.model, args.top_logprobs)
        samp_score, samp_err = get_echo_score(client, samp, args.model, args.top_logprobs)

        if orig_score is None or samp_score is None:
            n_skipped += 1
            if orig_err:
                print(f"\n  [skip original] {orig_err}")
            if samp_err:
                print(f"\n  [skip sampled]  {samp_err}")
            continue

        results.append({
            "original":      orig,
            "original_crit": orig_score,
            "sampled":       samp,
            "sampled_crit":  samp_score,
        })

    print(f"\nScored {len(results)}/{n} pairs  ({n_skipped} skipped)")

    if len(results) < 2:
        print("Not enough results to compute metrics.")
        return

    predictions = {
        "real":    [r["original_crit"] for r in results],
        "samples": [r["sampled_crit"]  for r in results],
    }

    print(f"Real    mean/std: {np.mean(predictions['real']):.3f} / {np.std(predictions['real']):.3f}")
    print(f"Sampled mean/std: {np.mean(predictions['samples']):.3f} / {np.std(predictions['samples']):.3f}")

    fpr, tpr, roc_auc = get_roc_metrics(predictions["real"], predictions["samples"])
    p, r, pr_auc = get_precision_recall_metrics(predictions["real"], predictions["samples"])

    print(f"\nEcho ({args.model})  ROC AUC: {roc_auc:.4f}   PR AUC: {pr_auc:.4f}")

    # Compare against our previous results
    print("\n--- Comparison (same 100 xsum samples) ---")
    prev = {
        "Fast-DetectGPT (GPT-2 Med, local)": 0.9892,
        "Perturbation-30 z (GPT-2 Med + T5-L)": 0.9885,
        "Perturbation-30 d (GPT-2 Med + T5-L)": 0.9858,
        "Adaptive-k d (GPT-2 Med + T5-L)": 0.9403,
    }
    for name, val in prev.items():
        print(f"  {name}: {val:.4f}")
    print(f"  Echo {args.model}: {roc_auc:.4f}  ← new")

    output = {
        "name":        f"echo_{args.model}",
        "info":        {"n_samples": len(results), "n_skipped": n_skipped, "model": args.model},
        "predictions": predictions,
        "raw_results": results,
        "metrics":     {"roc_auc": roc_auc, "fpr": fpr, "tpr": tpr},
        "pr_metrics":  {"pr_auc": pr_auc, "precision": p, "recall": r},
        "loss":        1 - pr_auc,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file",     type=str, default=DATA_FILE)
    parser.add_argument("--output_file",   type=str, default="results/echo_detect_results.json")
    parser.add_argument("--model",         type=str, default="gpt-4o-mini",
                        help="OpenAI model to use (gpt-4o-mini, gpt-4o, gpt-3.5-turbo-instruct)")
    parser.add_argument("--n_samples",     type=int, default=100)
    parser.add_argument("--top_logprobs",  type=int, default=20,
                        help="Number of top logprobs to request per token (max 20)")
    args = parser.parse_args()
    main(args)
