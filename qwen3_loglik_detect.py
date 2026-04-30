"""
Qwen3-8B-Base log-likelihood detection on HC3.

Uses the echo trick via Tinker API: ask the model to repeat the text verbatim,
collect output token logprobs, compute mean log-likelihood as the score.

Since top-k logprobs are not available, we use raw log p(x)/n_tokens rather
than the full Fast-DetectGPT discrepancy. This is the log-likelihood criterion
from the original DetectGPT paper, but with Qwen3-8B-Base as the scorer.

Usage:
    TINKER_API_KEY="..." python3 qwen3_loglik_detect.py
"""

import os, json, time
import numpy as np
import tqdm
from openai import OpenAI
from sklearn.metrics import roc_curve, auc, precision_recall_curve

BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
MODEL    = "Qwen/Qwen3-8B-Base"

SYSTEM_PROMPT = (
    "You are a text copy machine. "
    "Repeat the following text EXACTLY as given, word for word, with absolutely "
    "no changes, additions, commentary, or omissions. Output only the repeated text."
)

DATA_FILE = (
    "results/gpt2-medium-t5-large-temp/"
    "2026-04-23-10-19-55-879290-fp32-0.3-1-hc3-100/raw_data.json"
)


def get_loglik_score(client, text, max_retries=3):
    """Echo text, return mean token logprob. None on failure."""
    word_count = len(text.split())
    max_tokens = word_count * 3

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text},
                ],
                max_tokens=max_tokens,
                temperature=0,
                logprobs=True,
                top_logprobs=0,
                extra_body={"enable_thinking": False},
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                return None, f"API error: {e}"
            time.sleep(5 * (attempt + 1))

    choice = resp.choices[0]
    output  = choice.message.content or ""

    # alignment check
    input_start = " ".join(text.split()[:8])
    if input_start.lower() not in output.lower():
        return None, "misaligned echo"

    lp_data = choice.logprobs.content
    if not lp_data:
        return None, "no logprobs"

    logprobs = [t.logprob for t in lp_data]
    if len(logprobs) < 5:
        return None, f"too few tokens ({len(logprobs)})"

    return float(np.mean(logprobs)), None


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
    api_key = os.environ.get("TINKER_API_KEY")
    assert api_key, "Set TINKER_API_KEY"

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    with open(DATA_FILE) as f:
        data = json.load(f)

    originals = data["original"]
    sampled   = data["sampled"]
    n = len(originals)

    print(f"Model:   {MODEL}")
    print(f"Samples: {n}")
    print(f"Data:    {DATA_FILE}\n")

    results, n_skipped = [], 0

    for orig, samp in tqdm.tqdm(zip(originals, sampled), total=n, desc="Scoring"):
        orig_score, orig_err = get_loglik_score(client, orig)
        samp_score, samp_err = get_loglik_score(client, samp)

        if orig_score is None or samp_score is None:
            n_skipped += 1
            if orig_err: print(f"\n  [skip orig] {orig_err}")
            if samp_err: print(f"\n  [skip samp] {samp_err}")
            continue

        results.append({
            "original":      orig,
            "original_crit": orig_score,
            "sampled":       samp,
            "sampled_crit":  samp_score,
        })

    print(f"\nScored {len(results)}/{n} pairs  ({n_skipped} skipped)")
    if len(results) < 5:
        print("Not enough results.")
        return

    real_scores = [r["original_crit"] for r in results]
    samp_scores = [r["sampled_crit"]  for r in results]

    print(f"Human   mean loglik: {np.mean(real_scores):.4f}")
    print(f"ChatGPT mean loglik: {np.mean(samp_scores):.4f}")
    print(f"Separation:         {np.mean(samp_scores) - np.mean(real_scores):.4f}")

    fpr, tpr, roc_auc = get_roc_metrics(real_scores, samp_scores)
    p, r, pr_auc = get_pr_metrics(real_scores, samp_scores)

    print(f"\nQwen3-8B-Base log-likelihood  ROC AUC: {roc_auc:.4f}   PR AUC: {pr_auc:.4f}")

    print("\n--- Comparison (HC3, same 100 pairs) ---")
    prev = {
        "Pert-1 (z)":       0.7775,
        "Pert-10 (z)":      0.9429,
        "Adaptive-k (z)":   0.9545,
        "Fast-DetectGPT (GPT-2)": 0.9845,
    }
    for name, val in prev.items():
        print(f"  {name}: {val:.4f}")
    print(f"  Qwen3-8B-Base loglik: {roc_auc:.4f}  ← new")

    output = {
        "name":        f"qwen3_8b_base_loglik",
        "info":        {"n_samples": len(results), "n_skipped": n_skipped, "model": MODEL},
        "predictions": {"real": real_scores, "samples": samp_scores},
        "raw_results": results,
        "metrics":     {"roc_auc": roc_auc, "fpr": fpr, "tpr": tpr},
        "pr_metrics":  {"pr_auc": pr_auc, "precision": p, "recall": r},
        "loss":        1 - pr_auc,
    }

    out_path = "results/qwen3_8b_base_loglik_hc3.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
