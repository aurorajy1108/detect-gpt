"""
Masked prediction detection — validation experiment.

Setup (white-box):
  - Source model:  GPT-4o-mini  (generates the machine text)
  - Scorer:        GPT-4o-mini  (masked prediction)

For each text we sample K random word positions, mask each one, and ask
GPT-4o-mini to fill in the blank. The output logprob of the actual word
at that position gives us an approximate token-level log probability.

We then compute a discrepancy score analogous to Fast-DetectGPT:
  d = (sum log p(actual) - sum mean_ref) / sqrt(sum var_ref)

where mean_ref and var_ref are estimated from the top-20 logprobs returned
by the API at each masked position.

If this yields high AUROC, the masked prediction criterion is valid and
worth building into the DetectRouter pool extension.

Usage:
    OPENAI_API_KEY="sk-..." python3 masked_detect.py
    OPENAI_API_KEY="sk-..." python3 masked_detect.py --n_samples 30 --k_positions 20
"""

import os, json, time, random, argparse
import numpy as np
import tqdm
import datasets
from openai import OpenAI
from sklearn.metrics import roc_curve, auc, precision_recall_curve


# ── data ──────────────────────────────────────────────────────────────────────

def load_xsum_human_texts(n, cache_dir="~/.cache", min_words=100):
    d = datasets.load_dataset("xsum", split="train", cache_dir=cache_dir)
    texts = [x["document"].replace("\n", " ").strip() for x in d]
    texts = [t for t in texts if len(t.split()) >= min_words]
    random.shuffle(texts)
    return texts[:n]


def generate_gpt4o_mini_text(client, prompt_text, prompt_tokens=30, max_tokens=200):
    """Take first ~30 words as a prompt, generate a completion."""
    prompt = " ".join(prompt_text.split()[:prompt_tokens])
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are a news writer. Continue the following article naturally."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=1.0,
            )
            completion = resp.choices[0].message.content.strip()
            return prompt + " " + completion
        except Exception as e:
            print(f"  Generation error (attempt {attempt+1}): {e}")
            time.sleep(5)
    return None


# ── scoring ───────────────────────────────────────────────────────────────────

def masked_score_for_position(client, words, pos, model="gpt-4o-mini", top_logprobs=20):
    """
    Mask word at position `pos`, ask model to fill in the blank.
    Returns (actual_logprob, mean_ref, var_ref) or None on failure.
    """
    actual_word = words[pos]
    masked = words[:pos] + ["[BLANK]"] + words[pos+1:]
    masked_text = " ".join(masked)

    prompt = (
        f"Fill in [BLANK] with exactly one word that fits naturally.\n"
        f"Text: {masked_text}\n"
        f"Answer (one word only):"
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
                logprobs=True,
                top_logprobs=top_logprobs,
            )
            break
        except Exception as e:
            if attempt == 2:
                return None
            time.sleep(5 * (attempt + 1))

    content = resp.choices[0].logprobs.content
    if not content:
        return None

    # Use the first output token's logprob data
    token_data = content[0]
    top_lps = token_data.top_logprobs

    # Check if the actual word appears in the top predictions
    predicted_word = token_data.token.strip().lower()
    actual_lower = actual_word.strip().lower()

    # actual_logprob: find it in top_logprobs, or use the bottom of the list
    actual_logprob = None
    for lp in top_lps:
        if lp.token.strip().lower() == actual_lower:
            actual_logprob = lp.logprob
            break

    if actual_logprob is None:
        # Actual word not in top-20; assign log prob of the last entry (lower bound)
        actual_logprob = min(lp.logprob for lp in top_lps)

    # Estimate full distribution from top-k (renormalize)
    raw_lps = np.array([lp.logprob for lp in top_lps], dtype=np.float64)
    raw_probs = np.exp(raw_lps)
    total = raw_probs.sum()
    if total <= 0:
        return None
    probs = raw_probs / total
    lprobs = np.log(probs + 1e-30)

    mean_ref = (probs * lprobs).sum()
    var_ref  = (probs * np.square(lprobs)).sum() - mean_ref ** 2

    return actual_logprob, float(mean_ref), float(max(var_ref, 1e-10))


def get_masked_score(client, text, model="gpt-4o-mini", k=20, top_logprobs=20, seed=42):
    """
    Sample k random word positions, compute masked prediction discrepancy.
    Returns scalar score or None if too few valid positions.
    """
    words = text.split()
    if len(words) < 20:
        return None

    rng = random.Random(seed)
    # Avoid first and last 5 words (context boundaries)
    positions = list(range(5, len(words) - 5))
    positions = rng.sample(positions, min(k, len(positions)))

    log_likelihoods, mean_refs, var_refs = [], [], []

    for pos in positions:
        result = masked_score_for_position(client, words, pos, model, top_logprobs)
        if result is None:
            continue
        actual_lp, mean_ref, var_ref = result
        log_likelihoods.append(actual_lp)
        mean_refs.append(mean_ref)
        var_refs.append(var_ref)

    if len(log_likelihoods) < 5:
        return None

    ll_sum   = sum(log_likelihoods)
    mean_sum = sum(mean_refs)
    var_sum  = sum(var_refs)

    return (ll_sum - mean_sum) / np.sqrt(var_sum)


# ── metrics ───────────────────────────────────────────────────────────────────

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


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "Set OPENAI_API_KEY as an environment variable"
    client = OpenAI(api_key=api_key)

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Model:       {args.model}")
    print(f"Samples:     {args.n_samples}")
    print(f"K positions: {args.k_positions} per text")
    print()

    # ── Step 1: load human texts ──────────────────────────────────────────────
    print("Loading human texts from xsum...")
    human_texts = load_xsum_human_texts(args.n_samples, cache_dir=args.cache_dir)
    print(f"  Loaded {len(human_texts)} human texts")

    # ── Step 2: generate machine texts ───────────────────────────────────────
    print(f"\nGenerating {args.n_samples} texts with {args.model}...")
    machine_texts = []
    for text in tqdm.tqdm(human_texts, desc="Generating"):
        gen = generate_gpt4o_mini_text(client, text, prompt_tokens=args.prompt_tokens)
        machine_texts.append(gen)

    # ── Step 3: score all texts ───────────────────────────────────────────────
    print(f"\nScoring with masked prediction ({args.k_positions} positions/text)...")
    results = []
    n_skipped = 0

    for human, machine in tqdm.tqdm(
        zip(human_texts, machine_texts), total=len(human_texts), desc="Scoring pairs"
    ):
        if machine is None:
            n_skipped += 1
            continue

        human_score   = get_masked_score(client, human,   args.model, args.k_positions)
        machine_score = get_masked_score(client, machine, args.model, args.k_positions)

        if human_score is None or machine_score is None:
            n_skipped += 1
            continue

        results.append({
            "human":         human,
            "human_score":   float(human_score),
            "machine":       machine,
            "machine_score": float(machine_score),
        })

    print(f"\nScored {len(results)}/{len(human_texts)} pairs  ({n_skipped} skipped)")

    if len(results) < 5:
        print("Not enough results.")
        return

    # ── Step 4: compute metrics ───────────────────────────────────────────────
    real_scores    = [r["human_score"]   for r in results]
    machine_scores = [r["machine_score"] for r in results]

    print(f"\nHuman   mean/std: {np.mean(real_scores):.3f} / {np.std(real_scores):.3f}")
    print(f"Machine mean/std: {np.mean(machine_scores):.3f} / {np.std(machine_scores):.3f}")
    print(f"Separation:       {np.mean(machine_scores) - np.mean(real_scores):.3f}")

    fpr, tpr, roc_auc = get_roc_metrics(real_scores, machine_scores)
    p, r, pr_auc = get_pr_metrics(real_scores, machine_scores)

    print(f"\nMasked prediction ({args.model}) white-box:")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  PR  AUC: {pr_auc:.4f}")

    print("\n--- Context ---")
    print(f"  Echo gpt-4o-mini cross-model:        0.7778  (GPT-2 generated, GPT-4o-mini scored)")
    print(f"  Fast-DetectGPT GPT-2 white-box:      0.9892  (GPT-2 generated, GPT-2 scored)")
    print(f"  Masked {args.model} white-box: {roc_auc:.4f}  ← new")

    # ── Step 5: save ──────────────────────────────────────────────────────────
    output = {
        "name":     f"masked_{args.model}_whitebox",
        "info":     {"n_samples": len(results), "n_skipped": n_skipped,
                     "model": args.model, "k_positions": args.k_positions},
        "predictions": {"real": real_scores, "samples": machine_scores},
        "raw_results": results,
        "metrics":     {"roc_auc": roc_auc, "fpr": fpr, "tpr": tpr},
        "pr_metrics":  {"pr_auc": pr_auc, "precision": p, "recall": r},
        "loss": 1 - pr_auc,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        type=str, default="gpt-4o-mini")
    parser.add_argument("--n_samples",    type=int, default=50,
                        help="Number of text pairs to evaluate")
    parser.add_argument("--k_positions",  type=int, default=20,
                        help="Random word positions to mask per text")
    parser.add_argument("--prompt_tokens",type=int, default=30,
                        help="Words from human text used as generation prompt")
    parser.add_argument("--top_logprobs", type=int, default=20)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--cache_dir",    type=str, default="~/.cache")
    parser.add_argument("--output_file",  type=str,
                        default="results/masked_detect_whitebox_results.json")
    args = parser.parse_args()
    main(args)
