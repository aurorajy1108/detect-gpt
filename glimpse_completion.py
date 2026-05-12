"""
Glimpse + Fast-DetectGPT via OpenAI's completion API (legacy /v1/completions
endpoint with echo=True). This bypasses the echo trick: the API returns
logprobs ON THE INPUT TEXT directly, with no chat conditioning.

Models:
  - davinci-002    : ~13B base, $2/1M tokens
  - babbage-002    : ~1.3B base, $0.40/1M tokens

Usage:
    OPENAI_API_KEY=... python3 glimpse_completion.py [--model davinci-002]
"""

import os, json, math, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from openai import OpenAI
from sklearn.metrics import roc_curve, auc

DATA_FILE = "dataset_manifests/hc3_en_mixed_200.json"
DEFAULT_MODEL = "davinci-002"
TOP_K_VALUES = [5, 10, 20]
VOCAB_SIZE = 50257  # GPT-2/3 vocab (davinci-002 uses cl100k actually; close enough for tail estimate)


def score_one(client, model, text, max_retries=2):
    """Use /v1/completions echo=True logprobs=N to score the text directly.
    Returns list of (chosen_logprob, [top_logprobs]) for each token, or None.
    """
    for attempt in range(max_retries):
        try:
            resp = client.completions.create(
                model=model,
                prompt=text,
                echo=True,
                logprobs=20,    # top-20 logprobs per position (max OpenAI supports)
                max_tokens=0,   # don't generate, just score the prompt
                temperature=0,
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                return None, f"API error: {e}"
            time.sleep(3 * (attempt + 1))

    choice = resp.choices[0]
    lp_obj = choice.logprobs
    if lp_obj is None:
        return None, "no logprobs"
    tokens     = lp_obj.tokens or []
    chosen_lps = lp_obj.token_logprobs or []
    top_lps    = lp_obj.top_logprobs or []
    if len(tokens) < 5:
        return None, f"too few tokens ({len(tokens)})"

    # Token 0 has logprob=None (no prefix to condition on). Skip it.
    out = []
    for i in range(1, len(tokens)):
        chosen = chosen_lps[i]
        tops_dict = top_lps[i] if i < len(top_lps) else {}
        if chosen is None:
            continue
        # tops_dict is {token_str: logprob}; convert to list ranked by logprob desc
        tops = sorted(tops_dict.items(), key=lambda kv: -kv[1])
        # filter weird/dummy entries
        tops = [(t, float(lp)) for t, lp in tops if lp is not None and lp > -100.0]
        out.append((float(chosen), tops))
    return out, None


def fast_detect_glimpse(per_token, top_k, vocab_size=VOCAB_SIZE):
    if not per_token:
        return None

    nums, vars_ = [], []
    for chosen_lp, tops in per_token:
        tops_k = tops[:top_k]
        if not tops_k:
            continue
        log_p_top = np.array([lp for _, lp in tops_k])
        p_top     = np.exp(log_p_top)
        mass_top  = float(p_top.sum())
        if mass_top >= 0.999999:
            mu  = float((p_top * log_p_top).sum())
            sig2 = float((p_top * log_p_top**2).sum() - mu**2)
        else:
            M = max(1e-12, 1.0 - mass_top)
            n_tail = max(1, vocab_size - len(tops_k))
            log_tail = math.log(M) - math.log(n_tail)
            mu_top  = float((p_top * log_p_top).sum())
            mu_tail = M * log_tail
            mu = mu_top + mu_tail
            sec_top  = float((p_top * log_p_top**2).sum())
            sec_tail = M * log_tail**2
            sig2 = sec_top + sec_tail - mu**2
        sig2 = max(sig2, 1e-12)
        nums.append(chosen_lp - mu)
        vars_.append(sig2)

    if not nums:
        return None
    return float(np.sum(nums) / math.sqrt(np.sum(vars_)))


def auroc(real, samples):
    labels = [0]*len(real) + [1]*len(samples)
    scores = real + samples
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(auc(fpr, tpr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--out_path", type=str, default=None)
    args = ap.parse_args()

    if args.out_path is None:
        args.out_path = f"results/glimpse_completion_{args.model.replace('/', '_')}.json"

    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "Set OPENAI_API_KEY"
    client = OpenAI(api_key=api_key)

    with open(DATA_FILE) as f:
        records = json.load(f)[:args.n_samples]

    print(f"Scoring {len(records)} pairs via {args.model} (completion API, echo=True)")
    t0 = time.time()

    cache = {"orig": {}, "samp": {}}
    tasks = []
    for i, r in enumerate(records):
        tasks.append((i, "orig", r["original"]))
        tasks.append((i, "samp", r["sampled"]))

    def fetch(idx, kind, text):
        result, err = score_one(client, args.model, text)
        return idx, kind, result, err

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(fetch, i, k, t) for (i, k, t) in tasks]
        for n_done, fut in enumerate(as_completed(futures), 1):
            i, kind, result, err = fut.result()
            if result is None:
                print(f"[skip {kind} #{i}] {err}")
            cache[kind][i] = (result, err)
            if n_done % 25 == 0:
                print(f"  {n_done}/{len(tasks)} done ({(time.time()-t0):.0f}s elapsed)")

    print(f"\nAPI calls done in {(time.time()-t0):.0f}s")

    aurocs = {}
    for k in TOP_K_VALUES:
        real_scores, samp_scores, n_skip = [], [], 0
        for i in range(len(records)):
            o = cache["orig"].get(i, (None, "missing"))[0]
            s = cache["samp"].get(i, (None, "missing"))[0]
            if o is None or s is None:
                n_skip += 1; continue
            os_ = fast_detect_glimpse(o, k)
            ss_ = fast_detect_glimpse(s, k)
            if os_ is None or ss_ is None or math.isnan(os_) or math.isnan(ss_):
                n_skip += 1; continue
            real_scores.append(os_)
            samp_scores.append(ss_)
        a = auroc(real_scores, samp_scores) if len(real_scores) >= 5 else None
        print(f"top_k={k:>3}  n_used={len(real_scores)}  n_skip={n_skip}  ROC AUC={a}")
        aurocs[k] = {"auroc": a, "n_used": len(real_scores), "n_skip": n_skip,
                     "real_scores": real_scores, "samp_scores": samp_scores}

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump({
            "model": args.model,
            "n_samples": len(records),
            "data_file": DATA_FILE,
            "vocab_size_assumed": VOCAB_SIZE,
            "aurocs_by_top_k": aurocs,
        }, f, indent=2)
    print(f"\nSaved to {args.out_path}")


if __name__ == "__main__":
    main()
