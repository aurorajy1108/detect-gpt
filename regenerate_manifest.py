"""
Regenerate the `sampled` field of an HC3 manifest using a HuggingFace model.

Each entry in the input manifest has {prompt, original, sampled, source}; we
replace `sampled` with the model's own answer to `prompt`. Output is a new
manifest with the same shape, ready to feed into run.py via --data_file.

Usage:
    python regenerate_manifest.py \
        --model meta-llama/Llama-3.1-8B \
        --input_manifest dataset_manifests/hc3_en_mixed_200.json \
        --output_manifest dataset_manifests/hc3_en_regen_llama_3_1_8b.json \
        --n_samples 200 \
        --max_new_tokens 200
"""

import argparse
import json
import os
import time

import torch
import transformers
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input_manifest", required=True)
    ap.add_argument("--output_manifest", required=True)
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--cache_dir", default=os.path.expanduser("~/.cache"))
    ap.add_argument("--prompt_template", type=str, default=None,
                    help="Template with {prompt}. If None, just use the raw prompt.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {args.model}...")

    tok = transformers.AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, torch_dtype=dtype
    ).to(device)
    model.eval()

    with open(args.input_manifest) as f:
        records = json.load(f)
    records = records[: args.n_samples]
    print(f"Regenerating {len(records)} samples...")

    out_records = []
    t0 = time.time()
    for r in tqdm(records, desc="Generating"):
        prompt = r["prompt"]
        if args.prompt_template:
            prompt = args.prompt_template.format(prompt=prompt)

        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tok.pad_token_id,
            )
        # strip the prompt tokens
        new_tokens = out[0, inputs.input_ids.shape[1]:]
        generated = tok.decode(new_tokens, skip_special_tokens=True).strip().replace("\n", " ")
        out_records.append({**r, "sampled": generated})

    print(f"Generation took {time.time()-t0:.1f}s")
    os.makedirs(os.path.dirname(args.output_manifest) or ".", exist_ok=True)
    with open(args.output_manifest, "w") as f:
        json.dump(out_records, f, indent=2)
    print(f"Wrote {len(out_records)} records to {args.output_manifest}")


if __name__ == "__main__":
    main()
