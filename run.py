import matplotlib.pyplot as plt
import numpy as np
import datasets
import transformers
import re
import torch
import torch.nn.functional as F
import tqdm
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import argparse
import datetime
import os
import json
import functools
import hashlib
import custom_datasets
import model_backends
from multiprocessing.pool import ThreadPool
import time
from huggingface_hub.errors import OfflineModeIsEnabled
from pathlib import Path
from contextlib import contextmanager



# 15 colorblind-friendly colors
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
            "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
            "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")
resolved_model_backend = "auto"
current_loaded_model = None


def sanitize_name(value):
    return value.replace("/", "_")


def offline_mode_enabled():
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


@contextmanager
def temporarily_disable_hf_offline():
    old_hf = os.environ.pop("HF_HUB_OFFLINE", None)
    old_tf = os.environ.pop("TRANSFORMERS_OFFLINE", None)
    try:
        yield
    finally:
        if old_hf is not None:
            os.environ["HF_HUB_OFFLINE"] = old_hf
        if old_tf is not None:
            os.environ["TRANSFORMERS_OFFLINE"] = old_tf


def model_cache_root(model_name):
    return Path(cache_dir) / f"models--{model_name.replace('/', '--')}"


def build_merged_local_snapshot(model_name, required_files):
    root = model_cache_root(model_name)
    snapshots_dir = root / "snapshots"
    if not snapshots_dir.exists():
        return None

    merged_dir = root / "merged-local"
    merged_dir.mkdir(parents=True, exist_ok=True)

    available = {}
    for snapshot in snapshots_dir.iterdir():
        if not snapshot.is_dir():
            continue
        for filename in required_files:
            candidate = snapshot / filename
            if candidate.exists() and filename not in available:
                available[filename] = candidate

    if not all(name in available for name in required_files):
        return None

    for filename, source in available.items():
        target = merged_dir / filename
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(source)

    return str(merged_dir)


def get_gpt2_tokenizer():
    global GPT2_TOKENIZER
    if GPT2_TOKENIZER is not None:
        return GPT2_TOKENIZER
    try:
        GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained(
            'gpt2',
            cache_dir=cache_dir,
            local_files_only=offline_mode_enabled(),
        )
    except Exception:
        with temporarily_disable_hf_offline():
            GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
    return GPT2_TOKENIZER


def count_tokens_for_api_texts(texts):
    if args.tinker_model is not None and tinker_backend is not None:
        tokenizer = tinker_backend.tokenizer
    else:
        tokenizer = get_gpt2_tokenizer()
    return sum(len(tokenizer.encode(text)) for text in texts)


def using_local_transformers_backend():
    if args.openai_model is not None:
        return False
    if args.tinker_model is None:
        return True
    return resolved_model_backend == "hf"


def get_safe_args_dict(args):
    safe_args = dict(args.__dict__)
    for secret_key in ['openai_key', 'tinker_api_key']:
        if safe_args.get(secret_key):
            safe_args[secret_key] = "***REDACTED***"
    return safe_args


def build_result_folder_prefix(args, resolved_backend):
    sampling_string = "top_k" if args.do_top_k else ("top_p" if args.do_top_p else "temp")
    output_subfolder = f"{args.output_name}/" if args.output_name else ""
    if args.tinker_model is not None:
        backend_prefix = "hf-" if resolved_backend == "hf" else "tinker-"
        base_model_name = backend_prefix + sanitize_name(args.tinker_model)
    elif args.openai_model is None:
        base_model_name = sanitize_name(args.base_model_name)
    else:
        base_model_name = "openai-" + sanitize_name(args.openai_model)
    scoring_model_string = sanitize_name(f"-{args.scoring_model_name}") if args.scoring_model_name else ""
    mask_model_string = sanitize_name(args.mask_filling_model_name)
    return f"{output_subfolder}{base_model_name}{scoring_model_string}-{mask_model_string}-{sampling_string}"


def build_run_cache_signature(safe_args, resolved_backend):
    payload = {
        "resolved_model_backend": resolved_backend,
        "args": safe_args,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expected_result_filenames(args, resolved_backend, n_perturbation_list):
    expected = {"args.json", "raw_data.json"}
    local_backend = args.openai_model is None and (args.tinker_model is None or resolved_backend == "hf")

    if not args.skip_baselines:
        expected.update({
            "likelihood_threshold_results.json",
            "roberta-base-openai-detector_results.json",
            "roberta-large-openai-detector_results.json",
        })
        if local_backend:
            expected.update({
                "rank_threshold_results.json",
                "logrank_threshold_results.json",
                "entropy_threshold_results.json",
            })

    if not args.baselines_only:
        for n_perturbations in n_perturbation_list:
            expected.add(f"perturbation_{n_perturbations}_d_results.json")
            expected.add(f"perturbation_{n_perturbations}_z_results.json")

    return expected


def find_completed_cached_run(results_root, safe_args, signature, expected_files):
    if not os.path.isdir(results_root):
        return None

    for entry in sorted(os.listdir(results_root)):
        candidate = os.path.join(results_root, entry)
        if not os.path.isdir(candidate):
            continue

        meta_path = os.path.join(candidate, "run_cache_meta.json")
        if os.path.exists(meta_path):
            try:
                meta = json.load(open(meta_path, "r", encoding="utf-8"))
            except Exception:
                meta = None
            if meta and meta.get("signature") == signature and meta.get("completed") is True:
                if all(os.path.exists(os.path.join(candidate, name)) for name in expected_files):
                    return candidate

        args_path = os.path.join(candidate, "args.json")
        if not os.path.exists(args_path):
            continue
        try:
            saved_args = json.load(open(args_path, "r", encoding="utf-8"))
        except Exception:
            continue
        if saved_args == safe_args and all(os.path.exists(os.path.join(candidate, name)) for name in expected_files):
            return candidate

    return None


def load_base_model():
    print('MOVING BASE MODEL TO GPU...', end='', flush=True)
    start = time.time()
    global current_loaded_model
    if current_loaded_model == "base":
        print(f'DONE ({time.time() - start:.2f}s)')
        return
    try:
        if current_loaded_model == "mask":
            mask_model.cpu()
    except NameError:
        pass
    if using_local_transformers_backend() and base_model is not None:
        base_model.to(DEVICE)
    current_loaded_model = "base"
    print(f'DONE ({time.time() - start:.2f}s)')


def load_mask_model():
    print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    start = time.time()
    global current_loaded_model
    if current_loaded_model == "mask":
        print(f'DONE ({time.time() - start:.2f}s)')
        return
    if using_local_transformers_backend() and base_model is not None and current_loaded_model == "base":
        base_model.cpu()
    if not args.random_fills:
        mask_model.to(DEVICE)
    current_loaded_model = "mask"
    print(f'DONE ({time.time() - start:.2f}s)')


def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    generation_max_length = max(150, 8 * max(n_expected))
    outputs = mask_model.generate(
        **tokens,
        max_length=generation_max_length,
        do_sample=True,
        top_p=args.mask_top_p,
        num_return_sequences=1,
        eos_token_id=stop_id,
    )
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def resolve_failed_perturbations(perturbed_texts, original_texts, failed_idxs):
    for idx in failed_idxs:
        # Fall back to the original text so one stubborn sample doesn't stall the whole run.
        perturbed_texts[idx] = original_texts[idx]
    return perturbed_texts


def perturb_texts_(texts, span_length, pct, ceil_pct=False):
    if not args.random_fills:
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
        raw_fills = replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            if attempts > args.max_perturbation_retries:
                print(
                    f'WARNING: {len(idxs)} texts still have no fills after '
                    f'{args.max_perturbation_retries} retries. Falling back to original text.'
                )
                perturbed_texts = resolve_failed_perturbations(perturbed_texts, texts, idxs)
                break
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = replace_masks(masked_texts)
            extracted_fills = extract_fills(raw_fills)
            new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
    else:
        if args.random_fills_tokens:
            # tokenize base_tokenizer
            tokens = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
            valid_tokens = tokens.input_ids != base_tokenizer.pad_token_id
            replace_pct = args.pct_words_masked * (args.span_length / (args.span_length + 2 * args.buffer_size))

            # replace replace_pct of input_ids with random tokens
            random_mask = torch.rand(tokens.input_ids.shape, device=DEVICE) < replace_pct
            random_mask &= valid_tokens
            random_tokens = torch.randint(0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            # while any of the random tokens are special tokens, replace them with random non-special tokens
            while any(base_tokenizer.decode(x) in base_tokenizer.all_special_tokens for x in random_tokens):
                random_tokens = torch.randint(0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            tokens.input_ids[random_mask] = random_tokens
            perturbed_texts = base_tokenizer.batch_decode(tokens.input_ids, skip_special_tokens=True)
        else:
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
            perturbed_texts = masked_texts
            # replace each <extra_id_*> with args.span_length random words from FILL_DICTIONARY
            for idx, text in enumerate(perturbed_texts):
                filled_text = text
                for fill_idx in range(count_masks([text])[0]):
                    fill = random.sample(FILL_DICTIONARY, span_length)
                    filled_text = filled_text.replace(f"<extra_id_{fill_idx}>", " ".join(fill))
                assert count_masks([filled_text])[0] == 0, "Failed to replace all masks"
                perturbed_texts[idx] = filled_text

    return perturbed_texts


def perturb_texts(texts, span_length, pct, ceil_pct=False):
    chunk_size = args.chunk_size
    if '11b' in mask_filling_model_name:
        chunk_size //= 2

    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
    return outputs


def drop_last_word(text):
    return ' '.join(text.split(' ')[:-1])


def _openai_sample(p):
    if args.dataset != 'pubmed':  # keep Answer: prefix for pubmed
        p = drop_last_word(p)

    # sample from the openai model
    kwargs = { "engine": args.openai_model, "max_tokens": 200 }
    if args.do_top_p:
        kwargs['top_p'] = args.top_p
    
    r = openai.Completion.create(prompt=f"{p}", **kwargs)
    return p + r['choices'][0].text


def _tinker_sample(prompt):
    return tinker_backend.sample(
        prompt,
        max_tokens=200,
        temperature=1.0,
        top_p=args.top_p if args.do_top_p else None,
        top_k=args.top_k if args.do_top_k else None,
        num_samples=1,
    )[0]


def clean_generated_text(text):
    original = strip_newlines(text).strip()
    special_markers = [
        "<|endoftext|>",
        "<|end_of_text|>",
        "<|eot_id|>",
        "<｜end▁of▁sentence｜>",
    ]
    for marker in special_markers:
        text = text.replace(marker, " ")
    cleaned = strip_newlines(text).strip()
    return cleaned or original


def combine_prompt_and_generation(prompt, generated):
    prompt = strip_newlines(prompt).strip()
    generated = clean_generated_text(generated)
    if not prompt:
        return generated
    if generated.startswith(prompt):
        return generated
    return f"{prompt} {generated}".strip()


def build_prefixes_from_tokenizer(texts, prompt_tokens):
    prefixes = []
    for text in texts:
        encoded = base_tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        prefix_ids = encoded[:prompt_tokens]
        prefixes.append(base_tokenizer.decode(prefix_ids, skip_special_tokens=True))
    return prefixes


def build_generation_prompts(batch_records):
    prompts = []
    for record in batch_records:
        prompt = (record.get("prompt") or record["original"]).strip()
        if not args.use_dataset_samples and args.dataset in custom_datasets.PAIR_DATASETS and record.get("prompt"):
            prompt = (
                "Answer the following question in a complete paragraph.\n"
                f"Question: {prompt}\n"
                "Answer:"
            )
        prompts.append(prompt)
    return prompts


# sample from base_model using ****only**** the first 30 tokens in each example as context
def sample_from_model(texts, min_words=55, prompt_tokens=30, prompt_texts=None, max_tries=10):
    prompt_texts = prompt_texts or texts

    # encode each text as a list of token ids
    if args.tinker_model:
        if args.dataset in custom_datasets.PAIR_DATASETS:
            effective_prompts = prompt_texts
        else:
            effective_prompts = build_prefixes_from_tokenizer(prompt_texts, prompt_tokens)
        decoded = ['' for _ in range(len(prompt_texts))]
        tries = 0
        while (m := min(len(x.split()) for x in decoded)) < min_words:
            if tries >= max_tries:
                print()
                print(f"WARNING: hit max sample tries ({max_tries}); continuing with min words {m} < requested {min_words}")
                break
            if tries != 0:
                print()
                print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")
            decoded = []
            for prompt in effective_prompts:
                generated = _tinker_sample(prompt)
                if args.dataset in custom_datasets.PAIR_DATASETS:
                    decoded.append(clean_generated_text(generated))
                else:
                    decoded.append(combine_prompt_and_generation(prompt, generated))
            tries += 1
    else:
        if args.dataset == 'pubmed' and prompt_texts == texts:
            texts = [t[:t.index(custom_datasets.SEPARATOR)] for t in texts]
            all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
        else:
            effective_prompts = build_prefixes_from_tokenizer(prompt_texts, prompt_tokens)
            all_encoded = base_tokenizer(effective_prompts, return_tensors="pt", padding=True).to(DEVICE)

        if args.openai_model:
            # decode the prefixes back into text
            prefixes = base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
            pool = ThreadPool(args.batch_size)

            decoded = pool.map(_openai_sample, prefixes)
            decoded = [clean_generated_text(text) for text in decoded]
        else:
            decoded = ['' for _ in range(len(texts))]

            # sample from the model until we get a sample with at least min_words words for each example
            # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
            tries = 0
            while (m := min(len(x.split()) for x in decoded)) < min_words:
                if tries >= max_tries:
                    print()
                    print(f"WARNING: hit max sample tries ({max_tries}); continuing with min words {m} < requested {min_words}")
                    break
                if tries != 0:
                    print()
                    print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

                sampling_kwargs = {}
                if args.do_top_p:
                    sampling_kwargs['top_p'] = args.top_p
                elif args.do_top_k:
                    sampling_kwargs['top_k'] = args.top_k
                min_length = 50 if args.dataset in ['pubmed'] else 150
                outputs = base_model.generate(
                    **all_encoded,
                    min_length=min_length,
                    max_length=200,
                    do_sample=True,
                    remove_invalid_values=True,
                    **sampling_kwargs,
                    pad_token_id=base_tokenizer.eos_token_id,
                    eos_token_id=base_tokenizer.eos_token_id,
                )
                decoded = base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                decoded = [clean_generated_text(text) for text in decoded]
                tries += 1

    if args.openai_model or args.tinker_model:
        global API_TOKEN_COUNTER

        total_tokens = count_tokens_for_api_texts(decoded)
        API_TOKEN_COUNTER += total_tokens

    return decoded


def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])[:-1]
    labels = labels.view(-1)[1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean()


def sanitize_local_logits(logits):
    # Some local GPT-style checkpoints can emit rare NaN/Inf/extreme logits under the
    # current torch/transformers stack on this machine. Clean them before scoring.
    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(logits, min=-100.0, max=100.0)


def has_enough_tokens_for_local_scoring(text):
    if not text or not text.strip():
        return False
    tokenized = base_tokenizer(text, return_tensors="pt")
    return tokenized.input_ids.shape[-1] >= 2


# Get the log likelihood of each text under the base_model
def get_ll(text):
    if args.openai_model is None and args.tinker_model is None and not has_enough_tokens_for_local_scoring(text):
        return -100.0
    if args.tinker_model:
        return tinker_backend.get_mean_logprob(text)
    if args.openai_model:        
        kwargs = { "engine": args.openai_model, "temperature": 0, "max_tokens": 0, "echo": True, "logprobs": 0}
        r = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
        result = r['choices'][0]
        tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

        assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

        return np.mean(logprobs)
    else:
        with torch.no_grad():
            tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
            labels = tokenized.input_ids
            logits = base_model(**tokenized).logits
            logits = sanitize_local_logits(logits)
            value = get_likelihood(logits, labels).item()
            if not np.isfinite(value):
                return -100.0
            return value


def get_lls_local_batched(texts):
    if not texts:
        return []

    if args.tinker_model and resolved_model_backend == "hf" and hasattr(tinker_backend, "get_mean_logprobs"):
        return tinker_backend.get_mean_logprobs(texts, batch_size=args.batch_size)

    results = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        tokenized = base_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(DEVICE)

        with torch.no_grad():
            logits = base_model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized.get("attention_mask"),
            ).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = tokenized["input_ids"][:, 1:]
        if "attention_mask" in tokenized:
            shift_mask = tokenized["attention_mask"][:, 1:].float()
        else:
            shift_mask = torch.ones_like(shift_labels, dtype=torch.float32)

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs * shift_mask
        lengths = shift_mask.sum(dim=-1).clamp_min(1.0)
        batch_means = token_log_probs.sum(dim=-1) / lengths
        results.extend(batch_means.detach().cpu().tolist())

    return [float(x) for x in results]


def get_lls(texts):
    if using_local_transformers_backend():
        return get_lls_local_batched(texts)
    else:
        global API_TOKEN_COUNTER

        total_tokens = count_tokens_for_api_texts(texts)
        API_TOKEN_COUNTER += total_tokens * 2  # multiply by two because OpenAI double-counts echo_prompt tokens

        pool = ThreadPool(args.batch_size)
        return pool.map(get_ll, texts)


# get the average rank of each observed token sorted by model likelihood
def get_rank(text, log=False):
    assert using_local_transformers_backend(), "get_rank only implemented for local HuggingFace models"

    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:,:-1]
        logits = sanitize_local_logits(logits)
        labels = tokenized.input_ids[:,1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:,-1], matches[:,-2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1 # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()


# get average entropy of each token in the text
def get_entropy(text):
    assert using_local_transformers_backend(), "get_entropy only implemented for local HuggingFace models"

    if not has_enough_tokens_for_local_scoring(text):
        return 100.0

    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:,:-1]
        logits = sanitize_local_logits(logits)
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        value = -neg_entropy.sum(-1).mean().item()
        if not np.isfinite(value):
            return 100.0
        return value


def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)


# save the ROC curve for each experiment, given a list of output dictionaries, one for each experiment, using colorblind-friendly colors
def save_roc_curves(experiments):
    # first, clear plt
    plt.clf()

    for experiment, color in zip(experiments, COLORS):
        metrics = experiment["metrics"]
        plt.plot(metrics["fpr"], metrics["tpr"], label=f"{experiment['name']}, roc_auc={metrics['roc_auc']:.3f}", color=color)
        # print roc_auc for this experiment
        print(f"{experiment['name']} roc_auc: {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves ({base_model_name} - {args.mask_filling_model_name})')
    plt.legend(loc="lower right", fontsize=6)
    plt.savefig(f"{SAVE_FOLDER}/roc_curves.png")


# save the histogram of log likelihoods in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed
def save_ll_histograms(experiments):
    # first, clear plt
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)
            plt.hist([r["sampled_ll"] for r in results], alpha=0.5, bins='auto', label='sampled')
            plt.hist([r["perturbed_sampled_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed sampled')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.subplot(1, 2, 2)
            plt.hist([r["original_ll"] for r in results], alpha=0.5, bins='auto', label='original')
            plt.hist([r["perturbed_original_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed original')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(f"{SAVE_FOLDER}/ll_histograms_{experiment['name']}.png")
        except:
            pass


# save the histograms of log likelihood ratios in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed
def save_llr_histograms(experiments):
    # first, clear plt
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)

            # compute the log likelihood ratio for each result
            for r in results:
                r["sampled_llr"] = r["sampled_ll"] - r["perturbed_sampled_ll"]
                r["original_llr"] = r["original_ll"] - r["perturbed_original_ll"]
            
            plt.hist([r["sampled_llr"] for r in results], alpha=0.5, bins='auto', label='sampled')
            plt.hist([r["original_llr"] for r in results], alpha=0.5, bins='auto', label='original')
            plt.xlabel("log likelihood ratio")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(f"{SAVE_FOLDER}/llr_histograms_{experiment['name']}.png")
        except:
            pass


def get_perturbation_results(span_length=10, n_perturbations=1, n_samples=500):
    load_mask_model()

    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    original_text = data["original"]
    sampled_text = data["sampled"]

    perturb_fn = functools.partial(perturb_texts, span_length=span_length, pct=args.pct_words_masked)

    p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(n_perturbations)])
    p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])
    for _ in range(n_perturbation_rounds - 1):
        try:
            p_sampled_text, p_original_text = perturb_fn(p_sampled_text), perturb_fn(p_original_text)
        except AssertionError:
            break

    assert len(p_sampled_text) == len(sampled_text) * n_perturbations, f"Expected {len(sampled_text) * n_perturbations} perturbed samples, got {len(p_sampled_text)}"
    assert len(p_original_text) == len(original_text) * n_perturbations, f"Expected {len(original_text) * n_perturbations} perturbed samples, got {len(p_original_text)}"

    for idx in range(len(original_text)):
        results.append({
            "original": original_text[idx],
            "sampled": sampled_text[idx],
            "perturbed_sampled": p_sampled_text[idx * n_perturbations: (idx + 1) * n_perturbations],
            "perturbed_original": p_original_text[idx * n_perturbations: (idx + 1) * n_perturbations]
        })

    load_base_model()

    for res in tqdm.tqdm(results, desc="Computing log likelihoods"):
        p_sampled_ll = get_lls(res["perturbed_sampled"])
        p_original_ll = get_lls(res["perturbed_original"])
        res["original_ll"] = get_ll(res["original"])
        res["sampled_ll"] = get_ll(res["sampled"])
        res["all_perturbed_sampled_ll"] = p_sampled_ll
        res["all_perturbed_original_ll"] = p_original_ll
        res["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
        res["perturbed_original_ll"] = np.mean(p_original_ll)
        res["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1
        res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1

    return results


def run_perturbation_experiment(results, criterion, span_length=10, n_perturbations=1, n_samples=500):
    # compute diffs with perturbed
    predictions = {'real': [], 'samples': []}
    for res in results:
        if criterion == 'd':
            predictions['real'].append(res['original_ll'] - res['perturbed_original_ll'])
            predictions['samples'].append(res['sampled_ll'] - res['perturbed_sampled_ll'])
        elif criterion == 'z':
            if res['perturbed_original_ll_std'] == 0:
                res['perturbed_original_ll_std'] = 1
                print("WARNING: std of perturbed original is 0, setting to 1")
                print(f"Number of unique perturbed original texts: {len(set(res['perturbed_original']))}")
                print(f"Original text: {res['original']}")
            if res['perturbed_sampled_ll_std'] == 0:
                res['perturbed_sampled_ll_std'] = 1
                print("WARNING: std of perturbed sampled is 0, setting to 1")
                print(f"Number of unique perturbed sampled texts: {len(set(res['perturbed_sampled']))}")
                print(f"Sampled text: {res['sampled']}")
            predictions['real'].append((res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])
            predictions['samples'].append((res['sampled_ll'] - res['perturbed_sampled_ll']) / res['perturbed_sampled_ll_std'])

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    name = f'perturbation_{n_perturbations}_{criterion}'
    print(f"{name} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'name': name,
        'predictions': predictions,
        'info': {
            'pct_words_masked': args.pct_words_masked,
            'span_length': span_length,
            'n_perturbations': n_perturbations,
            'n_samples': n_samples,
        },
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }


def run_baseline_threshold_experiment(criterion_fn, name, n_samples=500):
    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    for batch in tqdm.tqdm(range(n_samples // batch_size), desc=f"Computing {name} criterion"):
        original_text = data["original"][batch * batch_size:(batch + 1) * batch_size]
        sampled_text = data["sampled"][batch * batch_size:(batch + 1) * batch_size]

        for idx in range(len(original_text)):
            original_crit = criterion_fn(original_text[idx])
            sampled_crit = criterion_fn(sampled_text[idx])
            if not np.isfinite(original_crit):
                original_crit = -100.0
            if not np.isfinite(sampled_crit):
                sampled_crit = -100.0
            results.append({
                "original": original_text[idx],
                "original_crit": original_crit,
                "sampled": sampled_text[idx],
                "sampled_crit": sampled_crit,
            })

    # compute prediction scores for real/sampled passages
    predictions = {
        'real': [x["original_crit"] for x in results],
        'samples': [x["sampled_crit"] for x in results],
    }

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"{name}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'name': f'{name}_threshold',
        'predictions': predictions,
        'info': {
            'n_samples': n_samples,
        },
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }


# strip newlines from each example; replace one or more newlines with a single space
def strip_newlines(text):
    return ' '.join(text.split())


# trim to shorter length
def trim_to_shorter_length(texta, textb):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
    texta = ' '.join(texta.split(' ')[:shorter_length])
    textb = ' '.join(textb.split(' ')[:shorter_length])
    return texta, textb


def truncate_to_substring(text, substring, idx_occurrence):
    # truncate everything after the idx_occurrence occurrence of substring
    assert idx_occurrence > 0, 'idx_occurrence must be > 0'
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]


def generate_samples(raw_data, batch_size):
    torch.manual_seed(42)
    np.random.seed(42)
    data = {
        "original": [],
        "sampled": [],
    }

    num_batches = max(1, (len(raw_data) + batch_size - 1) // batch_size)
    for batch in range(num_batches):
        print('Generating samples for batch', batch, 'of', num_batches)
        batch_records = raw_data[batch * batch_size:(batch + 1) * batch_size]
        if not batch_records:
            continue
        target_min_words = args.min_sample_words
        if args.dataset in ['pubmed']:
            target_min_words = 30
        if isinstance(batch_records[0], dict):
            original_text = [record["original"] for record in batch_records]
            prompt_text = build_generation_prompts(batch_records)
            if args.use_dataset_samples:
                sampled_text = [record.get("sampled") or "" for record in batch_records]
            else:
                sampled_text = sample_from_model(
                    original_text,
                    min_words=target_min_words,
                    prompt_texts=prompt_text,
                    max_tries=args.max_sample_tries,
                )
        else:
            original_text = batch_records
            sampled_text = sample_from_model(
                original_text,
                min_words=target_min_words,
                max_tries=args.max_sample_tries,
            )

        for o, s in zip(original_text, sampled_text):
            if args.dataset == 'pubmed':
                s = truncate_to_substring(s, 'Question:', 2)
                o = o.replace(custom_datasets.SEPARATOR, ' ')

            o, s = trim_to_shorter_length(o, s)

            # add to the data
            data["original"].append(o)
            data["sampled"].append(s)
    
    if args.pre_perturb_pct > 0:
        print(f'APPLYING {args.pre_perturb_pct}, {args.pre_perturb_span_length} PRE-PERTURBATIONS')
        load_mask_model()
        data["sampled"] = perturb_texts(data["sampled"], args.pre_perturb_span_length, args.pre_perturb_pct, ceil_pct=True)
        load_base_model()

    return data


def generate_data(dataset, key):
    # load data
    loaded_from_manifest = args.dataset_manifest is not None
    if loaded_from_manifest:
        with open(args.dataset_manifest, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    elif dataset in custom_datasets.DATASETS:
        data = custom_datasets.load_records(dataset, cache_dir=cache_dir)
    elif dataset in custom_datasets.PAIR_DATASETS:
        data = custom_datasets.load_records(
            dataset,
            cache_dir=cache_dir,
            split=args.dataset_split,
            source=args.dataset_source,
        )
    else:
        data = datasets.load_dataset(dataset, split=args.dataset_split, cache_dir=cache_dir)[key]

    # get unique examples, strip whitespace, and remove newlines
    # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
    # then take just the examples that are <= 512 tokens (for the mask model)
    # then generate n_samples samples

    if data and isinstance(data[0], dict):
        deduped = []
        seen = set()
        for record in data:
            normalized = {}
            for field in ['prompt', 'original', 'sampled', 'source']:
                value = record.get(field)
                if isinstance(value, str):
                    value = strip_newlines(value.strip())
                normalized[field] = value
            if not normalized['original']:
                continue
            dedupe_key = tuple(normalized.get(field) for field in ['prompt', 'original', 'sampled', 'source'])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            deduped.append(normalized)
        data = deduped
    else:
        # remove duplicates from the data
        data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

        # strip whitespace around each example
        data = [x.strip() for x in data]

        # remove newlines from each example
        data = [strip_newlines(x) for x in data]

    # try to keep only examples with > 250 words
    if dataset in ['writing', 'squad', 'xsum']:
        if data and isinstance(data[0], dict):
            long_data = [x for x in data if len(x['original'].split()) > 250]
        else:
            long_data = [x for x in data if len(x.split()) > 250]
        if len(long_data) > 0:
            data = long_data

    if not loaded_from_manifest:
        random.seed(0)
        random.shuffle(data)

        data = data[:5_000]

        # keep only examples with <= 512 tokens according to mask_tokenizer
        # this step has the extra effect of removing examples with low-quality/garbage content
        if data and isinstance(data[0], dict):
            tokenized_data = preproc_tokenizer([record['original'] for record in data])
            data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]
        else:
            tokenized_data = preproc_tokenizer(data)
            data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]

    # print stats about remainining data
    print(f"Total number of samples: {len(data)}")
    if data and isinstance(data[0], dict):
        print(f"Average number of words: {np.mean([len(x['original'].split()) for x in data])}")
    else:
        print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

    return generate_samples(data[:n_samples], batch_size=batch_size)


def load_base_model_and_tokenizer(name):
    if args.tinker_model is not None and not using_local_transformers_backend():
        return None, tinker_backend.tokenizer
    if args.tinker_model is not None and using_local_transformers_backend():
        return tinker_backend.model, tinker_backend.tokenizer

    if args.openai_model is None:
        print(f'Loading BASE model {args.base_model_name}...')
        base_model_kwargs = {}
        if 'gpt-j' in name or 'neox' in name:
            base_model_kwargs.update(dict(torch_dtype=torch.float16))
        if 'gpt-j' in name:
            base_model_kwargs.update(dict(revision='float16'))
        base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, cache_dir=cache_dir)
    else:
        base_model = None

    optional_tok_kwargs = {}
    if "facebook/opt-" in name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if args.dataset in ['pubmed'] or (
        base_model is not None and not getattr(base_model.config, "is_encoder_decoder", False)
    ):
        optional_tok_kwargs['padding_side'] = 'left'
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer


def eval_supervised(data, model):
    print(f'Beginning supervised evaluation with {model}...')
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(model, cache_dir=cache_dir).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)

    real, fake = data['original'], data['sampled']

    with torch.no_grad():
        # get predictions for real
        real_preds = []
        for batch in tqdm.tqdm(range(len(real) // batch_size), desc="Evaluating real"):
            batch_real = real[batch * batch_size:(batch + 1) * batch_size]
            batch_real = tokenizer(batch_real, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            real_preds.extend(detector(**batch_real).logits.softmax(-1)[:,0].tolist())
        
        # get predictions for fake
        fake_preds = []
        for batch in tqdm.tqdm(range(len(fake) // batch_size), desc="Evaluating fake"):
            batch_fake = fake[batch * batch_size:(batch + 1) * batch_size]
            batch_fake = tokenizer(batch_fake, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            fake_preds.extend(detector(**batch_fake).logits.softmax(-1)[:,0].tolist())

    predictions = {
        'real': real_preds,
        'samples': fake_preds,
    }

    fpr, tpr, roc_auc = get_roc_metrics(real_preds, fake_preds)
    p, r, pr_auc = get_precision_recall_metrics(real_preds, fake_preds)
    print(f"{model} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")

    # free GPU memory
    del detector
    torch.cuda.empty_cache()

    return {
        'name': model,
        'predictions': predictions,
        'info': {
            'n_samples': n_samples,
        },
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }


def use_transformers_backend(model_name):
    name = model_name.lower()
    return "olmo" in name or "gpt2" in name or "llama" in name


def resolve_model_backend(args):
    if args.tinker_model is None:
        return "auto"
    if args.model_backend == "auto":
        return "hf" if use_transformers_backend(args.tinker_model) else "tinker"
    return args.model_backend


if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_key', type=str, default="document")
    parser.add_argument('--dataset_split', type=str, default="train")
    parser.add_argument('--dataset_source', type=str, default=None)
    parser.add_argument('--dataset_manifest', type=str, default=None)
    parser.add_argument('--use_dataset_samples', action='store_true')
    parser.add_argument('--pct_words_masked', type=float, default=0.3) # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--n_perturbation_list', type=str, default="1,10")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--scoring_model_name', type=str, default="")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-large")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--base_half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--output_name', type=str, default="")
    parser.add_argument('--openai_model', type=str, default=None)
    parser.add_argument('--openai_key', type=str)
    parser.add_argument('--tinker_model', type=str, default=None)
    parser.add_argument('--tinker_tokenizer_name', type=str, default=None)
    parser.add_argument('--tinker_api_key', type=str, default=None)
    parser.add_argument('--model_backend', type=str, choices=['auto', 'tinker', 'hf'], default='auto')
    parser.add_argument('--hf_quantization', type=str, choices=['none', '4bit', '8bit'], default='none')
    parser.add_argument('--api_cache_dir', type=str, default="api_cache")
    parser.add_argument('--baselines_only', action='store_true')
    parser.add_argument('--skip_baselines', action='store_true')
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--pre_perturb_pct', type=float, default=0.0)
    parser.add_argument('--pre_perturb_span_length', type=int, default=5)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--random_fills_tokens', action='store_true')
    parser.add_argument('--min_sample_words', type=int, default=55)
    parser.add_argument('--max_sample_tries', type=int, default=10)
    parser.add_argument('--max_perturbation_retries', type=int, default=3)
    parser.add_argument('--cache_dir', type=str, default="~/.cache")
    parser.add_argument('--force_rerun', action='store_true')
    args = parser.parse_args()

    API_TOKEN_COUNTER = 0

    if args.openai_model is not None:
        import openai
        assert args.openai_key is not None, "Must provide OpenAI API key as --openai_key"
        openai.api_key = args.openai_key
    if args.openai_model is not None and args.tinker_model is not None:
        raise ValueError("Choose either --openai_model or --tinker_model, not both")
    resolved_model_backend = resolve_model_backend(args)

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

    safe_args = get_safe_args_dict(args)
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    result_folder_prefix = build_result_folder_prefix(args, resolved_model_backend)
    run_signature = build_run_cache_signature(safe_args, resolved_model_backend)
    expected_files = expected_result_filenames(args, resolved_model_backend, n_perturbation_list)
    results_root = os.path.join("results", result_folder_prefix)
    if not args.force_rerun:
        cached_run = find_completed_cached_run(results_root, safe_args, run_signature, expected_files)
        if cached_run is not None:
            print(f"Found completed cached run at: {os.path.abspath(cached_run)}")
            print("Skipping rerun. Use --force_rerun to ignore cache.")
            raise SystemExit(0)

    # define SAVE_FOLDER as the timestamp - base model name - mask filling model name
    # create it if it doesn't exist
    precision_string = "int8" if args.int8 else ("fp16" if args.half else "fp32")
    SAVE_FOLDER = f"tmp_results/{result_folder_prefix}/{START_DATE}-{START_TIME}-{precision_string}-{args.pct_words_masked}-{args.n_perturbation_rounds}-{args.dataset}-{args.n_samples}"
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_FOLDER)}")

    # write args to file
    with open(os.path.join(SAVE_FOLDER, "args.json"), "w") as f:
        json.dump(safe_args, f, indent=4)
    with open(os.path.join(SAVE_FOLDER, "run_cache_meta.json"), "w") as f:
        json.dump({"signature": run_signature, "completed": False}, f, indent=2)

    mask_filling_model_name = args.mask_filling_model_name
    n_samples = args.n_samples
    batch_size = args.batch_size
    n_perturbation_rounds = args.n_perturbation_rounds
    n_similarity_samples = args.n_similarity_samples

    cache_dir = os.path.expanduser(args.cache_dir)
    args.api_cache_dir = os.path.expanduser(args.api_cache_dir)
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")
    print(f"Using API cache dir {args.api_cache_dir}")

    GPT2_TOKENIZER = None

    tinker_backend = None
    if args.tinker_model is not None:
        backend_label = "Hugging Face" if resolved_model_backend == "hf" else "Tinker"
        print(f'Loading {backend_label} sampling backend {args.tinker_model}...')
        if resolved_model_backend == "hf":
            tinker_backend = model_backends.TransformersBackend(
                args.tinker_model,
                quantization=args.hf_quantization,
            )
        else:
            tinker_backend = model_backends.TinkerSamplingBackend(
                args.tinker_model,
                cache_dir=cache_dir,
                tokenizer_name=args.tinker_tokenizer_name,
                api_key=args.tinker_api_key,
                cache_root=args.api_cache_dir,
            )

    # generic generative model
    base_model, base_tokenizer = load_base_model_and_tokenizer(args.base_model_name)

    # mask filling t5 model
    if not args.baselines_only and not args.random_fills:
        int8_kwargs = {}
        half_kwargs = {}
        if args.int8:
            int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
        elif args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)
        print(f'Loading mask filling model {mask_filling_model_name}...')
        local_mask_model_dir = None
        if offline_mode_enabled():
            local_mask_model_dir = build_merged_local_snapshot(
                mask_filling_model_name,
                ["config.json", "model.safetensors"],
            )
        mask_model_source = local_mask_model_dir or mask_filling_model_name
        mask_model_load_kwargs = dict(
            use_safetensors=True,
            local_files_only=offline_mode_enabled(),
            **int8_kwargs,
            **half_kwargs,
            cache_dir=cache_dir,
        )
        try:
            mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                mask_model_source,
                **mask_model_load_kwargs,
            )
        except Exception:
            with temporarily_disable_hf_offline():
                mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                    mask_filling_model_name,
                    use_safetensors=True,
                    **int8_kwargs,
                    **half_kwargs,
                    cache_dir=cache_dir,
                )
        try:
            n_positions = mask_model.config.n_positions
        except AttributeError:
            n_positions = 512
    else:
        n_positions = 512
    preproc_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=512, cache_dir=cache_dir)
    mask_tokenizer_kwargs = dict(
        model_max_length=n_positions,
        cache_dir=cache_dir,
        local_files_only=offline_mode_enabled(),
    )
    if "t5" in mask_filling_model_name.lower():
        mask_tokenizer_kwargs["use_fast"] = False
    try:
        mask_tokenizer = transformers.AutoTokenizer.from_pretrained(
            mask_filling_model_name,
            **mask_tokenizer_kwargs,
        )
    except Exception:
        with temporarily_disable_hf_offline():
            retry_kwargs = dict(model_max_length=n_positions, cache_dir=cache_dir)
            if "t5" in mask_filling_model_name.lower():
                retry_kwargs["use_fast"] = False
            mask_tokenizer = transformers.AutoTokenizer.from_pretrained(
                mask_filling_model_name,
                **retry_kwargs,
            )
    if args.dataset in ['english', 'german']:
        preproc_tokenizer = mask_tokenizer

    load_base_model()

    print(f'Loading dataset {args.dataset}...')
    data = generate_data(args.dataset, args.dataset_key)
    if args.random_fills:
        FILL_DICTIONARY = set()
        for texts in data.values():
            for text in texts:
                FILL_DICTIONARY.update(text.split())
        FILL_DICTIONARY = sorted(list(FILL_DICTIONARY))

    if args.scoring_model_name and args.tinker_model is None:
        print(f'Loading SCORING model {args.scoring_model_name}...')
        del base_model
        del base_tokenizer
        torch.cuda.empty_cache()
        base_model, base_tokenizer = load_base_model_and_tokenizer(args.scoring_model_name)
        load_base_model()  # Load again because we've deleted/replaced the old model

    # write the data to a json file in the save folder
    with open(os.path.join(SAVE_FOLDER, "raw_data.json"), "w") as f:
        print(f"Writing raw data to {os.path.join(SAVE_FOLDER, 'raw_data.json')}")
        json.dump(data, f)

    if not args.skip_baselines:
        baseline_outputs = [run_baseline_threshold_experiment(get_ll, "likelihood", n_samples=n_samples)]
        if using_local_transformers_backend():
            rank_criterion = lambda text: -get_rank(text, log=False)
            baseline_outputs.append(run_baseline_threshold_experiment(rank_criterion, "rank", n_samples=n_samples))
            logrank_criterion = lambda text: -get_rank(text, log=True)
            baseline_outputs.append(run_baseline_threshold_experiment(logrank_criterion, "log_rank", n_samples=n_samples))
            entropy_criterion = lambda text: get_entropy(text)
            baseline_outputs.append(run_baseline_threshold_experiment(entropy_criterion, "entropy", n_samples=n_samples))

        baseline_outputs.append(eval_supervised(data, model='roberta-base-openai-detector'))
        baseline_outputs.append(eval_supervised(data, model='roberta-large-openai-detector'))

    outputs = []

    if not args.baselines_only:
        # run perturbation experiments
        for n_perturbations in n_perturbation_list:
            perturbation_results = get_perturbation_results(args.span_length, n_perturbations, n_samples)
            for perturbation_mode in ['d', 'z']:
                output = run_perturbation_experiment(
                    perturbation_results, perturbation_mode, span_length=args.span_length, n_perturbations=n_perturbations, n_samples=n_samples)
                outputs.append(output)
                with open(os.path.join(SAVE_FOLDER, f"perturbation_{n_perturbations}_{perturbation_mode}_results.json"), "w") as f:
                    json.dump(output, f)

    if not args.skip_baselines:
        # write likelihood threshold results to a file
        with open(os.path.join(SAVE_FOLDER, f"likelihood_threshold_results.json"), "w") as f:
            json.dump(baseline_outputs[0], f)

        if using_local_transformers_backend():
            # write rank threshold results to a file
            with open(os.path.join(SAVE_FOLDER, f"rank_threshold_results.json"), "w") as f:
                json.dump(baseline_outputs[1], f)

            # write log rank threshold results to a file
            with open(os.path.join(SAVE_FOLDER, f"logrank_threshold_results.json"), "w") as f:
                json.dump(baseline_outputs[2], f)

            # write entropy threshold results to a file
            with open(os.path.join(SAVE_FOLDER, f"entropy_threshold_results.json"), "w") as f:
                json.dump(baseline_outputs[3], f)
        
        # write supervised results to a file
        with open(os.path.join(SAVE_FOLDER, f"roberta-base-openai-detector_results.json"), "w") as f:
            json.dump(baseline_outputs[-2], f)
        
        # write supervised results to a file
        with open(os.path.join(SAVE_FOLDER, f"roberta-large-openai-detector_results.json"), "w") as f:
            json.dump(baseline_outputs[-1], f)

        outputs += baseline_outputs

    save_roc_curves(outputs)
    save_ll_histograms(outputs)
    save_llr_histograms(outputs)

    # move results folder from tmp_results/ to results/, making sure necessary directories exist
    new_folder = SAVE_FOLDER.replace("tmp_results", "results")
    if not os.path.exists(os.path.dirname(new_folder)):
        os.makedirs(os.path.dirname(new_folder))
    os.rename(SAVE_FOLDER, new_folder)
    with open(os.path.join(new_folder, "run_cache_meta.json"), "w") as f:
        json.dump({"signature": run_signature, "completed": True}, f, indent=2)

    print(f"Used an *estimated* {API_TOKEN_COUNTER} API tokens (may be inaccurate)")
