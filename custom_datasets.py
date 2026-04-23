import json
import random
import datasets

SEPARATOR = '<<<SEP>>>'


DATASETS = ['writing', 'english', 'german', 'pubmed', 'hc3']


def load_pubmed(cache_dir):
    data = datasets.load_dataset('pubmed_qa', 'pqa_labeled', split='train', cache_dir=cache_dir)
    
    # combine question and long_answer
    data = [f'Question: {q} Answer:{SEPARATOR}{a}' for q, a in zip(data['question'], data['long_answer'])]

    return data


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def load_writing(cache_dir=None):
    writing_path = 'data/writingPrompts'
    
    with open(f'{writing_path}/valid.wp_source', 'r') as f:
        prompts = f.readlines()
    with open(f'{writing_path}/valid.wp_target', 'r') as f:
        stories = f.readlines()
    
    prompts = [process_prompt(prompt) for prompt in prompts]
    joined = [process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]
    filtered = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]

    random.seed(0)
    random.shuffle(filtered)

    return filtered


def load_language(language, cache_dir):
    # load either the english or german portion of the wmt16 dataset
    assert language in ['en', 'de']
    d = datasets.load_dataset('wmt16', 'de-en', split='train', cache_dir=cache_dir)
    docs = d['translation']
    desired_language_docs = [d[language] for d in docs]
    lens = [len(d.split()) for d in desired_language_docs]
    sub = [d for d, l in zip(desired_language_docs, lens) if l > 100 and l < 150]
    return sub


def load_german(cache_dir):
    return load_language('de', cache_dir)


def load_english(cache_dir):
    return load_language('en', cache_dir)


def load_hc3(cache_dir, **kwargs):
    # Returns flat list of human texts (for compatibility with load() interface).
    # Use load_hc3_paired() when you need the ChatGPT answers too.
    pairs = load_hc3_paired(cache_dir, n_samples=None)
    return pairs["original"]


def load_hc3_paired(cache_dir, n_samples=None, min_words=55):
    """
    Load HC3 and return already-paired {"original": [...], "sampled": [...]}
    where original = human answer, sampled = ChatGPT answer.
    Skips pairs where either side is too short.
    """
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id='Hello-SimpleAI/HC3',
        filename='all.jsonl',
        repo_type='dataset',
        cache_dir=cache_dir,
    )
    original, sampled = [], []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            h = ex['human_answers'][0].strip().replace('\n', ' ') if ex['human_answers'] else None
            c = ex['chatgpt_answers'][0].strip().replace('\n', ' ') if ex['chatgpt_answers'] else None
            if not h or not c:
                continue
            if len(h.split()) < min_words or len(c.split()) < min_words:
                continue
            # trim each pair to same word count
            shorter = min(len(h.split()), len(c.split()))
            original.append(' '.join(h.split()[:shorter]))
            sampled.append(' '.join(c.split()[:shorter]))

    if n_samples is not None:
        original = original[:n_samples]
        sampled  = sampled[:n_samples]

    return {"original": original, "sampled": sampled}


def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        load_fn = globals()[f'load_{name}']
        return load_fn(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {name}')