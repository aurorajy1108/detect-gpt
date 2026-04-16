import random
import json
import datasets
from huggingface_hub import hf_hub_download

SEPARATOR = '<<<SEP>>>'


DATASETS = ['writing', 'english', 'german', 'pubmed']
PAIR_DATASETS = ['hc3', 'hc3_zh']


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


def _first_nonempty(value):
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _first_nonempty(item)
            if text:
                return text
    return None


def _infer_prompt(example):
    for key in ['question', 'prompt', 'instruction', 'input', 'query']:
        if key in example:
            text = _first_nonempty(example[key])
            if text:
                return text
    return None


def _infer_human_answer(example):
    for key in ['human_answers', 'human_answer', 'human', 'reference', 'references', 'answer_human']:
        if key in example:
            text = _first_nonempty(example[key])
            if text:
                return text
    return None


def _infer_model_answer(example):
    for key in ['chatgpt_answers', 'chatgpt_answer', 'chatgpt', 'model_answers', 'ai_answers', 'answer_chatgpt']:
        if key in example:
            text = _first_nonempty(example[key])
            if text:
                return text
    return None


def load_hc3_records(cache_dir, language='en', split='train', source=None):
    repo_id = 'Hello-SimpleAI/HC3-Chinese' if language == 'zh' else 'Hello-SimpleAI/HC3'
    filename = f"{source or 'all'}.jsonl"
    file_path = hf_hub_download(
        repo_id=repo_id,
        repo_type='dataset',
        filename=filename,
        cache_dir=cache_dir,
    )

    records = []
    with open(file_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            example = json.loads(line)
            original = _infer_human_answer(example)
            sampled = _infer_model_answer(example)
            prompt = _infer_prompt(example)
            if not original:
                continue
            records.append({
                'prompt': prompt,
                'original': original,
                'sampled': sampled,
                'source': source or 'all',
            })

    return records


def load_records(name, cache_dir, **kwargs):
    if name in DATASETS:
        return [{'original': text} for text in load(name, cache_dir=cache_dir, **kwargs)]
    if name == 'hc3':
        return load_hc3_records(cache_dir=cache_dir, language='en', **kwargs)
    if name == 'hc3_zh':
        return load_hc3_records(cache_dir=cache_dir, language='zh', **kwargs)
    raise ValueError(f'Unknown dataset {name}')


def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        load_fn = globals()[f'load_{name}']
        return load_fn(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {name}')
