import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import transformers


def _stable_key(payload):
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DiskCache:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key):
        return self.cache_dir / f"{key}.json"

    def get(self, payload):
        key = _stable_key(payload)
        path = self._path_for_key(key)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def set(self, payload, value):
        key = _stable_key(payload)
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            suffix=".tmp",
            dir=str(path.parent),
            prefix=f"{path.stem}.",
        )
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False)
        try:
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TinkerSamplingBackend:
    def __init__(self, model_name, cache_dir=None, tokenizer_name=None, api_key=None, cache_root=None):
        if api_key:
            os.environ["TINKER_API_KEY"] = api_key
        if not os.environ.get("TINKER_API_KEY"):
            raise ValueError("TINKER_API_KEY must be set or provided via --tinker_api_key")

        import tinker
        from tinker import types

        self.tinker = tinker
        self.types = types
        self.model_name = model_name
        self.service_client = tinker.ServiceClient()
        self.sampling_client = self.service_client.create_sampling_client(base_model=model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name or model_name,
            cache_dir=cache_dir,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.sample_cache = None
        self.logprob_cache = None
        if cache_root:
            backend_root = Path(cache_root) / "tinker" / model_name.replace("/", "__")
            self.sample_cache = DiskCache(backend_root / "samples")
            self.logprob_cache = DiskCache(backend_root / "logprobs")

    def _model_input(self, text):
        tokens = self.tokenizer.encode(text)
        return self.types.ModelInput.from_ints(tokens=tokens)

    def _run_async(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        return asyncio.run(coro)

    def sample(self, prompt, max_tokens=200, temperature=1.0, top_p=None, top_k=None, num_samples=1):
        cache_payload = {
            "kind": "sample",
            "model_name": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_samples": num_samples,
        }
        if self.sample_cache is not None:
            cached = self.sample_cache.get(cache_payload)
            if cached is not None:
                return cached["sequences"]

        params = dict(max_tokens=max_tokens, temperature=temperature)
        if top_p is not None:
            params["top_p"] = top_p
        if top_k is not None:
            params["top_k"] = top_k

        async def _sample():
            result = await self.sampling_client.sample_async(
                prompt=self._model_input(prompt),
                num_samples=num_samples,
                sampling_params=self.types.SamplingParams(**params),
            )
            return [
                self.tokenizer.decode(sequence.tokens, skip_special_tokens=True).strip()
                for sequence in result.sequences
            ]

        sequences = self._run_async(_sample())
        if self.sample_cache is not None:
            self.sample_cache.set(cache_payload, {"sequences": sequences})
        return sequences

    def _normalize_logprobs(self, result):
        if isinstance(result, list):
            values = result
        elif isinstance(result, dict):
            values = result.get("logprobs")
        else:
            values = getattr(result, "logprobs", None)

        if values is None:
            raise ValueError("Unsupported Tinker logprob response format")

        normalized = [value for value in values if value is not None]
        return normalized

    def get_mean_logprob(self, text):
        if not text.strip():
            return -100.0
        cache_payload = {
            "kind": "mean_logprob",
            "model_name": self.model_name,
            "text": text,
        }
        if self.logprob_cache is not None:
            cached = self.logprob_cache.get(cache_payload)
            if cached is not None:
                return float(cached["mean_logprob"])

        async def _compute():
            return await self.sampling_client.compute_logprobs_async(self._model_input(text))

        result = self._run_async(_compute())
        normalized = self._normalize_logprobs(result)
        mean_logprob = -100.0 if not normalized else float(np.mean(normalized))
        if self.logprob_cache is not None:
            self.logprob_cache.set(cache_payload, {"mean_logprob": mean_logprob})
        return mean_logprob
