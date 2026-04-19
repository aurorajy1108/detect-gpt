import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path


MODEL_PAIRS = [
    {
        "family": "qwen3_8b",
        "base": "Qwen/Qwen3-8B-Base",
        "post_trained": "Qwen/Qwen3-8B",
    },
    {
        "family": "qwen3_30b_a3b",
        "base": "Qwen/Qwen3-30B-A3B-Base",
        "post_trained": "Qwen/Qwen3-30B-A3B",
    },
    {
        "family": "llama3_1_8b",
        "base": "meta-llama/Llama-3.1-8B",
        "post_trained": "meta-llama/Llama-3.1-8B-Instruct",
    },
    {
        "family": "deepseek_v3_1",
        "base": "deepseek-ai/DeepSeek-V3.1-Base",
        "post_trained": "deepseek-ai/DeepSeek-V3.1",
    },
]

DATASET_MATRIX = [
    {
        "dataset": "hc3",
        "language": "en",
        "source": "finance",
        "mask_model": "t5-large",
    },
    {
        "dataset": "hc3_zh",
        "language": "zh",
        "source": "finance",
        "mask_model": "google/mt5-large",
    },
]


def stable_job_id(config):
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def apply_preset(args):
    if args.preset is None:
        return

    if args.preset == "qwen3_8b_formal":
        args.families = ["qwen3_8b"]
        args.languages = ["en", "zh"]
        args.stages = ["base", "post_trained"]
        args.answer_modes = ["dataset_answers", "regenerated_answers"]
        if args.n_samples == parser_defaults["n_samples"]:
            args.n_samples = 200
        if args.batch_size == parser_defaults["batch_size"]:
            args.batch_size = 10
        if args.n_perturbation_list == parser_defaults["n_perturbation_list"]:
            args.n_perturbation_list = "10"
        if args.output_prefix == parser_defaults["output_prefix"]:
            args.output_prefix = "formal_qwen3_8b"
        if args.marker_dir == parser_defaults["marker_dir"]:
            args.marker_dir = "batch_job_markers_qwen3_8b_formal"
        if args.regenerated_min_sample_words == parser_defaults["regenerated_min_sample_words"]:
            args.regenerated_min_sample_words = 30
        if args.max_sample_tries == parser_defaults["max_sample_tries"]:
            args.max_sample_tries = 10
    else:
        raise ValueError(f"Unknown preset {args.preset}")


def build_jobs(args):
    jobs = []
    for pair, dataset_cfg, answer_mode in itertools.product(
        MODEL_PAIRS,
        DATASET_MATRIX,
        ["dataset_answers", "regenerated_answers"],
    ):
        if args.languages and dataset_cfg["language"] not in args.languages:
            continue
        if args.answer_modes and answer_mode not in args.answer_modes:
            continue
        if args.families and pair["family"] not in args.families:
            continue

        for stage_name in ["base", "post_trained"]:
            if args.stages and stage_name not in args.stages:
                continue

            model_name = pair[stage_name]
            manifest_path = None
            if dataset_cfg["language"] == "en":
                manifest_path = args.dataset_manifest_en
            elif dataset_cfg["language"] == "zh":
                manifest_path = args.dataset_manifest_zh
            config = {
                "family": pair["family"],
                "stage": stage_name,
                "model_name": model_name,
                "dataset": dataset_cfg["dataset"],
                "language": dataset_cfg["language"],
                "source": Path(manifest_path).stem if manifest_path else dataset_cfg["source"],
                "mask_model": dataset_cfg["mask_model"],
                "answer_mode": answer_mode,
                "n_samples": args.n_samples,
                "batch_size": args.batch_size,
                "pct_words_masked": args.pct_words_masked,
                "span_length": args.span_length,
                "n_perturbation_list": args.n_perturbation_list,
                "n_perturbation_rounds": args.n_perturbation_rounds,
                "top_p": args.top_p,
                "dataset_manifest": manifest_path,
            }
            config["job_id"] = stable_job_id(config)
            jobs.append(config)
    return jobs


def build_command(job, args):
    output_name = f"{args.output_prefix}/{job['job_id']}_{job['family']}_{job['stage']}_{job['language']}_{job['answer_mode']}"
    command = [
        sys.executable,
        "run.py",
        "--dataset",
        job["dataset"],
        "--dataset_split",
        args.dataset_split,
        "--dataset_source",
        job["source"],
        "--n_samples",
        str(job["n_samples"]),
        "--batch_size",
        str(job["batch_size"]),
        "--pct_words_masked",
        str(job["pct_words_masked"]),
        "--span_length",
        str(job["span_length"]),
        "--n_perturbation_list",
        job["n_perturbation_list"],
        "--n_perturbation_rounds",
        str(job["n_perturbation_rounds"]),
        "--mask_filling_model_name",
        job["mask_model"],
        "--tinker_model",
        job["model_name"],
        "--base_model_name",
        job["model_name"],
        "--api_cache_dir",
        args.api_cache_dir,
        "--output_name",
        output_name,
        "--cache_dir",
        args.cache_dir,
        "--min_sample_words",
        str(args.min_sample_words),
        "--max_sample_tries",
        str(args.max_sample_tries),
    ]
    manifest_path = job.get("dataset_manifest")
    if manifest_path:
        command.extend(["--dataset_manifest", manifest_path])
    if args.do_top_p:
        command.extend(["--do_top_p", "--top_p", str(job["top_p"])])
    if job["answer_mode"] == "regenerated_answers" and args.regenerated_min_sample_words is not None:
        command.extend(["--min_sample_words", str(args.regenerated_min_sample_words)])
    if job["answer_mode"] == "dataset_answers":
        command.append("--use_dataset_samples")
    if args.skip_baselines:
        command.append("--skip_baselines")
    if args.baselines_only:
        command.append("--baselines_only")
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--pct_words_masked", type=float, default=0.3)
    parser.add_argument("--span_length", type=int, default=2)
    parser.add_argument("--n_perturbation_list", type=str, default="5")
    parser.add_argument("--n_perturbation_rounds", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.96)
    parser.add_argument("--do_top_p", action="store_true")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_manifest_en", type=str, default=None)
    parser.add_argument("--dataset_manifest_zh", type=str, default=None)
    parser.add_argument("--api_cache_dir", type=str, default="api_cache")
    parser.add_argument("--cache_dir", type=str, default="~/.cache")
    parser.add_argument("--min_sample_words", type=int, default=55)
    parser.add_argument("--regenerated_min_sample_words", type=int, default=30)
    parser.add_argument("--max_sample_tries", type=int, default=10)
    parser.add_argument("--output_prefix", type=str, default="modern_llm_batch")
    parser.add_argument("--marker_dir", type=str, default="batch_job_markers")
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--languages", nargs="*", choices=["en", "zh"], default=None)
    parser.add_argument(
        "--stages",
        nargs="*",
        choices=["base", "post_trained"],
        default=None,
    )
    parser.add_argument(
        "--answer_modes",
        nargs="*",
        choices=["dataset_answers", "regenerated_answers"],
        default=None,
    )
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--baselines_only", action="store_true")
    parser.add_argument("--print_only", action="store_true")
    args = parser.parse_args()

    global parser_defaults
    parser_defaults = {
        "n_samples": parser.get_default("n_samples"),
        "batch_size": parser.get_default("batch_size"),
        "n_perturbation_list": parser.get_default("n_perturbation_list"),
        "output_prefix": parser.get_default("output_prefix"),
        "marker_dir": parser.get_default("marker_dir"),
        "regenerated_min_sample_words": parser.get_default("regenerated_min_sample_words"),
        "max_sample_tries": parser.get_default("max_sample_tries"),
    }
    apply_preset(args)

    marker_dir = Path(args.marker_dir)
    marker_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args)
    print(f"Prepared {len(jobs)} jobs.")

    failures = []
    for index, job in enumerate(jobs, start=1):
        marker_path = marker_dir / f"{job['job_id']}.done.json"
        if marker_path.exists():
            print(f"[{index}/{len(jobs)}] Skipping completed job {job['job_id']}")
            continue

        command = build_command(job, args)
        printable = " ".join(shlex_quote(part) for part in command)
        print(f"[{index}/{len(jobs)}] {job['job_id']} -> {job['family']} {job['stage']} {job['language']} {job['answer_mode']}")
        print(printable)
        if args.print_only:
            continue

        result = subprocess.run(command, cwd=Path(__file__).resolve().parent)
        if result.returncode != 0:
            failures.append(job)
            print(f"Job failed with exit code {result.returncode}: {job['job_id']}")
            continue

        with open(marker_path, "w", encoding="utf-8") as handle:
            json.dump(job, handle, indent=2, ensure_ascii=False)

    if failures:
        print(f"{len(failures)} jobs failed.")
        sys.exit(1)
    print("All requested jobs completed.")


def shlex_quote(text):
    if not text or any(char.isspace() or char in "\"'\\$" for char in text):
        return json.dumps(text)
    return text


if __name__ == "__main__":
    main()
