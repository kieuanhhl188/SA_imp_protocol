#!/usr/bin/env python
"""Build one protocol-facing Phase 1 manifest from prepared metadata and offsets."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def repo_id(sample, dataset, index):
    for key in ("repo_id", "repository", "repo_name", "repo"):
        value = sample.get(key)
        if value:
            return str(value)
    return f"{dataset}:sample_{index}"


def query_group_id(sample, dataset, index):
    return str(sample.get("group_id", f"{dataset}:sample_{index}"))


def prompt_parts(dataset, sample):
    if dataset == "lcc":
        fixed_context = str(sample.get("context", ""))
        answers = sample.get("answers") or []
        return fixed_context, str(answers[0] if answers else sample.get("input", ""))
    if dataset == "repobench-p":
        fixed_context = str(sample.get("context", "")) + str(sample.get("input", ""))
        answers = sample.get("answers") or []
        return fixed_context, str(answers[0] if answers else sample.get("next_line", ""))
    raise ValueError(f"Unsupported LongBench dataset: {dataset}")


def build_manifest(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from squeezedattention.utils import truncate_fn
    from struct_clustering import assign_token_units, compact_unit_ids, parse_units

    cfg = os.path.join(REPO_ROOT, "LongBench", "config")
    model_paths = json.load(open(os.path.join(cfg, "model2path.json"), encoding="utf-8"))
    max_lengths = json.load(open(os.path.join(cfg, "model2maxlen.json"), encoding="utf-8"))
    prompts = json.load(open(os.path.join(cfg, "dataset2prompt.json"), encoding="utf-8"))
    model_path = model_paths[args.model]
    max_length = max_lengths[args.model]

    data = load_dataset("THUDM/LongBench", args.dataset, split="test")
    meta_dir = os.path.join(args.phase1_dir, args.model)
    meta_path = os.path.join(meta_dir, f"{args.dataset}_meta.jsonl")
    offsets_path = os.path.join(meta_dir, f"{args.dataset}_offsets.npz")
    if not os.path.exists(meta_path) or not os.path.exists(offsets_path):
        raise SystemExit(f"Missing prepared files: {meta_path} and/or {offsets_path}")

    metadata = {int(row["dataidx"]): row for row in load_jsonl(meta_path)}
    offsets = np.load(offsets_path)
    tok_slow = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tok_fast = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    prompt_format = prompts[args.dataset]
    prompt_only_format = prompts[args.dataset + "_prompt"]
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    written = 0
    n = len(data) if args.limit <= 0 else min(args.limit, len(data))
    with open(args.output, "w", encoding="utf-8") as manifest:
        for index in range(n):
            if index not in metadata:
                raise SystemExit(f"Metadata missing dataidx={index}")
            sample = data[index]
            meta = metadata[index]
            prompt_raw = prompt_format.format(**sample)
            prompt_only = prompt_only_format.format(**sample)
            prompt, shared_prefix = truncate_fn(
                prompt_raw, prompt_only, tok_slow, max_length, args.dataset, "cpu",
                model_name=args.model, force_chat=args.force_chat,
            )
            encoded = tok_fast(prompt, truncation=False, return_offsets_mapping=True,
                               add_special_tokens=True)
            token_ids = [int(value) for value in encoded["input_ids"]]
            char_offsets = [[int(a), int(b)] for a, b in encoded["offset_mapping"]]
            saved_char_offsets = np.asarray(offsets[f"offsets_{index}"]).astype(int).tolist()
            if char_offsets != saved_char_offsets:
                raise SystemExit(f"Character offsets changed for dataidx={index}")
            byte_offsets = np.asarray(offsets[f"offsets_bytes_{index}"]).astype(int).tolist()
            if len(byte_offsets) != len(char_offsets):
                raise SystemExit(f"Byte/character offset length mismatch for dataidx={index}")
            n_ctx = int(shared_prefix) - args.observation_window
            code_start = int(meta["code_char_start"])
            code_end = int(meta["code_char_end"])
            code = prompt[code_start:code_end]
            spans, ast_stats = parse_units(code, meta["language"], args.level)
            spans = [[int(start + code_start), int(end + code_start)] for start, end in spans]
            spans.append([0, len(prompt) + 1])
            starts = np.asarray(char_offsets[:max(0, n_ctx)], dtype=np.int64)[:, 0]
            import torch
            raw_units = assign_token_units(torch.from_numpy(starts), spans)
            unit_ids, _ = compact_unit_ids(raw_units)
            fixed_context, user_input = prompt_parts(args.dataset, sample)
            model_context = str(sample.get("context", ""))
            model_query_prefix = str(sample.get("input", ""))
            record = {
                "schema_version": 1,
                "dataset": args.dataset,
                "model": args.model,
                "sample_id": f"{args.dataset}:{index}",
                "repo_id": repo_id(sample, args.dataset, index),
                "query_group_id": query_group_id(sample, args.dataset, index),
                "query_count": 1,
                "query_mode": "single-query",
                "fixed_context": fixed_context,
                "user_input": user_input,
                "model_fixed_context": model_context,
                "model_query_prefix": model_query_prefix,
                "fixed_context_protocol_note": (
                    "context+input; current SA loader clusters context only"
                    if args.dataset == "repobench-p" else "context"
                ),
                "prompt": prompt,
                "language": meta["language"],
                "token_ids": token_ids,
                "offset_mapping": char_offsets,
                "offset_mapping_bytes": byte_offsets,
                "shared_prefix_length": int(shared_prefix),
                "observation_window": args.observation_window,
                "n_ctx": n_ctx,
                "ast_level": args.level,
                "ast_spans": spans,
                "unit_ids": [int(value) for value in unit_ids.tolist()],
                "ast_stats": ast_stats,
                "truncated": bool(meta["truncated"]),
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f">>> Wrote {written}/{len(data)} samples to {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--dataset", choices=["lcc", "repobench-p"], required=True)
    parser.add_argument("--phase1_dir", default=os.environ.get("SQA_PHASE1_DIR", "phase1_data"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--level", default="function")
    parser.add_argument("--observation_window", type=int, default=100)
    parser.add_argument("--force_chat", action="store_true")
    parser.add_argument("--limit", type=int, default=-1,
                        help="only build the first N samples for a smoke test")
    args = parser.parse_args()
    build_manifest(args)


if __name__ == "__main__":
    main()
