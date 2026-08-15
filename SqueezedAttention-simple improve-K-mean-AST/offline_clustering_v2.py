"""
offline_clustering_v2.py
========================
Extension của offline_clustering.py gốc, tích hợp:
  - Hướng 1: Layer-wise Adaptive Budget (entropy-based)
  - Hướng 2: AST-aware initialization cho code dataset

Đặt file này ở root của repo SqueezedAttention (cùng cấp với offline_clustering.py).

Usage examples (chi tiết trong README_EXTENSIONS.md):

  # Baseline (Squeezed Attention gốc):
  python offline_clustering_v2.py llama2-7b-32k --dataset trec --percent_clusters 5

  # Hướng 1: adaptive entropy-based budget
  python offline_clustering_v2.py llama2-7b-32k --dataset trec \\
      --percent_clusters 5 --adaptive_budget --budget_strategy linear

  # Hướng 2: code-aware (chỉ cho dataset code)
  python offline_clustering_v2.py llama2-7b-32k --dataset lcc \\
      --percent_clusters 5 --code_aware --code_language python

  # Combo:
  python offline_clustering_v2.py llama2-7b-32k --dataset repobench-p \\
      --percent_clusters 5 --adaptive_budget --code_aware
"""
import time
import os
import torch
import torch.nn as nn
import argparse
from utils.modelutils import *
from utils.datautils import *
from utils.model_parse import parse_model, get_layers
from tqdm import tqdm
import pickle
import numpy as np
import math
import sys
import textwrap
import shutil
import json

from squeezedattention.clustering import run_clustering, run_global_threshold
from squeezedattention.utils import build_chat, truncate_fn
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig

# Import extensions
from adaptive_budget import (
    compute_attention_entropy,
    profile_layer_entropies,
    allocate_budget_by_entropy,
    compute_total_budget,
    print_budget_summary,
)
from ast_clustering import (
    parse_code_to_scopes,
    map_tokens_to_scopes,
    compute_scope_centroids,
    compute_token_type_weights,
)


# ---------- Helper: chuẩn bị danh sách cluster size per-layer ----------

def get_per_layer_budgets(
    args,
    all_queries_layers,
    all_keys_layers,
    shared_prefix_length: int,
):
    """
    Trả về list[L] - số cluster cho mỗi layer.
    Nếu --adaptive_budget: dùng entropy-based; nếu không: uniform (gốc).
    """
    num_layers = len(all_keys_layers)
    percent = args.percent_clusters

    if not args.adaptive_budget:
        # Hành vi gốc: mỗi layer cùng K
        per_layer = max(1, int((percent / 100.0) * (shared_prefix_length - args.observation_window)))
        budgets = torch.full((num_layers,), per_layer, dtype=torch.long)
        return budgets, None  # None = không cần profiling

    # === Hướng 1: Entropy-based allocation ===
    print("\n[Hướng 1] Profiling layer entropies...")
    entropies = profile_layer_entropies(
        all_queries_layers,
        all_keys_layers,
        observation_window=args.observation_window,
    )

    total = compute_total_budget(
        shared_prefix_length,
        args.observation_window,
        percent,
        num_layers,
    )

    budgets = allocate_budget_by_entropy(
        entropies,
        total_budget=total,
        min_budget_per_layer=args.min_budget,
        strategy=args.budget_strategy,
    )

    print_budget_summary(entropies, budgets)
    return budgets, entropies


# ---------- Helper: chạy clustering với per-layer budget khác nhau ----------

def run_clustering_per_layer(
    all_keys_layers,
    budgets: torch.Tensor,
    observation_window: int,
    device,
    code_aware_info: dict = None,
):
    """
    Wrapper cho run_clustering nhưng cho phép K khác nhau mỗi layer.

    Squeezed Attention gốc gọi run_clustering với 1 K cho tất cả layer;
    ta cần override để mỗi layer có K riêng.

    Args:
        code_aware_info: dict với keys 'scope_ids', 'weights' nếu dùng Hướng 2.
                         None nếu standard mode.
    """
    centroids_tensor_dict = {}
    centroids_labels_dict = {}

    for layer_idx, keys in enumerate(all_keys_layers):
        k = int(budgets[layer_idx].item())

        if code_aware_info is not None:
            # Hướng 2: dùng scope-based init
            scope_ids = code_aware_info["scope_ids"]
            weights = code_aware_info["weights"]

            # Take only the fixed-context portion
            # Note: shape của keys thường là [B, H, S, D] hoặc [H, S, D]
            if keys.dim() == 4:
                keys_layer = keys[0]  # [H, S, D]
            else:
                keys_layer = keys

            ctx_keys = keys_layer[:, :-observation_window, :]  # exclude observation
            ctx_scope_ids = scope_ids[:ctx_keys.shape[1]]

            # Init centroids từ scope
            init_centroids = compute_scope_centroids(ctx_keys, ctx_scope_ids, k)

            # Run weighted K-means (refine)
            from ast_clustering import weighted_kmeans
            ctx_weights = weights[:ctx_keys.shape[1]]
            centroids, labels = weighted_kmeans(
                ctx_keys, ctx_weights, k,
                initial_centroids=init_centroids,
                num_iter=10, device=device,
            )

            centroids_tensor_dict[layer_idx] = centroids
            centroids_labels_dict[layer_idx] = labels
        else:
            # Standard: gọi run_clustering gốc cho từng layer
            # (Phải gọi từng layer vì K khác nhau)
            single_layer_keys = [keys]  # wrap to list
            ct, cl = run_clustering(
                single_layer_keys,
                k,
                observation_window=observation_window,
                device=device,
            )
            # ct, cl là dict {0: tensor}; rename key thành layer_idx
            centroids_tensor_dict[layer_idx] = ct[0]
            centroids_labels_dict[layer_idx] = cl[0]

    return centroids_tensor_dict, centroids_labels_dict


# ===================================================================
# MAIN
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="llama model to load")
    parser.add_argument('--output_path', type=str, default='output/')
    parser.add_argument(
        '--dataset', type=str, default='trec',
        choices=["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
                 "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
                 "samsum", "lcc", "repobench-p"],
    )
    parser.add_argument("--hierarchical_lookup", action="store_true")
    parser.add_argument("--percent_clusters", type=int, default=-1)
    parser.add_argument("--percent_clusters_l2", type=int, default=-1)
    parser.add_argument('--observation_window', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)

    # === Hướng 1 args ===
    parser.add_argument("--adaptive_budget", action="store_true",
                        help="Enable entropy-based layer-wise budget allocation")
    parser.add_argument("--budget_strategy", type=str, default="linear",
                        choices=["linear", "softmax", "pyramid", "inverse", "uniform"])
    parser.add_argument("--min_budget", type=int, default=2,
                        help="Min cluster count per layer")

    # === Hướng 2 args ===
    parser.add_argument("--code_aware", action="store_true",
                        help="Enable AST-aware clustering (chỉ cho dataset code)")
    parser.add_argument("--code_language", type=str, default="python",
                        choices=["python", "javascript", "java", "cpp", "go"])

    # === Log/debug ===
    parser.add_argument("--save_entropy_log", action="store_true",
                        help="Save layer entropies cho phân tích sau")

    args = parser.parse_args()

    DEV = torch.device(f"cuda:{args.device}")

    # Tag tên output để phân biệt experiment
    tag_parts = []
    if args.adaptive_budget:
        tag_parts.append(f"adaptive-{args.budget_strategy}")
    if args.code_aware:
        tag_parts.append(f"codeaware-{args.code_language}")
    tag = "_".join(tag_parts) if tag_parts else "baseline"
    args.output_path = os.path.join(args.output_path, tag)

    print(f"\n>>> Experiment tag: {tag}")
    print(f">>> Output path: {args.output_path}\n")

    # ===== Load model =====
    model2path = json.load(open("LongBench/config/model2path.json", "r"))
    model2maxlen = json.load(open("LongBench/config/model2maxlen.json", "r"))
    model_path = model2path[args.model]
    max_length = model2maxlen[args.model]

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = LlamaConfig.from_pretrained(model_path)
    config.return_qkv_states = True
    config._flash_attn_2_enabled = True
    config._attn_implementation = "flash_attention_2"
    model = LlamaForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16)
    model.eval()
    model = model.to(DEV)

    model_type = parse_model(model)
    layers = get_layers(model, model_type)

    # ===== Load dataset =====
    from datasets import load_dataset
    dataset = args.dataset
    dataset_name_prompt = dataset + '_prompt'
    data = load_dataset('THUDM/LongBench', dataset, split='test')

    dataset2prompt = json.load(open("LongBench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("LongBench/config/dataset2maxlen.json", "r"))
    prompt_format = dataset2prompt[dataset]
    prompt_only_format = dataset2prompt[dataset_name_prompt]
    data_all = [data_sample for data_sample in data]

    # ===== Code-aware check =====
    is_code_dataset = dataset in ["lcc", "repobench-p"]
    if args.code_aware and not is_code_dataset:
        print(f"[WARN] --code_aware bật nhưng dataset '{dataset}' không phải code dataset.")
        print(f"       Code datasets: lcc, repobench-p. Tắt code-aware cho an toàn.")
        args.code_aware = False

    # ===== Shared prefix length pre-computation =====
    shared_prefix_length = {}
    for i in range(len(data_all)):
        prompt = prompt_format.format(**data_all[i])
        prompt_only = prompt_only_format.format(**data_all[i])
        prompt, tspl = truncate_fn(prompt, prompt_only, tokenizer, max_length, dataset, DEV)
        shared_prefix_length[i] = tspl
        assert tspl > 0

    # ===== Hooks =====
    all_queries_layers = []
    all_keys_layers = []
    all_values_layers = []

    def get_attention_scores(module, inp, out):
        _, qkv, _ = out
        queries, keys, values = qkv
        sp_len = shared_prefix_length[dataidx]
        queries = queries[:, :, :sp_len]
        keys = keys[:, :, :sp_len]
        values = values[:, :, :sp_len]
        all_queries_layers.append(queries)
        all_keys_layers.append(keys)
        all_values_layers.append(values)

    for layer in layers:
        layer.self_attn.register_forward_hook(get_attention_scores)

    # Log entropy across dataset
    all_entropies_log = []

    # ===== Main loop =====
    for dataidx, d in enumerate(tqdm(data)):
        all_queries_layers.clear()
        all_keys_layers.clear()
        all_values_layers.clear()

        prompt = prompt_format.format(**d)
        prompt_only = prompt_only_format.format(**d)
        prompt, _ = truncate_fn(prompt, prompt_only, tokenizer, max_length, dataset, DEV)
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.to(DEV)

        print(f"dataidx: {dataidx} | input len: {len(input_ids[0])} | sp_len: {shared_prefix_length[dataidx]}")

        with torch.no_grad():
            model.generate(input_ids, do_sample=True, max_new_tokens=1,
                           use_cache=False, output_attentions=True)

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        sp_len = shared_prefix_length[dataidx]

        # ===== Hướng 1: per-layer budget =====
        budgets, entropies = get_per_layer_budgets(
            args, all_queries_layers, all_keys_layers, sp_len,
        )

        if args.save_entropy_log and entropies is not None:
            all_entropies_log.append(entropies.cpu().numpy())

        # ===== Hướng 2: code-aware info (nếu enable) =====
        code_aware_info = None
        if args.code_aware:
            # Lấy raw code từ data sample. Field tùy dataset:
            # lcc và repobench-p có field 'context' chứa code.
            code = d.get("context", "") or d.get("input", "")
            if not code:
                print(f"[WARN] dataidx {dataidx}: không tìm thấy code, skip code-aware")
            else:
                scopes = parse_code_to_scopes(code, args.code_language)
                scope_ids = map_tokens_to_scopes(code, tokenizer, scopes)
                weights = compute_token_type_weights(code, tokenizer)
                # Pad scope_ids và weights để match sp_len (do code chỉ là 1 phần prompt)
                # Trong thực tế, scope_ids dài bằng số token của code; ta pad zero ở đầu
                # cho phần prompt prefix (instruction)
                pad_len = sp_len - len(scope_ids)
                if pad_len > 0:
                    scope_ids = torch.cat([torch.full((pad_len,), -1, dtype=torch.long), scope_ids])
                    weights = torch.cat([torch.ones(pad_len), weights])
                elif pad_len < 0:
                    scope_ids = scope_ids[-sp_len:]
                    weights = weights[-sp_len:]

                code_aware_info = {"scope_ids": scope_ids, "weights": weights}
                print(f"[Hướng 2] Found {len(scopes)} scopes, {(scope_ids >= 0).sum().item()} tokens in scope")

        # ===== Run clustering =====
        if args.hierarchical_lookup:
            # Hierarchical: L2 trước, L1 sau (giữ logic gốc nhưng dùng adaptive K)
            # Note: hierarchical với adaptive còn cần thêm logic phức tạp.
            # Bản v0 này dùng adaptive cho L2, uniform percent_l2 cho L1.
            # TODO: extend hierarchical fully adaptive sau khi v0 stable.
            budgets_l2 = budgets
            per_layer_l1 = max(1, int((args.percent_clusters_l2 / 100.0) * (sp_len - args.observation_window)))
            budgets_l1 = torch.full((len(all_keys_layers),), per_layer_l1, dtype=torch.long)

            print("[Hierarchical] L2 budgets: ", budgets_l2.tolist())
            print("[Hierarchical] L1 budgets: ", budgets_l1.tolist())

            centroids_tensor_dict_l2, centroids_labels_dict_l2 = run_clustering_per_layer(
                all_keys_layers, budgets_l2, args.observation_window, DEV, code_aware_info,
            )
            # L1 cluster trên L2 centroids
            l2_centroids_list = [centroids_tensor_dict_l2[i].unsqueeze(0) for i in range(len(all_keys_layers))]
            centroids_tensor_dict_l1, centroids_labels_dict_l1 = run_clustering_per_layer(
                l2_centroids_list, budgets_l1, 0, DEV, None,  # L1 luôn không code-aware
            )

            num_lyrs = len(all_keys_layers)
            for i in range(num_lyrs):
                label_dict_l1 = centroids_labels_dict_l1[i]
                label_dict_l2 = centroids_labels_dict_l2[i]
                gathered_tensor = torch.gather(label_dict_l1, -1, label_dict_l2)
                centroids_labels_dict_l1[i] = gathered_tensor

            global_threshold_dict_l1 = run_global_threshold(
                all_keys_layers, all_queries_layers, centroids_tensor_dict_l1,
                centroids_labels_dict_l1, int(budgets_l1.float().mean().item()),
                observation_window=args.observation_window, device=DEV,
            )
            global_threshold_dict_l2 = run_global_threshold(
                all_keys_layers, all_queries_layers, centroids_tensor_dict_l2,
                centroids_labels_dict_l2, int(budgets_l2.float().mean().item()),
                observation_window=args.observation_window, device=DEV,
            )

            # Save - dùng tag để phân biệt
            os.makedirs(args.output_path, exist_ok=True)
            avg_l1 = int(budgets_l1.float().mean().item())
            avg_l2 = int(budgets_l2.float().mean().item())
            for k in centroids_tensor_dict_l1:
                centroids_tensor_dict_l1[k] = centroids_tensor_dict_l1[k].cpu()
                centroids_labels_dict_l1[k] = centroids_labels_dict_l1[k].cpu()
                centroids_tensor_dict_l2[k] = centroids_tensor_dict_l2[k].cpu()
                centroids_labels_dict_l2[k] = centroids_labels_dict_l2[k].cpu()

            # NOTE: tên file phải khớp modeling_llama.py (~L1457): 'hierarchical_centroids_*'
            torch.save(centroids_tensor_dict_l1, f'{args.output_path}/hierarchical_centroids_tensor_dict_L1_{dataidx}_{avg_l1}.pt')
            torch.save(centroids_labels_dict_l1, f'{args.output_path}/hierarchical_centroids_labels_dict_L1_{dataidx}_{avg_l1}.pt')
            torch.save(centroids_tensor_dict_l2, f'{args.output_path}/centroids_tensor_dict_{dataidx}_{avg_l2}.pt')
            torch.save(centroids_labels_dict_l2, f'{args.output_path}/centroids_labels_dict_{dataidx}_{avg_l2}.pt')
            torch.save(global_threshold_dict_l1, f'{args.output_path}/hierarchical_global_threshold_L1_{dataidx}_{avg_l1}.pt')
            torch.save(global_threshold_dict_l2, f'{args.output_path}/global_threshold_{dataidx}_{avg_l2}.pt')
            # Save budgets để online eval biết
            torch.save({"l1": budgets_l1, "l2": budgets_l2}, f'{args.output_path}/budgets_{dataidx}.pt')
        else:
            # Non-hierarchical
            centroids_tensor_dict, centroids_labels_dict = run_clustering_per_layer(
                all_keys_layers, budgets, args.observation_window, DEV, code_aware_info,
            )
            global_threshold_dict = run_global_threshold(
                all_keys_layers, all_queries_layers, centroids_tensor_dict,
                centroids_labels_dict, int(budgets.float().mean().item()),
                observation_window=args.observation_window, device=DEV,
            )

            os.makedirs(args.output_path, exist_ok=True)
            avg_k = int(budgets.float().mean().item())
            for k in centroids_tensor_dict:
                centroids_tensor_dict[k] = centroids_tensor_dict[k].cpu()
                centroids_labels_dict[k] = centroids_labels_dict[k].cpu()
            torch.save(centroids_tensor_dict, f'{args.output_path}/centroids_tensor_dict_{dataidx}_{avg_k}.pt')
            torch.save(centroids_labels_dict, f'{args.output_path}/centroids_labels_dict_{dataidx}_{avg_k}.pt')
            torch.save(global_threshold_dict, f'{args.output_path}/global_threshold_{dataidx}_{avg_k}.pt')
            torch.save(budgets, f'{args.output_path}/budgets_{dataidx}.pt')

        # Free memory
        num_layers = len(all_keys_layers)
        for _ in range(num_layers):
            del all_queries_layers[0]
            del all_keys_layers[0]
            del all_values_layers[0]

    # Save entropy log nếu enable
    if args.save_entropy_log and len(all_entropies_log) > 0:
        log_arr = np.stack(all_entropies_log)
        np.save(os.path.join(args.output_path, "entropy_log.npy"), log_arr)
        print(f"\n>>> Saved entropy log: shape {log_arr.shape}")

    print(f"\n>>> Done. Output in: {args.output_path}")
