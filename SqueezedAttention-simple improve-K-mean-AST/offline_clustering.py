import time
import os
import torch
import torch.nn as nn
import argparse
from utils.modelutils import *
from utils.datautils import *
from utils.model_parse import (
    parse_model,
    get_layers,
)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="llama model to load")

    parser.add_argument(
        '--output_path', type=str, default='output/'
    )

    parser.add_argument(
        '--dataset', type=str, default='trec', choices=["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                                                        "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                                                        "lcc", "repobench-p"]
    )

    parser.add_argument("--hierarchical_lookup", action="store_true")
    parser.add_argument("--force_chat", action="store_true",
                        help="ap chat template cho lcc/repobench-p. MAC DINH TAT "
                             "nen Phase 0 khong doi. Bat khi model chinh la Instruct, "
                             "va phai bat o MOI buoc")
    parser.add_argument("--percent_clusters", type=int, default=-1)
    parser.add_argument("--percent_clusters_l2", type=int, default=-1)
    parser.add_argument('--observation_window', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--limit', type=int, default=-1,
                        help="chi cluster N sample DAU (smoke test); -1 = ca dataset. "
                             "Phai truyen dung N nay cho LongBench/pred.py --limit, "
                             "vi pred.py tra centroid theo different_prefix_index = 0..N-1")

    args = parser.parse_args()
    DEV = torch.device(f"cuda:{args.device}")

    # get maxlen and model path
    model2path = json.load(open("LongBench/config/model2path.json", "r"))
    model2maxlen = json.load(open("LongBench/config/model2maxlen.json", "r"))
    model_path = model2path[args.model]
    max_length = model2maxlen[args.model]

    # load model
    # AutoConfig/AutoModelForCausalLM thay cho Llama* để dùng được cả Qwen2 (Phase 1).
    # LlamaConfig/LlamaForCausalLM vẫn được Auto* chọn đúng cho longchat/LLaMA-2-32K,
    # nên gate Phase 0 không đổi hành vi.
    from transformers import AutoConfig, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = AutoConfig.from_pretrained(model_path)
    config.return_qkv_states = True
    config._flash_attn_2_enabled = True
    config._attn_implementation = "flash_attention_2"

    # Qwen2: sliding window không tương thích Squeezed Attention (SA cần truy cập
    # mọi key của fixed context). Qwen2.5-Coder mặc định đã tắt; ép tắt cho chắc.
    if getattr(config, "use_sliding_window", False):
        print("[WARN] use_sliding_window=True -> ép về False cho Squeezed Attention")
        config.use_sliding_window = False

    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16)
    model.eval()
    model = model.to(DEV)

    print(f"num_attention_heads={config.num_attention_heads} | "
          f"num_key_value_heads={getattr(config, 'num_key_value_heads', config.num_attention_heads)} | "
          f"num_hidden_layers={config.num_hidden_layers}")

    # get model layers
    model_type = parse_model(model)
    layers = get_layers(model, model_type)

    # load longbench dataset
    from datasets import load_dataset
    dataset = args.dataset
    dataset_name_prompt = dataset + '_prompt'
    data = load_dataset('THUDM/LongBench', dataset, split='test')

    # define prompt format
    import json
    dataset2prompt = json.load(open("LongBench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("LongBench/config/dataset2maxlen.json", "r"))

    # load prompt format, and use first example in dataset as fixed context
    prompt_format = dataset2prompt[dataset]
    prompt_only_format = dataset2prompt[dataset_name_prompt]
    data_all = [data_sample for data_sample in data]

    # --limit: cat ca danh sach mau NGAY TU DAY, khong chi trong vong lap chinh.
    # Vong profiling ben duoi tokenize 2 lan/mau bang tokenizer CHAM; voi Qwen2 do la
    # BPE thuan Python nen quet ca 500 mau cho mot smoke test 10 mau la lang phi that su.
    n_total = len(data_all)
    if args.limit > 0 and args.limit < n_total:
        data_all = data_all[:args.limit]
        data = [data[i] for i in range(args.limit)]
        print(f"[limit] chi xu ly {len(data_all)}/{n_total} sample dau")

    # different prefix profiling offline (also need to account for truncation)
    shared_prefix_length = {}
    for i in range(len(data_all)):
        prompt = prompt_format.format(**data_all[i])
        prompt_only = prompt_only_format.format(**data_all[i])

        # perform truncation and get truncated shared prefix length
        prompt, truncated_shared_prefix_length = truncate_fn(prompt, prompt_only, tokenizer, max_length, dataset, DEV,
                                                   model_name=args.model,
                                                   force_chat=args.force_chat)
        shared_prefix_length[i] = truncated_shared_prefix_length
        assert (truncated_shared_prefix_length > 0) # else, truncated part of input context as well

    # add hooks to profile attn scores
    all_queries_layers = []
    all_keys_layers = []
    all_values_layers = []
    def get_attention_scores(module, inp, out):
        _, qkv, _ = out
        queries, keys, values = qkv
        sp_len = shared_prefix_length[dataidx]
        queries = queries[:,:,:sp_len]
        keys = keys[:,:,:sp_len]
        values = values[:,:,:sp_len]
        all_queries_layers.append(queries)
        all_keys_layers.append(keys)
        all_values_layers.append(values)

    # Attach the hook to each attention layer
    for layer in layers:
        layer.self_attn.register_forward_hook(get_attention_scores)

    # load dataset format
    for dataidx, d in enumerate(tqdm(data)):
        # Bỏ qua mẫu đã có kết quả -> chạy tiếp được sau khi job đứt giữa chừng.
        # Không đổi kết quả: chỉ tránh tính lại thứ đã nằm trên đĩa. Cần thiết vì clustering
        # chạy nhiều giờ và Phase 5/6 sẽ lặp lại hàng chục lượt.
        if not args.hierarchical_lookup:
            _sp = shared_prefix_length[dataidx]
            _nc = max(1, int((args.percent_clusters / 100.0) * (_sp - args.observation_window)))
            if os.path.exists(f'{args.output_path}/global_threshold_{dataidx}_{_nc}.pt'):
                continue

        all_queries_layers = []
        all_keys_layers = []
        all_values_layers = []

        prompt = prompt_format.format(**d)
        prompt_only = prompt_only_format.format(**d)

        # get truncated input prompt
        prompt, _ = truncate_fn(prompt, prompt_only, tokenizer, max_length, dataset, DEV,
                                                   model_name=args.model,
                                                   force_chat=args.force_chat)
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.to(DEV)

        print(f"dataidx: {dataidx} | length of input_ids: {len(input_ids[0])}")
        print(f"dataidx: {dataidx} | shared_prefix_length: {shared_prefix_length[dataidx]}")

        # run generation
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                do_sample=True,
                max_new_tokens=1,
                use_cache=False,
                output_attentions=True
            )

        # write out data
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        # determine num_centroids based on context length
        sp_len = shared_prefix_length[dataidx]
        percentage = ((args.percent_clusters * 1.0) / 100.0)
        num_centroids = int(percentage * (sp_len - args.observation_window))
        percentage_l2 = ((args.percent_clusters_l2 * 1.0) / 100.0)
        num_centroids_l2 = int(percentage_l2 * (sp_len - args.observation_window))

        if num_centroids < 1:
            num_centroids = 1
        if args.hierarchical_lookup:
            assert (num_centroids_l2 >= 1)

        # hierarchical
        if args.hierarchical_lookup:
            centroids_tensor_dict_l2, centroids_labels_dict_l2 = run_clustering(all_keys_layers,
                                                                                num_centroids_l2,
                                                                                observation_window=args.observation_window,
                                                                                device=DEV)
            centroids_tensor_dict_l1, centroids_labels_dict_l1 = run_clustering(centroids_tensor_dict_l2,
                                                                                num_centroids,
                                                                                observation_window=0,
                                                                                device=DEV)

            # update centroid_labels to convert L1 -> L2 mapping to be L1 -> keys for evaluation code
            num_lyrs = len(all_keys_layers)
            for i in range(num_lyrs):
                label_dict_l1 = centroids_labels_dict_l1[i]
                label_dict_l2 = centroids_labels_dict_l2[i]
                gathered_tensor = torch.gather(label_dict_l1, -1, label_dict_l2)
                centroids_labels_dict_l1[i] = gathered_tensor

            # run global threshold
            global_threshold_dict_l1 = run_global_threshold(
                all_keys_layers, all_queries_layers, centroids_tensor_dict_l1, centroids_labels_dict_l1, num_centroids,
                observation_window=args.observation_window,  device=DEV
            )

            # run global threshold (hierarchical lookup) using L2 denominator
            global_threshold_dict_l2 = run_global_threshold(
                all_keys_layers, all_queries_layers, centroids_tensor_dict_l2, centroids_labels_dict_l2, num_centroids_l2,
                observation_window=args.observation_window,  device=DEV
            )

            # save centroids tensor, labels, global threshold
            os.makedirs(args.output_path, exist_ok=True)
            for k,v in centroids_tensor_dict_l1.items():
                centroids_tensor_dict_l1[k] = centroids_tensor_dict_l1[k].cpu()
            for k,v in centroids_labels_dict_l1.items():
                centroids_labels_dict_l1[k] = centroids_labels_dict_l1[k].cpu()
            for k,v in centroids_tensor_dict_l2.items():
                centroids_tensor_dict_l2[k] = centroids_tensor_dict_l2[k].cpu()
            for k,v in centroids_labels_dict_l2.items():
                centroids_labels_dict_l2[k] = centroids_labels_dict_l2[k].cpu()

            # NOTE: tên file phải khớp với chỗ load trong
            # transformers/src/transformers/models/llama/modeling_llama.py (~L1457),
            # tức 'hierarchical_centroids_*' chứ không phải 'hierarchical_lookup_*'.
            torch.save(centroids_tensor_dict_l1, f'{args.output_path}/hierarchical_centroids_tensor_dict_L1_{dataidx}_{num_centroids}.pt')
            torch.save(centroids_labels_dict_l1, f'{args.output_path}/hierarchical_centroids_labels_dict_L1_{dataidx}_{num_centroids}.pt')
            torch.save(centroids_tensor_dict_l2, f'{args.output_path}/centroids_tensor_dict_{dataidx}_{num_centroids_l2}.pt')
            torch.save(centroids_labels_dict_l2, f'{args.output_path}/centroids_labels_dict_{dataidx}_{num_centroids_l2}.pt')
            torch.save(global_threshold_dict_l1, f'{args.output_path}/hierarchical_global_threshold_L1_{dataidx}_{num_centroids}.pt')
            torch.save(global_threshold_dict_l2, f'{args.output_path}/global_threshold_{dataidx}_{num_centroids_l2}.pt')

        else:
            # compute centroids
            centroids_tensor_dict, centroids_labels_dict = run_clustering(all_keys_layers,
                                                                          num_centroids,
                                                                          observation_window=args.observation_window,
                                                                          device=DEV)

            # run global threshold
            global_threshold_dict = run_global_threshold(
                all_keys_layers, all_queries_layers, centroids_tensor_dict, centroids_labels_dict, num_centroids,
                observation_window=args.observation_window, device=DEV
            )

            # save centroids tensor, labels, global threshold
            os.makedirs(args.output_path, exist_ok=True)
            for k,v in centroids_tensor_dict.items():
                centroids_tensor_dict[k] = centroids_tensor_dict[k].cpu()
            for k,v in centroids_labels_dict.items():
                centroids_labels_dict[k] = centroids_labels_dict[k].cpu()

            torch.save(centroids_tensor_dict, f'{args.output_path}/centroids_tensor_dict_{dataidx}_{num_centroids}.pt')
            torch.save(centroids_labels_dict, f'{args.output_path}/centroids_labels_dict_{dataidx}_{num_centroids}.pt')
            torch.save(global_threshold_dict, f'{args.output_path}/global_threshold_{dataidx}_{num_centroids}.pt')

        # free up memory by deleting all qkv from lists
        num_layers = len(all_keys_layers)
        for i in range(num_layers):
            del all_queries_layers[0]
            del all_keys_layers[0]
            del all_values_layers[0]
