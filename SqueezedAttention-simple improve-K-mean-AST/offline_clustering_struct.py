"""
offline_clustering_struct.py — Phase 2: offline clustering có ranh giới cấu trúc.

Song song với `offline_clustering.py` gốc (không sửa file đó, để gate Phase 0 giữ nguyên).
Sinh centroid/label/threshold đúng định dạng mà `LongBench/pred.py` nạp được.

BA NHÁNH ABLATION (protocol 2.5), tách bạch hoàn toàn:

    --method sa                K-means thuần trên toàn bộ key. Gọi thẳng
                               squeezedattention.clustering.run_clustering, KHÔNG đi qua
                               struct_clustering. Đây là baseline để đối chiếu — chạy qua
                               script này thay vì script gốc để loại trừ khác biệt do
                               môi trường.
    --method hard_boundary     K-means độc lập trong từng unit AST.        <- đề xuất 1
    --method struct_hierarchy  hard_boundary ở L2 + L1 = trung bình theo   <- đề xuất 2
                               unit cha. Ghi thêm file L1 cho hierarchical lookup.

Giữ nguyên Si/threshold/kernel (protocol 2.6): threshold vẫn do
`run_global_threshold` tính, y hệt mọi nhánh.

Cần chạy `scripts/prepare_code_data.py` trước để có offset token.

Ví dụ:
    python offline_clustering_struct.py qwen2.5-coder-7b-instruct \\
        --dataset lcc --method hard_boundary --level function \\
        --percent_clusters 5 --output_path clusters/lcc/

    # ablation cộng dồn trên cùng dữ liệu
    for M in sa hard_boundary struct_hierarchy; do
        python offline_clustering_struct.py qwen2.5-coder-7b-instruct \\
            --dataset lcc --method $M --output_path clusters_$M/lcc/
    done
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from squeezedattention.clustering import run_clustering, run_global_threshold  # noqa: E402
from squeezedattention.utils import truncate_fn  # noqa: E402
from struct_clustering import (  # noqa: E402
    LEVELS, parse_units, assign_token_units, compact_unit_ids,
    hard_boundary_kmeans, struct_hierarchy_l1, build_l1_groups,
    compute_token_type_weights,
)

METHODS = ("sa", "hard_boundary", "struct_hierarchy")


def load_phase1(phase1_dir, dataset):
    """Đọc offset token do scripts/prepare_code_data.py sinh ra."""
    meta_path = os.path.join(phase1_dir, f"{dataset}_meta.jsonl")
    npz_path = os.path.join(phase1_dir, f"{dataset}_offsets.npz")
    if not (os.path.exists(meta_path) and os.path.exists(npz_path)):
        raise SystemExit(
            f"[ERROR] thiếu dữ liệu Phase 1.4 tại {phase1_dir}.\n"
            f"        Chạy trước: python scripts/prepare_code_data.py <model> "
            f"--dataset {dataset}"
        )
    meta = {}
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            meta[d["dataidx"]] = d
    return meta, np.load(npz_path)


def build_unit_ids(prompt, rec, offsets, sp_len, obs_window, level, args):
    """
    Dựng unit_id cho phần fixed context (đã bỏ observation window).

    Trả về (unit_ids [n_ctx], token_weights hoặc None, thống kê).

    Lưu ý quan trọng: AST được parse trên PROMPT CUỐI CÙNG sau truncation, không phải file
    source gốc — vì offset của Phase 1.4 tính trên chuỗi đó. Sample bị truncate mất khúc
    giữa nên code hỏng cú pháp; tree-sitter vẫn parse được và sinh node ERROR, số lượng
    được ghi lại để lọc sau.
    """
    n_ctx = sp_len - obs_window
    starts = torch.from_numpy(offsets[:n_ctx, 0].astype(np.int64))

    code_start = rec["code_char_start"]
    code_end = rec["code_char_end"]
    code = prompt[code_start:code_end]

    spans, pstat = parse_units(code, rec["language"], level)
    # đưa span từ toạ độ trong `code` về toạ độ trong `prompt`
    spans = [(s + code_start, e + code_start) for s, e in spans]
    # thêm một span phủ cả prompt, để token phần chỉ dẫn (trước code) cũng có unit
    spans.append((0, len(prompt) + 1))

    unit_ids = assign_token_units(starts, spans)
    unit_ids, _ = compact_unit_ids(unit_ids)

    weights = None
    if args.token_weights:
        w_code = compute_token_type_weights(code, starts - code_start, rec["language"])
        # token ngoài vùng code giữ trọng số 1
        outside = (starts < code_start) | (starts >= code_end)
        w_code[outside] = 1.0
        weights = w_code

    stat = {
        "num_units": int(unit_ids.max()) + 1,
        "num_error_nodes": pstat["num_error_nodes"],
        "truncated": rec["truncated"],
    }
    return unit_ids, weights, stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=str)
    ap.add_argument("--dataset", type=str, default="lcc",
                    choices=["lcc", "repobench-p"])
    ap.add_argument("--output_path", type=str, default="output_struct/")
    ap.add_argument("--phase1_dir", type=str,
                    default=os.environ.get("SQA_PHASE1_DIR", "phase1_data"))

    # --- ablation ---
    ap.add_argument("--method", choices=METHODS, default="hard_boundary")
    ap.add_argument("--level", choices=LEVELS, default="function",
                    help="level cho unit L2 (ranh giới cứng). Phase 7 quét tham số này")
    ap.add_argument("--level_l1", choices=LEVELS, default="class",
                    help="level cho unit cha khi --method struct_hierarchy")
    ap.add_argument("--l1_weighted", action="store_true", default=True,
                    help="L1 centroid = trung bình CÓ trọng số theo số key (mặc định bật)")
    ap.add_argument("--no_l1_weighted", dest="l1_weighted", action="store_false")
    ap.add_argument("--token_weights", action="store_true",
                    help="[Hướng 2(b), TẮT SẴN] nhân trọng số theo loại token khi cập nhật "
                         "centroid. Không nằm trong ablation của protocol -> bật thì phải "
                         "báo cáo thành nhánh riêng")

    # --- ngân sách, giống offline_clustering.py ---
    ap.add_argument("--percent_clusters", type=int, default=5)
    ap.add_argument("--percent_clusters_l2", type=int, default=1,
                    help="phần trăm cho L1 khi hierarchical (bài: L1=1%%, L2=5%%)")
    ap.add_argument("--observation_window", type=int, default=100)
    ap.add_argument("--n_iter", type=int, default=10)
    ap.add_argument("--max_k_per_unit", type=int, default=64)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    DEV = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    model2path = json.load(open("LongBench/config/model2path.json", encoding="utf-8"))
    model2maxlen = json.load(open("LongBench/config/model2maxlen.json", encoding="utf-8"))
    model_path = model2path[args.model]
    max_length = model2maxlen[args.model]

    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = AutoConfig.from_pretrained(model_path)
    config.return_qkv_states = True
    config._flash_attn_2_enabled = True
    config._attn_implementation = "flash_attention_2"
    if getattr(config, "use_sliding_window", False):
        print("[WARN] use_sliding_window=True -> ép về False cho Squeezed Attention")
        config.use_sliding_window = False
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config,
                                                 torch_dtype=torch.bfloat16).eval().to(DEV)

    n_kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
    print(f">>> method={args.method} level={args.level} "
          f"level_l1={args.level_l1} token_weights={args.token_weights}")
    print(f">>> heads Q={config.num_attention_heads} KV={n_kv} "
          f"layers={config.num_hidden_layers}")

    layers = model.model.layers
    dataset2prompt = json.load(open("LongBench/config/dataset2prompt.json", encoding="utf-8"))
    prompt_format = dataset2prompt[args.dataset]
    prompt_only_format = dataset2prompt[args.dataset + "_prompt"]
    data = load_dataset("THUDM/LongBench", args.dataset, split="test")

    meta, offsets_npz = load_phase1(args.phase1_dir, args.dataset)

    # hook thu q/k giống offline_clustering.py.
    # `state` là hộp chứa để hook đọc được sp_len của sample hiện tại — dùng biến cục bộ
    # của main() thì hook (định nghĩa lồng nhưng gọi từ forward) sẽ đọc giá trị cũ.
    all_q, all_k = [], []
    state = {"sp_len": 0}

    def hook(module, inp, out):
        _, qkv, _ = out
        q, k, _v = qkv
        sp = state["sp_len"]
        all_q.append(q[:, :, :sp])
        all_k.append(k[:, :, :sp])

    for layer in layers:
        layer.self_attn.register_forward_hook(hook)

    os.makedirs(args.output_path, exist_ok=True)
    n = len(data) if args.limit <= 0 else min(args.limit, len(data))
    agg = {"error_nodes": 0, "truncated": 0, "units": [], "k1": []}

    for dataidx in tqdm(range(n)):
        d = data[dataidx]
        rec = meta.get(dataidx)
        if rec is None:
            print(f"[WARN] dataidx {dataidx} không có trong meta Phase 1.4, bỏ qua")
            continue

        prompt = prompt_format.format(**d)
        prompt_only = prompt_only_format.format(**d)
        prompt, shared_prefix_length = truncate_fn(
            prompt, prompt_only, tokenizer, max_length, args.dataset, DEV)
        if shared_prefix_length != rec["shared_prefix_length"]:
            raise SystemExit(
                f"[ERROR] dataidx {dataidx}: shared_prefix_length lệch với Phase 1.4 "
                f"({shared_prefix_length} vs {rec['shared_prefix_length']}). "
                f"Chạy lại prepare_code_data.py với cùng model/config."
            )

        state["sp_len"] = shared_prefix_length
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.to(DEV)
        all_q.clear()
        all_k.clear()
        with torch.no_grad():
            model.generate(input_ids, do_sample=False, max_new_tokens=1,
                           use_cache=False, output_attentions=True)

        n_ctx = shared_prefix_length - args.observation_window
        num_centroids = max(1, int(args.percent_clusters / 100.0 * n_ctx))

        if args.method == "sa":
            cent, lab = run_clustering(all_k, num_centroids,
                                       observation_window=args.observation_window, device=DEV)
        else:
            offs = offsets_npz[f"offsets_{dataidx}"]
            unit_ids, tw, st = build_unit_ids(prompt, rec, offs, shared_prefix_length,
                                              args.observation_window, args.level, args)
            agg["error_nodes"] += st["num_error_nodes"]
            agg["truncated"] += int(st["truncated"])
            agg["units"].append(st["num_units"])

            cent, lab = {}, {}
            for li in range(len(all_k)):
                k = all_k[li].squeeze(0).float()[:, :n_ctx, :]
                c, l, _ = hard_boundary_kmeans(
                    k, unit_ids, num_centroids, n_iter=args.n_iter,
                    max_k_per_unit=args.max_k_per_unit, device=DEV, token_weights=tw)
                cent[li], lab[li] = c, l

        thr = run_global_threshold(all_k, all_q, cent, lab, num_centroids,
                                   observation_window=args.observation_window, device=DEV)

        tag = f"{dataidx}_{num_centroids}"
        torch.save({k: v.cpu() for k, v in cent.items()},
                   f"{args.output_path}/centroids_tensor_dict_{tag}.pt")
        torch.save({k: v.cpu() for k, v in lab.items()},
                   f"{args.output_path}/centroids_labels_dict_{tag}.pt")
        torch.save(thr, f"{args.output_path}/global_threshold_{tag}.pt")

        # ---- tầng L1 theo cấu trúc ----
        if args.method == "struct_hierarchy":
            target_k1 = max(1, int(args.percent_clusters_l2 / 100.0 * n_ctx))
            offs = offsets_npz[f"offsets_{dataidx}"]
            unit_l1_raw, _, _ = build_unit_ids(prompt, rec, offs, shared_prefix_length,
                                               args.observation_window, args.level_l1, args)
            l1_groups, st1 = build_l1_groups(unit_ids, unit_l1_raw, target_k1)
            agg["k1"].append(st1["k1_actual"])

            c1, l1 = {}, {}
            for li in range(len(all_k)):
                a, b = struct_hierarchy_l1(cent[li], lab[li], l1_groups,
                                           weighted=args.l1_weighted)
                c1[li], l1[li] = a, b
            thr1 = run_global_threshold(all_k, all_q, c1, l1, st1["k1_actual"],
                                        observation_window=args.observation_window, device=DEV)

            t1 = f"{dataidx}_{st1['k1_actual']}"
            torch.save({k: v.cpu() for k, v in c1.items()},
                       f"{args.output_path}/hierarchical_centroids_tensor_dict_L1_{t1}.pt")
            torch.save({k: v.cpu() for k, v in l1.items()},
                       f"{args.output_path}/hierarchical_centroids_labels_dict_L1_{t1}.pt")
            torch.save(thr1,
                       f"{args.output_path}/hierarchical_global_threshold_L1_{t1}.pt")
            # K1 thực tế thường lệch danh nghĩa -> ghi lại để Phase 6 báo cáo budget đo thật
            torch.save({"k1_actual": st1["k1_actual"], "k1_target": target_k1,
                        "k1_raw": st1["k1_raw"], "mode": st1["k1_mode"]},
                       f"{args.output_path}/k1_stats_{dataidx}.pt")

    print(f"\n>>> Xong. Output: {args.output_path}")
    if agg["units"]:
        u = np.array(agg["units"])
        print(f"    unit/sample: min={u.min()} tb={u.mean():.1f} max={u.max()}")
        print(f"    sample bị truncate: {agg['truncated']}/{n}")
        print(f"    tổng node ERROR khi parse: {agg['error_nodes']}")
    if agg["k1"]:
        k1 = np.array(agg["k1"])
        print(f"    K1 thực tế: min={k1.min()} tb={k1.mean():.1f} max={k1.max()} "
              f"(danh nghĩa {args.percent_clusters_l2}% context)")


if __name__ == "__main__":
    main()
