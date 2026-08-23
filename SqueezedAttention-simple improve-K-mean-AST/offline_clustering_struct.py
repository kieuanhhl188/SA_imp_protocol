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

KHI SỐ UNIT VƯỢT NGÂN SÁCH CENTROID (`--on_budget_exceeded`)
------------------------------------------------------------
Ở level mịn, số unit cấu trúc có thể nhiều hơn số centroid mà ngân sách cho phép — mỗi unit
cần tối thiểu 1 centroid nên mẫu đó KHÔNG biểu diễn được. Đo ở 5% budget:

    level        LCC            RepoBench-P
    class        0/499          0/492
    function     0/499          2/492   (0,4%)
    block        13/499 (2,6%)  52/492  (10,6%)
    statement    386/499 (77%)  441/492 (90%)

Mặc định là **skip**: bỏ mẫu, ghi vào `feasibility_*.json`, chạy tiếp. KHÔNG raise (chết cả
run, mất GPU time của mọi mẫu sau) và KHÔNG tự gộp. Gộp rồi báo cáo như không có gì xảy ra
sẽ biến một giới hạn NGÂN SÁCH thành một kết luận về CẤU TRÚC: ở statement, gộp nghĩa là
gộp 32–67% số unit của 77–90% số mẫu, tức thứ được gọi là "statement" thực chất đã bị làm
thô về cỡ block, và đầu mịn của level sweep sẽ phẳng ra do chính thao tác đó.

Thứ tự chạy đã chốt: function trước (gần như không mẫu nào bị bỏ) -> block, ghi số mẫu bị
bỏ -> statement, KHÔNG gộp ở thí nghiệm chính, chỉ ghi bao nhiêu mẫu infeasible. Sau đó mới
quyết định có cần một biến thể budget-constrained (`--on_budget_exceeded merge`) hay không.

Bỏ mẫu làm tập còn lại THIÊN LỆCH (mẫu bị bỏ thường dài và cấu trúc mịn), nên điểm của
level này không so thẳng được với level khác chạy trên số mẫu khác. `feasibility_*.json`
ghi đúng danh sách dataidx để Phase 6 so trên tập giao.

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
from squeezedattention.utils import truncate_fn, apply_rope_scaling  # noqa: E402
from struct_clustering import (  # noqa: E402
    LEVELS, parse_units, assign_token_units, compact_unit_ids,
    hard_boundary_kmeans, struct_hierarchy_l1, build_l1_groups,
    compute_token_type_weights, merge_units_to_budget,
)

JSONL_HEAD = "Please complete the code given below. " + chr(10)
JSONL_TAIL = "Next line of code:" + chr(10)


def load_jsonl_contexts(data_dir):
    """Doc <data_dir>/contexts.jsonl — dinh dang do build_*.py sinh ra."""
    path = os.path.join(data_dir, "contexts.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"[ERROR] khong thay {path}")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append({"context": d["fixed_context"], "input": "",
                        "language": d.get("language", "python"),
                        "group_id": d["group_id"]})
    return out


METHODS = ("sa", "hard_boundary", "struct_hierarchy")


def load_phase1(phase1_dir, dataset, model, expect_n=None, expect_mode=None,
                expect_chat=None):
    """
    Đọc offset token do scripts/prepare_code_data.py sinh ra.

    Ưu tiên thư mục con theo model (`<phase1_dir>/<model>/`) — đó là nơi
    `scripts/phase1_gate.sh` ghi — rồi mới tới `<phase1_dir>/`.

    Hai lần kiểm bắt buộc, vì cả hai lỗi này đều KHÔNG crash mà chỉ cho ra kết quả sai:
      - offset của model khác: offset phụ thuộc tokenizer, dùng nhầm là mọi unit AST đều
        gán sai token.
      - thiếu mẫu: bản cũ chỉ in [WARN] rồi bỏ qua, nên một run tưởng là 500 mẫu có thể
        thực chất chỉ chạy 20 mẫu mà không ai nhận ra cho tới lúc đọc bảng kết quả.
    """
    cand = [os.path.join(phase1_dir, model), phase1_dir]
    for d in cand:
        meta_path = os.path.join(d, f"{dataset}_meta.jsonl")
        npz_path = os.path.join(d, f"{dataset}_offsets.npz")
        if os.path.exists(meta_path) and os.path.exists(npz_path):
            break
    else:
        raise SystemExit(
            f"[ERROR] thiếu dữ liệu Phase 1.4, đã tìm ở: {cand}\n"
            f"        Chạy trước: python scripts/prepare_code_data.py {model} "
            f"--dataset {dataset} --output_path {os.path.join(phase1_dir, model)}"
        )
    meta = {}
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            meta[d["dataidx"]] = d

    if expect_chat is not None:
        fcs = {bool(r.get("force_chat", False)) for r in meta.values()}
        if fcs != {expect_chat}:
            raise SystemExit(
                f"[ERROR] {meta_path} sinh voi force_chat={fcs}, run nay yeu cau "
                f"{expect_chat}. Chat template doi shared_prefix_length -> centroid "
                f"khong dung chung duoc."
            )
    if expect_mode is not None:
        modes = {r.get("fixed_context_mode", "longbench") for r in meta.values()}
        if modes != {expect_mode}:
            raise SystemExit(
                f"[ERROR] {meta_path} sinh voi fixed_context={modes}, run nay yeu cau "
                f"'{expect_mode}'.\n"
                f"        Hai che do cho shared_prefix_length khac nhau -> centroid khong "
                f"dung chung duoc. Sinh lai bang prepare_code_data.py "
                f"--fixed_context {expect_mode}"
            )
    models = {r.get("model") for r in meta.values()}
    if models != {model}:
        raise SystemExit(
            f"[ERROR] {meta_path} chứa offset của {models}, còn run này là '{model}'.\n"
            f"        Offset phụ thuộc tokenizer nên hai bộ KHÔNG dùng thay nhau được."
        )
    if expect_n is not None:
        missing = [i for i in range(expect_n) if i not in meta]
        if missing:
            raise SystemExit(
                f"[ERROR] Phase 1.4 chỉ có {len(meta)} mẫu, run này cần {expect_n} "
                f"(thiếu {len(missing)}, ví dụ {missing[:5]}).\n"
                f"        Sinh lại đủ rồi chạy scripts/check_phase1_data.py trước."
            )
    print(f">>> Phase 1.4: {meta_path} ({len(meta)} mẫu, model={model})")
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
    ap.add_argument("--data_source", choices=["longbench", "jsonl"], default="longbench",
                    help="jsonl: doc <data_dir>/contexts.jsonl. PHAI trung voi cach da "
                         "sinh du lieu Phase 1.4")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--rope_scaling", default=None,
                    help="dang 'dynamic:4' de dat 128K. Mac dinh tat -> giu native 32.768. "
                         "Dynamic NTK la phep dong nhat duoi native nen bat len khong doi "
                         "ket qua cua mau ngan")
    ap.add_argument("--force_chat", action="store_true",
                    help="phai TRUNG voi co da dung khi sinh du lieu Phase 1.4 va o pred.py")
    ap.add_argument("--fixed_context", choices=["full", "crossfile"], default="full",
                    help="phai TRUNG voi che do da dung khi sinh du lieu Phase 1.4")
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
    ap.add_argument("--max_k_per_unit", type=int, default=0,
                    help="trần centroid cho MỖI unit; 0 = chỉ chặn theo số token (mặc định). "
                         "Đặt số dương chỉ để ghìm bộ nhớ: trần quá thấp làm không tiêu hết "
                         "ngân sách và mẫu bị tính là infeasible dù ngân sách còn thừa")
    ap.add_argument("--on_budget_exceeded", choices=["skip", "merge", "fail"], default="skip",
                    help="khi số unit > ngân sách centroid. skip (MẶC ĐỊNH, dùng cho thí "
                         "nghiệm chính): bỏ mẫu, ghi vào feasibility_*.json, chạy tiếp. "
                         "merge: gộp unit liền kề cho vừa ngân sách — CHỈ dùng cho biến thể "
                         "budget-constrained và phải báo cáo riêng. fail: dừng cả run")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true",
                    help="tinh lai ca nhung mau da co du file tren dia. Mac dinh BO QUA "
                         "chung, de job dut giua chung chay tiep duoc")
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
    config = apply_rope_scaling(config, args.rope_scaling)
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
    if args.data_source == "jsonl":
        prompt_format = JSONL_HEAD + "{context}" + JSONL_TAIL
        prompt_only_format = JSONL_HEAD + "{context}"
        key_only = "jsonl"
    else:
        key_only = (args.dataset + "_prompt_full" if args.fixed_context == "full"
                    else args.dataset + "_prompt")
        prompt_only_format = dataset2prompt[key_only]
    print(f">>> fixed_context={args.fixed_context}  (template: {key_only})")
    if args.data_source == "jsonl":
        if not args.data_dir:
            raise SystemExit("[ERROR] --data_source jsonl can --data_dir")
        data = load_jsonl_contexts(args.data_dir)
    else:
        data = load_dataset("THUDM/LongBench", args.dataset, split="test")

    n_planned = len(data) if args.limit <= 0 else min(args.limit, len(data))
    meta, offsets_npz = load_phase1(args.phase1_dir, args.dataset, args.model,
                                    expect_n=n_planned, expect_mode=args.fixed_context,
                                    expect_chat=args.force_chat)

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
    n_skipped = 0
    # Sổ khả thi: Phase 6 CẦN file này để so các level trên cùng một tập mẫu. Bỏ mẫu là
    # làm tập còn lại thiên lệch (mẫu bị bỏ thường là file dài, cấu trúc mịn), nên điểm
    # của statement trên tập còn lại KHÔNG so thẳng được với function trên toàn bộ.
    feas = {"dataset": args.dataset, "method": args.method, "level": args.level,
            "percent_clusters": args.percent_clusters,
            "on_budget_exceeded": args.on_budget_exceeded,
            "n_requested": None, "feasible": [], "infeasible": [], "merged": []}

    for dataidx in tqdm(range(n)):
        d = data[dataidx]
        rec = meta.get(dataidx)
        if rec is None:  # load_phase1 đã chặn từ đầu; còn lại đây làm chốt an toàn
            raise SystemExit(f"[ERROR] dataidx {dataidx} không có trong meta Phase 1.4")

        prompt = prompt_format.format(**d)
        prompt_only = prompt_only_format.format(**d)
        prompt, shared_prefix_length = truncate_fn(
            prompt, prompt_only, tokenizer, max_length,
            ("lcc" if args.data_source == "jsonl" else args.dataset), DEV,
            model_name=args.model, force_chat=args.force_chat)
        if shared_prefix_length != rec["shared_prefix_length"]:
            raise SystemExit(
                f"[ERROR] dataidx {dataidx}: shared_prefix_length lệch với Phase 1.4 "
                f"({shared_prefix_length} vs {rec['shared_prefix_length']}). "
                f"Chạy lại prepare_code_data.py với cùng model/config."
            )

        n_ctx = shared_prefix_length - args.observation_window
        num_centroids = max(1, int(args.percent_clusters / 100.0 * n_ctx))

        # ---- BỎ QUA MẪU ĐÃ XONG ----
        # Giống `offline_clustering.py`, nhưng kiểm ĐỦ BỘ chứ không chỉ một file: pod đã
        # tự dựng lại 3 lần trong ngày 20-21/8, và một lần đứt đúng lúc ghi đã để lại 2 file
        # 0 byte. Kiểm một file thì mẫu hỏng dở sẽ bị coi là xong và không bao giờ sinh lại.
        # Chỉ so sánh sự tồn tại + kích thước > 0; tính toàn vẹn CRC để
        # `scripts/check_cluster_integrity.py` lo.
        tag = f"{dataidx}_{num_centroids}"
        need = [f"{args.output_path}/centroids_tensor_dict_{tag}.pt",
                f"{args.output_path}/centroids_labels_dict_{tag}.pt",
                f"{args.output_path}/global_threshold_{tag}.pt"]
        if args.method == "struct_hierarchy":
            # K1 chưa biết trước khi tính, nên chỉ cần có ÍT NHẤT một bộ L1 của dataidx này
            need.append(f"{args.output_path}/k1_stats_{dataidx}.pt")
        if (not args.overwrite
                and all(os.path.exists(f) and os.path.getsize(f) > 0 for f in need)):
            n_skipped += 1
            feas["feasible"].append(dataidx)
            continue

        state["sp_len"] = shared_prefix_length
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.to(DEV)
        all_q.clear()
        all_k.clear()
        with torch.no_grad():
            model.generate(input_ids, do_sample=False, max_new_tokens=1,
                           use_cache=False, output_attentions=True)

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

            # ---- CHÍNH SÁCH KHI SỐ UNIT VƯỢT NGÂN SÁCH CENTROID ----
            # Mặc định `skip`: mẫu này KHÔNG biểu diễn được ở level + ngân sách hiện tại,
            # ghi vào sổ rồi đi tiếp. KHÔNG raise (chết cả run, mất GPU time của các mẫu
            # sau) và KHÔNG tự gộp (biến một giới hạn ngân sách thành kết luận về cấu trúc).
            U_now = int(unit_ids.max()) + 1
            if U_now > num_centroids:
                feas["infeasible"].append(
                    {"dataidx": dataidx, "num_units": U_now, "budget": num_centroids,
                     "n_ctx": n_ctx, "language": rec["language"]})
                if args.on_budget_exceeded == "fail":
                    raise SystemExit(
                        f"[ERROR] dataidx {dataidx}: {U_now} unit > {num_centroids} centroid.")
                if args.on_budget_exceeded == "skip":
                    continue
                # merge: chỉ dùng cho biến thể budget-constrained, phải báo cáo riêng
                unit_ids, mst = merge_units_to_budget(unit_ids, num_centroids)
                feas["merged"].append({"dataidx": dataidx, **mst})
            feas["feasible"].append(dataidx)

            cent, lab = {}, {}
            for li in range(len(all_k)):
                k = all_k[li].squeeze(0).float()[:, :n_ctx, :]
                c, l, _ = hard_boundary_kmeans(
                    k, unit_ids, num_centroids, n_iter=args.n_iter,
                    max_k_per_unit=(args.max_k_per_unit or None), device=DEV,
                    token_weights=tw)
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

    # ---- sổ khả thi ----
    if args.method != "sa":
        feas["n_requested"] = n
        feas_path = (f"{args.output_path}/feasibility_{args.dataset}_{args.method}"
                     f"_{args.level}_pc{args.percent_clusters}.json")
        with open(feas_path, "w", encoding="utf-8") as f:
            json.dump(feas, f, ensure_ascii=False, indent=2)

        n_inf, n_mrg = len(feas["infeasible"]), len(feas["merged"])
        ratio = n_inf / max(n, 1)
        print(f"\n    KHẢ THI ({args.level}, ngân sách {args.percent_clusters}%):")
        print(f"      chạy được       : {len(feas['feasible'])}/{n}")
        print(f"      vượt ngân sách  : {n_inf}/{n} = {100 * ratio:.1f}%"
              f"   (chính sách: {args.on_budget_exceeded})")
        if n_mrg:
            fr = [m["frac_units_merged"] for m in feas["merged"]]
            print(f"      đã gộp          : {n_mrg} mẫu, gộp {100 * min(fr):.0f}–"
                  f"{100 * max(fr):.0f}% số unit của từng mẫu")
        print(f"      sổ khả thi      : {feas_path}")
        if n_skipped:
            print(f"      bỏ qua (đã có)  : {n_skipped}/{n} — dùng --overwrite để tính lại")

        if ratio > 0.10:
            print()
            print("  " + "!" * 66)
            print(f"  !! {100 * ratio:.1f}% MẪU KHÔNG BIỂU DIỄN ĐƯỢC ở level={args.level}, "
                  f"ngân sách {args.percent_clusters}%.")
            print("  !! Tập mẫu còn lại THIÊN LỆCH (mẫu bị bỏ thường dài và cấu trúc mịn).")
            print("  !! KHÔNG so điểm của level này với level khác trên số mẫu khác nhau.")
            print("  !! Phải so trên TẬP GIAO các mẫu khả thi ở mọi level, hoặc chỉ báo cáo")
            print("  !! level này dưới dạng thống kê infeasibility.")
            print("  " + "!" * 66)


if __name__ == "__main__":
    main()
