#!/usr/bin/env python
"""
prepare_code_data.py — Phase 1.4: sinh và lưu offset ký tự cho từng token.

VÌ SAO CẦN FILE NÀY
-------------------
Protocol Phase 1 yêu cầu "lưu kèm byte offset của từng token trong source". Không thể
tính offset trên source gốc, vì `squeezedattention/utils.py::truncate_fn` cắt **giữa**
prompt rồi decode-lại-encode:

    prompt = decode(tokens[:half]) + decode(tokens[-half:])

Sau bước đó chuỗi đưa vào model KHÔNG còn là source gốc — phần giữa biến mất và ranh giới
nối có thể lệch vài ký tự. Mọi offset tính trên source gốc đều sai.

Nên ở đây offset được tính trên **chuỗi prompt cuối cùng**, đúng cái model nhìn thấy.
Phase 2 vì vậy cũng phải parse AST trên chuỗi cuối cùng này, không phải trên file gốc.

HỆ QUẢ CHO PHASE 2 (đọc kỹ)
---------------------------
Với sample bị truncate, code đã mất phần giữa nên **không còn đúng cú pháp**. tree-sitter
vẫn parse được (nó error-tolerant) nhưng sẽ sinh node ERROR quanh chỗ nối. Trường
`truncated` trong output đánh dấu các sample này để Phase 2 quyết định: bỏ qua, xử lý
riêng, hay chấp nhận ERROR node.

TOKENIZER
---------
`offline_clustering.py` load tokenizer với `use_fast=False`, mà tokenizer chậm KHÔNG hỗ trợ
`return_offsets_mapping` (ném `NotImplementedError`). Script này load cả hai:
  - tokenizer CHẬM  -> chạy `truncate_fn`, để prompt sinh ra giống hệt offline_clustering
  - tokenizer NHANH -> lấy offset trên prompt cuối cùng
rồi **assert hai bên ra cùng token id**. Nếu lệch, offset sẽ không khớp với key vector mà
offline_clustering thu được -> script báo lỗi thay vì lặng lẽ cho ra dữ liệu sai.

OUTPUT
------
    <out>/<dataset>_meta.jsonl    metadata mỗi sample, một dòng một sample
    <out>/<dataset>_offsets.npz   mảng offset [num_tokens, 2] theo key "offsets_<dataidx>"

USAGE
-----
    python scripts/prepare_code_data.py longchat-v1.5-7b-32k --dataset lcc
    python scripts/prepare_code_data.py longchat-v1.5-7b-32k --dataset repobench-p --limit 5
    python scripts/prepare_code_data.py --self_test        # kiểm tra logic, không cần mạng
"""
import argparse
import json
import os
import sys

# Console Windows mac dinh cp1252 -> tieng Viet co dau se lam crash print().
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

# Task code của LongBench và ngôn ngữ mặc định để parse AST ở Phase 2.
# LCC gồm nhiều ngôn ngữ (Python/Java/C#); RepoBench-P là Python/Java.
# Ngôn ngữ thật nên suy ra từ nội dung, ở đây chỉ ghi mặc định để Phase 2 ghi đè.
DATASET_LANG_DEFAULT = {
    "lcc": "python",
    "repobench-p": "python",
}

# truncate_fn bỏ qua build_chat với các dataset này -> không chạm bug `model_name`
# chưa định nghĩa ở squeezedattention/utils.py:44
NO_CHAT_TEMPLATE = {"trec", "triviaqa", "samsum", "lcc", "repobench-p"}


# =====================================================================
# LOGIC LÕI — tách riêng để test được mà không cần mạng
# =====================================================================

def split_prompt_template(prompt_format):
    """
    Tách prompt template thành (phần đầu trước {context}, phần đuôi sau placeholder cuối).

    Ví dụ lcc:
        "Please complete the code given below. \\n{context}Next line of code:\\n"
        -> ("Please complete the code given below. \\n", "Next line of code:\\n")

    Ví dụ repobench-p:
        "Please complete the code given below. \\n{context}{input}Next line of code:\\n"
        -> ("Please complete the code given below. \\n", "Next line of code:\\n")
    """
    head_end = prompt_format.find("{")
    if head_end < 0:
        return "", ""
    head = prompt_format[:head_end]

    tail_start = prompt_format.rfind("}")
    tail = prompt_format[tail_start + 1:] if tail_start >= 0 else ""
    return head, tail


def locate_code_region(prompt, head, tail):
    """
    Xác định vùng code trong prompt CUỐI CÙNG (sau truncation).

    Trả về (start, end) là offset ký tự. Dùng string matching thay vì tin vào độ dài
    template, vì truncate_fn decode-lại nên có thể lệch khoảng trắng đầu chuỗi.
    """
    start = 0
    if head:
        if prompt.startswith(head):
            start = len(head)
        else:
            # truncate_fn có thể làm lệch vài ký tự đầu -> tìm trong 2x độ dài head
            idx = prompt.find(head.strip(), 0, max(len(head) * 2, 200))
            if idx >= 0:
                start = idx + len(head.strip())

    end = len(prompt)
    if tail:
        if prompt.endswith(tail):
            end = len(prompt) - len(tail)
        else:
            idx = prompt.rfind(tail.strip())
            if idx >= 0:
                end = idx

    if end < start:  # template không khớp -> lấy cả prompt, đánh dấu ở caller
        return 0, len(prompt)
    return start, end


def offsets_from_fast_tokenizer(prompt, tok_fast):
    """
    Lấy offset ký tự cho từng token của prompt cuối cùng.

    Trả về list [(start, end), ...] dài bằng số token.
    add_special_tokens=True để khớp với cách offline_clustering.py encode
    (`tokenizer(prompt, truncation=False, return_tensors="pt")`).
    """
    enc = tok_fast(prompt, truncation=False, return_offsets_mapping=True,
                   add_special_tokens=True)
    return list(enc["offset_mapping"]), list(enc["input_ids"])


def summarize_alignment(offsets, code_start, code_end, sp_len):
    """Đếm token nằm trong vùng code và trong phần fixed context, để sanity-check."""
    n_in_code = 0
    n_fixed_in_code = 0
    for i, (s, e) in enumerate(offsets):
        if s == e:          # special token (BOS) có offset rỗng
            continue
        if s >= code_start and e <= code_end:
            n_in_code += 1
            if i < sp_len:
                n_fixed_in_code += 1
    return n_in_code, n_fixed_in_code


# =====================================================================
# CHẠY THẬT
# =====================================================================

def run(args):
    import torch  # noqa: F401  (truncate_fn cần torch)
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from squeezedattention.utils import truncate_fn

    if args.dataset not in NO_CHAT_TEMPLATE:
        raise SystemExit(
            f"[ERROR] dataset '{args.dataset}' không nằm trong {sorted(NO_CHAT_TEMPLATE)}.\n"
            f"        Với dataset khác, truncate_fn gọi build_chat(prompt, model_name) mà\n"
            f"        `model_name` chưa định nghĩa trong scope -> NameError.\n"
            f"        Sửa squeezedattention/utils.py:44 trước khi dùng dataset đó."
        )

    model2path = json.load(open(os.path.join(REPO_ROOT, "LongBench/config/model2path.json"),
                                encoding="utf-8"))
    model2maxlen = json.load(open(os.path.join(REPO_ROOT, "LongBench/config/model2maxlen.json"),
                                  encoding="utf-8"))
    model_path = model2path[args.model]
    max_length = model2maxlen[args.model]

    print(f">>> Model:   {args.model}  ({model_path})")
    print(f">>> Dataset: {args.dataset}   max_length={max_length}")

    # CHẬM: dùng cho truncate_fn, để prompt giống hệt offline_clustering.py
    tok_slow = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    # NHANH: dùng cho offset_mapping
    tok_fast = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    dataset2prompt = json.load(open(os.path.join(REPO_ROOT, "LongBench/config/dataset2prompt.json"),
                                    encoding="utf-8"))
    prompt_format = dataset2prompt[args.dataset]
    prompt_only_format = dataset2prompt[args.dataset + "_prompt"]
    head, tail = split_prompt_template(prompt_format)

    data = load_dataset("THUDM/LongBench", args.dataset, split="test")
    n = len(data) if args.limit <= 0 else min(args.limit, len(data))
    print(f">>> Số sample xử lý: {n}/{len(data)}")

    os.makedirs(args.output_path, exist_ok=True)
    meta_path = os.path.join(args.output_path, f"{args.dataset}_meta.jsonl")
    npz_path = os.path.join(args.output_path, f"{args.dataset}_offsets.npz")

    import numpy as np
    from tqdm import tqdm

    offsets_store = {}
    n_mismatch = 0
    n_truncated = 0
    n_template_miss = 0

    with open(meta_path, "w", encoding="utf-8") as fmeta:
        for dataidx in tqdm(range(n)):
            d = data[dataidx]
            prompt_raw = prompt_format.format(**d)
            prompt_only = prompt_only_format.format(**d)

            len_before = len(tok_slow(prompt_raw, truncation=False).input_ids)

            # truncate_fn dùng tokenizer CHẬM, y hệt offline_clustering.py
            prompt, sp_len = truncate_fn(
                prompt_raw, prompt_only, tok_slow, max_length, args.dataset, "cpu"
            )
            truncated = len_before > max_length
            n_truncated += int(truncated)

            # offset lấy từ tokenizer NHANH trên prompt CUỐI CÙNG
            offsets, ids_fast = offsets_from_fast_tokenizer(prompt, tok_fast)
            ids_slow = tok_slow(prompt, truncation=False).input_ids

            ids_match = (list(ids_fast) == list(ids_slow))
            if not ids_match:
                n_mismatch += 1

            code_start, code_end = locate_code_region(prompt, head, tail)
            template_ok = not (code_start == 0 and code_end == len(prompt) and head)
            if not template_ok:
                n_template_miss += 1

            n_in_code, n_fixed_in_code = summarize_alignment(
                offsets, code_start, code_end, sp_len
            )

            rec = {
                "dataidx": dataidx,
                "dataset": args.dataset,
                "model": args.model,
                "language": args.language or DATASET_LANG_DEFAULT.get(args.dataset, "python"),
                "num_tokens": len(ids_fast),
                "shared_prefix_length": int(sp_len),
                "truncated": bool(truncated),
                "num_tokens_before_truncation": int(len_before),
                "code_char_start": int(code_start),
                "code_char_end": int(code_end),
                "num_tokens_in_code": int(n_in_code),
                "num_fixed_tokens_in_code": int(n_fixed_in_code),
                "fast_slow_ids_match": bool(ids_match),
                "template_located": bool(template_ok),
            }
            if args.save_prompt:
                rec["prompt"] = prompt
            fmeta.write(json.dumps(rec, ensure_ascii=False) + "\n")

            offsets_store[f"offsets_{dataidx}"] = np.asarray(offsets, dtype=np.int32)
            if args.save_prompt_npz:
                offsets_store[f"prompt_{dataidx}"] = np.array(prompt, dtype=object)

    np.savez_compressed(npz_path, **offsets_store)

    print()
    print(f">>> Đã ghi {meta_path}")
    print(f">>> Đã ghi {npz_path}")
    print(f"    sample bị truncate:            {n_truncated}/{n}")
    print(f"    template không định vị được:   {n_template_miss}/{n}")
    print(f"    token id fast != slow:         {n_mismatch}/{n}")

    if n_mismatch:
        print()
        print("  [!!] Tokenizer nhanh và chậm ra token id KHÁC NHAU ở một số sample.")
        print("       Offset sẽ KHÔNG khớp với key vector mà offline_clustering.py thu được,")
        print("       vì file đó dùng tokenizer chậm. Phải xử lý trước khi dùng cho Phase 2.")
        return 1
    if n_template_miss:
        print()
        print("  [!] Một số sample không định vị được vùng code theo template.")
        print("      Kiểm tra lại dataset2prompt.json cho dataset này.")
    return 0


# =====================================================================
# SELF TEST — không cần mạng, không cần GPU
# =====================================================================

class _MockFastTokenizer:
    """Tokenizer giả: tách theo khoảng trắng, có BOS, trả offset thật."""

    def __call__(self, text, truncation=False, return_offsets_mapping=False,
                 add_special_tokens=True, **kw):
        offsets, ids = [], []
        if add_special_tokens:
            offsets.append((0, 0))   # BOS: offset rỗng
            ids.append(1)
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            j = i
            while j < len(text) and not text[j].isspace():
                j += 1
            offsets.append((i, j))
            ids.append(hash(text[i:j]) % 30000)
            i = j
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


def self_test():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")
        if not cond:
            ok = False

    print("\n=== split_prompt_template ===")
    lcc_fmt = "Please complete the code given below. \n{context}Next line of code:\n"
    rb_fmt = "Please complete the code given below. \n{context}{input}Next line of code:\n"
    h1, t1 = split_prompt_template(lcc_fmt)
    h2, t2 = split_prompt_template(rb_fmt)
    check("lcc head", h1 == "Please complete the code given below. \n", repr(h1))
    check("lcc tail", t1 == "Next line of code:\n", repr(t1))
    check("repobench head", h2 == h1, repr(h2))
    check("repobench tail (qua {input})", t2 == "Next line of code:\n", repr(t2))

    print("\n=== locate_code_region ===")
    code = "def foo(a):\n    return a + 1\n"
    prompt = h1 + code + t1
    s, e = locate_code_region(prompt, h1, t1)
    check("định vị đúng vùng code", prompt[s:e] == code, repr(prompt[s:e]))

    # mô phỏng truncate_fn làm lệch khoảng trắng đầu chuỗi
    prompt_shift = h1.rstrip() + code + t1
    s2, e2 = locate_code_region(prompt_shift, h1, t1)
    check("chịu được lệch khoảng trắng đầu", prompt_shift[s2:e2].strip() == code.strip(),
          repr(prompt_shift[s2:e2][:30]))

    # template không khớp -> fallback cả prompt
    s3, e3 = locate_code_region("noise only", h1, t1)
    check("fallback khi template không khớp", (s3, e3) == (0, len("noise only")))

    print("\n=== offsets_from_fast_tokenizer ===")
    tok = _MockFastTokenizer()
    offsets, ids = offsets_from_fast_tokenizer(prompt, tok)
    check("số offset == số token", len(offsets) == len(ids), f"{len(offsets)} vs {len(ids)}")
    # mọi offset không rỗng phải cắt ra đúng chuỗi con không chứa khoảng trắng
    slices_ok = all(prompt[s:e] and not prompt[s:e][0].isspace()
                    for s, e in offsets if s != e)
    check("offset cắt ra đúng chuỗi con", slices_ok)
    check("BOS có offset rỗng", offsets[0] == (0, 0))

    print("\n=== summarize_alignment ===")
    n_in_code, n_fixed = summarize_alignment(offsets, s, e, sp_len=len(ids))
    tokens_in_code = [prompt[a:b] for a, b in offsets if a != b and a >= s and b <= e]
    check("đếm token trong vùng code", n_in_code == len(tokens_in_code),
          f"{n_in_code} vs {len(tokens_in_code)}")
    check("token vùng code không dính phần đuôi",
          all("Next" not in t for t in tokens_in_code))
    n_in_code2, n_fixed2 = summarize_alignment(offsets, s, e, sp_len=3)
    check("sp_len giới hạn được phần fixed", n_fixed2 <= 3 and n_fixed2 <= n_in_code2,
          f"n_fixed={n_fixed2}")

    print("\n=== round-trip: offset -> chuỗi con ===")
    # đây là bất biến Phase 2 sẽ dựa vào: token thứ i phủ prompt[offsets[i][0]:offsets[i][1]]
    rebuilt = "".join(prompt[a:b] for a, b in offsets)
    check("ghép các chuỗi con ra đúng prompt (bỏ khoảng trắng)",
          rebuilt == "".join(prompt.split()), f"len {len(rebuilt)}")

    print("\n" + ("TẤT CẢ PASS" if ok else "CÓ TEST FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None, help="tên model trong model2path.json")
    ap.add_argument("--dataset", default="lcc", choices=sorted(NO_CHAT_TEMPLATE))
    ap.add_argument("--output_path",
                    default=os.environ.get("SQA_PHASE1_DIR",
                                           os.path.join(REPO_ROOT, "phase1_data")))
    ap.add_argument("--limit", type=int, default=-1, help="chỉ xử lý N sample đầu; -1 = tất cả")
    ap.add_argument("--language", default=None, help="ghi đè ngôn ngữ mặc định của dataset")
    ap.add_argument("--save_prompt", action="store_true",
                    help="lưu luôn prompt cuối cùng vào jsonl (file to hơn nhiều)")
    ap.add_argument("--save_prompt_npz", action="store_true",
                    help="lưu prompt vào npz thay vì jsonl")
    ap.add_argument("--self_test", action="store_true",
                    help="kiểm tra logic offline, không cần mạng/GPU")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.model:
        ap.error("thiếu tham số `model` (hoặc dùng --self_test)")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
