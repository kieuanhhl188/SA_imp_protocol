#!/usr/bin/env python
"""
prepare_code_data.py — Phase 1.4: sinh và lưu offset BYTE và offset KÝ TỰ cho từng token.

VÌ SAO CẦN FILE NÀY
-------------------
Protocol Phase 1 yêu cầu "lưu kèm byte offset của từng token trong source". Tokenizer nhanh
chỉ trả offset KÝ TỰ (`return_offsets_mapping`), còn tree-sitter đánh địa chỉ theo BYTE.
Script lưu **cả hai**, để không ai phải đoán đang dùng hệ nào:

    offsets_<i>        offset ký tự — Phase 2 hiện chạy trong hệ này
    offsets_bytes_<i>  offset byte  — đúng chữ của protocol

Hai hệ chỉ trùng nhau khi văn bản thuần ASCII. Đo trên dữ liệu thật: LCC **0/500** mẫu có
ký tự non-ASCII, RepoBench-P **107/500**. Với 107 mẫu đó, dùng nhầm hệ là mọi span sau ký
tự non-ASCII đầu tiên đều lệch, và độ lệch cộng dồn.

Không thể tính offset trên source gốc, vì `squeezedattention/utils.py::truncate_fn` cắt
**giữa** prompt rồi decode-lại-encode:

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
    <out>/<dataset>_offsets.npz   hai mảng [num_tokens, 2] cho mỗi mẫu:
                                    "offsets_<i>"        offset KÝ TỰ
                                    "offsets_bytes_<i>"  offset BYTE (protocol Phase 1)

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

# NGÔN NGỮ — đọc từ trường `language` của TỪNG MẪU, không hardcode theo dataset.
#
# LongBench có sẵn trường `language` cho mọi mẫu code, giá trị trùng khít với khoá của
# `struct_clustering.NODE_TYPES`:
#     lcc          python 182 · java 160 · csharp 158
#     repobench-p  python 236 · java 264
#
# Bản trước ghi "python" cho tất cả. Hậu quả đo được (500 mẫu, level=function, budget 5%):
# parse Java/C# bằng parser Python cho **59,5% mẫu LCC chỉ còn ≤2 unit** — tức không còn
# ranh giới cấu trúc nào, `hard_boundary_kmeans` thoái hoá thành K-means thuần = đúng
# baseline SA. Ablation sẽ ra "Idea 1 không có tác dụng" mà không hề crash. Dùng đúng ngôn
# ngữ: trung vị unit 2 -> 15 (LCC), 2 -> 100 (RepoBench-P), số node ERROR trung vị 139 -> 0.
DATASET_LANG_DEFAULT = {
    "lcc": "python",
    "repobench-p": "python",
}

# Ngôn ngữ Phase 2 parse được (struct_clustering.NODE_TYPES). Gặp giá trị lạ thì dừng,
# KHÔNG âm thầm rơi về python — đó chính là lỗi cũ.
SUPPORTED_LANGS = {"python", "java", "csharp", "javascript", "typescript"}

# Chuẩn hoá vài cách viết khác của cùng một ngôn ngữ.
LANG_ALIAS = {
    "c#": "csharp", "cs": "csharp", "c_sharp": "csharp",
    "py": "python", "js": "javascript", "ts": "typescript",
}


def resolve_language(sample, dataset, override=None):
    """
    Ngôn ngữ của MỘT mẫu, theo thứ tự ưu tiên: --language > trường `language` của mẫu >
    mặc định theo dataset.

    Trả về (language, source) với source thuộc {'override', 'dataset_field', 'default'}
    để `_meta.jsonl` ghi lại được ngôn ngữ đến từ đâu — Phase 2 và người đọc log phân biệt
    được "biết chắc" với "đoán".
    """
    if override:
        lang = override
        src = "override"
    else:
        raw = (sample.get("language") or "").strip().lower()
        if raw:
            lang, src = LANG_ALIAS.get(raw, raw), "dataset_field"
        else:
            lang, src = DATASET_LANG_DEFAULT.get(dataset, "python"), "default"
    if lang not in SUPPORTED_LANGS:
        raise SystemExit(
            f"[ERROR] ngôn ngữ '{lang}' (nguồn: {src}) không nằm trong {sorted(SUPPORTED_LANGS)}.\n"
            f"        Phase 2 sẽ không parse được. Bổ sung node type vào "
            f"struct_clustering.NODE_TYPES trước, hoặc lọc mẫu này ra."
        )
    return lang, src

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


def char_to_byte_index(text):
    """
    Bảng tra `c2b[i]` = chỉ số BYTE của ký tự thứ i trong `text` (dài len(text)+1).
    Trả None nếu text thuần ASCII — khi đó byte và ký tự trùng nhau, khỏi tốn bộ nhớ.
    """
    if len(text.encode("utf-8")) == len(text):
        return None
    out = [0] * (len(text) + 1)
    b = 0
    for i, ch in enumerate(text):
        out[i] = b
        b += len(ch.encode("utf-8"))
    out[len(text)] = b
    return out


def offsets_to_bytes(offsets, prompt):
    """
    Đổi offset KÝ TỰ sang offset BYTE.

    VÌ SAO CÓ CẢ HAI: protocol Phase 1 yêu cầu "lưu kèm byte offset của từng token", còn
    `return_offsets_mapping` của tokenizer nhanh trả về offset ký tự. tree-sitter thì đánh
    địa chỉ theo byte. Hai hệ chỉ trùng nhau khi code thuần ASCII — đo trên dữ liệu thật:
    LCC 0/500 mẫu có non-ASCII, RepoBench-P **107/500**.

    Lưu cả hai để không ai phải đoán đang dùng hệ nào: `offsets_<i>` là ký tự,
    `offsets_bytes_<i>` là byte. Phase 2 hiện chạy trong hệ KÝ TỰ (đã kiểm chứng bằng phép
    thử vi sai trên 107 mẫu Unicode); byte offset dành cho công cụ làm việc trực tiếp với
    tree-sitter mà không muốn quy đổi.
    """
    c2b = char_to_byte_index(prompt)
    if c2b is None:
        return offsets
    return [(c2b[s], c2b[e]) for s, e in offsets]


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
    key_only = (args.dataset + "_prompt_protocol" if args.fixed_context == "protocol"
                else args.dataset + "_prompt")
    if key_only not in dataset2prompt:
        raise SystemExit(f"[ERROR] dataset2prompt.json thieu khoa '{key_only}'")
    prompt_only_format = dataset2prompt[key_only]
    print(f">>> fixed_context={args.fixed_context}  (template: {key_only})")
    head, tail = split_prompt_template(prompt_format)

    data = load_dataset("THUDM/LongBench", args.dataset, split="test")
    n = len(data) if args.limit <= 0 else min(args.limit, len(data))
    print(f">>> Số sample xử lý: {n}/{len(data)}")

    os.makedirs(args.output_path, exist_ok=True)
    meta_path = os.path.join(args.output_path, f"{args.dataset}_meta.jsonl")
    npz_path = os.path.join(args.output_path, f"{args.dataset}_offsets.npz")

    # Tên file KHÔNG chứa tên model, mà offset thì phụ thuộc hoàn toàn vào tokenizer.
    # Chạy script này cho Qwen sau khi đã chạy cho LongChat sẽ lặng lẽ đè mất bộ offset
    # 500 mẫu của LongChat — và Phase 2 thì không có cách nào biết mình đang đọc offset
    # của model nào. Chặn ở đây, bắt phải nói rõ ý định.
    if os.path.exists(meta_path) and not args.overwrite:
        prev_model = None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                first = f.readline()
            if first.strip():
                prev_model = json.loads(first).get("model")
        except Exception:
            pass
        if prev_model is not None and prev_model != args.model:
            raise SystemExit(
                f"[ERROR] {meta_path} đang chứa offset của model '{prev_model}',\n"
                f"        còn lệnh này chạy cho '{args.model}'. Offset phụ thuộc tokenizer\n"
                f"        nên hai bộ KHÔNG dùng thay nhau được.\n"
                f"        Chọn một:\n"
                f"          --output_path {os.path.join(args.output_path, args.model)}   (khuyên dùng)\n"
                f"          --overwrite                                    (bỏ bộ cũ đi)"
            )

    import numpy as np
    from tqdm import tqdm

    import collections
    lang_count = collections.Counter()
    lang_src_count = collections.Counter()

    offsets_store = {}
    n_mismatch = 0
    n_truncated = 0
    n_template_miss = 0

    with open(meta_path, "w", encoding="utf-8") as fmeta:
        for dataidx in tqdm(range(n)):
            d = data[dataidx]
            lang, lang_src = resolve_language(d, args.dataset, args.language)
            lang_count[lang] += 1
            lang_src_count[lang_src] += 1
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
                "language": lang,
                "language_source": lang_src,
                "num_tokens": len(ids_fast),
                "shared_prefix_length": int(sp_len),
                "truncated": bool(truncated),
                "num_tokens_before_truncation": int(len_before),
                "code_char_start": int(code_start),
                "code_char_end": int(code_end),
                "num_tokens_in_code": int(n_in_code),
                "num_fixed_tokens_in_code": int(n_fixed_in_code),
                "fast_slow_ids_match": bool(ids_match),
                "prompt_num_bytes": len(prompt.encode("utf-8")),
                "prompt_num_chars": len(prompt),
                "has_byte_offsets": True,
                "fixed_context_mode": args.fixed_context,
                "max_length": int(max_length),
                "template_located": bool(template_ok),
            }
            if args.save_prompt:
                rec["prompt"] = prompt
            fmeta.write(json.dumps(rec, ensure_ascii=False) + "\n")

            offsets_store[f"offsets_{dataidx}"] = np.asarray(offsets, dtype=np.int32)
            offsets_store[f"offsets_bytes_{dataidx}"] = np.asarray(
                offsets_to_bytes(offsets, prompt), dtype=np.int32)
            if args.save_prompt_npz:
                offsets_store[f"prompt_{dataidx}"] = np.array(prompt, dtype=object)

    np.savez_compressed(npz_path, **offsets_store)

    print()
    print(f">>> Đã ghi {meta_path}")
    print(f">>> Đã ghi {npz_path}")
    print(f"    sample bị truncate:            {n_truncated}/{n}")
    print(f"    template không định vị được:   {n_template_miss}/{n}")
    print(f"    token id fast != slow:         {n_mismatch}/{n}")
    print(f"    ngôn ngữ:                      "
          + ", ".join(f"{k}={v}" for k, v in sorted(lang_count.items())))
    print(f"    nguồn ngôn ngữ:                "
          + ", ".join(f"{k}={v}" for k, v in sorted(lang_src_count.items())))
    if lang_src_count.get("default"):
        print()
        print("  [!] Có mẫu KHÔNG có trường `language` -> rơi về mặc định theo dataset.")
        print("      Kiểm tra lại: đoán sai ngôn ngữ làm Phase 2 mất hết ranh giới cấu trúc.")

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

    print("\n=== offset BYTE (yêu cầu của protocol) ===")
    txt_ascii = "def f():\n    return 1\n"
    off_ascii = [(0, 3), (4, 5), (12, 18)]
    check("ASCII: byte == ký tự, không cấp phát bảng",
          char_to_byte_index(txt_ascii) is None
          and offsets_to_bytes(off_ascii, txt_ascii) == off_ascii)

    # 'ú' 2 byte, 'ế' 3 byte -> byte index chạy nhanh hơn char index
    txt_uni = "# chú ý\ndef f():\n    return 1\n"
    c2b = char_to_byte_index(txt_uni)
    check("Unicode: bảng dài len+1", c2b is not None and len(c2b) == len(txt_uni) + 1)
    check("số byte cuối bảng khớp encode()",
          c2b[len(txt_uni)] == len(txt_uni.encode("utf-8")),
          f"{c2b[len(txt_uni)]} vs {len(txt_uni.encode('utf-8'))}")

    # bất biến then chốt: cắt theo byte và cắt theo ký tự phải ra CÙNG một chuỗi
    raw = txt_uni.encode("utf-8")
    tok_char = [(i, i + 3) for i in range(0, len(txt_uni) - 3, 4)]
    tok_byte = offsets_to_bytes(tok_char, txt_uni)
    same = all(raw[b0:b1].decode("utf-8") == txt_uni[c0:c1]
               for (c0, c1), (b0, b1) in zip(tok_char, tok_byte))
    check("cắt theo byte ra đúng chuỗi như cắt theo ký tự", same)
    check("offset byte KHÁC offset ký tự khi có Unicode", tok_byte != tok_char)

    print("\n=== resolve_language ===")
    check("lấy đúng ngôn ngữ của mẫu",
          resolve_language({"language": "java"}, "lcc") == ("java", "dataset_field"))
    check("chuẩn hoá alias C# -> csharp",
          resolve_language({"language": "C#"}, "lcc") == ("csharp", "dataset_field"))
    check("--language ép đè trường của mẫu",
          resolve_language({"language": "java"}, "lcc", "python") == ("python", "override"))
    check("thiếu trường language -> mặc định theo dataset",
          resolve_language({}, "lcc") == ("python", "default"))
    try:
        resolve_language({"language": "brainfuck"}, "lcc")
        check("ngôn ngữ lạ phải dừng chương trình", False)
    except SystemExit:
        check("ngôn ngữ lạ phải dừng chương trình", True)

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
    ap.add_argument("--fixed_context", choices=["protocol", "longbench"], default="protocol",
                    help="dinh nghia fixed_context. protocol (MAC DINH): gom ca phan file "
                         "hien tai truoc con tro, dung chu cua protocol Phase 1. longbench: "
                         "chi gom {context}, giu nguyen quy uoc LongBench de so voi Table 2. "
                         "Chi khac nhau o repobench-p; lcc thi hai che do trung nhau")
    ap.add_argument("--language", default=None, help="ghi đè ngôn ngữ mặc định của dataset")
    ap.add_argument("--save_prompt", action="store_true",
                    help="lưu luôn prompt cuối cùng vào jsonl (file to hơn nhiều)")
    ap.add_argument("--save_prompt_npz", action="store_true",
                    help="lưu prompt vào npz thay vì jsonl")
    ap.add_argument("--self_test", action="store_true",
                    help="kiểm tra logic offline, không cần mạng/GPU")
    ap.add_argument("--overwrite", action="store_true",
                    help="cho phép ghi đè bộ offset đã có của MODEL KHÁC. "
                         "Mặc định script từ chối, vì offset phụ thuộc tokenizer")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.model:
        ap.error("thiếu tham số `model` (hoặc dùng --self_test)")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
