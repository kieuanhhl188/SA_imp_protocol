#!/usr/bin/env python
"""
check_phase1_data.py — GATE DỮ LIỆU PHASE 1, năm bước, chạy trên CPU.

KHÁC GÌ `check_phase1.py`
-------------------------
`check_phase1.py` kiểm bản PORT (Sq-70% có tệ hơn All-KV không) và cần GPU.
File này kiểm DỮ LIỆU mà Phase 2 sẽ đọc, không cần GPU, không cần model weight —
chỉ tokenizer. Chạy nó trước khi thuê GPU: mọi lỗi bắt được ở đây đều rẻ hơn 100 lần
so với bắt được sau một run clustering.

NĂM BƯỚC
--------
    [1] language      mỗi mẫu phải mang ĐÚNG ngôn ngữ của nó (python/java/csharp), không
                      phải mặc định theo dataset. Kiểm luôn hệ quả: parse bằng ngôn ngữ đó
                      có ra ranh giới cấu trúc thật không (U > 2), vì đó mới là thứ Phase 2
                      cần. Hardcode "python" cho Java/C# KHÔNG crash — nó chỉ làm
                      hard_boundary thoái hoá thành K-means thuần, tức ablation ra "không
                      khác gì SA". Đây là lỗi im lặng nguy hiểm nhất của Phase 1.
    [2] đủ 500 mẫu    meta phải phủ TOÀN BỘ dataset và npz phải có mảng offset cho từng
                      dataidx. Thiếu mẫu thì `offline_clustering_struct.py` chỉ in [WARN]
                      rồi bỏ qua -> chạy tưởng 500 mà thực ra 20.
    [3] offset        offset là KÝ TỰ, không phải byte. Năm bất biến: fast==slow token id;
                      số offset == num_tokens; offset không giảm; offset phủ kín prompt
                      không để khoảng trống; và span AST tính theo ký tự không lệch trên mẫu
                      có Unicode — kiểm bằng phép thử vi sai (thay mọi ký tự non-ASCII bằng
                      'x' thì số ký tự giữ nguyên còn số byte đổi, nên span PHẢI giống hệt).
                      Hai hiện tượng KHÔNG phải lỗi, được đếm và in ra dạng INFO: token
                      chồng offset khi BPE tách một ký tự CJK, và token cuối unit nuốt thêm
                      ký tự xuống dòng ngay sau unit.
    [4] fixed_context độ dài đúng và không mất context: với mẫu không bị truncate, `context`
                      gốc phải xuất hiện NGUYÊN VĂN trong prompt cuối; sp_len phải nằm trong
                      khoảng hợp lệ và n_ctx = sp_len - observation_window phải dương.
    [5] tổng kết      PASS/FAIL.

USAGE
-----
    python scripts/check_phase1_data.py qwen2.5-coder-7b --dataset lcc
    python scripts/check_phase1_data.py qwen2.5-coder-7b --dataset repobench-p --limit 50
"""
import argparse
import collections
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

SUPPORTED_LANGS = {"python", "java", "csharp", "javascript", "typescript"}


class Report:
    """Gom kết quả để in một lần, và để exit code phản ánh đúng trạng thái."""

    def __init__(self):
        self.failed = []
        self.warned = []

    def check(self, step, name, cond, detail=""):
        tag = "PASS" if cond else "FAIL"
        print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
        if not cond:
            self.failed.append(f"[{step}] {name}: {detail}")

    def warn(self, name, detail=""):
        print(f"  [WARN] {name}" + (f" — {detail}" if detail else ""))
        self.warned.append(f"{name}: {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--dataset", default="lcc", choices=["lcc", "repobench-p"])
    ap.add_argument("--phase1_dir",
                    default=os.environ.get("SQA_PHASE1_DIR",
                                           os.path.join(REPO_ROOT, "phase1_data")))
    ap.add_argument("--level", default="function",
                    help="level dùng để kiểm ranh giới cấu trúc (giống Phase 2)")
    ap.add_argument("--observation_window", type=int, default=100)
    ap.add_argument("--percent_clusters", type=int, default=5)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--max_degenerate_ratio", type=float, default=0.10,
                    help="tỉ lệ mẫu được phép có <=2 unit. Vượt ngưỡng là FAIL: nghĩa là "
                         "ranh giới cấu trúc gần như không tồn tại, Phase 2 sẽ đo vào hư không")
    args = ap.parse_args()

    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from squeezedattention.utils import truncate_fn
    from struct_clustering import parse_units, assign_token_units, compact_unit_ids

    rep = Report()

    model2path = json.load(open(os.path.join(REPO_ROOT, "LongBench/config/model2path.json"),
                                encoding="utf-8"))
    model2maxlen = json.load(open(os.path.join(REPO_ROOT, "LongBench/config/model2maxlen.json"),
                                  encoding="utf-8"))
    d2p = json.load(open(os.path.join(REPO_ROOT, "LongBench/config/dataset2prompt.json"),
                         encoding="utf-8"))
    model_path = model2path[args.model]
    max_length = model2maxlen[args.model]

    # Tìm theo đúng thứ tự mà offline_clustering_struct.py tìm: thư mục con theo model
    # trước (nơi phase1_gate.sh ghi), rồi mới tới thư mục gốc.
    cand = [os.path.join(args.phase1_dir, args.model), args.phase1_dir]
    for dpath in cand:
        meta_path = os.path.join(dpath, f"{args.dataset}_meta.jsonl")
        npz_path = os.path.join(dpath, f"{args.dataset}_offsets.npz")
        if os.path.exists(meta_path) and os.path.exists(npz_path):
            break
    else:
        raise SystemExit(f"[ERROR] không thấy dữ liệu {args.dataset} ở {cand}. "
                         f"Chạy scripts/prepare_code_data.py trước.")

    meta = {}
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            meta[d["dataidx"]] = d
    npz = np.load(npz_path)
    data = load_dataset("THUDM/LongBench", args.dataset, split="test")
    n_total = len(data)
    n = n_total if args.limit <= 0 else min(args.limit, n_total)

    print(f"\n{'=' * 70}")
    print(f"GATE DỮ LIỆU PHASE 1 — {args.dataset} · {args.model}")
    print(f"  meta: {meta_path}")
    print(f"  npz : {npz_path}")
    print(f"  kiểm {n}/{n_total} mẫu · level={args.level} · obs={args.observation_window}")
    print(f"{'=' * 70}")

    # ---------------------------------------------------------------
    # BƯỚC 2 (làm trước vì các bước sau cần meta đầy đủ)
    # ---------------------------------------------------------------
    print("\n=== BƯỚC 2 — đủ mẫu ===")
    # Với --limit, chỉ đòi phủ đủ phần sẽ dùng; nhưng nói rõ đây là bộ CHƯA đầy đủ, để
    # không ai nhầm một gate 20 mẫu với một gate 500 mẫu.
    missing_meta = [i for i in range(n) if i not in meta]
    rep.check(2, f"meta phủ đủ {n} mẫu sẽ dùng", not missing_meta,
              f"thiếu {len(missing_meta)} mẫu" if missing_meta else f"{len(meta)} mẫu trong meta")
    if n < n_total:
        rep.warn(f"chỉ kiểm {n}/{n_total} mẫu (--limit)",
                 "Phase 5/6 cần đủ dataset — nhớ chạy lại không giới hạn")
    missing_npz = [i for i in meta if f"offsets_{i}" not in npz]
    rep.check(2, "npz có mảng offset cho mọi dataidx trong meta", not missing_npz,
              f"thiếu {len(missing_npz)}" if missing_npz else f"{len(meta)} mảng")
    models_in_meta = {r.get("model") for r in meta.values()}
    rep.check(2, "meta chỉ chứa một model", len(models_in_meta) == 1, str(models_in_meta))
    rep.check(2, "model của meta khớp lệnh đang chạy", models_in_meta == {args.model},
              f"{models_in_meta} vs {args.model}")

    # ---------------------------------------------------------------
    # BƯỚC 1 — ngôn ngữ
    # ---------------------------------------------------------------
    print("\n=== BƯỚC 1 — ngôn ngữ ===")
    # So trên CÙNG TẬP CHỈ SỐ. Bản đầu so phân bố của meta (có thể chỉ 20 mẫu vì --limit)
    # với phân bố của cả 500 mẫu dataset -> FAIL giả, trong khi dữ liệu hoàn toàn đúng.
    idx_cmp = sorted(meta.keys())
    lang_meta = collections.Counter(meta[i]["language"] for i in idx_cmp)
    lang_data = collections.Counter(data[i]["language"] for i in idx_cmp)
    src = collections.Counter(r.get("language_source", "?") for r in meta.values())
    print(f"  meta   : {dict(sorted(lang_meta.items()))}   ({len(idx_cmp)} mẫu)")
    print(f"  dataset: {dict(sorted(lang_data.items()))}   (cùng {len(idx_cmp)} dataidx đó)")
    print(f"  nguồn  : {dict(sorted(src.items()))}")
    if len(idx_cmp) < n_total:
        full = collections.Counter(data[i]["language"] for i in range(n_total))
        print(f"  (toàn bộ dataset {n_total} mẫu: {dict(sorted(full.items()))})")
    rep.check(1, "ngôn ngữ trong meta khớp trường `language` của dataset",
              lang_meta == lang_data)
    rep.check(1, "mọi ngôn ngữ đều parse được ở Phase 2",
              set(lang_meta) <= SUPPORTED_LANGS, str(set(lang_meta) - SUPPORTED_LANGS))
    rep.check(1, "không mẫu nào phải đoán ngôn ngữ (source != 'default')",
              src.get("default", 0) == 0, f"{src.get('default', 0)} mẫu")
    per_sample_wrong = [i for i in range(n_total)
                        if i in meta and meta[i]["language"] != data[i]["language"]]
    rep.check(1, "đúng ngôn ngữ ở TỪNG mẫu, không chỉ đúng phân bố",
              not per_sample_wrong, f"{len(per_sample_wrong)} mẫu lệch")

    # ---------------------------------------------------------------
    # BƯỚC 3 + 4 — chạy lại đường dữ liệu của Phase 2 trên từng mẫu
    # ---------------------------------------------------------------
    print("\n=== BƯỚC 3 + 4 — offset và fixed_context (dựng lại prompt như Phase 2) ===")
    tok_slow = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    prompt_format = d2p[args.dataset]
    prompt_only_format = d2p[args.dataset + "_prompt"]

    bad_ids_match, bad_len, bad_sorted, bad_cover = [], [], [], []
    bad_sp, bad_nctx, bad_context_lost, bad_unit_span = [], [], [], []
    bad_unicode_span = []
    n_trunc, n_nonascii, degenerate, budget_short = 0, 0, [], []
    n_tok_cross, n_tok_total, n_unicode_code, n_overlap = 0, 0, 0, 0
    units_all = []

    for i in tqdm(range(n), disable=None):
        rec = meta.get(i)
        if rec is None:
            continue
        d = data[i]

        # --- dựng lại prompt đúng cách Phase 2 dựng ---
        prompt_raw = prompt_format.format(**d)
        prompt_only = prompt_only_format.format(**d)
        prompt, sp_len = truncate_fn(prompt_raw, prompt_only, tok_slow, max_length,
                                     args.dataset, "cpu")
        offs = npz[f"offsets_{i}"]

        # ---------------- BƯỚC 3 ----------------
        if not rec.get("fast_slow_ids_match", False):
            bad_ids_match.append(i)
        if len(offs) != rec["num_tokens"]:
            bad_len.append(i)
        starts_np = offs[:, 0]
        if len(starts_np) > 1 and bool((starts_np[1:] < starts_np[:-1]).any()):
            bad_sorted.append(i)
        # Offset phải phủ KÍN prompt, nhưng KHÔNG đòi lát cắt rời nhau.
        # Lý do: BPE byte-level của Qwen tách một ký tự nhiều byte (CJK, emoji) thành
        # nhiều token, và tokenizer quy các token con đó về cùng khoảng ký tự -> hai
        # token chồng lên nhau, ví dụ (225,227) rồi (226,227) cho chữ '尝试'. Không ký tự
        # nào bị mất, thứ tự start vẫn không giảm, cả hai token con rơi vào cùng một unit
        # -> Phase 2 không hề hấn gì. Đo trên RepoBench-P: 31/500 mẫu. Cái phải chặn là
        # KHOẢNG TRỐNG (ký tự không thuộc token nào), nên chỉ kiểm khoảng trống.
        pos = 0
        for a, b in offs:
            if a > pos:
                bad_cover.append(i)
                break
            pos = max(pos, int(b))
        else:
            if pos != len(prompt):
                bad_cover.append(i)
        if any(int(a) < int(prev_b) for (a, _), (_, prev_b)
               in zip(offs[1:], offs[:-1])):
            n_overlap += 1

        # ---------------- BƯỚC 4 ----------------
        if sp_len != rec["shared_prefix_length"]:
            bad_sp.append(i)
        n_ctx = sp_len - args.observation_window
        if n_ctx <= 0:
            bad_nctx.append(i)
        truncated = rec["truncated"]
        n_trunc += int(truncated)
        if not truncated and d["context"] not in prompt:
            # không truncate thì context gốc phải còn nguyên trong prompt
            bad_context_lost.append(i)

        if len(d["context"].encode("utf-8")) != len(d["context"]):
            n_nonascii += 1

        # ---------------- BƯỚC 1 (hệ quả) + BƯỚC 3 (bất biến span) ----------------
        if n_ctx <= 0:
            continue
        code = prompt[rec["code_char_start"]:rec["code_char_end"]]
        spans, _ = parse_units(code, rec["language"], args.level)
        spans = [(s + rec["code_char_start"], e + rec["code_char_start"]) for s, e in spans]
        spans.append((0, len(prompt) + 1))
        starts = torch.from_numpy(offs[:n_ctx, 0].astype("int64"))
        raw_ids = assign_token_units(starts, spans)
        unit_ids, _ = compact_unit_ids(raw_ids)
        U = int(unit_ids.max()) + 1
        units_all.append(U)
        if U <= 2:
            degenerate.append(i)
        k_total = max(1, int(args.percent_clusters / 100.0 * n_ctx))
        if U > k_total:
            budget_short.append(i)

        # Bất biến ĐÚNG của assign_token_units: token được gán theo ĐIỂM BẮT ĐẦU, nên
        # start phải nằm trong [span_start, span_end). Không đòi end <= span_end: token
        # cuối một hàm thường nuốt luôn ký tự xuống dòng ngay sau hàm (`)\n`), vượt biên
        # phải đúng 1 ký tự. Đó là hành vi bình thường, không phải lệch offset — đo trên
        # LCC: 9/2690 token mỗi mẫu, luôn là token cuối unit. Vẫn đếm để theo dõi.
        ends = torch.from_numpy(offs[:n_ctx, 1].astype("int64"))
        sp_start = torch.tensor([spans[int(u)][0] for u in raw_ids])
        sp_end = torch.tensor([spans[int(u)][1] for u in raw_ids])
        nonempty = ends > starts
        inside = ((starts >= sp_start) & (starts < sp_end)) | ~nonempty
        if not bool(inside.all()):
            bad_unit_span.append((i, int((~inside).sum())))
        n_tok_cross += int(((ends > sp_end) & nonempty).sum())
        n_tok_total += int(nonempty.sum())

        # KIỂM LỖI BYTE/KÝ TỰ (phép thử vi sai). tree-sitter đánh địa chỉ theo byte, offset
        # Phase 1.4 theo ký tự. Thay mọi ký tự non-ASCII bằng 'x': số KÝ TỰ không đổi, số
        # BYTE thì đổi, và cú pháp giữ nguyên (non-ASCII chỉ nằm trong comment/chuỗi).
        # Vậy span tính theo ký tự phải GIỐNG HỆT nhau. Lệch = quy đổi byte->ký tự sai.
        if len(code.encode("utf-8")) != len(code):
            n_unicode_code += 1
            ascii_code = "".join(c if ord(c) < 128 else "x" for c in code)
            spans_ascii, _ = parse_units(ascii_code, rec["language"], args.level)
            base = [(s - rec["code_char_start"], e - rec["code_char_start"])
                    for s, e in spans[:-1]]
            if sorted(base) != sorted(spans_ascii):
                bad_unicode_span.append(i)

    m = len(units_all)
    print()
    rep.check(3, "token id tokenizer nhanh == chậm", not bad_ids_match,
              f"{len(bad_ids_match)} mẫu lệch")
    rep.check(3, "số offset == num_tokens", not bad_len, f"{len(bad_len)} mẫu lệch")
    rep.check(3, "offset không giảm (điều kiện của searchsorted)", not bad_sorted,
              f"{len(bad_sorted)} mẫu")
    rep.check(3, "offset phủ kín prompt, không sót ký tự nào", not bad_cover,
              f"{len(bad_cover)} mẫu có khoảng trống" if bad_cover
              else f"{n} mẫu, 0 khoảng trống")
    if n_overlap:
        print(f"  [INFO] {n_overlap}/{n} mẫu có token chồng offset — BPE byte-level tách "
              f"ký tự nhiều byte (CJK) thành nhiều token; các token con cùng unit, vô hại")
    rep.check(3, "điểm bắt đầu của token nằm trong unit AST được gán",
              not bad_unit_span,
              f"{len(bad_unit_span)} mẫu, ví dụ {bad_unit_span[:3]}" if bad_unit_span
              else f"{n_tok_total} token")
    rep.check(3, "span AST không lệch trên mẫu Unicode (phép thử vi sai byte/ký tự)",
              not bad_unicode_span,
              f"{len(bad_unicode_span)} mẫu lệch: {bad_unicode_span[:5]}" if bad_unicode_span
              else (f"{n_unicode_code}/{n} mẫu có Unicode trong vùng code, span khớp tuyệt đối"
                    if n_unicode_code else f"0/{n} mẫu có Unicode — phép thử không áp dụng"))
    if n_tok_total:
        print(f"  [INFO] token vắt qua biên phải của unit (nuốt ký tự xuống dòng): "
              f"{n_tok_cross}/{n_tok_total} = {100 * n_tok_cross / n_tok_total:.2f}% "
              f"— bình thường với cách gán theo điểm bắt đầu")
    rep.check(4, "shared_prefix_length dựng lại khớp meta", not bad_sp,
              f"{len(bad_sp)} mẫu lệch")
    rep.check(4, "n_ctx = sp_len - obs > 0", not bad_nctx, f"{len(bad_nctx)} mẫu")
    rep.check(4, "mẫu không truncate giữ nguyên văn `context` gốc", not bad_context_lost,
              f"{len(bad_context_lost)} mẫu mất context")
    print(f"  [INFO] bị truncate: {n_trunc}/{n} mẫu "
          f"(mất khúc giữa -> cú pháp hỏng, tree-sitter sẽ sinh node ERROR)")

    if m:
        deg_ratio = len(degenerate) / m
        units_sorted = sorted(units_all)
        print(f"  [INFO] unit/mẫu ở level={args.level}: "
              f"trung vị={units_sorted[m // 2]} min={units_sorted[0]} max={units_sorted[-1]}")
        rep.check(1, f"ranh giới cấu trúc tồn tại thật (<={args.max_degenerate_ratio:.0%} "
                     f"mẫu có U<=2)", deg_ratio <= args.max_degenerate_ratio,
                  f"{len(degenerate)}/{m} = {deg_ratio:.1%}")
        if budget_short:
            rep.warn(f"{len(budget_short)}/{m} mẫu có U > ngân sách centroid "
                     f"({args.percent_clusters}%) ở level={args.level}",
                     "các mẫu này KHÔNG biểu diễn được ở level+ngân sách này; "
                     "offline_clustering_struct sẽ bỏ qua và ghi vào feasibility_*.json "
                     "(--on_budget_exceeded skip). Phải báo cáo con số này, và không so "
                     "điểm giữa các level trên số mẫu khác nhau")

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    if rep.failed:
        print(f"PHASE 1 DATA ({args.dataset}): FAIL — {len(rep.failed)} mục")
        for f in rep.failed:
            print(f"  - {f}")
        print("=" * 70)
        return 1
    print(f"PHASE 1 DATA ({args.dataset}): PASS"
          + (f"  ({len(rep.warned)} cảnh báo, không chặn)" if rep.warned else ""))
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
