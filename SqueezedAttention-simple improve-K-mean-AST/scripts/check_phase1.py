#!/usr/bin/env python
"""
check_phase1.py — gate cho ban port Squeezed Attention sang Qwen2 (Phase 1.5 + 1.6).

VI SAO KHONG DUNG check_gate.py
-------------------------------
check_gate.py so voi Table 2 cua bai. Table 2 chi co LongChat / LLaMA-2-32K / LWM —
**khong co Qwen2.5-Coder**. Khong co so doi chieu ngoai, nen gate Phase 1 phai dua vao
mot tieu chi NOI TAI:

    Sq-70% tren Qwen khong duoc te hon All-KV tren Qwen qua `--tolerance` diem.

Do la dung dieu Phase 0 da xac nhan cho duong LLaMA (Sq-70% 56.08 vs All-KV 54.83,
+1.25). Neu ban port GQA tra nham nhom centroid — loi de xay ra nhat, va la loi
KHONG crash — thi Sq-70% tut vai diem so voi All-KV. Do chinh la thu gate nay bat.

Ba dieu gate KHONG chung minh, phai ghi ro khi bao cao:
  1. Khong noi gi ve viec Qwen manh hay yeu hon LongChat — hai model khac nhau.
  2. Chay tren `--limit` mau dau thi phuong sai lon; dung de ket luan xu huong.
  3. Sq-70% ~ All-KV moi la dieu kien CAN. No khong loai duoc kha nang ca hai duong
     cung sai theo cung mot kieu (vd truncation hong lam ca hai cung tut).

USAGE
-----
    python scripts/check_phase1.py --model qwen2.5-coder-7b-instruct --task lcc --limit 20
"""
import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

sys.path.insert(0, HERE)
from check_gate import env_summary  # noqa: E402  (dung lai nguyen ban de log dong nhat)


def config_dir(model, use_centroids, percent_clusters, percentile, limit):
    """Dung lai dung quy uoc ten thu muc cua LongBench/pred.py."""
    name = (f"{model}_baseline" if not use_centroids
            else f"{model}_PC{percent_clusters}_PERC{percentile}")
    if limit > 0:
        name = f"{name}_lim{limit}"
    return name


def load_score(pred_dir, dirname, task):
    path = os.path.join(pred_dir, dirname, "result.json")
    if not os.path.exists(path):
        return None, path
    with open(path, "r", encoding="utf-8") as f:
        res = json.load(f)
    return res.get(task), path


def degenerate_ratio(pred_dir, dirname, task):
    """Ti le mau ma metric cham vao mot dong RONG.

    Bai hoc 17/8: gate bao PASS (Sq-70% 20.85 >= All-KV 17.60) trong khi ca hai con so
    deu vo nghia — model instruct sinh ra gan nhu khong gi ca, va `code_sim_score` cham
    vao dong rong. Tieu chi "Sq-70% khong te hon All-KV" khong the phat hien ca do, vi
    ca hai duong cung hong theo cung mot kieu. Nen phai kiem RIENG.

    Tra ve (ti_le, so_mau) hoac (None, 0) neu khong doc duoc file.
    """
    path = os.path.join(pred_dir, dirname, f"{task}.jsonl")
    if not os.path.exists(path):
        return None, 0
    n = n_empty = 0
    try:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            pred = json.loads(line).get("pred", "")
            chosen = ""
            for ln in pred.lstrip("\n").split("\n"):
                if ("`" not in ln) and ("#" not in ln) and ("//" not in ln):
                    chosen = ln
                    break
            n += 1
            n_empty += int(not chosen.strip())
    except Exception:
        return None, 0
    return (n_empty / n if n else None), n


def append_md_log(md_path, args, rows, verdict, env, meta):
    from datetime import datetime

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    badge = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "NO_DATA": "⬜ chưa có dữ liệu"}[verdict]

    scope = f"{args.limit} mẫu đầu" if args.limit > 0 else "toàn bộ dataset"

    lines = ["", f"### {ts} — Phase 1 gate (port Qwen2/GQA) — {args.model} — {badge}", ""]
    if args.run_note:
        lines += [f"> {args.run_note}", ""]

    lines += [
        f"- Tiêu chí: Sq-70% **không tệ hơn** All-KV quá ±{args.tolerance} điểm "
        f"(Table 2 không có Qwen nên không có mốc ngoài)",
        f"- Task `{args.task}`, phạm vi: **{scope}**",
        f"- pred_dir: `{args.pred_dir}`",
    ]
    if meta.get("tokenizer_check"):
        lines.append(f"- Phase 1.4 (offset/tokenizer): {meta['tokenizer_check']}")
    if meta.get("degenerate"):
        lines.append(f"- Dòng được metric chấm bị **rỗng**: {meta['degenerate']}")
    if env:
        lines.append(f"- GPU: `{env['gpu']}`  (CUDA_VISIBLE_DEVICES=`{env['cuda_visible']}`)")
        fork = "đúng fork" if env["is_fork"] else "**KHÔNG phải fork trong repo**"
        lines.append(f"- transformers `{env['transformers']}` ({fork}) | torch `{env['torch']}` "
                     f"| triton `{env['triton']}` | flash_attn `{env['flash_attn']}` "
                     f"| cuml `{env['cuml']}`")
        lines.append(f"- seed: `{env['seed']}`")
    else:
        lines.append("- env: *(chưa có `env_record.json`, chạy `scripts/record_env.py`)*")
    lines.append("")

    lines.append("| Config | Task | Điểm | Ghi chú |")
    lines.append("|---|---|---:|---|")
    for cfg, task, score, note in rows:
        s = f"{score:.2f}" if isinstance(score, float) else "-"
        lines.append(f"| {cfg} | {task} | {s} | {note} |")
    lines.append("")

    if meta.get("delta") is not None:
        lines.append(f"**Sq-70% − All-KV = {meta['delta']:+.2f} điểm** "
                     f"(Phase 0 trên LongChat: +1.25)")
        lines.append("")

    lines.append("Gate này chỉ chứng minh đường GQA nạp và tra đúng nhóm centroid. "
                 "Nó **không** so Qwen với LongChat và **không** thay cho Phase 5/6.")
    lines.append("")
    if args.console_log:
        lines += [f"Console log đầy đủ: [`{args.console_log}`]({args.console_log})", ""]
    lines += ["<!-- ghi chú tay bên dưới -->", ""]

    with open(md_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return md_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder-7b-instruct")
    ap.add_argument("--task", default="lcc")
    ap.add_argument("--pred_dir", default=os.path.join(REPO_ROOT, "LongBench", "pred"))
    ap.add_argument("--percent_clusters", type=int, default=5)
    ap.add_argument("--percentile", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--tolerance", type=float, default=2.0,
                    help="Sq-70% duoc phep thap hon All-KV toi da bao nhieu diem. "
                         "Mac dinh 2.0, cung muc da chot o Phase 0")
    ap.add_argument("--max_degenerate", type=float, default=0.25,
                    help="ti le toi da mau co dong-duoc-cham RONG. Vuot nguong nay thi "
                         "FAIL truoc khi so diem, vi diem khong con y nghia")
    ap.add_argument("--env_record", default=None)
    ap.add_argument("--console_log", default=None)
    ap.add_argument("--run_note", default=None)
    ap.add_argument("--tokenizer_check", default=None,
                    help="tom tat ket qua prepare_code_data.py de nhung vao nhat ky")
    ap.add_argument("--log_md", default=os.path.join(REPO_ROOT, "EXPERIMENT_LOG.md"))
    ap.add_argument("--no_log_md", action="store_true")
    args = ap.parse_args()

    d_base = config_dir(args.model, False, args.percent_clusters, args.percentile, args.limit)
    d_sq = config_dir(args.model, True, args.percent_clusters, args.percentile, args.limit)

    base, p_base = load_score(args.pred_dir, d_base, args.task)
    sq, p_sq = load_score(args.pred_dir, d_sq, args.task)

    print("=" * 66)
    print("  PHASE 1 GATE — port Squeezed Attention sang Qwen2 (GQA)")
    print(f"  Model:   {args.model}")
    print(f"  Task:    {args.task}" + (f"  (chi {args.limit} mau dau)" if args.limit > 0 else ""))
    print(f"  Tieu chi: Sq-70% >= All-KV - {args.tolerance}")
    print("=" * 66)
    print()
    print(f"  All-KV : {base if base is not None else 'THIEU'}   <- {p_base}")
    print(f"  Sq-70% : {sq if sq is not None else 'THIEU'}   <- {p_sq}")
    print()

    rows = []
    meta = {"tokenizer_check": args.tokenizer_check, "delta": None, "degenerate": None}

    # --- Chan output rong TRUOC khi so diem ---
    # Neu model sinh sai dinh dang thi moi con so sau day deu vo nghia, ke ca hieu so.
    deg_rows = []
    for label, dirname in (("All-KV", d_base), ("Sq-70%", d_sq)):
        ratio, n = degenerate_ratio(args.pred_dir, dirname, args.task)
        if ratio is not None:
            deg_rows.append((label, ratio, n))
    worst = max((r for _, r, _ in deg_rows), default=0.0)
    if deg_rows:
        meta["degenerate"] = "; ".join(f"{lb} {100*r:.0f}% ({n} mẫu)" for lb, r, n in deg_rows)
        print("  Dong duoc metric cham bi RONG: "
              + ", ".join(f"{lb} {100*r:.0f}%" for lb, r, _ in deg_rows))
        print()

    if worst > args.max_degenerate:
        print(f"  ❌ FAIL — {100*worst:.0f}% mau co dong cham RONG "
              f"(nguong {100*args.max_degenerate:.0f}%).")
        print("     Diem so KHONG phan anh chat luong retrieval, no phan anh viec model")
        print("     sinh sai DINH DANG. Hieu so Sq-70% vs All-KV o day vo nghia.")
        print("     Xem prediction tho:  python scripts/inspect_preds.py "
              f"{os.path.join(args.pred_dir, d_base, args.task + '.jsonl')}")
        print("     Nguyen nhan thuong gap: model instruct + prompt completion tho")
        print("     (LongBench bo chat template cho lcc/repobench-p) -> dung ban base.")
        if not args.no_log_md:
            env = env_summary(args.env_record)
            rows.append(("All KV", args.task, base, "output rỗng, điểm vô nghĩa"))
            rows.append(("Sq-70%", args.task, sq, "output rỗng, điểm vô nghĩa"))
            append_md_log(args.log_md, args, rows, "FAIL", env, meta)
        return 1

    if base is None or sq is None:
        verdict = "NO_DATA"
        rows.append(("All KV", args.task, base, "thiếu result.json" if base is None else ""))
        rows.append(("Sq-70%", args.task, sq, "thiếu result.json" if sq is None else ""))
        print("  ⬜ CHUA CO DU LIEU — chay pred.py + eval.py cho ca hai cau hinh truoc.")
    else:
        delta = sq - base
        meta["delta"] = delta
        ok = delta >= -args.tolerance
        verdict = "PASS" if ok else "FAIL"
        rows.append(("All KV", args.task, base, "trần accuracy"))
        rows.append(("Sq-70%", args.task, sq,
                     f"{'✅' if ok else '❌'} lệch {delta:+.2f} so với All-KV"))
        print(f"  Sq-70% - All-KV = {delta:+.2f}")
        print()
        if ok:
            print("  ✅ PASS — duong GQA nap va tra dung nhom centroid.")
            print("     Sq-70% khong tut so voi All-KV, cung chieu voi Phase 0 tren LongChat.")
        else:
            print(f"  ❌ FAIL — Sq-70% thap hon All-KV {-delta:.2f} diem (> {args.tolerance}).")
            print("     Nghi pham theo thu tu de kiem:")
            print("       1. repeat_interleave vs repeat -> tra nham nhom centroid")
            print("         (scripts/test_gqa_port.py bat duoc ca nay tren CPU)")
            print("       2. shared_prefix_length lech giua offline_clustering va pred")
            print("         (tokenizer fast/slow khac nhau)")
            print("       3. centroid sinh SAU repeat_kv -> 28 head thay vi 4")
            print("         (assert o modeling_qwen2 se bat, kiem log)")

    if not args.no_log_md:
        env = env_summary(args.env_record)
        path = append_md_log(args.log_md, args, rows, verdict, env, meta)
        print()
        print(f"  Da phu luc nhat ky: {path}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
