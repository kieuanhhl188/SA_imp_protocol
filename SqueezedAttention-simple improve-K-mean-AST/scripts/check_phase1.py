#!/usr/bin/env python
"""
check_phase1.py — gate cho ban port Squeezed Attention sang Qwen2 (Phase 1.5 + 1.6).

VI SAO KHONG DUNG check_gate.py
-------------------------------
check_gate.py so voi Table 2 cua bai. Table 2 chi co LongChat / LLaMA-2-32K / LWM —
**khong co Qwen2.5-Coder**. Khong co so doi chieu ngoai, nen gate Phase 1 phai dua vao
mot tieu chi NOI TAI:

    Sq-70% tren Qwen khong duoc kem hon All-KV MOT CACH CO Y NGHIA THONG KE.

Tieu chi la PAIRED TEST tren hieu so tung mau, khong phai nguong diem co dinh. Ly do
(do 18/8, n=20): ban port hong cho -42.30 voi p<0.0001 va tut deu 20/20 mau; ban port
dung cho -2.80 voi p=0.22 va 14/20 mau y het nhau. Mot nguong +-2.0 goi CA HAI la FAIL.
Tren 20 mau, SE cua rieng mot diem trung binh da vuot 5 diem — doc hieu cua hai trung
binh nhu mot con so tuyet doi la sai phuong phap. Van giu them chan `--max_drop` cho
truong hop hong nang ma n qua nho de dat y nghia thong ke.

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


def paired_diff(pred_dir, dir_a, dir_b, task):
    """So THEO CAP tu result_detail.json. Tra ve dict hoac None neu thieu du lieu.

    Nguong diem co dinh (vd +-2.0) khong phan biet duoc nhieu voi loi khi n nho: tren
    20 mau, SE cua rieng mot diem trung binh da vuot 5 diem. Nhung hai lan chay dung
    CUNG tap mau, CUNG model, chi khac co use_centroids -> thiet ke ghep cap, va dai
    luong dung la phan bo HIEU SO TUNG MAU.

    Do 18/8: ban port hong cho -42.30 voi p<0.0001 (tut deu 20/20); ban port dung cho
    -2.80 voi p=0.22 va 14/20 mau y het nhau. Nguong +-2.0 goi ca hai la FAIL.
    """
    import math

    def load(d):
        path = os.path.join(pred_dir, d, "result_detail.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                blk = json.load(f).get(task) or {}
            per = blk.get("per_sample") or {}
            return {int(k): float(v) for k, v in per.items()} or None
        except Exception:
            return None

    pa, pb = load(dir_a), load(dir_b)
    if not pa or not pb:
        return None
    common = sorted(set(pa) & set(pb))
    if len(common) < 2:
        return None

    diffs = [pb[i] - pa[i] for i in common]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    pos = sum(1 for d in diffs if d > 1e-9)
    neg = sum(1 for d in diffs if d < -1e-9)
    k, m = min(pos, neg), pos + neg
    pval = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / (2 ** m)) if m else 1.0
    return {
        "n": n, "same": n - pos - neg, "better": pos, "worse": neg,
        "mean": 100 * mean, "se": 100 * se,
        "ci_lo": 100 * (mean - 1.96 * se), "ci_hi": 100 * (mean + 1.96 * se),
        "p": pval,
    }


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

    lines = ["", f"### {ts} — Phase 1 gate — {args.model} — {badge}", ""]
    if args.run_note:
        lines += [f"> {args.run_note}", ""]

    lines += [
        f"- Tiêu chí: **paired test** trên hiệu số từng mẫu — FAIL nếu KTC95 nằm hẳn "
        f"dưới 0, hoặc tụt quá {args.max_drop} điểm. Table 2 không có Qwen nên không có "
        f"mốc ngoài; ngưỡng điểm cố định thì không phân biệt được nhiễu với lỗi ở n nhỏ",
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
    pd = meta.get("paired")
    if pd:
        lines.append(f"| Mẫu y hệt | Sq tốt hơn | Sq kém hơn | Hiệu số | KTC 95% | sign test |")
        lines.append("|---:|---:|---:|---:|---|---:|")
        lines.append(f"| {pd['same']}/{pd['n']} | {pd['better']} | {pd['worse']} | "
                     f"{pd['mean']:+.2f} ± {pd['se']:.2f} | "
                     f"[{pd['ci_lo']:+.2f}, {pd['ci_hi']:+.2f}] | p = {pd['p']:.4f} |")
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
    ap.add_argument("--model", default="longchat-v1.5-7b-32k")
    ap.add_argument("--task", default="lcc")
    ap.add_argument("--pred_dir", default=os.path.join(REPO_ROOT, "LongBench", "pred"))
    ap.add_argument("--percent_clusters", type=int, default=5)
    ap.add_argument("--percentile", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--tolerance", type=float, default=2.0,
                    help="Sq-70% duoc phep thap hon All-KV toi da bao nhieu diem. "
                         "Mac dinh 2.0, cung muc da chot o Phase 0")
    ap.add_argument("--max_drop", type=float, default=10.0,
                    help="tut qua bao nhieu diem thi FAIL bat ke y nghia thong ke. Chan "
                         "truong hop hong nang ma n qua nho de dat p nho")
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
    print("  PHASE 1 GATE — Sq-70% tra dung nhom centroid (paired test)")
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

        # Tieu chi: PAIRED TEST khi co du lieu tung mau, khong phai nguong diem co dinh.
        #   FAIL neu (a) kem hon co Y NGHIA THONG KE  -> khoang tin cay 95% nam han duoi 0
        #        hoac (b) tut qua --max_drop diem     -> chan hong nang du n qua nho de
        #                                                dat y nghia thong ke
        pd = paired_diff(args.pred_dir, d_base, d_sq, args.task)
        meta["paired"] = pd
        if pd:
            print(f"  So theo cap: {pd['same']}/{pd['n']} mau y het | "
                  f"{pd['better']} tot hon | {pd['worse']} kem hon")
            print(f"  Hieu so {pd['mean']:+.2f} +- {pd['se']:.2f}  "
                  f"KTC95 [{pd['ci_lo']:+.2f}, {pd['ci_hi']:+.2f}]  sign test p={pd['p']:.4f}")
            print()
            sig_worse = pd["ci_hi"] < 0
            catastrophic = pd["mean"] < -args.max_drop
            ok = not (sig_worse or catastrophic)
        else:
            print(f"  (thieu result_detail.json -> quay ve nguong diem +-{args.tolerance})")
            print()
            sig_worse = catastrophic = False
            ok = delta >= -args.tolerance

        verdict = "PASS" if ok else "FAIL"
        rows.append(("All KV", args.task, base, "trần accuracy"))
        rows.append(("Sq-70%", args.task, sq,
                     f"{'✅' if ok else '❌'} lệch {delta:+.2f} so với All-KV"))
        print(f"  Sq-70% - All-KV = {delta:+.2f}")
        print()
        if ok:
            print("  ✅ PASS — duong GQA nap va tra dung nhom centroid.")
            if pd and pd["mean"] < 0:
                print(f"     Sq-70% thap hon {-pd['mean']:.2f} diem nhung KHONG co y nghia "
                      f"thong ke (p={pd['p']:.3f}, KTC95 chua 0).")
                print(f"     {pd['same']}/{pd['n']} mau cho prediction y het nhau.")
                print("     Muon ket luan chac thi tang so mau, dung doc con so nay nhu that.")
        else:
            if pd and catastrophic:
                print(f"  ❌ FAIL — tut {-pd['mean']:.2f} diem, vuot muc hong nang "
                      f"({args.max_drop}).")
            elif pd:
                print(f"  ❌ FAIL — kem hon CO Y NGHIA THONG KE: KTC95 "
                      f"[{pd['ci_lo']:+.2f}, {pd['ci_hi']:+.2f}] nam han duoi 0.")
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
