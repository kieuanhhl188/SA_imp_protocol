#!/usr/bin/env python
"""
check_gate.py — Gate của Phase 0.

So kết quả reproduce với Table 2 của bài (Hooper et al., ACL 2025).
Protocol: nếu lệch > 0.3 điểm thì environment sai, phải sửa trước khi làm gì khác.

Usage:
    # sau khi chạy scripts/phase0_gate.sh
    python scripts/check_gate.py --pred_dir LongBench/pred --model longchat-v1.5-7b-32k

    # chỉ định config cụ thể
    python scripts/check_gate.py --pred_dir LongBench/pred \\
        --model longchat-v1.5-7b-32k --configs "All KV" "Sq-70%"

    # nới tolerance (ví dụ khi chạy subset)
    python scripts/check_gate.py --tolerance 1.0 ...

Exit code: 0 nếu tất cả config kiểm tra được đều PASS, 1 nếu có FAIL.
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
DEFAULT_REF = os.path.join(HERE, "reference_table2.json")

# Ánh xạ tên config trong Table 2 -> thư mục pred/ mà eval.py sinh ra.
# eval.py:
#   baseline           -> pred/{model}_baseline/
#   single-level       -> pred/{model}_PC{percent_clusters}_PERC{percentile}/
#   hierarchical       -> pred/{model}_PC1_{pc}_PERC1_{perc}_PC2_{pc2}_PERC2_{perc_lower}_lookup/
CONFIG_TO_DIR = {
    "All KV":   "{model}_baseline",
    "Sq-70%":   "{model}_PC5_PERC0.7",
    "Sq-80%":   "{model}_PC5_PERC0.8",
    "Sq-90%":   "{model}_PC5_PERC0.9",
    "H-Sq-90%": "{model}_PC1_1_PERC1_0.9_PC2_5_PERC2_0.5_lookup",
}

TASKS = ["lcc", "repobench-p"]


def load_result(pred_dir, dirname):
    path = os.path.join(pred_dir, dirname, "result.json")
    if not os.path.exists(path):
        return None, path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), path


def env_summary(env_path):
    """Tóm tắt env_record.json thành vài dòng để nhúng vào log."""
    if not env_path or not os.path.exists(env_path):
        return None
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            env = json.load(f)
    except Exception:
        return None
    pkgs = env.get("packages", {})
    def v(name):
        p = pkgs.get(name) or {}
        return p.get("version") if p.get("installed") else "THIẾU"
    return {
        "timestamp": env.get("timestamp"),
        "gpu": env.get("nvidia_smi", "").replace("\n", " | "),
        "torch": v("torch"),
        "transformers": v("transformers"),
        "triton": v("triton"),
        "flash_attn": v("flash_attn"),
        "cuml": v("cuml"),
        "is_fork": env.get("checks", {}).get("transformers_is_repo_fork"),
        "cuda_visible": (env.get("env_vars") or {}).get("CUDA_VISIBLE_DEVICES"),
        "seed": env.get("seed"),
    }


MD_HEADER = """# Nhật ký thí nghiệm — Structure-Aware Squeezed Attention

File này là **nguồn sự thật duy nhất** cho tiến độ và kết quả. Mỗi lần chạy gate,
`scripts/check_gate.py --log_md` tự phụ lục một mục vào cuối file. Ghi chú tay thì
thêm vào ngay dưới mục tương ứng.

Quy ước: mốc tham chiếu là Table 2, Hooper et al., ACL 2025 (`2025.acl-long.1568`).
Tolerance mặc định ±0.3 điểm theo protocol.

## Trạng thái các phase

| Phase | Nội dung | Trạng thái | Ghi chú |
|---|---|---|---|
| 0 | Môi trường + tái lập baseline SA | ⬜ chưa chạy | code+config đã xong, chờ GPU |
| 1 | Chuẩn bị dữ liệu code | ⬜ chưa xong | thiếu offset persist, CrossCodeEval, Qwen |
| 2 | Structure-aware clustering | ⬜ chưa làm | hard boundary chưa implement |
| 3 | Symbol / def-use signal | ⬜ chưa làm | |
| 4 | Incremental re-clustering | ⬜ chưa làm | |
| 5 | C2 retrieval quality | ⬜ chưa làm | chạy TRƯỚC Phase 6 |
| 6 | C1 accuracy@budget | ⬜ chưa làm | |
| 7 | C3 + phân tích | ⬜ chưa làm | |

## Nơi dữ liệu được ghi

| Loại | Đường dẫn |
|---|---|
| Centroid / label / threshold | `fixed-prompt-clusters/<dataset>/*.pt` |
| Prediction thô | `LongBench/pred/<config>/<dataset>.jsonl` |
| Điểm số | `LongBench/pred/<config>/result.json` |
| Môi trường | `phase0_results/env_record.json` + `_pip_freeze.txt` |
| Console log đầy đủ | `phase0_results/logs/<timestamp>_*.log` |
| Tổng hợp (file này) | `EXPERIMENT_LOG.md` |

---

## Lịch sử chạy

<!-- check_gate.py phụ lục bên dưới. Không xoá dòng này. -->
"""


def append_md_log(md_path, args, rows, n_pass, n_fail, n_skip, verdict, env):
    """Phụ lục một mục vào file nhật ký markdown."""
    from datetime import datetime

    os.makedirs(os.path.dirname(os.path.abspath(md_path)) or ".", exist_ok=True)
    if not os.path.exists(md_path):
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(MD_HEADER)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    badge = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "NO_DATA": "⬜ chưa có dữ liệu"}[verdict]

    lines = []
    lines.append("")
    lines.append(f"### {ts} — Phase 0 gate — {args.model} — {badge}")
    lines.append("")
    if args.run_note:
        lines.append(f"> {args.run_note}")
        lines.append("")

    lines.append(f"- Tolerance: ±{args.tolerance}")
    lines.append(f"- pred_dir: `{args.pred_dir}`")
    if env:
        lines.append(f"- GPU: `{env['gpu']}`  (CUDA_VISIBLE_DEVICES=`{env['cuda_visible']}`)")
        fork = "đúng fork" if env["is_fork"] else "**KHÔNG phải fork trong repo**"
        lines.append(f"- transformers `{env['transformers']}` ({fork}) | torch `{env['torch']}` "
                     f"| triton `{env['triton']}` | flash_attn `{env['flash_attn']}` | cuml `{env['cuml']}`")
        lines.append(f"- seed: `{env['seed']}`")
    else:
        lines.append("- env: *(chưa có `env_record.json`, chạy `scripts/record_env.py`)*")
    lines.append("")

    lines.append("| Config | Task | Expected | Actual | Delta | Status |")
    lines.append("|---|---|---:|---:|---:|---|")
    for cfg, task, exp, act, delta, status in rows:
        exp_s = f"{exp:.2f}" if isinstance(exp, float) else "-"
        act_s = f"{act:.2f}" if isinstance(act, float) else "-"
        del_s = f"{delta:+.2f}" if isinstance(delta, float) else "-"
        mark = {"PASS": "✅", "FAIL": "❌"}.get(status, "⬜")
        lines.append(f"| {cfg} | {task} | {exp_s} | {act_s} | {del_s} | {mark} {status} |")
    lines.append("")
    lines.append(f"**PASS={n_pass} · FAIL={n_fail} · SKIP={n_skip}**")
    lines.append("")
    if args.console_log:
        lines.append(f"Console log đầy đủ: [`{args.console_log}`]({args.console_log})")
        lines.append("")
    lines.append("<!-- ghi chú tay bên dưới -->")
    lines.append("")

    with open(md_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return md_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default=os.path.join(REPO_ROOT, "LongBench", "pred"),
                    help="thư mục pred/ do LongBench/eval.py sinh ra")
    ap.add_argument("--model", default="longchat-v1.5-7b-32k")
    ap.add_argument("--reference", default=DEFAULT_REF)
    ap.add_argument("--tolerance", type=float, default=0.3,
                    help="ngưỡng lệch tuyệt đối cho phép (protocol: 0.3)")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="chỉ kiểm tra các config này; mặc định kiểm tra mọi config tìm thấy")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--log_md", default=os.path.join(REPO_ROOT, "EXPERIMENT_LOG.md"),
                    help="file nhật ký markdown để phụ lục kết quả; '' để tắt")
    # Default PHẢI theo $SQA_RESULT_DIR: trên pod thư mục này nằm ở /workspace/phase0_results,
    # không phải trong repo. Bản cũ hard-code repo-relative nên khi chạy tay (không qua
    # phase0_gate.sh — script đó truyền --env_record tường minh) thì luôn báo "chưa có
    # env_record.json" dù file có thật → nhật ký ghi sai môi trường thành "không có".
    ap.add_argument("--env_record",
                    default=os.path.join(
                        os.environ.get("SQA_RESULT_DIR",
                                       os.path.join(REPO_ROOT, "phase0_results")),
                        "env_record.json"),
                    help="env_record.json để nhúng thông tin môi trường vào nhật ký "
                         "(mặc định $SQA_RESULT_DIR/env_record.json)")
    ap.add_argument("--console_log", default=None,
                    help="đường dẫn console log để trỏ tới từ nhật ký")
    ap.add_argument("--run_note", default="", help="ghi chú tự do cho lần chạy này")
    args = ap.parse_args()

    with open(args.reference, "r", encoding="utf-8") as f:
        ref_all = json.load(f)

    if args.model not in ref_all:
        avail = [k for k in ref_all if not k.startswith("_")]
        print(f"[ERROR] Không có số tham chiếu cho model '{args.model}'. Có: {avail}")
        return 2
    ref_model = ref_all[args.model]

    configs = args.configs if args.configs else list(CONFIG_TO_DIR.keys())

    rows = []
    n_pass = n_fail = n_skip = 0

    for cfg in configs:
        if cfg not in CONFIG_TO_DIR:
            print(f"[WARN] config '{cfg}' không có mapping thư mục, bỏ qua.")
            continue
        if cfg not in ref_model:
            print(f"[WARN] config '{cfg}' không có trong reference, bỏ qua.")
            continue

        dirname = CONFIG_TO_DIR[cfg].format(model=args.model)
        got, path = load_result(args.pred_dir, dirname)
        if got is None:
            rows.append((cfg, "-", None, None, None, "SKIP (chưa có result.json)"))
            n_skip += 1
            continue

        for task in args.tasks:
            expected = ref_model[cfg].get(task)
            if expected is None:
                continue
            if task not in got:
                rows.append((cfg, task, expected, None, None, "SKIP (thiếu task)"))
                n_skip += 1
                continue
            actual = got[task]
            delta = actual - expected
            ok = abs(delta) <= args.tolerance
            rows.append((cfg, task, expected, actual, delta, "PASS" if ok else "FAIL"))
            if ok:
                n_pass += 1
            else:
                n_fail += 1

    # in bảng
    print()
    print("=" * 78)
    print(f"  PHASE 0 GATE — model: {args.model}   tolerance: ±{args.tolerance}")
    print(f"  reference: Table 2, Hooper et al. ACL 2025")
    print("=" * 78)
    print(f"{'Config':<12} {'Task':<14} {'Expected':>9} {'Actual':>9} {'Delta':>8}  Status")
    print("-" * 78)
    for cfg, task, exp, act, delta, status in rows:
        exp_s = f"{exp:.2f}" if isinstance(exp, float) else "-"
        act_s = f"{act:.2f}" if isinstance(act, float) else "-"
        del_s = f"{delta:+.2f}" if isinstance(delta, float) else "-"
        print(f"{cfg:<12} {task:<14} {exp_s:>9} {act_s:>9} {del_s:>8}  {status}")
    print("-" * 78)
    print(f"  PASS={n_pass}  FAIL={n_fail}  SKIP={n_skip}")

    # phụ lục nhật ký markdown
    if n_fail > 0:
        verdict = "FAIL"
    elif n_pass == 0:
        verdict = "NO_DATA"
    else:
        verdict = "PASS"

    if args.log_md:
        try:
            md = append_md_log(args.log_md, args, rows, n_pass, n_fail, n_skip,
                               verdict, env_summary(args.env_record))
            print(f"\n  >>> Đã ghi nhật ký: {md}")
        except Exception as e:
            print(f"\n  [WARN] Không ghi được nhật ký {args.log_md}: {e}")

    if n_fail > 0:
        print()
        print("  >>> GATE FAIL. Theo protocol: environment sai, KHÔNG chạy tiếp Phase 1/2.")
        print("      Kiểm tra theo thứ tự:")
        print("        1. transformers đang dùng có phải bản fork trong repo không?")
        print("           python -c \"import transformers; print(transformers.__file__, transformers.__version__)\"")
        print("           -> phải trỏ vào <repo>/transformers/src/... và version 4.40.0.dev0")
        print("        2. max_length trong LongBench/config/model2maxlen.json = 31500?")
        print("        3. observation_window = 100 ở CẢ offline clustering lẫn pred.py (--obs_window)?")
        print("        4. Đã chạy đủ toàn bộ sample của task chưa (không cắt subset)?")
        print("        5. --percentile của pred.py có khớp mức sparsity không (Sq-70% -> 0.7)?")
        return 1

    if n_pass == 0:
        print()
        print("  >>> Chưa có kết quả nào để so. Chạy scripts/phase0_gate.sh trước.")
        return 1

    print()
    print("  >>> GATE PASS. Được phép sang Phase 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
