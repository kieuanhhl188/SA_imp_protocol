#!/usr/bin/env python
"""
record_env.py — ghi lại môi trường thí nghiệm.

Protocol Phase 0 yêu cầu: "Ghi lại version transformers/triton, GPU (nên cố định
1x H100 hoặc A100-80G cho mọi lần đo latency), seed."

Chạy TRƯỚC mỗi lần đo và commit file output kèm kết quả.

Usage:
    python scripts/record_env.py --out phase0_results/env_record.json
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime

# Console Windows mac dinh cp1252 -> tieng Viet co dau se lam crash print().
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# Package cần ghi version. transformers phải là bản FORK trong repo.
WATCH = [
    "torch", "transformers", "triton", "flash_attn", "datasets",
    "numpy", "sklearn", "cupy", "cuml", "tree_sitter", "accelerate",
]


def pkg_info(name):
    try:
        mod = __import__(name)
    except Exception as e:
        return {"installed": False, "error": f"{type(e).__name__}: {e}"}
    info = {
        "installed": True,
        "version": getattr(mod, "__version__", None),
        "path": getattr(mod, "__file__", None),
    }
    return info


def run(cmd):
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return out.stdout.strip() or out.stderr.strip()
    except Exception as e:
        return f"<failed: {e}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "phase0_results", "env_record.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--note", default="", help="ghi chú tự do, ví dụ 'phase0 gate lcc+rb'")
    ap.add_argument("--strict", action="store_true",
                    help="thoát lỗi nếu stack không đúng cấu hình Phase 0")
    args = ap.parse_args()

    rec = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "note": args.note,
        "seed": args.seed,
        "python": sys.version,
        "platform": platform.platform(),
        "repo_root": REPO_ROOT,
        "env_vars": {
            k: os.environ.get(k)
            for k in ["CUDA_VISIBLE_DEVICES", "CONDA_DEFAULT_ENV", "PYTORCH_CUDA_ALLOC_CONF"]
        },
        "packages": {name: pkg_info(name) for name in WATCH},
        "experiment_config": {
            "model": "longchat-v1.5-7b-32k",
            "datasets": ["lcc", "repobench-p"],
            "percent_clusters": 5,
            "percent_clusters_l1": 1,
            "percent_clusters_l2": 5,
            "observation_window": 100,
            "percentile_gate": 0.7,
            "percentile_lower": 0.5,
            "max_length": 31500,
            "seed": args.seed,
        },
    }

    # GPU
    rec["nvidia_smi"] = run("nvidia-smi --query-gpu=index,name,memory.total,driver_version "
                            "--format=csv,noheader")
    rec["nvcc"] = run("nvcc --version")

    # torch/cuda chi tiết
    try:
        import torch
        rec["torch_cuda"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as e:
        rec["torch_cuda"] = {"error": str(e)}

    # KIỂM TRA QUAN TRỌNG: transformers phải là bản fork trong repo
    tf = rec["packages"].get("transformers", {})
    tf_path = tf.get("path") or ""
    fork_path = os.path.join(REPO_ROOT, "transformers")
    rec["checks"] = {
        "transformers_is_repo_fork": os.path.realpath(fork_path) in os.path.realpath(tf_path),
        "transformers_version_expected": "4.40.0.dev0",
        "transformers_version_actual": tf.get("version"),
    }

    torch_info = rec["packages"].get("torch", {})
    triton_info = rec["packages"].get("triton", {})
    cuml_info = rec["packages"].get("cuml", {})
    rec["checks"].update({
        "python_3_10": sys.version_info[:2] == (3, 10),
        "torch_expected": torch_info.get("version") == "2.3.1+cu121",
        "triton_expected": triton_info.get("version") == "2.3.1",
        "cuml_expected": str(cuml_info.get("version", "")).replace(".", "").startswith("2406"),
    })
    try:
        import torch
        rec["checks"].update({
            "cuda_available": torch.cuda.is_available(),
            "one_gpu_visible": torch.cuda.device_count() == 1,
            "cuda_version_expected": torch.version.cuda == "12.1",
            "cuda_visible_is_zero": os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        })
    except Exception:
        rec["checks"].update({
            "cuda_available": False,
            "one_gpu_visible": False,
            "cuda_version_expected": False,
            "cuda_visible_is_zero": False,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    # pip freeze riêng ra file txt
    freeze_path = os.path.splitext(args.out)[0] + "_pip_freeze.txt"
    with open(freeze_path, "w", encoding="utf-8") as f:
        f.write(run(f'"{sys.executable}" -m pip freeze'))

    print(f">>> Đã ghi: {args.out}")
    print(f">>> Đã ghi: {freeze_path}")
    print()
    for name in WATCH:
        p = rec["packages"][name]
        if p["installed"]:
            print(f"  {name:<14} {str(p['version']):<16}")
        else:
            print(f"  {name:<14} THIẾU  ({p['error'][:60]})")
    print()
    print(f"  GPU: {rec['nvidia_smi']}")
    print()
    c = rec["checks"]
    if not c["transformers_is_repo_fork"]:
        print("  [!!] transformers KHÔNG phải bản fork trong repo:")
        print(f"       {tf_path}")
        print("       -> cd transformers && pip install -e .")
    if c["transformers_version_actual"] != c["transformers_version_expected"]:
        print(f"  [!!] transformers version {c['transformers_version_actual']} "
              f"!= {c['transformers_version_expected']} (khả năng đã bị LongBench/requirements.txt ghi đè)")

    if args.strict:
        required = [
            "transformers_is_repo_fork", "python_3_10", "torch_expected",
            "triton_expected", "cuml_expected", "cuda_available",
            "one_gpu_visible", "cuda_version_expected", "cuda_visible_is_zero",
        ]
        failed = [name for name in required if not rec["checks"].get(name, False)]
        if failed:
            print(f"  [!!] STRICT ENV FAIL: {', '.join(failed)}")
            sys.exit(1)


if __name__ == "__main__":
    main()
