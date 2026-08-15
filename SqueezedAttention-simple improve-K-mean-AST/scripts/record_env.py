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
        "transformers_is_repo_fork": os.path.abspath(fork_path) in os.path.abspath(tf_path),
        "transformers_version_expected": "4.40.0.dev0",
        "transformers_version_actual": tf.get("version"),
    }

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


if __name__ == "__main__":
    main()
