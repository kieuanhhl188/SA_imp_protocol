"""
analyze_entropy.py
==================
Phân tích layer entropy log để verify hypothesis Hướng 1.

Câu hỏi cần trả lời:
  1. Entropy có pattern rõ ràng giữa các layer không (variance đủ lớn)?
  2. Pattern có consistent across samples không?
  3. Pattern có khác giữa các dataset không?

Nếu (1) và (2) đều positive → green light cho Hướng 1.
Nếu (3) cũng positive → có thể có insight về dataset-specific allocation.

Usage:
    python analyze_entropy.py --log_dir output_gate_h1/baseline
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def load_entropy_logs(log_dir: str):
    """Load entropy log từ thư mục output."""
    log_path = os.path.join(log_dir, "entropy_log.npy")
    if not os.path.exists(log_path):
        # Fallback: tìm tất cả budgets_*.pt và compute back-entropy là không trực tiếp.
        # User cần chạy với --save_entropy_log
        raise FileNotFoundError(
            f"{log_path} không tồn tại. "
            "Chạy lại offline_clustering_v2.py với --save_entropy_log"
        )
    logs = np.load(log_path)  # [num_samples, num_layers]
    return logs


def plot_entropy_distribution(logs: np.ndarray, output_path: str = "entropy_plot.png"):
    """
    Plot 4 subplot:
      (a) Mean ± std entropy per layer
      (b) Heatmap [sample × layer]
      (c) Coefficient of variation per layer (consistency check)
      (d) Histogram of all entropy values
    """
    num_samples, num_layers = logs.shape
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Mean ± std per layer
    mean = logs.mean(axis=0)
    std = logs.std(axis=0)
    axes[0, 0].errorbar(range(num_layers), mean, yerr=std,
                        marker='o', capsize=3)
    axes[0, 0].set_xlabel("Layer index")
    axes[0, 0].set_ylabel("Attention entropy (nats)")
    axes[0, 0].set_title("(a) Mean entropy ± std per layer")
    axes[0, 0].grid(alpha=0.3)

    # (b) Heatmap
    im = axes[0, 1].imshow(logs, aspect='auto', cmap='viridis')
    axes[0, 1].set_xlabel("Layer index")
    axes[0, 1].set_ylabel("Sample index")
    axes[0, 1].set_title("(b) Entropy heatmap")
    plt.colorbar(im, ax=axes[0, 1])

    # (c) Coefficient of variation - consistency check
    cv = std / (mean + 1e-9)
    axes[1, 0].plot(range(num_layers), cv, marker='s', color='red')
    axes[1, 0].set_xlabel("Layer index")
    axes[1, 0].set_ylabel("CV = std/mean")
    axes[1, 0].set_title("(c) Coefficient of variation (lower = more consistent)")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].axhline(y=0.2, color='gray', linestyle='--', label='CV=0.2 (rule of thumb)')
    axes[1, 0].legend()

    # (d) Histogram
    axes[1, 1].hist(logs.flatten(), bins=50, edgecolor='black')
    axes[1, 1].set_xlabel("Entropy value")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("(d) Overall entropy distribution")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f">>> Saved plot to {output_path}")
    plt.close()


def print_summary(logs: np.ndarray):
    """In tóm tắt thống kê."""
    num_samples, num_layers = logs.shape

    print("\n" + "=" * 70)
    print("ENTROPY ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Num samples: {num_samples}")
    print(f"Num layers:  {num_layers}")

    mean = logs.mean(axis=0)
    std = logs.std(axis=0)

    print(f"\nMin entropy layer: {mean.argmin()} (value: {mean.min():.3f})")
    print(f"Max entropy layer: {mean.argmax()} (value: {mean.max():.3f})")
    print(f"Range:             {mean.max() - mean.min():.3f}")
    print(f"Mean entropy:      {mean.mean():.3f}")
    print(f"Spread (std/mean): {(std/mean).mean():.3f}")

    # Diagnostic: nếu range/mean < 0.1 → entropy quá uniform, Hướng 1 ít tác dụng
    diagnostic = (mean.max() - mean.min()) / mean.mean()
    print(f"\nDiagnostic ratio: {diagnostic:.3f}")
    if diagnostic < 0.1:
        print("⚠️  Entropy quá uniform giữa các layer.")
        print("   Hướng 1 có thể KHÔNG cho gain đáng kể.")
        print("   Consider: chuyển hướng hoặc tìm signal khác.")
    elif diagnostic < 0.3:
        print("⚠ Entropy có variance nhưng vừa phải.")
        print("   Hướng 1 có thể cho gain nhỏ (~1-2 điểm).")
        print("   Vẫn worth làm nhưng cần positioning cẩn thận.")
    else:
        print("✅ Entropy variance rõ rệt giữa các layer.")
        print("   Hướng 1 có potential cho gain đáng kể. GREEN LIGHT.")

    # Consistency check
    cv = std / (mean + 1e-9)
    print(f"\nAverage CV across layers: {cv.mean():.3f}")
    if cv.mean() < 0.2:
        print("✅ Entropy pattern consistent across samples → robust signal.")
    elif cv.mean() < 0.4:
        print("⚠ Entropy có noise vừa phải. Cần dùng calibration set lớn hơn.")
    else:
        print("⚠️  Entropy không stable across samples → khó dùng làm signal.")

    print("=" * 70 + "\n")


def compare_strategies(strategy_dirs: dict):
    """
    So sánh entropy log giữa các strategy khác nhau.

    Args:
        strategy_dirs: dict {'baseline': path, 'linear': path, ...}
    """
    for name, path in strategy_dirs.items():
        try:
            logs = load_entropy_logs(path)
            print(f"\n--- {name} ---")
            print_summary(logs)
        except FileNotFoundError as e:
            print(f"[SKIP] {name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Thư mục chứa entropy_log.npy")
    parser.add_argument("--plot_output", type=str, default="entropy_plot.png")
    args = parser.parse_args()

    logs = load_entropy_logs(args.log_dir)
    print_summary(logs)
    plot_entropy_distribution(logs, args.plot_output)
