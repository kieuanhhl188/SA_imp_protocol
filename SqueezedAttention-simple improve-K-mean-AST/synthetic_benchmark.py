"""
Synthetic benchmark cho value-aware retrieval.

Mục tiêu: dùng dữ liệu nhân tạo có kiểm soát để verify rằng value-aware 
thực sự cải thiện attention output approximation khi có value diversity cao.

Chạy được trên CPU, không cần model thực.
"""

import torch
import torch.nn.functional as F
import sys

from value_aware_clustering import (
    value_aware_kmeans,
    normalize_value_variance,
    _kmeans_cosine,
)
from value_aware_retrieval import (
    squeezed_attention_forward,
    baseline_full_attention,
    calibrate_threshold,
    compute_base_scores,
)


def synthetic_kv(H=8, N=512, D=64, num_groups=20, value_diversity=1.0, seed=0):
    """
    Tạo synthetic K, V có cấu trúc cluster rõ.
    
    Args:
        num_groups: số "topic" trong context (mỗi topic là một anchor key)
        value_diversity: 0 = values đồng nhất trong topic, 1.0 = values random hoàn toàn
    """
    torch.manual_seed(seed)
    # Anchors cho mỗi group, mỗi head
    anchors_k = F.normalize(torch.randn(H, num_groups, D), dim=-1)
    anchors_v = torch.randn(H, num_groups, D)

    # Phân bố tokens vào groups (uneven)
    group_assignments = torch.randint(0, num_groups, (H, N))

    # Keys = anchor + small noise
    keys = torch.zeros(H, N, D)
    values = torch.zeros(H, N, D)
    for h in range(H):
        for n in range(N):
            g = group_assignments[h, n].item()
            keys[h, n] = anchors_k[h, g] + 0.05 * torch.randn(D)
            keys[h, n] = F.normalize(keys[h, n], dim=-1)
            # Values: anchor + diversity * noise
            values[h, n] = anchors_v[h, g] + value_diversity * torch.randn(D)

    return keys, values, group_assignments


def run_one_trial(diversity, gamma, beta, num_queries=20, target_sparsity=0.85, verbose=False):
    """Chạy 1 trial: setup KV, cluster cả 2 cách, so sánh với full attention."""
    H, N, D = 8, 512, 64
    K = 32  # số clusters
    keys, values, _ = synthetic_kv(
        H=H, N=N, D=D, num_groups=15, value_diversity=diversity, seed=42
    )

    # Tạo queries: mỗi query gần với 1 anchor (mô phỏng query "tìm đến" 1 cluster)
    torch.manual_seed(99)
    queries = []
    for q in range(num_queries):
        # query = nhiễu xung quanh 1 key bất kì
        target_n = torch.randint(0, N, (1,)).item()
        q_vec = keys[:, target_n, :] + 0.1 * torch.randn(H, D)
        q_vec = F.normalize(q_vec, dim=-1)
        queries.append(q_vec)
    queries = torch.stack(queries)  # (num_queries, H, D)

    # ----- Cluster baseline (key-only, beta=0) -----
    kc_ko, _, lbl_ko, _ = value_aware_kmeans(
        keys, values, K, alpha=1.0, beta=0.0, num_iters=15
    )
    sizes_ko = torch.zeros(H, K)
    sizes_ko.scatter_add_(1, lbl_ko, torch.ones_like(lbl_ko, dtype=torch.float))
    nvar_ko = torch.zeros(H, K)  # tắt boost cho baseline

    # ----- Cluster value-aware -----
    kc_va, vc_va, lbl_va, vvar_va = value_aware_kmeans(
        keys, values, K, alpha=1.0, beta=beta, num_iters=15
    )
    sizes_va = torch.zeros(H, K)
    sizes_va.scatter_add_(1, lbl_va, torch.ones_like(lbl_va, dtype=torch.float))
    nvar_va = normalize_value_variance({0: vvar_va})[0]

    # ----- Calibrate thresholds (cùng target sparsity) -----
    T_ko = calibrate_threshold(
        queries, kc_ko, sizes_ko, nvar_ko, lbl_ko,
        target_sparsity=target_sparsity, gamma=0.0,
        num_threshold_search=80,
    )
    T_va = calibrate_threshold(
        queries, kc_va, sizes_va, nvar_va, lbl_va,
        target_sparsity=target_sparsity, gamma=gamma,
        num_threshold_search=80,
    )

    # ----- Đo trên test queries (khác calib queries để tránh overfit threshold) -----
    torch.manual_seed(2024)
    test_queries = []
    for q in range(num_queries):
        target_n = torch.randint(0, N, (1,)).item()
        q_vec = keys[:, target_n, :] + 0.1 * torch.randn(H, D)
        q_vec = F.normalize(q_vec, dim=-1)
        test_queries.append(q_vec)
    test_queries = torch.stack(test_queries)

    # Tính metrics
    ko_cos, ko_mse, ko_budget = [], [], []
    va_cos, va_mse, va_budget = [], [], []

    for q in test_queries:
        ref = baseline_full_attention(q, keys, values)

        ko_out, ko_info = squeezed_attention_forward(
            q, keys, values, kc_ko, sizes_ko, nvar_ko, lbl_ko,
            threshold=T_ko, gamma=0.0,
        )
        va_out, va_info = squeezed_attention_forward(
            q, keys, values, kc_va, sizes_va, nvar_va, lbl_va,
            threshold=T_va, gamma=gamma,
        )

        ko_cos.append(F.cosine_similarity(ko_out.flatten(), ref.flatten(), dim=0).item())
        ko_mse.append(F.mse_loss(ko_out, ref).item())
        ko_budget.append(ko_info["kv_budget"])

        va_cos.append(F.cosine_similarity(va_out.flatten(), ref.flatten(), dim=0).item())
        va_mse.append(F.mse_loss(va_out, ref).item())
        va_budget.append(va_info["kv_budget"])

    return {
        "ko_cos": sum(ko_cos) / len(ko_cos),
        "ko_mse": sum(ko_mse) / len(ko_mse),
        "ko_budget": sum(ko_budget) / len(ko_budget),
        "va_cos": sum(va_cos) / len(va_cos),
        "va_mse": sum(va_mse) / len(va_mse),
        "va_budget": sum(va_budget) / len(va_budget),
    }


def main():
    print("=" * 80)
    print("Synthetic benchmark: Value-aware vs Key-only Squeezed Attention")
    print("=" * 80)
    print()
    print("Setup: 8 heads, 512 keys, 32 clusters, sparsity target 85% (giữ 15%)")
    print("Test scenarios với value_diversity khác nhau.")
    print()

    print(f"{'Diversity':>10s} | {'Method':>12s} | {'CosSim':>8s} | {'MSE':>10s} | {'Budget':>8s}")
    print("-" * 80)

    results = []
    for diversity in [0.1, 0.5, 1.0, 2.0]:
        r = run_one_trial(diversity=diversity, gamma=0.3, beta=0.5)
        results.append((diversity, r))
        print(
            f"{diversity:>10.2f} | {'key-only':>12s} | "
            f"{r['ko_cos']:>8.4f} | {r['ko_mse']:>10.6f} | {r['ko_budget']*100:>7.2f}%"
        )
        print(
            f"{'':>10s} | {'value-aware':>12s} | "
            f"{r['va_cos']:>8.4f} | {r['va_mse']:>10.6f} | {r['va_budget']*100:>7.2f}%"
        )
        delta_cos = (r['va_cos'] - r['ko_cos']) * 100
        delta_mse_pct = (r['ko_mse'] - r['va_mse']) / max(r['ko_mse'], 1e-9) * 100
        print(
            f"{'':>10s} | {'Δ':>12s} | "
            f"{delta_cos:>+7.3f}pp | {delta_mse_pct:>+9.2f}% | "
            f"{(r['va_budget']-r['ko_budget'])*100:>+7.2f}%"
        )
        print()

    print("=" * 80)
    print("Diễn giải:")
    print("- Diversity thấp (0.1): values trong cùng key-cluster đã đồng nhất")
    print("  -> value-aware không có nhiều lợi thế, khác biệt nhỏ")
    print("- Diversity cao (1.0+): values trong cùng key-cluster rất khác nhau")
    print("  -> value-aware nên cải thiện rõ (Δ CosSim dương, Δ MSE âm)")
    print()
    print("Δ CosSim dương = value-aware tốt hơn key-only.")
    print("Δ Budget gần 0 = đảm bảo so sánh công bằng cùng KV cost.")
    print("=" * 80)

    # Verify ý tưởng đúng: ở diversity cao, value-aware nên thắng
    high_div_result = results[-1][1]
    if high_div_result['va_cos'] > high_div_result['ko_cos']:
        print("\n✓ VERIFIED: Value-aware cải thiện cos_sim ở diversity cao.")
    else:
        print("\n⚠ KHÔNG VERIFIED ở synthetic case này. Có thể cần tune γ, β.")
        print("  (Trên model thực, kết quả thường tốt hơn vì K, V có ngữ nghĩa)")


if __name__ == "__main__":
    main()
