"""
Adaptive layer-wise budget allocation for Squeezed Attention.

Hướng 1: Layer-wise Adaptive Cluster Budget
============================================
Quan sát: Attention entropy khác nhau giữa các layer. Layer có attention
concentrate (entropy thấp) cần ít cluster hơn vì top-k key đã rõ ràng;
layer có attention diffuse (entropy cao) cần nhiều cluster hơn để approximate
distribution.

Khác biệt với Wang & Gan SqueezeAttention (ICLR 2025):
- Họ: cosine similarity của hidden states, eviction-based
- Ta:  attention entropy, cluster-based (top of Squeezed Attention)

Author: <your name>
"""
import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional


# =====================================================================
# CORE: Attention Entropy Profiling
# =====================================================================

def compute_attention_entropy(
    queries: torch.Tensor,
    keys: torch.Tensor,
    observation_window: int = 100,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Tính attention entropy cho một layer.

    Entropy thấp -> attention concentrate -> dễ predict top-k -> ít cluster.
    Entropy cao -> attention diffuse -> cần nhiều cluster để approximate.

    Args:
        queries: [batch, num_heads, seq_len, head_dim] hoặc [num_heads, seq_len, head_dim]
        keys:    cùng shape với queries
        observation_window: số token cuối cùng dùng để tính entropy (mô phỏng user query)
        eps: tránh log(0)

    Returns:
        entropy: scalar tensor - entropy trung bình across heads của layer này.
                 Đơn vị: nats. Giá trị từ 0 (concentrate hoàn toàn) tới log(N) (uniform).
    """
    # Standardize shape: thêm batch dim nếu thiếu
    if queries.dim() == 3:
        queries = queries.unsqueeze(0)
        keys = keys.unsqueeze(0)

    B, H, S, D = queries.shape
    assert keys.shape == queries.shape, f"Q vs K shape mismatch: {queries.shape} vs {keys.shape}"

    # Lấy phần observation window cuối: đây là query simulate user input
    # Phần đầu (fixed context) làm key. Đây là setup của Squeezed Attention.
    obs_q = queries[:, :, -observation_window:, :]   # [B, H, obs, D]
    ctx_k = keys[:, :, :-observation_window, :]      # [B, H, S-obs, D]

    if ctx_k.shape[2] == 0:
        # Sequence quá ngắn, fallback an toàn
        return torch.tensor(0.0, device=queries.device)

    # Tính attention score: [B, H, obs, S-obs]
    # Note: chia sqrt(D) để giữ scale chuẩn của softmax attention
    scores = torch.matmul(obs_q, ctx_k.transpose(-2, -1)) / math.sqrt(D)
    attn = F.softmax(scores, dim=-1)  # [B, H, obs, S-obs]

    # Entropy mỗi query token: H = -Σ p log p
    entropy_per_query = -(attn * torch.log(attn + eps)).sum(dim=-1)  # [B, H, obs]

    # Trung bình across batch, heads, query tokens
    # Giữ thông tin per-head sẽ cần cho head-wise allocation (extension sau)
    layer_entropy = entropy_per_query.mean()  # scalar

    return layer_entropy


def profile_layer_entropies(
    all_queries_layers: List[torch.Tensor],
    all_keys_layers: List[torch.Tensor],
    observation_window: int = 100,
) -> torch.Tensor:
    """
    Profile entropy cho tất cả layer trong model.

    Args:
        all_queries_layers: list[L] tensors, mỗi tensor là Q của 1 layer
        all_keys_layers: list[L] tensors tương ứng K
        observation_window: như trên

    Returns:
        entropies: [L] tensor - entropy của mỗi layer
    """
    num_layers = len(all_queries_layers)
    entropies = torch.zeros(num_layers)

    for i in range(num_layers):
        q = all_queries_layers[i]
        k = all_keys_layers[i]
        entropies[i] = compute_attention_entropy(q, k, observation_window).cpu()

    return entropies


# =====================================================================
# CORE: Budget Allocation Strategies
# =====================================================================

def allocate_budget_by_entropy(
    layer_entropies: torch.Tensor,
    total_budget: int,
    min_budget_per_layer: int = 1,
    strategy: str = "linear",
    smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Phân bổ tổng budget cluster theo entropy mỗi layer.

    Strategy:
      "linear":   budget_i ∝ entropy_i  (entropy cao → nhiều cluster)
      "softmax":  budget_i ∝ softmax(entropy_i / T)
      "pyramid":  layer thấp được nhiều hơn (pyramid top-down)
      "inverse":  budget_i ∝ 1/entropy_i (chứng minh ngược lại, dùng ablation)

    Args:
        layer_entropies: [L] tensor
        total_budget: tổng số cluster trên toàn model
        min_budget_per_layer: tối thiểu mỗi layer (đảm bảo không có layer = 0)
        strategy: chiến lược phân bổ
        smoothing: tránh phân bổ quá lệch khi entropy chênh lớn

    Returns:
        budgets: [L] long tensor - số cluster cho mỗi layer, sum = total_budget
    """
    L = len(layer_entropies)
    assert total_budget >= L * min_budget_per_layer, (
        f"total_budget={total_budget} quá nhỏ so với L={L}, min={min_budget_per_layer}"
    )

    e = layer_entropies.float().clone()

    # Smoothing để tránh phân bổ quá cực đoan
    e = e + smoothing * e.mean()

    if strategy == "linear":
        weights = e / (e.sum() + 1e-9)
    elif strategy == "softmax":
        T = e.std() + 1e-6
        weights = F.softmax(e / T, dim=0)
    elif strategy == "pyramid":
        # Bias về layer thấp - inspired by PyramidKV
        layer_idx = torch.arange(L, dtype=torch.float32)
        pyramid_w = (L - layer_idx) / L
        combined = e * pyramid_w
        weights = combined / (combined.sum() + 1e-9)
    elif strategy == "inverse":
        # Negative control: ngược lại linear
        inv_e = 1.0 / (e + 1e-3)
        weights = inv_e / (inv_e.sum() + 1e-9)
    elif strategy == "uniform":
        # Baseline (giống Squeezed Attention gốc)
        weights = torch.ones(L) / L
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Reserve min budget per layer
    reserved = L * min_budget_per_layer
    remaining = total_budget - reserved

    budgets = torch.zeros(L, dtype=torch.long)
    budgets += min_budget_per_layer

    # Phân bổ phần còn lại theo weight
    extra = (weights * remaining).long()
    budgets += extra

    # Sửa lỗi rounding: ép sum đúng bằng total_budget
    diff = total_budget - budgets.sum().item()
    if diff > 0:
        # Phân vào các layer có weight cao nhất
        top_layers = torch.argsort(weights, descending=True)[:diff]
        budgets[top_layers] += 1
    elif diff < 0:
        # Trừ bớt từ các layer có weight thấp nhất (nhưng giữ min)
        bottom_layers = torch.argsort(weights, descending=False)
        idx = 0
        while diff < 0 and idx < L:
            l = bottom_layers[idx].item()
            if budgets[l] > min_budget_per_layer:
                budgets[l] -= 1
                diff += 1
            idx += 1

    assert budgets.sum().item() == total_budget, (
        f"Sum mismatch: {budgets.sum().item()} vs {total_budget}"
    )
    assert (budgets >= min_budget_per_layer).all(), "Some layer below min budget"

    return budgets


# =====================================================================
# UTILITY: Convert percentage budget to per-layer count
# =====================================================================

def compute_total_budget(
    shared_prefix_length: int,
    observation_window: int,
    percent_clusters: float,
    num_layers: int,
) -> int:
    """
    Tính total budget từ percentage, giống logic trong offline_clustering.py gốc.

    Trong Squeezed Attention gốc: num_centroids = percent * (sp_len - obs_window)
    Per-layer. Khi adaptive, ta nhân với num_layers để có tổng pool, rồi phân
    bổ lại.
    """
    per_layer_default = max(1, int((percent_clusters / 100.0) * (shared_prefix_length - observation_window)))
    total_budget = per_layer_default * num_layers
    return total_budget


# =====================================================================
# DIAGNOSTICS
# =====================================================================

def print_budget_summary(entropies: torch.Tensor, budgets: torch.Tensor):
    """In tóm tắt phân bổ để verify."""
    print("\n" + "=" * 70)
    print(f"{'Layer':<8} {'Entropy':<12} {'Budget':<10} {'% of total':<12}")
    print("-" * 70)
    total = budgets.sum().item()
    for i, (e, b) in enumerate(zip(entropies, budgets)):
        pct = 100.0 * b.item() / total
        print(f"{i:<8} {e.item():<12.4f} {b.item():<10} {pct:<12.2f}")
    print("-" * 70)
    print(f"Total: {total} | Mean entropy: {entropies.mean():.4f} | Std: {entropies.std():.4f}")
    print("=" * 70 + "\n")
