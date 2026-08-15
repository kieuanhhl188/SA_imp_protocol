"""
Online Value-Aware Retrieval cho Squeezed Attention.

Module này thay thế bước "centroid lookup" gốc khi inference:
- Tính S_i như cũ
- Boost theo value variance đã precompute
- Chọn các cluster có S_tilde_i > threshold

Ngoài ra, tính cả 'effective threshold' bù lại để giữ KV budget không tăng quá nhiều
do tác động của boost.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_base_scores(
    query: torch.Tensor,        # (H, D)  hoặc (B, H, D) - query token
    key_centroids: torch.Tensor,  # (H, K, D)
    cluster_sizes: torch.Tensor,  # (H, K)  - N_j: số keys trong mỗi cluster
) -> torch.Tensor:
    """
    Tính S_i theo công thức (1) trong paper:
        S_i = exp(q · C_i^T) / sum_j ( N_j * exp(q · C_j^T) )

    Returns:
        S: (H, K)
    """
    # Đảm bảo shape: (H, D)
    if query.dim() == 3:
        query = query.squeeze(0)  # (H, D)

    # Dot product q với centroids: (H, K)
    logits = torch.einsum("hd,hkd->hk", query, key_centroids)

    # Trừ max để stable (như Softmax stable)
    logits_max = logits.max(dim=-1, keepdim=True).values
    logits_shifted = logits - logits_max

    exp_logits = torch.exp(logits_shifted)  # (H, K)
    weighted = cluster_sizes * exp_logits   # N_j * exp(...)
    denom = weighted.sum(dim=-1, keepdim=True)  # (H, 1)

    S = exp_logits / (denom + 1e-12)
    return S


def value_aware_retrieve(
    query: torch.Tensor,            # (H, D)
    key_centroids: torch.Tensor,    # (H, K, D)
    cluster_sizes: torch.Tensor,    # (H, K)
    normalized_variance: torch.Tensor,  # (H, K)
    threshold: float,
    gamma: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trả về mask cluster và scores cho retrieval.
    
    Returns:
        cluster_mask: (H, K) bool - True nếu cluster được chọn
        adjusted_scores: (H, K) - dùng cho debug/logging
    """
    S = compute_base_scores(query, key_centroids, cluster_sizes)
    S_adjusted = S * (1.0 + gamma * normalized_variance)
    cluster_mask = S_adjusted > threshold
    return cluster_mask, S_adjusted


def keys_mask_from_clusters(
    cluster_mask: torch.Tensor,  # (H, K)
    labels: torch.Tensor,        # (H, N) - mỗi key thuộc cluster nào
) -> torch.Tensor:
    """
    Convert cluster mask -> per-token mask để load đúng các keys/values.
    
    Returns:
        key_mask: (H, N) bool
    """
    # gather: với mỗi (h, n), lấy cluster_mask[h, labels[h,n]]
    H, K = cluster_mask.shape
    # cluster_mask: (H, K) -> (H, K) bool
    # labels: (H, N) int64
    # Output: (H, N) bool
    key_mask = torch.gather(cluster_mask, 1, labels)
    return key_mask


def calibrate_threshold(
    queries_calib: torch.Tensor,         # (Q, H, D)
    key_centroids: torch.Tensor,         # (H, K, D)
    cluster_sizes: torch.Tensor,         # (H, K)
    normalized_variance: torch.Tensor,   # (H, K)
    labels: torch.Tensor,                # (H, N)
    target_sparsity: float = 0.9,
    gamma: float = 0.3,
    num_threshold_search: int = 50,
) -> float:
    """
    Tìm threshold global sao cho phần trăm keys bị prune ~= target_sparsity.
    
    Args:
        queries_calib: query tokens dùng để calibrate (typically 100 tokens cuối)
        target_sparsity: % keys muốn drop. 0.9 = giữ 10%
    
    Returns:
        threshold T thỏa mãn % keys giữ ≈ 1 - target_sparsity
    """
    Q = queries_calib.shape[0]
    H, N = labels.shape

    # Tính trung bình adjusted scores qua các query calib
    accumulated_scores = torch.zeros_like(cluster_sizes)
    for q_idx in range(Q):
        S = compute_base_scores(queries_calib[q_idx], key_centroids, cluster_sizes)
        S_adj = S * (1.0 + gamma * normalized_variance)
        accumulated_scores += S_adj
    avg_scores = accumulated_scores / Q  # (H, K)

    # Binary search threshold
    target_keep_ratio = 1.0 - target_sparsity
    # Sort flatten các điểm có thể là threshold
    # Để chính xác cần tính: với threshold T, % keys giữ = sum_{cluster: S>T} N_cluster / N_total

    # Cách đơn giản: tạo điểm threshold theo distribution của avg_scores
    # rồi check tỉ lệ
    candidates = torch.linspace(
        avg_scores.min().item(),
        avg_scores.max().item(),
        num_threshold_search,
        device=avg_scores.device,
    )

    best_T = candidates[0].item()
    best_diff = float("inf")
    for T in candidates:
        cluster_mask = avg_scores > T  # (H, K)
        # Số keys giữ = sum N_j khi cluster_mask
        kept = (cluster_mask.float() * cluster_sizes).sum(dim=-1)  # (H,)
        total = cluster_sizes.sum(dim=-1)  # (H,)
        keep_ratio = (kept / total.clamp(min=1)).mean().item()
        diff = abs(keep_ratio - target_keep_ratio)
        if diff < best_diff:
            best_diff = diff
            best_T = T.item()

    return best_T


def squeezed_attention_forward(
    query: torch.Tensor,           # (H, D) - 1 query token
    full_keys: torch.Tensor,       # (H, N, D)
    full_values: torch.Tensor,     # (H, N, D)
    key_centroids: torch.Tensor,   # (H, K, D)
    cluster_sizes: torch.Tensor,   # (H, K)
    normalized_variance: torch.Tensor,  # (H, K)
    labels: torch.Tensor,          # (H, N)
    threshold: float,
    gamma: float = 0.3,
    scale: float = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Một lượt forward attention dùng Value-Aware Squeezed Attention.
    
    Đây là implementation reference (không tối ưu kernel) để verify đúng đắn.
    
    Returns:
        output: (H, D)
        info: dict chứa các metric (kv budget, num_kept_keys, ...)
    """
    H, N, D = full_keys.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    # 1. Centroid lookup với value-aware boost
    cluster_mask, S_adj = value_aware_retrieve(
        query, key_centroids, cluster_sizes, normalized_variance, threshold, gamma
    )
    # 2. Mask keys
    key_mask = keys_mask_from_clusters(cluster_mask, labels)  # (H, N)

    # 3. Sparse attention: tính q·k^T, rồi softmax với mask
    # query: (H, D), full_keys: (H, N, D) -> attn_logits: (H, N)
    attn_logits = torch.einsum("hd,hnd->hn", query, full_keys) * scale
    # Mask out keys không trong cluster được chọn
    attn_logits = attn_logits.masked_fill(~key_mask, float("-inf"))

    attn_weights = F.softmax(attn_logits, dim=-1)
    # Output: (H, D)
    output = torch.einsum("hn,hnd->hd", attn_weights, full_values)

    info = {
        "num_clusters_kept": cluster_mask.float().sum(dim=-1).mean().item(),
        "num_keys_kept": key_mask.float().sum(dim=-1).mean().item(),
        "total_keys": N,
        "kv_budget": key_mask.float().mean().item(),
    }
    return output, info


def baseline_full_attention(
    query: torch.Tensor,        # (H, D)
    full_keys: torch.Tensor,    # (H, N, D)
    full_values: torch.Tensor,  # (H, N, D)
    scale: float = None,
) -> torch.Tensor:
    """Full attention - baseline để so sánh."""
    H, N, D = full_keys.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    attn_logits = torch.einsum("hd,hnd->hn", query, full_keys) * scale
    attn_weights = F.softmax(attn_logits, dim=-1)
    output = torch.einsum("hn,hnd->hd", attn_weights, full_values)
    return output
