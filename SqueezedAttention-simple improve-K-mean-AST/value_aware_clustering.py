"""
Value-Aware Clustering for Squeezed Attention.

Cải tiến chính so với clustering gốc:
1. Joint clustering trên concat(alpha*K, beta*V) thay vì chỉ K
2. Tính value variance per cluster để dùng làm tín hiệu boost score
3. Trả về thêm value_centroids và value_variance để dùng online

Có thể drop-in thay thế cho `squeezedattention.clustering.run_clustering`.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


def _kmeans_cosine(
    x: torch.Tensor,
    num_clusters: int,
    num_iters: int = 10,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    K-means trên đơn vị cầu (cosine similarity), batched theo head.
    
    Args:
        x: (H, N, D) - H heads, N points, D dims (đã được normalize L2)
        num_clusters: số cluster mong muốn
        num_iters: số vòng lặp Lloyd
        seed: random seed cho init
    
    Returns:
        centroids: (H, num_clusters, D) - đã normalize
        labels:    (H, N)
    """
    H, N, D = x.shape
    device = x.device
    dtype = x.dtype

    # Init: chọn ngẫu nhiên num_clusters điểm cho mỗi head
    g = torch.Generator(device=device).manual_seed(seed)
    init_idx = torch.randint(0, N, (H, num_clusters), generator=g, device=device)
    # gather: x[h, init_idx[h]]
    centroids = torch.gather(
        x, 1, init_idx.unsqueeze(-1).expand(-1, -1, D)
    ).clone()  # (H, K, D)

    for _ in range(num_iters):
        # cosine sim = dot product nếu cả hai đều normalized
        # sims: (H, N, K)
        sims = torch.bmm(x, centroids.transpose(1, 2))
        labels = sims.argmax(dim=-1)  # (H, N)

        # Cập nhật centroid = mean của các điểm cùng label, sau đó renormalize
        # Dùng scatter_add cho tốc độ
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(H, num_clusters, device=device, dtype=dtype)

        # Mở rộng labels để scatter add trên D
        labels_expanded = labels.unsqueeze(-1).expand(-1, -1, D)  # (H, N, D)
        new_centroids.scatter_add_(1, labels_expanded, x)

        ones = torch.ones(H, N, device=device, dtype=dtype)
        counts.scatter_add_(1, labels, ones)

        # Tránh chia 0 cho các cluster rỗng -> giữ centroid cũ
        empty_mask = counts == 0
        counts_safe = counts.clamp(min=1).unsqueeze(-1)
        new_centroids = new_centroids / counts_safe

        # Reset cluster rỗng về centroid cũ
        new_centroids = torch.where(
            empty_mask.unsqueeze(-1), centroids, new_centroids
        )

        # Renormalize lên đơn vị cầu
        new_centroids = F.normalize(new_centroids, dim=-1)
        centroids = new_centroids

    # Gán nhãn cuối cùng
    sims = torch.bmm(x, centroids.transpose(1, 2))
    labels = sims.argmax(dim=-1)
    return centroids, labels


def value_aware_kmeans(
    keys: torch.Tensor,
    values: torch.Tensor,
    num_clusters: int,
    alpha: float = 1.0,
    beta: float = 0.5,
    num_iters: int = 10,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Joint clustering trên concat(alpha*K_norm, beta*V_norm).
    
    Args:
        keys:   (H, N, D_k)  raw key vectors (chưa normalize)
        values: (H, N, D_v)  raw value vectors
        num_clusters: số cluster
        alpha: trọng số phần key trong joint vector (default 1.0 - giữ nguyên ưu thế)
        beta:  trọng số phần value (default 0.5 - đóng góp nhẹ hơn để không phá retrieval quality)
        num_iters: số iter của K-means
    
    Returns:
        key_centroids:   (H, K, D_k)  centroid trong key space (raw, đã chuẩn hóa)
        value_centroids: (H, K, D_v)  centroid trong value space (mean values theo cluster)
        labels:          (H, N)       cluster assignment cho mỗi key
        value_variance:  (H, K)       variance trung bình của values trong mỗi cluster
    """
    H, N, D_k = keys.shape
    _, _, D_v = values.shape
    device = keys.device

    # Chuẩn hóa K và V riêng để alpha/beta điều khiển trọng số đúng nghĩa
    keys_n = F.normalize(keys, dim=-1)
    values_n = F.normalize(values, dim=-1)

    # Joint feature: (H, N, D_k + D_v)
    joint = torch.cat([alpha * keys_n, beta * values_n], dim=-1)
    # Normalize lại để dùng cosine K-means
    joint = F.normalize(joint, dim=-1)

    # K-means trên joint space
    _, labels = _kmeans_cosine(joint, num_clusters, num_iters=num_iters, seed=seed)
    # labels: (H, N)

    # Tính key centroid (bằng mean keys trong cluster, sau đó normalize)
    # và value centroid (mean values - không normalize vì values đi vào output trực tiếp)
    key_centroids = torch.zeros(H, num_clusters, D_k, device=device, dtype=keys.dtype)
    value_centroids = torch.zeros(H, num_clusters, D_v, device=device, dtype=values.dtype)
    counts = torch.zeros(H, num_clusters, device=device, dtype=keys.dtype)

    labels_k = labels.unsqueeze(-1).expand(-1, -1, D_k)
    labels_v = labels.unsqueeze(-1).expand(-1, -1, D_v)

    key_centroids.scatter_add_(1, labels_k, keys)
    value_centroids.scatter_add_(1, labels_v, values)
    ones = torch.ones(H, N, device=device, dtype=keys.dtype)
    counts.scatter_add_(1, labels, ones)

    counts_safe = counts.clamp(min=1).unsqueeze(-1)
    key_centroids = key_centroids / counts_safe
    value_centroids = value_centroids / counts_safe

    # Chuẩn hóa key_centroid để tương thích với pipeline gốc
    # (Squeezed Attention dùng dot product q·C^T, các k đã norm)
    key_centroids = F.normalize(key_centroids, dim=-1)

    # Tính value variance per cluster: trung bình ||v - v_centroid||^2 trong cluster
    # Dùng để làm tín hiệu boost score online
    expanded_centroids = torch.gather(value_centroids, 1, labels_v)  # (H, N, D_v)
    sq_diff = (values - expanded_centroids).pow(2).sum(dim=-1)  # (H, N)

    var_sum = torch.zeros(H, num_clusters, device=device, dtype=keys.dtype)
    var_sum.scatter_add_(1, labels, sq_diff)
    value_variance = var_sum / counts.clamp(min=1)  # (H, K)

    return key_centroids, value_centroids, labels, value_variance


def run_value_aware_clustering(
    all_keys_layers: List[torch.Tensor],
    all_values_layers: List[torch.Tensor],
    num_centroids: int,
    observation_window: int = 100,
    alpha: float = 1.0,
    beta: float = 0.5,
    num_iters: int = 10,
    device: torch.device = torch.device("cuda:0"),
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Drop-in thay thế cho `run_clustering` của repo gốc, nhưng dùng joint K-V clustering.
    
    Args:
        all_keys_layers:   list of (B, H, N, D_k) - tensors từ hook, mỗi layer 1 phần tử
        all_values_layers: list of (B, H, N, D_v) - same shape
        num_centroids: số cluster mỗi head
        observation_window: số tokens cuối KHÔNG cluster (giữ nguyên để dùng exact)
        alpha, beta: trọng số K, V trong joint clustering
        num_iters: số iter K-means
        device: device để chạy
    
    Returns:
        key_centroids_dict: {layer_idx: (H, K, D_k)}
        value_centroids_dict: {layer_idx: (H, K, D_v)}
        labels_dict: {layer_idx: (H, N - obs_window)}
        value_variance_dict: {layer_idx: (H, K)}
    """
    key_centroids_dict = {}
    value_centroids_dict = {}
    labels_dict = {}
    value_variance_dict = {}

    num_layers = len(all_keys_layers)
    assert len(all_values_layers) == num_layers, "K and V must have same number of layers"

    for layer_idx in range(num_layers):
        keys_l = all_keys_layers[layer_idx].to(device)    # (B, H, N, D_k)
        values_l = all_values_layers[layer_idx].to(device)  # (B, H, N, D_v)

        # Bỏ batch dim (giả định B=1, giống pipeline gốc)
        if keys_l.dim() == 4:
            keys_l = keys_l.squeeze(0)
            values_l = values_l.squeeze(0)

        # Cắt observation_window cuối
        if observation_window > 0:
            keys_to_cluster = keys_l[:, :-observation_window, :]
            values_to_cluster = values_l[:, :-observation_window, :]
        else:
            keys_to_cluster = keys_l
            values_to_cluster = values_l

        # Đảm bảo có đủ tokens để cluster
        N_cluster = keys_to_cluster.shape[1]
        actual_num_centroids = min(num_centroids, N_cluster)
        if actual_num_centroids < 1:
            actual_num_centroids = 1

        kc, vc, labels, vvar = value_aware_kmeans(
            keys_to_cluster.float(),  # K-means nên chạy fp32 cho ổn định
            values_to_cluster.float(),
            actual_num_centroids,
            alpha=alpha,
            beta=beta,
            num_iters=num_iters,
            seed=layer_idx,  # seed khác nhau cho mỗi layer để init đa dạng
        )

        # Cast lại về dtype gốc của input
        target_dtype = all_keys_layers[layer_idx].dtype
        key_centroids_dict[layer_idx] = kc.to(target_dtype)
        value_centroids_dict[layer_idx] = vc.to(target_dtype)
        labels_dict[layer_idx] = labels  # int64 OK
        value_variance_dict[layer_idx] = vvar.to(target_dtype)

    return key_centroids_dict, value_centroids_dict, labels_dict, value_variance_dict


def normalize_value_variance(
    value_variance_dict: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """
    Chuẩn hóa value variance về [0, 1] per (layer, head) để dùng làm hệ số boost.
    
    Đây là bước quan trọng: variance thô có scale rất khác nhau giữa layers/heads,
    nên cần normalize trước khi dùng làm hệ số nhân.
    """
    normalized = {}
    for layer_idx, vvar in value_variance_dict.items():
        # vvar: (H, K)
        # Normalize per head: (var - min) / (max - min + eps)
        v_min = vvar.min(dim=-1, keepdim=True).values
        v_max = vvar.max(dim=-1, keepdim=True).values
        v_norm = (vvar - v_min) / (v_max - v_min + 1e-8)
        normalized[layer_idx] = v_norm
    return normalized


def value_aware_score_adjustment(
    base_scores: torch.Tensor,        # (H, K) - S_i thông thường
    normalized_variance: torch.Tensor, # (H, K) - đã normalize về [0,1]
    gamma: float = 0.3,
) -> torch.Tensor:
    """
    Áp dụng boost dựa trên value variance:
        S_tilde_i = S_i * (1 + gamma * sigma_v_normalized)
    
    Cluster có values khác nhau nhiều (variance cao) sẽ được boost thêm để tránh bỏ sót,
    vì khi đó việc đại diện bằng 1 centroid không đủ chính xác.
    
    Args:
        base_scores: softmax-normalized importance scores
        normalized_variance: variance đã chuẩn hóa [0,1]
        gamma: hệ số ảnh hưởng. gamma=0 -> giữ nguyên Squeezed Attention gốc.
               Khuyến nghị: 0.2 - 0.5
    
    Returns:
        adjusted_scores: (H, K)
    """
    return base_scores * (1.0 + gamma * normalized_variance)
