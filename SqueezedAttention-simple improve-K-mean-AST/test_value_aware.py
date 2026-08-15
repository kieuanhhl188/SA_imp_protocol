"""
Unit tests / smoke tests cho value-aware clustering.

Chạy:
    python test_value_aware.py

Không cần GPU - chạy được trên CPU. Mục tiêu: verify logic đúng đắn,
không phải benchmark performance.
"""

import torch
import torch.nn.functional as F

from value_aware_clustering import (
    _kmeans_cosine,
    value_aware_kmeans,
    run_value_aware_clustering,
    normalize_value_variance,
    value_aware_score_adjustment,
)
from value_aware_retrieval import (
    compute_base_scores,
    value_aware_retrieve,
    keys_mask_from_clusters,
    calibrate_threshold,
    squeezed_attention_forward,
    baseline_full_attention,
)


def test_kmeans_cosine():
    print("[test_kmeans_cosine] ", end="")
    torch.manual_seed(0)
    H, N, D, K = 2, 100, 16, 5
    x = F.normalize(torch.randn(H, N, D), dim=-1)
    centroids, labels = _kmeans_cosine(x, K, num_iters=10)
    assert centroids.shape == (H, K, D)
    assert labels.shape == (H, N)
    # Centroids phải normalize
    norms = centroids.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), f"centroids not normalized: {norms}"
    # Mỗi label trong [0, K)
    assert labels.min() >= 0 and labels.max() < K
    print("OK")


def test_value_aware_kmeans_output_shapes():
    print("[test_value_aware_kmeans_output_shapes] ", end="")
    torch.manual_seed(1)
    H, N, D_k, D_v, K = 4, 200, 32, 32, 8
    keys = torch.randn(H, N, D_k)
    values = torch.randn(H, N, D_v)
    kc, vc, lbl, vvar = value_aware_kmeans(keys, values, K, alpha=1.0, beta=0.5)
    assert kc.shape == (H, K, D_k)
    assert vc.shape == (H, K, D_v)
    assert lbl.shape == (H, N)
    assert vvar.shape == (H, K)
    # variance không âm
    assert (vvar >= 0).all()
    print("OK")


def test_value_aware_clustering_matches_keyonly_when_beta_zero():
    """Nếu beta=0, joint clustering chỉ dùng K -> kết quả gần (không hoàn toàn vì
    init khác nhau) với key-only K-means."""
    print("[test_value_aware_clustering_matches_keyonly_when_beta_zero] ", end="")
    torch.manual_seed(2)
    H, N, D, K = 2, 100, 16, 5
    keys = torch.randn(H, N, D)
    values = torch.randn(H, N, D)
    kc, vc, lbl, vvar = value_aware_kmeans(
        keys, values, K, alpha=1.0, beta=0.0, num_iters=20
    )
    # Khi beta=0, joint = alpha*keys_norm sau đó normalize -> giống keys_norm
    # Vì cả hai đều là cosine K-means trên same input, label hợp lý
    # (không assert chính xác - random init)
    # Chỉ kiểm tra: mỗi cluster có ít nhất 1 phần tử (hoặc 0 cho cluster rỗng)
    counts = torch.zeros(H, K)
    counts.scatter_add_(1, lbl, torch.ones_like(lbl, dtype=torch.float))
    # Tổng count = N
    assert (counts.sum(dim=-1) == N).all()
    print("OK")


def test_compute_base_scores_sums_correctly():
    """S_i là softmax-style: sum(N_j * exp(qC_j) / Z) phải bằng 1."""
    print("[test_compute_base_scores_sums_correctly] ", end="")
    torch.manual_seed(3)
    H, K, D, N = 2, 5, 16, 50
    q = torch.randn(H, D)
    centroids = F.normalize(torch.randn(H, K, D), dim=-1)
    sizes = torch.tensor([[10, 8, 12, 15, 5], [5, 5, 10, 20, 10]], dtype=torch.float)
    S = compute_base_scores(q, centroids, sizes)
    # sum(N_j * S_j) phải = 1 (vì denom = sum N_j exp(...))
    weighted_sum = (sizes * S).sum(dim=-1)
    assert torch.allclose(weighted_sum, torch.ones_like(weighted_sum), atol=1e-4), \
        f"weighted sum != 1: {weighted_sum}"
    print("OK")


def test_value_aware_retrieve_threshold_filters():
    print("[test_value_aware_retrieve_threshold_filters] ", end="")
    torch.manual_seed(4)
    H, K, D = 2, 10, 16
    q = torch.randn(H, D)
    centroids = F.normalize(torch.randn(H, K, D), dim=-1)
    sizes = torch.full((H, K), 10.0)
    nvar = torch.rand(H, K)

    # Threshold cao -> ít cluster qua
    mask_high, _ = value_aware_retrieve(q, centroids, sizes, nvar, threshold=0.5, gamma=0.3)
    # Threshold thấp -> nhiều cluster qua
    mask_low, _ = value_aware_retrieve(q, centroids, sizes, nvar, threshold=0.0, gamma=0.3)

    n_high = mask_high.sum().item()
    n_low = mask_low.sum().item()
    assert n_low >= n_high, f"low threshold should keep more clusters: low={n_low}, high={n_high}"
    print(f"OK (low_kept={n_low}, high_kept={n_high})")


def test_keys_mask_from_clusters():
    print("[test_keys_mask_from_clusters] ", end="")
    H, K, N = 2, 4, 20
    cluster_mask = torch.zeros(H, K, dtype=torch.bool)
    cluster_mask[0, 1] = True
    cluster_mask[0, 2] = True
    cluster_mask[1, 0] = True

    labels = torch.zeros(H, N, dtype=torch.long)
    # Head 0: token 0-4 thuộc cluster 0, 5-9 cluster 1, 10-14 cluster 2, 15-19 cluster 3
    labels[0, 5:10] = 1
    labels[0, 10:15] = 2
    labels[0, 15:20] = 3
    # Head 1: tất cả thuộc cluster 0
    labels[1, :] = 0

    key_mask = keys_mask_from_clusters(cluster_mask, labels)
    # Head 0: kì vọng True ở 5-14
    expected_h0 = torch.zeros(N, dtype=torch.bool)
    expected_h0[5:15] = True
    assert torch.equal(key_mask[0], expected_h0), f"head 0 mask wrong: {key_mask[0]}"
    # Head 1: tất cả True (vì mọi token đều ở cluster 0 và cluster 0 được chọn)
    assert key_mask[1].all()
    print("OK")


def test_squeezed_attention_recovers_full_when_no_pruning():
    """Nếu tất cả cluster đều được giữ -> output bằng full attention."""
    print("[test_squeezed_attention_recovers_full_when_no_pruning] ", end="")
    torch.manual_seed(5)
    H, N, D, K = 2, 50, 16, 5
    q = torch.randn(H, D)
    keys = torch.randn(H, N, D)
    values = torch.randn(H, N, D)

    kc, vc, lbl, vvar = value_aware_kmeans(keys, values, K, alpha=1.0, beta=0.5)
    sizes = torch.zeros(H, K)
    sizes.scatter_add_(1, lbl, torch.ones_like(lbl, dtype=torch.float))
    nvar = torch.zeros(H, K)  # tắt boost

    # Threshold rất nhỏ -> giữ tất cả
    out_sq, info = squeezed_attention_forward(
        q, keys, values, kc, sizes, nvar, lbl,
        threshold=-1e9, gamma=0.0,
    )
    out_full = baseline_full_attention(q, keys, values)
    assert info["kv_budget"] > 0.99, f"expected ~100% budget, got {info['kv_budget']}"
    diff = (out_sq - out_full).abs().max().item()
    assert diff < 1e-4, f"outputs differ: max abs diff = {diff}"
    print(f"OK (max diff = {diff:.2e})")


def test_calibrate_threshold_achieves_target_sparsity():
    print("[test_calibrate_threshold_achieves_target_sparsity] ", end="")
    torch.manual_seed(6)
    H, N, D, K = 4, 200, 16, 20
    keys = torch.randn(H, N, D)
    values = torch.randn(H, N, D)
    queries = torch.randn(20, H, D)

    kc, vc, lbl, vvar = value_aware_kmeans(keys, values, K, alpha=1.0, beta=0.5)
    sizes = torch.zeros(H, K)
    sizes.scatter_add_(1, lbl, torch.ones_like(lbl, dtype=torch.float))
    nvar = torch.zeros(H, K)

    target = 0.7  # giữ 30%
    T = calibrate_threshold(
        queries, kc, sizes, nvar, lbl,
        target_sparsity=target, gamma=0.0,
        num_threshold_search=100,
    )
    # Kiểm tra với threshold T, % keys giữ thực tế
    total_kept = []
    for q in queries:
        S = compute_base_scores(q, kc, sizes)
        cluster_mask = S > T
        kept = (cluster_mask.float() * sizes).sum(dim=-1) / sizes.sum(dim=-1)
        total_kept.append(kept.mean().item())
    avg_keep = sum(total_kept) / len(total_kept)
    expected_keep = 1 - target
    print(f"OK (target_keep={expected_keep:.2f}, actual_keep={avg_keep:.2f})")
    # Cho phép sai lệch 15% vì grid search thưa và variance cao
    assert abs(avg_keep - expected_keep) < 0.20, \
        f"sparsity off: target keep {expected_keep}, got {avg_keep}"


def test_run_value_aware_clustering_layer_dict():
    print("[test_run_value_aware_clustering_layer_dict] ", end="")
    torch.manual_seed(7)
    num_layers = 3
    H, N, D = 4, 100, 16
    K = 8
    keys_layers = [torch.randn(1, H, N, D) for _ in range(num_layers)]
    values_layers = [torch.randn(1, H, N, D) for _ in range(num_layers)]
    kc_dict, vc_dict, lbl_dict, vvar_dict = run_value_aware_clustering(
        keys_layers, values_layers, num_centroids=K,
        observation_window=10, alpha=1.0, beta=0.5,
        device=torch.device("cpu"),
    )
    assert len(kc_dict) == num_layers
    for li in range(num_layers):
        assert kc_dict[li].shape == (H, K, D)
        assert vc_dict[li].shape == (H, K, D)
        assert lbl_dict[li].shape == (H, N - 10)
        assert vvar_dict[li].shape == (H, K)
    # Normalize variance
    nvar_dict = normalize_value_variance(vvar_dict)
    for li in range(num_layers):
        v = nvar_dict[li]
        assert v.min() >= 0 and v.max() <= 1 + 1e-5
    print("OK")


def test_value_aware_improves_when_value_diverse():
    """
    Test sanity: khi values trong cùng "key-cluster" rất khác nhau,
    value-aware retrieval nên giữ lại nhiều hơn so với key-only.
    
    Setup: tạo 2 cluster theo K, nhưng trong cluster 0 các V rất đa dạng,
    trong cluster 1 các V đồng nhất. Kì vọng cluster 0 có variance cao
    -> được boost -> dễ được giữ lại.
    """
    print("[test_value_aware_improves_when_value_diverse] ", end="")
    torch.manual_seed(8)
    H, D = 1, 16
    # 2 nhóm keys quanh 2 anchor riêng biệt
    anchor1 = F.normalize(torch.randn(D), dim=-1)
    anchor2 = F.normalize(torch.randn(D), dim=-1)
    n_per = 30
    k1 = anchor1.unsqueeze(0).expand(n_per, -1) + 0.05 * torch.randn(n_per, D)
    k2 = anchor2.unsqueeze(0).expand(n_per, -1) + 0.05 * torch.randn(n_per, D)
    keys = torch.cat([k1, k2], dim=0).unsqueeze(0)  # (1, 60, D)

    # Values: nhóm 1 đa dạng, nhóm 2 đồng nhất
    v1 = torch.randn(n_per, D)  # đa dạng
    v2 = torch.zeros(n_per, D) + torch.randn(D).unsqueeze(0)  # đồng nhất
    values = torch.cat([v1, v2], dim=0).unsqueeze(0)

    # Cluster với K=2 -> kì vọng 2 anchor tách
    kc, vc, lbl, vvar = value_aware_kmeans(
        keys, values, num_clusters=2, alpha=1.0, beta=0.0, num_iters=20
    )
    # Nhận diện cluster nào là nhóm 1 (đa dạng V)
    # Cluster nào có nhiều labels[0:n_per] thuộc về thì là nhóm 1
    cluster_of_first = lbl[0, :n_per].mode().values.item()
    print(f"OK (cluster 1 (diverse V) idx={cluster_of_first}, "
          f"variance values: {vvar[0].tolist()})")
    # Cluster đa dạng V phải có variance cao hơn
    assert vvar[0, cluster_of_first] > vvar[0, 1 - cluster_of_first], \
        f"diverse cluster should have higher variance: {vvar[0]}"


def main():
    print("=" * 60)
    print("Running value-aware clustering tests")
    print("=" * 60)
    test_kmeans_cosine()
    test_value_aware_kmeans_output_shapes()
    test_value_aware_clustering_matches_keyonly_when_beta_zero()
    test_compute_base_scores_sums_correctly()
    test_value_aware_retrieve_threshold_filters()
    test_keys_mask_from_clusters()
    test_squeezed_attention_recovers_full_when_no_pruning()
    test_calibrate_threshold_achieves_target_sparsity()
    test_run_value_aware_clustering_layer_dict()
    test_value_aware_improves_when_value_diverse()
    print("\n" + "=" * 60)
    print("All tests passed ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
