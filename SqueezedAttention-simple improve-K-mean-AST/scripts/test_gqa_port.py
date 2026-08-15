#!/usr/bin/env python
"""
test_gqa_port.py — kiểm tra logic GQA của bản port Squeezed Attention sang Qwen2.

Chạy được trên CPU, không cần GPU / triton / flash-attn / cuml. Chỉ kiểm phần
biến đổi shape và ngữ nghĩa — thứ dễ sai âm thầm nhất khi port từ MHA sang GQA.

Ba bất biến được kiểm:
  1. Nhân bản head KV -> head Q đúng thứ tự (repeat_interleave, KHÔNG phải repeat/tile).
     Sai thứ tự thì query head vẫn tra được centroid, không crash, chỉ tra NHẦM nhóm.
  2. Với MHA (groups == 1) mọi biến đổi là no-op -> đường LLaMA của gate Phase 0 không đổi.
  3. Nhân bản centroid rồi tra cho ra đúng kết quả như tra trực tiếp trên head KV
     tương ứng, tức mỗi query head thấy đúng bộ centroid của nhóm mình (Appendix G).

Usage:
    python scripts/test_gqa_port.py
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)


def expand_kv_heads_to_query_heads(tensor, num_key_value_groups):
    """Bản sao của hàm trong modeling_qwen2.py, để test không phải import transformers."""
    if num_key_value_groups == 1:
        return tensor
    return torch.repeat_interleave(tensor, num_key_value_groups, dim=1)


def repeat_kv(hidden_states, n_rep):
    """Bản sao repeat_kv của transformers, để đối chiếu thứ tự head."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


OK = True


def check(name, cond, extra=""):
    global OK
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        OK = False


def main():
    torch.manual_seed(0)

    # Cấu hình thật của Qwen2.5-Coder-7B-Instruct
    H_Q, H_KV = 28, 4
    GROUPS = H_Q // H_KV          # 7
    S, D, K = 64, 16, 8

    print("=== 1. Thứ tự head khi nhân bản phải khớp repeat_kv ===")
    # keys [B, H_KV, S, D] -> repeat_kv -> [B, H_Q, S, D]
    keys = torch.randn(1, H_KV, S, D)
    keys_rep = repeat_kv(keys, GROUPS)
    check("repeat_kv ra đúng số head", keys_rep.shape == (1, H_Q, S, D), str(tuple(keys_rep.shape)))

    # centroids [B, H_KV, K, D] -> expand -> [B, H_Q, K, D]
    centroids = torch.randn(1, H_KV, K, D)
    centroids_exp = expand_kv_heads_to_query_heads(centroids, GROUPS)
    check("expand centroid ra đúng số head",
          centroids_exp.shape == (1, H_Q, K, D), str(tuple(centroids_exp.shape)))

    # Bất biến then chốt: query head h phải nhìn thấy KV head h // GROUPS,
    # và cả keys lẫn centroids phải đồng ý về ánh xạ đó.
    mapping_ok = True
    for h in range(H_Q):
        kv = h // GROUPS
        if not torch.equal(keys_rep[0, h], keys[0, kv]):
            mapping_ok = False
        if not torch.equal(centroids_exp[0, h], centroids[0, kv]):
            mapping_ok = False
    check("key và centroid cùng ánh xạ head Q -> head KV (h // groups)", mapping_ok)

    # Chứng minh repeat (tile) sẽ SAI: nó cho ánh xạ h % H_KV
    centroids_tiled = centroids.repeat(1, GROUPS, 1, 1)
    check("tile/repeat cho ánh xạ KHÁC -> đúng là phải dùng repeat_interleave",
          not torch.equal(centroids_tiled, centroids_exp))

    print("\n=== 2. MHA phải là no-op hoàn toàn ===")
    mha_keys = torch.randn(1, 32, S, D)
    mha_centroids = torch.randn(1, 32, K, D)
    mha_labels = torch.randint(0, K, (1, 32, S))
    check("expand centroid (groups=1) trả về đúng object",
          expand_kv_heads_to_query_heads(mha_centroids, 1) is mha_centroids)
    check("expand label (groups=1) trả về đúng object",
          expand_kv_heads_to_query_heads(mha_labels, 1) is mha_labels)
    check("repeat_kv (n_rep=1) trả về đúng object", repeat_kv(mha_keys, 1) is mha_keys)

    print("\n=== 3. num_keys tính trên label đã nhân bản ===")
    labels = torch.randint(0, K, (1, H_KV, S))
    labels_exp = expand_kv_heads_to_query_heads(labels, GROUPS)
    check("label expand ra đúng shape", labels_exp.shape == (1, H_Q, S), str(tuple(labels_exp.shape)))

    # Đây chính là chỗ bản port dễ sai: bản Llama dùng key_states.shape[1] làm số head,
    # nhưng ở Qwen2 tại thời điểm đó key_states VẪN còn H_KV head vì repeat_kv chưa chạy.
    num_heads_wrong = keys.shape[1]           # H_KV = 4  <- sai
    num_heads_right = labels_exp.shape[-2]    # H_Q = 28  <- đúng
    check("shape[-2] của label cho ra H_Q", num_heads_right == H_Q, str(num_heads_right))
    check("dùng key_states.shape[1] sẽ ra H_KV (bug nếu copy nguyên bản Llama)",
          num_heads_wrong == H_KV, str(num_heads_wrong))

    num_keys = torch.zeros((num_heads_right, K))
    for k in range(K):
        num_keys[:, k] = torch.sum(labels_exp == k, dim=-1)
    check("mỗi head đếm đủ S key", bool(torch.all(num_keys.sum(dim=-1) == S)))
    # head trong cùng nhóm phải có phân bố giống hệt nhau
    same_group_ok = all(
        torch.equal(num_keys[h], num_keys[(h // GROUPS) * GROUPS]) for h in range(H_Q)
    )
    check("head cùng nhóm KV có cùng phân bố cluster", same_group_ok)

    print("\n=== 4. Lookup trên centroid đã nhân bản == lookup trực tiếp trên head KV ===")
    # Mô phỏng phép cốt lõi của centroid_lookup: q @ centroid^T
    queries = torch.randn(1, H_Q, 4, D)
    scores_expanded = torch.matmul(queries[0], centroids_exp[0].transpose(1, 2))
    scores_manual = torch.stack(
        [queries[0, h] @ centroids[0, h // GROUPS].T for h in range(H_Q)], dim=0
    )
    check("điểm số khớp nhau", torch.allclose(scores_expanded, scores_manual, atol=1e-5),
          f"sai lệch max {float((scores_expanded - scores_manual).abs().max()):.2e}")

    # Và các query head trong cùng nhóm phải cho điểm KHÁC nhau (vì q khác nhau)
    # -> đúng tinh thần "each head independently selects" của Appendix G
    h0, h1 = 0, 1
    check("head cùng nhóm vẫn chọn độc lập (điểm khác nhau)",
          not torch.allclose(scores_expanded[h0], scores_expanded[h1]))

    print("\n=== 5. run_global_threshold: nhánh GQA ===")
    sys.path.insert(0, REPO_ROOT)
    src = open(os.path.join(REPO_ROOT, "squeezedattention", "clustering.py"),
               encoding="utf-8").read()
    check("clustering.py có nhánh GQA", "num_q_heads != num_kv_heads" in src)
    check("dùng repeat_interleave (không phải repeat)",
          "torch.repeat_interleave(centroids_tensor" in src)
    check("nhân bản cả key_shared_prefix", "repeat_interleave(keys_shared_prefix" in src)

    print("\n=== 6. modeling_qwen2: các điểm dễ sai ===")
    q = open(os.path.join(REPO_ROOT, "transformers/src/transformers/models/qwen2/modeling_qwen2.py"),
             encoding="utf-8").read()

    # QUAN TRỌNG: phải so thứ tự TRONG thân Qwen2FlashAttention2. Cả Qwen2Attention (eager)
    # và Qwen2SdpaAttention cũng có dòng repeat_kv, tìm trên cả file sẽ bắt nhầm.
    fa = q[q.index("class Qwen2FlashAttention2"):q.index("class Qwen2SdpaAttention")]
    check("chỉ có 1 repeat_kv(key_states) trong thân FlashAttention2",
          fa.count("key_states = repeat_kv(key_states, self.num_key_value_groups)") == 1)
    check("q/k/v cho clustering lấy TRƯỚC repeat_kv",
          fa.index("qkv_states_for_clustering = (query_states, key_states, value_states)")
          < fa.index("key_states = repeat_kv(key_states, self.num_key_value_groups)"))
    check("centroid lookup chạy TRƯỚC repeat_kv",
          fa.index("k_idx, head_kv_len, num_kv_blocks, head_start_block, _ = centroid_lookup")
          < fa.index("key_states = repeat_kv(key_states, self.num_key_value_groups)"))
    check("sparse attention chạy SAU repeat_kv",
          fa.index("key_states = repeat_kv(key_states, self.num_key_value_groups)")
          < fa.index("attn_output_merged, _, _ = dynamic_sparse_attention"))
    check("số head lấy từ centroid_labels.shape[-2]", "centroid_labels.shape[-2]" in q)
    check("chặn sliding window khi bật centroid", "assert not use_sliding_windows" in q)
    check("kiểm centroid đúng số head KV", "n_head_centroid == self.num_key_value_heads" in q)
    check("dùng num_hidden_layers, không dùng num_heads, cho vòng lặp layer",
          "for layer_num in range(self.num_hidden_layers):" in q)

    print("\n" + ("TẤT CẢ PASS" if OK else "CÓ TEST FAIL"))
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
