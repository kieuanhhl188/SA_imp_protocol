"""
Demo end-to-end Value-Aware Squeezed Attention trên model Hugging Face nhỏ.

Workflow:
1. Load model + tokenizer (mặc định Qwen2.5-1.5B-Instruct, ~3GB VRAM fp16)
2. Cung cấp 1 fixed context dài (vd: 1 bài báo / tài liệu)
3. Hook vào các layer attention để extract K, V của fixed context
4. Chạy 3 phương pháp clustering offline:
   (a) Key-only K-means (baseline Squeezed Attention gốc)
   (b) Value-aware K-means (cải tiến của chúng ta)
5. Với một query mới, so sánh attention output với:
   - Full attention (ground truth)
   - Key-only Squeezed Attention
   - Value-aware Squeezed Attention
6. Đo: cosine similarity giữa output, MSE, và KV budget thực tế

Cách chạy:
    python demo_value_aware.py --model Qwen/Qwen2.5-1.5B-Instruct --sparsity 0.9 --gamma 0.3

Yêu cầu VRAM: ~4-6GB cho Qwen2.5-1.5B (đủ chạy thoải mái trên 24GB).
"""

import argparse
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from value_aware_clustering import (
    run_value_aware_clustering,
    normalize_value_variance,
    value_aware_kmeans,
    _kmeans_cosine,
)
from value_aware_retrieval import (
    squeezed_attention_forward,
    baseline_full_attention,
    calibrate_threshold,
)


# ---------------------------------------------------------------------------
# Sample fixed context (1 đoạn dài về Squeezed Attention paper, tự sinh khoảng
# ~2-4K tokens - đủ để chứng minh ý tưởng mà không cần dataset lớn)
# ---------------------------------------------------------------------------
SAMPLE_CONTEXT = """
Large Language Models (LLMs) have seen rapid advancements in recent years, enabling
a range of downstream applications including Question Answering and analysis over
structured and unstructured documents. Performance on these tasks has benefited
from the increased context lengths of newer open-source and closed-source models,
as these tasks benefit from incorporating a large amount of input context in order
to condition the model to generate particular outputs. However, deployment of LLMs
for downstream applications is constrained by inference costs.

Long context-length applications have large memory capacity and memory bandwidth
requirements due to the size of the KV cache, which increases linearly with respect
to sequence length. For many applications such as in-context learning, document QA,
and code generation, over a series of prompts a large portion of the input context
is fixed. This fixed context, which may contain system instructions, documentation,
or few-shot examples, is extremely beneficial for tailoring the model to the target
application. However, increasing the length of the fixed context poses a significant
challenge for inference efficiency.

Squeezed Attention accelerates fixed context applications by accelerating the
attention computation. The method quickly identifies which keys in the fixed context
are important for a given query token. Offline prior to inference, the method clusters
the keys in the fixed context based on their semantic similarity and then represents
keys from the same cluster using a single representative key centroid. At inference
time, when the user input is received, the method retrieves the important keys by
first comparing the query tokens with the key centroids, rather than the entire set
of keys, in order to identify the important key clusters for the current query. Once
the important clusters are identified, the method retrieves their associated keys
and computes exact attention only with those high-scoring keys.

The hierarchical clustering and retrieval scheme efficiently narrows the search
space by first leveraging coarser-grained clusters and then refining the search
using fine-grained clusters. In contrast to existing solutions that identify less
important tokens once and discard them throughout the entire generation, this method
dynamically identifies and retrieves only the information that is semantically
relevant to each generation step. This allows the method to preserve generation
quality while reducing the number of KV cache entries loaded from memory by up to
8 times, including loading key centroids.

The semantic-based key clustering and retrieval works as follows: to cluster
non-consecutive keys by their semantic similarity, K-means clustering is performed
offline, representing all keys within each cluster with a single key centroid value.
This allows identifying semantically relevant keys for the query tokens during
inference by comparing the query against key clusters instead of the entire key set,
and only performing exact attention computation with the most relevant keys. Since
the number of key centroids is significantly smaller than the number of keys, the
memory overhead remains minimal.

A hierarchical version of the method reduces the memory and computational complexity
of the centroid lookup from linear to logarithmic with respect to the fixed context
length. The system implementation includes efficient Triton kernels for performing
the centroid comparison and computing sparse FlashAttention with only the important
keys, achieving 4.3 times and 4.2 times speedups during the prefill and decode phases
when running inference with long fixed context.

PreFixQA is a document QA benchmark which contains a selection of arXiv documents,
each with many synthetic user input question and answer pairs. This benchmark
facilitates research into fixed context methods by allowing evaluation of various
user inputs for each document.

The method is evaluated on long-context benchmarks including LongBench, RULER, and
PreFixQA. On LongBench, the method preserves accuracy with 3.1 times KV budget
reduction and achieves up to 8 times KV budget reduction with 0.5 point accuracy
degradation for the LLaMA-2-7B-32K, LWM-Text-Chat-1M, and Longchat-7B-v1.5-32K
models.
""" * 4  # nhân lên để có context dài hơn


SAMPLE_QUERIES = [
    "What is the main idea of Squeezed Attention?",
    "How much speedup does the method achieve during prefill?",
    "What benchmarks were used for evaluation?",
    "What is hierarchical centroid lookup?",
    "How does K-means clustering help here?",
]


# ---------------------------------------------------------------------------
# Hook helper: thu KV của một forward
# ---------------------------------------------------------------------------
class KVCollector:
    """Thu thập key, value của mỗi layer attention bằng forward hook."""

    def __init__(self, model):
        self.model = model
        self.keys_per_layer = []   # list[layer] of tensor (B, H, N, D)
        self.values_per_layer = []
        self.queries_per_layer = []
        self.handles = []

    def _make_hook(self, layer_idx):
        def hook(module, inputs, outputs):
            # Chúng ta sẽ patch ở chỗ khác, hook này chỉ là placeholder
            pass
        return hook

    def attach(self):
        """
        Cách dễ nhất là dùng `output_attentions=True` không đủ - không trả K,V.
        Ta sẽ monkey-patch forward của attention module để capture K, V sau RoPE.
        Ở đây dùng cách đơn giản: chạy forward với `use_cache=True` và lấy
        từ past_key_values.
        """
        pass  # No-op: dùng past_key_values thay vì hook

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def collect_kv_via_cache(model, input_ids):
    """
    Chạy 1 forward pass với use_cache=True, lấy past_key_values.
    Đây là cách chuẩn nhất - K, V trả về đã qua RoPE.
    
    Returns:
        keys_per_layer:   list of (B, H, N, D)
        values_per_layer: list of (B, H, N, D)
        queries_per_layer: None (Hugging Face không expose Q sau RoPE qua API)
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
    past = outputs.past_key_values  # tuple of (k, v) per layer
    keys = [layer_kv[0].detach() for layer_kv in past]
    values = [layer_kv[1].detach() for layer_kv in past]
    return keys, values


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------
def run_experiment(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 else torch.bfloat16

    print(f"=== Loading {args.model} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="eager",  # cần eager để past_key_values trả K,V đúng cấu trúc
    )
    model.eval()

    # ----- Tokenize fixed context -----
    print(f"=== Tokenizing fixed context (~{len(SAMPLE_CONTEXT)} chars) ===")
    fixed_ids = tokenizer(
        SAMPLE_CONTEXT,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_context,
    ).input_ids.to(device)
    N_fixed = fixed_ids.shape[1]
    print(f"Fixed context length: {N_fixed} tokens")

    # ----- Thu K, V từ fixed context -----
    print("=== Collecting K, V from fixed context ===")
    t0 = time.time()
    keys_layers, values_layers = collect_kv_via_cache(model, fixed_ids)
    print(f"Collected K, V in {time.time() - t0:.2f}s")
    num_layers = len(keys_layers)
    H = keys_layers[0].shape[1]
    D_k = keys_layers[0].shape[3]
    D_v = values_layers[0].shape[3]
    print(f"Layers: {num_layers}, Heads: {H}, D_k: {D_k}, D_v: {D_v}")

    # ----- Quyết định số centroid -----
    obs_window = args.obs_window
    N_to_cluster = N_fixed - obs_window
    num_centroids = max(1, int(args.percent_clusters / 100.0 * N_to_cluster))
    print(f"Num centroids per head: {num_centroids} (={args.percent_clusters}% of {N_to_cluster})")

    # ----- (A) Key-only clustering (baseline Squeezed Attention) -----
    print("=== Running KEY-ONLY clustering (baseline) ===")
    t0 = time.time()
    key_only_data = {}
    for layer_idx in range(num_layers):
        k_l = keys_layers[layer_idx].squeeze(0).to(device).float()  # (H, N, D)
        if obs_window > 0:
            k_l = k_l[:, :-obs_window, :]
        # Normalize cho K-means cosine (như trong code gốc)
        k_norm = F.normalize(k_l, dim=-1)
        centroids, labels = _kmeans_cosine(k_norm, num_centroids, num_iters=args.kmeans_iters)
        # Tính cluster sizes
        sizes = torch.zeros(H, num_centroids, device=device)
        sizes.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.float))
        key_only_data[layer_idx] = (centroids, labels, sizes)
    print(f"Key-only clustering done in {time.time() - t0:.2f}s")

    # ----- (B) Value-aware clustering -----
    print("=== Running VALUE-AWARE clustering (improvement) ===")
    t0 = time.time()
    kc_dict, vc_dict, labels_dict, vvar_dict = run_value_aware_clustering(
        keys_layers,
        values_layers,
        num_centroids=num_centroids,
        observation_window=obs_window,
        alpha=args.alpha,
        beta=args.beta,
        num_iters=args.kmeans_iters,
        device=device,
    )
    nvar_dict = normalize_value_variance(vvar_dict)
    print(f"Value-aware clustering done in {time.time() - t0:.2f}s")

    # Cluster sizes cho value-aware
    va_sizes = {}
    for li, lbl in labels_dict.items():
        s = torch.zeros(H, num_centroids, device=device)
        s.scatter_add_(1, lbl, torch.ones_like(lbl, dtype=torch.float))
        va_sizes[li] = s

    # ----- Đánh giá: với mỗi query, so sánh attention output -----
    print("\n=== EVALUATION ===")
    print(f"Sparsity target: {args.sparsity}, gamma: {args.gamma}\n")

    # Calibrate threshold cho cả 2 phương pháp dùng N tokens cuối làm "queries"
    # (mô phỏng calibration trong paper)
    print("=== Calibrating thresholds ===")
    # Lấy queries calib bằng cách chạy thêm 1 đoạn ngắn rồi lấy queries các layer
    # Ở demo này, ta dùng chính các keys cuối làm proxy cho queries
    # (vì query và key có distribution tương tự sau RoPE trong cùng head)

    # Dùng 50 keys cuối làm calibration queries
    n_calib = min(50, obs_window)

    # Tính thresholds per layer (đơn giản hóa: 1 threshold cho cả model)
    thresholds_keyonly = {}
    thresholds_valueaware = {}
    for layer_idx in range(num_layers):
        # Calib từ keys[-n_calib:]
        full_k = keys_layers[layer_idx].squeeze(0).to(device).float()  # (H, N, D)
        calib_q = full_k[:, -n_calib:, :].permute(1, 0, 2)  # (n_calib, H, D)

        # Key-only (gamma=0 -> tắt boost)
        kc_ko, lbl_ko, sz_ko = key_only_data[layer_idx]
        # Cần chuyển nvar về 0 để mô phỏng baseline
        zero_var = torch.zeros_like(sz_ko)
        T_ko = calibrate_threshold(
            calib_q, kc_ko, sz_ko, zero_var, lbl_ko,
            target_sparsity=args.sparsity, gamma=0.0,
            num_threshold_search=args.threshold_search,
        )
        thresholds_keyonly[layer_idx] = T_ko

        # Value-aware
        T_va = calibrate_threshold(
            calib_q,
            kc_dict[layer_idx].float(),
            va_sizes[layer_idx],
            nvar_dict[layer_idx].float(),
            labels_dict[layer_idx],
            target_sparsity=args.sparsity,
            gamma=args.gamma,
            num_threshold_search=args.threshold_search,
        )
        thresholds_valueaware[layer_idx] = T_va

    print(f"Threshold sample (layer 0): key-only={thresholds_keyonly[0]:.6e}, "
          f"value-aware={thresholds_valueaware[0]:.6e}")

    # ----- Loop qua các query -----
    metrics = {
        "key_only": {"cos": [], "mse": [], "budget": []},
        "value_aware": {"cos": [], "mse": [], "budget": []},
    }

    print("\n=== Running attention output comparison ===")
    for q_idx, query_text in enumerate(SAMPLE_QUERIES):
        print(f"\n--- Query {q_idx}: {query_text!r}")
        # Tokenize và concat sau fixed context
        q_ids = tokenizer(query_text, return_tensors="pt").input_ids.to(device)
        full_ids = torch.cat([fixed_ids, q_ids], dim=1)

        # Lấy queries (sau RoPE) cho từng layer.
        # Cách đơn giản: chạy forward với output_hidden_states - rồi tự tính.
        # Đơn giản hơn nữa: chạy thêm forward với full_ids, lấy last query token's keys
        # làm proxy cho query (cùng distribution sau RoPE).
        # Đây là approximation hợp lệ cho demo.
        all_k_full, all_v_full = collect_kv_via_cache(model, full_ids)
        # Query proxy = key của token cuối (lúc generation, q và k cùng được tính từ cùng hidden state)
        # NOTE: đây là approximation. Trong code production, cần lấy q_proj output thực sự.

        for layer_idx in range(num_layers):
            full_k = keys_layers[layer_idx].squeeze(0).to(device).float()  # (H, N, D)
            full_v = values_layers[layer_idx].squeeze(0).to(device).float()  # (H, N, D)
            # query proxy: lấy key của token cuối
            q_proxy = all_k_full[layer_idx].squeeze(0)[:, -1, :].to(device).float()  # (H, D)

            # Cắt phần fixed context (loại observation_window)
            if obs_window > 0:
                k_clustered = full_k[:, :-obs_window, :]
                v_clustered = full_v[:, :-obs_window, :]
            else:
                k_clustered = full_k
                v_clustered = full_v

            # Baseline full attention (chỉ tính trên phần được cluster)
            scale = 1.0 / (k_clustered.shape[-1] ** 0.5)
            ref_out = baseline_full_attention(q_proxy, k_clustered, v_clustered, scale)

            # Key-only Squeezed Attention
            kc_ko, lbl_ko, sz_ko = key_only_data[layer_idx]
            zero_var = torch.zeros_like(sz_ko)
            ko_out, ko_info = squeezed_attention_forward(
                q_proxy, k_clustered, v_clustered,
                kc_ko, sz_ko, zero_var, lbl_ko,
                threshold=thresholds_keyonly[layer_idx],
                gamma=0.0,
                scale=scale,
            )

            # Value-aware Squeezed Attention
            va_out, va_info = squeezed_attention_forward(
                q_proxy, k_clustered, v_clustered,
                kc_dict[layer_idx].float(),
                va_sizes[layer_idx],
                nvar_dict[layer_idx].float(),
                labels_dict[layer_idx],
                threshold=thresholds_valueaware[layer_idx],
                gamma=args.gamma,
                scale=scale,
            )

            # Tính cos similarity và MSE per head, average
            def cos_sim(a, b):
                return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

            def mse(a, b):
                return F.mse_loss(a, b).item()

            metrics["key_only"]["cos"].append(cos_sim(ko_out, ref_out))
            metrics["key_only"]["mse"].append(mse(ko_out, ref_out))
            metrics["key_only"]["budget"].append(ko_info["kv_budget"])

            metrics["value_aware"]["cos"].append(cos_sim(va_out, ref_out))
            metrics["value_aware"]["mse"].append(mse(va_out, ref_out))
            metrics["value_aware"]["budget"].append(va_info["kv_budget"])

        # Free memory cho query này
        del all_k_full, all_v_full
        torch.cuda.empty_cache()

    # ----- Print summary -----
    print("\n" + "=" * 70)
    print("SUMMARY (averaged over all queries x all layers x all heads)")
    print("=" * 70)
    for name, m in metrics.items():
        avg_cos = sum(m["cos"]) / len(m["cos"])
        avg_mse = sum(m["mse"]) / len(m["mse"])
        avg_budget = sum(m["budget"]) / len(m["budget"])
        print(
            f"{name:>15s}: "
            f"cos_sim_to_full={avg_cos:.4f}  "
            f"MSE={avg_mse:.6f}  "
            f"KV_budget={avg_budget*100:.2f}%"
        )

    # Diff
    cos_ko = sum(metrics["key_only"]["cos"]) / len(metrics["key_only"]["cos"])
    cos_va = sum(metrics["value_aware"]["cos"]) / len(metrics["value_aware"]["cos"])
    print(f"\nValue-aware improvement (cos_sim): {(cos_va - cos_ko)*100:+.3f} percentage points")
    print(
        "Higher cos_sim = output gần với full attention hơn = retrieval tốt hơn cho cùng KV budget."
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HF model id. Mặc định 1.5B chạy tốt trên 24GB.")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--fp16", action="store_true", help="Dùng fp16 thay vì bfloat16")
    p.add_argument("--max_context", type=int, default=4096)
    p.add_argument("--obs_window", type=int, default=64)
    p.add_argument("--percent_clusters", type=float, default=5.0,
                   help="% centroids so với độ dài fixed context (mặc định 5%)")
    p.add_argument("--kmeans_iters", type=int, default=10)
    p.add_argument("--sparsity", type=float, default=0.9,
                   help="% keys muốn drop. 0.9 = giữ ~10% keys.")
    p.add_argument("--gamma", type=float, default=0.3,
                   help="Hệ số boost variance. 0=tắt, 0.3 mặc định, 0.5 aggressive.")
    p.add_argument("--alpha", type=float, default=1.0, help="Trọng số K trong joint clustering")
    p.add_argument("--beta", type=float, default=0.5, help="Trọng số V trong joint clustering")
    p.add_argument("--threshold_search", type=int, default=80,
                   help="Số điểm để binary-search threshold")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
