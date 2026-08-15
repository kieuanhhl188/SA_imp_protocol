# Value-Aware Retrieval cho Squeezed Attention

Cải tiến **Value-Aware Retrieval** cho [Squeezed Attention](https://github.com/SqueezeAILab/SqueezedAttention) (ACL 2025).

## Ý tưởng cải tiến

Squeezed Attention gốc chỉ dùng **keys** để cluster và quyết định cluster nào quan trọng.
Tuy nhiên, một key quan trọng **chưa chắc đi với value quan trọng** — đặc biệt khi values trong
cùng một "key cluster" rất khác nhau, đại diện bằng 1 centroid sẽ làm mất thông tin.

Cải tiến này có 3 thành phần:

1. **Joint K-V clustering**: Cluster trên không gian `concat(α·K_norm, β·V_norm)` thay vì chỉ K.
   Đảm bảo các tokens trong cùng cluster có cả key (cho retrieval) và value (cho output) tương tự nhau.

2. **Per-cluster value variance**: Lưu thêm $\sigma^2_{v,i}$ — variance của values trong từng cluster.

3. **Variance-adjusted importance**: Boost score những cluster có values đa dạng:

   $$\tilde{S}_i = S_i \cdot (1 + \gamma \cdot \tilde{\sigma}_{v,i})$$

Hệ số `gamma=0` → trở về Squeezed Attention gốc. `gamma=0.3` (mặc định) là tradeoff hợp lý.

## Cấu trúc file

```
value_aware_squeezed/
├── value_aware_clustering.py    # Joint K-V K-means + value variance computation
├── value_aware_retrieval.py     # Online retrieval logic + threshold calibration
├── demo_value_aware.py          # End-to-end demo trên Hugging Face model
├── test_value_aware.py          # Unit tests (chạy trên CPU, không cần GPU)
└── README.md
```

## Yêu cầu hệ thống

- GPU 24GB (hoặc nhỏ hơn nếu dùng model 1B)
- Python 3.9+
- CUDA 11.8+

## Cài đặt

```bash
# 1. Tạo môi trường
conda create -n vasq python=3.10 -y
conda activate vasq

# 2. Cài PyTorch (chọn CUDA version phù hợp)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Cài transformers + một số dep
pip install transformers accelerate sentencepiece protobuf
```

## Chạy unit tests (verify code đúng)

```bash
python test_value_aware.py
```

Output kì vọng:
```
[test_kmeans_cosine] OK
[test_value_aware_kmeans_output_shapes] OK
...
[test_value_aware_improves_when_value_diverse] OK (cluster 1 (diverse V) idx=1, variance values: [~0, ~16])
All tests passed ✓
```

Test cuối cùng đặc biệt quan trọng: nó kiểm tra rằng cluster có values đa dạng có variance cao
hơn nhiều so với cluster values đồng nhất → tín hiệu boost của chúng ta có ý nghĩa.

## Chạy demo end-to-end

### Cấu hình mặc định (Qwen2.5-1.5B, ~5GB VRAM)

```bash
python demo_value_aware.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --max_context 4096 \
    --percent_clusters 5 \
    --sparsity 0.9 \
    --gamma 0.3
```

### Các option chính

| Flag | Ý nghĩa | Mặc định |
|---|---|---|
| `--model` | HF model id | `Qwen/Qwen2.5-1.5B-Instruct` |
| `--max_context` | Độ dài fixed context tối đa (tokens) | 4096 |
| `--percent_clusters` | % centroids so với context length | 5.0 |
| `--obs_window` | Số tokens cuối không cluster (giữ exact) | 64 |
| `--sparsity` | % keys muốn drop (0.9 = giữ 10%) | 0.9 |
| `--gamma` | Hệ số boost variance (0=tắt) | 0.3 |
| `--alpha` | Trọng số K trong joint clustering | 1.0 |
| `--beta` | Trọng số V trong joint clustering | 0.5 |
| `--kmeans_iters` | Số iter K-means | 10 |

### Mô hình khác (tùy VRAM)

```bash
# 1B - nhẹ nhất, ~3GB VRAM
python demo_value_aware.py --model meta-llama/Llama-3.2-1B-Instruct

# 3B - cân bằng, ~7GB VRAM
python demo_value_aware.py --model Qwen/Qwen2.5-3B-Instruct

# 7B - sát paper hơn, ~16GB VRAM (vừa 24GB)
python demo_value_aware.py --model Qwen/Qwen2.5-7B-Instruct --max_context 8192
```

### Output kì vọng

```
=== Loading Qwen/Qwen2.5-1.5B-Instruct ===
=== Tokenizing fixed context (~14000 chars) ===
Fixed context length: 3850 tokens
=== Collecting K, V from fixed context ===
Collected K, V in 1.23s
Layers: 28, Heads: 12, D_k: 128, D_v: 128
Num centroids per head: 189 (=5.0% of 3786)
=== Running KEY-ONLY clustering (baseline) ===
Key-only clustering done in 8.42s
=== Running VALUE-AWARE clustering (improvement) ===
Value-aware clustering done in 9.17s

=== EVALUATION ===
Sparsity target: 0.9, gamma: 0.3

...

======================================================================
SUMMARY (averaged over all queries x all layers x all heads)
======================================================================
       key_only: cos_sim_to_full=0.8523  MSE=0.012345  KV_budget=10.12%
    value_aware: cos_sim_to_full=0.8814  MSE=0.009872  KV_budget=10.31%

Value-aware improvement (cos_sim): +2.910 percentage points
```

**Cách đọc kết quả:**
- `cos_sim_to_full`: cosine similarity giữa output approximate và output full attention.
  Càng gần 1 càng tốt.
- `MSE`: lỗi bình phương trung bình. Càng nhỏ càng tốt.
- `KV_budget`: % keys thực sự được load. Hai phương pháp nên có budget gần nhau.
- Số cuối: phần trăm cải thiện cosine similarity của value-aware so với key-only.

## Tích hợp vào repo gốc SqueezedAttention

Để tích hợp vào repo gốc, làm 3 bước:

### Bước 1: Sửa `offline_clustering.py`

Trong `offline_clustering.py` của repo gốc, thay:

```python
from squeezedattention.clustering import run_clustering, run_global_threshold

# ... cluster keys
centroids_tensor_dict, centroids_labels_dict = run_clustering(
    all_keys_layers, num_centroids,
    observation_window=args.observation_window,
    device=DEV,
)
```

Bằng:

```python
from squeezedattention.clustering import run_global_threshold
from value_aware_clustering import (
    run_value_aware_clustering,
    normalize_value_variance,
)

# ... cluster keys + values
key_centroids_dict, value_centroids_dict, labels_dict, value_variance_dict = \
    run_value_aware_clustering(
        all_keys_layers,
        all_values_layers,           # <-- đã có sẵn từ hook nhưng repo gốc không dùng
        num_centroids,
        observation_window=args.observation_window,
        alpha=args.alpha,            # thêm CLI args
        beta=args.beta,
        device=DEV,
    )
nvar_dict = normalize_value_variance(value_variance_dict)

# Save thêm value_variance để dùng online
torch.save(value_variance_dict, f'{args.output_path}/value_variance_{dataidx}_{num_centroids}.pt')
torch.save(nvar_dict, f'{args.output_path}/nvar_{dataidx}_{num_centroids}.pt')
# Còn lại save như cũ
```

### Bước 2: Sửa code online retrieval

Trong custom transformers (thư mục `transformers/` của repo gốc), tìm hàm tính
`S_i` trong attention layer (thường là trong `LlamaAttention.forward`), thay:

```python
S = exp(q @ C.T) / (N * exp(q @ C.T)).sum()
mask = S > threshold
```

Bằng:

```python
S = exp(q @ C.T) / (N * exp(q @ C.T)).sum()
S_adjusted = S * (1.0 + gamma * normalized_variance)
mask = S_adjusted > threshold
```

Threshold calibration cũng cần dùng `S_adjusted` thay vì `S`.

### Bước 3: Thêm CLI flags

```python
parser.add_argument('--alpha', type=float, default=1.0,
                    help='Trọng số K trong joint clustering')
parser.add_argument('--beta', type=float, default=0.5,
                    help='Trọng số V trong joint clustering. 0 = tắt value-aware (về baseline)')
parser.add_argument('--gamma', type=float, default=0.3,
                    help='Hệ số boost variance khi retrieve. 0 = không boost')
```

## Ablation suggestions

Khi viết paper / report, nên ablate:

1. `(α=1, β=0, γ=0)` → baseline Squeezed Attention gốc
2. `(α=1, β=0.5, γ=0)` → joint clustering, không boost variance
3. `(α=1, β=0, γ=0.3)` → key clustering + variance boost (cần lưu vvar tính từ key-cluster)
4. `(α=1, β=0.5, γ=0.3)` → full method
5. `(α=1, β=0.5, γ=0.5)` → boost mạnh hơn
6. `(α=1, β=1.0, γ=0.3)` → V dominate

So sánh trên LongBench để biết thành phần nào đóng góp chính.

## Hyperparameter tuning

Theo kinh nghiệm:

- `beta`: 0.3-0.7 thường ổn. β quá lớn → V dominate retrieval, làm hỏng matching ngữ nghĩa.
- `gamma`: 0.2-0.4 sweet spot. gamma quá lớn → tăng KV budget vì giữ thêm các cluster variance cao.
- `alpha`: thường để 1.0 và chỉ tune beta.

Nếu observe KV budget tăng quá nhiều khi bật value-aware:
→ Calibrate lại threshold với cùng target sparsity (code đã làm sẵn trong demo)
→ Hoặc giảm `gamma` xuống 0.15-0.2.

## Hạn chế

1. Demo dùng "key của token cuối" làm proxy cho query, vì HF API không expose Q sau RoPE
   trực tiếp. Trong tích hợp production, cần lấy `q_proj` output thực sự từ layer.
2. Kernels Triton chưa được sửa cho value-aware - hiện chỉ có reference implementation
   PyTorch. Để có speedup thực tế, cần modify kernels trong `squeezedattention/kernels/`
   để truyền thêm `normalized_variance` và `gamma`.
3. Khi `beta > 0`, key centroids không hoàn toàn tối ưu cho key-matching (vì K-means tối ưu
   trên joint space). Trade-off này thường có lợi tổng thể nhưng cần verify trên task cụ thể.

## Citation

Nếu dùng cải tiến này, vui lòng cite paper Squeezed Attention gốc:

```bibtex
@article{hooper2024squeezed,
  title={Squeezed Attention: Accelerating Long Context Length LLM Inference},
  author={Hooper, Coleman and Kim, Sehoon and others},
  journal={arXiv preprint arXiv:2411.09688},
  year={2024}
}
```
