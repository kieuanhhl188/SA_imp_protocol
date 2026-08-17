# Tổng hợp kiến thức — Structure-Aware Squeezed Attention

Tài liệu này gom toàn bộ nền kiến thức cần để hiểu và làm việc trên repo: bài báo gốc,
Transformer, K-means, structure-aware clustering, và các khái niệm liên quan. Mọi công thức
đều được đối chiếu với code thật trong repo, có trỏ tới file/dòng.

Đây là tài liệu **kiến thức**, không phải nhật ký tiến độ (xem [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md))
và không phải hướng dẫn chạy (xem [docs/PHASE0.md](docs/PHASE0.md)).

**Mục lục**

1. [Bức tranh tổng thể](#1-bức-tranh-tổng-thể)
2. [Bảng ký hiệu](#2-bảng-ký-hiệu)
3. [Nền tảng Transformer](#3-nền-tảng-transformer)
4. [Bài báo gốc — Squeezed Attention](#4-bài-báo-gốc--squeezed-attention)
5. [K-means](#5-k-means)
6. [Structure-aware clustering (đóng góp của ta)](#6-structure-aware-clustering-đóng-góp-của-ta)
7. [Kiến thức liên quan khác](#7-kiến-thức-liên-quan-khác)
8. [Từ điển thuật ngữ](#8-từ-điển-thuật-ngữ)
9. [Cheat sheet công thức](#9-cheat-sheet-công-thức)
10. [Bản đồ file → khái niệm](#10-bản-đồ-file--khái-niệm)

---

## 1. Bức tranh tổng thể

### Vấn đề

Suy luận LLM với **prompt dài** rất đắt. Cụ thể có hai chi phí khác nhau, hay bị gộp làm một:

| Giai đoạn | Việc | Chi phí theo độ dài `S` |
|---|---|---|
| **Prefill** | nạp cả prompt, tính attention toàn bộ | `O(S²)` FLOPs |
| **Decode** | sinh từng token, mỗi token đọc lại toàn bộ KV cache | `O(S)` **băng thông bộ nhớ** mỗi token |

Với context 128K, KV cache của LLaMA-7B đã hơn 60 GB. Decode trở thành **memory-bound**:
GPU không thiếu FLOPs, nó thiếu băng thông để kéo KV cache từ HBM về mỗi bước.

### Quan sát mà bài báo khai thác

Trong rất nhiều ứng dụng thật, **phần lớn prompt là cố định giữa các query liên tiếp**:

- người dùng hỏi nhiều câu về **cùng một tài liệu**
- agent code hỏi nhiều câu về **cùng một codebase**
- system prompt + few-shot examples dùng lại cho mọi request

Gọi phần cố định đó là **fixed context** (còn gọi shared prefix), phần thay đổi là **user query**.

> **Premise then chốt:** nếu fixed context dùng lại nhiều lần, ta có thể trả **một lần** chi
> phí xử lý offline nó, rồi **khấu hao** qua mọi query sau đó. Đây là toàn bộ lý do tồn tại
> của phương pháp — và là lý do một benchmark mà mỗi query có context riêng
> (CrossCodeEval, RepoBench nguyên bản) **không đo được** gì cả (xem [§7.4](#74-benchmark)).

### Ý tưởng Squeezed Attention

```
                        OFFLINE (một lần / fixed context)
   fixed context ──► forward pass ──► key vectors K ──► K-means ──► centroids C + labels
                                                                    + threshold τ

                        ONLINE (mỗi user query)
   query q ──► so q với K centroid (rẻ: K ≈ 5%·S)
            ──► cluster nào điểm > τ thì "quan trọng"
            ──► chỉ tính attention với key thuộc các cluster đó   (~10-30% số key)
```

Thay vì `q · k_i` cho **mọi** `i ∈ [1, S]`, ta làm hai bước:
so sánh thô với `K` centroid rồi mới so chi tiết với các key được chọn.

### Ý tưởng cải tiến của ta (Structure-Aware)

K-means thuần chỉ nhìn khoảng cách trong không gian embedding. Với **code**, cấu trúc cú pháp
(function, class, block) là tín hiệu mạnh mà K-means bỏ qua: một cluster có thể vắt qua hai
function không liên quan chỉ vì embedding tình cờ gần nhau.

Đề xuất: **ép ranh giới cứng theo AST** — K-means chạy độc lập trong từng đơn vị cấu trúc,
không cluster nào được chứa token của hai function khác nhau.

Ba giả thuyết cần chứng minh:

| Claim | Nội dung | Đo bằng |
|---|---|---|
| **C1** | Accuracy cao hơn ở **cùng budget** | Phase 6 |
| **C2** | Retrieval tốt hơn: `Recall@budget` cao hơn (đây là **H0**, rẻ nhất, chạy trước) | Phase 5 |
| **C3** | Re-clustering **incremental** khi code bị sửa: nhanh hơn nhiều lần, mất < 0.3 điểm | Phase 4 + 7 |

---

## 2. Bảng ký hiệu

Ký hiệu dùng thống nhất trong tài liệu này và khớp với tên biến trong code.

| Ký hiệu | Tên trong code | Ý nghĩa |
|---|---|---|
| `S` | `shared_prefix_length` | độ dài fixed context (số token) |
| `W` | `observation_window` | cửa sổ quan sát, **100** token cuối của fixed context |
| `S_ctx` | `n_ctx = sp_len - obs_window` | phần fixed context thật sự được cluster = `S − W` |
| `L` | `num_hidden_layers` | số layer (LLaMA-7B: 32) |
| `H` | `num_attention_heads` | số **query head** (LLaMA-7B: 32; Qwen2.5-Coder-7B: 28) |
| `H_kv` | `num_key_value_heads` | số **KV head**. MHA: `H_kv = H`. GQA Qwen2.5-Coder: `H_kv = 4` |
| `G` | `num_key_value_groups` | `G = H / H_kv`, số query head dùng chung 1 KV head |
| `D` | `head_dim` | chiều của mỗi head (thường 128) |
| `d_model` | `hidden_size` | `H · D` (4096) |
| `K` | `num_clusters` | số centroid mức đơn tầng / mức L2. Mặc định `K = 5% · S_ctx` |
| `K1` | `num_clusters` (L1) | số centroid mức L1 khi hierarchical. Mục tiêu `1% · S_ctx` |
| `K2` | `num_clusters_l2` | số centroid mức L2 khi hierarchical = `5% · S_ctx` |
| `U` | `num_units` | số đơn vị cấu trúc (AST unit) trong một sample |
| `k_u` | `k_per_unit` | số centroid cấp cho unit `u` |
| `n_j` | `num_keys_per_cluster` | số key thuộc cluster `j` |
| `c_j` | `centroids_tensor` | vector centroid thứ `j` |
| `ℓ(t)` | `centroid_labels` | nhãn cluster của token `t` |
| `τ` | `global_threshold` | ngưỡng toàn cục để quyết định giữ cluster |
| `p` | `percentile` | mức sparsity: 0.7 / 0.8 / 0.9 |
| `S_i` | — | điểm quan trọng của cluster/token (xem [§4.3](#43-công-thức-điểm-s_i)) |

**Layout tensor** (rất hay nhầm, ghi rõ ở đây):

| Tensor | Shape | Ghi chú |
|---|---|---|
| key/query states | `[B, H, S, D]` | `B = 1`, không hỗ trợ batch |
| `centroids_tensor_dict[layer]` | `[1, H_kv, K, D]` | dict theo layer, **không** phải một tensor lớn |
| `centroids_labels_dict[layer]` | `[1, H_kv, S_ctx]` | dtype `int64` |
| `global_threshold_dict` | dict `{0.5: τ, 0.7: τ, 0.8: τ, 0.9: τ, 'shared_prefix_length': S, 'observation_window': W}` | |

⚠️ Online eval đọc `key_centroids.shape[2]` để lấy `K`
([modeling_llama.py:663](transformers/src/transformers/models/llama/modeling_llama.py#L663)).
Trả về `[H, K, D]` thay vì `[1, H, K, D]` sẽ **im lặng** lấy nhầm `D` làm `K`.

---

## 3. Nền tảng Transformer

### 3.1 Scaled dot-product attention

Với một head:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √D ) V
```

Từng phần tử, cho query token `i` và key token `t`:

```
        exp( q_i · k_t / √D )
a_it = ─────────────────────────
       Σ_s exp( q_i · k_s / √D )

out_i = Σ_t a_it · v_t
```

- **`q_i · k_t`** — tích vô hướng, đo mức "khớp" giữa query và key. Càng lớn càng liên quan.
- **`/ √D`** — *scaled*. Nếu các thành phần của `q`, `k` độc lập, phương sai của tích vô hướng
  tỉ lệ với `D`; chia `√D` giữ phương sai ~1 để softmax không bão hoà (gradient triệt tiêu).
- **`softmax`** — chuẩn hoá thành phân phối xác suất, tổng bằng 1. **Mẫu số** của softmax
  (`Σ_s exp(...)`) chính là chỗ mà Squeezed Attention phải **ước lượng**, vì nó không tính
  hết mọi key nữa. Xem [§4.3](#43-công-thức-điểm-s_i).
- **Causal mask** — token chỉ được nhìn về quá khứ: `a_it = 0` khi `t > i`.

### 3.2 Multi-head, MHA / MQA / GQA

Mỗi layer chia `d_model` thành `H` head chạy song song, mỗi head có `Q/K/V` riêng, rồi nối
lại và đi qua `o_proj`. Ý nghĩa: mỗi head học một kiểu quan hệ khác nhau.

Ba biến thể khác nhau ở **số KV head**:

| Biến thể | `H_kv` | KV cache | Model trong repo |
|---|---|---|---|
| **MHA** (Multi-Head Attention) | `= H` | lớn nhất | LLaMA-2-7B-32K, LongChat-v1.5-7B-32K |
| **GQA** (Grouped-Query Attention) | `1 < H_kv < H` | nhỏ hơn `G` lần | **Qwen2.5-Coder-7B: 28 Q / 4 KV → G = 7** |
| **MQA** (Multi-Query Attention) | `= 1` | nhỏ nhất | — |

GQA: `G` query head dùng chung một cặp `(K, V)`. Ánh xạ chuẩn của transformers là

```
query head h  →  KV head  ⌊h / G⌋
```

Trong code, `repeat_kv` nhân bản KV head lên đủ `H`. Điều này sinh ra **ba hệ quả quan trọng**
cho repo (xem [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) mục Phase 1.5/1.6):

1. **Cluster trên key TRƯỚC `repeat_kv`.** Sau `repeat_kv` các bản sao **giống hệt nhau** →
   cluster lại tốn gấp `G` lần công và dung lượng mà ra cùng kết quả. Nên centroid được lưu
   với `H_kv` head, rồi `repeat_interleave` lên `H` head ngay trước lookup
   ([clustering.py:138-148](squeezedattention/clustering.py#L138-L148)).
2. **`repeat_interleave` ≠ `repeat`.** `repeat_interleave(x, G, dim=0)` cho `h → ⌊h/G⌋` (đúng).
   `x.repeat(G, 1, 1)` cho `h → h mod H_kv` (**sai**) — mà vẫn chạy trơn tru, chỉ tra nhầm
   nhóm centroid rồi tụt accuracy không rõ lý do.
3. **Số head lookup** phải lấy từ `centroid_labels.shape[-2]`, không phải `key_states.shape[1]`,
   vì tại điểm đó `repeat_kv` chưa chạy.

Theo Appendix G của bài, mỗi query head **tự chọn tập key riêng** — nên phải nhân bản centroid
lên `H` head chứ không được gộp điểm của `G` head lại.

### 3.3 KV cache, prefill vs decode

Khi sinh token tự hồi quy, `K` và `V` của các token đã xử lý được **cache** lại để không phải
tính lại. Kích thước:

```
KV cache (bytes) = 2 · L · H_kv · D · S · sizeof(dtype)
```

Ví dụ LLaMA-7B (L=32, H_kv=32, D=128) bf16, S=31.500:
`2 · 32 · 32 · 128 · 31500 · 2 B ≈ 16,5 GB`.

Trong repo có một công thức thực dụng cho **peak VRAM** khi chạy offline clustering:

```
peak ≈ 13,5 GB (weights bf16)  +  S × 0,79 MB
```

Nguồn của `0,79 MB/token`: hook thu **cả Q, K, V** của mọi layer ở bf16:
`3 · 32 layer · 4096 · 2 B = 786 KB/token`. Bỏ `all_values_layers` (values **không hề được
dùng**) sẽ cắt 1/3 — đây là "lỗi đã biết" số 4 trong EXPERIMENT_LOG mục 8.

### 3.4 RoPE — Rotary Position Embedding

Vị trí được mã hoá bằng cách **xoay** `q` và `k` trong từng cặp chiều, góc xoay tỉ lệ với vị
trí. Kết quả: `q_i · k_t` chỉ phụ thuộc vào **khoảng cách tương đối** `i − t`.

Hai điểm liên quan trực tiếp tới repo:

- **Clustering phải chạy trên key SAU khi apply RoPE.** Hook được gắn ở
  `layer.self_attn` và trả về `qkv` đã qua `apply_rotary_pos_emb`
  ([modeling_llama.py:633](transformers/src/transformers/models/llama/modeling_llama.py#L633)).
  Cluster trên key trước RoPE sẽ không khớp với thứ mà attention online thật sự dùng.
- **`rope_scaling`** — kéo dài context ngoài mức train bằng cách nội suy góc.
  LongChat-v1.5-7B-32K dùng `linear factor 8` (train 4K → dùng 32K).

### 3.5 FlashAttention và Triton

**FlashAttention** tính attention mà **không bao giờ materialize** ma trận `S × S`. Nó chia
`K/V` thành block, chạy softmax **online** (giữ running max `m` và running sum `ℓ`, rescale khi
gặp giá trị lớn hơn). Kết quả: bộ nhớ `O(S)` thay vì `O(S²)`, và nhanh hơn vì ít đọc/ghi HBM.

**Triton** là DSL để viết GPU kernel bằng Python. Repo có 3 kernel trong
[squeezedattention/kernels.py](squeezedattention/kernels.py):

| Kernel | Việc |
|---|---|
| `_fwd_centroid_simple_kernel_qk` | tính điểm query→centroid, ra mask key nào giữ |
| `_fwd_kernel_qk` | sparse FlashAttention trên tập key đã chọn |
| `_reduce_kernel_qk` | gộp kết quả các block (merge running max/sum) |

⚠️ Kernel dùng `tl.math.exp2` — API **Triton 2.x**. Triton 3.x đổi API. Đây là lý do stack bị
ghim ở `torch 2.3.1 + triton 2.3.1` (xem EXPERIMENT_LOG mục "stack đã kiểm chứng").

⚠️ `kernels.py` dòng 3 có `import pytest` sót từ repo gốc, mà `modeling_llama.py` import
kernels ở top-level → **thiếu `pytest` thì `from transformers import LlamaForCausalLM` cũng chết**.

---

## 4. Bài báo gốc — Squeezed Attention

> Hooper, Kim, Mohammadzadeh, Maheswaran, Paik, Mahoney, Keutzer, Gholami.
> *Squeezed Attention: Accelerating Long Context Length LLM Inference.*
> arXiv:2411.09688 · ACL 2025 (`2025.acl-long.1568`).

### 4.1 Kiến trúc hai pha

**Pha offline** — chạy một lần cho mỗi fixed context ([offline_clustering.py](offline_clustering.py)):

1. Forward toàn bộ prompt qua model, hook thu `Q, K` của mọi layer
   (`config.return_qkv_states = True` mở đường trả qkv trong forward đã patch).
2. Cắt lấy `S` token đầu (fixed context), bỏ `W = 100` token cuối (observation window).
3. `run_clustering` — K-means trên key của **từng layer, từng head độc lập**.
4. `run_global_threshold` — dùng query trong observation window để **calibrate** ngưỡng `τ`.
5. Lưu ra đĩa: `centroids_tensor_dict_*.pt`, `centroids_labels_dict_*.pt`, `global_threshold_*.pt`.

**Pha online** — mỗi user query ([modeling_llama.py:598-820](transformers/src/transformers/models/llama/modeling_llama.py#L598-L820)):

1. Nạp centroid/label/threshold của sample tương ứng (`different_prefix_index`).
2. `centroid_lookup` — tính điểm query→centroid, so với `τ`, ra `mask [H, S]`.
3. `compute_k_idx_optimized` — mask → danh sách chỉ số key được chọn, đóng block cho kernel.
4. Ba nhánh attention được **gộp** lại:
   - `causal_attention_kernel` — query mới ↔ chính nó (causal)
   - `dynamic_sparse_attention` — query mới ↔ **tập key đã chọn** của fixed context
   - `_flash_attention_forward` — attention nội bộ trong fixed context (không đổi)

`reset_context = hidden_states.shape[1] > 1` — nghĩa là centroid được nạp lại đúng một lần ở
bước prefill của mỗi sample, không nạp lại ở từng bước decode.

### 4.2 Observation window — vì sao có

`W = 100` token **cuối** của fixed context được **giữ nguyên, không cluster**. Hai vai trò:

1. **Proxy để calibrate.** Lúc offline ta chưa biết user sẽ hỏi gì. 100 token cuối của context
   là thứ gần nhất với "query sẽ tới" → dùng làm query giả để đo phân phối điểm và chọn `τ`.
2. **Luôn được giữ online.** Trong kernel, các token obs window được gán nhãn đặc biệt `K`
   (một chỉ số ngoài dải `[0, K)`) và điểm được đặt cứng bằng `1/num_query_blocks` để tổng
   thành 1 — luôn `> τ`, nên **không bao giờ bị loại**
   ([modeling_llama.py:453, 673-675](transformers/src/transformers/models/llama/modeling_llama.py#L453)).

Nguồn: Appendix C của bài.

### 4.3 Công thức điểm `S_i`

Đây là **trái tim của phương pháp**. Code:
[squeezedattention/clustering.py:150-181](squeezedattention/clustering.py#L150-L181).

**Bước 1 — điểm query → centroid.** Với query `i` trong observation window và centroid `j`:

```
A_ij = (q_i · c_j) / √D
```

**Bước 2 — gán điểm centroid xuống từng token.** Token `t` thuộc cluster `ℓ(t)`:

```
score(t, i) = A_{i, ℓ(t)}
```

Tức là: *mọi token trong cùng một cluster nhận cùng một điểm*. Đây chính là phép xấp xỉ —
ta thay `q_i · k_t` bằng `q_i · c_{ℓ(t)}`.

**Bước 3 — ước lượng mẫu số softmax.** Không tính được `Σ_t exp(q_i·k_t/√D)` vì không muốn
đụng vào mọi key. Thay bằng: mỗi cluster đóng góp `n_j` lần giá trị của centroid nó:

```
Z_i ≈ Σ_j  n_j · exp(A_ij)          ← n_j = số key trong cluster j
```

**Bước 4 — điểm chuẩn hoá dạng softmax.**

```
                 exp( A_{i, ℓ(t)} )
p(t, i)  =  ───────────────────────
                      Z_i
```

**Bước 5 — trung bình qua mọi query của observation window.**

```
            1   W
S_t   =    ───  Σ   p(t, i)
            W  i=1
```

Kết quả `full_centroid_scores` có shape `[L, H, S_ctx]` — một điểm cho mỗi (layer, head, token).

**Bước 6 — ngưỡng toàn cục.**

```
τ_p  =  Quantile_p ( toàn bộ S_t qua MỌI layer, MỌI head, MỌI token )
        với p ∈ {0.5, 0.7, 0.8, 0.9}
```

Điểm cần nhớ: đây là **một con số duy nhất cho cả model**, không phải per-layer hay per-head.
Vì vậy các layer/head khác nhau tự nhiên sẽ giữ số key khác nhau — layer nào có attention
tập trung thì ít key vượt ngưỡng hơn. Đó là cơ chế *adaptive tự nhiên* của bài.

⚠️ `qlist = [0.5, 0.7, 0.8, 0.9]` **hard-code** tại
[clustering.py:191](squeezedattention/clustering.py#L191). Muốn mức sparsity khác phải sửa
`qlist` **và chạy lại toàn bộ offline clustering**.

**Online** dùng đúng công thức đó, nhưng query là user query thật, và điểm được kernel tính
rồi `gather` xuống token qua `centroid_labels`
([modeling_llama.py:474-484](transformers/src/transformers/models/llama/modeling_llama.py#L474-L484)):

```
mask[h, t] = ( avg_score[h, ℓ(t)] > τ_p )
```

### 4.4 Hierarchical (phân tầng)

Với context rất dài, `K = 5%·S` vẫn lớn (S=128K → K=6400 centroid để so mỗi query). Bài thêm
một tầng nữa:

```
                  K1 = 1%·S centroid          ← thô, lọc bỏ 50% key
   query ──────►  L1 lookup, ngưỡng τ_{0.5}
                        │
                        ▼  key còn sống
                  K2 = 5%·S centroid          ← mịn, chọn tập cuối
                  L2 lookup, ngưỡng τ_p
```

**Cách bài dựng L1:** chạy K-means **trên chính các centroid L2**
(`run_clustering(centroids_tensor_dict_l2, num_centroids, observation_window=0)`) — tức
"K-means của K-means" ([offline_clustering.py:186-189](offline_clustering.py#L186-L189)).
Rồi `torch.gather(label_l1, -1, label_l2)` để đổi ánh xạ `L1→L2` thành `L1→key`.

**Online**, hierarchical được **mô phỏng bằng hai lần lookup một tầng** rồi `AND` hai mask
([modeling_llama.py:677-723](transformers/src/transformers/models/llama/modeling_llama.py#L677-L723)).
Chi tiết tinh tế: `num_keys_l2` được nhân với `l1_mask` — tức cluster L2 đã bị L1 loại thì
`n_j = 0`, không còn đóng góp vào mẫu số `Z_i` nữa.

Tham số chốt (Section 6.1): **L1 = 1%, L2 = 5%, ngưỡng L1 loại 50% key** (`--percentile_lower 0.5`).

⚠️ Bug đã sửa trong Phase 0: offline lưu tên `hierarchical_lookup_*` còn online load
`hierarchical_centroids_*` → hierarchical crash. Xem EXPERIMENT_LOG mục 6, bug #2.

### 4.5 Budget — cách tính, và vì sao 0.325

**Budget** = phần trăm KV cache thực sự phải nạp, **đã tính cả metadata centroid**. Đây là
trục so sánh của mọi thí nghiệm ("matched budget"), nên phải hiểu chính xác.

Metadata: centroid chỉ là **key**, không có value. Mà KV cache gồm cả K lẫn V. Nên:

```
metadata (đơn tầng)     = 5% key          = 5%/2  = 2,5% KV cache
metadata (phân tầng)    = (5% + 1%) key   = 6%/2  = 3,0% KV cache
```

Kiểm chứng với Table 2:

| Config | Key giữ lại | + metadata | = Budget | Table 2 |
|---|---|---|---|---|
| Sq-70% | 30% | 2,5% | 0,325 | **0.325** ✓ |
| Sq-80% | 20% | 2,5% | 0,225 | **0.225** ✓ |
| Sq-90% | 10% | 2,5% | 0,125 | **0.125** ✓ |
| H-Sq-90% | ~9,2% | 3,0% | 0,122 | **0.122** ✓ |

Đây là lý do **`K1` lệch danh nghĩa thì budget lệch** — và là lý do `build_l1_groups` phải ép
`K1` về đúng 1% (xem [§6.6](#66-vấn-đề-k1-và-build_l1_groups)).

### 4.6 Số cần khớp — Table 2 (gate Phase 0)

LongChat-7B-v1.5-32K, hai task code của LongBench. Tolerance **±0.3 điểm**.

| Config | Budget | LCC | RepoBench-P |
|---|---|---:|---:|
| All KV | 1.000 | 56.64 | 53.20 |
| **Sq-70%** | 0.325 | **56.93** | **54.64** |
| Sq-80% | 0.225 | 57.17 | 52.83 |
| Sq-90% | 0.125 | 56.95 | 51.57 |
| H-Sq-90% | 0.122 | 57.20 | 51.89 |

Nhận xét đáng chú ý: ở Sq-70% và Sq-80%, điểm **cao hơn All-KV**. Không phải lỗi — bỏ bớt key
nhiễu đôi khi giúp model. Đây là hiện tượng quen thuộc trong sparse attention.

⚠️ **Đừng dùng số trong `LongBench/README.md`** (LCC 53.0 / RB 55.3). Đó là All-KV của
LongBench gốc với prompt/truncation khác — so vào sẽ FAIL oan.

### 4.7 Chi phí offline — đã đo thật

Từ EXPERIMENT_LOG (đo trên A100-SXM 80GB):

| | LCC | RepoBench-P |
|---|---|---|
| Độ dài trung bình | 4.290 token | ~15.900 token |
| Thời gian/sample | **42,5 giây** | **~4-5 phút** |
| Đĩa/sample | ~146 MB | ~517-748 MB |
| GPU util | 54-61% (≈40% chờ I/O) | — |
| Tổng 500 sample | ~6 giờ / 73 GB | ~37 giờ / 325 GB |

**Công thức đĩa: ≈ 34 KB/token.** Suy ra được:

```
centroid: L · H · K · D · 4 B (fp32),  với K = 5%·S_ctx
labels:   L · H · S · 8 B (int64)

S=15.900:  32·32·790·128·4 = 414 MB  +  32·32·15900·8 = 130 MB  =  544 MB
           → 544 MB / 15.900 token ≈ 34 KB/token          (đo thật: 517 MB ✓)
```

**Cải tiến đã xác định, chưa áp dụng:** lưu centroid ở **fp16** → giảm nửa dung lượng và phần
lớn thời gian chờ I/O. Centroid chỉ dùng để so cosine, fp16 dư chính xác.

> **Ghi nhớ khi viết paper:** clustering là **chi phí offline một lần**, KHÔNG nằm trong
> inference latency. Phải nói rõ điều này, nhưng cũng phải báo cáo nó (đó chính là claim C3).

---

## 5. K-means

### 5.1 Bài toán

Cho `n` điểm `x_1..x_n ∈ ℝ^D`, tìm `K` centroid `c_1..c_K` cực tiểu **quán tính** (inertia,
within-cluster sum of squares):

```
J  =  Σ_{t=1}^{n}  ‖ x_t − c_{ℓ(t)} ‖²      với  ℓ(t) = argmin_j ‖x_t − c_j‖²
```

Bài toán NP-hard nói chung. Thực tế dùng **thuật toán Lloyd**, hội tụ về cực tiểu địa phương.

### 5.2 Thuật toán Lloyd

Lặp hai bước tới khi ổn định (hoặc hết `n_iter`):

```
(1) ASSIGN   ℓ(t) ← argmin_j ‖x_t − c_j‖²           gán mỗi điểm về centroid gần nhất
(2) UPDATE   c_j  ← mean{ x_t : ℓ(t) = j }          centroid ← trung bình cụm của nó
```

Đảm bảo `J` không tăng qua mỗi vòng. Không đảm bảo tìm được tối ưu toàn cục.

**Khởi tạo** quyết định rất nhiều tới kết quả:

| Cách | Mô tả | Dùng ở đâu |
|---|---|---|
| **random** | chọn ngẫu nhiên `K` điểm | — |
| **k-means++** | chọn tuần tự, điểm càng xa centroid đã chọn càng dễ được chọn (xác suất ∝ `D(x)²`) | `run_clustering` — cuML, `init='k-means++'`, `random_state=0` |
| **linspace tất định** | chia đều theo chỉ số trong unit, **không RNG** | `hard_boundary_kmeans` — để tái lập tuyệt đối |

**Cluster rỗng**: nếu không điểm nào được gán vào `c_j`, `mean` là `0/0`.
- `run_clustering` gốc: đặt centroid về **vector 0**
  ([clustering.py:86](squeezedattention/clustering.py#L86)).
- `hard_boundary_kmeans`: **giữ nguyên centroid cũ**, tránh NaN
  ([struct_clustering.py:414-416](struct_clustering.py#L414-L416)).

### 5.3 Cosine vs Euclid — điểm rất dễ nhầm trong repo này

`run_clustering` làm một việc **không đối xứng**, cần nắm rõ:

```python
data_normalized = F.normalize(head_data, p=2, dim=-1)   # chuẩn hoá L2
kmeans.fit(data_cp)                                     # ← K-means trên vector ĐÃ chuẩn hoá
...
cluster_keys = head_data[mask]                          # ← nhưng lấy key GỐC
centroid = torch.mean(cluster_keys, dim=0)              # ← centroid = mean của key GỐC
```

Tức là: **gán cụm theo hướng (cosine), nhưng centroid tính theo độ lớn thật (Euclid)**.

Vì sao hợp lý: trên hình cầu đơn vị, `‖x − c‖² = 2 − 2·(x·c)` → cực tiểu khoảng cách Euclid
**tương đương** cực đại cosine similarity. Nên bước ASSIGN chính là clustering theo cosine.
Còn centroid phải giữ **độ lớn thật** vì online nó được dùng trong `q · c / √D` — một tích vô
hướng chưa chuẩn hoá, độ lớn có ý nghĩa.

`hard_boundary_kmeans` thì dùng `torch.cdist` trên key **gốc** (Euclid thuần) — đây là một
**khác biệt có thật** giữa hai nhánh, cần ghi nhớ khi đọc kết quả ablation.

### 5.4 Độ phức tạp — và vì sao chi phí là bậc hai theo `S`

Lloyd, mỗi vòng: `O(n · K · D)`. Ở đây `n = S_ctx`, và chạy cho **mỗi head, mỗi layer**:

```
Chi phí  =  n_iter · L · H · S · K · D
```

Vì bài đặt `K = 5% · S`:

```
Chi phí  ∝  S²      ← BẬC HAI theo độ dài context
```

Đây là lý do các ước tính thời gian trong EXPERIMENT_LOG **sai nhiều lần**: đoán sai độ dài
2 lần → sai thời gian 4 lần. Với `S = 15.900`: `10 · 32 · 32 · 15900 · 790 · 128 ≈ 1,6 · 10^13`
phép — hàng chục giây GPU chỉ riêng K-means, chưa kể I/O.

Nút thắt thứ hai, ít ai để ý: `run_global_threshold` có vòng lặp Python qua `K`, chạy
`num_layers` lần → **25.000–36.000 vòng/sample**, mỗi vòng thao tác trên tensor `[32, S, 100]`
([clustering.py:159-168](squeezedattention/clustering.py#L159-L168)). Đây là phần chiếm phần lớn
thời gian, không phải K-means.

### 5.5 Weighted K-means

Mỗi điểm có trọng số `w_t ≥ 0`. Chỉ **bước UPDATE** đổi:

```
              Σ_{t ∈ cluster j}  w_t · x_t
c_j    =     ───────────────────────────────
              Σ_{t ∈ cluster j}  w_t
```

Bước ASSIGN **không đổi** (vẫn `argmin` khoảng cách thuần). Đây đúng là cách Hướng 2(b)
được cài trong `hard_boundary_kmeans(token_weights=...)`.

⚠️ Mẫu số là **tổng trọng số**, không phải số đếm. Không được `clamp(min=1)`: với trọng số
nhỏ hơn 1 (dấu câu 0.5×) clamp sẽ bóp méo trung bình. Chỉ được chặn chia cho 0
([struct_clustering.py:409-412](struct_clustering.py#L409-L412)).

### 5.6 Cài đặt GPU trong repo

| | `run_clustering` (SA gốc) | `hard_boundary_kmeans` (ta) |
|---|---|---|
| Thư viện | **cuML** (RAPIDS) qua cupy + dlpack | **PyTorch thuần** |
| Vòng lặp | Python `for H in range(num_heads)` | vector hoá theo bucket |
| Init | k-means++ (`random_state=0`) | linspace tất định |
| `max_iter` | 300 | 10 (`--n_iter`) |
| Metric ASSIGN | cosine (vector đã chuẩn hoá) | Euclid (`torch.cdist`) |

**dlpack** là chuẩn trao đổi tensor zero-copy giữa framework — dùng để đưa tensor PyTorch sang
cupy/cuML mà không copy qua CPU:
`torch tensor → to_dlpack → cupy → cuML KMeans → toDlpack → torch`.

⚠️ `squeezedattention/clustering.py` import `cupy` + `cuml` ở **top-level** → thiếu RAPIDS là
crash ngay lúc import, kể cả khi chỉ muốn chạy nhánh không dùng cuML.

---

## 6. Structure-aware clustering (đóng góp của ta)

Module: [struct_clustering.py](struct_clustering.py) · Script: [offline_clustering_struct.py](offline_clustering_struct.py)

### 6.1 Vì sao cấu trúc lại quan trọng với code

K-means chỉ biết khoảng cách embedding. Với code, ba quan sát:

1. Token trong cùng một **function** thường liên quan về ngữ nghĩa hơn hai token cùng embedding
   nhưng ở hai file khác nhau.
2. Khi user hỏi về một function, **toàn bộ** function đó nên được truy xuất — không phải một
   nửa nằm ở cluster này, nửa kia ở cluster khác.
3. Cluster vắt qua ranh giới function làm điểm `S_i` bị **nhiễu**: cluster vừa chứa code liên
   quan vừa chứa code không liên quan → hoặc kéo cả rác vào, hoặc bỏ sót cả phần liên quan.

Thêm một lợi ích cho claim C3: nếu cluster **không vắt qua** ranh giới function, thì sửa một
function chỉ cần re-cluster đúng các unit bị đụng → **incremental re-clustering** khả thi.
Với K-means thuần, sửa một dòng có thể đổi assignment ở mọi nơi.

### 6.2 AST và tree-sitter

**AST** (Abstract Syntax Tree) — cây cú pháp trừu tượng, biểu diễn cấu trúc chương trình:
`module → class → function → block → statement → expression → token`.

**tree-sitter** — thư viện parser incremental, **chịu lỗi** (error-tolerant): code sai cú pháp
vẫn parse được, chỗ hỏng thành node `ERROR`. Đây là tính chất **bắt buộc** ở đây, vì:

> Sample bị truncate mất phần giữa → code **không còn đúng cú pháp**. `truncate_fn` cắt
> `max_length/2` token đầu + `max_length/2` token cuối rồi nối lại
> ([utils.py:32-38](squeezedattention/utils.py#L32-L38)). Trường `truncated` trong meta của
> Phase 1.4 đánh dấu sẵn để quyết định bỏ qua hay chấp nhận.

Mỗi node có `start_byte`, `end_byte` — **byte offset** trong source. Đây là cầu nối giữa AST và
token: `assign_token_units` khớp `token_starts` (offset ký tự của từng token, do Phase 1.4 sinh)
với các span của node.

⚠️ Repo dùng **API mới** `tree_sitter >= 0.22` (`Language(mod.language())`), **không** dùng gói
`tree_sitter_languages` (gói đó không cài được). File cũ `ast_clustering.py` vẫn dùng gói cũ và
có fallback regex — không dùng nữa.

### 6.3 Năm level cấu trúc

```python
LEVELS = ("file", "class", "function", "block", "statement")   # thô → mịn
```

Hỗ trợ 5 ngôn ngữ: `python`, `java`, `javascript`, `typescript`, `csharp`
([struct_clustering.py:37-91](struct_clustering.py#L37-L91)).

Quy tắc quan trọng: **level thô được gộp vào level mịn**. `level="function"` cũng nhận `class`
(một class không có method vẫn là một đơn vị); `level="block"` nhận cả function lẫn class.
Thêm nữa, luôn có một span phủ **toàn bộ file**. Hai điều này bảo đảm:

> **Mọi token đều có ít nhất một unit bao nó — không token nào bị bỏ rơi**, và các level xếp
> thành một hierarchy thật (lồng nhau).

Phase 7.3 sẽ **quét tham số `--level`** để vẽ hình sensitivity (statement vs block vs function).

### 6.4 Gán `unit_id` — `assign_token_units`

Mỗi token được gán vào **span NHỎ NHẤT bao nó**.

Thuật toán: sắp span theo kích thước **giảm dần** rồi ghi đè. Span nhỏ xử lý sau nên thắng.
Mỗi span chỉ tốn 2 lần `searchsorted`:

```
Độ phức tạp:  O(U log S)      thay cho  O(S × U) của bản cũ
```

Con số cụ thể: `S = 31.000`, `U = 500` → bản cũ là **15 triệu vòng lặp Python**. Đó là lý do
`map_tokens_to_scopes` trong `ast_clustering.py` không dùng được.

`compact_unit_ids` ép `unit_id` về dải liên tục `[0, U)`, bỏ các span không chứa token nào.

### 6.5 Đề xuất 1 — Hard boundary K-means

**Khác biệt then chốt so với bản cũ**, cần nói rõ trong paper:

| | `ast_clustering.py` (cũ) | `struct_clustering.py` (mới) |
|---|---|---|
| AST dùng để | **khởi tạo** centroid | **ràng buộc** assignment |
| Sau đó | K-means chạy tự do trên toàn bộ token | K-means chạy **độc lập trong từng unit** |
| Kết quả | cluster **vẫn vắt qua** biên function | cluster **không bao giờ** vắt qua |
| Tên đúng | "AST-aware **init**" | "**hard boundary**" |

**Bất biến cần test:** duyệt mọi cluster của mọi head — không cluster nào chứa token của hai
unit khác nhau. Đây là test quan trọng nhất của Phase 2.

#### Phân bổ ngân sách — `allocate_centroids`

Ràng buộc (protocol 2.3):

```
(a)  k_u ≥ 1                        mỗi unit ít nhất 1 centroid
(b)  k_u ≤ min(n_u, max_k_per_unit) không nhiều cluster hơn điểm; trần mặc định 64
(c)  Σ_u k_u = K                    TỔNG đúng bằng budget → so công bằng với SA
```

Cách chia: cấp 1 cho mỗi unit trước, phần dư chia **tỉ lệ theo số token**, còn thừa thì rải
theo thứ tự unit lớn trước.

> **Quyết định thiết kế:** nếu `K < U` (nhiều unit hơn ngân sách) thì hàm **raise**, không im
> lặng cắt bớt. Cắt bớt sẽ phá tính "cùng budget" — nền tảng của mọi so sánh trong protocol.
> Cách xử lý đúng là dùng level **thô hơn** (function thay vì statement) hoặc tăng
> `percent_clusters`.

#### Vector hoá theo bucket lũy-thừa-2

Vấn đề: các unit có kích thước rất khác nhau → không batch trực tiếp được.

Giải pháp: gom unit theo kích thước **làm tròn lên lũy thừa 2** (`_bucket_by_size`), padding
trong mỗi bucket là bounded, rồi chạy K-means batch trên tensor `[H, B, P, D]`:

```
S = 16K, H = 32, K = 800  →  110 lần gọi kernel, 3,2 giây trên CPU
                             (vòng lặp Python kiểu cũ: 256.000 vòng)
```

Số lần gọi kernel là `O(log(max_unit_size))` mỗi vòng lặp, không phải `O(U)` hay `O(U × H)`.

Cập nhật centroid dùng `scatter_add_` thay vì vòng lặp — đây cũng chính là cách nên dùng để
tối ưu `run_global_threshold` nếu về sau cần.

### 6.6 Đề xuất 2 — StructHierarchy

Thay tầng L1 "K-means của K-means" của bài bằng **cấu trúc thật**:

```
Protocol:  Level-2 centroid = trong-function
           Level-1 centroid = trung bình theo function/file
```

Công thức (`struct_hierarchy_l1`, `weighted=True` mặc định):

```
                Σ_{j : parent(j) = g}  w_{h,j} · c2_{h,j}
c1_{h,g}   =   ─────────────────────────────────────────       w_{h,j} = |cluster j| ở head h
                Σ_{j : parent(j) = g}  w_{h,j}
```

**Tính chất quan trọng của `weighted=True`:** L1 centroid **đúng bằng trung bình toàn bộ key**
của nhóm. Chứng minh ngắn:

```
Σ_j w_j · c2_j  =  Σ_j Σ_{t ∈ cluster j} k_t  =  Σ_{t ∈ nhóm g} k_t
Σ_j w_j         =  | nhóm g |
```

`weighted=False` (trung bình cộng đơn thuần các centroid L2) làm cluster 3 token nặng ngang
cluster 200 token — giữ lại chỉ để ablation.

⚠️ **Bug đã từng mắc, test bắt được:** dùng số đếm cluster của **head 0** cho mọi head. Ranh
giới cứng chỉ bảo đảm mọi head có **cùng phân hoạch theo unit**; bên trong một unit thì mỗi
head chia cluster khác nhau → `w_{h,j}` khác nhau theo head. Test "weighted L1 == trung bình
toàn bộ key" lệch `6e-01` → lộ ra.

Một lợi ích phụ: `labels_l1` ánh xạ **thẳng** key → cluster L1, nên bỏ được bước
`gather(L1→L2→key)` mà bài gốc phải làm.

### 6.7 Vấn đề `K1` và `build_l1_groups`

**Vấn đề.** "Trung bình theo function/file" cho ra số nhóm = số unit cha mà code **tình cờ** có:

- 5 class → `K1 = 5` → bộ lọc L1 **vô dụng** (5 centroid không lọc nổi gì)
- 300 function → `K1` vượt xa 1% danh nghĩa → **budget lệch** (xem [§4.5](#45-budget--cách-tính-và-vì-sao-0325))

Mà "Ghi chú kiểm soát" của protocol cấm đúng chuyện này: mọi so sánh phải ở **cùng budget đo thực tế**.

**Cách xử lý**, giữ nguyên tính cấu trúc:

| Tình huống | Hành động |
|---|---|
| `K1_raw > target` | **GỘP** các unit cha **liền kề** (code liền nhau), cân theo số token |
| `K1_raw < target` | **TÁCH** mỗi unit cha thành nhiều nhóm con, mỗi nhóm là một **dãy unit L2 liền kề** |

Cả hai đều **không cắt đôi unit L2** → hierarchy vẫn lồng nhau đúng.

⚠️ Trần khi tách: một unit cha chỉ tách được tối đa bằng **số unit con** của nó. Không thể tách
2 function thành 3 nhóm. Đây là lý do `allocate_centroids` có tham số `caps` — bug thứ hai mà
test bắt được.

`K1` thực tế được ghi ra `k1_stats_<dataidx>.pt` để Phase 6 **báo cáo budget đo thật**, không
phải budget danh nghĩa.

### 6.8 Token-type weighting — Hướng 2(b), TẮT SẴN

Trọng số theo lớp node AST:

| Lớp | Trọng số | Lý do |
|---|---|---|
| `identifier` | 1.5 | mang nhiều thông tin ngữ nghĩa nhất |
| `literal` | 1.2 | |
| `keyword` / `other` | 1.0 | |
| `punctuation` | 0.5 | `(`, `)`, `,` — gần như không mang thông tin |
| `comment` | 0.5 | |

Cải tiến so với bản cũ: phân loại bằng **node type của tree-sitter** thay vì regex trên chuỗi
token đã decode. Bản cũ không phân biệt được identifier với keyword — `return` và `total` đều
khớp `^[a-zA-Z_][a-zA-Z0-9_]{1,}$`.

> **Mặc định TẮT.** Protocol Phase 2 chỉ có 3 nhánh ablation và mục 2.6 yêu cầu giữ nguyên
> Si/threshold/kernel. Bật lên là thêm một biến vào thí nghiệm → **phải báo cáo thành nhánh
> riêng**, không được trộn vào con số của `+HardBoundary`. Test xác nhận: tắt cờ ra kết quả
> **trùng bit-for-bit** với đường không có cờ.

### 6.9 Ba nhánh ablation

```bash
python offline_clustering_struct.py <model> --dataset lcc --method {sa|hard_boundary|struct_hierarchy}
```

| `--method` | Nội dung | Ghi chú |
|---|---|---|
| `sa` | K-means thuần toàn bộ key | gọi **thẳng** `run_clustering` gốc — để mọi nhánh đi qua cùng một đường code, loại trừ khác biệt do môi trường |
| `hard_boundary` | K-means độc lập trong từng unit AST | **đề xuất 1** |
| `struct_hierarchy` | hard_boundary ở L2 + L1 theo unit cha | **đề xuất 2**, ghi thêm file L1 |

Nguyên tắc xuyên suốt (protocol 2.6): module **chỉ sinh centroid + label**, cùng layout
`[1,H,K,D]` / `[1,H,S]`. Threshold vẫn do `run_global_threshold` tính y hệt mọi nhánh. Kernel
không đụng tới. Nhờ vậy mọi khác biệt trong kết quả **chỉ có thể đến từ clustering**.

`offline_clustering.py` và `modeling_llama.py` **không bị sửa** → gate Phase 0 nguyên vẹn.

---

## 7. Kiến thức liên quan khác

### 7.1 Phổ các phương pháp nén KV cache

Cần phân biệt rõ để positioning paper đúng:

| Nhóm | Cơ chế | Ví dụ | Key bị bỏ có lấy lại được? |
|---|---|---|---|
| **Eviction** | **xoá vĩnh viễn** key ít quan trọng khỏi cache | H2O, SnapKV, StreamingLLM | ❌ Không |
| **Selection / sparse retrieval** | **giữ đủ** cache, mỗi query chỉ **nạp** một tập con | **Squeezed Attention**, QUEST, ClusterKV | ✅ Có |
| **Quantization** | giảm số bit của K/V | KIVI, KVQuant | không mất token |
| **Kiến trúc** | giảm `H_kv` ngay từ khi train | GQA, MQA, MLA | — |

Squeezed Attention thuộc nhóm **selection** — quan trọng vì nó không mất thông tin vĩnh viễn,
chỉ đánh đổi băng thông.

**Baseline phải tích hợp cho Phase 6** (chưa có trong repo):

- **QUEST** — chia key thành **page** liên tục theo vị trí, tóm tắt mỗi page bằng min/max
  theo từng chiều, ước lượng cận trên của điểm attention để chọn page. Khác SA ở chỗ: nhóm
  theo **vị trí**, không phải theo **nội dung**.
- **ClusterKV** — cũng cluster theo ngữ nghĩa, gần SA nhất.

Điểm so sánh đáng nói của ta: QUEST nhóm theo vị trí liền kề — mà *ranh giới AST cũng là ranh
giới vị trí liền kề*, chỉ khác là **có ý nghĩa cú pháp**. Đây là một trục thảo luận tốt cho paper.

### 7.2 Novelty — các công trình dễ bị nhầm

Từ [README_EXTENSIONS.md](README_EXTENSIONS.md):

1. **Wang & Gan (ICLR 2025) "SqueezeAttention"** — **trùng tên** nhưng là paper khác. Họ làm
   layer-wise budget cho phương pháp **eviction**, tín hiệu là cosine similarity của hidden
   state. Phải nhấn mạnh khác biệt.
2. **Multipole Attention (NeurIPS 2025)** — follow-up của **chính Hooper** cho reasoning task,
   có hierarchical clustering. Đảm bảo positioning không bị nhầm.
3. **AST-aware cho KV cache** — **chưa có ai làm trực tiếp**. Đây là contribution sạch nhất.

### 7.3 Hai hướng khác đã có trong repo (không nằm trong protocol)

| Hướng | File | Trạng thái |
|---|---|---|
| **Layer-wise Adaptive Budget** — phân bổ centroid theo attention entropy mỗi layer | [adaptive_budget.py](adaptive_budget.py) | prototype, không dùng trong Phase 2 |
| **Value-Aware Clustering** — cluster trên `concat(α·K, β·V)` thay vì chỉ K | [value_aware_clustering.py](value_aware_clustering.py) | prototype |

**Attention entropy** (dùng ở Hướng 1):

```
Entropy(head) = − Σ_t  a_t · log a_t          đơn vị: nats,  từ 0 tới log(N)
```

Entropy thấp → attention tập trung → dễ đoán top-k → cần ít cluster.
Entropy cao → attention phân tán → cần nhiều cluster để xấp xỉ.

Lưu ý: threshold toàn cục của SA **đã tạo ra một dạng adaptive tự nhiên** theo layer (xem
[§4.3](#43-công-thức-điểm-s_i) bước 6) — nên gain của Hướng 1 chưa chắc còn nhiều. `analyze_entropy.py`
có gate: diagnostic ratio `< 0.1` → dừng, `> 0.3` → green light.

⚠️ Đường `--code_aware` của `offline_clustering_v2.py` **chưa từng chạy thành công**: tokenizer
load `use_fast=False` nhưng `map_tokens_to_scopes` gọi `return_offsets_mapping=True` →
`NotImplementedError`. Đây chính là lý do Phase 1.4 phải tách ra làm script riêng dùng
tokenizer **nhanh**.

### 7.4 Benchmark

#### LongBench LCC + RepoBench-P (đang dùng)

| | LCC | RepoBench-P |
|---|---|---|
| Task | hoàn thành dòng code kế tiếp | như trên, có cross-file context |
| Số sample | 500 | 500 |
| Độ dài (đo thật) | trung vị 3.080, trung bình 4.290, max 31.494 token | ~15.900 token trung bình |
| Metric | `code_sim_score` | `code_sim_score` |

**`code_sim_score`** ([LongBench/metrics.py:80](LongBench/metrics.py#L80)): lấy **dòng đầu tiên
không chứa** `` ` ``, `#`, `//` của prediction, so với ground truth bằng `fuzz.ratio`
(Levenshtein-based edit similarity), chia 100. Nói cách khác: **edit similarity của một dòng**.

**Bài học quy đổi độ dài:** LongBench công bố LCC "1235 words". Quy đổi 2 token/word là **sai** —
với code thực tế là **3,5 token/word** (4290/1235). Áp đúng tỉ lệ cho RepoBench-P:
`4206 × 3,5 = 14.700` ≈ 15.900 đo được. Ghi lại để không lặp lại sai lầm này.

#### CrossCodeEval / RepoBench v1.1 — đã khảo sát, KHÔNG dùng nguyên bản

Kết luận từ [docs/PHASE1_DATASETS.md](docs/PHASE1_DATASETS.md), đo trên dữ liệu tải thật:

| Vấn đề | Chi tiết |
|---|---|
| **Context quá ngắn** | trung vị 1.4–3.6K token, xa mức 20–150K protocol giả định. `crossfile_context` chỉ là top-5 chunk đã retrieve |
| **Context KHÔNG dùng chung** | trong 1500 sample Python CrossCodeEval: 211 repo có >1 sample, **chỉ 2 repo** dùng chung context. Repo `hq0709-Depth-NeuS`: 42 sample → 42 context khác nhau |

Vấn đề thứ hai **phá thẳng premise của SA**. Nếu mỗi query có context riêng thì phải chạy lại
clustering cho từng query → toàn bộ giá trị của phương pháp biến mất.

Chính bài gốc thừa nhận khoảng trống này (Section 5): *"there is currently no benchmark designed
to test this scenario... Recent long context benchmarks do not evaluate the handling of multiple
queries on the same document"* — và đó là lý do họ tự dựng **PreFixQA**.

**Đường đi khả thi** (quyết định D4, chưa chốt): clone repo gốc về, lấy **toàn bộ repo làm fixed
context**, tái dùng chính nhãn có sẵn (`groundtruth` / `next_line`) làm query. Đạt cả hai yêu cầu
cùng lúc: context 20–150K **và** nhiều query trên cùng context. Đã kiểm: 15/15 repo RepoBench
nhiều sample nhất còn clone được.

#### Contamination

`created_at` của **toàn bộ** 4017 dòng RepoBench v1.1 đều là năm **2023**. Qwen2.5-Coder train
tới ~2024 → gần như chắc chắn đã thấy các repo này.

Xử lý: so sánh **tương đối** giữa các method ở cùng budget vẫn hợp lệ (mọi method chạy trên cùng
model bị nhiễm), nhưng **số tuyệt đối phải ghi rõ caveat** trong paper.

### 7.5 Metric cho các claim

**C2 — Retrieval quality (Phase 5, chạy TRƯỚC Phase 6):**

Dựa trên baseline "Ideal" của bài (Appendix H).

```
K*     =  tập key "lý tưởng": tính full attention query → toàn bộ fixed key,
          lấy top-p theo threshold
K_m    =  tập key mà method m chọn, ở CÙNG budget

                  | K_m ∩ K* |
Recall@budget  = ───────────────
                     | K* |
```

Metric phụ: precision, và **attention mass recovered** = tổng attention weight của `K_m` trên `K*`.
(Recall đếm số key; attention mass đo *lượng xác suất* thu hồi được — quan trọng hơn khi phân
phối lệch mạnh.)

Pass nếu structure-aware recall cao hơn SA **có ý nghĩa thống kê ở ≥ 2 mức budget** (paired test).

> **Nếu C2 fail thì H0 sai → DỪNG, không chạy C1/C3.** Đây là gate rẻ nhất của toàn bộ nghiên cứu.

**C1 — Accuracy@matched-budget (Phase 6):** completion → Exact Match + Edit Similarity;
pass@1 nếu có test; QA → F1. **Budget phải đo thực tế**, không phải danh nghĩa.

**C3 — Incremental (Phase 4 + 7):** `t_full` vs `t_incr` theo kích thước diff (1 function /
1 file / 10% repo) + `Δaccuracy`. Claim: speedup nhiều lần, `Δaccuracy < 0.3` điểm.

### 7.6 Kỷ luật thí nghiệm

Áp dụng xuyên suốt, đây là những điểm reviewer TDSC/KSE hay soi:

- [ ] Mọi so sánh ở **cùng budget đo thực tế**, kể cả overhead metadata centroid.
- [ ] GQA: nêu rõ chiến lược chọn key per-head, khớp cấu hình QUEST khi so.
- [ ] Repo dữ liệu **sau mốc train** của model để tránh contamination — ghi rõ nguồn/năm.
- [ ] Cố định seed, báo cáo **mean ± std qua ≥ 3 seed** cho các con số accuracy chính.
- [ ] Nêu rõ clustering là **chi phí offline một lần**, KHÔNG nằm trong inference latency.
- [ ] Latency Phase 7: **1 GPU cố định** (đã chốt A100-SXM — đổi sang PCIe là phải đo lại hết).

### 7.7 Cạm bẫy môi trường đã trả giá

Tóm tắt từ EXPERIMENT_LOG (~5 giờ pod ≈ $8 để tìm ra tổ hợp chạy được):

| Bẫy | Hệ quả | Cách tránh |
|---|---|---|
| Python 3.12 của image RunPod | torch 2.3.1 không kéo triton → kernel `tl.math.exp2` chết; RAPIDS cp312 đòi `numpy≥2` còn torch 2.3 cần `numpy<2`; flash-attn không có wheel cp312 | dựng venv **Python 3.10** |
| `uv venv` không cài pip | `pip` rơi vào Python hệ thống | dùng `python -m pip` |
| `pip install flash-attn` | build từ nguồn **2,5 giờ** | cài thẳng URL wheel từ GitHub Releases |
| `datasets` đời mới | kéo `pyarrow≥21` → phá cudf 24.6 → phá cuML | ghim `datasets==2.20.0` + `pyarrow 16.1` |
| `pip install -r LongBench/requirements.txt` | ghim `transformers==4.31.0`, **ghi đè bản fork** → mọi cờ `use_centroids` biến mất, số trông "gần đúng" một cách gây nhầm lẫn | **tuyệt đối không chạy** |
| `df -h /workspace` | hiện dung lượng **cả cụm MooseFS** (404 TB), không phải hạn mức volume → `Disk quota exceeded` trong khi `df` báo còn hàng trăm TB | xem dashboard RunPod, hoặc thử `dd` rồi xoá |
| LongBench custom loader hỏi `[y/N]` | job dài không người trông sẽ **treo** | `HF_DATASETS_TRUST_REMOTE_CODE=1` |

**Stack đã kiểm chứng:** Python 3.10 · torch/triton 2.3.1+cu121 · flash-attn 2.6.3 (wheel cp310)
· cuML 24.6.1 / cupy 13.6.0 · numpy 1.26.4 / pyarrow 16.1.0 / datasets 2.20.0 · transformers
4.40.0.dev0 (fork, editable).

---

## 8. Từ điển thuật ngữ

| Thuật ngữ | Giải thích |
|---|---|
| **Fixed context / shared prefix** | Phần prompt **không đổi** giữa các query. Đối tượng của toàn bộ tối ưu hoá offline. |
| **User input / query** | Phần prompt **thay đổi** mỗi lần hỏi. |
| **Observation window** | 100 token cuối của fixed context, **không cluster**. Dùng làm query giả để calibrate threshold, và luôn được giữ khi inference. |
| **Centroid** | Vector đại diện một cluster key. Bằng trung bình các key trong cluster. |
| **Label (`centroid_labels`)** | Ánh xạ token → cluster. Shape `[1, H, S]`. |
| **Global threshold `τ`** | **Một** con số cho cả model, so với điểm `S_t` để quyết định giữ hay bỏ. Không per-layer, không per-head. |
| **Sparsity / percentile `p`** | Tỉ lệ key bị **loại**. Sq-70% = loại 70%, giữ 30%. |
| **Budget** | Phần trăm KV cache thực sự phải nạp, **đã tính metadata centroid**. Trục so sánh chính. |
| **Matched budget** | Đặt budget của mọi method về cùng một mức trước khi so accuracy. Bắt buộc. |
| **Hierarchical lookup** | Hai tầng centroid: L1 thô lọc bớt, L2 mịn chọn tập cuối. |
| **Hard boundary** | Ranh giới cứng theo AST: cluster **không được** vắt qua hai unit. Đề xuất 1 của ta. |
| **AST-aware init** | Chỉ dùng AST để **khởi tạo** centroid, K-means vẫn chạy tự do. Bản cũ, **không phải** hard boundary. |
| **Unit** | Một đơn vị cấu trúc (file/class/function/block/statement) — miền mà K-means được phép chạy trong đó. |
| **Ablation** | Bật/tắt từng thành phần để đo đóng góp riêng của nó. Ở đây: SA → +HardBoundary → +StructHierarchy → +SymbolSignal. |
| **Gate** | Thí nghiệm chốt chặn: không đạt thì không đi tiếp. Phase 0 gate = tái lập Table 2 ±0.3. |
| **Prefill / Decode** | Prefill = xử lý cả prompt một lần (compute-bound). Decode = sinh từng token (memory-bound). |
| **KV cache** | Bộ nhớ lưu `K, V` của token đã xử lý để không tính lại. |
| **GQA** | Grouped-Query Attention: nhiều query head dùng chung một KV head. |
| **RoPE** | Rotary Position Embedding: mã hoá vị trí bằng phép xoay, cho khoảng cách tương đối. |
| **FlashAttention** | Attention không materialize ma trận `S×S`, dùng online softmax. |
| **Triton** | DSL viết GPU kernel bằng Python. |
| **dlpack** | Chuẩn trao đổi tensor zero-copy giữa framework (torch ↔ cupy). |
| **cuML / RAPIDS** | Thư viện ML chạy trên GPU của NVIDIA. `run_clustering` dùng `cuml.cluster.KMeans`. |
| **tree-sitter** | Parser incremental, **chịu lỗi** — code hỏng cú pháp vẫn parse được, sinh node `ERROR`. |
| **Byte offset** | Vị trí byte trong source. Cầu nối giữa node AST và token. |
| **Symbol / def-use** | Idea 2 (Phase 3): dùng identifier và quan hệ định nghĩa–sử dụng làm tín hiệu retrieval bổ sung. |
| **Incremental re-clustering** | Idea 3 (Phase 4): code bị sửa thì chỉ re-cluster các unit bị đụng. |
| **Contamination** | Model đã thấy dữ liệu test lúc train → số tuyệt đối không đáng tin. |
| **Recall@budget** | Metric chính của C2: tỉ lệ key lý tưởng mà method truy xuất được, ở cùng budget. |
| **Attention mass recovered** | Tổng attention weight (chứ không phải số key) thu hồi được. |
| **Edit similarity** | `fuzz.ratio` — độ tương đồng Levenshtein, thang 0-1. Metric của LCC/RepoBench-P. |

---

## 9. Cheat sheet công thức

**Attention**
```
a_it = exp(q_i·k_t/√D) / Σ_s exp(q_i·k_s/√D)          out_i = Σ_t a_it · v_t
```

**KV cache**
```
bytes = 2 · L · H_kv · D · S · sizeof(dtype)
```

**Peak VRAM offline clustering (repo này)**
```
≈ 13,5 GB (weights bf16) + S × 0,79 MB     ← 0,79 MB = 3(QKV) · 32 layer · 4096 · 2 B
```

**Đĩa cho centroid + label**
```
≈ 34 KB/token       (fp32 centroid + int64 label, K = 5%·S_ctx)
```

**K-means (Lloyd)**
```
ASSIGN:  ℓ(t) ← argmin_j ‖x_t − c_j‖²
UPDATE:  c_j  ← Σ_{ℓ(t)=j} w_t·x_t  /  Σ_{ℓ(t)=j} w_t     (w_t ≡ 1 nếu không trọng số)

Chi phí = n_iter · L · H · S · K · D  ∝ S²   khi K = 5%·S
```

**Cosine ↔ Euclid trên hình cầu đơn vị**
```
‖x − c‖² = 2 − 2·(x·c)      → argmin Euclid ≡ argmax cosine
```

**Điểm Squeezed Attention**
```
A_ij  = q_i · c_j / √D
Z_i   = Σ_j n_j · exp(A_ij)                        ← ước lượng mẫu số softmax
S_t   = (1/W) · Σ_i  exp(A_{i,ℓ(t)}) / Z_i
τ_p   = Quantile_p( S_t qua MỌI layer, head, token ),   p ∈ {0.5, 0.7, 0.8, 0.9}
mask  = ( S_t > τ_p )
```

**Budget**
```
budget = (1 − p) + metadata          metadata = 2,5% (đơn tầng)  |  3,0% (phân tầng)
Sq-70%: 0,30 + 0,025 = 0,325 ✓
```

**Phân bổ centroid theo unit**
```
Σ_u k_u = K            k_u ≥ 1            k_u ≤ min(n_u, max_k_per_unit)
```

**L1 centroid theo cấu trúc (weighted)**
```
c1_g = Σ_{j: parent(j)=g} w_j·c2_j / Σ_{j: parent(j)=g} w_j      w_j = |cluster j| (TỪNG HEAD)
     = trung bình toàn bộ key của nhóm g
```

**Recall@budget**
```
Recall = |K_m ∩ K*| / |K*|
```

**Quy đổi độ dài code**
```
1 word ≈ 3,5 token       (KHÔNG phải 2 — sai lầm đã mắc)
```

---

## 10. Bản đồ file → khái niệm

### Lõi Squeezed Attention (gốc, không sửa)

| File | Khái niệm |
|---|---|
| [squeezedattention/clustering.py](squeezedattention/clustering.py) | `run_clustering` (K-means cuML) · `run_global_threshold` (công thức `S_i` + calibrate `τ`) · nhánh GQA |
| [squeezedattention/kernels.py](squeezedattention/kernels.py) | 3 Triton kernel: centroid lookup, sparse attention, reduce |
| [squeezedattention/utils.py](squeezedattention/utils.py) | `truncate_fn` (cắt giữa, tính `shared_prefix_length`) · `build_chat` |
| [offline_clustering.py](offline_clustering.py) | pipeline offline: hook Q/K → cluster → threshold → lưu `.pt` |
| [transformers/.../llama/modeling_llama.py](transformers/src/transformers/models/llama/modeling_llama.py) | đường online: nạp centroid, lookup, gộp 3 nhánh attention |
| [transformers/.../qwen2/modeling_qwen2.py](transformers/src/transformers/models/qwen2/modeling_qwen2.py) | port sang GQA (Phase 1.5/1.6) |

### Đóng góp của ta (Phase 2)

| File | Khái niệm |
|---|---|
| [struct_clustering.py](struct_clustering.py) | `parse_units` · `assign_token_units` · `allocate_centroids` · `hard_boundary_kmeans` · `build_l1_groups` · `struct_hierarchy_l1` · `compute_token_type_weights` |
| [offline_clustering_struct.py](offline_clustering_struct.py) | script offline 3 nhánh ablation |
| [scripts/prepare_code_data.py](scripts/prepare_code_data.py) | Phase 1.4 — offset ký tự từng token, có self-test |
| [scripts/test_struct_clustering.py](scripts/test_struct_clustering.py) | 72 test CPU, gồm test bất biến ranh giới cứng |
| [scripts/test_gqa_port.py](scripts/test_gqa_port.py) | 20 test shape cho port GQA |

### Hạ tầng thí nghiệm

| File | Việc |
|---|---|
| [configs/phase0.sh](configs/phase0.sh) | **nguồn duy nhất** của mọi tham số. Mọi phase phải `source`, không hard-code |
| [scripts/phase0_gate.sh](scripts/phase0_gate.sh) | chạy trọn gate |
| [scripts/check_gate.py](scripts/check_gate.py) | so kết quả vs Table 2 (±0.3), tự ghi vào EXPERIMENT_LOG |
| [scripts/reference_table2.json](scripts/reference_table2.json) | số Table 2 |
| [scripts/record_env.py](scripts/record_env.py) | dump version/GPU/seed, kiểm transformers có đúng fork |
| [scripts/setup_pod.sh](scripts/setup_pod.sh) | dựng môi trường theo stack đã kiểm chứng |

### Prototype không nằm trong protocol

| File | Hướng |
|---|---|
| [adaptive_budget.py](adaptive_budget.py) | Hướng 1 — layer-wise budget theo entropy |
| [ast_clustering.py](ast_clustering.py) | Hướng 2 **bản cũ** (AST-init, không phải hard boundary). Chỉ `parse_code_to_scopes` còn dùng lại được |
| [value_aware_clustering.py](value_aware_clustering.py) | cluster trên `concat(α·K, β·V)` |
| [offline_clustering_v2.py](offline_clustering_v2.py) | H1 + H2 bản cũ. Đường `--code_aware` **chưa từng chạy thành công** |

### Tài liệu

| File | Nội dung |
|---|---|
| [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | **nguồn sự thật duy nhất** cho tiến độ, kết quả, quyết định D1-D4 |
| [docs/PHASE0.md](docs/PHASE0.md) | hướng dẫn dựng môi trường + chạy gate |
| [docs/PHASE1_DATASETS.md](docs/PHASE1_DATASETS.md) | khảo sát CrossCodeEval / RepoBench trên dữ liệu thật |
| [docs/PATCHING_EVAL.md](docs/PATCHING_EVAL.md) | patch cần cho eval khi K không đồng nhất (**chưa apply**) |
| [README.md](README.md) | README gốc của bài |
| [README_EXTENSIONS.md](README_EXTENSIONS.md) | hai hướng cũ + ghi chú novelty |
| **summarize_knowledge.md** | ← file này |

---

## Phụ lục — Những chỗ dễ sai nhất

Gom lại để tra nhanh. Điểm chung của gần hết danh sách này: **không crash, chỉ sai lệch**.

1. **Shape `[H,K,D]` vs `[1,H,K,D]`** — online đọc `shape[2]` làm `K`, thiếu batch dim sẽ lấy nhầm `D`.
2. **`repeat` vs `repeat_interleave`** (GQA) — cả hai chạy trơn tru, một cái tra nhầm nhóm centroid.
3. **Số đếm cluster của head 0 dùng cho mọi head** — ranh giới cứng chỉ đảm bảo cùng phân hoạch theo **unit**, không phải cùng cluster.
4. **Trần số unit con khi tách L1** — 2 function không thể tách thành 3 nhóm.
5. **`clamp(min=1)` cho mẫu số weighted K-means** — bóp méo trung bình khi trọng số < 1.
6. **`pip install -r LongBench/requirements.txt`** — ghi đè transformers fork, số trông "gần đúng".
7. **Số trong `LongBench/README.md`** — không phải mốc để so, dùng vào sẽ FAIL oan.
8. **`pred.py` ghi jsonl append** — chạy lại là nhân đôi prediction, `eval.py` ra số sai mà không báo lỗi. Đã thêm `--overwrite`.
9. **`seed_everything` không kế thừa qua `mp.spawn`** — phải seed lại trong process con.
10. **Offset phải tính trên prompt CUỐI CÙNG sau truncation**, không phải source gốc — lệch offset nghĩa là gán `unit_id` cho sai key vector.
11. **`df -h` trên MooseFS** — báo dung lượng cả cụm, không phải hạn mức volume.
12. **`build_chat(prompt, model_name)`** ([squeezedattention/utils.py:44](squeezedattention/utils.py#L44)) — `model_name` chưa định nghĩa trong scope → `NameError`. Chỉ nổ với dataset ngoài `{trec, triviaqa, samsum, lcc, repobench-p}`.
