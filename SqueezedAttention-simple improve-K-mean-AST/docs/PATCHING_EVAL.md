# Patching `run_evaluation.sh` cho Adaptive Budget

Online evaluation pipeline của Squeezed Attention gốc giả định **mọi layer có cùng K cluster**. Với Hướng 1 (adaptive budget), K khác nhau giữa các layer → cần patch nhẹ.

## File cần sửa

### 1. `LongBench/pred.py` (hoặc tên file tương đương)

Tìm chỗ load centroid, ví dụ:

```python
# Code gốc (giả định uniform K)
centroids = torch.load(f"{centroids_path}/centroids_tensor_dict_{dataidx}_{num_centroids}.pt")
```

Sửa thành:

```python
# Patched: load với adaptive K
# Đầu tiên load budgets để biết K thực sự của từng layer
budgets_path = f"{centroids_path}/budgets_{dataidx}.pt"
if os.path.exists(budgets_path):
    budgets = torch.load(budgets_path)
    if isinstance(budgets, dict):  # hierarchical case
        budgets_l2 = budgets["l2"]
        avg_k = int(budgets_l2.float().mean().item())
    else:
        avg_k = int(budgets.float().mean().item())
    # File save name dùng avg_k (xem offline_clustering_v2.py)
    centroids = torch.load(f"{centroids_path}/centroids_tensor_dict_{dataidx}_{avg_k}.pt")
    labels = torch.load(f"{centroids_path}/centroids_labels_dict_{dataidx}_{avg_k}.pt")
    threshold = torch.load(f"{centroids_path}/global_threshold_{dataidx}_{avg_k}.pt")
else:
    # Fallback: behavior gốc với K uniform
    centroids = torch.load(f"{centroids_path}/centroids_tensor_dict_{dataidx}_{num_centroids}.pt")
    ...
```

### 2. Attention forward (`squeezedattention/` package)

Trong file thực hiện attention với squeezed clusters (thường có hook hoặc custom forward), sửa để chấp nhận K khác nhau mỗi layer:

```python
# Code gốc giả định cùng số centroid cho mọi layer
def squeezed_attn_forward(self, query, key, value, centroids, labels, threshold):
    K = centroids.shape[1]  # uniform K
    ...

# Patched: K được suy ra từ shape của centroids của LAYER ĐÓ
def squeezed_attn_forward(self, query, key, value, centroids, labels, threshold):
    K_this_layer = centroids.shape[1]  # K riêng cho layer này
    ...
```

Vì shape của centroid tensor `[H, K, D]` đã encode K, hầu hết các operation không cần K explicit — chỉ cần code không hardcode K. Check kỹ với `grep`:

```bash
grep -rn "num_centroids" squeezedattention/
grep -rn "K = " squeezedattention/
```

### 3. Run script

Trong `LongBench/run_evaluation.sh`, đặt path output đúng:

```bash
# Cho baseline
CENTROIDS_DIR="../output_full/baseline"

# Cho adaptive linear
CENTROIDS_DIR="../output_full/adaptive-linear"

# Cho code-aware
CENTROIDS_DIR="../output_full/codeaware-python"

# Cho combo
CENTROIDS_DIR="../output_full/adaptive-linear_codeaware-python"
```

## Kiểm tra patch đúng chưa

Chạy 1 sample với verbose mode, in ra K mỗi layer:

```python
# Trong pred.py
for layer_idx, c in centroids.items():
    print(f"Layer {layer_idx}: K = {c.shape[1]}")
```

Output kỳ vọng:
- Baseline: tất cả layer cùng K
- Adaptive: K khác nhau

## Nếu Squeezed Attention gốc dùng hardcoded K trong Triton kernel

Vấn đề nghiêm trọng hơn: nếu kernel Triton compile với K cố định, K thay đổi mỗi layer sẽ recompile mỗi lần → chậm. Hai cách giải quyết:

**Approach 1 (Easy):** Round K lên power-of-2 hoặc multiple-of-32 để giảm số biến thể kernel cần compile. Trade-off: budget không exact bằng số nguyên bạn muốn.

**Approach 2 (Hard):** Dùng `tl.constexpr` cho K khi compile, cache theo K. Triton hỗ trợ điều này nhưng cần code review kỹ.

Cho v0 paper, **Approach 1** đủ — focus vào accuracy first, optimization speed sau.

## Common bugs

### Bug 1: Centroid tensor shape mismatch
Triệu chứng: `RuntimeError: size mismatch` khi attention
Nguyên nhân: code giả định tất cả layer cùng K nhưng concat tensor
Fix: process layer-by-layer thay vì batch all layers

### Bug 2: Budgets file không được load
Triệu chứng: K luôn = num_centroids (uniform)
Nguyên nhân: code load wrong path
Fix: check `os.path.exists(budgets_path)` debug print

### Bug 3: Result giống hệt baseline dù adaptive
Triệu chứng: accuracy không khác baseline
Có thể: (a) clustering vẫn dùng uniform K, (b) eval load wrong centroids, (c) entropy variance quá nhỏ
Debug: in ra số cluster mỗi layer **trong eval**, không chỉ trong clustering.
