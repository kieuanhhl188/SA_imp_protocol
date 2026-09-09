# Phase 2 — L2 = BLOCK, LongChat-7B, LCC, 200 mẫu (9/9/2026)

Mục đích: ở `--level function` thì `build_l1_groups` chẻ L1 tới đúng số function →
`struct_hierarchy` ≡ `hard_boundary`. Chạy lại ở `--level block` để L1 = function trở
thành một phép GỘP thật (nhiều block → 1 function), xem đề xuất 2 có tách được khỏi đề
xuất 1 không.

Lệnh: `scripts/run_phase2_block_lcc.sh` (nhánh `sa` tái dùng `/workspace/p2-longchat/sa/lcc`,
độc lập với level). Centroid ~78 GB ở `/workspace/p2-longchat-block/`, KHÔNG vào git.

## Clustering

| | hard_boundary (block) | struct_hierarchy (block, L1=function) |
|---|---|---|
| Khả thi | 198/200 | 198/200 |
| Vượt ngân sách (skip) | 2/200 = 1,0% — `dataidx` 87, 99 | idem |
| unit L2 / mẫu | tb 68,8 · max 219 | tb 67,6 · max 203 |
| K1 thực tế (tầng L1) | — | **median 26 · tb 34,8 · max 179** · mode split 168 / merge 28 / as-is 2 |

So với function-level: K1 ở function là median 16,5 ≡ số function (mode split 199/1).
Ở block, K1 (median 26) nằm GIỮA số function (~16) và số block (~68) → tầng L1 giờ là
tầng trung gian thật, và nhánh *merge* của `build_l1_groups` lần đầu chạy đáng kể (28 mẫu).

## Bất biến — `p2_invariants_block.log` · TẤT CẢ QUA ✅

| Kiểm | Kết quả |
|---|---|
| [B] cùng budget K ba nhánh | khớp 3 nhánh |
| [A] `hard_boundary` block vắt biên | **0,0% — 0/198 mẫu** |
| [A] `struct_hierarchy` block vắt biên | **0,0% — 0/198 mẫu** |
| [A] `sa` (đối chứng) | vắt biên **trung vị 57,5%** (p25 44,8 · p75 65,2 · min 1,3 · max 90,0) — 200/200 mẫu >0 |
| [C] shape `[1,32,K,128]` · ô rỗng | đúng · 0,0% |

`sa` vắt biên cao hơn hẳn function-level (32,3% → 57,5%): ranh giới block dày hơn nên
K-means thuần vi phạm nhiều hơn → can thiệp block **có liều lớn hơn**.

## Phase 5 — C2 recall@budget · `phase5_lcc_block.json` · n=98

| method | recall@70 | recall@80 | recall@90 | mass@70 |
|---|---:|---:|---:|---:|
| `sa` | 74,39 | 69,49 | 62,45 | 96,8 |
| `hard_boundary` (block) | 71,15 | 65,34 | 57,01 | 92,4 |
| `struct_hierarchy` (block) | 71,15 | 65,34 | 57,01 | 92,4 |

### Hai kết luận

1. **`struct_hierarchy` VẪN ≡ `hard_boundary` từng chữ số** — KHÔNG phải vì L1 ≡ L2 (ở
   block, L1=26 thực sự thô hơn L2=68), mà vì **`phase5_recall.py` không hề đọc file
   hierarchical L1** (`hierarchical_centroids_*_L1_*.pt`). Nó chỉ đo recall phẳng trên L2,
   mà L2 của `struct_hierarchy` do CHÍNH `hard_boundary_kmeans` sinh ra → trùng bit.
   → Muốn tách đề xuất 2 khỏi đề xuất 1 phải có: (a) biến thể Phase 5 truy hồi phân tầng
   (định tuyến qua centroid L1 → tinh trong nhóm L1 trúng), hoặc (b) Phase 6 accuracy@budget
   với kernel phân tầng online. Đổi level KHÔNG sửa được điều này.

2. **Ranh giới block mịn hơn → recall vs ideal THẤP HƠN.** hard_boundary@70 tụt từ 73,35
   (function) xuống 71,15 (block). Càng ép ranh giới cấu trúc chặt, recall so với oracle
   càng tụt — **cùng chiều với kết quả function-level, ngược chiều giả thuyết Idea 1**, và
   biên độ lớn hơn.

### Kiểm định ghép cặp (bootstrap 20.000, gộp lớp theo mẫu)

| config | sp70 | sp80 | sp90 |
|---|---|---|---|
| LongChat function (n=100) | −1,00 [−1,29; −0,74] | −1,49 [−1,83; −1,18] | −2,33 [−2,75; −1,95] |
| LongChat **block** (n=98) | −3,24 [−3,61; −2,88] | −4,15 [−4,58; −3,71] | −5,44 [−5,95; −4,92] |
| Qwen function (n=300, PHASE5_RESULTS.md) | −0,89 [−1,01; −0,77] | −1,13 [−1,26; −1,00] | −1,37 [−1,51; −1,23] |

**Cả 9 khoảng tin cậy loại trừ 0, tất cả âm.** Tiêu chí C2 (5.5) — *"pass nếu structure-aware
recall CAO HƠN SA có ý nghĩa ở ≥2 mức budget"* — **fail ở cả 3 cấu hình**, lệch về phía
ngược lại. Block làm khoảng cách rộng gấp ~3 lần function.

## Việc còn lại

- Cần cơ chế đo dùng tới tầng L1 thì `struct_hierarchy` mới khác `hard_boundary` được.
- Level sweep: statement (bỏ 77–90% mẫu — xem D6), RepoBench-P (context dài hơn).
- Kiểm định ghép cặp cho Phase 5 (bootstrap như Phase 0).
