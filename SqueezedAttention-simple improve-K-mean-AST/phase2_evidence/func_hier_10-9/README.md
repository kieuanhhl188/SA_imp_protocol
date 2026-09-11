# Phase 5 — lookup PHÂN TẦNG 2 bước, LongChat-7B, LCC, 200 mẫu (10/9/2026)

Lần **đầu tiên** `phase5_recall.py` thực sự đọc tầng L1
(`hierarchical_centroids_*_L1_*.pt`) và định tuyến 2 bước — cơ chế do commit `23473cc`
("mo phong lookup 2 buoc") thêm vào. Các lượt trước (31/8 function, 9/9 block) chỉ đo
recall **phẳng** trên L2, nên `struct_hierarchy` luôn ≡ `hard_boundary` từng bit và câu
hỏi "đề xuất 2 có tách khỏi đề xuất 1 không" chưa được trả lời. README block 9/9 ghi rõ
việc này còn thiếu.

## Lệnh

```
python phase5_recall.py longchat-v1.5-7b-32k --dataset lcc \
    --cluster_dir "sa=/workspace/p2-longchat/sa/lcc" \
    --cluster_dir "hard_boundary=/workspace/p2-longchat/hard_boundary/lcc" \
    --cluster_dir "struct_hierarchy=/workspace/p2-longchat/struct_hierarchy/lcc" \
    --phase1_dir "$SQA_PHASE1_DIR/longchat-v1.5-7b-32k" \
    --sparsity 50 60 70 80 90 --limit 200 --hierarchical --l1_ratios 1.0 0.9 0.7 0.5 \
    --out phase5_lcc_func_hier_sp5.json
```

- Cây centroid function-level `/workspace/p2-longchat/` (200 mẫu, ~85 GB) — **KHÔNG vào git**.
- `n = 200 mẫu × 3 lớp (first/mid/last) = 600` lượt quan sát mỗi ô.
- **`--sparsity 50 60` nằm NGOÀI protocol.** Protocol Phase 5 chốt budget ∈ {70, 80, 90%};
  50/60 chỉ thêm để kiểm tính đơn điệu, xem phụ lục.

## Cơ chế lookup 2 bước (`recall_hierarchical`, `phase5_recall.py:118`)

```
Bước 1 (L1):  s1 = q · centroid_L1     → giữ top ⌈r·K1⌉ nhóm L1 gần query nhất
Bước 2 (L2):  key thuộc nhóm L1 bị loại → điểm = −inf; lấy top-N phần còn lại
```

`r = 1.0` giữ hết nhóm → trùng đúng recall phẳng (kiểm chứng cài đặt). `r < 1.0` là
đề xuất 2 thật sự: lọc thô ở L1 trước, tinh ở L2 sau.

## Recall — `phase5_lcc_func_hier_sp5.json`

| nhánh / r | sp50 | sp60 | **sp70** | **sp80** | **sp90** |
|---|---:|---:|---:|---:|---:|
| `sa` (r=1.0) | 0,8185 | 0,7831 | **0,7433** | **0,6944** | **0,6242** |
| `hard_boundary` (r=1.0) | 0,8146 | 0,7769 | 0,7340 | 0,6801 | 0,6014 |
| `struct_hierarchy` r=1.0 | 0,8146 | 0,7769 | 0,7340 | 0,6801 | 0,6014 |
| `struct_hierarchy` r=0,9 | 0,7950 | 0,7598 | 0,7198 | 0,6691 | 0,5934 |
| `struct_hierarchy` r=0,7 | 0,7258 | 0,6969 | 0,6647 | 0,6232 | 0,5578 |
| `struct_hierarchy` r=0,5 | 0,6395 | 0,6094 | 0,5850 | 0,5581 | 0,5121 |

`mass` (attention giữ lại / oracle) cùng xu hướng: `sa` 0,94–0,98; `struct_hierarchy` tụt
từ 0,94 (r=1.0) xuống 0,78 (r=0,5) ở sp90.

## Kiểm định ghép cặp — `phase5_func_hier_sp5_bootstrap.txt`

`python scripts/phase5_bootstrap.py phase5_lcc_func_hier_sp5.json`
(gộp 3 lớp theo mẫu → resample 200 mẫu, bootstrap 20.000, percentile 95%).

Hiệu số ×100 = nhánh − `sa`:

| config | sp70 | sp80 | sp90 |
|---|---|---|---|
| `hard_boundary` = `struct_hierarchy` r=1.0 | −0,93 [−1,12; −0,76] | −1,43 [−1,65; −1,22] | −2,28 [−2,55; −2,02] |
| `struct_hierarchy` r=0,9 | −2,35 [−2,70; −2,04] | −2,54 [−2,87; −2,23] | −3,09 [−3,42; −2,77] |
| `struct_hierarchy` r=0,7 | −7,86 [−8,82; −6,99] | −7,12 [−8,08; −6,27] | −6,64 [−7,53; −5,85] |
| `struct_hierarchy` r=0,5 | −15,83 [−16,97; −14,72] | −13,63 [−14,76; −12,55] | −11,22 [−12,24; −10,25] |

**Tất cả 15 khoảng tin cậy (5 sp × 3 dòng chọn) loại trừ 0, tất cả âm.**

## Ba kết luận

1. **`struct_hierarchy` r=1.0 ≡ `hard_boundary` từng chữ số** — vẫn đúng, và đúng theo
   thiết kế: L2 của cả hai do cùng `hard_boundary_kmeans` sinh; `r=1.0` không loại nhóm
   nào. Đây là test cài đặt, PASS.

2. **Bật lọc thô L1 (`r < 1.0`) chỉ làm MẤT recall, đơn điệu.** sp70: −0,93 → −2,35 →
   −7,86 → −15,83 khi `r` đi 1.0 → 0,9 → 0,7 → 0,5. Lookup 2 bước (cơ chế thật của đề
   xuất 2) **kém hơn hẳn** tìm phẳng (đề xuất 1) ở mọi cấu hình. Khớp bất đẳng thức:
   `recall(hierarchical, r) ≤ recall(flat)`, dấu = chỉ khi `r → 1.0` — một tầng chỉ mục
   thô lossy không thể vượt trần "không lọc".

3. **C2 FAIL cho đề xuất 2, y như đề xuất 1.** Tiêu chí 5.5 (*"pass nếu structure-aware
   recall CAO HƠN SA có ý nghĩa ở ≥2 budget"*) fail ở cả 3 mức protocol 70/80/90 — và
   lệch về phía ngược lại. Trần của đề xuất 2 = đề xuất 1 = `hard_boundary`, vốn đã thua
   `sa` (KTC loại 0). Không giá trị `l1_ratio` nào lật được điều này.

Đối chiếu lượt 31/8 (function, n=100, phẳng): sp70 −1,00 [−1,29; −0,74] → nay n=200
−0,93 [−1,12; −0,76]. Nhất quán, KTC hẹp hơn.

## Không làm (có lý do)

- **Sinh lại block-level rồi chạy hierarchical**: cây `/workspace/p2-longchat-block/` đã
  mất khỏi MooseFS (xem `../block_longchat_9-9/`). Function-level + bất đẳng thức ở kết
  luận 2 đã đủ; block chỉ làm gap rộng hơn (~3× theo lượt 9/9), không lật kết luận.
- **Cluster lại tầng L1 (k-means-của-k-means)** thay vì "trung bình theo function" của
  protocol: rời định nghĩa đề xuất 2, và vẫn dính bất đẳng thức kết luận 2.
- **`--limit` > 200**: cây centroid chỉ có 200 mẫu; KTC hiện đã đủ hẹp.

## Còn treo (không chặn kết luận C2)

- LCC context ~3k token — quá ngắn cho structure-aware clustering. Cần RepoBench v1.1
  (context dài) để nói structure-aware "vô dụng ở mọi độ dài" hay chỉ "ở context ngắn".
- Phase 6: accuracy@budget end-task với kernel phân tầng online.
