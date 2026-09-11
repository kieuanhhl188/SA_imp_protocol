# Bất biến [E] tầng L1 — struct_hierarchy, LongChat-7B, LCC, 200 mẫu (10/9/2026)

Đóng nốt ~10% còn lại của đề xuất 2: đến 9/9 tầng L1 **chưa có bất biến nào** —
`check_phase2_invariants.py` chỉ đọc nhãn L2. Nay thêm [E] và chạy trên toàn bộ 200
mẫu function-level còn nguyên ở `/workspace/p2-longchat/struct_hierarchy/lcc/`.

## Lệnh

```
python scripts/check_phase2_invariants.py \
    --cluster_dir "struct_hierarchy=/workspace/p2-longchat/struct_hierarchy/lcc" \
    --model longchat-v1.5-7b-32k --level function \
    --phase1_dir /workspace/phase1_data/longchat-v1.5-7b-32k --dataset lcc \
    --checks BCE
```

`--checks BCE` bỏ [A] (dựng lại 200 prompt, ~2h — struct_hierarchy đã qua [A] ở
`../full200_longchat_31-8/`) và [D] (chỉ cho `sa`). Chạy CPU ~5 phút.

## Sáu kiểm tra của [E]

| | Nội dung | Kết quả 200/200 |
|---|---|---|
| E1 | `K1 ≤ K2` | ✅ |
| E2 | mỗi cluster L2 thuộc **đúng 1** nhóm L1 (hierarchy lồng nhau, không vắt) | ✅ mọi head, mọi layer |
| E3 | `labels_l1` giống nhau mọi head (nhóm L1 là cấu trúc, không per-head) | ✅ |
| E4 | độ dài `labels_l1` == `n_ctx` | ✅ |
| E5 | centroid L1 ≈ trung bình **có trọng số** (theo #key) các centroid L2 thành viên | ✅ `rel ≤ 1,7e-7` |
| E6 | `K1` khớp hậu tố tên file + `k1_stats.k1_actual` | ✅ |

**`✅ MỌI BẤT BIẾN QUA`** — xem `p2_invariants_L1_func.log`.

## Số liệu tầng L1 (200 mẫu)

| | min / trung vị / max |
|---|---|
| K1 (nhóm L1 thực tế) | 2 / **17** / 175 |
| K2 (cluster L2) | 59 / 154 / 897 |
| mode `build_l1_groups` | split 199 · merge 1 (khớp 31/8) |
| chi phí metadata `(K1+K2)/n_ctx` | 5,0% / 5,6% / 6,0% (L2 riêng cố định 5,0%) |

- **E5 là điểm mấu chốt**: `rel ~1e-7` xác nhận file L1 đúng là *trung bình có trọng số
  của centroid L2* — tức "L1 = trung bình theo function/file" của protocol, **không phải
  k-means-của-k-means** như bài gốc.
- **E2** xác nhận hierarchy lồng nhau: không cluster L2 nào bị chẻ qua 2 nhóm L1 — điều
  kiện để lookup 2 tầng (`recall_hierarchical`) đúng nghĩa.
- Chi phí metadata thật: tầng L1 thêm trung bình **~0,6 điểm phần trăm** ngân sách centroid
  (5,0% → 5,6%), khớp khoảng "đơn tầng 2,5% → phân tầng 3%" của protocol theo tỷ lệ.

## Trạng thái đề xuất 2 sau entry này

SA + đề xuất 1 + đề xuất 2 — **machinery + bất biến XONG** (function-level).
C2 cho đề xuất 2: **FAIL** (xem `../func_hier_10-9/`). Đề xuất 2 giờ 100% ở mức Phase 2.
