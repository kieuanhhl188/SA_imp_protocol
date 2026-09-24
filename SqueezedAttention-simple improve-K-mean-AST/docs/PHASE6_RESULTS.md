# Phase 6 (C1) — Độ chính xác end-task · RepoBench-P / `class` · **KHÔNG ĐẠT**

Chốt 24/9/2026 · LongChat-v1.5-7B-32K · RepoBench-P · **n = 200** (cùng 200 mẫu, cùng thứ tự
cho cả 3 cấu hình) · ngân sách centroid 5% (`PC5`) · sparsity 70% (`PERC0.7`) · 1 seed (42)

Nguồn: `LongBench/pred/longchat-v1.5-7b-32k_*_lim200_runfull200*/` ·
số in ra: [phase6_evidence/repobench_class_24-9/](../phase6_evidence/repobench_class_24-9/) ·
tái lập: `python scripts/phase6_bootstrap.py`

---

## Vì sao chạy cấu hình này

Phase 5 (C2, recall@budget) — sweep 4 mức `--level` trên RepoBench-P ngày 12–13/9: **`class` là cấu
hình duy nhất pass cổng C2** (≥2 mức budget cao hơn `sa` có ý nghĩa thống kê):

| Phase 5 · `hard_boundary(class) − sa` · recall | sp70 | sp80 | sp90 |
|---|---:|---:|---:|
| Hiệu số (điểm %) | **+0,30** | **+0,16** | −0,09 |
| KTC95 loại 0 | ✅ | ✅ | ✅ (âm) |

Xem [PHASE5_RESULTS.md](PHASE5_RESULTS.md) và EXPERIMENT_LOG mục 6, entry 12/9 (d). Phase 6 kiểm
tra xem mức tăng recall đó có chuyển thành độ chính xác sinh code thật hay không.

> ⚠️ **Mới chạy sp70.** sp80 — mức pass C2 thứ hai — **chưa chạy end-task**.

## Kết quả

| Cấu hình | EM (%) | ES | ΔES so với All-KV [KTC95] | ΔES so với SA-70% [KTC95] | T / B / H so với SA-70% | Thời gian/mẫu (s) | Đỉnh VRAM max / TB (GiB) |
|---|---:|---:|---|---|---:|---:|---:|
| All-KV (cache đầy đủ) | **12,00** | **58,15** | — | +0,69 [−0,33; +1,99] | 10 / 10 / 180 | **3,81 ± 1,92** | **47,14** / 28,38 |
| SA-70% (K-means thuần) | 11,00 | 57,45 | −0,69 [−1,99; +0,33] | — | — | 71,49 ± 46,46 | 48,66 / 29,08 |
| Class-70% (`hard_boundary`, đề xuất) | 10,50 | 56,36 | −1,78 [−3,71; +0,13] | **−1,09 [−3,25; +1,12]** | **30 / 42 / 128** | 70,61 ± 44,72 | 48,66 / 29,08 |

*T / B / H = số mẫu Thắng / Bại / Hoà theo ES.* Kiểm định là bootstrap ghép cặp: resample
chỉ số mẫu 20.000 lần, KTC95 theo percentile, seed 0, cùng một bộ resample cho mọi cặp. p hai
phía theo bootstrap:

| Cặp | ΔES | p | ΔEM [KTC95] | Mẫu giống hệt |
|---|---:|---:|---|---:|
| SA-70% − All-KV | −0,69 | 0,225 | −1,00 [−2,50; +0,00] | 180/200 (90%) |
| Class-70% − All-KV | −1,78 | 0,067 | −1,50 [−4,00; +0,50] | 132/200 (66%) |
| Class-70% − SA-70% | −1,09 | **0,322** | −0,50 [−2,50; +1,50] | 128/200 (64%) |

`compare_runs.py` (KTC xấp xỉ chuẩn ±1,96·SE) cho cùng kết luận: Class−SA −1,09 [−3,25;
+1,07], sign test p = 0,19; Class−All-KV −1,78 [−3,69; +0,13], p = 0,11.

Các mẫu Class-70% thua nặng nhất so với SA-70% (ES): idx 31 72→0 · idx 10 93→25 · idx 95
100→41 · idx 122 89→42 · idx 118 64→18 · idx 30 79→34.

## Chạy thế nào (23/9 16:08 → 24/9 00:22 UTC)

Script: [scripts/run_phase6_full200_repobench.sh](../scripts/run_phase6_full200_repobench.sh)
(bản gốc `/workspace/run_full_200.sh`, chép vào repo nguyên văn). Log tổng:
[master.log](../phase6_evidence/repobench_class_24-9/master.log) (bản đầy đủ ở
`/workspace/logs_full_run_20260923_160815/`).

Tham số chung: `--model longchat-v1.5-7b-32k --task repobench-p --limit 200 --fixed_context full --resume`.
Hai nhánh centroid thêm `--use_centroids --percent_clusters 5 --percentile 0.7 --obs_window 100`.

| Nhánh | `--path_to_clusters` | `--run_tag` | Thời gian chạy |
|---|---|---|---|
| All-KV | — | `full200` | 16:08 → 16:23 (15 phút) |
| SA-70% | `/workspace/p2-longchat-repobench/sa/` | `full200sa` | 16:23 → 20:24 (4 h 01) |
| Class-70% | `/workspace/p2-longchat-repobench/hard_boundary_class/` | `full200class` | 20:24 → 00:22 (3 h 58) |

**Centroid của nhánh Class đã được xác nhận.** File
[feasibility_repobench-p_hard_boundary_class_pc5.json](../phase6_evidence/repobench_class_24-9/feasibility_repobench-p_hard_boundary_class_pc5.json)
trong thư mục centroid ghi `method: hard_boundary`, `level: class`, `percent_clusters: 5`,
`n_requested: 200`. Đây cũng là thư mục mà `scripts/run_phase2_class_repobench.sh` sinh ra cho
Phase 5, tức **cùng bộ centroid đã pass C2**.

Trước đó có 3 lần khởi chạy hỏng, không sinh dữ liệu: `logs_full_run_20260923_155531` và
`…_155958` (sai đường dẫn `pred.py`), `…_160021` (`--model longchat-7b-v1.5-32k` không hợp
lệ). Bản script dùng tham số `--dataset/--method all_kv|sa|class_level` chưa từng chạy được,
vì `pred.py` không có các tham số đó.

## Số liệu được tạo ra thế nào (24/9)

**Bước 1 — so theo cặp và bootstrap.** Lệnh dự kiến ban đầu,
`python scripts/compare_runs.py LongBench/pred/longchat-v1.5-7b-32k_*_lim200_runfull200*`,
**lỗi** `unrecognized arguments`: `compare_runs.py` chỉ nhận đúng 2 file `result_detail.json`,
KTC là ±1,96·SE và không có EM. Vì vậy đã làm hai việc:
- Chạy `compare_runs.py --task repobench-p` cho từng cặp (3 cặp) →
  [step1_compare_runs.txt](../phase6_evidence/repobench_class_24-9/step1_compare_runs.txt)
- Viết `scripts/phase6_bootstrap.py` cho EM, bootstrap B=20.000 và W/L/T →
  [bootstrap.txt](../phase6_evidence/repobench_class_24-9/bootstrap.txt)

**Bước 2 — thời gian và VRAM**, đọc từ `_logs/repobench-p.stats.jsonl` (`gen_time_s`,
`peak_alloc_gib`) → [step2_runtime_vram.txt](../phase6_evidence/repobench_class_24-9/step2_runtime_vram.txt).
Log đủ 200 `dataidx`, không trùng. Độ dài context (TB 14.367 token) và số token sinh ra (53–55)
tương đương giữa 3 cấu hình, nên chênh lệch thời gian không đến từ output dài hơn.

| Cấu hình | TB ± SD (s) | Trung vị (s) | Tổng (s) | ms/token | VRAM alloc max / TB | Reserved max |
|---|---:|---:|---:|---:|---:|---:|
| All-KV | 3,81 ± 1,92 | 3,28 | 761 | 70,3 | 47,14 / 28,38 | 57,19 |
| SA-70% | 71,49 ± 46,46 | 60,58 | 14.298 | 1.336,7 | 48,66 / 29,08 | 62,61 |
| Class-70% | 70,61 ± 44,72 | 59,33 | 14.122 | 1.286,4 | 48,66 / 29,08 | 62,62 |

## Đọc kết quả này thế nào

- **C1 không được xác nhận.** Class-70% không tốt hơn SA-70% có ý nghĩa thống kê: ΔES
  −1,09, KTC95 [−3,25; +1,12], p = 0,32. Ước lượng điểm còn **ngược hướng** giả thuyết, và
  trong 72 mẫu có khác biệt thì Class thua 42, thắng 30.
- **Recall +0,30 không chuyển thành độ chính xác.** Mức tăng recall ở Phase 5 quá nhỏ so với
  nhiễu của ES end-task (KTC rộng ~4,4 điểm ở n=200). Ngược lại, ranh giới cứng `class` làm
  output lệch khỏi All-KV ở **34%** mẫu, trong khi SA-70% chỉ lệch 10%. Thứ quyết định ES là
  *những key nào* được giữ ở mẫu khó, không phải tổng recall trung bình.
- **Chưa nói được "không kém hơn" (non-inferior).** Cận dưới KTC so với All-KV là −3,71 ES.
  Muốn kết luận "không kém hơn" phải có một ngưỡng mất mát chấp nhận được đặt **trước khi
  chạy**, và ngưỡng đó phải ≥ ~4 ES. Protocol chưa có ngưỡng như vậy, nên không được viết
  thành "tương đương".
- **Về hiệu năng: không đo được lợi ích.** Cả hai nhánh 70% chậm hơn All-KV ~18,5× và tốn
  VRAM đỉnh nhiều hơn (+1,5 GiB). Đây là chi phí của **cách cài đặt mô phỏng** (giữ nguyên KV
  đầy đủ, cộng thêm centroid và lookup; `gen_time_s` gồm cả `torch.load` centroid trong
  forward, xem comment trong `pred.py`), không phải chi phí của phương pháp. Không được dùng
  cột này để kết luận bất cứ điều gì về hiệu năng.

**Câu viết cho luận văn:** *"Trên 200 mẫu RepoBench-P, ranh giới cứng cấp class không khác
biệt có ý nghĩa thống kê so với K-means thuần (ΔES = −1,09, KTC95 [−3,25; +1,12]); C1 không
được hỗ trợ ở cỡ mẫu này."*

## Giới hạn — phải nêu kèm khi trích số

| | |
|---|---|
| Chỉ 1 seed | Protocol yêu cầu mean ± std qua ≥3 seed cho số accuracy chính (`pred.py --seed`). Decode là greedy nên seed ít ảnh hưởng, nhưng chưa kiểm chứng |
| Chỉ sp70 | sp80 (mức pass C2 thứ hai) chưa chạy end-task |
| EM là định nghĩa tự đặt | Pipeline không có sẵn EM. Dòng được so là dòng mà `code_sim_score` trích ra, strip hai đầu rồi so nguyên văn. EM chỉ đổi ở 5/200 mẫu, không phân biệt được các cấu hình |
| ES của Class làm tròn | eval.py ghi 56,36; trung bình per-sample ra 56,365, script in 56,37. Cùng một số liệu |

## Việc tiếp theo nếu còn muốn cứu C1

1. Chạy sp80 cho cả `sa` và `class` (n=200). Chỉ đáng chạy nếu vẫn cần trả lời đủ 2 mức pass
   C2. Với kết quả sp70 như trên, khả năng đảo chiều là thấp.
2. Thêm 2 seed cho sp70 để tròn yêu cầu protocol trước khi đưa bảng vào luận văn.
3. Cho `pred.py` ghi `path_to_clusters`/`level` vào `_logs/`. Lần này phải lần ngược qua script
   chạy và file feasibility mới xác nhận được cấu hình; lần sau output nên tự ghi lại.
