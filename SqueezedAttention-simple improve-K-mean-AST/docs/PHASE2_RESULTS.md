# Phase 2 — Bảng kết quả

> ## ⚠️ ĐỔI PHẠM VI 28/8/2026 — Phase 2 chuyển sang LongChat-7B, LCC
>
> Model chính đổi từ `qwen2.5-coder-7b-instruct` **sang `longchat-v1.5-7b-32k`**, LCC-only,
> **KHÔNG `--force_chat`** — khớp phạm vi đã thu hẹp ở Phase 0 ([configs/phase0.sh](../configs/phase0.sh))
> và Phase 1 ([configs/phase1.sh](../configs/phase1.sh)), dùng đúng model của bài gốc (Table 2).
>
> **Mọi con số bên dưới là hồ sơ lượt Qwen 22/8, giữ lại làm tham chiếu — KHÔNG phải trạng
> thái hiện tại.** Phải chạy lại cả ba nhánh trên LongChat: `bash scripts/run_phase2_phase5_lcc.sh`.
>
> Ba điều đổi theo model, phải đọc lại khi có số LongChat:
> - **GQA → MHA.** Bảng 1 đếm "28 lớp × 4 KV head" là Qwen. LongChat là **32 lớp × 32 head**
>   (`num_key_value_heads = num_heads = 32`). Không có xử lý per-head kiểu QUEST App. G.
> - **Dung lượng ×8.** Centroid lưu theo số KV head → Bảng 6 (5,8 / 5,8 / 7,8 GB) thành
>   ~46 / ~46 / ~62 GB. Ba nhánh ~150 GB, sát trần volume 200 GB.
> - **Bất biến D** đối chiếu `sa` với `offline_clustering.py` — reference dir phải là bộ
>   LongChat (`fixed-prompt-clusters/longchat-v1.5-7b-32k/lcc`), không phải bộ Qwen.
>
> Không đổi theo model (tính chất của LCC): Bảng 5 (K1 thực tế bị chặn ở số function),
> chính sách D6 (skip khi vượt ngân sách), thứ tự level function → block → statement.

---

Chốt 22/8/2026 · ~~Qwen2.5-Coder-7B (base)~~ · LongBench LCC **500/500 mẫu** · A100-80GB
Ngân sách centroid 5% · observation window 100 · `level=function` · `level_l1=class`

Nguồn: `phase2_invariants_v2.log` · `feasibility_lcc_*_function_pc5.json` · `k1_stats_*.pt`
Chi tiết quyết định: [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md) · Phase 1: [PHASE1_RESULTS.md](PHASE1_RESULTS.md)

> **Phase 2 đo VIỆC THI HÀNH, không đo CHẤT LƯỢNG.** Không có accuracy, không có recall ở đây.
> Nó trả lời: ranh giới cứng có được thi hành đúng không, can thiệp mạnh tới đâu, và ba nhánh
> có so sánh được với nhau không. Đo chất lượng là việc của Phase 5 (C2) và Phase 6 (C1).

---

## Bảng 1 — KẾT QUẢ CHÍNH: ranh giới cứng có ràng buộc vào cái gì thật không

Đếm cluster chứa token của **từ hai đơn vị AST trở lên**, cộng dồn qua 28 lớp × 4 KV head,
trên toàn bộ 500 mẫu.

| Nhánh | Cluster vắt qua ranh giới AST |
|---|---|
| `sa` — K-means thuần (bài gốc) | **trung vị 44,5%** · p25 36,0% · p75 52,6% · max 88,0% |
| `hard_boundary` — đề xuất 1 | **0,0% — đúng 500/500 mẫu** |
| `struct_hierarchy` — đề xuất 2 | **0,0% — đúng 500/500 mẫu** |

**Hai kết luận, cả hai đều cần thiết:**

1. **Ranh giới cứng được thi hành tuyệt đối.** Không một cluster nào trong 500 mẫu vi phạm.
   Đây là định nghĩa của `+HardBoundary`; vỡ bất biến này thì tên gọi không còn đúng.
2. **Can thiệp có liều lớn.** K-means thuần trộn token từ nhiều function ở **gần một nửa**
   số cluster. Nếu con số này gần 0 thì `hard_boundary` sẽ cho ra phân hoạch gần trùng `sa`,
   và mọi chênh lệch accuracy đo được sau này sẽ vô nghĩa — không phải vì ý tưởng sai mà vì
   can thiệp chưa từng xảy ra. Đo được 44,5% nghĩa là **có đủ chỗ cho hiệu ứng tồn tại**.

Mẫu có tỉ lệ thấp nhất (0,3% · 0,7% · 1,0%) đều là mẫu **U=2**, tức file gần như không có
cấu trúc function — không có ranh giới nào để cấm. Không phải phản chứng.

---

## Bảng 2 — Ba nhánh có so sánh được với nhau không

| Kiểm | Kết quả |
|---|---|
| Mẫu chạy được | `hard_boundary` **500/500** · `struct_hierarchy` **500/500** |
| Mẫu bị bỏ vì vượt ngân sách | **0** cả hai nhánh |
| Mẫu phải gộp unit | **0** cả hai nhánh |
| Tổng K danh nghĩa | bằng nhau giữa ba nhánh |
| Shape centroid `[1,H,K,D]` · nhãn `[1,H,n_ctx]` | đúng, mọi mẫu |

Ba nhánh chạy trên **đúng cùng 500 mẫu, cùng ngân sách** — điều kiện tiên quyết để so sánh
accuracy ở Phase 6 mà không cần hiệu chỉnh gì.

---

## Bảng 3 — Ngân sách HIỆU DỤNG: ô centroid không được dùng

Đếm hàng centroid toàn 0 (cuML cấp phát đủ K hàng nhưng không dùng hết).

| Nhánh | Ô centroid rỗng (trung bình 500 mẫu) |
|---|---:|
| `sa` | **0,71%** |
| `hard_boundary` | **0,00%** |
| `struct_hierarchy` | **0,00%** |

**Phải ghi vào bài.** Hai nhánh cấu trúc dùng **hết** ngân sách, còn K-means thuần lãng phí
0,71%. Nghĩa là ở cùng `K` danh nghĩa, `hard_boundary` có nhiều hơn ~0,7% centroid **thực
dùng**. Chênh lệch nhỏ nhưng có hệ thống, và nó là hệ quả trực tiếp của ràng buộc "mỗi unit
tối thiểu 1 centroid". Khi báo cáo accuracy phải nói rõ điều này, không được để người đọc
hiểu rằng hai bên có ngân sách hiệu dụng y hệt nhau.

---

## Bảng 4 — Bất biến D: nhánh `sa` có tái lập bản gốc không

Nhánh `sa` của `offline_clustering_struct.py` phải cho ra centroid trùng với
`offline_clustering.py` gốc — đây là **nhóm đối chứng âm**, kiểm cái giá đỡ (hook, truncation,
threshold) chứ không kiểm Squeezed Attention.

| | |
|---|---|
| Mẫu lệch quá ngưỡng 5% | **55/500 (11%)** |
| Dải khoảng cách tập hợp (chuẩn hoá) | **5,7e-03 … 3,4e-01** |

**Đọc thế nào.** 445/500 mẫu khớp trong 5%. Thước đo là **Hausdorff một chiều lấy max**: chỉ
cần một centroid duy nhất rơi khác chỗ là con số vọt lên, dù 99% centroid còn lại trùng khít.

Bằng chứng ủng hộ cách đọc này: `median norm` của centroid **trùng nhau tới 4 chữ số ở cả 28
lớp** giữa hai lượt chạy. Nếu giá đỡ sai thì thống kê tổng thể phải lệch, không thể trùng.

> ### ⚠️ ĐÍNH CHÍNH 24/8 — nguyên nhân ghi ở bản trước là SAI
>
> Bản trước quy phần lệch cho *"cuML khởi tạo ngẫu nhiên, không ghim `random_state`"* và đề
> xuất *"ghim seed rồi sinh lại, ~45 phút GPU"*.
>
> **Seed đã ghim sẵn.** [squeezedattention/clustering.py:69](../squeezedattention/clustering.py#L69)
> đặt `random_state=0`, và `git show b03a63d` cho thấy dòng đó có từ **first commit** — tức
> từ code gốc của Squeezed Attention, không phải thứ ai đó quên. Cả hai bên của phép đối
> chiếu đều gọi đúng hàm `run_clustering` đó, nên cả hai đều đang chạy với seed ghim. **Cách
> sửa được đề xuất là no-op**: chạy 45 phút GPU rồi ra đúng con số cũ.
>
> Nguyên nhân thật chưa biết, và ba khả năng còn lại đòi ba cách xử lý khác nhau:
>
> | | Khả năng | Nếu đúng thì phải làm gì |
> |---|---|---|
> | 1 | cuML **không tất định dù đã ghim seed** — Lloyd iteration reduce bằng atomic trên GPU, thứ tự cộng khác nhau mỗi lượt, rồi `tol=1e-4` chặn sớm ở vòng khác | Không sửa được bằng seed. Phải báo cáo một **ngưỡng sàn** của phép đo, và đổi metric sang loại ổn định số: `inertia`, `ARI` |
> | 2 | **Key vector** không giống nhau giữa hai lượt forward | Nặng hơn k-means nhiều: mọi con số Phase 2 sinh ở hai thời điểm khác nhau đều không đối chiếu được |
> | 3 | Thư mục reference sinh bởi **code/config khác** (transformers, `rope_scaling`, `force_chat`, `fixed_context`, `maxlen`) | Sinh lại reference bằng đúng cấu hình. Không liên quan gì tới seed |
>
> **Cách tách:** [scripts/diag_invariant_d.py](../scripts/diag_invariant_d.py) — ba tầng đo
> lồng nhau, mỗi tầng loại một khả năng. T1 forward hai lần cùng prompt trong cùng process
> (loại 2); T2 gọi `run_clustering` hai lần trên **cùng** key A (đo riêng khả năng 1, cho ra
> ngưỡng sàn); T3 so kết quả T2 với file trên đĩa — `T3 ≈ T2` thì là nhiễu cuML, `T3 ≫ T2`
> thì còn nguyên nhân thứ ba. Mặc định 3 mẫu, **~2–3 phút GPU** — rẻ hơn ~15 lần so với cách
> cũ, và trả lời đúng câu hỏi hơn.
>
> Chưa chạy. Đến khi chạy thì Bảng 4 vẫn là **giới hạn đã biết**, không phải kết luận
> "port sai" — nhưng cũng **không được viết nguyên nhân là seed**.

---

## Bảng 5 — Tầng L1 của `struct_hierarchy` KHÔNG phải "1% context"

| | Trung vị | Trung bình | Max |
|---|---:|---:|---:|
| K1 **thực tế** | **14** | **18,2** | 181 |
| K1 danh nghĩa (1% context) | 22 | 30,7 | — |
| Mẫu bị chặn dưới mức danh nghĩa | **355/500 (71%)** | | |

Nguyên nhân: `build_l1_groups` không thể tạo nhiều nhóm hơn số unit con. LCC có trung vị ~15
function mỗi mẫu, trong khi 1% context cho phép ~22 nhóm — nên 71% số mẫu bị chặn ở **số
function**, và tầng L1 thực chất là *"trung bình theo function"*.

**Ba hệ quả bắt buộc:**

1. **Không viết "L1 = 1% context"** trong bài. Phải lấy K1 đo được từ `k1_stats_*.pt`.
   Budget vẫn hợp lệ (K1 nhỏ hơn danh nghĩa, không lớn hơn) nhưng con số phải đúng.
2. **Không mô tả đây là hierarchy ba tầng class→function→token.** Cờ `--level_l1` không hề
   tham gia ở 71% số mẫu.
3. **Muốn khảo sát `--level_l1` phải làm trên RepoBench-P** (chỉ 29,5% mẫu bị chặn) hoặc trên
   LCC với L2 ở level mịn hơn. Quét `--level_l1` trên LCC ở `level=function` sẽ cho **cùng
   một kết quả với mọi giá trị** — rất dễ đọc nhầm thành "hierarchy không nhạy với level".

---

## Bảng 6 — Chi phí và độ tin cậy vận hành

| | |
|---|---|
| Thời gian clustering | ~30–45 phút mỗi nhánh (500 mẫu), ~2 giờ cả ba |
| Dung lượng | 5,8 GB (`sa`) · 5,8 GB (`hard_boundary`) · 7,8 GB (`struct_hierarchy`) |
| Sau warm-up | ~2–4 giây mỗi mẫu |
| **File hỏng do ổ mạng** | **3/6.500 file (0,05%)** — rỗng 0 byte, không có exception nào |

Ba file rỗng xuất hiện trong một lượt chạy **hoàn tất bình thường**, trên `/workspace` là
MooseFS qua mạng. `torch.save` tạo được metadata nhưng dữ liệu không tới đích, Python không
nhận lỗi. Bắt được nhờ `check_cluster_integrity.py` mở lại từng file bằng CRC.

→ **Quy tắc vận hành**: chạy kiểm toàn vẹn sau **mọi** job dài, không chỉ khi nghi ngờ. Và
kiểm toàn vẹn ≠ kiểm đầy đủ: CRC nói từng file còn sống, không nói mẫu nào thiếu file.

---

## Trạng thái mục theo protocol

| # | Việc | Trạng thái |
|---|---|---|
| 2.1 | Parse AST bằng tree-sitter, có offset | ✅ 5 level · 5 ngôn ngữ |
| 2.2 | Gán `unit_id` cho từng key token | ✅ 500/500 mẫu |
| 2.3 | **Hard boundary** — K-means trong từng unit | ✅ **0% vắt biên, 500/500** |
| 2.4 | **StructHierarchy** — L2 + L1 theo unit cha | ✅ chạy được; ⚠️ K1 thực tế ≠ danh nghĩa |
| 2.5 | Ablation tách bạch SA / +HB / +SH | ✅ ba nhánh, cùng 500 mẫu, cùng budget |
| 2.6 | Giữ nguyên Si, threshold, kernel | ✅ threshold do `run_global_threshold` tính, mọi nhánh |

---

## Những gì Phase 2 CHƯA trả lời

- **Ranh giới cứng có làm chất lượng tốt lên không.** Chưa đo. Đó là Phase 5 (recall@budget)
  và Phase 6 (accuracy@budget).
- **Chi phí clustering có tăng không.** Đã đo thời gian tổng nhưng chưa tách riêng so với `sa`
  ở cùng điều kiện.
- **Kết quả trên RepoBench-P.** Mới chạy LCC. Dữ liệu Phase 1 của RepoBench-P đã sẵn sàng.
- **Các level khác `function`.** `block` bỏ 2,6% mẫu (LCC) và 10,6% (RepoBench-P);
  `statement` bỏ 77–90% — xem quyết định D6.
