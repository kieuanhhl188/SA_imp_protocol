# Phase 5 (C2) — Recall@budget · **KẾT QUẢ ÂM**

> **Cập nhật 13/9/2026 — sweep đủ 4 mức `--level` (class/function/block/statement) trên
> RepoBench-P, ranh giới L2 (không phải L1 hierarchy).** Xu hướng đơn điệu: `class` (thô hơn
> function) **DƯƠNG** ở sp70/80 (+0,30/+0,16, KTC loại 0) nhưng âm ở sp90 (−0,09) · `function`
> âm cả 3 (−1,33/−1,77/−2,45) · `block` âm cả 3 (−3,65/−4,43/−5,57, n=195) · `statement` (mịn
> nhất) âm cả 3 và **nặng nhất đo được** trong toàn bộ Phase 5 (−6,75/−7,95/−9,83), nhưng
> ⚠️ chỉ đo trên **n=56/200** — 72% mẫu vượt ngân sách centroid ở mức này bị bỏ (chính sách
> skip), tập còn lại thiên lệch về mẫu ít cấu trúc hơn, nên không so trực tiếp 1-1 với 3 mức
> kia được, chỉ là quan sát củng cố thêm. Coi như đã trả lời câu "chưa thử level thô/mịn hơn
> trước khi kết luận H0" nêu ở entry 12/9 (c) của EXPERIMENT_LOG.md — không có đảo chiều bất
> ngờ ở phía mịn, tín hiệu dương chỉ tồn tại quanh `class`/sp thấp. Chi tiết đầy đủ + nguồn dữ
> liệu: [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md) mục 6, entry 2026-09-12 (d) và 2026-09-13
> (e)/(f).

> **Cập nhật 12/9/2026 — hierarchy L1 THẬT trên RepoBench-P (không còn giả "TB theo
> function"), vẫn FAIL và TỆ HƠN bản giả.** `--percent_clusters_l2 0.1` ép ngân sách L1 chặt
> hơn số class thật (trung vị 17 class/mẫu) → nhánh *merge* của `build_l1_groups` lần đầu
> chạy thật: 139/199 mẫu merge (trước đó 11/9: 199/199 đi *split*, L1 chỉ là "trung bình theo
> function", không phải hierarchy). Bootstrap ghép cặp `struct_hierarchy(r) − sa`, n=197:
> r=0,9: −2,24/−2,40/−2,84 · r=0,7: −6,58/−5,70/−5,09 · r=0,5: **−14,39/−11,75/−9,21** ở sp
> 70/80/90 — mọi KTC loại 0 và âm, nặng hơn rõ rệt so với bản hierarchy giả (r=0,5 giả:
> −7,55/−5,87/−4,85). Đóng hướng thoát "hierarchy chưa được thử thật" còn treo từ 9–11/9.
> Bằng chứng: `EXPERIMENT_LOG.md` mục 6 entry 12/9 (b) · `phase2_evidence/repobench_l1pc01_12-9/`.
>
> **Cập nhật 9/9/2026 — kết quả âm lặp lại trên LongChat.** Chạy thêm 2 cấu hình, kiểm định
> ghép cặp `hard_boundary − sa`:
> - LongChat / LCC / `function` (n=100): −1,00 / −1,49 / −2,33 ở sp 70/80/90 — KTC loại 0
> - LongChat / LCC / **`block`** (n=98): −3,24 / −4,15 / −5,44 — KTC loại 0, âm nặng hơn ~3×
>
> **9/9 khoảng tin cậy (3 cấu hình × 3 budget) đều loại trừ 0 và âm.** Bằng chứng:
> `phase2_evidence/full200_longchat_31-8/` · `phase2_evidence/block_longchat_9-9/`.
> `struct_hierarchy` vẫn trùng `hard_boundary` từng chữ số — lý do đã xác định, xem
> EXPERIMENT_LOG mục 6 entry 9/9 (L2 trùng bit theo thiết kế + tầng L1 chưa có consumer).

Chốt 24/8/2026 · Qwen2.5-Coder-7B-**Instruct** · `force_chat` · `fixed_context=full` · maxlen 31.500
LongBench LCC · **300 mẫu × 10 lớp** (0,3,6,…,27) · `level=function` · ngân sách centroid 5%

Nguồn: `phase5_lcc_full.json` · `phase5_lcc.json` · Phase 2: [PHASE2_RESULTS.md](PHASE2_RESULTS.md)

---

## Kết quả

| | sp70 | sp80 | sp90 |
|---|---|---|---|
| `sa` — K-means thuần | **68,64% / 92,68%** | **62,60% / 89,16%** | **54,23% / 82,69%** |
| `hard_boundary` — đề xuất 1 | 67,75% / 91,98% | 61,47% / 87,98% | 52,86% / 80,82% |

*(recall / attention-mass thu hồi được)*

**Kiểm định ghép cặp ở mức mẫu** — gộp 10 lớp của mỗi mẫu trước, nên n = 300 mẫu độc lập,
không phải 3.000 phép đo tương quan:

| Sparsity | Hiệu số | KTC95 | Số mẫu `hard_boundary` thắng |
|---|---:|---|---:|
| 70% | **−0,89** điểm | [−1,01; −0,77] | 56/300 |
| 80% | **−1,13** | [−1,26; −1,00] | 39/300 |
| 90% | **−1,37** | [−1,51; −1,23] | **24/300** |

Cả ba khoảng tin cậy **không chứa 0**. Tiêu chí protocol — *"Pass nếu structure-aware recall
cao hơn SA có ý nghĩa thống kê ở ≥2 mức budget"* — **không đạt**, và lệch về phía ngược lại.

## Không phải hiện tượng của một vài lớp

Hiệu số theo từng lớp ở sparsity 70:

| Lớp | 0 | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 24 | 27 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sa` | 67,4 | 66,3 | 67,7 | 68,5 | 70,4 | 69,4 | 73,5 | 70,4 | 67,3 | 65,7 |
| `hard_boundary` | 67,0 | 65,6 | 66,5 | 67,7 | 69,0 | 68,4 | 72,7 | 69,6 | 65,8 | 65,2 |
| Hiệu | −0,44 | −0,64 | −1,14 | −0,77 | −1,38 | −0,99 | −0,75 | −0,82 | **−1,52** | −0,45 |

**10/10 lớp âm.** Đây là lý do phải mở rộng từ 3 lớp lên 10 trước khi kết luận: nếu chỉ vài
lớp âm thì câu chuyện sẽ khác hẳn.

## Đọc kết quả này thế nào

Ghép với số đo của Phase 2 sẽ ra một phát biểu chặt:

> K-means thuần vắt qua ranh giới AST ở **44,5%** số cluster (Phase 2). Cấm những lần vắt đó
> làm recall giảm **~1 điểm** một cách nhất quán (Phase 5). Vậy các lần vắt biên **mang thông
> tin hữu ích**, không phải nhiễu.

Nói cách khác: **ranh giới cú pháp không trùng với ranh giới ngữ nghĩa của attention.** Các
key mà truy vấn cần thường nằm rải ở nhiều function; ép mỗi function thành một cụm riêng làm
centroid tóm tắt kém đi.

**Về độ lớn, phải nói cho đúng mức.** Giảm ~1 điểm trên nền ~68 điểm, trong khi can thiệp
thay đổi 44,5% số cluster. Nghĩa là phần lớn việc vắt biên là **trung tính**, và cấm nó phải
trả một cái giá nhỏ nhưng có hệ thống. Không được viết thành "cấu trúc phá hoại nghiêm trọng".

## Những gì kết quả này KHÔNG nói (bảng gốc 24/8 — chỉ áp dụng cho cấu hình Qwen/LCC ở trên)

| | |
|---|---|
| Đây là **cận trên** của recall thực tế | Thước đo cắt top-N theo thứ hạng; cài đặt thật dùng ngưỡng toàn cục nên còn mất thêm. Sai số này như nhau cho mọi nhánh nên so sánh vẫn hợp lệ |

Ba dòng từng đứng ở đây — *"đề xuất 2 chưa được đo"*, *"chỉ mới level=function"*, *"chỉ mới
LCC"* — **đã lỗi thời, đã xoá**. Tại thời điểm 24/8 đúng là cả ba đều đúng; sau các lượt 9/9,
10/9, 11/9, 12/9 (xem banner đầu file) cả ba đều đã được đo, trên nhiều cấu hình, và không có
cấu hình nào đảo ngược kết luận.

## Tổng hợp toàn bộ 5 cấu hình đã đo (chốt 12/9/2026)

| # | Model / dataset / level | n | sp70 | sp80 | sp90 |
|---|---|---:|---:|---:|---:|
| 1 | Qwen-fn / LCC (bảng ở trên) | 300 | −0,89 | −1,13 | −1,37 |
| 2 | LongChat-fn / LCC | 100 | −1,00 | −1,49 | −2,33 |
| 3 | LongChat-block / LCC | 98 | −3,24 | −4,15 | −5,44 |
| 4 | LongChat-fn / RepoBench-P (đề xuất 2, hierarchy **giả** — L1=1%, luôn *split*) | 197 | −1,33 | −1,77 | −2,45 |
| 5 | LongChat-fn / RepoBench-P (đề xuất 2, hierarchy **thật** — L1=0,1%, phần lớn *merge*) | 197 | −1,33 (r=1,0) → **−14,39 ở r=0,5** | −1,77 → **−11,75** | −2,45 → **−9,21** |

Tất cả là hiệu số `(nhánh − sa)`, đơn vị điểm phần trăm. **15/15 khoảng tin cậy (5 cấu hình ×
3 mức budget) loại trừ 0 và âm.** Cấu hình #5 cho thấy: hierarchy càng "thật" (đúng nghĩa
class→function→token, không phải nguỵ trang) thì càng tệ, không phải càng tốt — bác bỏ khả
năng đây chỉ là vấn đề đo chưa đúng cách.

## Theo protocol thì làm gì

> *"Tiêu chí: C2 pass nếu structure-aware recall cao hơn SA có ý nghĩa thống kê ở ≥2 mức
> budget. **Nếu không → H0 yếu, xem lại định nghĩa unit/level trước khi bỏ.**"*

**Không chạy Phase 6** — vế đầu của tiêu chí rõ ràng không đạt, 5/5 cấu hình. Nhưng câu lệnh
thứ hai của chính tiêu chí này ("xem lại định nghĩa unit/level trước khi bỏ") là một bước
**bắt buộc, chưa làm**: mọi cấu hình đã thử (5/5) dùng `--level function` hoặc `--level block`
làm ranh giới L2 — tức **bằng hoặc mịn hơn function**. Xu hướng đo được nhất quán là "mịn hơn
→ tệ hơn" (block tệ gấp ~3× function trên cùng dataset). **`--level class` — ranh giới THÔ
hơn function — chưa từng được thử làm ranh giới L2 chính** (chỉ dùng cho tầng L1 phụ trợ của
`struct_hierarchy`). Nếu xu hướng "mịn hơn → tệ hơn" tiếp diễn tuyến tính theo chiều ngược
lại, ranh giới thô hơn có thể thu hẹp gap, thậm chí đảo chiều — **khả năng này chưa bị loại
trừ**, nên "H0 sai" hiện tại đúng nghĩa "H0 yếu" theo protocol, không phải kết luận đóng.

`--level class` chỉ có ý nghĩa nơi code thật sự có class: **RepoBench-P có trung vị 17
class/mẫu** (đo được từ `k1_raw` ở thí nghiệm L1 12/9); LCC (code-completion snippet, nhiều
hàm rời rạc không nằm trong class) nhiều khả năng suy biến gần về `sa` ở mức này — không đáng
thử trước. Việc còn lại, rẻ (nhánh `sa` không phụ thuộc `--level`, tái dùng được dữ liệu đã
có; chỉ cần sinh `hard_boundary` mới ở `level=class` trên RepoBench-P + Phase 5): xem
[EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md) mục 6, entry 2026-09-12 (c).

Ba việc rẻ từng đề xuất ở bản 9/9 (LCC `level=block`, RepoBench-P, đo đề xuất 2 cho đúng bằng
lookup phân tầng thật) đã làm đủ cả ba, cả ba đều âm. Đây là bằng chứng mạnh cho hướng "H0
sai", nhưng **chưa phải kết luận cuối** cho tới khi mục "xem lại định nghĩa unit/level" ở trên
được đóng.
