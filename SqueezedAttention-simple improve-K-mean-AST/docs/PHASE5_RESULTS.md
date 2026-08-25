# Phase 5 (C2) — Recall@budget · **KẾT QUẢ ÂM**

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

## Những gì kết quả này KHÔNG nói

| | |
|---|---|
| **Đề xuất 2 (`struct_hierarchy`) chưa được đo** | Thước đo chấm điểm key bằng centroid **L2**, mà L2 của `struct_hierarchy` chính là centroid của `hard_boundary`. Tầng L1 chỉ tham gia ở bước lookup phân tầng — `phase5_recall.py` không mô phỏng bước đó. Hai nhánh ra số **trùng khít từng chữ số**, đó là hệ quả của phép đo chứ không phải phát hiện |
| Chỉ mới `level=function` | LCC có trung vị 15 đơn vị/mẫu. Level `block` (trung vị 48) hoặc `statement` chưa thử |
| Chỉ mới LCC | RepoBench-P có trung vị **107 đơn vị/mẫu** — cấu trúc dày hơn 7 lần. Chưa đo |
| Đây là **cận trên** của recall thực tế | Thước đo cắt top-N theo thứ hạng; cài đặt thật dùng ngưỡng toàn cục nên còn mất thêm. Sai số này như nhau cho mọi nhánh nên so sánh vẫn hợp lệ |

## Theo protocol thì làm gì

> *"Nếu C2 fail thì H0 sai → dừng, không chạy C1/C3."*

Không chạy Phase 6 trên cấu hình này. Ước tính chi phí tiết kiệm được: **90–390 giờ A100**
cho grid đầy đủ bốn dataset.

**Ba việc rẻ nên làm trước khi kết luận về cả hướng đi:**

1. **LCC ở `level=block`** — kiểm xem độ mịn của đơn vị có phải là vấn đề không.
   Phase 2 hai nhánh ~30 phút + Phase 5 ~20 phút.
2. **RepoBench-P** — nơi cấu trúc dày gấp 7 lần. Nếu ranh giới cấu trúc giúp ở đâu thì phải
   là ở đó. Phase 2 hai nhánh ~18 giờ.
3. **Đo đề xuất 2 cho đúng** — cần mô phỏng lookup phân tầng (L1 lọc trước, L2 lọc sau),
   hoặc đo bằng `centroid_lookup` thật thay vì xếp hạng.

Nếu cả ba đều âm thì đây là kết quả chính của dự án, và là một kết quả **có giá trị công
bố**: một giả thuyết hợp lý về mặt trực giác, được kiểm bằng phép đo trực tiếp, và bị bác bỏ
bằng số liệu nhất quán trên 10 lớp × 3 mức ngân sách × 300 mẫu.
