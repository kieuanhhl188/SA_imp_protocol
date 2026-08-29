# Phase 1 — Tóm tắt báo cáo

Chốt 20/8/2026 · Qwen2.5-Coder-7B (base) · A100-80GB
Chi tiết đầy đủ: [PHASE1_RESULTS.md](PHASE1_RESULTS.md) (8 bảng)

---

> ## ⚠️ ĐỔI PHẠM VI 28/8/2026 — quay về LongChat-7B + LCC-only
>
> Model chính đổi từ Qwen **trở lại `longchat-v1.5-7b-32k`**, LCC-only, khớp Phase 0. Mục 1–6
> bên dưới là hồ sơ lượt Qwen 20/8, giữ làm tham chiếu.
>
> **Trạng thái Phase 1 cho LongChat (29/8):** ✅ gate dữ liệu 1.4 **PASS** — `check_phase1_data.py
> longchat-v1.5-7b-32k --dataset lcc`, cả 5 bước. 500 mẫu LCC, 2.094.562 token kiểm, 0 lệch
> token id, offset byte↔ký tự khớp tuyệt đối 500/500, 1/500 truncate, 2,4% mẫu suy biến, unit
> trung vị 15. Sản phẩm: `phase1_data/longchat-v1.5-7b-32k/`.
>
> Accuracy lấy từ Phase 0 (**n=3**, 29/8): All-KV **54,83 ± 0,00** · Sq-70% **56,36 ± 0,28** ·
> hiệu ghép cặp +1,25 / +1,80 / +1,54 (hai lượt gần nhất bootCI95 loại trừ 0). Chi tiết:
> [PHASE1_RESULTS.md](PHASE1_RESULTS.md).

---

## 1. Kết quả chính

Bản port Squeezed Attention sang Qwen2 (GQA) **chạy đúng**. LongBench LCC, 20 mẫu:

| | Điểm |
|---|---:|
| All-KV (trần accuracy) | **65,35** |
| Sq-70% (bỏ 70% key) | **62,55** |
| Hiệu số | −2,80 · KTC95 `[−7,12; +1,52]` · p = 0,22 |

**Chênh lệch không có ý nghĩa thống kê.** 14/20 mẫu cho prediction *y hệt nhau* dù đã bỏ 70%
số key — bằng chứng đường GQA trả đúng nhóm centroid.

⏳ Đang chạy lại trên 500 mẫu; KTC95 sẽ hẹp từ ±4,3 xuống ±0,9.

**Ngược chiều với hai mốc tham chiếu:**

| | LongChat (tái lập) | Bài gốc (Table 2) | Qwen (đo được) |
|---|---:|---:|---:|
| Hiệu số Sq-70% − All-KV | +1,25 | +0,29 | **−2,80** |

---

## 2. Dữ liệu đã chuẩn bị

500 mẫu LCC + 500 mẫu RepoBench-P, có offset ký tự từng token, **gate 5 bước PASS cả hai**:

| | LCC | RepoBench-P |
|---|---:|---:|
| Token đã kiểm | 1.559.310 | 4.882.207 |
| Lệch token id · mẫu mất context | **0 · 0** | **0 · 0** |
| Đơn vị cấu trúc / mẫu (trung vị) | 15 | 100 |

Số đo trên máy Windows và pod A100 **trùng khít, không lệch một token**.

---

## 3. Chi phí đo được (dữ liệu C3 đầu tiên)

Ba lần chạy độc lập cho cùng kết quả:

| Bước | s/mẫu | So với All-KV |
|---|---:|---:|
| pred All-KV | 2,3 | 1,0× |
| **pred Sq-70%** | **19,3** | **8,4×** |

Chi phí nằm ở **decode**, không phải prefill. LCC có ngữ cảnh trung vị 2.194 token — nhiều
khả năng **dưới điểm hoà vốn** của phương pháp; bài gốc đo ở 32K+.

---

## 4. Ba lỗi đáng kể nhất (trong 8 lỗi đã sửa)

Cả ba đều **không crash**, chỉ làm sai lệch kết quả:

| Lỗi | Hậu quả nếu không sửa |
|---|---|
| Ngưỡng `NaN` do `exp` tràn | Sq-70% ra **23,05** thay vì 62,55 |
| `language` hardcode `"python"` | 63,6% mẫu LCC là Java/C# → **59,5% mẫu mất hết ranh giới cấu trúc**, ablation Idea 1 vô hiệu mà không báo lỗi |
| Trộn byte với ký tự trong span AST | 107/500 mẫu RepoBench-P lệch span cộng dồn |

Hai lỗi đầu nằm ở **code dùng chung với bài gốc**, chỉ bộc lộ trên họ model Qwen2.

---

## 5. Giới hạn phải ghi vào bài

**LCC và RepoBench-P không kiểm chứng được premise của Squeezed Attention.** Mỗi mẫu là một
cặp `(context, query)` độc lập → tỉ lệ khấu hao thực tế là **1 query / 1 lần clustering**,
trong khi PreFixQA của bài gốc có ~24.

| Claim | Bị ảnh hưởng? |
|---|---|
| C2 — recall@budget | không |
| C1 — accuracy@budget | không (mọi method cùng điều kiện) |
| **C3 — chi phí khấu hao** | **có** |

Xử lý: bổ sung **RepoBench v1.1** ở Phase 6 (1.646 fixed context ≥16K dùng chung, nhãn thật).

---

## 6. Trạng thái

| Hạng mục | |
|---|---|
| 1.1–1.7 theo protocol | ✅ **hoàn tất**, gate PASS |
| Còn lại (cần GPU) | chạy port gate 500 mẫu — **đang chạy** |
| Còn lại (CPU) | loader RepoBench v1.1 — chặn Phase 6, không chặn Phase 2/5 |

**Phase 1 đủ điều kiện mở Phase 2.**
