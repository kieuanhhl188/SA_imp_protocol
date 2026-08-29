# Phase 1 — Bảng kết quả

Chốt 20/8/2026. Mọi số đều **đo được**, không có ước lượng.
Nguồn chi tiết: [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md) · [PHASE1_DATASETS.md](PHASE1_DATASETS.md)

Môi trường: A100-SXM4-80GB · torch 2.3.1+cu121 · transformers 4.40.0.dev0 (fork trong repo)
· triton 2.3.1 · flash-attn 2.6.3 · cuML 24.06 · seed 42

---

> ## ⚠️ ĐỔI PHẠM VI 28/8/2026 — quay về LongChat-7B + LCC-only
>
> Model chính đổi từ `qwen2.5-coder-7b-instruct` **trở lại `longchat-v1.5-7b-32k`**, chỉ LCC,
> **không `--force_chat`** — khớp phạm vi đã thu hẹp ở Phase 0 ([PHASE0.md §8](PHASE0.md)) và
> dùng đúng model của bài gốc (Table 2). Toàn bộ Bảng 1–8 bên dưới là **hồ sơ lượt Qwen 20/8**,
> giữ làm tham chiếu, không phải trạng thái hiện tại.
>
> **Phần RIÊNG còn lại của Phase 1 cho LongChat** = dữ liệu 1.4 (offset byte + ký tự) + gate
> dữ liệu 5 bước, đều CPU. Phần accuracy trùng Phase 0, không đo lại ở đây.
>
> ### ✅ Gate dữ liệu LongChat — PASS 29/8/2026
>
> `python scripts/check_phase1_data.py longchat-v1.5-7b-32k --dataset lcc` → **PASS toàn bộ**.
> Sản phẩm: `phase1_data/longchat-v1.5-7b-32k/` (`lcc_meta.jsonl` 500 dòng + `lcc_offsets.npz`
> 1000 mảng: `offsets_` ký tự + `offsets_bytes_` byte).
>
> | Bước | Kết quả |
> |---|---|
> | 1 — ngôn ngữ đúng từng mẫu | ✅ `python 182 · java 160 · csharp 158` · 0 mẫu lệch · 0 mẫu phải đoán |
> | 2 — đủ mẫu (meta + npz) | ✅ 500/500 |
> | 3 — offset: fast==slow · không giảm · phủ kín · byte↔ký tự | ✅ 0 lệch token id · hở ≤8 ký tự · **byte cắt ra đúng chuỗi ký tự 500/500** |
> | 4 — `fixed_context`: `sp_len` khớp · `n_ctx>0` · context không mất | ✅ 0 mẫu lệch · 0 mẫu mất context |
> | 5 — tổng kết | ✅ **PASS** |
>
> | Số đo | LongChat / LCC |
> |---|---:|
> | Token đã kiểm | **2.094.562** (khác Qwen 1.559.310 do tokenizer LLaMA) |
> | Mẫu bị truncate | 1/500 |
> | Token vắt biên unit | 1 / 2.094.562 ≈ 0,00% |
> | Unit/mẫu ở level function (trung vị · min · max) | 15 · 2 · 280 |
> | Mẫu suy biến (U ≤ 2) | 12/500 (2,4%) |
> | Mẫu chứa Unicode | 0/500 — phép thử vi sai không áp dụng (LCC thuần ASCII) |
>
> **Accuracy lấy từ Phase 0** (LongChat / LCC, **n=3**, chốt 29/8): All-KV **54,83 ± 0,00** ·
> Sq-70% **56,36 ± 0,28** (56,08 / 56,63 / 56,38, đều seed K-means 0) · hiệu ghép cặp từng lượt
> +1,25 / +1,80 / +1,54 — hai lượt gần nhất bootstrap KTC95 loại trừ 0.

---

## Bảng 1 — Kết quả chính: bản port Squeezed Attention sang Qwen2 (GQA)

Qwen2.5-Coder-7B **base**, LongBench LCC, 20 mẫu đầu, Sq-70% (5% centroid, percentile 0,7)

| Cấu hình | Điểm `code_sim_score` |
|---|---:|
| All-KV (trần accuracy) | **65,35** |
| Sq-70% | **62,55** |

**So sánh theo cặp** — cùng 20 mẫu, cùng model, chỉ khác cờ `use_centroids`

| Đại lượng | Giá trị |
|---|---:|
| Hiệu số trung bình | **−2,80** |
| Độ lệch chuẩn của hiệu số | 9,85 |
| Sai số chuẩn | 2,20 |
| Khoảng tin cậy 95% | **[−7,12 · +1,52]** |
| Sign test (2 phía) | **p = 0,2188** |
| Wilcoxon signed-rank | **p = 0,2188** |
| Mẫu cho prediction **y hệt nhau** | **14/20 (70%)** |
| Sq-70% tốt hơn / kém hơn | 1 / 5 |
| Dòng được metric chấm bị rỗng | 0% / 0% |

→ **Chênh lệch không có ý nghĩa thống kê** — khoảng tin cậy chứa 0.

**Phân bố độ tụt — dồn vào một mẫu**

| dataidx | All-KV | Sq-70% | hiệu |
|---:|---:|---:|---:|
| **7** | 98,0 | 57,0 | **−41,0** |
| 5 | 36,0 | 22,0 | −14,0 |
| 3 | 32,0 | 27,0 | −5,0 |
| 11 | 100,0 | 96,0 | −4,0 |
| 14 | 24,0 | 23,0 | −1,0 |
| 12 | 18,0 | 27,0 | **+9,0** |

Riêng mẫu 7 đóng góp −2,05 trong tổng −2,80.

---

## Bảng 2 — Đối chiếu với Phase 0 và bài gốc

| Cấu hình | LongChat đo được<br>(500 mẫu) | Table 2<br>Hooper et al. | Qwen base đo được<br>(20 mẫu) |
|---|---:|---:|---:|
| All-KV | 54,83 | 56,64 | **65,35** |
| Sq-70% | 56,08 | 56,93 | **62,55** |
| **Hiệu số** | **+1,25** | **+0,29** | **−2,80** |

Qwen ra **âm** trong khi cả hai mốc kia đều dương. Chưa có ý nghĩa thống kê nên chưa kết
luận, nhưng phải theo dõi ở cỡ mẫu lớn.

---

## Bảng 3 — Gate dữ liệu 5 bước, 500 + 500 mẫu

`scripts/check_phase1_data.py` — CPU, ~2 phút mỗi dataset

| Bước | Nội dung | LCC | RepoBench-P |
|---|---|---|---|
| 1 | Ngôn ngữ đúng ở **từng mẫu** | ✅ | ✅ |
| 2 | Đủ mẫu (meta + npz) | ✅ 500/500 | ✅ 500/500 |
| 3 | Offset: fast==slow · không giảm · phủ kín · không lệch byte/ký tự | ✅ | ✅ |
| 4 | `fixed_context`: `sp_len` khớp · `n_ctx>0` · context không mất | ✅ | ✅ |
| 5 | Tổng kết | ✅ **PASS** | ✅ **PASS** (1 cảnh báo) |

**Số đo chi tiết** — trùng khít giữa máy Windows (CPU) và pod A100, không lệch một token

| | LCC | RepoBench-P |
|---|---:|---:|
| Token kiểm | 1.559.310 | 4.882.207 |
| Lệch token id fast/slow | **0** | **0** |
| Mẫu mất context | **0** | **0** |
| Mẫu bị truncate | 1/500 | 8/500 |
| Token vắt biên unit | 0,48% | 1,06% |
| Unit/mẫu (trung vị · min · max) | 15 · 2 · 280 | 100 · 4 · 669 |
| Mẫu suy biến (U ≤ 2) | 12/500 (2,4%) | 0/500 |
| Mẫu chứa Unicode — phép thử vi sai | 0/500 | **107/500, span khớp tuyệt đối** |
| Thời gian sinh offset (pod) | 37 giây | 4 phút 51 |

---

## Bảng 4 — Phân bố ngôn ngữ

Lấy từ trường `language` của **từng mẫu**, không suy đoán theo dataset.

| Ngôn ngữ | LCC | RepoBench-P |
|---|---:|---:|
| Python | 182 | 236 |
| Java | 160 | 264 |
| C# | 158 | 0 |
| **Không phải Python** | **63,6%** | **53,0%** |

Trước 20/8 mọi mẫu bị hardcode `"python"`. Tác động của việc sửa:

| | Trước | Sau |
|---|---:|---:|
| Unit/mẫu LCC (trung vị) | 2 | **15** |
| Unit/mẫu RepoBench-P (trung vị) | 2 | **100** |
| Mẫu suy biến LCC (U ≤ 2) | **59,5%** | 2,4% |
| Node `ERROR` của tree-sitter (trung vị) | 139 | **0** |

→ Không sửa thì `hard_boundary_kmeans` chạy trên **một unit duy nhất** ở 59,5% mẫu LCC, tức
**đúng bằng baseline SA**. Ablation Idea 1 sẽ ra "không khác gì SA" mà không crash, không
cảnh báo — một quyết định cài đặt hoá trang thành một phát hiện khoa học.

---

## Bảng 5 — Chi phí đo được

20 mẫu LCC, A100-80GB, **ba lần chạy độc lập cho cùng kết quả**

| Bước | Thời gian 20 mẫu | s/mẫu | So với All-KV |
|---|---|---:|---:|
| Offline clustering | 1:44 – 1:46 | 5,2 | — *(chi phí một lần)* |
| `pred.py` All-KV | 0:45 – 0:47 | 2,3 | 1,0× |
| **`pred.py` Sq-70%** | **6:21 – 6:33** | **19,3** | **8,4×** |

Chi phí nằm ở **decode**, không phải prefill: bản instruct (sinh prediction gần rỗng) chỉ mất
1:36 ở cùng bước. LCC có `n_ctx` trung vị 2.194 token — nhiều khả năng **dưới điểm hoà vốn**;
bài gốc đo ở 32K+.

**So với LongChat** — 32 head KV so với 4 head KV của Qwen

| | LongChat | Qwen | Tỉ lệ |
|---|---:|---:|---:|
| Clustering LCC | 42,5 s/mẫu | **5,4 s/mẫu** | 7,9× |
| Đĩa | ~146 MB/mẫu | **~10 MB/mẫu** | 14× |
| Cả 500 mẫu LCC | 6 giờ · 68 GB | **~45 phút · ~5 GB** | — |

---

## Bảng 6 — D6: số mẫu khả thi theo độ mịn ranh giới

991 mẫu (đã bỏ mẫu truncate), ngân sách centroid 5%

| Level | LCC chạy được | RepoBench-P | Ghi chú |
|---|---:|---:|---|
| class | 499/499 | 492/492 | |
| **function** | **499/499** | **490/492** | ← chạy trước ở Phase 2 |
| block | 486/499 (97,4%) | 440/492 (89,4%) | ghi số mẫu bỏ |
| statement | **113/499 (22,6%)** | **51/492 (10,4%)** | không gộp trong thí nghiệm chính |

⚠️ Không được so điểm `statement` trên ~23% mẫu với điểm `function` trên 100% mẫu. Phải so
trên **tập giao** các mẫu khả thi ở mọi level — `feasibility_*.json` ghi đúng danh sách
`dataidx` chứ không chỉ số đếm.

---

## Bảng 7 — Khảo sát benchmark cho fixed context dài

Đo 19/8 trên dữ liệu tải thật, không theo mô tả trong paper.

| | CrossCodeEval | **RepoBench v1.1** | **RepoEval** |
|---|---:|---:|---:|
| Fixed context ≥16k **dùng chung** | **0** | **1.646** (Py + Java) | 16 repo |
| Query / context | ≤3 (Python) | trung vị 3 · max 50 | **200** (line/api) · 46 (function) |
| Độ dài context dựng được | max 10,2K tok | max 99,9K tok | **192K – 1,19M tok** |
| Repo gốc kèm theo | ❌ xin qua email | ⚠️ clone lại được | ✅ ship sẵn 43 MB |
| Kết luận | **loại** — 9/9 biến thể quá ngắn | **dùng ngay** | **khớp premise nhất** |

Mốc so sánh: PreFixQA của bài gốc có **~24 query/context**, chứng minh ở **128K**.

---

## Bảng 8 — Lỗi tìm được và đã sửa

| # | Lỗi | Triệu chứng nếu không sửa |
|---|---|---|
| 1 | `pred.py` thiếu Qwen trong `choices` | argparse từ chối — lộ ngay |
| 2 | `pred.py` dùng tokenizer **nhanh**, clustering dùng **chậm** | assert nổ sau khi nạp xong model 15 GB |
| 3 | **`exp` tràn** trong `run_global_threshold` → ngưỡng `NaN` | **Sq-70% 23,05 thay vì 62,55**. Không crash, không assert |
| 4 | **Cluster rỗng gây `0/0`** — lộ ra sau khi sửa #3 | như trên |
| 5 | **`language` hardcode `"python"`** | 59,5% mẫu LCC còn ≤2 unit → ablation Idea 1 vô hiệu |
| 6 | **Trộn byte với ký tự** trong span AST | 107/500 mẫu RepoBench-P lệch span cộng dồn |
| 7 | Bất biến của gate quá chặt | FAIL giả trên hành vi bình thường của BPE |
| 8 | Gate so phân bố ngôn ngữ trên **hai tập mẫu khác nhau** khi có `--limit` | FAIL giả |

Lỗi #3 và #4 nằm ở **code dùng chung với bài gốc**, không phải ở bản port — chỉ nổ trên họ
model có massive activations (Qwen2), LLaMA không chạm tới.

---

## Đối chiếu từng yêu cầu của protocol — chốt 23/8

Protocol Phase 1 có **8 yêu cầu thực** (bỏ 2 mục trong thẻ `<chưa cần làm bây giờ>`).

| # | Protocol yêu cầu | Trạng thái | Bằng chứng |
|---|---|---|---|
| A | *"một fixed context → **nhiều** user input query"* | ✅ **đạt 23/8** | RepoBench v1.1: **1.735 context · 7.080 query**, trung vị 3 query/context, max 50 |
| B1 | LongBench RepoBench-P và LCC | ✅ | 500 + 500 mẫu, gate PASS |
| B2 | CrossCodeEval và RepoEval / RepoBench | ✅ CCE + RepoBench · ⏸️ RepoEval | CCE **9.928 mẫu · 4 ngôn ngữ**. RepoEval bỏ có chủ đích — không có trong grid Phase 6, và RepoBench v1.1 đã phủ mục đích |
| C1 | `fixed_context` = cross-file + phần file trước con trỏ | ✅ | Cờ `--fixed_context full` (mặc định), đóng dấu vào meta, `load_phase1` chặn khi lệch |
| C2 | `user_input` = dòng/khối cần hoàn thành | ✅ | |
| C3 | **Lưu byte offset** từng token | ✅ | Lưu **cả byte lẫn ký tự**; bất biến "byte cắt ra đúng chuỗi như ký tự" khớp tuyệt đối trên 100% mẫu của cả 4 bộ |
| D1 | Model long-context, mở, đủ mạnh — đề xuất **Qwen2.5-Coder-7B-Instruct (128K)** | ✅ Instruct · ⚠️ 128K | Instruct + chat template: **67,80**. 128K cần YaRN mà fork không có → đã port RoPE scaling, bật bằng `--rope_scaling dynamic:4` |
| D2 | GQA per-head theo QUEST Appendix G | ✅ | 4 KV head → 28 Q head, xác nhận trên GPU |

## Bốn dataset đã sẵn sàng

Tất cả sinh cùng cấu hình: **Instruct · `force_chat` · `full` · maxlen 31.500**.

| | LCC | RepoBench-P | CrossCodeEval | **RepoBench v1.1** |
|---|---:|---:|---:|---:|
| Context | 500 | 500 | 9.928 | **1.735** |
| Query | 500 | 500 | 9.928 | **7.080** |
| Query/context | 1 | 1 | 1 | **3 (max 50)** |
| Ngôn ngữ | py·java·cs | py·java | py·java·cs·**ts** | py·java |
| `n_ctx` trung vị | 2.294 | 8.328 | 946 | **16.154** |
| Đơn vị AST/mẫu | 15 | 107 | 6 | 98 |
| Mẫu suy biến (U≤2) | 2,4% | 0% | **21,4%** | 0,2% |
| Truncate ở 31.500 | 1/500 | 8/500 | 0 | **298 (17,2%)** |
| Gate | PASS | PASS | PASS | PASS |

## Ba chỗ lệch protocol còn lại — đều có số liệu chống lưng

| Lệch | Lý do đo được |
|---|---|
| **128K → 31.500** | Config của Qwen2.5-Coder-7B(-Instruct): `max_position_embeddings=32768`, `rope_scaling=None`. "128K" trên model card cần YaRN, mà fork transformers **không có nhánh scaling nào cho Qwen2**. Đã port Dynamic NTK (là phép đồng nhất dưới 32.768 — kiểm bằng so tensor), bật bằng cờ. Xem D7 |
| **Bỏ RepoEval** | Chỉ xuất hiện 1 lần trong protocol, không có trong grid Phase 6 lẫn bảng chính. RepoBench v1.1 phủ cùng mục đích với chi phí thấp hơn nhiều |
| **Offset trên prompt SAU truncation** | `truncate_fn` decode-lại-encode nên offset trên source gốc chắc chắn sai. Không có lựa chọn khác |

## Hai tính chất của dữ liệu phải nêu trong bài

**CrossCodeEval có rất ít cấu trúc AST.** 21,4% mẫu chỉ ≤2 đơn vị, và **0/9.928 mẫu chứa Unicode parse sạch** — chunk retrieve ra là đoạn cắt giữa hàm, không có `function_definition` trọn vẹn. Đề xuất 1 gần như không có ranh giới để ràng buộc trên 1/5 bộ này.

**RepoBench v1.1 mất 17,2% mẫu vì trần 31.500**, đúng những mẫu dài nhất. Muốn dùng đúng mục đích long-context phải chạy với `--rope_scaling dynamic:4`.

---

## Giới hạn phải ghi vào bài

**LCC và RepoBench-P không kiểm chứng được premise của SA.** Mỗi mẫu là một cặp
`(context, next_line)` độc lập — tỉ lệ khấu hao thực tế là **1 query / 1 lần clustering**,
không phải ~24 như PreFixQA của bài gốc.

| Claim | Bị ảnh hưởng? |
|---|---|
| C2 — recall@budget (Phase 5) | ❌ không — đo trên *một* query là đủ |
| C1 — accuracy@budget (Phase 6) | ❌ không — mọi method chịu cùng điều kiện |
| **C3 — chi phí khấu hao** | ✅ **có** — trên hai bộ này chi phí không được chia cho gì cả |

Chính bài gốc thừa nhận khoảng trống này (Section 5: *"there is currently no benchmark
designed to test this scenario"*) và tự dựng PreFixQA vì thế.

**Đã xử lý xong 23/8**: loader RepoBench v1.1 cho **1.735 fixed context dùng chung bởi 7.080
query**, trung vị 3 query/context. Đó là bộ duy nhất trong dự án đo được C3.

Nhưng phải nói cho đúng mức: **trung vị 3 query/context, không phải ~24 như PreFixQA**. Chi
phí clustering được khấu hao qua 3 truy vấn chứ không phải 24, nên mọi phát biểu về tiết kiệm
phải dựa trên con số 3 — không được mượn tỉ lệ của bài gốc.

Và với `maxlen=31500`, **17,2% mẫu của bộ đó bị cắt giữa**. Muốn đo C3 ở đúng độ dài mà bộ
này cung cấp (trung vị 19,8K, max 99,9K token) thì bắt buộc chạy `--rope_scaling dynamic:4`.
