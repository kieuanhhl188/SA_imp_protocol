# Phase 1.2 — Khảo sát CrossCodeEval / RepoBench

> ## ⚠️ ĐÍNH CHÍNH 19/8/2026 — kết luận cũ về RepoBench SAI
>
> Khảo sát 15/8 kết luận RepoBench "context quá ngắn" và "context không dùng chung", rồi
> khuyến nghị **không** dùng bộ này ở dạng nguyên bản. **Cả hai vế đều sai**, do hai lỗi
> phương pháp độc lập:
>
> **Lỗi 1 — đọc nửa dữ liệu.** Chỉ đọc `cross_file_first` **shard 0**, mà shard đó sắp theo
> `level` nên chứa toàn mẫu ngắn. Thực tế split này có **2 shard, 8.033 dòng**.
>
> **Lỗi 2 — nhóm sai khoá.** Nhóm theo `repo_name` rồi hỏi "mọi sample của repo có dùng chung
> MỘT context không". Quá chặt: một repo có thể có vài bộ context, mỗi bộ được nhiều query
> dùng chung. Phải nhóm theo `(repo_name, bộ context)`.
>
> ### Số đo lại, toàn bộ dữ liệu (19/8)
>
> | | Đo 15/8 (shard 0) | **Đo lại (đủ)** |
> |---|---:|---:|
> | Số dòng `cross_file_first` | 4.017 | **8.033** |
> | Nhóm `level` | chỉ tới `16k` | **`2k…16k, 24k, 32k, 64k, 128k`** |
> | `token_num` trung vị | 3.614 | **10.826** |
> | `token_num` max | 14.177 | **99.376** |
>
> Phân bố `level` đầy đủ của `cross_file_first`:
> `2k=1000 · 4k=1000 · 8k=1000 · 12k=1000 · 16k=1000 · 24k=1000 · 32k=1000 · 64k=912 · 128k=121`
>
> ### Context CÓ dùng chung — nhóm theo `(repo, bộ context)`
>
> | `cross_file_first` | |
> |---|---:|
> | Nhóm có ≥2 query | **1.308** |
> | Query trong các nhóm đó | **4.830 (60,1% toàn bộ)** |
> | **Nhóm vừa dài ≥16k vừa dùng chung** | **732 nhóm · 3.496 query** |
> | Context nhóm dài: trung vị · max | **17.886 · 99.376 token** |
> | Query/nhóm: trung vị · max | 3 · **30** |
>
> `cross_file_random` tương tự: 699 nhóm ≥16k, 3.096 query.
>
> ### Kết luận mới
>
> **RepoBench v1.1 dùng được ở dạng nguyên bản.** Không cần clone repo, không cần dựng
> benchmark mới, không cần pipeline LLM sinh câu hỏi. Chỉ cần loader gom theo
> `(repo_name, bộ context)` → mỗi nhóm là một `fixed_context` với nhiều `user_input`.
>
> Hai điều vẫn phải ghi vào bài:
> - **Trung vị 3 query/context** (PreFixQA của bài có ~24) → mức khấu hao chi phí clustering
>   thấp hơn nhiều, không được nói quá.
> - **Contamination**: `created_at` toàn 2023, Qwen2.5-Coder train tới ~2024.
>
> ---
>
> ## Đo lại đầy đủ 19/8 — cả ba bộ
>
> | | CrossCodeEval | RepoBench v1.1 | **RepoEval** |
> |---|---:|---:|---:|
> | Số fixed context | ~90 nhóm (Py) | **1.308** (732 nhóm ≥16k) | **16 repo** |
> | **Query / context** | ≤3 (Py) | trung vị 3, max 30 | **200** (line/api) · 46 (function) |
> | Độ dài context dựng được | 1,4K · max 8,6K tok | 10,8K · max 99,4K tok | **192K – 1,19M tok** |
> | Repo gốc kèm theo | ❌ xin qua email | ⚠️ clone lại được | ✅ **ship sẵn 43 MB** |
>
> ### CrossCodeEval — kết luận cũ ĐÚNG, nhưng vì lý do khác
>
> Đo lại cả 4 ngôn ngữ với khoá nhóm đúng `(repo, bộ context)`:
>
> | Ngôn ngữ | n | Context trung vị | Nhóm ≥2 query | Query/nhóm max | Khoá **cũ** (sai) |
> |---|---:|---:|---:|---:|---:|
> | Python | 2.665 | ~1.376 tok | 90 (7,1%) | 3 | 2/370 (0,5%) |
> | Java | 2.139 | ~1.667 tok | 143 (15,2%) | 14 | 3/204 (1,5%) |
> | C# | 1.768 | ~1.080 tok | 173 (24,5%) | 24 | 0/94 (0%) |
> | TypeScript | 3.356 | ~1.429 tok | 561 (37,8%) | 7 | 5/178 (2,8%) |
>
> Khoá đúng nâng tỉ lệ dùng chung lên đáng kể, nhưng **không cứu được** bộ này: lý do loại bỏ
> thật sự là **độ dài**. `crossfile_context` là top-5 chunk retrieve, p95 chỉ ~3.700 ký tự —
> bị chặn về mặt cấu trúc, không biến thể nào dài ra được.
>
> ### RepoEval — chưa từng đo ở khảo sát 15/8
>
> | Nhóm task | Repo | Query | Query/repo |
> |---|---:|---:|---:|
> | line + api level | 8 | 1.600 | **200** (đều nhau) |
> | function level | 8 | 455 | trung vị 46, max 146 |
>
> Hai bộ repo **rời nhau** → tổng **16 repo**.
>
> Kích thước repo (chỉ file `.py`, ước 3,5 ký tự/token):
>
> | line/api | ~token | | function | ~token |
> |---|---:|---|---|---:|
> | huggingface_diffusers | 1.187.670 | | leopard-ai_betty | 191.602 |
> | opendilab_ACE | 756.181 | | facebookresearch_omnivore | 119.853 |
> | pytorch_rl | 646.722 | | google_lightweight_mmm | 107.167 |
> | alibaba_FederatedScope | 524.393 | | CarperAI_trlx | 104.598 |
> | google_vizier | 447.130 | | deepmind_tracr | 96.011 |
> | nerfstudio-project_nerfstudio | 295.262 | | lucidrains_imagen-pytorch | 73.955 |
> | huggingface_evaluate | 242.383 | | maxhumber_redframes | 34.961 |
> | awslabs_fortuna | 192.725 | | amazon-science_patchcore-inspection | 26.185 |
>
> **Cả 8 repo line/api đều vượt 128K token.** Mọi repo function-level đều ≥26K.
>
> ⚠️ `prompt` trong file phân phối đã bị **cắt sẵn** theo ngân sách 1k/2k/4k của model năm
> 2023 (tên file ghi rõ). Dùng nguyên `prompt` là tự giới hạn ở 1-4K token. Phải dựng lại
> fixed context từ repo mới ra dải dài — và **cách ghép/cắt file trở thành một lựa chọn thiết
> kế phải ghi rõ trong bài**, vì nó ảnh hưởng thẳng tới kết quả.
>
> ### Khuyến nghị sau khi đo đủ
>
> **Đi cả hai đường, chúng bù nhau đúng chỗ yếu của nhau:**
>
> | | RepoBench v1.1 | RepoEval |
> |---|---|---|
> | Điểm mạnh | dùng ngay, 732 context dài, nhãn thật | **200 query/context**, repo tới 1,19M token |
> | Điểm yếu | **chỉ ~3 query/context** | phải tự dựng fixed context |
> | Công việc | viết loader gom theo `(repo, bộ context)` | ghép file repo + cắt tới độ dài đích |
>
> Cả hai dùng **nhãn thật**, không cần pipeline RepoPreFixQA (LLM sinh câu hỏi +
> self-consistency 5 lần + LLM-as-judge) mà protocol dự kiến.
>
> **Ba điều phải ghi vào bài:** contamination (repo 2022-2023, Qwen2.5-Coder train tới ~2024);
> RepoEval chỉ có 16 repo nên phải báo cáo theo từng repo chứ không chỉ một số trung bình; và
> cách dựng fixed context của RepoEval là lựa chọn thiết kế, không phải dữ liệu có sẵn.
>
> ### CrossCodeEval — đã đo đủ 9 biến thể × 4 ngôn ngữ (bản gốc GitHub)
>
> Độ dài context (ký tự, `prompt` + `crossfile_context`), Python:
>
> | Biến thể | trung vị | p95 | max |
> |---|---:|---:|---:|
> | `oracle_bm25` | 4.816 | 13.504 | 35.517 |
> | `oracle_openai` | 4.787 | 13.635 | 35.591 |
> | `oracle_unixcoder` | 4.853 | 13.636 | 35.477 |
> | `rg1_bm25` | 4.638 | 13.313 | 35.486 |
> | `rg1_openai` | 4.655 | 13.315 | 34.849 |
> | `rg1_unixcoder` | 4.660 | 13.466 | 35.570 |
>
> Sáu biến thể retrieval chênh nhau **dưới 5%**, trần max ~35,6K ký tự ≈ **10,2K token**.
> Đổi retriever hay đổi loại đều không làm context dài ra. **Kết luận loại bỏ vì độ dài là
> vững, đã kiểm trên toàn bộ biến thể chứ không phải suy đoán.**
>
> Chi tiết cần nêu nếu về sau ai dùng bộ này: **lựa chọn retriever ảnh hưởng mạnh tới tỉ lệ
> dùng chung** — Python `rg1`: bm25 5,3% vs unixcoder 1,9%; TypeScript `oracle`: bm25 37,8%
> vs unixcoder 18,6%. Không được nói chung chung "CrossCodeEval", phải ghi rõ biến thể.
>
> ### ⚠️ Một con số suýt thành "phát hiện" sai — ghi lại để không lặp
>
> Biến thể `line_completion.jsonl` báo tỉ lệ dùng chung **96-99,6%**:
>
> ```
> python/line_completion.jsonl        370 nhom  2564 query  96.2%
> typescript/line_completion.jsonl    178 nhom  3341 query  99.6%
> ```
>
> **Đây là rác.** Biến thể này **không có** `crossfile_context`, nên hàm trích trả về chuỗi
> rỗng cho mọi dòng và khoá nhóm `(repo, hash(""))` gom hết query cùng repo vào một nhóm.
> Con số đó chỉ đo "có bao nhiêu repo nhiều mẫu", không đo chia sẻ context.
>
> Cùng họ với lỗi nhóm sai khoá ở RepoBench: **khoá nhóm không nói lên điều mình tưởng nó
> nói.** Lần này bắt được vì con số đẹp bất thường. Quy tắc rút ra: khi một tỉ lệ chia sẻ
> vọt lên gần 100%, kiểm xem trường dùng làm khoá có rỗng không **trước khi** mừng.
>
> ### RepoBench Java — đo xong, TỐT HƠN Python
>
> | `cross_file_first` | Python | **Java** |
> |---|---:|---:|
> | Số dòng | 8.033 | **8.722** |
> | Mức `128k` | 121 | **722** |
> | Mức `64k` | 912 | **1.000** |
> | `token_num` trung vị · p95 · max | 10.826 · 38.719 · 99.376 | 11.840 · **69.581** · 99.951 |
> | **Nhóm ≥16k và dùng chung** | 732 nhóm · 3.496 query | **914 nhóm · 3.352 query** |
> | Query/nhóm max | 30 | **50** |
>
> Java có **gấp 6 lần** số mẫu mức `128k` và p95 độ dài cao gần gấp đôi. Ba split nhất quán:
> `cross_file_first` 914 nhóm · `cross_file_random` 912 · `in_file` 896.
>
> Gộp Python + Java, riêng `cross_file_first`: **1.646 fixed context ≥16k, 6.848 query.**
>
> Giả định "Java chỉ làm tăng số lượng" **sai** — nó dịch hẳn phân bố độ dài lên trên. Nếu
> chỉ chạy một ngôn ngữ cho thí nghiệm long-context thì nên chọn **Java**, không phải Python.
>
> ### Đã đo hết
>
> Không còn khoảng trống nào trong ba bộ dữ liệu.
>
> ---
>
> Phần bên dưới giữ nguyên làm hồ sơ những gì đã kết luận ngày 15/8. **Mọi con số về độ dài
> và tỉ lệ dùng chung của RepoBench trong đó đều đã bị thay thế bởi bảng trên.** Phần
> CrossCodeEval chưa đo lại — có thể mắc cùng hai lỗi này.


Khảo sát thực hiện 15/8/2026 trên dữ liệu tải thật, không phải theo mô tả trong paper.

- CrossCodeEval: `git clone --depth 1 https://github.com/amazon-science/cceval` →
  `data/crosscodeeval_data.tar.xz` (81 MB nén, 888 MB sau giải nén)
- RepoBench v1.1: `huggingface.co/datasets/tianyang/repobench_python_v1.1`, split
  `cross_file_first` shard 0 (31 MB parquet, 4017 dòng)

---

## 0. Protocol nói gì về hai bộ này

Toàn bộ, nguyên văn — 4 chỗ, đều **ngoài** thẻ `<chưa cần làm bây giờ>`:

**Phase 1:**
> **Dataset dùng ngay (không cần tự build)**
> LongBench RepoBench-P (RB) và LCC — có sẵn trong pipeline SA, cho phép so trực tiếp với số của bài.
> **CrossCodeEval và RepoEval / RepoBench — completion cấp repository, có cross-file context (đúng fixed context dài).**

**Phase 6:**
> Chạy full grid: {models} × {datasets: RB, LCC, **CrossCodeEval**, RepoPreFixQA} × {methods} × {sparsity 70/80/90%}.

**Bảng kết quả tối thiểu:**
> Accuracy@matched-budget: All-KV / QUEST / ClusterKV / SA / Ours, trên RB+LCC+**CrossCodeEval**+RepoPreFixQA

Không có gì thêm. Protocol **không** chỉ định: dùng biến thể nào (`rg1` hay `oracle`,
retriever nào), ngôn ngữ nào, metric nào, số sample, hay cách dựng split fixed/user.

### Ba nhận xét khi đối chiếu với dữ liệu thật

1. **Cả hai mệnh đề mô tả đều không đúng.** *"không cần tự build"* → thực tế phải build,
   vì context không dùng chung giữa các query. *"đúng fixed context dài"* → thực tế trung vị
   1.4–3.6K token. Đây là dòng duy nhất protocol mô tả hai bộ.

2. **RepoEval và RepoBench chỉ xuất hiện đúng một lần**, rồi biến mất — không có trong grid
   Phase 6, không có trong bảng chính. Thực chất protocol chỉ cam kết dùng **CrossCodeEval**;
   RepoEval/RepoBench là gợi ý bỏ ngỏ.

3. **Mâu thuẫn nội tại trong protocol.** Bảng chính và grid Phase 6 (đều ngoài thẻ) cần
   **RepoPreFixQA**, nhưng việc dựng RepoPreFixQA lại nằm **trong** thẻ
   `<chưa cần làm bây giờ>`. Tức bảng chứng minh C1 phụ thuộc một bộ dữ liệu mà protocol nói
   chưa cần làm. Sớm muộn cũng phải mở thẻ đó, không thì bảng chính thiếu một cột.

---

## Kết luận ngắn

Protocol Phase 1 viết: *"CrossCodeEval và RepoEval / RepoBench — completion cấp repository,
có cross-file context (đúng fixed context dài)"*.

**Cả hai bộ, ở dạng phân phối sẵn, đều KHÔNG thoả điều đó.** Hai lý do, lý do thứ hai
nghiêm trọng hơn nhiều:

1. Context ngắn — trung vị 1.4–3.6K token, xa mức 20–150K mà protocol giả định.
2. **Context là kết quả retrieval theo từng query, không dùng chung giữa các query cùng repo.**
   Điều này phá thẳng vào premise của Squeezed Attention.

---

## 1. CrossCodeEval

### Schema (`<lang>/line_completion_oracle_bm25.jsonl`)

| Trường | Kiểu | Nội dung |
|---|---|---|
| `prompt` | str | code trong file, phần trước con trỏ |
| `groundtruth` | str | dòng cần hoàn thành |
| `right_context` | str | code sau con trỏ |
| `metadata` | dict | `task_id`, `repository`, `file`, `context_start_lineno`, `groundtruth_start_lineno`, `right_context_start_lineno` |
| `crossfile_context` | dict | `text` (đã format thành comment), `list` = `[{retrieved_chunk, filename, score}]` |

Ba biến thể × 3 retriever: `line_completion` (không cross-file), `_rg1_*` (retrieval),
`_oracle_*` (retrieval có tham chiếu gold).

### Quy mô

| Ngôn ngữ | Số sample |
|---|---|
| Python | 2665 |
| Java | 2139 |
| TypeScript | 3356 |
| C# | 1768 |

Python: **471 repo**, trung bình 5.7 sample/repo, nhiều nhất 69 sample.

### Độ dài (ký tự, 800 sample Python đầu)

| File | prompt | crossfile_context | tổng |
|---|---|---|---|
| `line_completion` | 288 / 2814 / 24219 | 0 | 288 / 2814 / 24219 |
| `oracle_bm25` | 288 / 2814 / 24219 | 1808 / 2837 / 9494 | 2367 / **5652** / 27403 |
| `rg1_bm25` | 288 / 2814 / 24219 | 750 / 2581 / 4366 | 1780 / **5396** / 27367 |

(min / trung bình / max). Trung bình ~5.6K ký tự ≈ **1.4K token**. Lớn nhất ~27K ký tự ≈ 7K token.

`crossfile_context` chỉ là **top-5 chunk đã retrieve**, không phải toàn bộ repo.

### Context KHÔNG dùng chung giữa các query

Trong 1500 sample Python đầu:
- 211 repo có >1 sample
- **chỉ 2 repo** có mọi sample dùng chung một `crossfile_context`
- repo `hq0709-Depth-NeuS-49d93d4`: 42 sample → **42 context khác nhau**

### Repo gốc không kèm theo

README của bộ dữ liệu: *"Please reach out to us via email if you need the raw data for the
repos used to create CrossCodeEval."*

---

## 2. RepoBench v1.1 (Python, `cross_file_first`, shard 0)

### Schema

| Trường | Nội dung |
|---|---|
| `repo_name` | dạng `owner/repo` — **sạch, dùng để clone lại được ngay** |
| `file_path` | đường dẫn file trong repo |
| `context` | list `{identifier, path, snippet}` — snippet đã retrieve, trung bình 3 cái |
| `import_statement` | các import của file hiện tại |
| `token_num` | số token của context |
| `cropped_code` / `all_code` | code trong file trước con trỏ |
| `next_line` | dòng cần hoàn thành |
| `gold_snippet_index` | snippet nào chứa đáp án |
| `created_at` | thời điểm tạo repo |
| `level` | nhóm độ dài: `2k`/`4k`/`8k`/`12k`/`16k` |

### Quy mô và độ dài

- 4017 dòng, **1722 repo**, trung bình 2.3 sample/repo, nhiều nhất 6
- `token_num`: min 641, trung vị **3614**, max **14177**
- Phân bố `level`: 2k=1000, 4k=1000, 8k=1000, 12k=1000, 16k=17

### Context cũng không dùng chung

962 repo có >1 sample, chỉ **153 repo** dùng chung một bộ context.

### Contamination

`created_at` của **toàn bộ 4017 dòng đều là năm 2023**. Qwen2.5-Coder huấn luyện trên dữ
liệu tới khoảng 2024 → gần như chắc chắn đã thấy các repo này.

---

## 3. Vì sao điểm 2 là vấn đề nghiêm trọng

Premise của Squeezed Attention: *một fixed context, nhiều user query*, nhờ đó chi phí
clustering được khấu hao qua nhiều query. Nếu mỗi query có context riêng thì phải chạy lại
clustering cho từng query — toàn bộ giá trị của phương pháp biến mất.

Chính bài gốc đã thừa nhận khoảng trống này (Section 5): *"there is currently no benchmark
designed to test this scenario... Recent long context benchmarks do not evaluate the handling
of multiple queries on the same document"* — và đó là lý do họ tự dựng PreFixQA.

Nói cách khác: **CrossCodeEval và RepoBench không giải quyết được vấn đề mà protocol nghĩ
là chúng giải quyết.** Chúng có cùng hạn chế như LongBench LCC/RepoBench-P đang dùng.

---

## 4. Đường đi khả thi: dựng lại fixed context từ repo gốc

Cả hai bộ đều ghi lại repo nguồn. Clone repo về, lấy **toàn bộ repo làm fixed context**, và
tái dùng chính các nhãn có sẵn (`groundtruth` / `next_line` + vị trí con trỏ) làm query.

Kết quả sẽ đạt cả hai yêu cầu của protocol cùng lúc: fixed context 20–150K token, **và**
nhiều query trên cùng một fixed context.

**Kiểm tra tính khả thi (15/8/2026):** 15/15 repo RepoBench nhiều sample nhất còn truy cập
được trên GitHub, kích thước 0.1–32 MB.

| | CrossCodeEval | RepoBench v1.1 |
|---|---|---|
| Định danh repo | `turboderp-exllama-a544085` — owner/repo/hash dính liền, **khó tách** vì tên có dấu gạch | `DLYuanGod/TinyGPT-V` — **sạch** |
| Commit cụ thể | có (hash ngắn trong tên) | không, chỉ có `created_at` |
| Sample/repo | tb 5.7, max **69** | tb 2.3, max 6 |
| Số repo (Python) | 471 | 1722 |

→ **CrossCodeEval tốt hơn cho mục đích này** vì mật độ query/repo cao gấp 2.5 lần và có
commit hash để tái lập chính xác. Đổi lại phải xử lý việc tách `owner-repo-hash`, có thể
dùng GitHub search API.

So với pipeline RepoPreFixQA của protocol (sinh câu hỏi bằng LLM + self-consistency 5 lần +
LLM-as-judge), cách này **rẻ hơn nhiều** vì nhãn đã có sẵn và là nhãn thật, không phải nhãn máy sinh.

---

## 5. Khuyến nghị

1. **Không viết loader cho hai bộ ở dạng nguyên bản.** Context quá ngắn và không dùng chung
   → không đo được cái mà protocol muốn đo.
2. **Gộp mục 1.2 với RepoPreFixQA (⏸️).** Chúng thực chất là cùng một việc: dựng benchmark
   fixed-context-dài. Dùng CrossCodeEval làm nguồn repo + nhãn thay vì sinh mới từ đầu.
3. **Vẫn giữ contamination caveat.** Repo đều từ 2023. So sánh tương đối giữa các method ở
   cùng budget vẫn hợp lệ (mọi method chạy trên cùng model bị nhiễm), nhưng số tuyệt đối
   phải ghi rõ trong paper.
4. **Không nằm trên đường tới hạn 22/8.** Phase 2 và Phase 5 chạy trọn vẹn với LongBench
   LCC + RepoBench-P.
