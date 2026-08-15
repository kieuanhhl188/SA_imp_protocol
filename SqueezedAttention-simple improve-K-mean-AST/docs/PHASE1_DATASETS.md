# Phase 1.2 — Khảo sát CrossCodeEval / RepoBench

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
