# Nhật ký thí nghiệm — Structure-Aware Squeezed Attention

File này là **nguồn sự thật duy nhất** cho tiến độ và kết quả. Mỗi lần chạy gate,
`scripts/check_gate.py --log_md` tự phụ lục một mục vào cuối file (mục "Lịch sử chạy").
Ghi chú tay thì viết ngay dưới mục tương ứng, chỗ có comment `<!-- ghi chú tay bên dưới -->`.

Mốc tham chiếu: Table 2, Hooper et al., ACL 2025 (`2025.acl-long.1568`).
Tolerance ±0.3 điểm theo protocol.

---

## 1. Kế hoạch theo phase

Ký hiệu: ✅ xong · 🟡 một phần · ❌ chưa làm · ⏸️ hoãn (protocol đánh dấu *chưa cần làm bây giờ*)

### Tổng quan

| Phase | Nội dung | Hạn | Tiến độ |
|---|---|---|---|
| 0 | Môi trường + tái lập baseline SA | — | 🟡 code xong, chưa chạy |
| 1 | Chuẩn bị dữ liệu code | — | 🟡 5/6 mục, chờ chạy trên pod |
| 2 | Structure-aware clustering (Idea 1) | **22/8** | 🟡 6/6 có code, chưa chạy GPU |
| 3 | Symbol / def-use signal (Idea 2) | **30/8** | ❌ 0/4 |
| 4 | Incremental re-clustering (Idea 3) | **8/9** | ❌ 0/4 |
| 5 | C2 retrieval quality — chạy TRƯỚC Phase 6 | — | ❌ 0/5 |
| 6 | C1 accuracy@budget end-task | — | ❌ 0/5 |
| 7 | C3 + phân tích | — | ❌ 0/4 |

Thứ tự chạy theo protocol: `Phase 0 → Phase 1 → Phase 5 (C2, quyết định H0, dừng sớm nếu fail) → Phase 6 (C1) + Phase 4/7 (C3)`.
Phase 2 và 3 là phần *cài đặt* mà Phase 5/6 sẽ đo.

---

### Phase 0 — Môi trường + tái lập baseline SA · 🟡

Mục tiêu: dựng lại đúng pipeline SA để mọi cải tiến là ablation trên cùng một nền.

| # | Việc | Trạng thái | Chi tiết |
|---|---|---|---|
| 0.1 | Clone repo gốc, dựng env | ✅ | Repo + fork transformers 4.40.0.dev0 có sẵn. `requirements.txt` đã viết đủ (cuml, cupy, triton, flash-attn) |
| 0.2 | Chốt config mặc định (5%, 1%/5%, obs 100, 32K) | ✅ | [configs/phase0.sh](configs/phase0.sh) |
| 0.3 | Ghi version transformers/triton, GPU, seed | ✅ | [scripts/record_env.py](scripts/record_env.py), tự kiểm tra transformers có đúng fork |
| 0.4 | Script chạy gate LCC/RB | ✅ | [scripts/phase0_gate.sh](scripts/phase0_gate.sh) + [scripts/check_gate.py](scripts/check_gate.py), tolerance ±0.3 |
| 0.5 | Số đích từ Table 2 | ✅ | [scripts/reference_table2.json](scripts/reference_table2.json), xem mục 2 |
| 0.6 | Sửa bug chặn gate | ✅ | 5 bug, xem mục 6 |
| 0.7 | **Cài đặt thật trên pod** | ❌ | Chờ thuê A100 80GB SXM + volume 200GB. Script: [scripts/setup_pod.sh](scripts/setup_pod.sh) |
| 0.8 | **Chạy gate, khớp Table 2 ±0.3** | ❌ | Việc duy nhất còn lại của Phase 0 |

**Còn lại:**
1. Thuê pod theo cấu hình mục 5, cài theo [docs/PHASE0.md](docs/PHASE0.md).
2. Chạy thử **3 sample** trước để đo giây/sample và MB/sample — hai con số chưa biết, cần để ước lượng job full và dung lượng đĩa.
3. `bash scripts/phase0_gate.sh`. Kết quả tự ghi vào mục "Lịch sử chạy".

**Đã kiểm, không phải lo:** 8 tham số SA đều có default trong `configuration_llama.py` (nên `pred.py` không set `return_qkv_states` vẫn chạy); `reset_context = hidden_states.shape[1] > 1` nên centroid nạp lại đúng mỗi sample; `rope_scaling` linear factor 8 của LongChat được fork chấp nhận.

---

### Phase 1 — Chuẩn bị dữ liệu code · 🟡

Protocol ghi "đã xong" nhưng thực tế mới có phần LongBench. Cần đúng cấu trúc *một fixed context → nhiều user query* thì premise của SA mới áp dụng.

| # | Việc | Trạng thái | Chi tiết |
|---|---|---|---|
| 1.1 | LongBench LCC + RepoBench-P | ✅ | Có sẵn trong pipeline, metric `code_sim_score`, so trực tiếp được với Table 2 |
| 1.2 | CrossCodeEval + RepoEval/RepoBench | ⚠️ | **Đã khảo sát dữ liệu thật — giả định của protocol KHÔNG đúng.** Context của cả hai bộ là retrieval theo từng query, không dùng chung, và chỉ dài 1.4–3.6K token. Xem [docs/PHASE1_DATASETS.md](docs/PHASE1_DATASETS.md). Khuyến nghị gộp với RepoPreFixQA thay vì viết loader |
| 1.3 | Chuẩn hoá split `fixed_context` / `user_input` | 🟡 | Cơ chế có (`{dataset}_prompt` + `truncate_fn`) nhưng **lệch định nghĩa protocol** — xem quyết định D2 mục 7 |
| 1.4 | Lưu offset ký tự từng token | 🟡 | **Code xong**, chưa chạy thật (cần mạng tải dataset + tokenizer). [scripts/prepare_code_data.py](scripts/prepare_code_data.py) — offset tính trên prompt **cuối cùng sau truncation**, không phải source gốc. Self-test 12/12 pass |
| 1.5 | Model chính Qwen2.5-Coder-7B-Instruct (128K) | 🟡 | **Code xong**, chưa chạy thật. Port SA sang `models/qwen2/` (config + attention forward + `Qwen2Model.forward`), mở `pred.py` và `offline_clustering.py` cho model ngoài Llama, thêm entry vào `model2path`/`model2maxlen` |
| 1.6 | GQA: chọn key per-head, khớp cấu hình QUEST (Appendix G) | 🟡 | **Code xong**. Appendix G: mỗi query head **tự chọn key riêng**. Cài bằng `repeat_interleave` centroid/label từ 4 head KV lên 28 head Q. `run_global_threshold` cũng có nhánh GQA, no-op khi MHA. Test 20/20 pass ([scripts/test_gqa_port.py](scripts/test_gqa_port.py)) |
| — | ⏸️ RepoPreFixQA (đóng góp benchmark, làm song song) | ⏸️ | 30-50 repo Python/Java từ GitHub 2025+ (tránh contamination), mỗi repo cắt fixed context 20-150K token; sinh query theo pipeline PreFixQA (Section E), filter self-consistency 5 lần + LLM-as-judge, tái dùng prompt Appendix E.3. **Thực chất trùng việc với 1.2** — xem quyết định D4 |
| — | ⏸️ Model cross-check | ⏸️ | DeepSeek-Coder-V2-Lite hoặc CodeLlama-13B |

**Còn lại, theo thứ tự:** chạy 1.4 trên pod → 1.5 + 1.6 (chặn Phase 6, và chặn Phase 5 nếu muốn chạy trên code model, phụ thuộc quyết định D1) → 1.2.

**Ghi chú 1.4 — ba điều Phase 2 phải biết:**
1. Offset tính trên **prompt cuối cùng sau truncation**. Phase 2 vì vậy cũng phải parse AST trên chuỗi đó, không phải trên file source gốc.
2. Sample bị truncate mất phần giữa → code **không còn đúng cú pháp**, tree-sitter sẽ sinh node `ERROR` quanh chỗ nối. Trường `truncated` trong meta đánh dấu sẵn để Phase 2 quyết định bỏ qua hay chấp nhận.
3. `offline_clustering.py` dùng tokenizer **chậm** (không hỗ trợ `offset_mapping`), script này dùng tokenizer **nhanh** để lấy offset rồi **assert hai bên ra cùng token id**. Lệch là báo lỗi và exit 1, thay vì lặng lẽ sinh dữ liệu sai — vì offset lệch nghĩa là `unit_id` gán sai key vector.

---

### Phase 2 — Structure-aware clustering (Idea 1) · 🟡 · **hạn 22/8**

Ý tưởng: đặt ranh giới **cứng** theo AST, cluster embedding bên trong mỗi đơn vị cấu trúc. Hierarchy = token → statement/block → function → file.

| # | Việc | Trạng thái | Chi tiết |
|---|---|---|---|
| 2.1 | Parse AST bằng tree-sitter, có byte offset | ✅ | `parse_units` — 5 level (`file`/`class`/`function`/`block`/`statement`), 5 ngôn ngữ. Dùng API tree_sitter mới, **không cần** `tree_sitter_languages` (gói đó không cài được). Level thô gộp vào level mịn nên mọi token đều có unit bao |
| 2.2 | Gán `unit_id` cho từng key token ở từng level | ✅ | `assign_token_units` — sắp span theo kích thước giảm dần rồi ghi đè bằng `searchsorted`, O(U log S) thay cho O(S×U) của bản cũ. Offset lấy từ Phase 1.4 nên không đụng lỗi `use_fast=False` |
| 2.3 | **Hard boundary** — K-means độc lập trong từng unit, unit nhỏ → 1 centroid, tổng K vẫn ~5% | 🟡 | **Code xong + đã nối pipeline** (`--method hard_boundary`). Test bất biến: không cluster nào vắt qua hai unit. Chờ chạy GPU |
| 2.4 | **StructHierarchy** — L2 = trong-function, L1 = trung bình theo function/file | 🟡 | **Code xong + đã nối pipeline** (`--method struct_hierarchy`). `build_l1_groups` ép K1 về đúng 1% context, ghi K1 thực tế ra `k1_stats_*.pt`. `weighted=True` mặc định: L1 centroid đúng bằng trung bình toàn bộ key của nhóm |
| 2.5 | Ablation tách bạch: SA / +HardBoundary / +StructHierarchy | ✅ | `offline_clustering_struct.py --method {sa,hard_boundary,struct_hierarchy}`. Nhánh `sa` gọi thẳng `run_clustering` gốc để mọi nhánh đi qua cùng một đường code |
| 2.6 | Giữ nguyên Si, threshold, kernel | ✅ | `struct_clustering.py` **chỉ** sinh centroid + label, cùng layout `[1,H,K,D]`/`[1,H,S]` với `run_clustering`. Token-type weighting (Hướng 2(b) sẵn có trong repo) giữ lại làm cờ `token_weights`, **mặc định tắt**; test xác nhận tắt cờ ra kết quả trùng bit-for-bit |

**Vấn đề kỹ thuật phải xử lý khi viết lại:**
- Hiệu năng: `weighted_kmeans` lặp Python `for h in range(H): for k in range(K)` — với H=32, K≈1500 là ~48K vòng/layer × 32 layer × 10 iter. Thực tế treo máy. Baseline dùng cuML KMeans trên GPU. Thay bằng `torch.searchsorted` cho mapping và `scatter_add_` cho centroid update.
- Shape: `run_clustering` trả `[1,H,K,D]` còn `weighted_kmeans` trả `[H,K,D]`. Online eval đọc `key_centroids.shape[2]` làm K → với `[H,K,D]` sẽ lấy nhầm `D`.
- Output của `offline_clustering_v2.py` lưu tên `..._{avg_k}.pt` nhưng online tính lại `num_clusters` từ `percent_clusters` → chưa bao giờ load được. [docs/PATCHING_EVAL.md](docs/PATCHING_EVAL.md) mô tả patch cần làm nhưng **chưa apply**.

**Khuyến nghị:** viết module mới `struct_clustering.py` thay vì sửa `ast_clustering.py`. Phần dùng lại được chỉ có `parse_code_to_scopes`.

---

### Phase 3 — Symbol / def-use làm tín hiệu retrieval (Idea 2) · ❌ · **hạn 30/8**

| # | Việc | Trạng thái | Chi tiết |
|---|---|---|---|
| 3.1 | Symbol index offline: `identifier -> [token positions]` | ❌ | v1 chỉ cần string matching identifier (rẻ, đủ mạnh) |
| 3.2 | v2 scope-aware | ❌ | tree-sitter query hoặc Jedi cho def-use chính xác |
| 3.3 | Hợp nhất với Si: `S_final = Si + λ · symbol_hit(cluster, query_identifiers)` | ❌ | λ calibrate như threshold (Appendix C) |
| 3.4 | Ablation `+SymbolSignal` bật/tắt, quét λ ∈ {0, 0.5, 1, 2} | ❌ | |

Phụ thuộc: 1.4 (byte offset) và 2.2 (unit mapping).

---

### Phase 4 — Incremental re-clustering (Idea 3) · ❌ · **hạn 8/9**

| # | Việc | Trạng thái | Chi tiết |
|---|---|---|---|
| 4.1 | Lưu state `unit_id -> {centroids, key_ids, value_stats}` | ❌ | |
| 4.2 | Mô phỏng edit bằng diff thật | ❌ | Lấy commit kế tiếp của repo, hoặc sinh edit tổng hợp (thêm/xoá/sửa 1 function) |
| 4.3 | Xác định `changed_units`, chỉ re-cluster unit đó | ❌ | Dùng ranh giới của Phase 2 |
| 4.4 | Đo `t_full`, `t_incr`, `Δaccuracy` | ❌ | Baseline `t_full` theo bài ~23-24 phút/128K |

Phụ thuộc: 2.3 (hard boundary định nghĩa ra unit).

---

### Phase 5 — C2 retrieval quality · ❌ · **chạy TRƯỚC Phase 6**

Bằng chứng trực tiếp và rẻ nhất cho H0. Dựa trên baseline "Ideal" của bài (Appendix H). **Nếu C2 fail thì H0 sai → dừng, không chạy C1/C3.**

| # | Việc | Trạng thái | Chi tiết |
|---|---|---|---|
| 5.1 | Tính full attention query→toàn bộ fixed key, lấy top-p theo threshold làm tập ideal `K*` | ❌ | |
| 5.2 | Với mỗi method lấy tập `K_m` ở cùng budget | ❌ | SA, +HardBoundary, +StructHierarchy, +SymbolSignal |
| 5.3 | Metric chính: `Recall@budget` = số key trong `K_m ∩ K*` chia cho số key trong `K*` | ❌ | Phụ: precision, attention-mass recovered (tổng attention weight của `K_m` trên `K*`) |
| 5.4 | Quét budget ∈ {70, 80, 90%}, vẽ Recall vs budget | ❌ | |
| 5.5 | Paired test qua các mẫu | ❌ | Pass nếu structure-aware recall cao hơn SA có ý nghĩa thống kê ở **≥2 mức budget** |

---

### Phase 6 — C1 accuracy@budget end-task · ❌

| # | Việc | Trạng thái | Chi tiết |
|---|---|---|---|
| 6.1 | Full grid `{models} × {RB, LCC, CrossCodeEval, RepoPreFixQA} × {methods} × {70/80/90%}` | ❌ | |
| 6.2 | **Matched-budget bắt buộc** | ❌ | Đặt budget từng method về cùng % KV cache loaded, set token budget động theo từng mẫu. Báo cáo budget **đo thực tế**, không phải danh nghĩa |
| 6.3 | Baseline so sánh | ❌ | All-KV (trần), SA, **QUEST**, **ClusterKV** — hai cái sau chưa có trong repo, phải tích hợp |
| 6.4 | Metric theo task | ❌ | Completion → Exact Match + Edit Similarity; pass@1 nếu có test; QA → F1 |
| 6.5 | Ablation cộng dồn SA → +HardBoundary → +StructHierarchy → +SymbolSignal | ❌ | |

---

### Phase 7 — C3 + phân tích · ❌

| # | Việc | Trạng thái | Chi tiết |
|---|---|---|---|
| 7.1 | Bảng `t_full` vs `t_incr` theo kích thước diff (1 function / 1 file / 10% repo) + `Δaccuracy` | ❌ | Claim: speedup nhiều lần, Δaccuracy < 0.3 điểm |
| 7.2 | Latency inference | ❌ | Tái dùng harness kernel của bài (`triton.testing.do_bench`), đo prefill (1K/4K) + decode, so FlashAttention/FlashDecoding. `squeezedattention/kernels.py` có sẵn, harness benchmark thì chưa |
| 7.3 | Sensitivity granularity | ❌ | statement vs block vs function làm ranh giới cứng. Vẽ accuracy/recall theo level. Thường là hình "đắt" nhất của paper |
| 7.4 | Skewness theo construct | ❌ | Phỏng theo Appendix B.2 — attention có skew khác nhau theo loại node (delimiter, identifier, comment)? Nếu có, củng cố lập luận threshold nên nhận biết cấu trúc |

---

### Bảng/hình tối thiểu cho paper

| Bảng/Hình | Nội dung | Chứng minh | Từ phase |
|---|---|---|---|
| Bảng chính | Accuracy@matched-budget: All-KV / QUEST / ClusterKV / SA / Ours trên RB+LCC+CrossCodeEval+RepoPreFixQA | C1 | 6 |
| Hình Recall | Recall@budget vs sparsity, Ours vs SA | C2 (H0) | 5 |
| Bảng ablation | +HardBoundary → +Hierarchy → +Symbol | phân rã đóng góp | 2, 3, 6 |
| Bảng incremental | `t_full` vs `t_incr` vs `Δaccuracy` theo diff size | C3 | 4, 7 |
| Hình granularity | accuracy/recall theo level cấu trúc | sensitivity | 7 |

### Ghi chú kiểm soát (áp dụng xuyên suốt)

- [ ] Mọi so sánh ở cùng budget **đo thực tế**, kể cả overhead metadata (centroid) như bài tính.
- [ ] GQA: nêu rõ chiến lược chọn key per-head, khớp cấu hình QUEST khi so.
- [ ] Repo dữ liệu sau mốc train của model để tránh contamination — ghi rõ nguồn/năm.
- [ ] Cố định seed, báo cáo **mean ± std qua ≥3 seed** cho các con số accuracy chính (điểm reviewer TDSC/KSE hay soi). `pred.py` đã có `--seed`.
- [ ] Nêu rõ clustering là chi phí offline một lần, **không** nằm trong inference latency.

---

## 2. Mốc cần khớp — Phase 0 gate

LongChat-7B-v1.5-32K, hai task code của LongBench.

| Config | Budget | LCC | RepoBench-P |
|---|---|---:|---:|
| All KV | 1.000 | 56.64 | 53.20 |
| **Sq-70%** | 0.325 | **56.93** | **54.64** |
| Sq-80% | 0.225 | 57.17 | 52.83 |
| Sq-90% | 0.125 | 56.95 | 51.57 |
| H-Sq-90% | 0.122 | 57.20 | 51.89 |

⚠️ Không dùng số trong `LongBench/README.md` (LCC 53.0 / RB 55.3) — đó là All-KV của
LongBench gốc với prompt/truncation khác, so vào sẽ FAIL oan.

---

## 3. Dữ liệu được ghi ở đâu

| Loại | Đường dẫn | Sinh bởi |
|---|---|---|
| Centroid / label / threshold | `fixed-prompt-clusters/<dataset>/*.pt` | `offline_clustering.py` |
| Prediction thô | `LongBench/pred/<config>/<dataset>.jsonl` | `LongBench/pred.py` |
| Điểm số | `LongBench/pred/<config>/result.json` | `LongBench/eval.py` |
| Offset token (Phase 1.4) | `phase1_data/<dataset>_meta.jsonl` + `_offsets.npz` | `scripts/prepare_code_data.py` |
| Môi trường | `phase0_results/env_record.json`, `_pip_freeze.txt` | `scripts/record_env.py` |
| Console log đầy đủ | `phase0_results/logs/<timestamp>_phase0_gate.log` | `scripts/phase0_gate.sh` |
| **Tổng hợp (file này)** | `EXPERIMENT_LOG.md` | `scripts/check_gate.py --log_md` |

Quy ước tên thư mục `<config>` do `eval.py` sinh:
- All-KV → `<model>_baseline`
- Single-level → `<model>_PC<percent>_PERC<percentile>`
- Hierarchical → `<model>_PC1_<pc>_PERC1_<perc>_PC2_<pc2>_PERC2_<perc_lower>_lookup`

⚠️ Ước tính dung lượng: centroid lưu fp32, label lưu int64 → khoảng **1.1 GB/sample** ở
context 31.5K. Cả LCC + RepoBench-P (500 sample mỗi task) ước tính **150-250 GB**,
nhân đôi nếu chạy `--full` (hierarchical lưu thêm bộ L1/L2). Kiểm tra dung lượng đĩa
trước khi chạy full.

---

## 4. Cấu hình đã chốt

Nguồn duy nhất: [configs/phase0.sh](configs/phase0.sh). Mọi phase sau phải `source`, không hard-code lại.

| Tham số | Giá trị | Nguồn |
|---|---|---|
| centroid single-level | 5% fixed context | Section 6.1 |
| hierarchical L1 / L2 | 1% / 5% | Section 6.1 |
| observation window | 100 token cuối, giữ nguyên | Appendix C |
| ngưỡng L1 hierarchical | loại 50% key → `--percentile_lower 0.5` | Section 6.1 |
| max context | 32K (`model2maxlen` = 31500) | Appendix F |
| seed | 42 | — |

Ánh xạ sparsity: Sq-70% ⇒ `--percentile 0.7`, Sq-80% ⇒ `0.8`, Sq-90% ⇒ `0.9`.
Danh sách quantile khả dụng `[0.5, 0.7, 0.8, 0.9]` hard-code ở
[squeezedattention/clustering.py:173](squeezedattention/clustering.py#L173) — muốn mức khác
phải sửa `qlist` **và** chạy lại offline clustering.

---

## 5. Hạ tầng đã chốt

**GPU: RunPod A100 80GB SXM, On-Demand.** Chốt ngày 2026-08-15.

*(Ban đầu chốt PCIe $1.39/h; đổi sang SXM $1.59/h cùng ngày — lý do ở dưới.)*

| Hạng mục | Giá trị | Lý do |
|---|---|---|
| GPU | A100 80GB **SXM** | Chênh 14% giá nhưng nhanh hơn ~5-15% (2039 vs 1935 GB/s, TDP 400W vs 300W) → **tổng chi phí gần như hoà**. Quyết định thật nằm ở trần **7 GPU/pod** so với 2 của PCIe |
| Loại thuê | **On-Demand** | Spot/Interruptible bị kill giữa chừng, job clustering nhiều giờ sẽ mất trắng |
| Số GPU khởi đầu | **1** | Smoke test + Phase 0 chỉ cần 1. Trần 7 là quyền chọn cho sau, không thuê 7 ngay |
| Network volume | **200 GB**, mount `/workspace` | **Không hoãn được tới Phase 3** — gate Phase 0 đầy đủ đã cần ~178 GB. Xem mục 3 |
| CUDA / PyTorch | template **CUDA 12.x + torch 2.3-2.4** nếu có | ⚠️ Tránh torch 2.8 / CUDA 13: fork là transformers 4.40.0.dev0 (3/2024), khoảng cách API quá xa. Nếu buộc dùng thì kiểm kỹ bước [2]/[3] của `setup_pod.sh` |
| RAM / vCPU | 117 GB / 16 vCPU | Thừa. cuML chạy GPU-side, vCPU chỉ lo tokenize + load dataset |

**Vì sao trần 7 GPU đáng giá.** Phase 6 phải chạy full grid `{models} × {4 dataset} × {3 method}
× {3 mức sparsity}` — hàng chục lượt clustering, mà clustering song song hoàn toàn theo sample.
Thêm nữa `pred.py` **đã sẵn** hỗ trợ đa GPU (`world_size = torch.cuda.device_count()`, spawn
một process mỗi GPU, [pred.py:198-207](LongBench/pred.py#L198-L207)) — cắm thêm GPU là eval
nhanh lên tuyến tính, không sửa dòng nào. Trần 2 của PCIe sẽ thành nút thắt đúng lúc cần nhất.
*(`offline_clustering.py` thì chưa shard — vòng lặp tuần tự, cần thêm cờ `--shard i/N` khi dùng
nhiều GPU.)*

**VRAM vs đĩa — hai thứ khác nhau, cùng bằng 80 nên dễ nhầm:**

| | VRAM 80 GB | Đĩa |
|---|---|---|
| Chứa | weights 13.5 GB + cache Q/K/V 32 layer (~25 GB ở 31.5K) + tensor tạm | file `.pt` do `torch.save` ghi ra |
| Cần | **~42 GB** đỉnh → dư gần gấp đôi | **~178 GB** cho gate đầy đủ |
| Vòng đời | giải phóng ngay sau mỗi sample | tích luỹ qua 500 sample, giữ tới khi eval đọc xong |

Ước tính peak VRAM: `13.5 GB (weights bf16) + S × 0.79 MB`.
Còn ~34 GB nếu bỏ `all_values_layers` (xem mục 8) — nhưng D3 đã chốt không patch.

**Chi phí ước tính cho Phase 0 + 1 + 2:** 12-24 giờ GPU × $1.59 = **$19-38**, cộng volume
200 GB ~$3/tuần. Cơ sở: bài gốc ghi clustering ~24 phút/context 128K, chi phí K-means scale
bậc hai theo S (vì `K = 5%·S`), quy về độ dài thật của LCC/RB rồi nhân hệ số an toàn 2-3×.
**Đây là ngoại suy từ một con số trong paper, chưa đo trực tiếp — sai số có thể ±2 lần.**

Biến môi trường cần set trước khi chạy, để dữ liệu nằm trên volume và không tải lại weights
mỗi lần restart pod (`configs/phase0.sh` đọc được qua `${VAR:-default}`, không phải sửa file):

```bash
export HF_HOME=/workspace/hf
export SQA_CLUSTER_DIR=/workspace/fixed-prompt-clusters
export SQA_RESULT_DIR=/workspace/phase0_results
export SQA_PHASE1_DIR=/workspace/phase1_data
export CUDA_VISIBLE_DEVICES=0
```

[scripts/setup_pod.sh](scripts/setup_pod.sh) tự sinh `/workspace/env.sh` chứa các biến này.

⚠️ **Ràng buộc kéo dài tới Phase 7.** Protocol yêu cầu cố định 1× H100 hoặc A100-80G cho mọi
lần đo latency. Đã chốt **SXM** thì Phase 7 phải giữ nguyên SXM — đổi sang PCIe là phải đo lại
toàn bộ latency. `record_env.py` tự ghi tên đầy đủ (`NVIDIA A100-SXM4-80GB`) vào
`env_record.json` và vào mục "Lịch sử chạy" bên dưới.

*(Dùng nhiều GPU cho clustering/eval thì không sao — đó là chi phí offline, không nằm trong
inference latency. Riêng benchmark latency Phase 7 luôn chạy 1 GPU.)*

---

## 6. Thay đổi code

### 2026-08-15 — Chuẩn bị Phase 0

**Thêm mới**
- `requirements.txt` — dependency đầy đủ (cuml, cupy, triton, flash-attn…)
- `configs/phase0.sh` — chốt cấu hình dùng chung
- `scripts/reference_table2.json` — số Table 2 cho LongChat / LLaMA-2-32K / LWM
- `scripts/check_gate.py` — so kết quả vs Table 2, tolerance 0.3, tự ghi nhật ký
- `scripts/record_env.py` — dump version/GPU/seed, tự kiểm tra transformers có đúng fork
- `scripts/phase0_gate.sh` — chạy trọn gate, tee console ra file
- `docs/PHASE0.md`, `EXPERIMENT_LOG.md`

**Bug đã sửa**

| # | Bug | File |
|---|---|---|
| 1 | `model2path.json` trỏ 4 path chết `/home/chooper/...` | `LongBench/config/model2path.json` |
| 2 | Offline lưu `hierarchical_lookup_*`, online load `hierarchical_centroids_*` → hierarchical crash | `offline_clustering.py`, `offline_clustering_v2.py` |
| 3 | `pred.py` ghi jsonl chế độ append → chạy lại là nhân đôi prediction, `eval.py` ra số sai mà không báo lỗi. Thêm `--overwrite` | `LongBench/pred.py` |
| 4 | `seed_everything` chỉ chạy ở process cha, `mp.spawn` không kế thừa RNG → seed lại trong con | `LongBench/pred.py` |
| 5 | Thêm `--seed` (chuẩn bị mean±std ≥3 seed) | `LongBench/pred.py` |

### 2026-08-16 — Sự cố hết quota đĩa ở mẫu 113/500, và bản vá chạy tiếp

**Chuyện gì xảy ra.** Clustering LCC chết ở mẫu 113/500 sau 1 giờ 40 phút:

```
RuntimeError: [enforce fail at inline_container.cc:764] . PytorchStreamWriter failed writing file data/1
tee: /workspace/logs/2_clustering.log: Disk quota exceeded
```

Volume thực tế được cấp ~50-55 GB, không phải 200 GB như tôi khuyến nghị. Lúc vỡ đang dùng
53 GB: `venv310` 15 GB + `hf` 14 GB + centroid 16 GB + `.cache` 6,8 GB + repo 1,2 GB.
Mà riêng centroid cho đủ 500 mẫu LCC cần **~71 GB**.

**`df` không phát hiện được.** `/workspace` là MooseFS dùng chung; `df` báo dung lượng cả cụm
(404 TB, còn 121 TB) chứ không biết gì về hạn mức riêng. Ghi vượt hạn mức thì báo
`Disk quota exceeded` trong khi `df` vẫn hiện hàng trăm terabyte trống. Muốn biết hạn mức thật
phải xem dashboard RunPod, hoặc thử ghi (`dd`) rồi xoá.

**Bản vá: bỏ qua mẫu đã có kết quả** ([offline_clustering.py](offline_clustering.py) trong
vòng lặp chính). Trước mỗi mẫu, kiểm tra `global_threshold_{dataidx}_{K}.pt` đã tồn tại chưa;
có rồi thì `continue`.

Không đổi kết quả — chỉ tránh tính lại thứ đã nằm trên đĩa. Cứu được 1 giờ 40 phút của lần
chạy này, nhưng lý do chính là **Phase 5/6 sẽ chạy clustering hàng chục lượt, mỗi lượt nhiều
giờ**; đứt giữa chừng là chuyện chắc chắn xảy ra, và trước bản vá thì mỗi lần đứt là mất toàn
bộ tiến độ.

Ghi chú: mẫu 113 có 2 file dở dang (`centroids_tensor` đã ghi, `centroids_labels` thì chưa).
Bản vá kiểm tra `global_threshold` — file ghi **sau cùng** — nên mẫu dở dang sẽ được làm lại
đúng như phải thế.

**Ba lỗi vận hành trong lúc xử lý, đều của tôi:**
1. Vòng chờ dùng `pgrep -f offline_clustering.py`, mà dòng lệnh của **chính nó** cũng chứa
   chuỗi đó → tự khớp → lặp vô hạn, bước 3-6 không bao giờ chạy. Đổi sang `kill -0 <PID>`.
2. `pgrep -f "offline_clustering.py --dataset lcc"` không khớp vì tên model nằm giữa
   (`offline_clustering.py longchat-... --dataset lcc`). Dùng `pgrep -af offline_clustering`.
3. Tin vào PID mà bash in ra sau `&`. Với lệnh dạng `source ... && cd ... && nohup python ... &`
   thì `&` áp lên **cả chuỗi**: bash tạo subshell, subshell thoát sau khi `nohup` tiếp quản,
   PID đó bị hệ thống cấp lại cho tiến trình khác. Phải lấy PID bằng `pgrep -af <script>`.

### 2026-08-16 — Gate LCC: phân bố độ dài thật + chi phí đo được

**Phase 1.4 trên toàn bộ 500 sample LCC** (40 giây): `1/500` truncate, `0/500` lệch template,
**`0/500` lệch token id**. Xác nhận ở quy mô đầy đủ, không chỉ 3 sample.

**Phân bố độ dài LCC** (đọc từ `lcc_meta.jsonl`, không phải ước lượng):

| | token |
|---|---|
| min | 1.280 |
| trung vị | 3.080 |
| trung bình | 4.290 |
| max | 31.494 |
| **đĩa cần** | **73 GB** |

**Tìm ra nguồn gốc sai lầm ước tính hôm qua.** LongBench công bố LCC "1235 words",
RepoBench-P "4206 words". Tôi quy đổi **2 token/word**; thực tế code là **3,5 token/word**
(4290 / 1235). Áp tỉ lệ đúng cho RepoBench-P: 4206 × 3,5 = 14.700 — khớp với 15.900 đo được.
Dùng 3,5 ngay từ đầu thì đã không lệch.

**Chi phí clustering LCC đo được**

| | |
|---|---|
| Tốc độ | **42,5 giây/sample** |
| Tổng 500 sample | **~6 giờ** |
| GPU utilization | **54-61%** |

Ước tính của tôi là 1,5-4 giờ → sai tiếp, lần này 1,5-4 lần (lần trước 10-60 lần).

**Vì sao chậm hơn dự đoán bậc hai.** Theo tỉ lệ S² từ RepoBench-P (S=15.900 → 240s), LCC với
S=4.290 lẽ ra chỉ 17,5 giây. Thực tế 42,5 giây. Phần dư ~25 giây/sample là **ghi đĩa**:
mỗi sample ~146 MB lên MooseFS ≈ 5,8 MB/s. GPU 54-61% xác nhận: khoảng 40% thời gian chờ I/O,
không phải nghẽn hoàn toàn ở I/O cũng không phải hoàn toàn ở tính toán.

**Cải tiến cho Phase 5/6** (lúc đó clustering chạy hàng chục lượt): lưu centroid ở **fp16**
thay fp32 → giảm nửa dung lượng ghi (73 → 37 GB) và phần lớn thời gian chờ I/O. Centroid chỉ
dùng để so cosine, fp16 dư chính xác. Không áp dụng cho lần gate này vì D3 đã chốt chạy trên
code nguyên bản.

### 2026-08-16 — ĐO THẬT: chi phí clustering, và ước tính thời gian của tôi sai nặng

Chạy `offline_clustering.py` trên RepoBench-P thật (LongChat, 5% centroid), dừng sau 2 sample.

**Số đo**

| Sample | K | S (token) | Dung lượng | Thời gian |
|---|---|---|---|---|
| 0 | 789 | ~15.900 | 517 MB | ~4 phút |
| 1 | 1141 | ~22.900 | 748 MB | ~5 phút |

*(S suy ra từ tên file: `K = 5% × (S − 100)`)*

**Công thức đĩa đúng, ước tính thời gian sai 10-60 lần**

- Đĩa: công thức 34 KB/token dự đoán 540 MB và 779 MB → thực tế 517 MB và 748 MB. **Khớp.**
- Thời gian: tôi đoán 5-30 giây/sample → thực tế **~5 phút/sample**.

Hai nguyên nhân. Một, tôi giả định RepoBench-P trung bình ~8K token dựa trên "Avg len 4206
words" của LongBench, thực tế hai sample đầu là 15,9K và 22,9K — mà chi phí K-means scale
**bậc hai** theo S nên sai độ dài 2 lần thành sai thời gian 4 lần. Hai, `run_global_threshold`
lặp Python qua **K × num_layers** = 789-1141 × 32 ≈ 25.000-36.000 vòng mỗi sample, mỗi vòng
thao tác trên tensor `[32, S, 100]`.

**Tính lại cho gate đầy đủ**

| | RepoBench-P | LCC (ước tính) | Tổng |
|---|---|---|---|
| Thời gian | ~37 giờ | ~2 giờ | **~39 giờ** |
| Đĩa | ~325 GB | ~68 GB | **~393 GB** |
| Tiền @ $1,60/h | $59 | $3 | **~$62** |

So với ước tính ban đầu của tôi ($19-38, volume 200 GB): **gấp đôi cả tiền lẫn đĩa**.
Volume 200 GB không đủ.

**Quyết định: gate chỉ trên LCC** (~2 giờ, ~68 GB, ~$3).

Gate tồn tại để trả lời đúng một câu hỏi — môi trường có tái lập được số của bài không.
LCC trả lời được câu đó (mốc Sq-70% = 56,93) với 1/20 chi phí. Chạy thêm 37 giờ trên
RepoBench-P chỉ để xác nhận lại điều LCC đã xác nhận là lãng phí. Bổ sung RepoBench-P sau,
khi có lý do cụ thể.

Hai đường thay thế nếu về sau cần RepoBench-P đầy đủ: chia shard cho nhiều GPU (trần 7 của
SXM → ~5 giờ, cần thêm cờ `--shard i/N` vào `offline_clustering.py`), hoặc tối ưu vòng lặp
Python trong `run_global_threshold`.

### 2026-08-16 — Dựng môi trường trên pod: stack đã kiểm chứng

Mất ~5 giờ pod (~$8) mới ra tổ hợp chạy được. **Ghi lại để không phải mò lại.**
Bản đầy đủ: `/workspace/working_env.txt` trên pod, và [scripts/setup_pod.sh](scripts/setup_pod.sh)
đã viết lại theo đúng luồng này.

| Gói | Bản | Ghi chú |
|---|---|---|
| Python | **3.10** (venv riêng, dựng bằng `uv`) | image RunPod mặc định 3.12 — không dùng được |
| torch / triton | 2.3.1+cu121 / **2.3.1** | |
| flash-attn | 2.6.3 (wheel cp310) | |
| cuML / cupy | 24.6.1 / 13.6.0 | |
| numpy / pyarrow / datasets | 1.26.4 / 16.1.0 / 2.20.0 | |
| transformers | 4.40.0.dev0 (fork, editable) | |

**Sáu cái bẫy, mỗi cái mất từ 20 phút tới 2,5 giờ**

1. **Python 3.12 của image làm hỏng ba thứ cùng lúc.** torch 2.3.1 khai báo
   `triton==2.3.1 ; python_version < "3.12"` nên trên 3.12 nó *không* kéo triton, để nguyên
   triton 3.4 của image — mà kernel dùng `tl.math.exp2` (API Triton 2.x). RAPIDS bản cp312
   đòi `numpy>=2` còn torch 2.3 cần `numpy<2`. flash-attn không có wheel cp312 cho torch2.3.
   → Dựng venv Python 3.10, cả ba tự tan.
2. **`uv venv` không cài pip vào venv** → gõ `pip` rơi vào pip hệ thống, cài nhầm vào Python
   3.12. Ba lần cài torch đầu tiên đều mất trắng vì lỗi này. → Dùng `python -m pip`.
3. **pip 23.0.1 do `ensurepip` cấp có bug chuẩn hoá tên**: gặp `Jinja2` báo "inconsistent
   Name: expected 'jinja2'", bỏ wheel, quay sang build sdist rồi chết vì thiếu `flit_core`.
   → Nâng pip trước khi cài gì khác.
4. **`pip install flash-attn` build từ nguồn 2,5 giờ** (~5 job song song trên 16 nhân, ~$4
   tiền GPU). Wheel dựng sẵn chỉ có trên GitHub Releases, tên phải khớp *đồng thời* torch
   minor + cxx11abi + python + cuda. → Cài thẳng URL wheel, 2 phút.
5. **`datasets` đời mới kéo pyarrow≥21 → phá cudf 24.6** (`pyarrow.lib has no attribute
   PyExtensionType`) → phá luôn cuML. → Ghim `datasets==2.20.0` + `pyarrow 16.1`.
6. **`squeezedattention/kernels.py` dòng 3 có `import pytest`** (sót từ repo gốc), mà
   `modeling_llama.py` import kernels ở top-level → thiếu `pytest` thì
   `from transformers import LlamaForCausalLM` cũng chết.

**Hai lỗi của tôi trong script, đã sửa**
- `setup_pod.sh` bước kiểm transformers so `REPO_ROOT` (đường dẫn symlink `/workspace/sa`)
  với `transformers.__file__` (đường dẫn đã giải symlink) → báo "SAI: không phải bản fork"
  trong khi fork hoàn toàn đúng. Đã đổi sang `os.path.realpath` cả hai vế.
- `requirements.txt` ghim `numpy<2` mà không lường RAPIDS 26.8 đòi `numpy>=2` — chính là
  bẫy #1. Đã viết lại toàn bộ file kèm lý do từng ràng buộc.

**Ba chi tiết vận hành**
- `df -h /workspace` hiện dung lượng **cả cụm MooseFS** (404T), không phải hạn mức volume.
  Phải xem trên dashboard RunPod.
- `/workspace` là **network filesystem**, không phải ổ cục bộ — ghi 500 file × ~272 MB có
  thể chậm hơn dự tính.
- LongBench có custom loader, hỏi `[y/N]` khi load → job dài không người trông sẽ **treo**.
  Đặt `HF_DATASETS_TRUST_REMOTE_CODE=1` (đã đưa vào `env.sh`).

**Đã xác nhận chạy thật trên GPU**
- `tl.math.exp2` biên dịch OK trên triton 2.3.1 — câu hỏi treo suốt buổi đã có đáp án
- `run_clustering` chạy đúng shape trên đường cupy + cuML + dlpack + torch
- 3 bộ test CPU: PASS
- `prepare_code_data.py` trên RepoBench-P thật: **0/3 truncate, 0/3 lệch template,
  0/3 lệch token id** — lần đầu code Phase 1.4 chạy trên dữ liệu thật

### 2026-08-15 — Phase 2: structure-aware clustering

**File mới**
- [struct_clustering.py](struct_clustering.py) — module lõi
- [offline_clustering_struct.py](offline_clustering_struct.py) — script offline, 3 nhánh ablation
- [scripts/test_struct_clustering.py](scripts/test_struct_clustering.py) — 72 test, chạy CPU

**Không sửa `offline_clustering.py` và `modeling_llama.py`** → gate Phase 0 nguyên vẹn.

| Hàm | Vai trò |
|---|---|
| `parse_units` | AST 5 level (`file`/`class`/`function`/`block`/`statement`), 5 ngôn ngữ |
| `assign_token_units` | token → unit nhỏ nhất bao nó, O(U log S) |
| `allocate_centroids` | chia ngân sách centroid theo unit, tổng đúng bằng budget |
| `hard_boundary_kmeans` | **đề xuất 1** — K-means độc lập trong từng unit |
| `build_l1_groups` | ép K1 về đúng mục tiêu bằng gộp/tách unit |
| `struct_hierarchy_l1` | **đề xuất 2** — L1 = trung bình L2 theo unit cha |
| `compute_token_type_weights` | Hướng 2(b), **tắt sẵn** |

**Bốn quyết định thiết kế**

1. **Ranh giới cứng thật, không phải init.** Bản cũ dùng AST để *khởi tạo* centroid rồi thả
   K-means chạy tự do → assignment vẫn vượt biên function. Test bất biến ở đây: duyệt mọi
   cluster của mọi head, không cluster nào chứa token của hai unit khác nhau.
2. **Vector hoá theo bucket lũy-thừa-2.** S=16K, H=32, K=800 → 110 lần gọi kernel, 3.2s trên
   CPU. Vòng lặp Python kiểu cũ ở cùng cấu hình là 256.000 vòng.
3. **Ngân sách không đủ thì raise, không im lặng cắt.** Cắt bớt sẽ phá tính "cùng budget" —
   nền tảng của mọi so sánh trong protocol.
4. **Khởi tạo tất định** (linspace trong unit, không RNG) → tái lập tuyệt đối, và seed chỉ
   ảnh hưởng đúng phần mình muốn nó ảnh hưởng.

**Vấn đề K1 và cách xử lý.** "Trung bình theo function/file" cho ra số nhóm = số unit cha mà
code *tình cờ* có: 5 class → K1=5 (bộ lọc L1 vô dụng), 300 function → K1 vượt xa 1% danh
nghĩa. Metadata centroid **được tính vào KV budget** nên K1 lệch là budget lệch.
`build_l1_groups` gộp unit cha liền kề (khi thừa) hoặc tách thành dãy unit con liền kề (khi
thiếu) để chạm đúng mục tiêu, vẫn không cắt đôi unit L2. K1 thực tế được ghi ra
`k1_stats_<dataidx>.pt` để Phase 6 báo cáo budget đo thật.

**Hai bug tự tạo, test bắt được** — cả hai đều thuộc loại *không crash, chỉ sai lệch*:
- Trọng số L1 dùng số đếm cluster của head 0 cho mọi head. Ranh giới cứng chỉ đảm bảo cùng
  phân hoạch theo **unit**; trong một unit thì mỗi head chia cluster khác nhau. Test
  "weighted L1 == trung bình toàn bộ key" lệch 6e-01 → lộ. Đã đếm riêng từng head.
- `allocate_centroids` không biết trần số unit con: 2 function không thể tách thành 3 nhóm,
  nhưng hàm phân bổ theo số token nên cấp 3 → K1 ra 5 thay vì 6. Đã thêm tham số `caps`.

**Chưa chạy GPU.** Toàn bộ test dùng `torch.randn`, chưa từng chạy trên key vector thật.

### 2026-08-15 — Phase 1.2: khảo sát CrossCodeEval / RepoBench

Tải dữ liệu thật về đo, không dựa vào mô tả trong paper. Báo cáo đầy đủ:
[docs/PHASE1_DATASETS.md](docs/PHASE1_DATASETS.md).

**Kết quả: giả định của protocol về hai bộ này không đúng.**

| | CrossCodeEval | RepoBench v1.1 |
|---|---|---|
| Quy mô (Python) | 2665 sample / 471 repo | 4017 sample / 1722 repo |
| Sample mỗi repo | tb 5.7, max 69 | tb 2.3, max 6 |
| Độ dài context | tb ~1.4K token, max ~7K | trung vị 3.6K, max 14K |
| Năm tạo repo | 2023 | 2023 (toàn bộ) |

Hai vấn đề:
1. **Context ngắn** — protocol giả định 20-150K token, thực tế trung vị 1.4-3.6K.
   `crossfile_context` chỉ là top-5 chunk đã retrieve, repo gốc không kèm theo.
2. **Context không dùng chung giữa các query cùng repo** — nghiêm trọng hơn nhiều.
   Đếm trên 1500 sample Python đầu của CrossCodeEval: 211 repo có >1 sample, **chỉ 2 repo**
   có mọi sample dùng chung context. Repo `hq0709-Depth-NeuS`: 42 sample → 42 context khác
   nhau. RepoBench tương tự: 962 repo đa sample, chỉ 153 dùng chung.

Điều 2 phá thẳng premise của SA (*một fixed context, nhiều query* → khấu hao chi phí
clustering). Bài gốc đã thừa nhận khoảng trống này ở Section 5 và đó là lý do họ dựng
PreFixQA. Hai bộ này có **đúng cùng hạn chế** như LongBench LCC/RepoBench-P.

→ Không viết loader cho hai bộ ở dạng nguyên bản. Xem quyết định D4.

⚠️ Dữ liệu đã tải (`cceval/data/`, 888 MB sau giải nén) nằm trong scratchpad của phiên làm
việc, **sẽ mất khi hết phiên**. Cần giữ thì copy sang volume trước.

### 2026-08-15 — Phase 1.5 + 1.6: port Squeezed Attention sang Qwen2 (GQA)

Quyết định D1: **chọn Qwen2.5-Coder-7B-Instruct làm model chính**, port ngay.
Gate Phase 0 vẫn chạy LongChat để khớp Table 2 — hai việc độc lập.

**File sửa**
- `transformers/.../qwen2/configuration_qwen2.py` — thêm 10 tham số SA + `different_prefix_index`
- `transformers/.../qwen2/modeling_qwen2.py` — port toàn bộ đường SA
- `squeezedattention/clustering.py` — `run_global_threshold` thêm nhánh GQA
- `offline_clustering.py`, `LongBench/pred.py` — `AutoConfig`/`AutoModelForCausalLM` thay `Llama*`
- `LongBench/config/model2path.json`, `model2maxlen.json` — thêm `qwen2.5-coder-7b-instruct`

**Bốn quyết định thiết kế**

1. **Không sửa một dòng nào của `modeling_llama.py`.** Các kernel wrapper (`centroid_lookup`,
   `dynamic_sparse_attention`, `causal_attention_kernel`, `compute_k_idx_optimized`) đều là
   module-level nên `modeling_qwen2.py` import thẳng, không copy ~500 dòng. Đường Llama của
   gate Phase 0 giữ nguyên tuyệt đối.

2. **Cluster trên key TRƯỚC `repeat_kv`.** GQA nhân bản head KV thành các bản sao *giống hệt*.
   Cluster sau `repeat_kv` sẽ tốn gấp 7 lần công và dung lượng mà ra đúng cùng kết quả.
   Nên `return_qkv_states` trả về key 4 head, còn centroid được `repeat_interleave` lên 28
   head ngay trước lookup.

3. **`repeat_interleave`, không phải `repeat`.** `repeat_kv` của transformers ánh xạ
   query head `h` → KV head `h // groups`. `tensor.repeat()` cho ánh xạ `h % H_kv` — khác
   hẳn, mà vẫn chạy trơn tru không crash, chỉ tra nhầm nhóm centroid. Test bắt riêng ca này.

4. **Số head lookup lấy từ `centroid_labels.shape[-2]`.** Bản Llama dùng `key_states.shape[1]`;
   copy nguyên sang Qwen2 sẽ ra 4 thay vì 28 vì `repeat_kv` chưa chạy tại điểm đó.

**Chốt chặn tự động**
- `assert` centroid có đúng `num_key_value_heads` head khi nạp — sai là báo ngay, thay vì
  âm thầm tra nhầm rồi tụt accuracy không rõ lý do.
- `assert not use_sliding_windows` khi bật centroid — sliding window cắt key theo cửa sổ,
  mâu thuẫn với giả định SA truy cập được toàn bộ fixed context. Qwen2.5-Coder mặc định đã
  tắt; `offline_clustering.py` và `pred.py` ép tắt thêm lần nữa.
- `Qwen2SdpaAttention` nhận `**kwargs` và raise `NotImplementedError` nếu bị truyền centroid.

### 2026-08-15 — Phase 1.4: offset token

- `scripts/prepare_code_data.py` — sinh offset ký tự cho từng token trên prompt **cuối cùng
  sau truncation**, kèm vị trí vùng code, `shared_prefix_length`, cờ `truncated`.
  Có `--self_test` chạy offline không cần mạng/GPU (12/12 pass).
- Kiểm tra an toàn tích hợp sẵn: assert token id của tokenizer nhanh == tokenizer chậm.
  Lệch là exit 1, vì offset lệch nghĩa là Phase 2 sẽ gán `unit_id` cho sai key vector.

---

## 7. Việc đang chặn & quyết định cần chốt

### Việc chặn

| # | Việc | Chặn phase | Ghi chú |
|---|---|---|---|
| B1 | Chạy gate thật trên GPU | 1, 2 | Cần A100 80GB. Peak mem ước tính ~42 GB ở context 31.5K |
| ~~B2~~ | ~~Persist offset sau truncation~~ | — | ✅ Code xong (`scripts/prepare_code_data.py`), còn chạy thật trên pod |
| ~~B3~~ | ~~Hard-boundary clustering (2.3)~~ | — | ✅ Code xong + test bất biến; còn nối vào pipeline offline |
| ~~B4~~ | ~~Port SA sang `modeling_qwen2.py`~~ | — | ✅ Code xong, chờ chạy thật |
| ~~B5~~ | ~~Kiểm chứng GQA~~ | — | ✅ Code xong + test shape 20/20; còn cần xác nhận trên GPU thật |

### Quyết định cần chốt

**Ký hiệu:** `B<n>` = việc đang **B**locking (chặn tiến độ, thuần kỹ thuật, tự làm được).
`D<n>` = **D**ecision — chỗ rẽ hướng nghiên cứu, phải người phụ trách chốt vì mỗi lựa chọn
dẫn tới khối lượng công việc và kết luận khoa học khác nhau. Đánh số để tham chiếu lại.

| | Nội dung một dòng | Trạng thái |
|---|---|---|
| D1 | Port Qwen2.5-Coder ngay hay hoãn tới trước Phase 6? | ✅ chốt 15/8 — port ngay, Qwen là model chính |
| D2 | `fixed_context` của RepoBench-P có gồm in-file prefix không? | ✅ chốt 15/8 — giữ nguyên LongBench |
| D3 | Patch tiết kiệm VRAM trước khi chạy gate? | ✅ chốt 15/8 — không patch |
| D4 | Lấy benchmark fixed-context-dài từ đâu? | ⬜ chưa chốt — chặn Phase 6 |


**D1 — Port Qwen ngay, hay hoãn?** ✅ **ĐÃ CHỐT 15/8: port ngay, Qwen2.5-Coder-7B-Instruct là model chính.** Phase nào yêu cầu gì thì chạy đúng cái đó — gate Phase 0 vẫn dùng LongChat để khớp Table 2.

- *Hoãn port:* giữ được hạn 22/8 cho Phase 2. Đổi lại Phase 5 — thí nghiệm rẻ nhất và quyết định H0 — chạy trên model không phải code model, làm yếu kết luận.
- *Port ngay:* Phase 5 chạy đúng model chính, nhưng port + kiểm GQA là việc nặng, nhiều khả năng trượt 22/8.

**D2 — Định nghĩa `fixed_context` cho RepoBench-P.** ✅ **ĐÃ CHỐT 15/8: giữ nguyên LongBench, áp định nghĩa protocol cho CrossCodeEval + RepoPreFixQA.**

Protocol nói `fixed_context = cross-file context + phần file hiện tại trước cursor`. Nhưng
[dataset2prompt.json:22-23](LongBench/config/dataset2prompt.json#L22-L23) đang đặt
`repobench-p` = `{context}{input}` còn `repobench-p_prompt` = `{context}` — tức in-file prefix
bị tính vào phần **query**, không phải fixed context.

- Sửa cho khớp protocol → **không so được với Table 2 nữa**, mất luôn giá trị của gate Phase 0.
- Giữ nguyên → lệch định nghĩa protocol trên LongBench.
- *Đề xuất:* giữ nguyên cho LongBench (bảo toàn khả năng so với bài), áp định nghĩa protocol cho CrossCodeEval + RepoPreFixQA.

**D3 — Có patch tiết kiệm VRAM trước khi chạy gate không?** ✅ **ĐÃ CHỐT 15/8: không patch, chạy gate trên code nguyên bản.**

**D4 — Nguồn cho benchmark fixed-context-dài?** *(chưa chốt — chặn Phase 6, không chặn 22/8)*

Khảo sát 15/8 cho thấy mục 1.2 và RepoPreFixQA ⏸️ **thực chất là cùng một việc**: dựng
benchmark có fixed context dài dùng chung nhiều query. Khác nhau ở nguồn nhãn:

| | Tận dụng CrossCodeEval | Pipeline RepoPreFixQA của protocol |
|---|---|---|
| Nguồn repo | 471 repo Python, có commit hash | 30-50 repo GitHub 2025+ |
| Nhãn | có sẵn, là nhãn **thật** | LLM sinh + self-consistency 5 lần + LLM-as-judge |
| Chi phí | thấp | cao |
| Contamination | **có** (repo 2023, Qwen train tới ~2024) | tránh được |
| Query/repo | tb 5.7, max 69 | ~24 (theo bài) |

Cả hai đều phải clone repo về và lấy toàn bộ repo làm fixed context. Đã kiểm: 15/15 repo
RepoBench nhiều sample nhất còn clone được (0.1–32 MB).

*Đề xuất:* làm đường CrossCodeEval trước để có kết quả sớm, bổ sung repo 2025+ sau để trả
lời câu hỏi contamination của reviewer.

⚠️ Lưu ý về protocol: bảng chính và grid Phase 6 (đều **ngoài** thẻ `<chưa cần làm bây giờ>`)
đều cần RepoPreFixQA, trong khi việc dựng nó nằm **trong** thẻ. Sớm muộn phải mở thẻ đó,
không thì bảng chứng minh C1 thiếu một cột.

Hook thu cả `all_values_layers` nhưng values không hề được dùng → bỏ đi cắt 1/3 QKV cache
(42 GB → ~34 GB). Thêm `.cpu()` cho keys/queries thì còn ~15 GB. Nếu patch thì **phải patch
trước khi chạy gate**, để số Phase 0 sinh ra từ đúng code sẽ dùng cho các phase sau.

---

## 8. Lỗi đã biết, chưa chặn Phase 0

- [squeezedattention/utils.py:44](squeezedattention/utils.py#L44) — `build_chat(prompt, model_name)`
  dùng biến `model_name` chưa định nghĩa trong scope → `NameError`. Chỉ nổ với dataset
  **ngoài** `{trec, triviaqa, samsum, lcc, repobench-p}`. Gate không chạm, nhưng sẽ nổ ở
  Phase 6 nếu thêm task QA.
- `README_EXTENSIONS.md` trỏ tới `LongBench/run_evaluation.sh` — file không tồn tại.
  Dùng `scripts/phase0_gate.sh`.
- Đường `--code_aware` của `offline_clustering_v2.py` **chưa từng chạy thành công**:
  tokenizer load `use_fast=False` nhưng `map_tokens_to_scopes` gọi `return_offsets_mapping=True`
  → `NotImplementedError`.
- Hook trong `offline_clustering.py` thu cả `all_values_layers` nhưng values **không hề được
  dùng** — lãng phí 1/3 QKV cache (8.3 GB ở context 31.5K).

---

## Lịch sử chạy

<!-- check_gate.py phụ lục bên dưới. Không xoá dòng này. -->

*(chưa có lần chạy nào)*
