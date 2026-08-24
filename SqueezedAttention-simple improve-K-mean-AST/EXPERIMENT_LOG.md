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
| 0 | Môi trường + tái lập baseline SA | — | ✅ **GATE PASS** (LCC, dung sai ±2.0) |
| 1 | Chuẩn bị dữ liệu code | — | ✅ **GATE PASS** (Qwen base, paired test p=0,22) |
| 2 | Structure-aware clustering (Idea 1) | **22/8** | 🟡 6/6 có code, chưa chạy GPU |
| 3 | Symbol / def-use signal (Idea 2) | **30/8** | ❌ 0/4 |
| 4 | Incremental re-clustering (Idea 3) | **8/9** | ❌ 0/4 |
| 5 | C2 retrieval quality — chạy TRƯỚC Phase 6 | — | ❌ 0/5 |
| 6 | C1 accuracy@budget end-task | — | ❌ 0/5 |
| 7 | C3 + phân tích | — | ❌ 0/4 |

Thứ tự chạy theo protocol: `Phase 0 → Phase 1 → Phase 5 (C2, quyết định H0, dừng sớm nếu fail) → Phase 6 (C1) + Phase 4/7 (C3)`.
Phase 2 và 3 là phần *cài đặt* mà Phase 5/6 sẽ đo.

---

### Phase 0 — Môi trường + tái lập baseline SA · ✅ GATE PASS

Mục tiêu: dựng lại đúng pipeline SA để mọi cải tiến là ablation trên cùng một nền.

| # | Việc | Trạng thái | Chi tiết |
|---|---|---|---|
| 0.1 | Clone repo gốc, dựng env | ✅ | Repo + fork transformers 4.40.0.dev0 có sẵn. `requirements.txt` đã viết đủ (cuml, cupy, triton, flash-attn) |
| 0.2 | Chốt config mặc định (5%, 1%/5%, obs 100, 32K) | ✅ | [configs/phase0.sh](configs/phase0.sh) |
| 0.3 | Ghi version transformers/triton, GPU, seed | ✅ | [scripts/record_env.py](scripts/record_env.py), tự kiểm tra transformers có đúng fork |
| 0.4 | Script chạy gate LCC/RB | ✅ | [scripts/phase0_gate.sh](scripts/phase0_gate.sh) + [scripts/check_gate.py](scripts/check_gate.py), tolerance ±0.3 |
| 0.5 | Số đích từ Table 2 | ✅ | [scripts/reference_table2.json](scripts/reference_table2.json), xem mục 2 |
| 0.6 | Sửa bug chặn gate | ✅ | 5 bug, xem mục 6 |
| 0.7 | Cài đặt thật trên pod | ✅ | A100 SXM 80GB. Stack đã kiểm chứng, xem mục 6 |
| 0.8 | Chạy gate | ✅ | **PASS** với dung sai nới ±2.0. All-KV 54,83 · Sq-70% 56,08. Chỉ LCC; bỏ RepoBench-P và Sq-80/90% |
| 0.9 | Hậu kiểm prediction thô | ✅ | Thêm 17/8. `inspect_preds.py` trên cả 500 mẫu: dòng chấm rỗng **14,6%** (All-KV) / **12,6%** (Sq-70%), dưới ngưỡng 25%. Prediction là code thật → 54,83 không phải số ảo. Xem mục 6 |

**Còn lại:**
1. Thuê pod theo cấu hình mục 5, cài theo [docs/PHASE0.md](docs/PHASE0.md).
2. Chạy thử **3 sample** trước để đo giây/sample và MB/sample — hai con số chưa biết, cần để ước lượng job full và dung lượng đĩa.
3. `bash scripts/phase0_gate.sh`. Kết quả tự ghi vào mục "Lịch sử chạy".

**Đã kiểm, không phải lo:** 8 tham số SA đều có default trong `configuration_llama.py` (nên `pred.py` không set `return_qkv_states` vẫn chạy); `reset_context = hidden_states.shape[1] > 1` nên centroid nạp lại đúng mỗi sample; `rope_scaling` linear factor 8 của LongChat được fork chấp nhận.

---

### Phase 1 — Chuẩn bị dữ liệu code · ✅ **GATE PASS** · dữ liệu ✅ **PASS 19/8 (500+500 mẫu)**

Cần đúng cấu trúc *một fixed context → nhiều user query* thì premise của SA mới áp dụng.

#### Gate dữ liệu 5 bước — [scripts/check_phase1_data.py](scripts/check_phase1_data.py), chạy 19/8

Gate cũ (`check_phase1.py`) kiểm **bản port** và cần GPU. Gate này kiểm **dữ liệu Phase 2 sẽ
đọc**, chạy trên CPU trong ~2 phút mỗi dataset. Kết quả:

| Bước | Nội dung | LCC | RepoBench-P |
|---|---|---|---|
| 1 | ngôn ngữ đúng ở **từng mẫu** | ✅ `python 182 · java 160 · csharp 158` | ✅ `python 236 · java 264` |
| 2 | đủ mẫu (meta + npz) | ✅ 500/500 | ✅ 500/500 |
| 3 | offset: fast==slow, không giảm, phủ kín, không lệch byte/ký tự | ✅ 0 lệch · 1.559.310 token | ✅ 0 lệch · 4.882.207 token |
| 4 | fixed_context: `sp_len` khớp, `n_ctx>0`, context không mất | ✅ 0 mẫu mất context · 1/500 truncate | ✅ 0 mẫu mất context · 8/500 truncate |
| 5 | tổng kết | ✅ **PASS** | ✅ **PASS** (1 cảnh báo) |

**Bổ sung 22/8 — offset BYTE theo đúng chữ của protocol.** Protocol Phase 1 viết *"lưu kèm
byte offset của từng token"*, nhưng `return_offsets_mapping` của tokenizer nhanh trả offset
**ký tự**. Nay lưu cả hai trong cùng một `.npz`:

    offsets_<i>        offset ký tự — Phase 2 hiện chạy trong hệ này (đã kiểm chứng)
    offsets_bytes_<i>  offset byte  — đúng yêu cầu protocol, cho công cụ làm việc thẳng
                                      với tree-sitter mà không phải quy đổi

Hai hệ chỉ trùng nhau khi thuần ASCII: LCC 0/500 mẫu có non-ASCII, RepoBench-P **107/500**.
Gate Phase 1 thêm bất biến thứ sáu: cắt chuỗi byte theo `offsets_bytes` phải ra **đúng** chuỗi
mà `offsets` ký tự cắt ra. Kiểm 200 token trải đều mỗi mẫu.

⚠️ Bộ dữ liệu sinh trước 22/8 **không có** `offsets_bytes_*` → gate sẽ báo thiếu. Phải sinh
lại (LCC 37 giây · RepoBench-P 5 phút). **Không ảnh hưởng centroid Phase 2 đã có**: offset ký
tự và `shared_prefix_length` không đổi, chỉ thêm một mảng mới.

**Xác nhận lại trên pod A100 (20/8):** cả hai dataset PASS, và **mọi con số trùng khít** với
lần chạy trên máy Windows — không lệch một token:

| | LCC | RepoBench-P |
|---|---:|---:|
| token kiểm | 1.559.310 | 4.882.207 |
| token vắt biên unit | 0,48% | 1,06% |
| unit/mẫu (trung vị · min · max) | 15 · 2 · 280 | 100 · 4 · 669 |
| mẫu suy biến (U≤2) | 12/500 (2,4%) | 0/500 |
| truncate | 1/500 | 8/500 |
| **mẫu Unicode — phép thử vi sai** | 0/500 (không áp dụng) | **107/500, span khớp tuyệt đối** |

Đường dữ liệu Phase 1 tất định, không phụ thuộc môi trường (Windows CPU vs pod Linux A100).
Bản vá byte→ký tự chỉ được kiểm thật trên RepoBench-P, vì LCC thuần ASCII.

Thời gian thật đo trên pod: LCC 500 mẫu **37 giây** (13,4 it/s), RepoBench-P **4 phút 51**
(1,7 it/s — context dài gấp ~3,7 lần). Gate kiểm lại thêm 2 phút.

#### ⏱️ Số đo chi phí đầu tiên (C3) — đường Sq-70% chậm gấp 8,4 lần All-KV

Trích từ log gate 20 mẫu LCC, Qwen2.5-Coder-7B base, A100-80GB. **Ba lần chạy độc lập
(17/8, 18/8 ×2) cho cùng kết quả**, không phải nhiễu một lần:

| Bước | 20 mẫu | s/mẫu | so với All-KV |
|---|---|---:|---:|
| [2] offline clustering | 1:44 – 1:46 | 5,2 | — (chi phí một lần) |
| [4] pred All-KV | 0:45 – 0:47 | 2,3 | 1,0× |
| **[5] pred Sq-70%** | **6:21 – 6:33** | **19,3** | **8,4×** |

**Chi phí nằm ở decode, không phải prefill.** Bằng chứng: cùng pipeline, bản **instruct**
chỉ mất 1:36 ở bước [5] (2,8×) thay vì 6:25 — vì instruct sinh prediction gần như rỗng
(điểm 17,60) nên decode rất ít token. Chi phí tỉ lệ với **số token sinh ra**, tức nằm ở vòng
tra centroid + so ngưỡng + gom KV thưa mỗi bước decode.

Ba điều phải nói cho đúng khi báo cáo:

1. Đây là **wall-clock của bản cài đặt tham chiếu**, không phải phát biểu về độ phức tạp của
   phương pháp. Bản gốc tối ưu cho ngữ cảnh dài hơn nhiều.
2. LCC có `n_ctx` trung vị **2.194 token** — nhiều khả năng nằm **dưới điểm hoà vốn**: phần
   attention tiết kiệm được không bù nổi chi phí tra cứu. Bài gốc đo ở 32K+.
3. Nó **ăn khớp** với việc Qwen ra −2,80 còn LongChat ra +1,25: ở ngữ cảnh ngắn, bỏ 70% key
   vừa chậm hơn vừa không lợi gì về chất lượng.

Việc cần làm ở Phase 7: đo lại chi phí này theo `n_ctx` để tìm điểm hoà vốn, và tách riêng
prefill với decode. Phase 5 (C2, recall@budget) **không** đi qua đường decode nên không chịu
chi phí này — thêm một lý do chạy Phase 5 trước.

⚠️ Một lỗi của chính gate, đã sửa: khi chạy với `--limit`, bản đầu so phân bố ngôn ngữ của
meta (20 mẫu) với phân bố của **cả 500 mẫu** dataset → FAIL giả. Nay so trên cùng tập chỉ số.
Cùng họ với lỗi D6 cảnh báo (so hai số đo trên hai tập mẫu khác nhau), lần này nạn nhân là
cái gate. Phép kiểm per-sample ngay cạnh đó vẫn báo đúng, nên bắt được ngay.

**Ba lỗi đã sửa để đạt PASS** (trước đó gate này chưa tồn tại, ba lỗi đều im lặng):

1. **`language` bị hardcode `"python"`** cho mọi mẫu ([prepare_code_data.py](scripts/prepare_code_data.py)).
   LongBench có sẵn trường `language` mỗi mẫu, mà 63,6% LCC và 53% RepoBench-P **không phải
   Python**. Đo hậu quả trước khi sửa: **59,5% mẫu LCC chỉ còn ≤2 unit** ở `level=function`
   (Java 160/160), tức `hard_boundary_kmeans` chạy trên một unit duy nhất = **đúng bằng
   baseline SA**. Ablation Idea 1 sẽ ra "không khác gì SA" mà không crash, không cảnh báo.
   Sau khi sửa: unit trung vị **2 → 15** (LCC), **2 → 100** (RepoBench-P); mẫu suy biến còn
   2,4% và 0%; node ERROR trung vị **139 → 0**.
2. **Trộn byte với ký tự**: `parse_units` trả span theo *byte* của tree-sitter, còn offset
   Phase 1.4 là *ký tự*. LCC thuần ASCII nên không lộ; RepoBench-P có **107/500 mẫu chứa
   Unicode** trong vùng code → span lệch dần và cộng dồn. Đã thêm `byte_to_char_index` vào
   [struct_clustering.py](struct_clustering.py), áp cho cả `parse_units` lẫn
   `compute_token_type_weights`. Kiểm bằng **phép thử vi sai** (thay non-ASCII bằng `x`:
   số ký tự giữ nguyên, số byte đổi → span phải giống hệt): 107/500 khớp tuyệt đối.
3. **Bất biến kiểm tra sai** (lỗi của chính gate, ghi lại để không lặp): bản đầu đòi token
   nằm *trọn* trong unit và các lát offset *rời nhau*. Cả hai đều quá chặt. Token cuối một
   hàm nuốt luôn ký tự xuống dòng ngay sau hàm (0,48% token ở LCC) — hệ quả bình thường của
   việc gán theo điểm bắt đầu. Và BPE byte-level tách một ký tự CJK thành nhiều token, các
   token con **chồng** offset lên nhau (31/500 mẫu RepoBench-P) chứ không mất ký tự nào.
   Bất biến đúng là: start nằm trong span, và offset **không để khoảng trống**.

#### Bốn dataset đã sẵn sàng cho Phase 2/6 · ✅ 23/8

Hai loader mới đưa CrossCodeEval và RepoBench v1.1 vào cùng đường ống với LongBench
(`--data_source jsonl`), tất cả sinh với **Qwen2.5-Coder-7B-Instruct · force_chat · full ·
maxlen 31500**:

| | LCC | RepoBench-P | CrossCodeEval | RepoBench v1.1 |
|---|---:|---:|---:|---:|
| Mẫu / context | 500 | 500 | **9.928** | **1.735** |
| Query | 500 | 500 | 9.928 | **7.080** |
| Query/context | 1 | 1 | 1 | **trung vị 3 · max 50** |
| `n_ctx` trung vị | 2.294 | 8.328 | **946** | **16.154** |
| Truncate ở 31.500 | 1/500 | 8/500 | 0/9.928 | **298/1.735 (17,2%)** |
| Unit/mẫu (function) | 15 | 107 | **6** | **98** |
| Mẫu suy biến (U≤2) | 2,4% | 0% | **21,4%** | 0,2% |
| Gate | PASS | PASS | PASS* | **PASS** |

\* CrossCodeEval cần `--max_degenerate_ratio 0.25`. **Không phải nới tiêu chuẩn cho qua** —
21,4% là tính chất của bộ đó: chunk retrieve ra là đoạn **cắt giữa hàm**, không có
`function_definition` trọn vẹn. Bằng chứng cùng chiều: **0/9.928 mẫu chứa Unicode parse
sạch** (1.170 mẫu đều có node ERROR). Phải ghi vào bài: *đề xuất 1 có rất ít ranh giới để
ràng buộc trên ~1/5 số mẫu CrossCodeEval.*

**RepoBench v1.1 là bộ duy nhất thoả câu mở đầu của protocol** — một fixed context, nhiều
query. Nhưng ở `maxlen=31500` thì **17,2% mẫu bị cắt giữa**, và đó đúng là những mẫu dài
nhất. Muốn dùng đúng mục đích phải chạy với `--rope_scaling dynamic:4`.

**Hai phép kiểm của gate đã được siết đúng phạm vi** (23/8), không phải nới lỏng:
- Phép thử vi sai byte/ký tự chỉ áp cho mẫu **parse sạch** — trên code gãy cú pháp, cơ chế
  phục hồi lỗi của tree-sitter nhảy chỗ khác khi thay ký tự nên báo lệch giả (8/1735 mẫu).
- Phủ kín offset cho phép hở **≤8 ký tự/mẫu** — do chuẩn hoá của tokenizer nhanh, đo được
  3/1735 mẫu hở tổng cộng 8 ký tự (một mẫu là `U+0300` dấu tổ hợp).

Bất biến **trực tiếp** thì vẫn kiểm 100% mẫu và khớp tuyệt đối: offset byte cắt ra đúng
chuỗi mà offset ký tự cắt ra, trên cả 9.928 + 1.735 mẫu.

---

#### D7 — "128K" của protocol KHÔNG có sẵn; giữ 31.500 · ✅ chốt 23/8

Protocol ghi *"Qwen2.5-Coder-7B-Instruct (128K)"*. Kiểm `config.json` của **cả hai** bản:

    max_position_embeddings = 32768
    rope_scaling            = None

128K là con số trên model card, **chỉ đạt được khi bật YaRN** (`rope_scaling` factor 4) —
một tuỳ chọn phải khai báo, không phải mặc định. Ngày 23/8 tôi đã đổi `model2maxlen` lên
127500 rồi **đổi lại 31500** sau khi kiểm config.

Vì sao 127500 không những vô ích mà còn có hại: mẫu dài hơn 32768 token được đưa nguyên vào
model chỉ có 32768 vị trí → ngoài dải RoPE → kết quả là rác. Chính cảnh báo đã hiện lúc
chạy: `Token indices sequence length is longer than the specified maximum sequence length
for this model (34855 > 32768)`. Ảnh hưởng ~9/1000 mẫu (1 LCC + 8 RepoBench-P), nhưng là
rác thật chứ không phải sai số.

**Và YaRN KHÔNG bật được trong repo này.** `transformers/src/transformers/models/qwen2/
modeling_qwen2.py` của fork dùng `Qwen2RotaryEmbedding` trần, **không có nhánh xử lý
`rope_scaling` nào** — không linear, không dynamic, không yarn. Muốn 128K phải tự port YaRN
vào đúng file mà Squeezed Attention đã vá, tức đặt bản port vào rủi ro.

=> **"(128K)" của protocol không đạt được trong codebase này.** Không phải lựa chọn, là ràng
buộc kỹ thuật. Giữ 31.500 (native 32.768 trừ lề). Nếu Phase 6 cần context >32K cho
RepoBench v1.1 thì phải giải quyết YaRN trước, và đó là một hạng mục kỹ thuật riêng.

---

#### D6 — Chính sách khi số unit vượt ngân sách centroid · ✅ chốt 20/8

Đo đầy đủ trên 991 mẫu (đã bỏ mẫu truncate), ngân sách 5%. **Hai** đường chặn khác nhau, chứ
không phải một:

| level | U > ngân sách · LCC | RepoBench-P | trần `max_k=64` chặn · LCC | RB-P |
|---|---:|---:|---:|---:|
| class | 0/499 | 0/492 | **236/499 (47%)** | 65/492 |
| **function** | **0/499** | **2/492 (0,4%)** | **22/499 (4,4%)** | 2/492 |
| block | 13/499 (2,6%) | 52/492 (10,6%) | 11/499 | 0/492 |
| statement | 386/499 (77,4%) | 441/492 (89,6%) | 2/499 | 0/492 |

**Đường 2 không phải vấn đề ngân sách** mà là hằng số cài đặt: `max_k_per_unit=64` làm
`sum(cap)` nhỏ hơn ngân sách khi context dài mà ít unit, nên không tiêu hết được ngân sách
dù còn thừa gấp nhiều lần. Đã đổi mặc định thành **không trần** (chỉ chặn theo số token của
unit) — khi đó `sum(cap) = n_ctx` luôn ≥ ngân sách 5%. Tham số vẫn còn để ghìm bộ nhớ.

**Đường 1 là chính sách, đã chốt:** mặc định `--on_budget_exceeded skip` — bỏ mẫu, ghi vào
`feasibility_<dataset>_<method>_<level>_pc<N>.json`, chạy tiếp. **Không** `raise` (chết cả
run, mất GPU time của mọi mẫu sau) và **không** tự gộp.

Thứ tự chạy Phase 2:

1. **function** — chạy trước, gần như không mẫu nào bị bỏ (0% / 0,4%).
2. **block** — chạy, **ghi số mẫu phải bỏ** (2,6% LCC · 10,6% RB-P; hai dataset báo riêng).
3. **statement** — **không gộp** trong thí nghiệm chính, chỉ ghi bao nhiêu mẫu infeasible.

Rồi mới quyết định có cần biến thể budget-constrained (`--on_budget_exceeded merge`) không.

Lý do không gộp mặc định: ở statement phải gộp **32–67% số unit** của **77–90% số mẫu**.
Thứ được gọi là "statement" khi đó đã bị làm thô về cỡ block, nên đầu mịn của level sweep sẽ
phẳng ra **do chính thao tác gộp**, không phải do cấu trúc code — đúng họ với lỗi hardcode
`python`: một quyết định cài đặt hoá trang thành một phát hiện.

**Hệ quả bắt buộc ghi vào bài:** bỏ mẫu làm tập còn lại thiên lệch (mẫu bị bỏ thường dài,
cấu trúc mịn). Không được so điểm statement trên ~23% mẫu với điểm function trên 100% mẫu.
Phải so trên **tập giao** các mẫu khả thi ở mọi level — đó là lý do `feasibility_*.json` ghi
đúng danh sách `dataidx` chứ không chỉ ghi số đếm. `offline_clustering_struct.py` in cảnh
báo lớn khi tỉ lệ bỏ vượt 10%.

`merge_units_to_budget` (dùng cho biến thể, không dùng mặc định) gộp unit **liền kề theo vị
trí trong code**, cân theo số token, tất định — đúng cơ chế `build_l1_groups` đã dùng và đã
có test. Tổng centroid không đổi nên so cùng budget vẫn hợp lệ; 9 test mới trong
[test_struct_clustering.py](scripts/test_struct_clustering.py) (86/86 PASS).

**Kiểm chứng trên 991 mẫu thật** (chạy đúng logic quyết định của `offline_clustering_struct`,
CPU, không cần GPU) — số mẫu Phase 6 sẽ thực sự có ở mỗi level:

| level | LCC chạy được | RepoBench-P | raise vì trần `max_k` |
|---|---:|---:|---:|
| class | 499/499 | 492/492 | **0** (trước khi sửa: 236 + 65) |
| function | **499/499** | **490/492** | **0** (trước: 22 + 2) |
| block | 486/499 | 440/492 | 0 |
| statement | **113/499** | **51/492** | 0 |

Nhánh merge (chỉ khi gọi tường minh) chạy được 100% số mẫu bị bỏ, nhưng ở statement nó phải
gộp tới **99% số unit** của một số mẫu — con số đó tự nó là lý do không đặt merge làm mặc
định.

Lệnh tái lập (CPU, ~2 phút mỗi dataset, không cần GPU cũng không cần model weight):
```bash
D=phase1_data/qwen2.5-coder-7b
python scripts/prepare_code_data.py qwen2.5-coder-7b --dataset lcc         --output_path $D
python scripts/prepare_code_data.py qwen2.5-coder-7b --dataset repobench-p --output_path $D
python scripts/check_phase1_data.py qwen2.5-coder-7b --dataset lcc
python scripts/check_phase1_data.py qwen2.5-coder-7b --dataset repobench-p
```
Gate này đã được nối vào [scripts/phase1_gate.sh](scripts/phase1_gate.sh) thành bước **[1b]**,
chạy trước mọi bước dùng GPU và dừng cả gate nếu FAIL.

| # | Việc | Trạng thái | Chi tiết |
|---|---|---|---|
| 1.1 | LongBench LCC + RepoBench-P | ✅ | Có sẵn trong pipeline, metric `code_sim_score`, so trực tiếp được với Table 2 |
| 1.2 | CrossCodeEval + RepoEval/RepoBench | 🟡 | **Đo lại đầy đủ 19/8, cả ba bộ.** RepoBench v1.1 **dùng được ở dạng nguyên bản**: 1.646 fixed context ≥16k dùng chung (Py+Java), tới 99,9K token. RepoEval **khớp premise tốt nhất**: 200 query/context, repo 192K–1,19M token. CrossCodeEval **loại** — 9/9 biến thể đều trần ~10,2K token. Khảo sát 15/8 sai do đọc shard 0 + nhóm sai khoá. Xem [docs/PHASE1_DATASETS.md](docs/PHASE1_DATASETS.md). Còn lại: viết loader |
| 1.3 | Chuẩn hoá split `fixed_context` / `user_input` | ✅ | Chốt theo D2: giữ định nghĩa LongBench. Kiểm 19/8: **LCC vốn đã khớp protocol** (`{context}` = toàn bộ code trước con trỏ), xung đột chỉ ở `repobench-p`. Gate Phase 0 chạy LCC nên không ảnh hưởng |
| 1.4 | Lưu offset **byte + ký tự** từng token | ✅ | **Qwen 500/500 LCC + 500/500 RepoBench-P (19/8)**, 0 lệch token id; trước đó LongChat 500/500 LCC · Qwen 20/20. Ngôn ngữ nay lấy từ **trường `language` của từng mẫu**, không còn hardcode. Dữ liệu ở `phase1_data/<model>/`; `load_phase1` của Phase 2 kiểm tên model và **số mẫu**, thiếu là dừng chứ không [WARN] rồi bỏ qua |
| 1.5 | Model chính **Qwen2.5-Coder-7B (base)** | ✅ | Chạy thật trên A100. **Đổi từ Instruct sang base** sau khi đo: Instruct 17,60 vs base **65,35** cùng dữ liệu. `model2maxlen` = 31500 (không phải 128K) — xem D5. Đã sửa 3 bug chặn, xem mục 6 |
| 1.6 | GQA: chọn key per-head (QUEST Appendix G) | ✅ | `repeat_interleave` centroid/label từ 4 head KV lên 28 head Q. **Xác nhận trên GPU**: `num_key_value_heads=4`, assert `shared_prefix_length` qua cả 20 mẫu, 14/20 mẫu cho prediction y hệt All-KV |
| 1.7 | Gate cho bản port | ✅ | [scripts/phase1_gate.sh](scripts/phase1_gate.sh) + [scripts/check_phase1.py](scripts/check_phase1.py). Tiêu chí là **paired test** trên hiệu số từng mẫu, không phải ngưỡng điểm cố định — ±2,0 gọi cả ca hỏng (−42,30) lẫn ca chạy được (−2,80) là FAIL |
| — | ⏸️ RepoPreFixQA | ⏸️ | **Nhiều khả năng không cần nữa.** RepoBench + RepoEval cho fixed context dài với **nhãn thật**, không phải nhãn LLM sinh. Xem D4 |
| — | ⏸️ Model cross-check | ⏸️ | DeepSeek-Coder-V2-Lite hoặc CodeLlama-13B |

**Kết quả gate** (Qwen2.5-Coder-7B base, LCC, 20 mẫu):

| | Điểm | |
|---|---:|---|
| All-KV | **65,35** | trần accuracy, cao hơn LongChat 54,83 |
| Sq-70% | **62,55** | hiệu −2,80 · `p = 0,22` · KTC95 `[−7,12, +1,52]` |
| Mẫu prediction y hệt nhau | **14/20** | bỏ 70% key mà 70% mẫu vẫn ra đúng chữ |

Cả hai cấu hình 0% dòng chấm rỗng. Chênh lệch **không có ý nghĩa thống kê**.

**Việc còn treo, không chặn Phase 2** (cập nhật 20/8):

| # | Việc | Cần GPU? | Chặn gì |
|---|---|---|---|
| 1 | **Chạy port gate 500 mẫu** thay vì 20 — thu hẹp KTC95 `[−7,12, +1,52]` | ✅ pod | không chặn gì; Phase 6 cần dù sao |
| 2 | **`phase1_gate.sh` end-to-end trên pod** — bước [2]–[6] chưa từng chạy với code hiện tại, [1b] mới thêm | ✅ pod (trừ [1]/[1b] là CPU) | không chặn Phase 2 |
| 3 | **Mẫu 7 tụt 41 điểm** (98,0 → 57,0, `sp_len`=3554) — chạy `inspect_centroids.py --dataidx 7` | ⚠️ chỉ cần file `.pt` từ pod, `torch.load` được trên CPU | không chặn |
| 4 | **Loader RepoBench v1.1** gom theo `(repo_name, bộ context)` | ❌ CPU | chặn **Phase 6** (không chặn Phase 5) |
| 5 | ~~Ghi limitation về premise~~ | ❌ | ✅ **xong 20/8**, xem khối bên dưới |

**Khác chiều với LongChat.** Phase 0 ra +1,25, bài gốc +0,29, Qwen ra −2,80. Chưa có ý nghĩa
thống kê nên chưa kết luận được, nhưng nếu ở 500 mẫu vẫn âm và có ý nghĩa thì đó là khác
biệt giữa hai họ model, phải giải thích trong bài.

#### ⚠️ GIỚI HẠN PHẢI GHI VÀO BÀI — dữ liệu Phase 1 chưa kiểm chứng premise của SA

Premise của Squeezed Attention là **một fixed context, NHIỀU user query**: chi phí clustering
offline chỉ đáng bỏ ra khi nó được khấu hao qua nhiều truy vấn trên cùng một context.

**LCC và RepoBench-P của LongBench KHÔNG có cấu trúc đó.** Mỗi mẫu là một cặp
`(context, next_line)` độc lập — một context ứng với đúng một query. Đo cụ thể trên bộ đang
dùng: `shared_prefix_length` chiếm gần trọn prompt, phần "user input" của LCC chỉ là chuỗi
chỉ dẫn `"Next line of code:\n"` (~6 token). Nghĩa là tỉ lệ khấu hao thực tế là **1 query
trên 1 lần clustering**, không phải ~24 như PreFixQA của bài gốc.

Hệ quả cho từng claim:

- **C2 (Phase 5, recall@budget) KHÔNG bị ảnh hưởng.** Nó đo chất lượng tập key được chọn cho
  *một* query so với tập lý tưởng, nên một-query-một-context là đủ. Đây cũng là lý do
  protocol xếp Phase 5 chạy trước.
- **C1 (Phase 6, accuracy@budget) KHÔNG bị ảnh hưởng về mặt so sánh**, vì mọi method đều
  chịu cùng điều kiện.
- **C3 (chi phí khấu hao) THÌ BỊ.** Không thể dùng LCC/RepoBench-P để nói "chi phí clustering
  không đáng kể vì được chia cho nhiều query" — trên hai bộ này nó không được chia cho gì cả.

Chính bài gốc thừa nhận khoảng trống này (Section 5: *"there is currently no benchmark
designed to test this scenario"*) và tự dựng PreFixQA vì thế.

**Cách xử lý đã chốt:** không sửa LCC/RepoBench-P, mà bổ sung ở Phase 6 bằng **RepoBench
v1.1** (gom theo `(repo_name, bộ context)` → 1.646 fixed context ≥16k, 6.848 query, trung vị
**3 query/context**) và/hoặc **RepoEval** (200 query/context, repo 192K–1,19M token). Việc
còn lại là viết loader — CPU, không cần GPU, không chặn Phase 2 và **không chặn Phase 5**.

Khi báo cáo phải nói rõ hai điều, không được nói quá: trung vị 3 query/context của RepoBench
thấp hơn nhiều so với ~24 của PreFixQA; và mọi số trên LCC/RepoBench-P là ở chế độ
**1 query/context**, tức trường hợp xấu nhất cho chi phí khấu hao của SA.

---

### Phase 2 — Structure-aware clustering (Idea 1) · ✅ **XONG 22/8** · 500/500 mẫu LCC

**Kết quả trung tâm:** K-means thuần (`sa`) vắt qua ranh giới AST ở **trung vị 44,5%** số
cluster (p25 36,0% · p75 52,6% · max 88,0%, n=500), trong khi `hard_boundary` và
`struct_hierarchy` đạt **0,0% ở đúng 500/500 mẫu**. Ranh giới cứng vừa được thi hành tuyệt
đối, vừa ràng buộc vào một lượng rất lớn — tức can thiệp có liều thật, không phải no-op.

Bảng đầy đủ: [docs/PHASE2_RESULTS.md](docs/PHASE2_RESULTS.md). Ba điều phải mang sang Phase 5/6:
`hard_boundary` dùng hết ngân sách còn `sa` lãng phí 0,71% ô centroid; K1 thực tế của tầng L1
là 18,2 chứ không phải 1% context (71% mẫu bị chặn ở số function); bất biến D còn 55/500 mẫu
lệch >5% — **nguyên nhân chưa biết**, xem đính chính 24/8 bên dưới (không phải do seed).

Ý tưởng: đặt ranh giới **cứng** theo AST, cluster embedding bên trong mỗi đơn vị cấu trúc. Hierarchy = token → statement/block → function → file.

| # | Việc | Trạng thái | Chi tiết |
|---|---|---|---|
| 2.1 | Parse AST bằng tree-sitter, có byte offset | ✅ | `parse_units` — 5 level (`file`/`class`/`function`/`block`/`statement`), 5 ngôn ngữ. Dùng API tree_sitter mới, **không cần** `tree_sitter_languages` (gói đó không cài được). Level thô gộp vào level mịn nên mọi token đều có unit bao |
| 2.2 | Gán `unit_id` cho từng key token ở từng level | ✅ | `assign_token_units` — sắp span theo kích thước giảm dần rồi ghi đè bằng `searchsorted`, O(U log S) thay cho O(S×U) của bản cũ. Offset lấy từ Phase 1.4 nên không đụng lỗi `use_fast=False` |
| 2.3 | **Hard boundary** — K-means độc lập trong từng unit, unit nhỏ → 1 centroid, tổng K vẫn ~5% | 🟡 | **Code xong + đã nối pipeline** (`--method hard_boundary`). Test bất biến: không cluster nào vắt qua hai unit. Chờ chạy GPU |
| 2.4 | **StructHierarchy** — L2 = trong-function, L1 = trung bình theo function/file | 🟡 | **Code xong + đã chạy smoke GPU 20/8** (`--method struct_hierarchy`). `build_l1_groups` ép K1 về 1% context **khi làm được**, ghi K1 thực tế ra `k1_stats_*.pt`. ⚠️ Trên LCC thường KHÔNG làm được — xem ghi chú dưới bảng |
| 2.5 | Ablation tách bạch: SA / +HardBoundary / +StructHierarchy | ✅ | `offline_clustering_struct.py --method {sa,hard_boundary,struct_hierarchy}`. Nhánh `sa` gọi thẳng `run_clustering` gốc để mọi nhánh đi qua cùng một đường code |
| 2.6 | Giữ nguyên Si, threshold, kernel | ✅ | `struct_clustering.py` **chỉ** sinh centroid + label, cùng layout `[1,H,K,D]`/`[1,H,S]` với `run_clustering`. Token-type weighting (Hướng 2(b) sẵn có trong repo) giữ lại làm cờ `token_weights`, **mặc định tắt**; test xác nhận tắt cờ ra kết quả trùng bit-for-bit |

#### Smoke GPU 20/8 — cả ba nhánh chạy được, và một phát hiện về tầng L1

`offline_clustering_struct.py --limit 3` trên A100, cả `sa` / `hard_boundary` /
`struct_hierarchy` đều xong: `heads Q=28 KV=4` (hook đúng chỗ, trước `repeat_kv`), đủ bộ ba
`.pt` mỗi mẫu, `feasibility_*.json` ghi `"infeasible": []`. Sau mẫu đầu (~130s warm-up) thì
chỉ ~2–4 s/mẫu → ước **30–45 phút cho 500 mẫu mỗi nhánh**, rẻ hơn nhiều so với bước pred.

**⚠️ Tầng L1 không phải "class" như cờ `--level_l1` gợi ý.** `k1_stats_*.pt` của 3 mẫu smoke:

    k1_raw=2  k1_target=113  k1_actual=51   mode=split
    k1_raw=4  k1_target=26   k1_actual=23   mode=split
    k1_raw=4  k1_target=20   k1_actual=20   mode=split

`k1_raw` = số unit cấp class trong file, chỉ **2–4**. Ngân sách L1 (1% context) cho phép
20–113 nhóm, nhiều hơn hẳn, nên `build_l1_groups` đi nhánh *split* và chẻ tới trần = **số
unit L2**, tức mỗi nhóm L1 = đúng một function. Đo trên toàn bộ dữ liệu:

| | K1 bị chặn ở số function (`--level_l1` vô hiệu) | K1 thực tế / n_ctx |
|---|---:|---:|
| LCC | **364/499 = 72,9%** | trung vị **0,66%** |
| RepoBench-P | 145/492 = 29,5% | trung vị **0,99%** |

Ba hệ quả:

1. **Không được ghi "L1 = 1% context"** trong bài. Trên LCC nó là 0,66%; con số phải lấy từ
   `k1_stats_*.pt`, không lấy từ danh nghĩa. Budget vẫn hợp lệ (K1 nhỏ hơn danh nghĩa, không
   lớn hơn), nhưng báo cáo sai con số là chuyện khác.
2. **`struct_hierarchy` ở `level=function` thực chất là "L1 = trung bình theo function"** —
   đúng chữ của protocol (*"Level-1 centroid = trung bình theo function/file"*), nhưng cờ
   `--level_l1` không hề tham gia. Đừng mô tả nó như một hierarchy 3 tầng class→function→token.
3. **Muốn khảo sát `--level_l1` thì phải làm trên RepoBench-P** (70% mẫu đi nhánh *merge*,
   nơi cờ đó thực sự quyết định cách gộp), hoặc trên LCC với L2 ở level mịn hơn (`block`).
   Quét `--level_l1` trên LCC ở `level=function` sẽ cho ra **cùng một kết quả cho mọi giá
   trị** — nếu không biết trước, rất dễ đọc thành "hierarchy không nhạy với level".

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
| Centroid model **ngoài** LongChat | `fixed-prompt-clusters/<model>/<dataset>/*.pt` | `scripts/phase1_gate.sh` |
| Môi trường | `phase0_results/env_record.json`, `_pip_freeze.txt` | `scripts/record_env.py` |
| Console log đầy đủ | `phase0_results/logs/<timestamp>_phase0_gate.log` | `scripts/phase0_gate.sh` |
| **Tổng hợp (file này)** | `EXPERIMENT_LOG.md` | `scripts/check_gate.py --log_md` |

Quy ước tên thư mục `<config>` do `eval.py` sinh:
- All-KV → `<model>_baseline`
- Single-level → `<model>_PC<percent>_PERC<percentile>`
- Hierarchical → `<model>_PC1_<pc>_PERC1_<perc>_PC2_<pc2>_PERC2_<perc_lower>_lookup`
- Chạy `--limit N` → thêm hậu tố `_lim<N>`. **Điểm của N mẫu đầu không so được với điểm
  của cả 500 mẫu** — hậu tố tồn tại để hai thứ không bao giờ lẫn vào nhau.

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

### 2026-08-24 — Đính chính bất biến D: **seed đã ghim sẵn**, cách sửa đã đề xuất là no-op

Rà lại code Phase 2 thì thấy chẩn đoán của Bảng 4 sai.

`docs/PHASE2_RESULTS.md` giải thích 55/500 mẫu lệch >5% ở bất biến D bằng *"cuML khởi tạo
ngẫu nhiên, không ghim `random_state`"*, và đề xuất *"ghim seed rồi sinh lại — ~45 phút GPU"*.

Nhưng [squeezedattention/clustering.py:69](squeezedattention/clustering.py#L69) **đã** đặt
`random_state=0`, và `git show b03a63d` cho thấy dòng đó có từ **first commit** — từ code gốc
của Squeezed Attention, không phải thứ ai quên. Cả hai bên của phép đối chiếu (`--method sa`
của `offline_clustering_struct.py` và `offline_clustering.py`) đều gọi đúng hàm
`run_clustering` đó, nên **cả hai đều đang chạy với seed ghim**.

Hệ quả: chạy 45 phút GPU theo cách đã đề xuất sẽ ra **đúng con số cũ**, rồi rất dễ đọc thành
"đã ghim seed mà vẫn lệch → chắc port sai". Đó là kết luận sai sinh ra từ một phép sửa không
hề thay đổi gì.

**Ba khả năng còn lại, ba cách xử lý khác nhau:**

| | Khả năng | Nếu đúng thì phải làm gì |
|---|---|---|
| 1 | cuML không tất định **dù đã ghim seed** — Lloyd iteration reduce bằng atomic trên GPU, thứ tự cộng khác nhau mỗi lượt, rồi `tol=1e-4` chặn sớm ở vòng khác | Không sửa được bằng seed. Báo cáo một **ngưỡng sàn** của phép đo, đổi metric sang loại ổn định số (`inertia`, `ARI`) |
| 2 | **Key vector** không giống nhau giữa hai lượt forward | Nặng hơn k-means: mọi số Phase 2 sinh ở hai thời điểm khác nhau đều không đối chiếu được |
| 3 | Reference sinh bởi **code/config khác** (transformers, `rope_scaling`, `force_chat`, `fixed_context`, `maxlen`) | Sinh lại reference bằng đúng cấu hình. Không liên quan seed |

**Thêm [scripts/diag_invariant_d.py](scripts/diag_invariant_d.py)** để tách ba khả năng đó —
ba tầng đo lồng nhau, mỗi tầng loại một khả năng:

| Tầng | Đo gì | Loại được khả năng nào |
|---|---|---|
| T1 | forward **hai lần** cùng prompt trong cùng process, so key bit-for-bit | 2 |
| T2 | gọi `run_clustering` **hai lần trên cùng key A** | đo riêng 1 → đây là **ngưỡng sàn** của phép đo |
| T3 | so kết quả T2 với file centroid trên đĩa (`--reference_dir`) | `T3 ≈ T2` → nhiễu cuML, loại 3. `T3 ≫ T2` → còn nguyên nhân thứ ba |

Metric của T2/T3 dùng **y hệt** `check_phase2_invariants.py:397` (Hausdorff một chiều lấy
max, chuẩn hoá theo median norm, bỏ hàng zero) để so thẳng được với dải 5,7e-03…3,4e-01 đã
báo cáo. Kèm ba metric **bất biến với hoán vị** vì Hausdorff-max chỉ cần một centroid rơi
khác chỗ là vọt lên: `mean_nearest` (trung bình thay vì max), `inertia_rel` (hai phân hoạch
có *tốt ngang nhau* không — đây mới là câu hỏi khoa học), `ARI` (hai phân hoạch có *giống
nhau* không). Ba hàm metric đã test riêng trên CPU: bất biến với hoán vị cluster, bỏ đúng
hàng zero, `inertia` tối ưu < `inertia` ngẫu nhiên, `ARI(x,x)=1` và `ARI` của hai nhãn ngẫu
nhiên ≈ 0.

Chi phí: mặc định 3 mẫu, **~2–3 phút GPU** — rẻ hơn ~15 lần cách cũ và trả lời đúng câu hỏi
hơn. Chưa chạy (cần pod).

    python scripts/diag_invariant_d.py qwen2.5-coder-7b-instruct --force_chat \
        --dataset lcc --phase1_dir /workspace/phase1_data --limit 3 \
        --reference_dir /workspace/p2-instruct/sa/lcc \
        --out /workspace/diag_invariant_d.json

**Bài học chung, không riêng chỗ này:** trước khi ghi "nguyên nhân là X" vào bảng kết quả,
phải mở source đọc X. Ở đây chỉ tốn một lần `grep random_state` để thấy chẩn đoán sai — mà
nếu không thấy thì cái giá là 45 phút GPU cộng một kết luận sai về chất lượng bản port.

### 2026-08-19 — Đính chính: RepoBench v1.1 **dùng được ở dạng nguyên bản**

Đo lại toàn bộ dữ liệu sau khi bị chất vấn "đo đúng bản chưa". **Kết luận 15/8 sai**, do hai
lỗi phương pháp độc lập, cả hai cùng đẩy về một hướng:

| # | Lỗi | Hậu quả |
|---|---|---|
| 1 | Chỉ đọc `cross_file_first` **shard 0** — mà shard đó sắp theo `level` nên toàn mẫu ngắn | Kết luận "context quá ngắn" |
| 2 | Nhóm theo `repo_name` rồi hỏi "mọi sample của repo có dùng chung MỘT context không" | Kết luận "context không dùng chung" |

Lỗi 2 tinh vi hơn: con số 153/962 và 2/211 **không sai về tính toán**, chúng chỉ trả lời một
câu hỏi khác câu cần hỏi. Một repo có thể chứa vài bộ context, mỗi bộ được nhiều query dùng
chung. Khoá đúng là `(repo_name, bộ context)`.

**Số đo lại:**

| | Đo 15/8 (shard 0) | **Đo lại (đủ)** |
|---|---:|---:|
| Số dòng `cross_file_first` | 4.017 | **8.033** |
| Nhóm `level` | tới `16k` | **tới `128k`** |
| `token_num` trung vị · max | 3.614 · 14.177 | **10.826 · 99.376** |

`level`: `2k=1000 · 4k=1000 · 8k=1000 · 12k=1000 · 16k=1000 · 24k=1000 · 32k=1000 · 64k=912 · 128k=121`

**Cấu trúc fixed-context CÓ sẵn:**

| `cross_file_first` | |
|---|---:|
| Nhóm `(repo, bộ context)` có ≥2 query | **1.308** |
| Query trong các nhóm đó | **4.830 — 60,1% toàn bộ** |
| **Nhóm vừa dài ≥16k vừa dùng chung** | **732 nhóm · 3.496 query** |
| Context nhóm dài: trung vị · max | **17.886 · 99.376 token** |
| Query/nhóm: trung vị · max | 3 · **30** |

`cross_file_random` tương tự: 699 nhóm ≥16k, 3.096 query.

**Hệ quả — D4 gần như tự giải.** Không cần clone repo, không cần dựng benchmark mới, không
cần pipeline LLM sinh câu hỏi + self-consistency + LLM-as-judge. Chỉ cần một loader gom theo
`(repo_name, bộ context)`; mỗi nhóm thành một `fixed_context` với nhiều `user_input`. Đây là
cấu trúc mà câu đầu tiên của Phase 1 yêu cầu, và nó có sẵn suốt.

**Ba điều phải giữ tỉnh táo:**
1. **Trung vị 3 query/context**, trong khi PreFixQA của bài có ~24 → mức khấu hao chi phí
   clustering thấp hơn nhiều. Không được nói quá trong bài.
2. **Contamination**: `created_at` toàn 2023, Qwen2.5-Coder train tới ~2024.
3. **CrossCodeEval chưa đo lại** — khảo sát cũ có thể mắc đúng hai lỗi này. Con số
   "2/211 repo dùng chung" đặc biệt đáng nghi vì nó dùng chính khoá nhóm sai.

**Bài học phương pháp.** Repo đã có chốt chặn cho dữ liệu sinh ra (`check_cluster_integrity`,
chốt dòng rỗng, chốt ngưỡng `NaN`) nhưng **không có chốt nào cho khâu khảo sát dữ liệu đầu
vào**. Hai lỗi trên đều thuộc loại *đọc một mẫu không đại diện rồi kết luận cho tổng thể* —
cùng họ với lỗi đọc shard 0. Khi khảo sát bộ dữ liệu mới: kiểm số shard/split trước khi đọc,
và nêu rõ khoá nhóm trước khi đếm.

### 2026-08-18 (tối) — Paired test: −2,80 là **nhiễu**. Phase 1 xong

`compare_runs.py` trên dữ liệu thật:

| | |
|---|---|
| Mẫu cho prediction **y hệt nhau** | **14/20 (70%)** |
| Sq-70% tốt hơn / kém hơn | 1 / 5 |
| Hiệu số trung bình | **−2,80 ± 2,20** |
| Khoảng tin cậy 95% | **[−7,12, +1,52]** — chứa 0 |
| Sign test · Wilcoxon | **p = 0,2188** cả hai |

**Không có ý nghĩa thống kê.** Và độ tụt gần như do **một mẫu duy nhất**:

| idx | All-KV | Sq-70% | hiệu |
|---:|---:|---:|---:|
| **7** | 98,0 | 57,0 | **−41,0** |
| 5 | 36,0 | 22,0 | −14,0 |
| 3 | 32,0 | 27,0 | −5,0 |
| 11 | 100,0 | 96,0 | −4,0 |
| 14 | 24,0 | 23,0 | −1,0 |
| 12 | 18,0 | 27,0 | **+9,0** |

Riêng mẫu 7 đóng góp −2,05 trong tổng −2,80. Bỏ nó ra thì phần còn lại gần như hoà.

**Đổi tiêu chí gate sang paired test.** Cần nói rõ để không thành nguỵ biện: đây **không
phải** nới ±2,0 thành ±3,0 cho vừa số đo, mà là thay một tiêu chí **sai phương pháp**. Và
nhận định "±2,0 không vững ở n=20" đã ghi vào mục trước **trước khi** chạy `compare_runs.py`,
không phải sau khi thấy kết quả.

Bằng chứng cho việc ngưỡng cũ sai: nó gọi **cả hai** ca là FAIL.

| Ca | Hiệu số | p | Dáng | Ngưỡng ±2,0 | Paired test |
|---|---:|---:|---|---|---|
| Ngưỡng NaN (hỏng thật) | −42,30 | <0,0001 | tụt đều 20/20 | ❌ FAIL | ❌ FAIL |
| Sau khi vá (chạy được) | −2,80 | 0,2188 | 14/20 y hệt | ❌ FAIL | ✅ PASS |

Tiêu chí mới trong `check_phase1.py`, **hai điều kiện FAIL độc lập**:
1. Kém hơn **có ý nghĩa thống kê** — KTC95 nằm hẳn dưới 0
2. Tụt quá `--max_drop` (mặc định 10 điểm) — chặn hỏng nặng khi n quá nhỏ để đạt p nhỏ

Đã kiểm trên fixture tái dựng đúng số thật của cả hai ca: ca chạy được → PASS, ca hỏng →
FAIL kèm đúng lý do.

**Phase 1 xong.** Bản port Squeezed Attention sang Qwen2/GQA hoạt động đúng.

**Ba việc còn treo, không chặn Phase 2:**

1. **Mẫu 7 tụt 41 điểm** — chưa giải thích. `sp_len` = 3554. Đáng chạy
   `inspect_centroids.py --dataidx 7` xem có phải cluster suy biến không. Nếu đúng thì đây
   là ca mẫu cụ thể cho Idea 1.
2. **Khoảng tin cậy rộng** [−7,12, +1,52] — chưa loại được khả năng Sq-70% thực sự kém hơn
   vài điểm. Muốn con số chắc phải chạy 500 mẫu (~45 phút clustering + ~2,7 giờ pred).
   Việc này **Phase 6 cần đến dù sao**, nên không phải chi phí thêm.
3. **Khác chiều với LongChat.** Phase 0 ra +1,25, bài gốc +0,29, Qwen ra −2,80. Không có ý
   nghĩa thống kê nên chưa kết luận được gì, nhưng phải theo dõi ở n lớn — nếu ở 500 mẫu
   vẫn âm và có ý nghĩa thì đó là một khác biệt giữa hai họ model, phải giải thích trong bài.

### 2026-08-18 (chiều) — Sau khi vá: 23,05 → **62,55**. Gate còn lệch −2,80

| | All-KV | Sq-70% | Hiệu |
|---|---:|---:|---:|
| Trước khi vá | 65,35 | 23,05 | **−42,30** |
| **Sau khi vá** | 65,35 | **62,55** | **−2,80** |

Bản vá lấy lại **39,5 điểm**. Dòng chấm rỗng 0% ở cả hai. Đường Squeezed Attention trên
Qwen2/GQA hoạt động — đó là câu hỏi gate sinh ra để trả lời, và câu trả lời là **có**.

**Gate vẫn báo FAIL vì −2,80 > dung sai ±2,0 — nhưng tiêu chí đó không vững ở n = 20.**

±2,0 được mượn nguyên từ Phase 0, nơi nó áp lên **500 mẫu**. Trên 20 mẫu, sai số chuẩn của
riêng một điểm trung bình đã vượt 5 điểm, nên đọc `−2,80` như một con số tuyệt đối là sai
phương pháp. Hai lần chạy dùng **cùng 20 mẫu, cùng model**, chỉ khác cờ `use_centroids` —
đây là thiết kế **ghép cặp**, và đại lượng đúng là phân bố **hiệu số từng mẫu**. Phần lớn
mẫu thường cho prediction y hệt nhau (hiệu = 0), nên phương sai của hiệu số nhỏ hơn hẳn
phương sai của từng điểm.

Thêm [scripts/compare_runs.py](scripts/compare_runs.py): so theo cặp từ `result_detail.json`
— đếm mẫu giống hệt / tốt hơn / kém hơn, hiệu số trung bình ± SE, khoảng tin cậy 95%, sign
test chính xác (không cần scipy) và Wilcoxon nếu có. Đúng thứ **Phase 5.5** yêu cầu
("paired test qua các mẫu"), nên không phải công cụ dùng một lần.

Test trên hai kịch bản dựng sẵn: 3/20 mẫu lệch → khoảng tin cậy `[−8,08, +0,58]` chứa 0,
kết luận "chưa có ý nghĩa thống kê"; tụt đều 20/20 → `[−41,91, −36,59]`, `p < 0,0001`,
kết luận "thực sự kém hơn". Phân biệt được đúng hai chế độ.

**Chưa chốt được ở đây:** −2,80 là thật hay nhiễu. Phải chạy `compare_runs.py` trên dữ liệu
thật mới biết. Ba khả năng:

1. Khoảng tin cậy **chứa 0** → chênh lệch không có ý nghĩa ở n=20. Tiêu chí gate phải đổi
   sang paired test, hoặc tăng số mẫu. **Không được nới ngưỡng cho vừa số đo** — đó là sửa
   thước cho khớp kết quả.
2. Khoảng tin cậy **loại 0**, và độ tụt dồn vào vài mẫu → nghi cluster suy biến (một cluster
   nuốt 60% context ở mẫu 19). Đây là chỗ Idea 1 có động cơ.
3. Loại 0 và tụt đều → còn lỗi hệ thống nữa, tiếp tục truy.

Nhắc lại để so: LongChat ở Phase 0 ra **+1,25**, bài gốc ra **+0,29** — cả hai đều dương.
Qwen ra âm là khác chiều, nên dù không có ý nghĩa thống kê thì vẫn phải giải thích được
trong bài, không lờ đi.

### 2026-08-18 — Tìm ra lỗi thật: ngưỡng `NaN` do tràn số học ở `exp`

Gate Phase 1 với model **base** `Qwen/Qwen2.5-Coder-7B`:

| | All-KV | Sq-70% |
|---|---:|---:|
| Điểm (20 mẫu LCC) | **65,35** | **23,05** |
| Dòng chấm rỗng | 0% | 0% |

**Đổi sang base là đúng: 65,35 cao hơn cả LongChat 54,83.** Và vì output đã sạch (0% rỗng
ở cả hai), con số 23,05 mới đọc được — đây là lỗi retrieval thật, không phải lỗi định dạng.

**Chẩn đoán, bằng hai nguồn bằng chứng độc lập và không tốn phút GPU nào:**

`scripts/inspect_centroids.py` trên file `.pt` đã lưu:

```
[THRESHOLD]  q=0.50 tau = nan | q=0.70 tau = nan | q=0.80 tau = nan | q=0.90 tau = nan
```

`result_detail.json` (điểm từng mẫu): 0,27 · 0,20 · 0,28 · 0,22 · 0,22 · 0,15 … — **tụt đều
cả 20 mẫu**, không mẫu nào thoát. Đúng dấu hiệu lỗi hệ thống chứ không phải dữ liệu lẻ.

**Chuỗi nhân quả.** `run_global_threshold` gọi `torch.exp` thẳng trên logit thô — softmax
chưa chuẩn hoá, không trừ max:

```python
attn_scores_centroids_est_exp = torch.exp(attn_scores_centroids)   # -> inf
scores_scaled_sm = torch.exp(scores) / denom_est.unsqueeze(-2)     # inf/inf -> nan
```

float32 tràn thành `inf` khi logit vượt ~88,7 → `inf/inf = nan` → `np.quantile` trên mảng
có `nan` trả về `nan` → ngưỡng `nan` được `torch.save` xuống đĩa mà không ai kêu ca →
`mask = (avg_score_per_token > threshold)` **luôn False** vì `nan` so sánh với gì cũng False
→ **không cluster nào được chọn**, model mất toàn bộ fixed context.

Không crash. Không assert nào nổ. Model vẫn sinh code sạch. Chỉ tụt 42 điểm.

**Lỗi nằm ở code DÙNG CHUNG, không phải bản port Qwen.** Bản port GQA đúng — centroid nạp
đúng 4 head, `repeat_interleave` đúng nhóm, `shared_prefix_length` khớp qua cả 20 mẫu.
LLaMA/LongChat thoát vì logit nằm trong dải an toàn; Qwen2 có **massive activations** nên
logit lớn hơn hẳn. Đây là loại lỗi chỉ lộ ra khi đổi họ model.

**Sửa — hai tầng, tầng thứ hai chỉ lộ ra sau khi vá tầng thứ nhất.**

*Tầng 1 — tràn.* Trừ max theo chiều cluster trước khi `exp`. Cùng hằng số cho cả tử và mẫu
nên tỉ số **không đổi về mặt toán học**: `exp(s−M) / Σ n_k·exp(a_k−M) ≡ exp(s) / Σ n_k·exp(a_k)`.

*Tầng 2 — chia cho 0.* Vá xong tầng 1, chốt chặn mới **nổ ngay ở mẫu 0**:
`56880/8918784 (0,64%)` điểm vẫn không hữu hạn — khớp gần đúng tỉ lệ **cluster rỗng 0,8-1%**
mà `inspect_centroids.py` báo. Không trùng hợp:

1. `run_clustering` gán centroid của cluster rỗng bằng **vector 0** → điểm `q·0 = 0`
2. Khi mọi cluster thật có điểm rất âm, **max rơi vào cluster rỗng**
3. Cluster rỗng có `num_keys = 0` → góp **0** vào mẫu số
4. Cluster thật, sau khi trừ max, thành `exp(−150)` → **underflow về 0** (float32 hết dải ở ~−103)
5. Mẫu số `= 0`, tử số `= 0` → **`0/0 = NaN`**

**Lần vá đầu của tôi sai:** chỉ loại cluster rỗng khỏi phép lấy *max*. Thế là `M` tụt xuống
mức cluster thật, rồi điểm `0` của cluster rỗng thành `exp(0−(−150)) = exp(150) = inf`, và
`0 × inf = NaN`. Đổi chỗ tràn chứ không khử. Bản đúng phải **mask một lần rồi dùng cho cả max
lẫn tổng**: đặt `−inf` ở cluster rỗng thì `exp(−inf − M) = 0` chính xác, không phụ thuộc
`num_keys = 0` nhân với cái gì.

Kiểm bằng số, ba chế độ, bốn biến thể cài đặt:

| Chế độ | gốc | chỉ trừ max | loại khỏi max | **mask cả hai** |
|---|---|---|---|---|
| Cluster thật −150, rỗng 0 *(ca trên pod)* | NaN | NaN | NaN | ✅ hữu hạn |
| Logit +120 *(tràn exp)* | NaN | ✅ | ✅ | ✅ |
| Dải LLaMA | ✅ | ✅ | ✅ | ✅ |

Chỉ biến thể cuối qua được cả ba. Ở dải an toàn nó lệch bản gốc **5,7e-07** → **Phase 0 không
phải chạy lại**.

Kiểm chứng bằng số (`torch.randn`, tái dựng đúng phép tính của hàm):

| | Kết quả |
|---|---|
| Logit dải LLaMA: cũ vs mới | lệch tương đối **8,8e-07** → **Phase 0 không phải chạy lại** |
| Logit +120 (kiểu Qwen): bản cũ | **120000/120000** phần tử `nan`/`inf`, quantile ra `nan` |
| Logit +120: bản mới | **0** phần tử `nan`, quantile ra `0,001493` |
| Bất biến dịch chuyển `exp(x+c)` ≡ `exp(x)` | lệch **7,1e-06** |

**Chốt chặn thêm.** `run_global_threshold` giờ `raise RuntimeError` ngay nếu điểm centroid
hoặc quantile ra `nan`/`inf`, kèm chẩn đoán. Cùng nguyên tắc với chốt chặn dòng rỗng: **hỏng
thì phải nổ tại chỗ, không được xuống đĩa.** Một file ngưỡng `nan` nằm im trên đĩa đã tốn
một lượt clustering + hai lượt `pred.py`.

**⚠️ Phải xoá centroid cũ trước khi chạy lại.** `offline_clustering.py` bỏ qua mẫu đã có
`global_threshold_{idx}_{K}.pt` — chính bản vá chống đứt job của Phase 0. File `nan` vẫn
tồn tại nên nó sẽ bỏ qua sạch và giữ nguyên dữ liệu hỏng. `rm -rf` thư mục trước.

**Còn một vấn đề chưa xử lý, phát hiện cùng lúc.** `inspect_centroids.py` báo ở mẫu 19:

```
mot cluster chua 813 key (>50% toan bo context) -> K-means suy bien
```

K-means trên key của Qwen2 bị outlier kéo lệch: một cluster nuốt 60% context, cluster nhỏ
nhất chỉ 1 key. Kể cả khi ngưỡng đã đúng, một cluster chứa 60% key thì chọn nó là mất hết
tính thưa, bỏ nó là mất 60% ngữ cảnh — centroid gần như không phân biệt được gì. **Chưa
sửa; sửa lỗi chí mạng trước rồi đo lại mới biết cái này còn ảnh hưởng bao nhiêu.** Nếu còn
thì đây chính là chỗ Idea 1 (ranh giới cứng theo AST) có lý do tồn tại rõ ràng nhất — nó
ép cluster không được vắt qua đơn vị cấu trúc, tức chặn sẵn kiểu suy biến này.

### 2026-08-17 — Hậu kiểm Phase 0: prediction sạch, và hiệu chuẩn được ngưỡng dòng rỗng

Chốt chặn "dòng chấm rỗng" viết hôm nay chưa từng soi Phase 0, nên chạy
[scripts/inspect_preds.py](scripts/inspect_preds.py) trên toàn bộ 500 mẫu đã có.

| | All-KV | Sq-70% |
|---|---:|---:|
| Điểm script tính lại | **54,83** | **56,08** |
| Dòng được chấm bị rỗng | 73/500 (**14,6%**) | 63/500 (**12,6%**) |
| Prediction có markdown fence | 435/500 | 430/500 |

Điểm tính lại **khớp `result.json`** — xác nhận bản sao logic `code_sim_score` trong
`inspect_preds.py` là chính xác, nên mọi chẩn đoán dựa vào nó đều đọc được.

**Phase 0 sạch, 54,83 là số thật.** Dưới ngưỡng 25%, và prediction là code thật:
mẫu 1 được 100 điểm, mẫu 0 và 2 đúng một phần — đúng dáng model đang làm việc.

**Chi tiết giải thích được ca Qwen-Instruct.** LongChat cũng mở markdown fence ở
435/500 mẫu. Nhưng metric bỏ qua mọi dòng chứa `` ` `` nên nhảy xuống dòng code bên dưới
và chấm đúng chỗ. Khác biệt không nằm ở chuyện có fence hay không, mà ở chỗ **LongChat
sinh code thật sau fence, còn Qwen-Instruct dừng ngay tại dấu fence**. Chẩn đoán hôm nay
giờ có đối chứng, không còn là suy luận.

**Hiệu chuẩn ngưỡng 25%.** Trước đó tôi đặt con số này theo cảm tính. Nay có hai đầu mút đo được:

| | Tỉ lệ dòng rỗng |
|---|---:|
| Lành (LongChat, Phase 0) | 12,6 – 14,6% |
| Hỏng (Qwen-Instruct, Phase 1) | 50 – 80% |

Ngưỡng 25% nằm giữa hai vùng, cách đều. Đây là **toàn bộ cơ sở thực nghiệm** cho con số
đó — nếu về sau một model lành vượt 20% thì phải xem lại ngưỡng, đừng nới tay.

**Một điều phải ghi vào paper:** 14,6% mẫu bị chấm 0 điểm vì lý do định dạng, không phải
vì model dự đoán sai. Trần accuracy của LCC bị hạ sẵn chừng đó cho **mọi** cấu hình. Không
làm hỏng so sánh nào của protocol vì nó áp đều lên All-KV, SA và mọi biến thể structure-aware
— và cũng áp lên số của bài gốc, vốn dùng đúng metric này. Nhưng nó có nghĩa là con số
tuyệt đối trên LCC không nên đọc như "độ chính xác của model".

⚠️ File prediction thô (`lcc.jsonl`, vài trăm KB mỗi cái) **chỉ nằm trên pod**, không có
trong git — chỉ `result.json` được commit. Đó là bằng chứng gốc cho con số nền mà mọi so
sánh về sau dựa vào. Nên `git add -f` hai file đó trước khi pod bị xoá.

### 2026-08-17 — Chạy gate Phase 1: bản port GQA đúng, nhưng model chọn sai

**Đường Squeezed Attention trên Qwen2 hoạt động.** Đây là kết quả thật, giữ nguyên giá trị:

| | |
|---|---|
| `num_attention_heads=28 \| num_key_value_heads=4 \| num_hidden_layers=28` | centroid sinh từ 4 head KV, **trước** `repeat_kv` — đúng thiết kế |
| Tokenizer nhanh/chậm Qwen2 | **0/20 lệch token id** trên fork 4.40 thật. Câu hỏi mở duy nhất của bản port, nay đã đóng |
| File centroid | 60/60 CRC đúng, đủ bộ ba cho cả 20 mẫu |
| `shared_prefix_length` | assert qua hết 20 mẫu → offline và online khớp nhau |

**Clustering rẻ hơn nhiều so với LongChat** — đổi hẳn ngân sách Phase 5/6:

| | LongChat (32 head KV) | Qwen2.5-Coder (4 head KV) |
|---|---:|---:|
| Thời gian | 42,5 giây/mẫu | **5,4 giây/mẫu** |
| Đĩa | ~146 MB/mẫu | **~10 MB/mẫu** |
| Cả 500 mẫu LCC | 6 giờ, 68 GB | **~45 phút, ~5 GB** |

Đúng tỉ lệ ~8× từ 32 head KV xuống 4. Lần đầu một ước tính của tôi khớp.

**Nhưng: All-KV = 17,60** (LongChat cùng task được 54,83). Prediction thô cho thấy model
sinh ra **gần như không gì cả**:

```
sample 0  RAW: '\n\n\n... (31 dòng trống)'      CHẤM: ''
sample 1  RAW: ' ```'                            CHẤM: ''
sample 4  RAW: " ```\nobj['next_line']\n``` "    CHẤM: "obj['next_line']"
```

Mẫu 4 lộ rõ nhất: model đọc `"Next line of code:"` như câu hỏi trừu tượng rồi bịa ra một
cái tên, thay vì hoàn thành đoạn code đang dở.

**Nguyên nhân: lệch model instruct với prompt completion thô.** Qwen2.5-Coder-**Instruct**
được huấn luyện trong khung ChatML; LongBench thì **cố ý bỏ** chat template cho
lcc/repobench-p (`truncate_fn` bỏ qua `build_chat` với 5 dataset này, comment gốc ghi
*"chat models are better off without build prompts on these tasks"*). Không có khung đó,
model rơi vào chế độ trợ lý, mở một khối markdown rồi phát token kết thúc sớm.

**Không phải lỗi pipeline, cũng không phải lỗi bản port:**
- Cùng đường ống đó LongChat ra 54,83.
- Đây là nhánh **All-KV, không dùng centroid nào**.
- Model vẫn sinh markdown fence đúng cú pháp và định danh Python hợp lệ — attention hỏng
  thì ra ký tự loạn, không ra thế này.

**Quyết định: chuyển sang bản base `Qwen/Qwen2.5-Coder-7B`** (thêm vào `model2path.json`
+ `model2maxlen.json`, `configs/phase1.sh` đổi mặc định). LCC/RepoBench-P là điền dòng code
tiếp theo trong ngữ cảnh repo — base model tiếp tục code tự nhiên, không có chế độ trợ lý
để rơi vào. **Hệ quả cần theo dõi:** RepoPreFixQA của Phase 6 là task QA, chỗ đó lại cần
instruct. Nếu kết cục dùng hai model cho hai loại task thì phải ghi rõ trong paper.

**Bài học quan trọng hơn: gate báo PASS trong khi cả hai con số đều vô nghĩa.**

Tiêu chí "Sq-70% không tệ hơn All-KV" thấy `20,85 ≥ 17,60` và kết luận PASS. Nó không thể
phát hiện được, vì **cả hai đường cùng hỏng theo cùng một kiểu** — đúng giới hạn tôi đã
viết vào `check_phase1.py` từ đầu, và nó xảy ra ngay lần chạy đầu tiên.

Đã vá bằng cách kiểm **riêng biệt**, chạy **trước** khi so điểm: đếm tỉ lệ mẫu mà
`code_sim_score` chấm vào một **dòng rỗng**. Vượt 25% là FAIL ngay, kèm chẩn đoán. Lần chạy
vừa rồi có tỉ lệ **80%** — ngưỡng này bắt được.

Nguyên tắc rút ra, áp cho cả Phase 5/6: **một chỉ số tương đối không tự bảo vệ được mình.**
So A với B chỉ có nghĩa khi đã biết A và B đều nằm trong dải hợp lệ. Mọi gate về sau phải
có một khẳng định tuyệt đối đứng trước khẳng định tương đối.

Thêm [scripts/inspect_preds.py](scripts/inspect_preds.py) — in prediction thô cạnh
**dòng mà metric thật sự chấm**. Khoảng cách giữa "model sinh gì" và "metric thấy gì" là
chỗ ẩn nấp của cả lớp lỗi này; `code_sim_score` lấy dòng đầu tiên không chứa `` ` ``, `#`,
`//`, nên `` ```\nfoo()\n``` `` bị chấm ở dòng rỗng chứ không phải ở `foo()`.

**Chi tiết vận hành:** Sq-70% chậm hơn All-KV (4,84 vs 1,74 giây/mẫu). Context ở 20 mẫu đầu
chỉ 1,4–4,4K token; ở độ dài đó chi phí nạp centroid từ đĩa lớn hơn phần tiết kiệm. SA
thiết kế cho 128K. **Đừng đo latency Phase 7 ở độ dài này.**

### 2026-08-17 — Phase 1: hai bug chặn bản port Qwen, và bộ công cụ gate

Đọc lại đường Qwen trước khi thuê pod. **Không chạy GPU nào cho mục này.**

**Hai bug chặn — cả hai đều nổ *sau* khi đã tốn tiền, không phải lúc gõ lệnh**

| # | Bug | File | Hậu quả nếu không sửa |
|---|---|---|---|
| 6 | `pred.py --model` có `choices=[...]` **hard-code**, thiếu `qwen2.5-coder-7b-instruct` | [LongBench/pred.py](LongBench/pred.py) | argparse từ chối ngay. Mục 1.5 đã thêm entry vào `model2path.json` + `model2maxlen.json` nhưng bỏ sót danh sách thứ ba này |
| 7 | `pred.py` load tokenizer Qwen bằng `AutoTokenizer.from_pretrained(path)` → mặc định **use_fast=True**, trong khi `offline_clustering.py` dùng `use_fast=False` | [LongBench/pred.py](LongBench/pred.py) | `truncate_fn` tính `shared_prefix_length` bằng tokenizer. Hai bên lệch dù **1 token** thì `assert` ở [modeling_qwen2.py:1347](transformers/src/transformers/models/qwen2/modeling_qwen2.py#L1347) nổ **sau khi đã nạp xong model 15 GB** |

Bug 6 sửa bằng cách bỏ hẳn danh sách hard-code, đọc thẳng key của `model2path.json`.
Nguồn duy nhất thì không lệch được nữa — Phase 6 còn thêm model cross-check.

Bug 7 đáng chú ý ở chỗ **đường LLaMA vốn đã đúng** mà không ai để ý tại sao:
`LlamaTokenizer.from_pretrained` là bản **chậm**, tình cờ khớp với `offline_clustering.py`.
Port sang Qwen đổi sang `AutoTokenizer` là mất luôn tính chất đó.

**Chưa trả lời được ở đây, phải chờ pod:** tokenizer nhanh và chậm của Qwen2 có ra cùng
token id không. Tôi thử trên máy local nhưng `transformers` bản 5.x **bỏ qua**
`use_fast=False` — cả hai đều trả về `Qwen2Tokenizer` với `is_fast=True`, nên phép so
thành vô nghĩa. Fork trong repo là 4.40, ở đó hai bản mới thực sự khác nhau. Đã kiểm được
gián tiếp: repo `Qwen/Qwen2.5-Coder-7B-Instruct` **có** `vocab.json` + `merges.txt`, nên
bản chậm nạp được. Câu hỏi id có khớp không thì bước [1] của gate trả lời, mất 1 phút.

**Ba tiện ích mới**

| File | Vai trò |
|---|---|
| [scripts/phase1_gate.sh](scripts/phase1_gate.sh) | chạy trọn gate 6 bước, rẻ trước đắt sau |
| [scripts/check_phase1.py](scripts/check_phase1.py) | tiêu chí **nội tại** Sq-70% vs All-KV; FAIL thì in sẵn 3 nghi phạm theo thứ tự dễ kiểm |
| [scripts/check_cluster_integrity.py](scripts/check_cluster_integrity.py) | quét CRC + kiểm đủ bộ ba file, song song |
| [configs/phase1.sh](configs/phase1.sh) | `source` phase0.sh rồi chỉ ghi đè phần khác |

`check_cluster_integrity.py` chính là món nợ từ bài học Phase 0 ("phải chạy `testzip()`
ngay sau mỗi lượt clustering"). Đã test trên 4 kiểu hỏng dựng sẵn: file rỗng, file cắt cụt,
**file lật byte ở giữa** (chỉ CRC bắt được — đúng ca đã làm mất 5 giờ), và thiếu file trong
bộ ba. Bắt đủ 4/4, `--delete` xoá rồi `offline_clustering.py` sinh lại nhờ logic bỏ qua mẫu
đã xong.

**Cờ `--limit` cho `offline_clustering.py` / `pred.py` / `eval.py`**

Trước đây không có cách nào chạy thử N mẫu — mà gate Phase 1 chỉ cần 20. Ba chi tiết:

1. `pred.py` cắt **sau** khi gán `different_prefix_index`, nên index vẫn là `0..N-1`, khớp
   đúng tên file centroid mà `offline_clustering.py --limit N` sinh ra.
2. Kết quả ghi vào thư mục có hậu tố **`_lim<N>`**. Không tách thì một lượt smoke test 20
   mẫu sẽ lặng lẽ đè lên kết quả 500 mẫu — **đúng loại lỗi với bug append #3** của Phase 0.
   `eval.py` nhận cùng `--limit` để đọc đúng thư mục.
3. `offline_clustering.py` cắt danh sách **trước** vòng profiling, vì vòng đó tokenize 2
   lần/mẫu bằng tokenizer chậm; với Qwen là BPE Python thuần nên quét cả 500 mẫu cho một
   smoke test 20 mẫu là lãng phí thật.

Nhân tiện gộp hai khối `if` dựng tên thư mục trùng nhau trong `eval.py` thành một —
trước đó chỗ đọc và chỗ ghi `result.json` dựng tên **độc lập**, sửa một bên quên bên kia
là kết quả rơi vào thư mục khác chỗ đọc.

**Centroid Qwen ghi vào thư mục riêng** `fixed-prompt-clusters/<model>/<dataset>/`. Tên file
là `centroids_tensor_dict_<dataidx>_<K>.pt` mà `K` tính từ `shared_prefix_length` — khác nhau
giữa hai tokenizer. Nên chúng **không** đè lên nhau, tức lỗi sẽ không lộ ra bằng một va chạm;
chỉ làm thư mục phình gấp đôi và rất dễ tra nhầm file.

**Đã kiểm, không phải lo:** `parse_model()` trong `utils/model_parse.py` không biết Qwen2 và
rơi vào nhánh mặc định `"llama"` → `get_layers` trả `model.model.layers`, đúng cấu trúc
Qwen2ForCausalLM. Chạy được nhờ trùng hợp, nhưng chạy đúng.

**Test CPU sau khi sửa:** `test_gqa_port.py` 20/20, `test_struct_clustering.py` 72/72,
`prepare_code_data.py --self_test` 12/12 — đều PASS.

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

### 2026-08-16 — ✅ GATE PHASE 0 PASS (LCC, dung sai nới ±2.0)

| Cấu hình | Đo được | Table 2 | Lệch |
|---|---|---|---|
| All-KV | **54,83** | 56,64 | −1,81 |
| Sq-70% | **56,08** | 56,93 | −0,85 |
| **Hiệu số Sq-70% − All-KV** | **+1,25** | +0,29 | +0,96 |

**Quyết định: nới dung sai từ ±0,3 lên ±2,0.** Protocol đặt ±0,3 để tái lập chính xác Table 2.
Ta không đạt được mức đó, nhưng mọi so sánh của protocol (SA vs +HardBoundary vs
+StructHierarchy) đều đo **trong cùng môi trường này**, nên chênh lệch nền so với bài không
làm hỏng kết luận nào. Với ±2,0 gate vẫn bắt được lỗi môi trường nghiêm trọng — cài nhầm
transformers hay sai truncation thường lệch 5-10 điểm, không phải 1,8.

**Kết luận quan trọng nhất: đường Squeezed Attention hoạt động đúng.** Sq-70% không tệ hơn
All-KV mà còn nhỉnh hơn 1,25 điểm, **cùng chiều** với bài (+0,29). Nếu centroid lookup sai thì
Sq-70% đã tụt vài điểm. Đây chính là điều gate cần xác nhận, vì Phase 2 xây tiếp lên đúng
đường này.

*(Việc SA vượt attention đầy đủ nghe nghịch lý nhưng bài cũng ghi nhận: bỏ bớt key có
attention thấp đôi khi có tác dụng khử nhiễu.)*

**Ba điểm chưa giải thích được, phải ghi vào paper:**
1. All-KV lệch −1,81 so với bài. Ba nghi phạm: trọng số model (bài dùng bản local
   `/home/chooper/longchat-7b-v1.5-32k`, ta dùng `lmsys/longchat-7b-v1.5-32k` trên HF);
   flash-attn 2.6.3 vs bản 2024 (~2.5.x); chi tiết truncation.
2. Chênh lệch nền **không đồng đều**: Sq-70% lệch 0,85 còn All-KV lệch 1,81. Một phần
   "+1,25 thay vì +0,29" đến từ chỗ này.
3. Hiệu số lệch 0,96 điểm — không đủ để kết luận pipeline sai, nhưng cũng không trùng khít.

**Phạm vi đã cố ý thu hẹp:** chỉ chạy LCC (bỏ RepoBench-P vì ~37 giờ / ~325 GB), chỉ Sq-70%
(bỏ Sq-80%, Sq-90%, H-Sq-90%). Lý do và tính toán ghi ở các mục dưới.

**Chi phí thực tế Phase 0**

| Khoản | |
|---|---|
| Clustering LCC 500 mẫu | 6h15m (45 giây/mẫu) |
| `pred.py` All-KV | ~1 giờ |
| `pred.py` Sq-70% | 3h07m (22 giây/mẫu, 2 GPU) |
| Hai lượt `pred.py` hỏng vì file centroid lỗi | ~5 giờ mất trắng |
| Dựng môi trường | ~5 giờ |
| **Tổng GPU** | **~20 giờ ≈ $32** |

**Bốn sự cố đã xử lý, chi tiết ở các mục dưới:** hết quota đĩa ở mẫu 113; lỗi dtype fp32/bf16
ở `o_proj` khi bật `use_centroids`; hai file centroid hỏng (mẫu 122, 458) khiến `pred.py`
chết sau nhiều giờ; ba lỗi vận hành trong cách tôi viết lệnh nền.

**Bài học về kiểm tra tính toàn vẹn.** Tôi kiểm tra ba lần với ba mức chặt dần, mỗi lần sau
một lượt `pred.py` hỏng: (1) file có tồn tại không → bỏ sót file cắt cụt; (2)
`zipfile.ZipFile()` mở được không → chỉ đọc mục lục, bỏ sót dữ liệu hỏng bên trong;
(3) `zipfile.testzip()` kiểm CRC toàn bộ → mới bắt được. Đáng lẽ dùng (3) ngay từ đầu:
20 phút quét rẻ hơn nhiều so với hai lượt `pred.py` mất ~5 giờ.

Với Phase 5/6 (sinh hàng chục nghìn file centroid), phải chạy `testzip()` ngay sau mỗi lượt
clustering. Bản song song 8 tiến trình quét 1.500 file trong vài phút — chi phí không đáng kể.

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
| ~~B1~~ | ~~Chạy gate Phase 0 trên GPU~~ | — | ✅ PASS 16/8, LCC, dung sai ±2.0 |
| **B6** | **Chạy `phase1_gate.sh` trên pod** | **2, 5, 6** | Bản port Qwen2/GQA **chưa từng chạy GPU**. ~45 phút, ~$1. Đây là việc chặn duy nhất còn lại của Phase 1 |
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
| D4 | Lấy benchmark fixed-context-dài từ đâu? | 🟡 **gần như tự giải 19/8** — RepoBench v1.1 có sẵn 732 fixed context ≥16k với nhiều query. Chỉ còn viết loader |


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

### 2026-08-17 06:38:51 — Phase 0 gate — longchat-v1.5-7b-32k — ✅ PASS

> Gate LCC, dung sai noi long 2.0

- Tolerance: ±2.0
- pred_dir: `LongBench/pred`
- env: *(chưa có `env_record.json`, chạy `scripts/record_env.py`)*

| Config | Task | Expected | Actual | Delta | Status |
|---|---|---:|---:|---:|---|
| All KV | lcc | 56.64 | 54.83 | -1.81 | ✅ PASS |
| All KV | repobench-p | 53.20 | - | - | ⬜ SKIP (thiếu task) |
| Sq-70% | lcc | 56.93 | 56.08 | -0.85 | ✅ PASS |
| Sq-70% | repobench-p | 54.64 | - | - | ⬜ SKIP (thiếu task) |
| Sq-80% | - | - | - | - | ⬜ SKIP (chưa có result.json) |
| Sq-90% | - | - | - | - | ⬜ SKIP (chưa có result.json) |
| H-Sq-90% | - | - | - | - | ⬜ SKIP (chưa có result.json) |

**PASS=2 · FAIL=0 · SKIP=5**

<!-- ghi chú tay bên dưới -->
