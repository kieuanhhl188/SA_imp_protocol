# Runbook — chạy trên pod A100

Sổ tay vận hành cho RunPod A100-80GB SXM. Mỗi phase: chạy lệnh gì, mất bao lâu,
**kết quả cuối cùng nằm ở đâu**.

Tài liệu này mô tả **những gì script thực sự làm**, không phải ý định. Chỗ nào quy ước
lưu trữ chưa nhất quán thì có ghi rõ ở [§8](#8-chỗ-quy-ước-còn-lệch).

---

## 1. Vào phiên làm việc

Endpoint SSH **đổi mỗi lần khởi động pod**. Cập nhật `~/.ssh/config` ở máy Windows trước:

```
Host runpod
    HostName <IP mới>
    User root
    Port <port mới>
    IdentityFile C:/Users/Admin/.ssh/id_ed25519
    Compression yes
```

Rồi trên pod, mọi phiên đều bắt đầu bằng:

```bash
source /workspace/env.sh
cd "/workspace/SA_imp_protocol/SqueezedAttention-simple improve-K-mean-AST"
```

`/workspace/env.sh` do [scripts/setup_pod.sh](../scripts/setup_pod.sh) sinh ra, nó kích hoạt
venv Python 3.10 và đặt:

| Biến | Giá trị | Dùng cho |
|---|---|---|
| `HF_HOME` | `/workspace/hf` | weights, khỏi tải lại sau restart |
| `SQA_CLUSTER_DIR` | `/workspace/fixed-prompt-clusters` | centroid Phase 0/1 |
| `SQA_RESULT_DIR` | `/workspace/phase0_results` | env record + log + tổng hợp |
| `SQA_PHASE1_DIR` | `/workspace/phase1_data` | offset token Phase 1 |
| `CUDA_VISIBLE_DEVICES` | `0` | cố định 1 GPU cho mọi lần đo |
| `HF_DATASETS_TRUST_REMOTE_CODE` | `1` | không có là job dài **treo** ở prompt `[y/N]` |

Pod mới hoàn toàn thì dựng môi trường bằng `bash scripts/setup_pod.sh` (~30 phút, đã chặn
sẵn 5 cái bẫy phiên bản — đọc phần đầu file đó trước khi đổi bất cứ version nào).

**Luôn chạy job dài trong `tmux`.** Rớt SSH là mất cả buổi.

```bash
tmux new -s p0          # tạo
tmux attach -t p0       # quay lại
# Ctrl-B rồi D để thoát mà job vẫn chạy
```

### 1.1 Điều kiện chạy các phase 

Các phase **không nối tiếp nhau** trừ Phase 5. Phase 2 KHÔNG cần Phase 0, KHÔNG cần
`phase1_gate.sh --full` — chỉ cần dữ liệu 1.4.

| Phase | Lệnh | Điều kiện tiên quyết | GPU | Weight model | Đĩa thêm | Thời gian |
|---|---|---|:--:|:--:|---|---|
| setup | `bash scripts/setup_pod.sh` | pod có driver NVIDIA + `/workspace` | – | – | ~15 GB (venv) | ~30 ph |
| **0** | `bash scripts/repro_lcc.sh --seeds "0 1 2"` | setup | ✅ | ✅ | ~70 GB/seed | 6h15 + ~3h/seed |
| **1** dữ liệu | `bash scripts/phase1_gate.sh --data-only` | setup | – | – | vài MB | ~2 ph (CPU) |
| **1** `--full` *(tuỳ chọn)* | `bash scripts/phase1_gate.sh --full --skip-cluster` | setup + dữ liệu 1.4 + centroid Phase 0 | ✅ | ✅ | +70 GB nếu không skip | ≈ Phase 0 |
| **2** | `LIMIT_P2=200 bash scripts/run_phase2_phase5_lcc.sh` | setup + **dữ liệu 1.4** | ✅ | ✅ | ~88 GB | 7–10h |
| **5** | trong script Phase 2, hoặc `python phase5_recall.py …` | setup + dữ liệu 1.4 + **centroid Phase 2** | ✅ | ✅ | vài MB | ~20 ph |

Weight model (`longchat-v1.5-7b-32k`, ~14 GB) **tự tải** về `$HF_HOME` ở lần chạy GPU đầu.

### 1.2 Kết nối trên pod mới

Mọi thứ các phase cần đều nằm trên `/workspace` (venv, weight, dữ liệu 1.4, centroid).
`/workspace` là network volume — sống độc lập với container.

**Pod mới, gắn lại đúng volume `/workspace`** (trường hợp thường):

```bash
source /workspace/env.sh
cd "/workspace/SA_imp_protocol/SqueezedAttention-simple improve-K-mean-AST"
ls /workspace/venv310/bin/python \
   /workspace/phase1_data/longchat-v1.5-7b-32k/lcc_meta.jsonl && echo "OK — chạy thẳng phase cần"
```

Có `OK` → không dựng lại gì, chạy thẳng Phase 0/2/5. Chỉ cần cập nhật `~/.ssh/config` ở
Windows (endpoint đổi mỗi lần khởi động).

**Pod mới, volume trắng** — mất hết, dựng lại theo thứ tự:

```bash
bash scripts/setup_pod.sh                       # 1. môi trường, ~30 ph
source /workspace/env.sh
bash scripts/phase1_gate.sh --data-only         # 2. dữ liệu 1.4, ~2 ph
LIMIT_P2=200 bash scripts/run_phase2_phase5_lcc.sh   # 3. Phase 2 + 5 (weight tự tải)
```

Centroid Phase 0 (`fixed-prompt-clusters_seed*`) chỉ sinh lại khi cần chạy Phase 0 —
`bash scripts/repro_lcc.sh --seeds "0 1 2"`, hoặc chép từ backup.

---

## 2. Kết quả cuối cùng của từng phase nằm ở đâu

Bảng tra nhanh. Chi tiết ở mục của từng phase.

| Phase | Kết quả cuối cùng | Trung gian (xoá được) |
|---|---|---|
| **0** | `/workspace/phase0_results/repro_lcc_<TS>.md` + `.json`<br>`LongBench/pred/longchat-v1.5-7b-32k_{baseline,PC5_PERC0.7}_run<TAG>/result.json` | `/workspace/fixed-prompt-clusters_seed<N>/lcc/` (~70 GB/seed) |
| **1.4** | `/workspace/phase1_data/longchat-v1.5-7b-32k/lcc_{meta.jsonl,offsets.npz}` — offset byte + ký tự từng token (đầu vào Phase 2) | — |
| **1** (accuracy) | = Phase 0 (`repro_lcc.sh`). Chỉ khi chạy `phase1_gate.sh --full`: `LongBench/pred/longchat-v1.5-7b-32k_{baseline,PC5_PERC0.7}_*/result.json` | `/workspace/fixed-prompt-clusters/longchat-v1.5-7b-32k/lcc/` (~70 GB — nên symlink Phase 0) |
| **2** | `/workspace/p2_invariants_longchat.log` — kết quả kiểm bất biến | `/workspace/p2-longchat/{sa,hard_boundary,struct_hierarchy}/lcc/` (~150 GB) |
| **5** | `/workspace/phase5_lcc.json` — recall@budget (C2) | `/workspace/phase5_smoke.json` (smoke 3 mẫu) |
| 3, 4, 6, 7 | **chưa có code** | — |

Ba thứ dùng chung cho mọi phase:

| | Đường dẫn |
|---|---|
| Console log đầy đủ | `/workspace/phase0_results/logs/<TS>_<tên_gate>.log` |
| Bản ghi môi trường | `/workspace/phase0_results/env_record*.json` + `_pip_freeze.txt` |
| Nhật ký tổng | `EXPERIMENT_LOG.md` trong repo — **nguồn sự thật duy nhất** |

> **Kết quả cần giữ chỉ vài MB.** Toàn bộ dung lượng lớn là centroid trung gian. Với LongChat
> trên LCC là **~70 GB/seed**

---

## 3. Phase 0 — môi trường + tái lập baseline SA

Phạm vi đã chốt: **LCC × LongChat-7B-v1.5-32K**, hai cấu hình **All-KV** và **Sq-70%**.
Không so với Table 2 của bài; mốc là mean ± std của chính đường ống này.
Chi tiết trong [docs/PHASE0.md](PHASE0.md).

```bash
# dùng lại centroid seed 0 đã có (đừng sinh lại, mất 6h15)
ln -sfn /workspace/fixed-prompt-clusters /workspace/fixed-prompt-clusters_seed0

bash scripts/repro_lcc.sh --limit 20 --seeds "0" --skip-cluster   # smoke, ~10 phút
bash scripts/repro_lcc.sh --seeds "0"   --skip-cluster            # ~3h10
bash scripts/repro_lcc.sh --seeds "1 2" --purge-after             # 2 seed mới
bash scripts/repro_lcc.sh --seeds "0 1 2" --aggregate-only        # gộp mean±std
```

**Thời gian thật đo trên A100-80GB, 500 mẫu LCC:**

| Bước | Thời gian |
|---|---|
| Offline clustering (dùng chung mọi seed) | 6h15 |
| `pred.py` All-KV | **16 phút 29** |
| `pred.py` Sq-70% | **~2h40** (16,5–21 s/mẫu) |
| Quét CRC 1500 file centroid | vài phút |

**Kết quả cuối:**

```
/workspace/phase0_results/repro_lcc_<TS>.md      <- bảng mean ± std, đọc cái này
/workspace/phase0_results/repro_lcc_<TS>.json    <- cùng nội dung, dạng máy đọc
LongBench/pred/longchat-v1.5-7b-32k_baseline_runs0/result.json
LongBench/pred/longchat-v1.5-7b-32k_PC5_PERC0.7_runs0/result.json
```

Mỗi lượt lặp có hậu tố `_run<TAG>` riêng, smoke test có thêm `_lim<N>` — hai lượt không bao
giờ ghi đè nhau.

**Mốc đã biết:** All-KV = **54,83** trên 500 mẫu. Con số này tái lập được bit-for-bit giữa
lượt 17/8 và lượt 27/8. Lệch khỏi nó nghĩa là môi trường đã đổi.

---

## 4. Phase 1 — chuẩn bị dữ liệu code

### Phải chạy: một lệnh, CPU, ~1–2 phút

```bash
bash scripts/phase1_gate.sh --data-only
```

Sinh dữ liệu 1.4 (offset byte + ký tự từng token của LCC, 500 mẫu — `prepare_code_data.py`)
rồi gate 5 bước (`check_phase1_data.py`: ngôn ngữ đúng từng mẫu · đủ 500 mẫu · offset
fast==slow, phủ kín, byte↔ký tự khớp · `fixed_context` không mất khúc nào).

Không cần GPU, không cần weight model — chạy được cả trên máy Windows. `--data-only` mặc
định chạy **toàn bộ 500 mẫu** (Phase 2 đọc cả 500).

**Sản phẩm — đầu vào bắt buộc của Phase 2:**

```
/workspace/phase1_data/longchat-v1.5-7b-32k/lcc_meta.jsonl
/workspace/phase1_data/longchat-v1.5-7b-32k/lcc_offsets.npz
```

Hai file này còn đó thì **không cần chạy lại**. `/workspace` là network volume, sống qua
restart — **pod mới gắn lại đúng volume này thì file vẫn còn**, chỉ chạy lại khi:
volume trắng (pod mới + volume mới), hoặc bộ 1.4 là bản **trước 22/8** (thiếu
`offsets_bytes_*`). Sinh lại chỉ 2 phút CPU, tất định — nghi ngờ thì cứ chạy.

Đã chạy 29/8/2026, PASS (tất định, LCC thuần ASCII): 2.094.562 token, 0 lệch fast/slow,
byte↔ký tự khớp 500/500, `shared_prefix_length` khớp meta, function trung vị 15 unit/mẫu,
12/500 mẫu suy biến (U≤2), 1/500 truncate.

### KHÔNG chạy phần accuracy ở đây

`phase1_gate.sh` bỏ `--data-only` sẽ chạy tiếp bước [2]–[6] (pred All-KV + Sq-70% + paired
test). Với LongChat (MHA, đi thẳng `modeling_llama`) mấy bước đó **trùng hoàn toàn Phase 0**
— cùng model, cùng LCC, cùng centroid. Lấy số từ Phase 0: All-KV **54,83** · Sq-70%
**56,08** · hiệu ghép cặp **+1,25** (p=0,39, KTC95 [−0,10; +2,59]).

`bash scripts/phase1_gate.sh --full` chỉ để **một lần kiểm độc lập** bằng paired test của
[check_phase1.py](../scripts/check_phase1.py) — **không bắt buộc** cho Phase 2/5. Nó sinh
thêm ~70 GB centroid (thư mục riêng), trong khi đĩa là ràng buộc chặt nhất của pod
([§9.1](#91-moosefs-cắt-cụt-file-và-df-nói-dối-về-đĩa)). Tiết kiệm bằng `--skip-cluster` +
symlink centroid Phase 0:

```bash
mkdir -p /workspace/fixed-prompt-clusters/longchat-v1.5-7b-32k
ln -sfn /workspace/fixed-prompt-clusters_seed0/lcc \
        /workspace/fixed-prompt-clusters/longchat-v1.5-7b-32k/lcc
bash scripts/phase1_gate.sh --full --skip-cluster
```

Kết quả bản `--full`: `LongBench/pred/longchat-v1.5-7b-32k_{baseline,PC5_PERC0.7}/result.json`
+ `phase0_results/{env_record_phase1.json,logs/<TS>_phase1_gate.log}`. `phase1_gate.sh` gọi
`check_phase1.py --no_log_md` nên không đụng `EXPERIMENT_LOG.md`.

### Cấu hình & lưu ý

- **Chốt 28/8: `longchat-v1.5-7b-32k`, LCC-only, KHÔNG `--force_chat`** — khớp Phase 0
  ([PHASE0.md §8](PHASE0.md)). Cấu hình ở [configs/phase1.sh](../configs/phase1.sh)
  (`source` `phase0.sh` rồi chỉ ghi đè `SQA_MODEL_CODE` + `SQA_FORCE_CHAT=0`).
- ⚠️ Bộ 1.4 sinh **trước 22/8** thiếu `offsets_bytes_*` và đã bị thư mục Qwen ghi đè — phải
  sinh lại. Sinh lại **không** ảnh hưởng centroid Phase 0/2 đã có (offset ký tự và
  `shared_prefix_length` không đổi, chỉ thêm mảng byte mới).
- Bước 1.6 (GQA per-head, QUEST App. G) **N/A** với LongChat (`num_key_value_heads =
  num_heads = 32`). Chỉ bật khi thêm model cross-check có GQA.

---

## 5. Phase 2 — structure-aware clustering (Idea 1)

> **Phase 2 đo VIỆC THI HÀNH, không đo CHẤT LƯỢNG.** Nó trả lời: ranh giới cứng có được thi
> hành đúng không (`hard_boundary` phải 0% cluster vắt biên), can thiệp mạnh tới đâu (`sa`
> vắt biên bao nhiêu %), ba nhánh có cùng ngân sách không. "Ranh giới cứng có làm retrieval
> tốt lên không" là **Phase 5 (C2)** và **Phase 6 (C1)**.

Ba nhánh chạy trên cùng một nền: `sa` (đối chứng, bằng SA gốc), `hard_boundary`,
`struct_hierarchy`.

### Bước 0 — tiên quyết: dữ liệu Phase 1.4 cho LongChat (CPU, ~1–2 phút)

```bash
bash scripts/phase1_gate.sh --data-only
ls /workspace/phase1_data/longchat-v1.5-7b-32k/lcc_meta.jsonl   # phải có
```
`run_phase2_phase5_lcc.sh` dừng ngay nếu thiếu file này (khác tokenizer → offset khác).

### Bước 1 — smoke 3 mẫu · ✅ ĐÃ CHẠY 29/8

```bash
LIMIT_P2=3 bash scripts/run_phase2_phase5_lcc.sh
```
Kết quả 29/8: cả 3 nhánh xong, `p2_invariants_longchat.log` xanh — `hard_boundary` +
`struct_hierarchy` 0,0% vắt biên, `sa` 22,8–36,7%, Phase 5 smoke recall 0,748.

### Bước 2 — chạy `LIMIT_P2=200` (KHÔNG full 500)

```bash
LIMIT_P2=200 bash scripts/run_phase2_phase5_lcc.sh
```

Full 500 tốn **~232 GB đĩa** (vượt volume 200 GB) và **~18–24h**. `LIMIT_P2=200` → ~88 GB,
~7–10h, và Phase 5 vốn chỉ đánh giá `--limit 100` nên không mất độ phủ. Dùng 150 nếu muốn
đệm đĩa rộng hơn (~66 GB). Xem [EXPERIMENT_LOG.md "Smoke GPU 29/8"](../EXPERIMENT_LOG.md).

Thứ tự trong script **có chủ đích**: chạy `hard_boundary` trước, rồi smoke Phase 5 trên 3 mẫu
ngay lập tức. Nếu `phase5_recall.py` hỏng thì hỏng sớm chứ không phải sau nhiều giờ.

**CHỐT 28/8: model = `longchat-v1.5-7b-32k`, LCC-only, KHÔNG `--force_chat`** (khớp Phase 0/1;
LongChat là MHA đi thẳng `modeling_llama`, không phải bản Instruct, lcc trong `NO_CHAT_TEMPLATE`).
Script `run_phase2_phase5_lcc.sh` nay `source configs/phase1.sh` nên lấy đúng
`SQA_MODEL_CODE` / `SQA_FORCE_CHAT` / `SQA_PHASE1_DIR`.

Cấu hình còn lại, không đổi: `--dataset lcc`, `--level function --level_l1 class`,
`--percent_clusters 5`, `--observation_window 100`.

**Thời gian** (clustering scale ~8× theo số KV head — Phase 0: LongChat 6h15 vs Qwen 45 phút;
lượt Qwen full 3 nhánh + Phase 5 = 4h50): full 500 ~18–24h · **`LIMIT_P2=200` ~7–10h** ·
`LIMIT_P2=150` ~5–7h. Nhánh `sa` một mình ≈ Phase 0 `offline_clustering.py` ≈ 6h15 ở full.
Mỗi nhánh chạy forward pass RIÊNG (không cache qkv giữa các nhánh). Chạy lại được —
`offline_clustering_struct.py` bỏ qua mẫu đã đủ file.

⚠️ **Dung lượng: ~33 KB/token** (đo 29/8, không còn ngoại suy ×8). `sa` / `hard_boundary`
~69 GB mỗi nhánh ở full 500 · `struct_hierarchy` ~94 GB → **ba nhánh ~232 GB, vượt volume
200 GB**. Cộng 68 GB centroid seed-0 phải giữ tại chỗ. `LIMIT_P2=200` → ~88 GB (26/26/36).

**Kết quả cuối:**

```
/workspace/p2_invariants_longchat.log    <- kiểm bất biến Phase 2, ĐỌC CÁI NÀY
$P2_DIR/{sa,hard_boundary,struct_hierarchy}/lcc/   <- centroid ba nhánh, ~88 GB voi LIMIT_P2=200
```

`$P2_DIR` mặc định là `/workspace/p2-longchat`, đổi bằng biến môi trường `P2_DIR`.
Chạy Phase 5 riêng thì phải `export P2_DIR=/workspace/p2-longchat` trước (§6 dùng biến này).

### Bất biến D — giới hạn đã biết, chưa giải

Nhánh `sa` của `offline_clustering_struct.py` phải tái lập `offline_clustering.py` gốc. Lượt
Qwen: **55/500 mẫu (11%) lệch >5%**, nguyên nhân **chưa biết** (KHÔNG phải seed — chẩn đoán
cũ sai). Khi có centroid LongChat, chạy chẩn đoán (rẻ, ~2–3 phút GPU, 3 mẫu):

```bash
python scripts/diag_invariant_d.py longchat-v1.5-7b-32k --dataset lcc \
    --phase1_dir /workspace/phase1_data \
    --reference_dir /workspace/fixed-prompt-clusters_seed0/lcc \
    --out /workspace/diag_invd_longchat.json
```
T1 (loại "key vector khác giữa 2 forward") · T2 (nhiễu cuML thuần, ra ngưỡng sàn) · T3 (so
với file trên đĩa). `T3 ≈ T2` → nhiễu cuML. `T3 ≫ T2` → reference sinh bằng config khác.
Chi tiết: [PHASE2_RESULTS.md Bảng 4](PHASE2_RESULTS.md).

### `--level_l1` vô hiệu trên LCC ở `level=function`

71% mẫu bị chặn K1 ở số function (LCC trung vị ~15 function < ngân sách L1 ~22). `struct_hierarchy`
thực chất là "L1 = trung bình theo function" — 2 tầng, **không phải** class→function→token.
Quét `--level_l1` trên LCC/function ra cùng kết quả mọi giá trị; muốn khảo sát phải để L2 ở
`block`.

---

## 6. Phase 5 — C2 recall@budget

Chạy kèm trong script Phase 2 ở trên, hoặc riêng (đặt `P2` cho khớp `$P2_DIR` của script):

```bash
P2=/workspace/p2-longchat
python phase5_recall.py longchat-v1.5-7b-32k --dataset lcc \
    --cluster_dir "sa=$P2/sa/lcc" \
    --cluster_dir "hard_boundary=$P2/hard_boundary/lcc" \
    --cluster_dir "struct_hierarchy=$P2/struct_hierarchy/lcc" \
    --sparsity 70 80 90 --limit 100 --out /workspace/phase5_lcc.json
```

**Kết quả cuối:**

```
/workspace/phase5_lcc.json       <- recall@budget cho 3 nhánh × 3 mức sparsity
/workspace/phase5_smoke.json     <- smoke 3 mẫu, KHÔNG phải kết quả
```

Phase 5 quyết định H0 theo protocol — fail thì dừng sớm, không chạy Phase 6.

---

## 7. Phase 3, 4, 6, 7 — chưa có code

| Phase | Nội dung | Trạng thái |
|---|---|---|
| 3 | Symbol / def-use signal (Idea 2) | chưa viết |
| 4 | Incremental re-clustering (Idea 3) | chưa viết |
| 6 | C1 accuracy@budget end-task | chưa viết — cần loader RepoBench v1.1 trước |
| 7 | C3 chi phí khấu hao + phân tích | chưa viết |

Mục này cập nhật khi có script. Thứ tự chạy theo protocol:
`Phase 0 → 1 → 5 (quyết định H0) → 6 + 4/7`.

---

## 8. Chỗ quy ước còn lệch

Ghi lại để biết mà đọc đúng chỗ, **chưa sửa**:

1. **Phase 0/1 ghi vào thư mục có tên**, Phase 2/5 ghi **file rời ở gốc `/workspace`**
   (`phase5_lcc.json`, `p2_invariants_longchat.log`) lẫn với hàng chục file log khác.
   Nhất quán hơn thì nên là `/workspace/phase2_results/` và `/workspace/phase5_results/`.
2. **`SQA_RESULT_DIR` tên là `phase0_results` nhưng chứa cả env record của Phase 1**
   (`env_record_phase1.json`) và log của mọi gate.
3. **`$P2_DIR` mặc định `/workspace/p2-longchat`.** Lượt Qwen cũ nằm ở `/workspace/p2-instruct`
   và `/workspace/struct-clusters` (19 GB) — KHÔNG dùng lại được cho LongChat (khác tokenizer,
   khác số head). Kiểm `du -sh /workspace/p2-*` rồi xoá bộ Qwen nếu cần chỗ.

---

## 9. Lưu ý riêng của pod này

### 9.1 MooseFS cắt cụt file, và `df` nói dối về đĩa

`/workspace` không phải ổ đĩa trong máy mà là **MooseFS** — một hệ thống file mạng phân
tán, dữ liệu nằm trên cụm máy chủ khác, mount vào pod qua FUSE. Nó cư xử khác ổ local ở
hai chỗ chết người.

Thứ nhất: vượt hạn mức thì `torch.save` **không raise** — file bị cắt cụt, vòng resume của
`offline_clustering.py` thấy file *tồn tại* nên bỏ qua mẫu hỏng, rồi `pred.py` chạy vài giờ
mới chết vì `PytorchStreamReader failed reading zip archive`.

Chỉ kiểm CRC mới bắt được. Chạy sau **mọi** lượt clustering, kể cả khi tái dùng centroid cũ:

```bash
python scripts/check_cluster_integrity.py /workspace/fixed-prompt-clusters/lcc/ --expect 500
python scripts/check_cluster_integrity.py /workspace/fixed-prompt-clusters/lcc/ --delete
```

`--delete` xoá **cả bộ ba file** của mẫu hỏng — chỉ xoá file hỏng thì `global_threshold`
còn lại làm resume bỏ qua mẫu đó mãi mãi. (27/8: 5/1500 file hỏng từ lượt 17/8, `pred.py`
chết ở mẫu 226/500.)

Thứ hai: `df /workspace` báo dung lượng **cả cụm** (404 TB), không có `mfsgetquota` — hạn
mức thật chỉ thấy trên dashboard RunPod. (16/8: tưởng 200 GB, thật ~50 GB, clustering chết
ở mẫu 113.)

### 9.2 Git treo trên MooseFS

`git fetch`/`pull` ghi ref → **kẹt trong FUSE**, để lại `refs/remotes/origin/main.lock`,
mọi lệnh git sau đó hỏng, tiến trình vào trạng thái `D` không kill được. Đã xảy ra 3 lần.
**Đừng bấm Sync/Pull trong panel Source Control của VS Code trên pod.**

Gỡ khi đã kẹt (file lock đã chứa đúng SHA, chỉ cần làm nốt):

```bash
ps aux | grep "[g]it fetch"          # chắc chắn không còn tiến trình sống
cd /workspace/SA_imp_protocol/.git/refs/remotes/origin && mv main.lock main
```

Sync code an toàn: `git checkout <sha> -- <path>` (object tải rồi thì không cần mạng), hoặc
scp từ Windows — quote **một lớp**, scp đời mới dùng SFTP nên không expand shell ở đầu xa:

```bash
R="/workspace/SA_imp_protocol/SqueezedAttention-simple improve-K-mean-AST"
scp scripts/repro_lcc.sh runpod:"$R/scripts/"
```

Commit/push làm ở Windows, pod chỉ nhận.

### 9.3 Lượt chạy dở đội lốt kết quả

`pred.py` cũ thoát 0 kể cả khi worker chết giữa chừng → `eval.py` chấm phần dở, ghi
`result.json` như thật. Đã chặn ba tầng ở `ef2c98e` (`pred.py` kiểm `exitcode`,
`eval.py --expect N`, `aggregate_runs.py` in cột số mẫu). Chạy tay không qua `repro_lcc.sh`
thì tự đếm trước khi chấm:

```bash
wc -l < pred/longchat-v1.5-7b-32k_PC5_PERC0.7_runs0/lcc.jsonl    # phải đúng 500
```

### 9.4 `model2maxlen` phải là 31500

Không phải 32768. LongChat có đúng 32768 vị trí RoPE; đặt sát trần thì token tràn dải và
**kết quả là rác, không phải sai số**. Đổi con số này còn làm `shared_prefix_length` khác
→ tên file centroid khác → `pred.py` chết vì không tìm thấy file. Chốt ở
[EXPERIMENT_LOG](../EXPERIMENT_LOG.md) mục **D7**.

```bash
grep longchat LongBench/config/model2maxlen.json    # phải ra 31500
```

---

## 10. Đưa kết quả về máy Windows

Kết quả chỉ vài MB, kéo về rồi ghi vào `EXPERIMENT_LOG.md` ở máy Windows và push.

```bash
scp runpod:/workspace/phase0_results/repro_lcc_*.md  ./
scp runpod:/workspace/phase5_lcc.json                ./
scp -r runpod:"/workspace/SA_imp_protocol/SqueezedAttention-simple improve-K-mean-AST/LongBench/pred/longchat-v1.5-7b-32k_baseline_runs0" ./
```

**Một chiều duy nhất: pod sinh số → Windows ghi nhật ký → push.** Đừng để pod tự ghi vào
`EXPERIMENT_LOG.md` (file có trong git), vì hai bên cùng sửa là conflict — đã xảy ra 26/8.

`repro_lcc.sh` và `phase1_gate.sh` đã theo quy ước này (không truyền `--log_md` / gọi
`check_phase1.py --no_log_md`). Chạy `check_phase1.py` tay thì tự thêm `--no_log_md` —
mặc định của nó ghi thẳng vào file trong git. Đã lỡ ghi thì `git checkout HEAD -- <file>`
sau khi copy bản pod ra ngoài.
