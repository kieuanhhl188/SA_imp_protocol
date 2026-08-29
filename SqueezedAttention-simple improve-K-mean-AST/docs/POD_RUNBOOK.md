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
> trên LCC là **~70 GB/seed**; với Qwen (4 head KV thay vì 32) chỉ **~5 GB**. Sinh lại được
> bất cứ lúc nào từ đúng seed, nên xoá thoải mái sau khi đã có `result.json`.

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

**CHỐT 28/8: model = `longchat-v1.5-7b-32k`, LCC-only, KHÔNG `--force_chat`.**
Quay về đúng model của bài gốc (Table 2) và khớp phạm vi đã thu hẹp ở Phase 0
([docs/PHASE0.md §8](PHASE0.md)). Cấu hình ở [configs/phase1.sh](../configs/phase1.sh) —
nó `source` `phase0.sh` rồi chỉ ghi đè `SQA_MODEL_CODE` + `SQA_FORCE_CHAT=0`.

### Phần RIÊNG của Phase 1 = dữ liệu 1.4 + gate dữ liệu — **chạy CPU**

```bash
bash scripts/phase1_gate.sh --data-only     # 500 mẫu, ~1-2 phút CPU (không GPU)
```

`--data-only` mặc định chạy **toàn bộ 500 mẫu** (không phải smoke 20) — Phase 2 đọc cả 500.

Việc nó làm:

| Bước | Nội dung | Output |
|---|---|---|
| [1] `prepare_code_data.py` | sinh offset **byte + ký tự** từng token của LCC (500 mẫu), `language` lấy per-sample | `/workspace/phase1_data/longchat-v1.5-7b-32k/lcc_{meta.jsonl,offsets.npz}` |
| [1b] `check_phase1_data.py` | gate 5 bước: ngôn ngữ đúng từng mẫu · đủ 500 mẫu · offset fast==slow, phủ kín, byte↔ký tự khớp · `fixed_context` không mất khúc nào | PASS/FAIL trên console |

Số đã biết (tất định, LCC thuần ASCII): ~1.559.310 token, 0 lệch fast/slow, unit/mẫu
trung vị 15, 12/500 mẫu suy biến (U≤2), 1/500 truncate. **Kỳ vọng PASS.**

Bước này chạy được cả trên máy Windows — không bắt buộc lên pod.

### Phần accuracy (Sq-70% vs All-KV trên LongChat/LCC) = **KHÔNG chạy ở đây**

LongChat là MHA (không GQA) và đi thẳng đường `modeling_llama` gốc → bước [2]–[6] của
`phase1_gate.sh` **trùng hoàn toàn với Phase 0** (`repro_lcc.sh`: cùng model, cùng LCC,
cùng centroid). Lấy số từ đó: All-KV **54,83** · Sq-70% **56,08** · hiệu ghép cặp **+1,25**
(p=0,39, KTC95 [−0,10; +2,59]).

Chỉ chạy `bash scripts/phase1_gate.sh --full` khi muốn một lần **kiểm độc lập** bằng
paired test của [check_phase1.py](../scripts/check_phase1.py). Lưu ý:
- Nó sinh **thêm ~70 GB** centroid ở `/workspace/fixed-prompt-clusters/longchat-v1.5-7b-32k/lcc/`
  (thư mục riêng), trong khi đĩa là ràng buộc chặt nhất của pod ([§9.1](#91-moosefs-cắt-cụt-file-im-lặng), [§9.2](#92-df-không-cho-biết-hạn-mức)).
- Muốn tiết kiệm: `--skip-cluster` rồi symlink centroid Phase 0 vào đúng chỗ:
  ```bash
  mkdir -p /workspace/fixed-prompt-clusters/longchat-v1.5-7b-32k
  ln -sfn /workspace/fixed-prompt-clusters_seed0/lcc \
          /workspace/fixed-prompt-clusters/longchat-v1.5-7b-32k/lcc
  bash scripts/phase1_gate.sh --full --skip-cluster
  ```
- `phase1_gate.sh` gọi `check_phase1.py --no_log_md` → **không** đụng `EXPERIMENT_LOG.md`.

**Kết quả cuối (bản `--full` nếu chạy):**

```
LongBench/pred/longchat-v1.5-7b-32k_baseline{,_lim20}/result.json
LongBench/pred/longchat-v1.5-7b-32k_PC5_PERC0.7{,_lim20}/result.json
/workspace/phase0_results/env_record_phase1.json
/workspace/phase0_results/logs/<TS>_phase1_gate.log
```

> ⚠️ Bộ dữ liệu 1.4 sinh **trước 22/8** (LongChat cũ) không có `offsets_bytes_*` và đã bị
> thư mục Qwen ghi đè — phải sinh lại. Sinh lại **không** ảnh hưởng centroid Phase 2/Phase 0
> đã có: offset ký tự và `shared_prefix_length` không đổi, chỉ thêm một mảng byte mới.

> Bước 1.6 (GQA per-head, QUEST App. G) là **N/A** với LongChat (`num_key_value_heads = 32
> = num_heads`). Chỉ bật lại khi thêm model cross-check có GQA.

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

### Bước 1 — smoke 3 mẫu để biết s/mẫu thật TRƯỚC khi chạy full

```bash
LIMIT_P2=3 bash scripts/run_phase2_phase5_lcc.sh
```
Chưa đo clustering trên LongChat bao giờ. Nhân s/mẫu × 500 × 3 nhánh rồi mới quyết.

### Bước 2 — chạy full

```bash
bash scripts/run_phase2_phase5_lcc.sh
```

Thứ tự trong script **có chủ đích**: chạy `hard_boundary` trước, rồi smoke Phase 5 trên 3 mẫu
ngay lập tức. Nếu `phase5_recall.py` hỏng thì hỏng sớm chứ không phải sau nhiều giờ.

**CHỐT 28/8: model = `longchat-v1.5-7b-32k`, LCC-only, KHÔNG `--force_chat`** (khớp Phase 0/1;
LongChat là MHA đi thẳng `modeling_llama`, không phải bản Instruct, lcc trong `NO_CHAT_TEMPLATE`).
Script `run_phase2_phase5_lcc.sh` nay `source configs/phase1.sh` nên lấy đúng
`SQA_MODEL_CODE` / `SQA_FORCE_CHAT` / `SQA_PHASE1_DIR`.

Cấu hình còn lại, không đổi: `--dataset lcc`, `--level function --level_l1 class`,
`--percent_clusters 5`, `--observation_window 100`.

Thời gian: **chưa đo trên LongChat**. Mốc thô — Phase 0 `offline_clustering.py` mất ~6h15 cho
LCC 500 mẫu (MHA). Ước mỗi nhánh ~4–8 giờ (dùng smoke ở Bước 1 để chốt).
Chạy lại được — `offline_clustering_struct.py` bỏ qua mẫu đã đủ file.

⚠️ **Dung lượng: LongChat tốn gấp ~8 lần Qwen** (32 KV head vs 4). Lượt Qwen 22/8: 5,8 / 5,8 /
7,8 GB. Ngoại suy LongChat: ~46 / ~46 / ~62 GB → ba nhánh ~150 GB, sát trần volume 200 GB.
Nhẹ hơn: `LIMIT_P2=200 bash scripts/run_phase2_phase5_lcc.sh`.

**Kết quả cuối:**

```
/workspace/p2_invariants_longchat.log    <- kiểm bất biến Phase 2, ĐỌC CÁI NÀY
$P2_DIR/{sa,hard_boundary,struct_hierarchy}/lcc/   <- centroid ba nhánh, ~150 GB
```

`$P2_DIR` mặc định là `/workspace/p2-longchat`, đổi bằng biến môi trường `P2_DIR`.

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

Chạy kèm trong script Phase 2 ở trên, hoặc riêng:

```bash
python phase5_recall.py longchat-v1.5-7b-32k --dataset lcc \
    --cluster_dir "sa=$P2_DIR/sa/lcc" \
    --cluster_dir "hard_boundary=$P2_DIR/hard_boundary/lcc" \
    --cluster_dir "struct_hierarchy=$P2_DIR/struct_hierarchy/lcc" \
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

## 9. Cạm bẫy trên pod này

Bốn thứ đã cắn thật, mất tổng cộng hơn 10 giờ GPU.

### 9.1 MooseFS cắt cụt file im lặng

`/workspace` là MooseFS qua FUSE. Vượt hạn mức thì **`torch.save` không raise** — file bị cắt
cụt, và vòng resume của `offline_clustering.py` thấy file *tồn tại* nên **bỏ qua đúng mẫu
hỏng**. `pred.py` chạy vài giờ rồi mới chết vì
`PytorchStreamReader failed reading zip archive`.

Chỉ kiểm CRC mới bắt được. Chạy sau **mọi** lượt clustering, kể cả khi tái dùng centroid cũ:

```bash
python scripts/check_cluster_integrity.py /workspace/fixed-prompt-clusters/lcc/ --expect 500
python scripts/check_cluster_integrity.py /workspace/fixed-prompt-clusters/lcc/ --delete
```

`--delete` xoá **cả bộ ba file** của mẫu hỏng, không chỉ file hỏng — nếu chỉ xoá file hỏng
thì `global_threshold` còn lại làm resume bỏ qua mẫu đó mãi mãi.

Sự cố 27/8: 5/1500 file hỏng có sẵn từ lượt 17/8, `pred.py` chạy 226/500 mẫu rồi chết.

### 9.2 `df` không cho biết hạn mức

`df /workspace` báo dung lượng **cả cụm MooseFS** (404 TB), không biết gì về hạn mức của bạn.
Không có `mfsgetquota`. **Chỉ dashboard RunPod mới cho biết con số thật.**

Ngày 16/8 volume thật chỉ được cấp ~50-55 GB trong khi tôi tưởng là 200 GB — job clustering
chết ở mẫu 113/500.

### 9.3 Git treo trên MooseFS

Thao tác ghi ref của git (`git fetch`/`pull`) **kẹt trong FUSE** và để lại
`refs/remotes/origin/main.lock`, làm mọi lệnh git sau đó hỏng. Tiến trình rơi vào trạng thái
`D` — `kill` không có tác dụng. Đã xảy ra 3 lần.

Đọc/ghi file thường thì bình thường (~600 MB/s). Chỉ khoá POSIX là hỏng.

**Đừng bấm Sync/Pull trong panel Source Control của VS Code trên pod.**

Gỡ khi đã kẹt — file lock đã chứa sẵn đúng SHA, chỉ cần làm nốt việc git bỏ dở:

```bash
ps aux | grep "[g]it fetch"          # kiểm không còn tiến trình sống
kill <pid>
cd /workspace/SA_imp_protocol/.git/refs/remotes/origin
mv main.lock main
```

**Cách đồng bộ code an toàn** — object đã tải về rồi thì không cần mạng:

```bash
cd /workspace/SA_imp_protocol
P="SqueezedAttention-simple improve-K-mean-AST"
git checkout <sha> -- "$P/LongBench/pred.py" "$P/scripts/repro_lcc.sh"
```

Hoặc bỏ git hẳn, scp từ Windows (quote **một lớp**, scp đời mới dùng SFTP nên không expand
shell ở đầu xa):

```bash
R="/workspace/SA_imp_protocol/SqueezedAttention-simple improve-K-mean-AST"
scp scripts/repro_lcc.sh runpod:"$R/scripts/"
```

Commit và push làm ở máy Windows, pod chỉ nhận.

### 9.4 Lượt chạy dở đội lốt kết quả

`pred.py` cũ `join()` mà không kiểm `exitcode`: worker chết vì exception nhưng tiến trình cha
vẫn thoát 0, `set -e` không bắt, `eval.py` chấm 226 mẫu và ghi `result.json` như thật.

Đã chặn ba tầng (commit `ef2c98e`): `pred.py` kiểm `exitcode` mọi worker; `eval.py --expect N`
từ chối ghi khi thiếu mẫu; `aggregate_runs.py` in cột `số mẫu` và báo `[SAI]` khi các cấu hình
lệch nhau.

Nếu chạy tay không qua `repro_lcc.sh`, **luôn đếm dòng trước khi chấm điểm**:

```bash
wc -l < pred/longchat-v1.5-7b-32k_PC5_PERC0.7_runs0/lcc.jsonl    # phải đúng 500
```

### 9.5 `model2maxlen` phải là 31500

Không phải 32768. LongChat có đúng 32768 vị trí RoPE; đặt sát trần thì token tràn ra ngoài dải
và **kết quả là rác, không phải sai số**. 31500 = native trừ lề, đã chốt ở
[EXPERIMENT_LOG](../EXPERIMENT_LOG.md) mục **D7**.

Đổi con số này còn làm `shared_prefix_length` khác đi → `num_centroids` khác → tên file centroid
khác → `pred.py` chết giữa chừng vì không tìm thấy file.

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
`EXPERIMENT_LOG.md` (file có trong git), vì hai bên cùng sửa là conflict — đã xảy ra với 51
dòng gate tự phụ lục ngày 26/8.

`scripts/repro_lcc.sh` đã theo quy ước này: nó ghi ra `$SQA_RESULT_DIR` và **không** truyền
`--log_md`, nên không đụng vào file trong git.

`scripts/phase1_gate.sh` (từ 28/8) đã gọi `check_phase1.py --no_log_md` nên **không** đụng
`EXPERIMENT_LOG.md`. Nếu chạy `check_phase1.py` tay thì tự thêm `--no_log_md` — mặc định của
nó vẫn ghi thẳng vào file trong git.

Đã lỡ ghi rồi thì cất phần pod thêm ra ngoài trước khi đồng bộ code — xem [§9.3](#93-git-treo-trên-moosefs):

```bash
cp EXPERIMENT_LOG.md /workspace/EXPERIMENT_LOG_pod_<ngày>.md
git checkout HEAD -- "SqueezedAttention-simple improve-K-mean-AST/EXPERIMENT_LOG.md"
```
