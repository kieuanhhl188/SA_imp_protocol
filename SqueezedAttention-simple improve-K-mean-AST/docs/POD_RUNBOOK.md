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
| **1** | `EXPERIMENT_LOG.md` (mục "Lịch sử chạy", `check_phase1.py` tự phụ lục)<br>`LongBench/pred/qwen2.5-coder-7b-instruct_*/result.json` | `/workspace/fixed-prompt-clusters/qwen2.5-coder-7b-instruct/lcc/` (~5,5 GB) |
| **1.4** | `/workspace/phase1_data/<model>/` — offset byte + ký tự từng token | — |
| **2** | `/workspace/p2_invariants_instruct.log` — kết quả kiểm bất biến | `$P2_DIR/{sa,hard_boundary,struct_hierarchy}/lcc/` (~19 GB) |
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

## 4. Phase 1 — chuẩn bị dữ liệu code + port sang Qwen2 (GQA)

Model: `qwen2.5-coder-7b-instruct`, **bắt buộc `--force_chat` ở mọi bước**. Cấu hình ở
[configs/phase1.sh](../configs/phase1.sh) — nó `source` `phase0.sh` rồi chỉ ghi đè phần khác.

```bash
bash scripts/phase1_gate.sh --data-only    # bước [1] không cần GPU, ~1 phút
bash scripts/phase1_gate.sh                # 20 mẫu đầu, ~30-45 phút
bash scripts/phase1_gate.sh --full         # cả 500 mẫu LCC
bash scripts/phase1_gate.sh --skip-cluster # centroid đã có
```

Gate chạy 6 bước theo thứ tự **rẻ trước đắt sau**, hỏng ở đâu dừng ngay ở đó: kiểm tokenizer
(CPU) → offline clustering → CRC → pred All-KV → pred Sq-70% → so sánh.

Tiêu chí PASS là **nội tại**, không phải tái lập số của bài: Sq-70% ≥ All-KV − tolerance,
đánh giá bằng **paired test trên hiệu số từng mẫu** ([scripts/check_phase1.py](../scripts/check_phase1.py)).
Table 2 không có Qwen nên không có mốc ngoài nào để so.

**Kết quả cuối:**

```
EXPERIMENT_LOG.md                       <- check_phase1.py tự phụ lục mục PASS/FAIL
LongBench/pred/qwen2.5-coder-7b-instruct_baseline{,_lim20}/result.json
LongBench/pred/qwen2.5-coder-7b-instruct_PC5_PERC0.7{,_lim20}/result.json
/workspace/phase0_results/env_record_phase1.json
/workspace/phase0_results/logs/<TS>_phase1_gate.log
```

**Dữ liệu Phase 1.4** (offset token, đầu vào của Phase 2) sinh riêng và **lưu riêng**:

```bash
python scripts/prepare_code_data.py qwen2.5-coder-7b-instruct --dataset lcc \
    --output_path /workspace/phase1_data/qwen2.5-coder-7b-instruct
python scripts/check_phase1_data.py qwen2.5-coder-7b-instruct --dataset lcc
```

```
/workspace/phase1_data/<model>/     <- offset BYTE + KÝ TỰ từng token
```

Thời gian: LCC 500 mẫu **37 giây**, RepoBench-P **4 phút 51**. Bước này là CPU, chạy được cả
trên máy Windows.

> ⚠️ Bộ dữ liệu sinh **trước 22/8** không có `offsets_bytes_*` → gate báo thiếu, phải sinh
> lại. Không ảnh hưởng centroid Phase 2 đã có.

---

## 5. Phase 2 — structure-aware clustering (Idea 1)

Ba nhánh chạy trên cùng một nền: `sa` (đối chứng, bằng SA gốc), `hard_boundary`,
`struct_hierarchy`. Một script làm cả Phase 2 rồi Phase 5 luôn:

```bash
bash scripts/run_phase2_phase5_lcc.sh
```

Thứ tự trong script **có chủ đích**: chạy `hard_boundary` trước, rồi smoke Phase 5 trên 3 mẫu
ngay lập tức. Nếu `phase5_recall.py` hỏng thì hỏng sau ~1 giờ chứ không phải sau ~3 giờ.

Cấu hình chốt 23/8, không đổi giữa chừng: `--force_chat`, `--dataset lcc`,
`--level function --level_l1 class`, `--percent_clusters 5`, `--observation_window 100`.

Thời gian: mỗi nhánh ~55-70 phút, ba nhánh ~3 giờ, Phase 5 ~1 giờ.
Chạy lại được — `offline_clustering_struct.py` bỏ qua mẫu đã đủ file.

**Kết quả cuối:**

```
/workspace/p2_invariants_instruct.log    <- kiểm bất biến Phase 2, ĐỌC CÁI NÀY
$P2_DIR/{sa,hard_boundary,struct_hierarchy}/lcc/   <- centroid ba nhánh, ~19 GB
```

`$P2_DIR` mặc định là `/workspace/p2-instruct`, đổi bằng biến môi trường `P2_DIR`.

---

## 6. Phase 5 — C2 recall@budget

Chạy kèm trong script Phase 2 ở trên, hoặc riêng:

```bash
python phase5_recall.py qwen2.5-coder-7b-instruct --force_chat --dataset lcc \
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
   (`phase5_lcc.json`, `p2_invariants_instruct.log`) lẫn với hàng chục file log khác.
   Nhất quán hơn thì nên là `/workspace/phase2_results/` và `/workspace/phase5_results/`.
2. **`SQA_RESULT_DIR` tên là `phase0_results` nhưng chứa cả env record của Phase 1**
   (`env_record_phase1.json`) và log của mọi gate.
3. **`$P2_DIR` mặc định `/workspace/p2-instruct` nhưng dữ liệu thật đang nằm ở
   `/workspace/struct-clusters`** (19 GB, từ lượt chạy trước với `P2_DIR` khác).
   Kiểm bằng `du -sh /workspace/struct-clusters/*` trước khi cho rằng phải chạy lại.

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

⚠️ **`scripts/phase1_gate.sh` thì ngược lại.** `check_phase1.py` có `--log_md` mặc định trỏ
thẳng vào `EXPERIMENT_LOG.md`, nên chạy gate Phase 1 trên pod **sẽ** sửa file trong git. Muốn
giữ một chiều thì thêm cờ tắt:

```bash
python scripts/check_phase1.py ... --no_log_md
```

Đã lỡ ghi rồi thì cất phần pod thêm ra ngoài trước khi đồng bộ code — xem [§9.3](#93-git-treo-trên-moosefs):

```bash
cp EXPERIMENT_LOG.md /workspace/EXPERIMENT_LOG_pod_<ngày>.md
git checkout HEAD -- "SqueezedAttention-simple improve-K-mean-AST/EXPERIMENT_LOG.md"
```
