# Phase 0 — Môi trường + tái lập baseline SA

Nền cho toàn bộ protocol: dựng lại đúng đường ống SA để mọi cải tiến về sau là một
ablation trên cùng một nền.

> **Thu hẹp phạm vi (27/8).** Cả bài chỉ kiểm chứng khả thi trên **một** cặp
> **LCC × LongChat-7B-v1.5-32K**, và **không** còn so kết quả với Table 2 của Hooper
> et al. Mốc bây giờ là **chính đường ống này**: chạy lặp nhiều lượt rồi lấy mean ± std
> cho từng cấu hình (All-KV, Sq-70%). Xem [§3](#3-tái-lập-nhiều-lượt-mean--std) và
> [§8](#8-những-gì-đã-bỏ-khỏi-phase-0) cho những gì bị bỏ.

---

## 1. Dựng môi trường

```bash
conda create -n sqa python=3.10 -y && conda activate sqa

# torch trước (chọn cu121/cu118 theo driver)
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# RAPIDS — squeezedattention/clustering.py import cupy + cuml ở top-level, thiếu là crash ngay
pip install cuml-cu12 cupy-cuda12x --extra-index-url=https://pypi.nvidia.com

# transformers PHẢI là bản fork trong repo (đã patch Squeezed Attention vào models/llama)
cd transformers && pip install -e . && cd ..
pip install -e .
```

⚠️ **Tuyệt đối không** `pip install -r LongBench/requirements.txt`. File đó ghim `transformers==4.31.0` và sẽ ghi đè bản fork → mọi cờ `use_centroids` biến mất, model chạy như bản gốc và số sẽ trông "gần đúng" một cách gây nhầm lẫn.

Kiểm tra:
```bash
python scripts/record_env.py --out phase0_results/env_record.json --note "setup check"
```
Phải thấy `transformers` trỏ vào `<repo>/transformers/src/...` và version `4.40.0.dev0`.

---

## 2. Chạy tái lập

```bash
export CUDA_VISIBLE_DEVICES=0        # cố định 1 GPU cho mọi lần đo
bash scripts/repro_lcc.sh            # 3 seed K-means, All-KV + Sq-70%, rồi gộp mean±std
```

Script tự làm: ghi env → offline clustering (mọi seed trong **một** lượt forward) →
`pred.py` → `eval.py` → `scripts/aggregate_runs.py`.

Chạy lại từng phần:
```bash
bash scripts/repro_lcc.sh --skip-cluster        # centroid đã có
bash scripts/repro_lcc.sh --aggregate-only      # chỉ gộp lại kết quả đã chạy
bash scripts/repro_lcc.sh --seeds "0 1 2 3 4"   # nhiều lượt hơn
bash scripts/repro_lcc.sh --limit 20            # smoke test
```

Centroid seed 0 của lượt 17/8 dùng lại được — nó vốn đã sinh bằng `random_state=0`:
```bash
ln -s "$PWD/fixed-prompt-clusters" "$PWD/fixed-prompt-clusters_seed0"
```

**Chi phí GPU** (A100-80GB, 500 mẫu LCC, đo từ log 17/8): offline clustering ~6h15 (dùng
chung cho mọi seed) · `pred.py` All-KV ~16 phút/lượt · `pred.py` Sq-70% ~3h07/lượt.
Ba seed ≈ **16 giờ**.

### ⚠️ Đĩa là ràng buộc chặt hơn GPU

Centroid LCC 500 mẫu với LongChat = **~68-71 GB / seed** (32 head KV, fp32 + label int64).
Ba seed cùng lúc ≈ **210 GB**. Volume 200 GB không chứa nổi; ngày 16/8 volume thật chỉ
được cấp ~50-55 GB và job chết ở mẫu 113/500.

`df` **không** phát hiện được: `/workspace` là MooseFS dùng chung, `df` báo dung lượng cả
cụm chứ không biết hạn mức riêng. Vượt hạn mức thì `torch.save` không raise — file bị cắt
cụt im lặng, và vòng resume của `offline_clustering.py` thấy file tồn tại nên **bỏ qua đúng
mẫu hỏng**. Vì vậy `repro_lcc.sh` luôn chạy `check_cluster_integrity.py` (kiểm CRC từng
file) sau mỗi lượt clustering.

**Trình tự an toàn cho 3 seed** (đỉnh ~140 GB, tổng ~16 giờ GPU):
```bash
ln -s /workspace/fixed-prompt-clusters /workspace/fixed-prompt-clusters_seed0
bash scripts/repro_lcc.sh --seeds "0"   --skip-cluster       # dùng lại centroid 17/8
bash scripts/repro_lcc.sh --seeds "1 2" --purge-after        # 1 lượt forward cho 2 seed
bash scripts/repro_lcc.sh --seeds "0 1 2" --aggregate-only   # gộp mean±std
```
Volume nhỏ hơn thì chạy từng seed một, mỗi lượt kèm `--purge-after` (đỉnh ~70 GB, nhưng
tốn 3 lượt forward → ~28 giờ). `--purge-after` chỉ xoá thư mục khớp `*_seed*`, và với
symlink thì chỉ gỡ link — không đụng vào `fixed-prompt-clusters` gốc.

---

## 3. Tái lập nhiều lượt: mean ± std

Không còn số đích từ ngoài. Cái được báo cáo là:

| Cấu hình | n | mean | std | min | max | số mẫu đổi điểm giữa các lượt |
|---|---|---|---|---|---|---|
| baseline (All-KV) | 3 | 54,83 | 0,00 | 54,83 | 54,83 | 0/500 (mọi cặp) |
| PC5_PERC0.7 (Sq-70%) | 3 | 56,36 | 0,28 | 56,08 | 56,63 | 66/500 (17↔28/8) · 13/500 (28↔29/8) |

n=3 chốt 29/8 (17/8 · 28/8 · 29/8), **cả ba đều seed K-means 0** → std 0,28 là sàn nhiễu
kernel SA, chưa phải phương sai theo seed. Kết quả:
`LongBench/pred/longchat-v1.5-7b-32k_{baseline,PC5_PERC0.7}_runs0{,b}/`.

Hiệu số ghép cặp Sq-70% − All-KV trên từng mẫu (bootstrap KTC95, 20.000 lần):

| Lượt | Hiệu TB | KTC95 bootstrap | % lấy mẫu lại ≤ 0 |
|---|---:|---|---:|
| 17/8 | +1,25 | [−0,10; +2,59] *(xấp xỉ chuẩn, p=0,39)* | — |
| 28/8 | +1,80 | [+0,56; +3,13] | 0,2% |
| 29/8 | +1,54 | [+0,31; +2,87] | 0,7% |

Hai lượt gần nhất loại trừ 0. Hiệu ứng nhỏ, cỡ +1 đến +2 điểm.

### ⚠️ Đường ống này gần như tất định — đọc trước khi diễn giải std

Trước bản vá 27/8 thì **lặp lại bao nhiêu lượt cũng ra std = 0,00**, vì:

- `pred.py` giải mã tham lam (`do_sample=False`, `num_beams=1`) → `--seed` của
  torch/numpy không đổi được output một chút nào;
- [squeezedattention/clustering.py](../squeezedattention/clustering.py) **hardcode**
  `KMeans(random_state=0)` → `--seed` không bao giờ tới được K-means, centroid không đổi.

Nguồn phương sai thật của Squeezed Attention là **khởi tạo K-means**. Bản vá 27/8 đưa
seed vào `random_state`, và `offline_clustering.py --seeds 0 1 2` sinh nhiều bộ centroid
**trong cùng một lượt forward** (lượt forward chiếm phần lớn thời gian và không phụ
thuộc seed). Vì vậy trong `repro_lcc.sh`, **một lượt = một seed K-means**, không phải
chạy lại `pred.py` trên cùng bộ centroid.

All-KV không đụng centroid nên vẫn không có nguồn ngẫu nhiên nào; nó chạy đủ N lượt để
cho **sàn nhiễu phần cứng** đem so với std của Sq-70%. Kỳ vọng std ≈ 0,00 — khác 0 đáng
kể tức là môi trường không ổn định và mọi so sánh sau phải tính đến điều đó.

`aggregate_runs.py` in thẳng cột "số mẫu đổi điểm giữa các lượt" và cảnh báo khi cột đó
bằng 0, để không đọc nhầm "std = 0" thành "đã đo được phương sai".

---

## 4. Cấu hình đã chốt

Tất cả nằm trong [configs/phase0.sh](../configs/phase0.sh). Mọi phase sau phải `source` file này, không hard-code lại.

| Tham số | Giá trị | Nguồn |
|---|---|---|
| số centroid single-level | 5% fixed context | Section 6.1 |
| hierarchical L1 / L2 | 1% / 5% | Section 6.1 |
| observation window | 100 token cuối, giữ nguyên không cluster | Appendix C |
| ngưỡng L1 (hierarchical) | loại 50% key → `--percentile_lower 0.5` | Section 6.1 |
| max context | 32K (`model2maxlen` = 31500) | Appendix F |
| seed torch/numpy | 42 | — (không ảnh hưởng output: giải mã tham lam) |
| seed K-means | 0, 1, 2 | — (nguồn phương sai duy nhất) |

**Ánh xạ sparsity → CLI:** Sq-70% ⇒ `--percentile 0.7`, Sq-80% ⇒ `0.8`, Sq-90% ⇒ `0.9`. Danh sách quantile khả dụng là `[0.5, 0.7, 0.8, 0.9]` (hard-code trong [squeezedattention/clustering.py](../squeezedattention/clustering.py#L173)) — muốn mức khác phải sửa `qlist` **và** chạy lại offline clustering.

---

## 5. Đã sửa trong Phase 0

| Sửa gì | File |
|---|---|
| Thêm `requirements.txt` đầy đủ (cuml, cupy, triton, flash-attn…) | [requirements.txt](../requirements.txt) |
| `model2path.json` trỏ path chết `/home/chooper/...` → HF repo id | [LongBench/config/model2path.json](../LongBench/config/model2path.json) |
| **Bug**: offline lưu `hierarchical_lookup_*`, online load `hierarchical_centroids_*` → hierarchical crash | [offline_clustering.py](../offline_clustering.py), [offline_clustering_v2.py](../offline_clustering_v2.py) |
| **Bug**: `pred.py` ghi jsonl chế độ append, chạy lại là nhân đôi prediction → `eval.py` ra số sai. Thêm `--overwrite` + cảnh báo | [LongBench/pred.py](../LongBench/pred.py) |
| **Bug**: `seed_everything` chỉ chạy ở process cha, `mp.spawn` không kế thừa. Seed lại trong con | [LongBench/pred.py](../LongBench/pred.py) |
| Thêm `--seed` (chuẩn bị cho mean±std ≥3 seed) | [LongBench/pred.py](../LongBench/pred.py) |
| **Bug**: `KMeans(random_state=0)` hardcode → `--seed` không tới được K-means, mọi lượt chạy ra centroid y hệt và std luôn = 0 | [squeezedattention/clustering.py](../squeezedattention/clustering.py) |
| `--seeds` + `{seed}` trong `--output_path`: nhiều bộ centroid từ **một** lượt forward | [offline_clustering.py](../offline_clustering.py) |
| `--run_tag`: tách thư mục kết quả của các lượt lặp cùng cấu hình (không có nó thì lượt sau ghi đè lượt trước) | [LongBench/pred.py](../LongBench/pred.py), [LongBench/eval.py](../LongBench/eval.py) |

---

## 6. Còn nợ (cần GPU hoặc quyết định)

- [ ] **Chạy `scripts/repro_lcc.sh` trên pod** — chưa từng chạy với code sau bản vá 27/8.
- [x] Ghi environment record — cần ghi lại trên pod mới sau khi chạy `setup_pod.sh --strict`.
- [ ] Quyết định: port Squeezed Attention sang `modeling_qwen2.py` ngay, hay giữ LLaMA cho Phase 0–2 và hoãn Qwen2.5-Coder tới trước Phase 6. Patch hiện **chỉ tồn tại trong `models/llama/`**; `models/qwen2/` hoàn toàn nguyên bản.
- [ ] Kiểm chứng GQA: đường code centroid lookup chưa từng chạy với `num_key_value_heads < num_heads` (LLaMA-2-7B-32K là MHA). Bắt buộc trước khi dùng Qwen.

---

## 7. Lỗi khác đã biết, chưa chặn Phase 0

- [squeezedattention/utils.py:44](../squeezedattention/utils.py#L44) — `build_chat(prompt, model_name)` dùng biến `model_name` chưa định nghĩa trong scope → `NameError`. Chỉ nổ với dataset **ngoài** `{trec, triviaqa, samsum, lcc, repobench-p}`. Gate không chạm tới, nhưng sẽ nổ ở Phase 6 nếu thêm task QA.
- `README_EXTENSIONS.md` trỏ tới `LongBench/run_evaluation.sh` — file này **không tồn tại**. Dùng `scripts/repro_lcc.sh` thay thế.

---

## 8. Những gì đã bỏ khỏi Phase 0

Bỏ vì phạm vi thu lại còn "LCC × LongChat-7B, kiểm chứng khả thi". Code không xoá, chỉ
không còn nằm trên đường chạy mặc định — cần thì bật lại được.

| Bỏ gì | Vì sao | Hiện trạng |
|---|---|---|
| **RepoBench-P** | Không nằm trong phạm vi. Context dài gấp ~3,7 lần → clustering đắt hơn nhiều mà không trả lời thêm câu hỏi nào. Phase 1 vốn đã chốt LCC-only. | `SQA_CODE_DATASETS=("lcc")` |
| **Sq-80%, Sq-90%, H-Sq-90%** | Chỉ cần một điểm sparsity để so All-KV ↔ SA. Sq-70% là mức bài gốc dùng làm mốc chính. | `repro_lcc.sh` không chạy; `pred.py` vẫn nhận `--percentile` |
| **Chạy nhánh hierarchical (L1/L2)** | Là biến thể của SA, không phải nền để so. **Cấu hình 1%/5% + `percentile_lower 0.5` vẫn chốt trong `phase0.sh`** — chốt giá trị và chạy nhánh là hai việc khác nhau. | `offline_clustering.py --hierarchical_lookup` vẫn chạy được, không phải sửa config |
| **So với Table 2 + tolerance ±0,3** | Người dùng chốt không so nữa. Số của bài phụ thuộc prompt/truncation/phần cứng của họ; chênh 1,8 điểm trên All-KV không phân định được là môi trường sai hay là khác biệt hợp lệ, và nó chặn nhầm cả những lượt chạy tốt. | `scripts/check_gate.py` + `scripts/reference_table2.json` **để lại nhưng không còn dùng** — `check_phase1.py` vẫn import `env_summary` từ đó |
| **`scripts/phase0_gate.sh`** | Thay bằng `scripts/repro_lcc.sh`. | Giữ làm tham chiếu lịch sử |
| **Mục 0.5 "số đích từ Table 2", 0.8 "gate"** | Không còn tiêu chí PASS/FAIL từ ngoài. | Xem `EXPERIMENT_LOG.md` mục Phase 0 |

**Không** bỏ: `record_env.py` (`--strict`), `inspect_preds.py` (hậu kiểm prediction thô),
`compare_runs.py` (paired test), và toàn bộ 5 bug đã sửa ở [§5](#5-đã-sửa-trong-phase-0).
Những cái đó kiểm tính đúng đắn của đường ống, không phụ thuộc việc có mốc ngoài hay không.
