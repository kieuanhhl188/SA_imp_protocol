# Phase 0 — Môi trường + tái lập baseline SA

Gate của toàn bộ protocol. **Không khớp số thì không chạy Phase 1/2.**

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

## 2. Chạy gate

```bash
export CUDA_VISIBLE_DEVICES=0        # cố định 1 GPU cho mọi lần đo
bash scripts/phase0_gate.sh          # All-KV + Sq-70%, đủ để gate
bash scripts/phase0_gate.sh --full   # thêm Sq-80/90% + H-Sq-90%
```

Script tự làm: ghi env → offline clustering → `pred.py` → `eval.py` → `check_gate.py`.

Chạy lại từng phần:
```bash
bash scripts/phase0_gate.sh --skip-cluster   # centroid đã có, chỉ chạy lại eval
python scripts/check_gate.py --model longchat-v1.5-7b-32k --pred_dir LongBench/pred
```

---

## 3. Số cần khớp (±0.3 điểm)

LongChat-7B-v1.5-32K, Table 2 của Hooper et al. (ACL 2025, `2025.acl-long.1568`):

| Config | Budget | LCC | RepoBench-P |
|---|---|---|---|
| All KV | 1.000 | 56.64 | 53.20 |
| **Sq-70%** | 0.325 | **56.93** | **54.64** |
| Sq-80% | 0.225 | 57.17 | 52.83 |
| Sq-90% | 0.125 | 56.95 | 51.57 |
| H-Sq-90% | 0.122 | 57.20 | 51.89 |

Toàn bộ bảng (kèm LLaMA-2-7B-32K, LWM) nằm trong [scripts/reference_table2.json](../scripts/reference_table2.json).

⚠️ **Đừng dùng số trong `LongBench/README.md`** (LCC 53.0 / RB 55.3). Đó là số All-KV của LongBench gốc với prompt/truncation khác — không phải mốc để so.

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
| seed | 42 | — |

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

---

## 6. Còn nợ (cần GPU hoặc quyết định)

- [ ] **Chạy lại strict gate thật** — artifact cũ chỉ đạt khi nới tolerance lên ±2.0.
- [x] Ghi environment record — cần ghi lại trên pod mới sau khi chạy `setup_pod.sh --strict`.
- [ ] Quyết định: port Squeezed Attention sang `modeling_qwen2.py` ngay, hay giữ LLaMA cho Phase 0–2 và hoãn Qwen2.5-Coder tới trước Phase 6. Patch hiện **chỉ tồn tại trong `models/llama/`**; `models/qwen2/` hoàn toàn nguyên bản.
- [ ] Kiểm chứng GQA: đường code centroid lookup chưa từng chạy với `num_key_value_heads < num_heads` (LLaMA-2-7B-32K là MHA). Bắt buộc trước khi dùng Qwen.

---

## 7. Lỗi khác đã biết, chưa chặn Phase 0

- [squeezedattention/utils.py:44](../squeezedattention/utils.py#L44) — `build_chat(prompt, model_name)` dùng biến `model_name` chưa định nghĩa trong scope → `NameError`. Chỉ nổ với dataset **ngoài** `{trec, triviaqa, samsum, lcc, repobench-p}`. Gate không chạm tới, nhưng sẽ nổ ở Phase 6 nếu thêm task QA.
- `README_EXTENSIONS.md` trỏ tới `LongBench/run_evaluation.sh` — file này **không tồn tại**. Dùng `scripts/phase0_gate.sh` thay thế.
