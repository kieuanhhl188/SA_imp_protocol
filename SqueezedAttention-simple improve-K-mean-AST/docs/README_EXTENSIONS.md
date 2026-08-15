# Extensions cho Squeezed Attention

Implement 2 hướng cải tiến trên top of [SqueezeAILab/SqueezedAttention](https://github.com/SqueezeAILab/SqueezedAttention):

- **Hướng 1: Layer-wise Adaptive Cluster Budget** — phân bổ budget cluster theo attention entropy mỗi layer
- **Hướng 2: Code-aware Clustering** — dùng AST + token-type để init centroid cho code dataset

---

## ⚠️ Lưu ý quan trọng về Novelty

Trước khi bắt đầu, bạn cần biết những công trình có thể bị overlap để **positioning paper đúng**:

1. **Wang & Gan (ICLR 2025) "SqueezeAttention"** — paper khác (trùng tên!) đã làm layer-wise budget cho **eviction-based** method (H2O, sliding window). Bạn phải nhấn mạnh: contribution của bạn là **cluster-based** (top of Hooper's Squeezed Attention), và signal là **attention entropy** (không phải cosine similarity của hidden state).

2. **Multipole Attention (NeurIPS 2025)** — follow-up của chính Hooper cho reasoning task, có hierarchical clustering. Đảm bảo positioning của bạn không bị nhầm với họ.

3. **AST-aware cho KV cache** — chưa có ai làm trực tiếp. Đây là contribution sạch nhất.

---

## Cài đặt

### Bước 1: Clone và setup repo gốc

```bash
git clone https://github.com/SqueezeAILab/SqueezedAttention.git
cd SqueezedAttention

# Tạo conda env theo hướng dẫn gốc
conda create --name squeezed python=3.9 -y
conda activate squeezed

cd transformers
pip install -e .
cd ..
pip install -e .
```

### Bước 2: Cài thêm dependency cho extensions

```bash
# Cho AST parsing (Hướng 2)
pip install tree-sitter tree-sitter-languages

# Cho phân tích/plot (Hướng 1)
pip install matplotlib numpy
```

### Bước 3: Copy các file extension vào repo

Copy 3 file mới sau vào **root của repo** (cùng cấp với `offline_clustering.py`):

```
SqueezedAttention/
├── offline_clustering.py        # gốc (giữ nguyên)
├── offline_clustering_v2.py     # ← mới (Hướng 1 + 2)
├── adaptive_budget.py           # ← mới (Hướng 1 module)
├── ast_clustering.py            # ← mới (Hướng 2 module)
├── scripts/
│   ├── preliminary_gate_h1.sh   # ← mới
│   ├── full_experiment.sh       # ← mới
│   └── analyze_entropy.py       # ← mới
├── LongBench/                   # gốc
├── squeezedattention/           # gốc
├── transformers/                # gốc
└── utils/                       # gốc
```

---

## ⚙️ Hardware Setup cho 16GB VRAM

Repo gốc dùng `Llama-2-7B-32K` (MHA, no quantization) → cần ~50GB context-32K. **Không fit 16GB.**

**Giải pháp cho bạn:**

### Option A: Llama-3.2-3B (recommend cho prototyping)
- 3B params, GQA → KV cache nhỏ
- Context 8K-16K thoải mái với FP16
- Sửa `LongBench/config/model2path.json`:
  ```json
  {
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct"
  }
  ```
- Sửa `LongBench/config/model2maxlen.json`:
  ```json
  {
    "llama3.2-3b": 8192
  }
  ```

### Option B: Llama-2-7B-32K với 4-bit quantization
- Cần thêm bitsandbytes
- Sửa load_model trong `offline_clustering_v2.py`:
  ```python
  from transformers import BitsAndBytesConfig
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
  )
  model = LlamaForCausalLM.from_pretrained(
      model_path, config=config,
      quantization_config=bnb_config,
  )
  ```

### Option C: Smaller context cho prototyping
- Set `max_length = 4096` trong `model2maxlen.json`
- Đủ để verify pipeline, sau scale up

**Khuyến nghị workflow:** Bắt đầu với Option A + context 4K → verify pipeline → mới scale lên 7B với 4-bit.

---

## Chạy Thí Nghiệm

### Bước 1: Preliminary Gate Experiment (BẮT BUỘC, không skip)

Trước khi đầu tư thời gian vào full experiment, **xác minh hypothesis** rằng entropy có pattern đủ rõ giữa các layer.

```bash
bash scripts/preliminary_gate_h1.sh
```

Sau khi xong (~30 phút trên RTX 4080):

```bash
python scripts/analyze_entropy.py --log_dir output_gate_h1/baseline
```

Đọc kỹ output:
- **Diagnostic ratio < 0.1**: entropy quá uniform → Hướng 1 không cho gain. STOP, chuyển hướng khác.
- **Diagnostic ratio 0.1-0.3**: Có signal nhưng vừa phải. Tiếp tục nhưng kỳ vọng gain nhỏ.
- **Diagnostic ratio > 0.3**: Signal rõ. GREEN LIGHT.

Đồng thời check `entropy_plot.png` — bạn cần thấy:
1. Subplot (a): có variance rõ giữa các layer
2. Subplot (c): CV thấp (< 0.3) → consistent across samples

### Bước 2: Online Evaluation cho gate experiment

Sau khi clustering xong, chạy online eval (giống quy trình gốc Squeezed Attention):

```bash
cd LongBench

# Edit run_evaluation.sh để point tới output của bước 1
# Cụ thể: set CENTROIDS_DIR=../output_gate_h1/baseline (cho baseline)
# Và ../output_gate_h1/adaptive-linear (cho adaptive)

bash run_evaluation.sh
```

So sánh accuracy:
- **adaptive-linear > baseline** với margin ≥ 1 điểm → Hướng 1 promising
- **adaptive-linear ≈ baseline** → cần thêm strategy hoặc rethink

### Bước 3: Full Experiment

Chỉ chạy nếu Bước 1-2 cho green light:

```bash
bash scripts/full_experiment.sh
```

Estimated time: 6-24 giờ tùy hardware.

---

## Kiểm tra correctness của code

Trước khi run full, test nhanh với 1 sample để đảm bảo không crash:

```bash
# Subset 1 sample của TREC
python offline_clustering_v2.py llama3.2-3b \
    --dataset trec \
    --output_path output_test \
    --percent_clusters 10 \
    --adaptive_budget \
    --device 0
```

Nếu chạy được 1-2 sample không lỗi → pipeline OK.

---

## Cách Cấu trúc Experiment cho Paper

### Bảng chính (cho Hướng 1)

| Method | TREC | TriviaQA | SAMSum | QASPer | MFQA | Avg |
|--------|------|----------|--------|--------|------|-----|
| Squeezed (baseline, K=5%) | ? | ? | ? | ? | ? | ? |
| Adaptive-Linear (5%) | ? | ? | ? | ? | ? | ? |
| Adaptive-Pyramid (5%) | ? | ? | ? | ? | ? | ? |
| Adaptive-Inverse (5%, neg control) | ? | ? | ? | ? | ? | ? |

→ Mong đợi: Linear/Pyramid > Baseline > Inverse. Nếu Inverse thắng baseline, đó là tín hiệu xấu (signal sai chiều).

### Pareto curve (KV budget vs Accuracy)

Sweep `percent_clusters` ∈ {2, 5, 10, 20} cho từng method, plot accuracy vs budget. Adaptive nên có curve trên baseline.

### Ablation cho Hướng 1

- Strategy: linear vs softmax vs pyramid vs inverse
- Min budget: 1, 2, 5, 10
- Calibration set size: 5, 20, 100 samples

### Bảng cho Hướng 2

| Method | LCC | RepoBench-P | Avg |
|--------|-----|-------------|-----|
| Squeezed (baseline) | ? | ? | ? |
| Code-aware only | ? | ? | ? |
| Adaptive-Linear only | ? | ? | ? |
| Combo (Hướng 1 + 2) | ? | ? | ? |

### Diagnostic plots

- Heatmap entropy [layer × sample]
- Distribution cluster size khi adaptive vs uniform
- t-SNE của centroid với code-aware vs random init (cho 1-2 sample đại diện)

---

## Troubleshooting

### Lỗi OOM (Out of Memory)

```python
# Giảm max_length trong model2maxlen.json
# Hoặc dùng torch.cuda.empty_cache() sau mỗi sample
```

### `tree_sitter_languages` không cài được

Fallback: code sẽ tự động dùng regex (chậm + kém chính xác hơn nhưng vẫn run).

### Output không tương thích với `run_evaluation.sh` gốc

`run_evaluation.sh` load centroid từ `--centroids_path`. File output của bạn:
```
output_full/adaptive-linear/centroids_tensor_dict_<dataidx>_<avg_k>.pt
output_full/adaptive-linear/budgets_<dataidx>.pt
```

Trong `run_evaluation.sh`, set:
```bash
CENTROIDS_DIR=output_full/adaptive-linear
```

Online eval code có thể cần sửa nhẹ để load `budgets_<dataidx>.pt` thay vì assume uniform K. Xem `PATCHING_EVAL.md` để biết chi tiết.

### Entropy giống nhau giữa các layer

Có thể model dùng `flash_attention_2` đã optimize aggressive, dẫn tới Q/K capture không chuẩn. Thử:
```python
config._attn_implementation = "eager"  # thay vì "flash_attention_2"
```

---

## Roadmap nghiên cứu

### Phase 1: Verify (tuần 1-2)
- [ ] Setup environment + verify baseline reproduce
- [ ] Gate experiment Hướng 1
- [ ] Quyết định: tiếp tục H1 hay pivot

### Phase 2: Implement (tuần 3-6)
- [ ] Full sweep budget percentages
- [ ] Multiple budget strategies
- [ ] Hướng 2 trên LCC, RepoBench-P

### Phase 3: Analysis (tuần 7-10)
- [ ] Wall-clock benchmark (không chỉ FLOPs)
- [ ] Ablation đầy đủ
- [ ] Visualization: heatmap, Pareto curve, t-SNE

### Phase 4: Writing (tuần 11-14)
- [ ] Draft paper (8 trang ACL format)
- [ ] Internal review
- [ ] Submit workshop (ICML ES-FoMo, NeurIPS Efficient ML)

### Phase 5: Full submission (tuần 15+)
- [ ] Rebut workshop reviews
- [ ] Extend to ICLR/NeurIPS main conference

---

## Câu hỏi cho advisor / discussion

1. Calibration set: dùng cùng dataset hay cross-dataset? Generalization có quan trọng không?
2. Adaptive budget có nên cần re-calibrate khi model thay đổi không?
3. Wall-clock speedup: bao nhiêu là threshold để publish? (>10%? >20%?)
4. Code-aware: scope-based init có overlap với QUEST page-based không?
