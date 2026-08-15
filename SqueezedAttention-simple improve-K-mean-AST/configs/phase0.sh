#!/bin/bash
# ====================================================================
# configs/phase0.sh — CẤU HÌNH CHUNG, CHỐT MỘT LẦN
#
# Mọi phase sau (1..7) phải `source` file này thay vì hard-code lại,
# để ablation luôn chạy trên cùng một nền.
#
# Nguồn số: Hooper et al., ACL 2025, Section 6.1 + Appendix C + F.
# ====================================================================

# --- Model dùng cho gate Phase 0 (khớp Table 2 của bài) ---
export SQA_MODEL="longchat-v1.5-7b-32k"

# --- Task code trong LongBench ---
export SQA_CODE_DATASETS=("lcc" "repobench-p")

# --- Clustering (Section 6.1) ---
# single-level: số centroid = 5% chiều dài fixed context
export SQA_PERCENT_CLUSTERS=5
# hierarchical: L1 = 1%, L2 = 5%
export SQA_PERCENT_CLUSTERS_L1=1
export SQA_PERCENT_CLUSTERS_L2=5

# --- Calibration threshold (Appendix C) ---
# 100 token cuối của fixed context, giữ nguyên (không cluster)
export SQA_OBS_WINDOW=100

# --- Sparsity setting. Ánh xạ sang --percentile của pred.py ---
# Sq-70% -> 0.7 | Sq-80% -> 0.8 | Sq-90% -> 0.9
# (qlist trong squeezedattention/clustering.py là [0.5, 0.7, 0.8, 0.9])
export SQA_PERCENTILE_GATE=0.7

# --- Hierarchical: ngưỡng L1 loại bỏ 50% key trước khi lookup L2 (Section 6.1) ---
# pred.py gọi cái này là --percentile_lower
export SQA_PERCENTILE_LOWER=0.5

# --- Context ---
# 32K max context (Appendix F). model2maxlen.json đã set 31500 cho longchat.
export SQA_MAX_CONTEXT=32768

# --- Seed ---
export SQA_SEED=42

# --- Đường dẫn (đổi theo máy bạn) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SQA_REPO_ROOT="$REPO_ROOT"
export SQA_CLUSTER_DIR="${SQA_CLUSTER_DIR:-$REPO_ROOT/fixed-prompt-clusters}"
export SQA_RESULT_DIR="${SQA_RESULT_DIR:-$REPO_ROOT/phase0_results}"

# --- GPU. Đặt CUDA_VISIBLE_DEVICES trước khi source nếu muốn cố định. ---
# pred.py tự spawn 1 process / GPU thấy được -> muốn 1 GPU thì set CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SQA_DEVICE=0   # index TRONG danh sách GPU đã visible
