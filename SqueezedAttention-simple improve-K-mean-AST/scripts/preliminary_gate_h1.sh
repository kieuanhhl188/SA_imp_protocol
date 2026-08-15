#!/bin/bash
# ====================================================================
# preliminary_gate_h1.sh
# Mục đích: Verify hypothesis Hướng 1 trước khi commit full experiment.
#
# Chạy trên 1 dataset nhỏ (TREC), save entropy log,
# kiểm tra có pyramid pattern không.
#
# Estimated time: ~30 phút trên 1 GPU 16GB với Llama-2-7B
# ====================================================================

set -e  # exit nếu có lỗi

MODEL="llama2-7b-32k"       # Đổi thành model bạn dùng
DATASET="trec"               # Bắt đầu với TREC (ngắn, fast)
DEVICE=0
PERCENT=5                    # 5% budget (tương đối nhỏ)
OUT_DIR="output_gate_h1"

echo "=================================================="
echo "  Hướng 1 Gate Experiment"
echo "  Model:   $MODEL"
echo "  Dataset: $DATASET"
echo "=================================================="

# Step 1: Baseline (Squeezed Attention gốc)
echo ""
echo ">>> [1/3] Baseline (uniform K per layer)"
python offline_clustering_v2.py $MODEL \
    --dataset $DATASET \
    --output_path $OUT_DIR \
    --percent_clusters $PERCENT \
    --device $DEVICE \
    --save_entropy_log

# Step 2: Adaptive linear
echo ""
echo ">>> [2/3] Adaptive Linear Strategy"
python offline_clustering_v2.py $MODEL \
    --dataset $DATASET \
    --output_path $OUT_DIR \
    --percent_clusters $PERCENT \
    --adaptive_budget --budget_strategy linear \
    --device $DEVICE \
    --save_entropy_log

# Step 3: Inverse (negative control - ablation)
# Nếu linear thắng inverse rõ ràng, đó là evidence rằng entropy signal đúng
echo ""
echo ">>> [3/3] Adaptive Inverse (negative control)"
python offline_clustering_v2.py $MODEL \
    --dataset $DATASET \
    --output_path $OUT_DIR \
    --percent_clusters $PERCENT \
    --adaptive_budget --budget_strategy inverse \
    --device $DEVICE

echo ""
echo "=================================================="
echo "  Gate experiment xong. Tiếp theo:"
echo "  1. Run LongBench/run_evaluation.sh trên cả 3 setup"
echo "  2. Compare scores. Nếu adaptive > baseline > inverse"
echo "     → green light cho Hướng 1."
echo "=================================================="
