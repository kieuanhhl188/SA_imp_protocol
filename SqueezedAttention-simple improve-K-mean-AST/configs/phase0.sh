#!/bin/bash
# ====================================================================
# configs/phase0.sh — CẤU HÌNH CHUNG, CHỐT MỘT LẦN
#
# Mọi phase sau (1..7) phải `source` file này thay vì hard-code lại,
# để ablation luôn chạy trên cùng một nền.
#
# Nguồn số: Hooper et al., ACL 2025, Section 6.1 + Appendix C + F.
# ====================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- Model dùng cho Phase 0 ---
# CHỐT: cả bài chỉ kiểm chứng khả thi trên MỘT cặp (LCC, LongChat-7B-v1.5-32K).
export SQA_MODEL="longchat-v1.5-7b-32k"

# --- Task code trong LongBench ---
# CHỈ LCC. RepoBench-P đã bị bỏ khỏi phạm vi: context dài gấp ~3,7 lần nên clustering
# đắt hơn nhiều, mà nó không trả lời thêm câu hỏi nào khi mục tiêu chỉ là "cải tiến có
# khả thi không". Phase 1 (configs/phase1.sh) đã chốt LCC-only từ trước, đây là chỗ
# Phase 0 khớp lại cho thống nhất.
export SQA_CODE_DATASETS=("lcc")

# --- Clustering (Section 6.1) ---
# single-level: số centroid = 5% chiều dài fixed context
export SQA_PERCENT_CLUSTERS=5
# hierarchical: L1 = 1%, L2 = 5%
# ĐÃ CHỐT theo bài, dùng chung cho mọi thí nghiệm — kể cả khi Phase 0 chưa chạy nhánh này.
# Chốt giá trị và chạy nhánh là hai việc khác nhau: giá trị phải cố định từ đầu để bất kỳ
# phase nào bật hierarchical lên cũng nằm trên cùng một nền, không ai tự đặt lại con số.
# Phạm vi chạy hiện tại: chỉ LCC, All-KV + Sq-70% (xem scripts/repro_lcc.sh).
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
# pred.py gọi cái này là --percentile_lower. Cũng đã chốt, cùng lý do như L1/L2 ở trên.
export SQA_PERCENTILE_LOWER=0.5

# --- Context ---
# 32K max context (Appendix F). model2maxlen.json đã set 31500 cho longchat.
export SQA_MAX_CONTEXT=32768

# --- Seed ---
export SQA_SEED=42

# --- Seed K-means cho tái lập lặp nhiều lần ---
#
# ĐỌC KỸ: đây là NGUỒN NGẪU NHIÊN DUY NHẤT của đường ống.
#   * pred.py giải mã tham lam (do_sample=False, num_beams=1) → SQA_SEED ở trên không
#     đổi được output một chút nào.
#   * squeezedattention/clustering.py trước đây hardcode KMeans(random_state=0) → chạy
#     lại bao nhiêu lần cũng ra centroid y hệt.
# Cộng lại: nếu chỉ chạy lại pred.py nhiều lượt thì std = 0,00 và không đo được gì.
# Muốn mean±std có nghĩa thì phải sinh centroid với các seed K-means khác nhau.
export SQA_SEEDS=(0 1 2)

# Mỗi seed một thư mục centroid riêng. '{seed}' được offline_clustering.py thay thế.
# Dẫn xuất từ SQA_CLUSTER_DIR nên khi pod đã export SQA_CLUSTER_DIR=/workspace/... thì
# centroid của mọi seed vẫn nằm trên network volume, không rơi vào container disk.
# (Định nghĩa SQA_CLUSTER_DIR nằm dưới phần Đường dẫn; pattern được đặt ở đó, không ở đây.)

# --- Đường dẫn (đổi theo máy bạn) ---
export SQA_REPO_ROOT="$REPO_ROOT"
export SQA_CLUSTER_DIR="${SQA_CLUSTER_DIR:-$REPO_ROOT/fixed-prompt-clusters}"
# Phải đặt SAU SQA_CLUSTER_DIR. Xem chú thích ở mục Seed K-means.
export SQA_CLUSTER_DIR_PATTERN="${SQA_CLUSTER_DIR_PATTERN:-${SQA_CLUSTER_DIR}_seed{seed}}"
export SQA_RESULT_DIR="${SQA_RESULT_DIR:-$REPO_ROOT/phase0_results}"

# --- GPU. Đặt CUDA_VISIBLE_DEVICES trước khi source nếu muốn cố định. ---
# pred.py tự spawn 1 process / GPU thấy được -> muốn 1 GPU thì set CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SQA_DEVICE=0   # index TRONG danh sách GPU đã visible
