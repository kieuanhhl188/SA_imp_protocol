#!/bin/bash
# ====================================================================
# phase0_gate.sh — GATE MÔI TRƯỜNG
#
# Tái lập số Table 2 của Hooper et al. (ACL 2025) trên hai task code
# của LongBench: LCC và RepoBench-P, model LongChat-7B-v1.5-32K.
#
# Mốc cần khớp (±0.3 điểm):
#   All KV  : LCC 56.64 | RB 53.20
#   Sq-70%  : LCC 56.93 | RB 54.64   <- gate chính
#   Sq-80%  : LCC 57.17 | RB 52.83
#   Sq-90%  : LCC 56.95 | RB 51.57
#   H-Sq-90%: LCC 57.20 | RB 51.89
#
# Nếu KHÔNG khớp -> environment sai, dừng, không chạy Phase 1/2.
#
# Usage:
#   bash scripts/phase0_gate.sh                 # All KV + Sq-70% (đủ để gate)
#   bash scripts/phase0_gate.sh --full          # thêm Sq-80/90 + H-Sq-90
#   bash scripts/phase0_gate.sh --skip-cluster  # bỏ qua offline clustering (đã chạy rồi)
#
# Thời gian: offline clustering là phần đắt nhất (~vài chục phút/task trên 1 GPU).
# ====================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../configs/phase0.sh"
cd "$SQA_REPO_ROOT"

RUN_FULL=0
SKIP_CLUSTER=0
for arg in "$@"; do
  case $arg in
    --full) RUN_FULL=1 ;;
    --skip-cluster) SKIP_CLUSTER=1 ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

MODEL="$SQA_MODEL"
CLUSTER_DIR="$SQA_CLUSTER_DIR"
mkdir -p "$SQA_RESULT_DIR"

# --- Ghi toàn bộ console ra file, đồng thời vẫn hiện trên màn hình ---
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SQA_RESULT_DIR/logs"
mkdir -p "$LOG_DIR"
CONSOLE_LOG="$LOG_DIR/${TS}_phase0_gate.log"
# đường dẫn tương đối để link trong markdown chạy được
CONSOLE_LOG_REL="phase0_results/logs/${TS}_phase0_gate.log"
exec > >(tee -a "$CONSOLE_LOG") 2>&1
set -o pipefail

echo "=================================================================="
echo "  PHASE 0 GATE"
echo "  Model:        $MODEL"
echo "  Datasets:     ${SQA_CODE_DATASETS[*]}"
echo "  Cluster dir:  $CLUSTER_DIR"
echo "  GPU visible:  $CUDA_VISIBLE_DEVICES"
echo "  Console log:  $CONSOLE_LOG"
echo "  Nhật ký:      $SQA_REPO_ROOT/EXPERIMENT_LOG.md"
echo "  Bắt đầu:      $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

# ---------- 0. Ghi lại môi trường ----------
echo ""
echo ">>> [0] Ghi môi trường"
python scripts/record_env.py \
    --out "$SQA_RESULT_DIR/env_record.json" \
    --seed "$SQA_SEED" \
    --note "phase0 gate: $MODEL on ${SQA_CODE_DATASETS[*]}"

# ---------- 1. Offline clustering ----------
if [ "$SKIP_CLUSTER" -eq 0 ]; then
  for DATASET in "${SQA_CODE_DATASETS[@]}"; do
    echo ""
    echo ">>> [1] Offline clustering (single-level, ${SQA_PERCENT_CLUSTERS}%) — $DATASET"
    python offline_clustering.py "$MODEL" \
        --dataset "$DATASET" \
        --output_path "${CLUSTER_DIR}/${DATASET}/" \
        --percent_clusters "$SQA_PERCENT_CLUSTERS" \
        --observation_window "$SQA_OBS_WINDOW" \
        --device "$SQA_DEVICE"

    if [ "$RUN_FULL" -eq 1 ]; then
      echo ""
      echo ">>> [1b] Offline clustering (hierarchical, L1=${SQA_PERCENT_CLUSTERS_L1}% / L2=${SQA_PERCENT_CLUSTERS_L2}%) — $DATASET"
      python offline_clustering.py "$MODEL" \
          --dataset "$DATASET" \
          --output_path "${CLUSTER_DIR}/${DATASET}/" \
          --hierarchical_lookup \
          --percent_clusters "$SQA_PERCENT_CLUSTERS_L1" \
          --percent_clusters_l2 "$SQA_PERCENT_CLUSTERS_L2" \
          --observation_window "$SQA_OBS_WINDOW" \
          --device "$SQA_DEVICE"
    fi
  done
else
  echo ""
  echo ">>> [1] BỎ QUA offline clustering (--skip-cluster)"
fi

# ---------- 2. Online evaluation ----------
cd "$SQA_REPO_ROOT/LongBench"

# 2a. Baseline All-KV (không dùng centroid) — trần accuracy
for DATASET in "${SQA_CODE_DATASETS[@]}"; do
  echo ""
  echo ">>> [2a] All-KV baseline — $DATASET"
  python pred.py --model "$MODEL" --task "$DATASET" --seed "$SQA_SEED" --overwrite
done
python eval.py --model "$MODEL"

# 2b. Sq-70% — GATE CHÍNH
for DATASET in "${SQA_CODE_DATASETS[@]}"; do
  echo ""
  echo ">>> [2b] Sq-70% (percentile=$SQA_PERCENTILE_GATE) — $DATASET"
  python pred.py --model "$MODEL" --task "$DATASET" \
      --use_centroids \
      --percent_clusters "$SQA_PERCENT_CLUSTERS" \
      --percentile "$SQA_PERCENTILE_GATE" \
      --obs_window "$SQA_OBS_WINDOW" \
      --path_to_clusters "${CLUSTER_DIR}/" \
      --seed "$SQA_SEED" --overwrite
done
python eval.py --model "$MODEL" --use_centroids \
    --percent_clusters "$SQA_PERCENT_CLUSTERS" --percentile "$SQA_PERCENTILE_GATE"

# 2c. Sq-80% / Sq-90% / H-Sq-90% (chỉ khi --full)
if [ "$RUN_FULL" -eq 1 ]; then
  for PERC in 0.8 0.9; do
    for DATASET in "${SQA_CODE_DATASETS[@]}"; do
      echo ""
      echo ">>> [2c] Sq-$(python -c "print(int(float('$PERC')*100))")% — $DATASET"
      python pred.py --model "$MODEL" --task "$DATASET" \
          --use_centroids \
          --percent_clusters "$SQA_PERCENT_CLUSTERS" \
          --percentile "$PERC" \
          --obs_window "$SQA_OBS_WINDOW" \
          --path_to_clusters "${CLUSTER_DIR}/" \
          --seed "$SQA_SEED" --overwrite
    done
    python eval.py --model "$MODEL" --use_centroids \
        --percent_clusters "$SQA_PERCENT_CLUSTERS" --percentile "$PERC"
  done

  for DATASET in "${SQA_CODE_DATASETS[@]}"; do
    echo ""
    echo ">>> [2d] H-Sq-90% (L1=${SQA_PERCENT_CLUSTERS_L1}%, L2=${SQA_PERCENT_CLUSTERS_L2}%, "
    echo "         L2 threshold=0.9, L1 threshold=${SQA_PERCENTILE_LOWER}) — $DATASET"
    python pred.py --model "$MODEL" --task "$DATASET" \
        --use_centroids --hierarchical_lookup \
        --percent_clusters "$SQA_PERCENT_CLUSTERS_L1" \
        --percent_clusters_l2 "$SQA_PERCENT_CLUSTERS_L2" \
        --percentile 0.9 \
        --percentile_lower "$SQA_PERCENTILE_LOWER" \
        --obs_window "$SQA_OBS_WINDOW" \
        --path_to_clusters "${CLUSTER_DIR}/" \
        --seed "$SQA_SEED" --overwrite
  done
  python eval.py --model "$MODEL" --use_centroids --hierarchical_lookup \
      --percent_clusters "$SQA_PERCENT_CLUSTERS_L1" \
      --percent_clusters_l2 "$SQA_PERCENT_CLUSTERS_L2" \
      --percentile 0.9 --percentile_lower "$SQA_PERCENTILE_LOWER"
fi

# ---------- 3. Kiểm tra gate + ghi nhật ký ----------
cd "$SQA_REPO_ROOT"
echo ""
echo ">>> Kết thúc: $(date '+%Y-%m-%d %H:%M:%S')"

if [ "$RUN_FULL" -eq 1 ]; then
  NOTE="Full gate (All-KV, Sq-70/80/90%, H-Sq-90%)"
else
  NOTE="Gate rút gọn (All-KV + Sq-70%)"
fi
if [ "$SKIP_CLUSTER" -eq 1 ]; then
  NOTE="$NOTE, dùng lại centroid có sẵn"
fi

GATE_RC=0
python scripts/check_gate.py \
    --model "$MODEL" \
    --pred_dir LongBench/pred \
    --tolerance 0.3 \
    --env_record "$SQA_RESULT_DIR/env_record.json" \
    --console_log "$CONSOLE_LOG_REL" \
    --run_note "$NOTE" || GATE_RC=$?

echo ""
echo "Console log:  $CONSOLE_LOG"
echo "Nhật ký:      $SQA_REPO_ROOT/EXPERIMENT_LOG.md"

# cho tee trong process substitution kịp flush trước khi shell thoát
sleep 1
exit $GATE_RC
