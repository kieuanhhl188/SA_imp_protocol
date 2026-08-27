#!/bin/bash
# ====================================================================
# repro_lcc.sh — TÁI LẬP BASELINE SA TRÊN LCC + LongChat-7B, NHIỀU LƯỢT
#
# Thay cho scripts/phase0_gate.sh. Khác ở ba điểm:
#   1. Chỉ LCC. Bỏ RepoBench-P.
#   2. Chỉ All-KV và Sq-70%. Bỏ Sq-80%, Sq-90%, H-Sq-90%.
#   3. KHÔNG so với Table 2 của bài. Mốc là chính đường ống này: chạy N lượt rồi lấy
#      mean ± std cho từng cấu hình.
#
# NGUỒN NGẪU NHIÊN — đọc trước khi diễn giải std
# ----------------------------------------------
# pred.py giải mã tham lam (do_sample=False, num_beams=1), nên chạy lại cùng một bộ
# centroid sẽ ra output y hệt. Phương sai thật của Squeezed Attention nằm ở K-means:
# mỗi seed cho một phân hoạch khác, một tập centroid khác. Vì vậy mỗi lượt ở đây =
# một seed K-means riêng, không phải chạy lại pred.py với cùng centroid.
#
# All-KV không dùng centroid nên KHÔNG có nguồn ngẫu nhiên nào. Vẫn chạy đủ N lượt vì
# rẻ (~16 phút/lượt) và để có sàn nhiễu phần cứng đem so với std của Sq-70%. Kỳ vọng
# std của All-KV ≈ 0,00 — nếu khác 0 đáng kể thì bản thân môi trường đang không ổn định
# và mọi so sánh sau đều phải tính đến điều đó.
#
# CHI PHÍ (đo trên A100-80GB, 500 mẫu LCC)
# ----------------------------------------
#   Offline clustering : ~6h15 cho lượt forward, DÙNG CHUNG cho mọi seed
#                        (offline_clustering.py --seeds chạy K-means nhiều lần trên
#                        cùng một lượt forward → thêm seed chỉ tốn phần K-means)
#   pred All-KV        : ~16 phút / lượt
#   pred Sq-70%        : ~3h07 / lượt
#   → 3 seed ≈ 6h15 + 3×16ph + 3×3h07 ≈ 16 giờ.
#
# ĐÃ CÓ SẴN centroid seed 0 ở fixed-prompt-clusters/lcc/ (lượt 17/8). Dùng lại được:
#   ln -s /workspace/fixed-prompt-clusters /workspace/fixed-prompt-clusters_seed0
# Lưu ý: bộ đó sinh bằng KMeans(random_state=0), đúng bằng seed 0 sau bản vá này.
#
# ⚠️ ĐĨA LÀ RÀNG BUỘC CHẶT HƠN GPU
# --------------------------------
# Centroid LCC 500 mẫu với LongChat = **~68-71 GB / seed** (32 head KV, fp32 + label int64).
# Ba seed cùng lúc ≈ 205 GB. Volume 200 GB không chứa nổi, và ngày 16/8 volume thật chỉ
# được cấp ~50-55 GB.
#
# Tệ hơn: `/workspace` là MooseFS dùng chung nên `df` báo dung lượng CẢ CỤM, không biết gì
# về hạn mức riêng. Ghi vượt hạn mức thì `torch.save` KHÔNG raise — file bị cắt cụt im
# lặng, và vòng resume của offline_clustering.py thấy file tồn tại nên bỏ qua đúng mẫu
# hỏng đó. Vì vậy script này luôn chạy check_cluster_integrity.py (kiểm CRC) sau mỗi lượt
# clustering, và có --purge-after để xoá centroid của một seed ngay sau khi seed đó đã
# pred xong.
#
# TRÌNH TỰ AN TOÀN cho 3 seed trên volume ~150 GB (đỉnh ~136 GB):
#   ln -s /workspace/fixed-prompt-clusters /workspace/fixed-prompt-clusters_seed0
#   bash scripts/repro_lcc.sh --seeds "0"   --skip-cluster                # dùng lại 17/8
#   bash scripts/repro_lcc.sh --seeds "1 2" --purge-after                 # 1 lượt forward
#   bash scripts/repro_lcc.sh --seeds "0 1 2" --aggregate-only            # gộp
# Volume nhỏ hơn thì chạy từng seed một, mỗi lượt kèm --purge-after.
#
# USAGE
#   bash scripts/repro_lcc.sh                    # 3 seed, đầy đủ
#   bash scripts/repro_lcc.sh --seeds "0 1 2 3 4"
#   bash scripts/repro_lcc.sh --skip-cluster     # centroid đã có
#   bash scripts/repro_lcc.sh --aggregate-only   # chỉ gộp kết quả đã chạy
#   bash scripts/repro_lcc.sh --purge-after      # xoá centroid mỗi seed sau khi pred xong
#   bash scripts/repro_lcc.sh --limit 20         # smoke test
# ====================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../configs/phase0.sh"
cd "$SQA_REPO_ROOT"

SKIP_CLUSTER=0
AGGREGATE_ONLY=0
PURGE_AFTER=0
LIMIT=-1
SEEDS=("${SQA_SEEDS[@]}")

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-cluster)   SKIP_CLUSTER=1; shift ;;
    --aggregate-only) AGGREGATE_ONLY=1; SKIP_CLUSTER=1; shift ;;
    --purge-after)    PURGE_AFTER=1; shift ;;
    --seeds)          read -r -a SEEDS <<< "$2"; shift 2 ;;
    --limit)          LIMIT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

MODEL="$SQA_MODEL"
DATASET="lcc"
LIMIT_ARG=""
# Không viết `[ ... ] && LIMIT_ARG=...`: dưới `set -e`, test trả về sai ở cuối một
# danh sách lệnh sẽ làm script thoát ngay. LIMIT mặc định là -1 nên đó là ca thường gặp.
if [ "$LIMIT" -gt 0 ]; then
  LIMIT_ARG="--limit $LIMIT"
fi

mkdir -p "$SQA_RESULT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SQA_RESULT_DIR/logs"
mkdir -p "$LOG_DIR"
CONSOLE_LOG="$LOG_DIR/${TS}_repro_lcc.log"
exec > >(tee -a "$CONSOLE_LOG") 2>&1
set -o pipefail

# run_tag của lượt ứng với seed S là "sS" -> thư mục pred/<config>_runsS
RUN_TAGS=()
for S in "${SEEDS[@]}"; do RUN_TAGS+=("s${S}"); done

# thư mục centroid của một seed
cluster_root_for() { echo "${SQA_CLUSTER_DIR_PATTERN/\{seed\}/$1}"; }

# Xoá centroid của một seed sau khi seed đó đã pred + eval xong.
# Hai chốt trước khi rm: đường dẫn PHẢI khớp pattern (chứa '_seed'). Symlink thì chỉ gỡ
# link, không đụng vào đích — thư mục fixed-prompt-clusters của lượt 17/8 hay được
# symlink vào seed 0, xoá nhầm là mất 6 giờ GPU.
purge_seed() {
  local root="$1"
  case "$root" in
    *_seed*) ;;
    *) echo "    [BỎ QUA purge] '$root' không khớp pattern '*_seed*'"; return 0 ;;
  esac
  if [ -L "$root" ]; then
    echo "    purge: '$root' là symlink -> chỉ gỡ link, giữ nguyên đích"
    rm "$root"
  elif [ -d "$root" ]; then
    echo "    purge: xoá $(du -sh "$root" 2>/dev/null | cut -f1) tại $root"
    rm -rf "$root"
  else
    echo "    [BỎ QUA purge] '$root' không tồn tại"
  fi
}

echo "=================================================================="
echo "  TÁI LẬP BASELINE SA — LCC"
echo "  Model:        $MODEL"
echo "  Dataset:      $DATASET"
echo "  Cấu hình:     All-KV, Sq-70% (percentile=$SQA_PERCENTILE_GATE)"
echo "  Seed K-means: ${SEEDS[*]}   (run_tag: ${RUN_TAGS[*]})"
echo "  Centroid:     $SQA_CLUSTER_DIR_PATTERN/$DATASET/"
echo "  GPU visible:  $CUDA_VISIBLE_DEVICES"
echo "  Console log:  $CONSOLE_LOG"
echo "  Purge sau pred: $([ "$PURGE_AFTER" -eq 1 ] && echo "CÓ" || echo "không")"
echo "  Bắt đầu:      $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

if [ "$AGGREGATE_ONLY" -eq 0 ] && [ "$SKIP_CLUSTER" -eq 0 ]; then
  NEED=$(( ${#SEEDS[@]} * 70 ))
  echo ""
  echo "!!  Centroid LCC LongChat ~70 GB/seed -> ${#SEEDS[@]} seed cần ~${NEED} GB CÙNG LÚC."
  echo "    /workspace là MooseFS: 'df' báo dung lượng cả cụm, KHÔNG thấy hạn mức của bạn."
  echo "    Kiểm hạn mức thật trên dashboard RunPod trước khi để job chạy qua đêm."
  echo "    Không đủ chỗ thì chạy ít seed hơn mỗi lượt, kèm --purge-after."
  echo ""
fi

if [ "$AGGREGATE_ONLY" -eq 0 ]; then
  # ---------- 0. Môi trường ----------
  echo ""
  echo ">>> [0] Ghi môi trường"
  python scripts/record_env.py \
      --out "$SQA_RESULT_DIR/env_record.json" \
      --seed "$SQA_SEED" \
      --note "repro LCC: $MODEL, seed K-means ${SEEDS[*]}" \
      --strict

  # ---------- 1. Offline clustering, tất cả seed trong MỘT lượt forward ----------
  if [ "$SKIP_CLUSTER" -eq 0 ]; then
    echo ""
    echo ">>> [1] Offline clustering ${SQA_PERCENT_CLUSTERS}% — seed ${SEEDS[*]}"
    python offline_clustering.py "$MODEL" \
        --dataset "$DATASET" \
        --output_path "${SQA_CLUSTER_DIR_PATTERN}/${DATASET}/" \
        --seeds "${SEEDS[@]}" \
        --percent_clusters "$SQA_PERCENT_CLUSTERS" \
        --observation_window "$SQA_OBS_WINDOW" \
        --device "$SQA_DEVICE" \
        $LIMIT_ARG

    # Bắt buộc, không phải tuỳ chọn: MooseFS cắt cụt file im lặng khi vượt hạn mức và
    # torch.save KHÔNG raise. Vòng resume thấy file tồn tại nên bỏ qua đúng mẫu hỏng.
    # Chỉ kiểm CRC (zipfile.testzip) mới bắt được — rẻ hơn nhiều so với một lượt pred.py
    # chạy vài giờ rồi mới chết. Sự cố 16/8: hai file hỏng, mất ~5 giờ GPU.
    EXPECT_N=500
    if [ "$LIMIT" -gt 0 ]; then EXPECT_N="$LIMIT"; fi
    for S in "${SEEDS[@]}"; do
      echo ""
      echo ">>> [1b] Kiểm toàn vẹn centroid — seed $S"
      python scripts/check_cluster_integrity.py \
          "$(cluster_root_for "$S")/${DATASET}/" --expect "$EXPECT_N"
    done
  else
    echo ""
    echo ">>> [1] BỎ QUA offline clustering"
  fi

  # ---------- 2. Chạy từng lượt ----------
  cd "$SQA_REPO_ROOT/LongBench"
  for S in "${SEEDS[@]}"; do
    TAG="s${S}"
    CROOT="$(cluster_root_for "$S")"

    echo ""
    echo ">>> [2a] All-KV — lượt $TAG"
    python pred.py --model "$MODEL" --task "$DATASET" \
        --run_tag "$TAG" --seed "$SQA_SEED" --overwrite $LIMIT_ARG
    python eval.py --model "$MODEL" --run_tag "$TAG" $LIMIT_ARG

    echo ""
    echo ">>> [2b] Sq-70% — lượt $TAG (centroid: $CROOT)"
    python pred.py --model "$MODEL" --task "$DATASET" \
        --use_centroids \
        --percent_clusters "$SQA_PERCENT_CLUSTERS" \
        --percentile "$SQA_PERCENTILE_GATE" \
        --obs_window "$SQA_OBS_WINDOW" \
        --path_to_clusters "${CROOT}/" \
        --run_tag "$TAG" --seed "$SQA_SEED" --overwrite $LIMIT_ARG
    python eval.py --model "$MODEL" --use_centroids \
        --percent_clusters "$SQA_PERCENT_CLUSTERS" \
        --percentile "$SQA_PERCENTILE_GATE" --run_tag "$TAG" $LIMIT_ARG

    # Chỉ purge SAU khi eval đã ghi xong result.json — điểm số nằm trong pred/, centroid
    # không cần nữa và sinh lại được bất cứ lúc nào bằng chính seed đó.
    if [ "$PURGE_AFTER" -eq 1 ]; then
      echo ""
      echo ">>> [2c] Giải phóng đĩa — seed $S"
      purge_seed "$CROOT"
    fi
  done
  cd "$SQA_REPO_ROOT"
fi

# ---------- 3. Gộp mean ± std ----------
CFG_SQ="PC${SQA_PERCENT_CLUSTERS}_PERC${SQA_PERCENTILE_GATE}"
if [ "$LIMIT" -gt 0 ]; then
  CFG_BASE="baseline_lim${LIMIT}"
  CFG_SQ="${CFG_SQ}_lim${LIMIT}"
else
  CFG_BASE="baseline"
fi

echo ""
echo ">>> [3] Gộp kết quả"
python scripts/aggregate_runs.py \
    --model "$MODEL" --task "$DATASET" \
    --config "$CFG_BASE" --config "$CFG_SQ" \
    --run_tags "${RUN_TAGS[@]}" \
    --out "$SQA_RESULT_DIR/repro_lcc_${TS}" \
    --note "seed K-means ${SEEDS[*]}; console log ${CONSOLE_LOG}"

echo ""
echo ">>> Kết thúc: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Console log:  $CONSOLE_LOG"
echo "Tổng hợp:     $SQA_RESULT_DIR/repro_lcc_${TS}.md"

sleep 1
