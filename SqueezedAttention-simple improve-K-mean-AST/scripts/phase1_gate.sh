#!/bin/bash
# ====================================================================
# phase1_gate.sh — GATE PHASE 1: ban port Squeezed Attention sang Qwen2 (GQA)
#
# Phase 1.5 + 1.6 da co code va pass test CPU (scripts/test_gqa_port.py 20/20),
# nhung CHUA TUNG chay tren GPU. Script nay chay het duong ong tren N mau dau
# cua LCC de xac nhan bon thu, theo thu tu RE TRUOC DAT SAU — hong o buoc nao
# thi dung ngay o do, khong tra tien GPU cho buoc sau:
#
#   [1] Phase 1.4 cho Qwen  — tokenizer nhanh/cham co ra cung token id khong?
#                             (~1 phut, CPU. Lech la moi offset deu sai.)
#   [2] offline_clustering  — hook tra key 4 head (TRUOC repeat_kv) va cuML chay duoc
#   [3] integrity           — file centroid CRC dung, du bo ba cho moi dataidx
#   [4] pred All-KV         — duong khong-centroid cua Qwen chay va sinh output that
#   [5] pred Sq-70%         — duong centroid + GQA lookup chay
#   [6] check_phase1        — Sq-70% khong te hon All-KV -> tra dung nhom centroid
#
# Tieu chi PASS: Sq-70% >= All-KV - tolerance. Table 2 KHONG co Qwen nen day la
# tieu chi noi tai, khong phai tai lap so cua bai. Chi tiet o scripts/check_phase1.py.
#
# Usage:
#   bash scripts/phase1_gate.sh                  # 20 mau dau (mac dinh)
#   bash scripts/phase1_gate.sh --limit 50
#   bash scripts/phase1_gate.sh --full           # ca 500 mau LCC
#   bash scripts/phase1_gate.sh --skip-cluster   # dung lai centroid da co
#   bash scripts/phase1_gate.sh --data-only      # chi buoc [1], khong dung GPU
#   bash scripts/phase1_gate.sh --model qwen2.5-coder-7b-instruct   # doi model
#
# Chi phi uoc tinh cho 20 mau: ~30-45 phut ke ca tai model 15 GB (~$1).
# Qwen2.5-Coder-7B co 4 head KV (vs 32 cua LongChat) -> centroid nho hon ~8 lan,
# nen ca thoi gian clustering lan dung luong dia deu thap hon Phase 0 dang ke.
# DAY LA UOC TINH, CHUA DO. Phase 0 cho thay uoc tinh cua toi tung sai 1,5-60 lan.
# ====================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../configs/phase1.sh"
cd "$SQA_REPO_ROOT"

LIMIT="$SQA_PHASE1_LIMIT"
SKIP_CLUSTER=0
DATA_ONLY=0
MODEL_OVERRIDE=""
while [ $# -gt 0 ]; do
  case $1 in
    --limit) LIMIT="$2"; shift 2 ;;
    --model) MODEL_OVERRIDE="$2"; shift 2 ;;
    --full) LIMIT=0; shift ;;
    --skip-cluster) SKIP_CLUSTER=1; shift ;;
    --data-only) DATA_ONLY=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

MODEL="${MODEL_OVERRIDE:-$SQA_MODEL_CODE}"
DATASET="$SQA_PHASE1_TASK"

# Centroid cua Qwen phai nam THU MUC RIENG, khong tron voi centroid LongChat cua
# Phase 0. Ly do: ten file la `centroids_tensor_dict_<dataidx>_<K>.pt`, ma K tinh tu
# shared_prefix_length — von khac nhau giua hai tokenizer. Chung se KHONG de len nhau,
# nen loi se khong lo ra bang mot va cham; chi lam thu muc phinh gap doi va rat de
# tra nham file. Tach ra la xong.
#
# pred.py tu noi them '<dataset>/' vao --path_to_clusters, nen offline_clustering
# phai ghi vao dung "$CLUSTER_ROOT/$DATASET/".
CLUSTER_ROOT="$SQA_CLUSTER_DIR/$MODEL"
mkdir -p "$SQA_RESULT_DIR" "$SQA_PHASE1_DIR" "$CLUSTER_ROOT"

# --limit 0 / --full -> khong truyen co --limit cho script con
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
  LIMIT_ARG="--limit $LIMIT"
  SCOPE="$LIMIT mau dau"
else
  LIMIT_ARG=""
  SCOPE="toan bo dataset"
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SQA_RESULT_DIR/logs"
mkdir -p "$LOG_DIR"
CONSOLE_LOG="$LOG_DIR/${TS}_phase1_gate.log"
CONSOLE_LOG_REL="phase0_results/logs/${TS}_phase1_gate.log"
exec > >(tee -a "$CONSOLE_LOG") 2>&1
set -o pipefail

echo "=================================================================="
echo "  PHASE 1 GATE — port Squeezed Attention sang Qwen2 (GQA)"
echo "  Model:        $MODEL"
echo "  Dataset:      $DATASET   ($SCOPE)"
echo "  Cluster dir:  $CLUSTER_ROOT"
echo "  GPU visible:  $CUDA_VISIBLE_DEVICES"
echo "  Console log:  $CONSOLE_LOG"
echo "  Bat dau:      $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

# ---------- 0. Ghi lai moi truong ----------
echo ""
echo ">>> [0] Ghi moi truong"
python scripts/record_env.py \
    --out "$SQA_RESULT_DIR/env_record_phase1.json" \
    --seed "$SQA_SEED" \
    --note "phase1 gate: $MODEL on $DATASET ($SCOPE)"

# ---------- 1. Phase 1.4 cho Qwen: offset + kiem tokenizer ----------
# Chay TRUOC MOI THU vi day la buoc re nhat va la buoc de hong nhat khi doi model:
# offline_clustering.py dung tokenizer CHAM, script nay dung tokenizer NHANH de lay
# offset roi assert hai ben ra cung token id. Lech la exit 1 ngay tai day, thay vi
# lo ra sau nhieu gio clustering duoi dang unit_id gan sai key vector.
#
# Voi LongChat da chay that: 500/500 khop. Voi Qwen thi CHUA — Qwen2 dung BPE kieu
# GPT-2 (vocab.json + merges.txt) chu khong phai sentencepiece, ban cham la Python
# thuan, nen day la cau hoi mo that su chu khong phai thu tuc.
echo ""
echo ">>> [1] Phase 1.4 — offset token + kiem tokenizer nhanh/cham ($MODEL)"
#
# Ghi vao THU MUC RIENG theo model. File output ten la '<dataset>_meta.jsonl', khong
# co ten model trong do — chay de len bo offset 500 mau cua LongChat la mat, ma Phase 2
# thi khong co cach nao biet minh dang doc offset cua model nao.
PHASE14_LOG="$LOG_DIR/${TS}_phase1_prepare_data.log"
python scripts/prepare_code_data.py "$MODEL" \
    --dataset "$DATASET" \
    --output_path "$SQA_PHASE1_DIR/$MODEL" \
    $LIMIT_ARG 2>&1 | tee "$PHASE14_LOG"

TOK_SUMMARY="$(grep -E 'token id fast|sample bi truncate|template khong' "$PHASE14_LOG" \
               | tr -s ' ' | paste -sd '; ' - || echo 'khong doc duoc')"
echo ""
echo "    Tom tat: $TOK_SUMMARY"

if [ "$DATA_ONLY" -eq 1 ]; then
  echo ""
  echo ">>> --data-only: dung tai day, khong dung GPU."
  sleep 1
  exit 0
fi

# ---------- 2. Offline clustering ----------
if [ "$SKIP_CLUSTER" -eq 0 ]; then
  echo ""
  echo ">>> [2] Offline clustering (single-level, ${SQA_PERCENT_CLUSTERS}%) — $DATASET"
  echo "        Cho doi dong 'num_key_value_heads=4' o dau log: do la xac nhan"
  echo "        centroid duoc sinh TRUOC repeat_kv. Neu ra 28 thi dung ngay."
  python offline_clustering.py "$MODEL" \
      --dataset "$DATASET" \
      --output_path "${CLUSTER_ROOT}/${DATASET}/" \
      --percent_clusters "$SQA_PERCENT_CLUSTERS" \
      --observation_window "$SQA_OBS_WINDOW" \
      --device "$SQA_DEVICE" \
      $LIMIT_ARG
else
  echo ""
  echo ">>> [2] BO QUA offline clustering (--skip-cluster)"
fi

# ---------- 3. Kiem toan ven file centroid ----------
# Bai hoc Phase 0: hai file centroid hong lam mat ~5 gio pred.py. Quet CRC o day
# ton vai phut. Lam LUON, khong doi den khi hong.
echo ""
echo ">>> [3] Kiem toan ven file centroid (CRC)"
EXPECT_ARG=""
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then EXPECT_ARG="--expect $LIMIT"; fi
python scripts/check_cluster_integrity.py "${CLUSTER_ROOT}/${DATASET}/" \
    --jobs 8 $EXPECT_ARG

# ---------- 4. All-KV baseline ----------
cd "$SQA_REPO_ROOT/LongBench"
echo ""
echo ">>> [4] All-KV baseline — $DATASET"
python pred.py --model "$MODEL" --task "$DATASET" --seed "$SQA_SEED" --overwrite $LIMIT_ARG
python eval.py --model "$MODEL" $LIMIT_ARG

# ---------- 5. Sq-70% qua duong GQA ----------
echo ""
echo ">>> [5] Sq-70% (percentile=$SQA_PERCENTILE_GATE) — $DATASET"
python pred.py --model "$MODEL" --task "$DATASET" \
    --use_centroids \
    --percent_clusters "$SQA_PERCENT_CLUSTERS" \
    --percentile "$SQA_PERCENTILE_GATE" \
    --obs_window "$SQA_OBS_WINDOW" \
    --path_to_clusters "${CLUSTER_ROOT}/" \
    --seed "$SQA_SEED" --overwrite $LIMIT_ARG
python eval.py --model "$MODEL" --use_centroids \
    --percent_clusters "$SQA_PERCENT_CLUSTERS" --percentile "$SQA_PERCENTILE_GATE" $LIMIT_ARG

# ---------- 6. Kiem gate + ghi nhat ky ----------
cd "$SQA_REPO_ROOT"
echo ""
echo ">>> Ket thuc: $(date '+%Y-%m-%d %H:%M:%S')"

NOTE="Gate port Qwen2/GQA tren $DATASET ($SCOPE)"
if [ "$SKIP_CLUSTER" -eq 1 ]; then NOTE="$NOTE, dung lai centroid co san"; fi

GATE_RC=0
python scripts/check_phase1.py \
    --model "$MODEL" \
    --task "$DATASET" \
    --pred_dir LongBench/pred \
    --percent_clusters "$SQA_PERCENT_CLUSTERS" \
    --percentile "$SQA_PERCENTILE_GATE" \
    --tolerance "$SQA_PHASE1_TOLERANCE" \
    --env_record "$SQA_RESULT_DIR/env_record_phase1.json" \
    --console_log "$CONSOLE_LOG_REL" \
    --tokenizer_check "$TOK_SUMMARY" \
    --run_note "$NOTE" \
    $LIMIT_ARG || GATE_RC=$?

echo ""
echo "Console log:  $CONSOLE_LOG"
echo "Nhat ky:      $SQA_REPO_ROOT/EXPERIMENT_LOG.md"

sleep 1
exit $GATE_RC
