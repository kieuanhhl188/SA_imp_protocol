#!/bin/bash
# ====================================================================
# phase1_gate.sh — GATE PHASE 1: chuan bi du lieu code + kiem duong nap centroid
#
# CHOT 28/8: model chinh = LongChat-7B-v1.5-32K, LCC-only (configs/phase1.sh).
#
#   [1]  Phase 1.4          — sinh offset BYTE + KY TU tung token; tokenizer
#                             nhanh/cham co ra cung token id khong? (~1 phut, CPU)
#   [1b] check_phase1_data  — gate du lieu 5 buoc (CPU): ngon ngu tung mau, du mau,
#                             span AST khong lech byte/ky tu, fixed_context khong mat
#   [2] offline_clustering  — hook tra key TRUOC repeat_kv va cuML chay duoc
#   [3] integrity           — file centroid CRC dung, du bo ba cho moi dataidx
#   [4] pred All-KV         — duong khong-centroid chay va sinh output that
#   [5] pred Sq-70%         — duong nap + tra centroid chay
#   [6] check_phase1        — Sq-70% khong te hon All-KV -> tra dung nhom centroid
#
# >>> VOI LongChat (MHA, khong GQA), buoc [2]-[6] TRUNG voi Phase 0 (repro_lcc.sh):
#     cung model, cung LCC, cung duong modeling_llama. Phan RIENG that su cua Phase 1
#     la [1]+[1b] — chay CPU. Khuyen dung:
#         bash scripts/phase1_gate.sh --data-only
#     roi lay ket qua accuracy tu Phase 0. Chi chay het [2]-[6] khi muon mot lan
#     kiem doc lap bang paired test cua check_phase1.py — va luu y no sinh THEM ~70 GB
#     centroid o thu muc rieng (xem CLUSTER_ROOT ben duoi), trong khi dia la rang buoc
#     chat nhat cua pod. Khi do nen --skip-cluster + symlink toi centroid Phase 0.
#
# Tieu chi PASS: Sq-70% >= All-KV - tolerance, danh gia bang paired test tren hieu so
# tung mau. Chi tiet o scripts/check_phase1.py.
#
# Usage:
#   bash scripts/phase1_gate.sh --data-only      # [1]+[1b], khong dung GPU  <-- MAC DINH nen dung
#   bash scripts/phase1_gate.sh                  # 20 mau dau, het duong ong
#   bash scripts/phase1_gate.sh --limit 50
#   bash scripts/phase1_gate.sh --full           # ca 500 mau LCC
#   bash scripts/phase1_gate.sh --skip-cluster   # dung lai centroid da co
#   bash scripts/phase1_gate.sh --model qwen2.5-coder-7b-instruct   # doi model (can SQA_FORCE_CHAT=1)
#
# Chi phi [2]-[6] cho LongChat: bang Phase 0 — clustering ~6h15/500 mau, ~70 GB/seed.
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

# Chat template: bat buoc voi ban Instruct, phai dong bo o MOI buoc.
# Thieu co o mot buoc -> shared_prefix_length lech -> assert no sau khi nap model 15 GB.
if [ "${SQA_FORCE_CHAT:-0}" = "1" ]; then CHAT_ARG="--force_chat"; else CHAT_ARG=""; fi
echo ">>> force_chat: ${SQA_FORCE_CHAT:-0}"
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
echo "  PHASE 1 GATE — chuan bi du lieu code + kiem duong nap centroid"
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

# ---------- 1. Phase 1.4: offset + kiem tokenizer ----------
# Chay TRUOC MOI THU vi day la buoc re nhat va la buoc de hong nhat khi doi model:
# offline_clustering.py dung tokenizer CHAM, script nay dung tokenizer NHANH de lay
# offset roi assert hai ben ra cung token id. Lech la exit 1 ngay tai day, thay vi
# lo ra sau nhieu gio clustering duoi dang unit_id gan sai key vector.
#
# LongChat (sentencepiece): da chay that truoc day, 500/500 khop — nhung bo cu sinh
# TRUOC 22/8 khong co offsets_bytes_* nen phai sinh lai. Sinh lai KHONG dung centroid
# da co: offset ky tu va shared_prefix_length khong doi, chi them mot mang moi.
echo ""
echo ">>> [1] Phase 1.4 — offset token + kiem tokenizer nhanh/cham ($MODEL)"
#
# Ghi vao THU MUC RIENG theo model. File output ten la '<dataset>_meta.jsonl', khong
# co ten model trong do — chay de len bo offset 500 mau cua LongChat la mat, ma Phase 2
# thi khong co cach nao biet minh dang doc offset cua model nao.
PHASE14_LOG="$LOG_DIR/${TS}_phase1_prepare_data.log"
python scripts/prepare_code_data.py "$MODEL" $CHAT_ARG \
    --dataset "$DATASET" \
    --output_path "$SQA_PHASE1_DIR/$MODEL" \
    $LIMIT_ARG 2>&1 | tee "$PHASE14_LOG"

TOK_SUMMARY="$(grep -E 'token id fast|sample bi truncate|template khong' "$PHASE14_LOG" \
               | tr -s ' ' | paste -sd '; ' - || echo 'khong doc duoc')"
echo ""
echo "    Tom tat: $TOK_SUMMARY"

# ---------- 1b. Gate du lieu 5 buoc (CPU) ----------
# Chay NGAY sau khi sinh offset, TRUOC moi buoc dung GPU. Bat bon thu ma buoc [1] khong
# bat: ngon ngu tung mau co dung khong, du mau khong, span AST co lech byte/ky tu khong,
# fixed_context co mat khuc nao khong. Tat ca deu la loi IM LANG — khong crash, chi lam
# ket qua sai. Xem scripts/check_phase1_data.py.
echo ""
echo ">>> [1b] Gate du lieu Phase 1 (CPU, khong can GPU)"
python scripts/check_phase1_data.py "$MODEL" $CHAT_ARG \
    --dataset "$DATASET" \
    --phase1_dir "$SQA_PHASE1_DIR" \
    $LIMIT_ARG 2>&1 | tee "$LOG_DIR/${TS}_phase1_data_gate.log"
DATA_GATE_RC="${PIPESTATUS[0]}"
if [ "$DATA_GATE_RC" -ne 0 ]; then
  echo ""
  echo ">>> [1b] FAIL — dung lai, khong tra tien GPU cho du lieu hong."
  exit 1
fi

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
  echo "        Kiem dong 'num_key_value_heads=' o dau log: phai bang so head KV THAT"
  echo "        cua model (LongChat=32, Qwen2.5-Coder=4). Ra so head Q la hook sai cho,"
  echo "        centroid bi sinh SAU repeat_kv -> dung ngay."
  python offline_clustering.py "$MODEL" $CHAT_ARG \
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
python pred.py --model "$MODEL" --task "$DATASET" --seed "$SQA_SEED" --overwrite $CHAT_ARG $LIMIT_ARG
python eval.py --model "$MODEL" $LIMIT_ARG

# ---------- 5. Sq-70% qua duong GQA ----------
echo ""
echo ">>> [5] Sq-70% (percentile=$SQA_PERCENTILE_GATE) — $DATASET"
python pred.py --model "$MODEL" --task "$DATASET" $CHAT_ARG \
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

NOTE="Gate Phase 1 ($MODEL) tren $DATASET ($SCOPE)"
if [ "$SKIP_CLUSTER" -eq 1 ]; then NOTE="$NOTE, dung lai centroid co san"; fi

# --no_log_md: KHONG ghi vao EXPERIMENT_LOG.md (file trong git). Pod chi sinh so;
# ghi nhat ky lam o may Windows roi push — xem docs/POD_RUNBOOK.md §10.
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
    --no_log_md \
    $LIMIT_ARG || GATE_RC=$?

echo ""
echo "Console log:  $CONSOLE_LOG"
echo "Nhat ky:      $SQA_REPO_ROOT/EXPERIMENT_LOG.md"

sleep 1
exit $GATE_RC
