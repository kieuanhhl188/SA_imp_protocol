#!/bin/bash
# ====================================================================
# run_phase2_block_lcc.sh — Phase 2 voi L2 = BLOCK tren LCC (LongChat-7B).
#
#   LIMIT_P2=200 bash scripts/run_phase2_block_lcc.sh
#
# Muc dich: o --level function thi build_l1_groups che L1 toi so function ->
# struct_hierarchy == hard_boundary tung chu so. Voi L2 = block, L1 = function
# la mot phep GOP that (nhieu block -> 1 function), nen struct_hierarchy tach
# duoc khoi hard_boundary. Xem EXPERIMENT_LOG Phase 2 / muc "Smoke GPU 20/8".
#
# - KHONG chay lai nhanh `sa` (doc lap voi level) — dung lai /workspace/p2-longchat/sa/lcc
# - hard_boundary + struct_hierarchy chay o --level block
# - struct_hierarchy: --level_l1 function  (protocol: "L1 = trung binh theo function/file")
# - on_budget_exceeded=skip (mac dinh): block tren LCC bo ~2,6% mau, ghi feasibility_*.json
# ====================================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/configs/phase1.sh"
# shellcheck disable=SC1091
[ -f /workspace/env.sh ] && source /workspace/env.sh

MODEL="${SQA_MODEL_CODE:-longchat-v1.5-7b-32k}"
P2_FN="${P2_FN_DIR:-/workspace/p2-longchat}"          # nhanh sa (level-independent) lay o day
P2="${P2_DIR:-/workspace/p2-longchat-block}"          # ket qua block ghi o day
PHASE1_DIR="${SQA_PHASE1_DIR:-$REPO_ROOT/phase1_data}"
LIMIT_P5="${LIMIT_P5:-100}"
LIMIT_P2="${LIMIT_P2:--1}"

FORCE_CHAT=()
[ "${SQA_FORCE_CHAT:-0}" = "1" ] && FORCE_CHAT=(--force_chat)

COMMON=("${FORCE_CHAT[@]}" --dataset lcc --level block
        --percent_clusters 5 --observation_window 100
        --phase1_dir "$PHASE1_DIR")
[ "$LIMIT_P2" != "-1" ] && COMMON+=(--limit "$LIMIT_P2")

echo "=================================================================="
echo "  Phase 2 (L2=block) + Phase 5 — LCC"
echo "  Model     : $MODEL   (force_chat=${SQA_FORCE_CHAT:-0})"
echo "  sa reuse  : $P2_FN/sa/lcc"
echo "  Output    : $P2"
echo "  LIMIT_P2  : $LIMIT_P2   LIMIT_P5: $LIMIT_P5"
echo "  Bat dau   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

if [ ! -f "$PHASE1_DIR/$MODEL/lcc_meta.jsonl" ]; then
  echo "[ERROR] thieu du lieu Phase 1.4 cho $MODEL tai $PHASE1_DIR/$MODEL/"
  exit 1
fi
if [ ! -d "$P2_FN/sa/lcc" ]; then
  echo "[ERROR] thieu nhanh sa tai $P2_FN/sa/lcc — chay run_phase2_phase5_lcc.sh truoc"
  exit 1
fi

echo ""
echo "########## nhanh 1/2: hard_boundary (L2=block) ##########"
python offline_clustering_struct.py "$MODEL" "${COMMON[@]}" \
    --method hard_boundary --output_path "$P2/hard_boundary/lcc/"

echo ""
echo "########## nhanh 2/2: struct_hierarchy (L2=block, L1=function) ##########"
python offline_clustering_struct.py "$MODEL" "${COMMON[@]}" \
    --method struct_hierarchy --level_l1 function \
    --output_path "$P2/struct_hierarchy/lcc/"

echo ""
echo "########## Kiem toan ven file ##########"
for B in hard_boundary struct_hierarchy; do
  echo "-- $B"
  python scripts/check_cluster_integrity.py "$P2/$B/lcc" | tail -3
done

echo ""
echo "########## Kiem bat bien Phase 2 (level=block) ##########"
python scripts/check_phase2_invariants.py \
    --cluster_dir "sa=$P2_FN/sa/lcc" \
    --cluster_dir "hard_boundary=$P2/hard_boundary/lcc" \
    --cluster_dir "struct_hierarchy=$P2/struct_hierarchy/lcc" \
    --model "$MODEL" --level block \
    --phase1_dir "$PHASE1_DIR/$MODEL" --dataset lcc \
    2>&1 | tee /workspace/p2_invariants_block.log | tail -8 || true

echo ""
echo "########## PHASE 5 — C2 recall@budget (level=block) ##########"
python phase5_recall.py "$MODEL" "${FORCE_CHAT[@]}" --dataset lcc \
    --cluster_dir "sa=$P2_FN/sa/lcc" \
    --cluster_dir "hard_boundary=$P2/hard_boundary/lcc" \
    --cluster_dir "struct_hierarchy=$P2/struct_hierarchy/lcc" \
    --sparsity 70 80 90 --limit "$LIMIT_P5" --out /workspace/phase5_lcc_block.json

echo ""
echo "=================================================================="
echo "  XONG — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  /workspace/phase5_lcc_block.json"
echo "  /workspace/p2_invariants_block.log"
echo "=================================================================="
