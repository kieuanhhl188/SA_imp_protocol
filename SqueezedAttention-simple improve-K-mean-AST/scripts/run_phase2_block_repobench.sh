#!/bin/bash
# ====================================================================
# run_phase2_block_repobench.sh — dinh vi ranh gioi "giup/hai" cua tieu chi C2 (Phase 5)
# tren RepoBench-P, giua muc class (da DUONG o sp70/sp80, entry 2026-09-12 (d)) va muc
# function (da AM o ca 3 muc sparsity, entry 11/9).
#
# ---- VI SAO CAN SCRIPT NAY ----
# `--level class` (THO hon function) tren RepoBench-P cho KTC(hard_boundary-sa) DUONG va
# loai tru 0 o sp70 (+0.30) va sp80 (+0.16), nhung quay lai AM o sp90 (-0.09) — xem entry
# 2026-09-12 (d). `--level function` (MIN hon class) tren cung dataset da AM o ca 3 muc.
# Ranh gioi "giup/hai" vi vay nam dau do GIUA class va function. `--level block` MIN hon
# function mot bac nua — chay no de xac nhan xu huong "min hon -> te hon" tiep tuc (cung
# chieu voi ket qua LCC truoc day, block te ~3x function) hay dao chieu bat ngo.
#
# ---- VI SAO REPOBENCH-P, KHONG PHAI LCC ----
# LCC context ngan (~3.1K token trung vi), cau truc mong -> block/statement de suy bien.
# RepoBench-P dai hon (~4x), giu du don vi o muc block de phep so co nghia.
#
# ---- CHI PHI ----
# Nhanh `sa` KHONG doc --level (offline_clustering_struct.py dong sa goi thang
# run_clustering, bo qua unit_ids) -> tai dung thang $P2/sa/repobench-p/ da co san,
# KHONG sinh lai. Chi can sinh moi `hard_boundary --level block`, uoc ~30-45 phut GPU
# (cung co so voi luot --level class truoc, cung dataset). Phase 5 khong can --hierarchical
# vi dang kiem ranh gioi L2, khong phai tang L1.
#
#   bash scripts/run_phase2_block_repobench.sh
#
# Chay lai duoc: offline_clustering_struct.py bo qua mau da co du file.
# ====================================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/configs/phase1.sh"
# shellcheck disable=SC1091
[ -f /workspace/env.sh ] && source /workspace/env.sh

MODEL="${SQA_MODEL_CODE:-longchat-v1.5-7b-32k}"
DATASET=repobench-p
P2="${P2_DIR:-/workspace/p2-longchat-repobench}"
PHASE1_DIR="${SQA_PHASE1_DIR:-$REPO_ROOT/phase1_data}"
LIMIT="${LIMIT:-200}"

SA_DIR="$P2/sa/$DATASET"
OUT="$P2/hard_boundary_block/$DATASET/"

echo "=================================================================="
echo "  Phase 2 (--level block) + Phase 5 — $DATASET"
echo "  Model      : $MODEL"
echo "  Phase 1.4  : $PHASE1_DIR/$MODEL"
echo "  sa (tai dung): $SA_DIR"
echo "  Output moi : $OUT"
echo "  LIMIT      : $LIMIT"
echo "  Bat dau    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

if [ ! -f "$PHASE1_DIR/$MODEL/${DATASET}_meta.jsonl" ]; then
  echo "[ERROR] thieu du lieu Phase 1.4 cho $MODEL tai $PHASE1_DIR/$MODEL/"
  echo "        Sinh truoc (CPU): SQA_PHASE1_TASK=$DATASET bash scripts/phase1_gate.sh --data-only"
  exit 1
fi

if [ ! -d "$SA_DIR" ]; then
  echo "[ERROR] khong thay $SA_DIR — can chay run_phase2_phase5_repobench.sh truoc"
  echo "        (hoac sua SA_DIR/P2_DIR neu du lieu nam o duong dan khac)"
  exit 1
fi

echo ""
echo "########## SMOKE 3 mau — bat loi som, do s/mau ##########"
SMOKE_T0=$(date +%s)
python offline_clustering_struct.py "$MODEL" --dataset "$DATASET" \
    --level block --percent_clusters 5 --observation_window 100 \
    --phase1_dir "$PHASE1_DIR" --limit 3 \
    --method hard_boundary --output_path "$P2/_smoke/hard_boundary_block/$DATASET/"
echo ">>> SMOKE xong sau $(( $(date +%s) - SMOKE_T0 ))s (3 mau)"
echo ">>> Kiem tra so unit/mau o log tren — neu qua thap (gan 1, suy bien ve sa) thi"
echo "    --level block khong co y nghia tren tap nay, dung lai truoc khi chay full."

echo ""
echo "########## hard_boundary --level block (full) ##########"
python offline_clustering_struct.py "$MODEL" --dataset "$DATASET" \
    --level block --percent_clusters 5 --observation_window 100 \
    --phase1_dir "$PHASE1_DIR" --limit "$LIMIT" \
    --method hard_boundary --output_path "$OUT"

echo ""
echo "########## Kiem toan ven file ##########"
python scripts/check_cluster_integrity.py "$OUT" --expect "$LIMIT"

echo ""
echo "########## Kiem bat bien Phase 2 (A/B/C — khong co tang L1 o day) ##########"
python scripts/check_phase2_invariants.py \
    --cluster_dir "$OUT" --method hard_boundary \
    --model "$MODEL" \
    --phase1_dir "$PHASE1_DIR/$MODEL" --dataset "$DATASET" \
    --checks ABC \
    2>&1 | tee /workspace/p2_invariants_repobench_block.log | tail -8 || true

echo ""
echo "########## PHASE 5 — C2 recall@budget (sa vs hard_boundary --level block) ##########"
python phase5_recall.py "$MODEL" --dataset "$DATASET" \
    --cluster_dir "sa=$SA_DIR" \
    --cluster_dir "hard_boundary_block=$OUT" \
    --sparsity 70 80 90 --limit "$LIMIT" \
    --skip_idx 21 102 \
    --out /workspace/phase5_repobench_block.json

echo ""
echo "########## Bootstrap ghep cap ##########"
python scripts/phase5_bootstrap.py /workspace/phase5_repobench_block.json \
    | tee /workspace/phase5_repobench_block_bootstrap.txt

echo ""
echo "=================================================================="
echo "  XONG — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Ket qua: /workspace/phase5_repobench_block.json"
echo "           /workspace/phase5_repobench_block_bootstrap.txt"
echo "           /workspace/p2_invariants_repobench_block.log"
echo "  Doc ket qua: neu KTC(hard_boundary_block - sa) van loai tru 0 va am o ca 3 muc"
echo "  budget -> muc 'xem lai dinh nghia unit/level' coi nhu da lam du, H0 chuyen tu"
echo "  YEU sang SAI han. Neu KTC chua 0 hoac duong o >=1 muc -> H0 van yeu, ranh gioi"
echo "  tho hon co giup that, phai bao cao lai, KHONG duoc dong Idea 1."
echo "=================================================================="
