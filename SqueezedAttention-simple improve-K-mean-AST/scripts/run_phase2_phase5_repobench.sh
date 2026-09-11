#!/bin/bash
# ====================================================================
# run_phase2_phase5_repobench.sh — mot phien pod: Phase 2 (3 nhanh) roi Phase 5 (C2)
# tren RepoBench-P (LongBench). Doi song song voi run_phase2_phase5_lcc.sh.
#
#   LIMIT_P2=200 bash scripts/run_phase2_phase5_repobench.sh
#
# ---- VI SAO RepoBench-P ----
# LCC shared_prefix trung vi ~3,1K token — qua ngan cho structure-aware clustering, va
# tang L1 bi vo hieu o 73% mau (build_l1_groups che toi so function). RepoBench-P trung
# vi ~12,5K token (~4x), tang L1 co hieu luc o ~70% mau -> day la noi de xuat 1 + de
# xuat 2 co co hoi tach khoi `sa`. Phase 5 remaining item (b) trong EXPERIMENT_LOG.
#
# ---- CHOT: model = longchat-v1.5-7b-32k, KHONG --force_chat (repobench-p thuoc
#      NO_CHAT_TEMPLATE, giong lcc) ----
#
# ---- 2 MAU LECH TOKENIZER (fast != slow): dataidx 21, 102 ----
# check_phase1_data.py bao FAIL o kiem [3]. offline_clustering (tok cham) va offset npz
# (tok nhanh) khong khop cho 2 mau nay -> nhan unit AST lech. Chinh sach D6: so tren tap
# giao mau kha thi. Ta VAN cluster het (2 mau, chi phi khong dang ke) nhung LOAI 21 & 102
# o buoc phase5 qua co --skip_idx 21 102 (per_sample khong chua 2 mau nay). Ghi ro con so nay.
#
# ---- DUNG LUONG ----
# RepoBench-P ~4x token so LCC -> centroid file ~4x lon. LCC 200 mau 3 nhanh = 85 GB;
# uoc RepoBench-P 200 ~ 300-350 GB. Volume /workspace hien du (kiem `df -h /workspace`).
#
# ---- THOI GIAN ----
# CHUA do tr— clustering scale ~bac hai theo shared_prefix (K = 5%*S). SMOKE 3 mau
# chay TRUOC full branch 1 de lay s/mau that. Xem bien SMOKE_ONLY.
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
LIMIT_P5="${LIMIT_P5:-200}"
LIMIT_P2="${LIMIT_P2:--1}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"       # 1 = chi chay smoke 3 mau hard_boundary roi dung

FORCE_CHAT=()
[ "${SQA_FORCE_CHAT:-0}" = "1" ] && FORCE_CHAT=(--force_chat)

COMMON=("${FORCE_CHAT[@]}" --dataset "$DATASET" --level function --level_l1 class
        --percent_clusters 5 --observation_window 100
        --phase1_dir "$PHASE1_DIR")
[ "$LIMIT_P2" != "-1" ] && COMMON+=(--limit "$LIMIT_P2")

echo "=================================================================="
echo "  Phase 2 + Phase 5 — $DATASET"
echo "  Model     : $MODEL   (force_chat=${SQA_FORCE_CHAT:-0})"
echo "  Phase 1.4 : $PHASE1_DIR/$MODEL"
echo "  Output    : $P2"
echo "  LIMIT_P2  : $LIMIT_P2   LIMIT_P5: $LIMIT_P5   SMOKE_ONLY: $SMOKE_ONLY"
echo "  Bat dau   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

if [ ! -f "$PHASE1_DIR/$MODEL/${DATASET}_meta.jsonl" ]; then
  echo "[ERROR] thieu du lieu Phase 1.4 cho $MODEL tai $PHASE1_DIR/$MODEL/"
  echo "        Sinh truoc (CPU): SQA_PHASE1_TASK=$DATASET bash scripts/phase1_gate.sh --data-only"
  exit 1
fi

echo ""
echo "########## SMOKE Phase 2 + Phase 5 (3 mau) — bat loi som, do s/mau ##########"
SMOKE_T0=$(date +%s)
python offline_clustering_struct.py "$MODEL" "${FORCE_CHAT[@]}" --dataset "$DATASET" \
    --level function --level_l1 class --percent_clusters 5 --observation_window 100 \
    --phase1_dir "$PHASE1_DIR" --limit 3 \
    --method hard_boundary --output_path "$P2/_smoke/hard_boundary/$DATASET/"
python phase5_recall.py "$MODEL" "${FORCE_CHAT[@]}" --dataset "$DATASET" \
    --cluster_dir "hard_boundary=$P2/_smoke/hard_boundary/$DATASET" \
    --sparsity 70 --limit 3 --out /workspace/phase5_repobench_smoke.json
echo ">>> SMOKE xong sau $(( $(date +%s) - SMOKE_T0 ))s (3 mau, 1 nhanh)"

if [ "$SMOKE_ONLY" = "1" ]; then
  echo ">>> SMOKE_ONLY=1 — dung tai day. Xem s/mau o tren roi chon LIMIT_P2."
  exit 0
fi

echo ""
echo "########## nhanh 1/3: hard_boundary ##########"
python offline_clustering_struct.py "$MODEL" "${COMMON[@]}" \
    --method hard_boundary --output_path "$P2/hard_boundary/$DATASET/"

echo ""
echo "########## nhanh 2/3: struct_hierarchy ##########"
python offline_clustering_struct.py "$MODEL" "${COMMON[@]}" \
    --method struct_hierarchy --output_path "$P2/struct_hierarchy/$DATASET/"

echo ""
echo "########## nhanh 3/3: sa (doi chung) ##########"
python offline_clustering_struct.py "$MODEL" "${COMMON[@]}" \
    --method sa --output_path "$P2/sa/$DATASET/"

echo ""
echo "########## Kiem toan ven file ##########"
for B in sa hard_boundary struct_hierarchy; do
  echo "-- $B"
  python scripts/check_cluster_integrity.py "$P2/$B/$DATASET" | tail -3
done

echo ""
echo "########## Kiem bat bien Phase 2 ##########"
python scripts/check_phase2_invariants.py \
    --cluster_dir "sa=$P2/sa/$DATASET" \
    --cluster_dir "hard_boundary=$P2/hard_boundary/$DATASET" \
    --cluster_dir "struct_hierarchy=$P2/struct_hierarchy/$DATASET" \
    --model "$MODEL" \
    --phase1_dir "$PHASE1_DIR/$MODEL" --dataset "$DATASET" \
    2>&1 | tee /workspace/p2_invariants_repobench.log | tail -8 || true

echo ""
echo "########## PHASE 5 — C2 recall@budget (phang + phan tang) ##########"
python phase5_recall.py "$MODEL" "${FORCE_CHAT[@]}" --dataset "$DATASET" \
    --cluster_dir "sa=$P2/sa/$DATASET" \
    --cluster_dir "hard_boundary=$P2/hard_boundary/$DATASET" \
    --cluster_dir "struct_hierarchy=$P2/struct_hierarchy/$DATASET" \
    --sparsity 70 80 90 --limit "$LIMIT_P5" --hierarchical --l1_ratios 1.0 0.9 0.7 0.5 \
    --skip_idx 21 102 \
    --out /workspace/phase5_repobench.json

echo ""
echo "=================================================================="
echo "  XONG — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Ket qua: /workspace/phase5_repobench.json"
echo "           /workspace/p2_invariants_repobench.log"
echo "  Phan tich: python scripts/phase5_bootstrap.py /workspace/phase5_repobench.json"
echo "  (dataidx 21 & 102 da bi loai o buoc phase5 qua --skip_idx)"
echo "=================================================================="
