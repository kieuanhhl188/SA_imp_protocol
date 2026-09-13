#!/bin/bash
# ====================================================================
# run_phase2_statement_repobench.sh — muc MIN NHAT trong 5 muc (file/class/function/block/
# statement) cua tieu chi C2 (Phase 5) tren RepoBench-P. Da do: class DUONG o sp70/sp80
# (entry 12/9 (d)), function AM ca 3 muc (entry 11/9), block AM ca 3 muc va NANG NHAT
# (entry 13/9 (e)). Statement la buoc cuoi cung con lai de xac nhan xu huong don dieu
# "min hon -> hai nhieu hon" co tiep tuc hay khong.
#
# ---- CANH BAO QUAN TRONG: NGAN SACH — doc truoc khi doc ket qua ----
# D6 (Phase 1, do tren 991 mau that): o level=statement, RepoBench-P co **441/492 (89,6%)**
# mau vuot ngan sach centroid 5% — nghia la voi chinh sach mac dinh `on_budget_exceeded=skip`
# (giong het class/function/block da chay), PHAN LON mau se bi LOAI, chi con ~10% (~20/200)
# mau kha thi. Ghep cap tren n~20 se co KTC bootstrap rat rong — dung doc con so diem giua
# nhu the "manh" bang cac muc truoc (n~195). Day la ly do statement KHONG duoc dat lam mac
# dinh trong thi nghiem chinh (xem D6, doc/PHASE1_RESULTS hoac EXPERIMENT_LOG muc D6).
# Van chay skip (khong merge) de giu dung nghia "statement that", nhat quan voi class/
# function/block da do — merge se lam "statement" bien thanh xap xi "block" ve kich thuoc.
#
# ---- CHI PHI ----
# Nhanh `sa` KHONG doc --level (offline_clustering_struct.py dong sa goi thang
# run_clustering, bo qua unit_ids) -> tai dung thang $P2/sa/repobench-p/ da co san,
# KHONG sinh lai. Chi can sinh moi `hard_boundary --level statement`. Nhieu unit hon block/
# function nen co the cham hon (nhieu centroid nho hon), nhung it mau kha thi hon nen tong
# thoi gian pred/eval it hon — chua do truoc, xem SMOKE.
#
#   bash scripts/run_phase2_statement_repobench.sh
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
OUT="$P2/hard_boundary_statement/$DATASET/"

echo "=================================================================="
echo "  Phase 2 (--level statement) + Phase 5 — $DATASET"
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
    --level statement --percent_clusters 5 --observation_window 100 \
    --phase1_dir "$PHASE1_DIR" --limit 3 \
    --method hard_boundary --output_path "$P2/_smoke/hard_boundary_statement/$DATASET/"
echo ">>> SMOKE xong sau $(( $(date +%s) - SMOKE_T0 ))s (3 mau)"
echo ">>> Kiem tra so unit/mau o log tren — neu qua thap (gan 1, suy bien ve sa) thi"
echo "    --level statement khong co y nghia tren tap nay, dung lai truoc khi chay full."

echo ""
echo "########## hard_boundary --level statement (full) ##########"
python offline_clustering_struct.py "$MODEL" --dataset "$DATASET" \
    --level statement --percent_clusters 5 --observation_window 100 \
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
    2>&1 | tee /workspace/p2_invariants_repobench_statement.log | tail -8 || true

echo ""
echo "########## PHASE 5 — C2 recall@budget (sa vs hard_boundary --level statement) ##########"
python phase5_recall.py "$MODEL" --dataset "$DATASET" \
    --cluster_dir "sa=$SA_DIR" \
    --cluster_dir "hard_boundary_statement=$OUT" \
    --sparsity 70 80 90 --limit "$LIMIT" \
    --skip_idx 21 102 \
    --out /workspace/phase5_repobench_statement.json

echo ""
echo "########## Bootstrap ghep cap ##########"
python scripts/phase5_bootstrap.py /workspace/phase5_repobench_statement.json \
    | tee /workspace/phase5_repobench_statement_bootstrap.txt

echo ""
echo "=================================================================="
echo "  XONG — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Ket qua: /workspace/phase5_repobench_statement.json"
echo "           /workspace/phase5_repobench_statement_bootstrap.txt"
echo "           /workspace/p2_invariants_repobench_statement.log"
echo "  Doc ket qua: neu KTC(hard_boundary_statement - sa) van loai tru 0 va am o ca 3 muc"
echo "  budget -> muc 'xem lai dinh nghia unit/level' coi nhu da lam du, H0 chuyen tu"
echo "  YEU sang SAI han. Neu KTC chua 0 hoac duong o >=1 muc -> H0 van yeu, ranh gioi"
echo "  tho hon co giup that, phai bao cao lai, KHONG duoc dong Idea 1."
echo "=================================================================="
