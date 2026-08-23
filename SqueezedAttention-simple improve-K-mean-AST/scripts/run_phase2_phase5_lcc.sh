#!/bin/bash
# ====================================================================
# run_phase2_phase5_lcc.sh — mot phien pod: Phase 2 (3 nhanh) roi Phase 5 (C2) tren LCC.
#
#   bash scripts/run_phase2_phase5_lcc.sh
#
# Cau hinh da chot 23/8, KHONG doi giua chung:
#   model = qwen2.5-coder-7b-instruct · force_chat · fixed_context=full · maxlen 31500
#
# THU TU CO CHU DICH: chay nhanh `hard_boundary` truoc, roi SMOKE Phase 5 tren 3 mau
# ngay lap tuc. Ly do: phase5_recall.py chua tung chay tren GPU. Neu no hong thi hong sau
# ~1 gio chu khong phai sau ~3 gio, va hai nhanh con lai chua ton gio nao.
#
# Uoc tinh (ngoai suy tu moc do that: 4,97 giay/mau tren LCC Instruct):
#   moi nhanh ~55-70 phut · ba nhanh ~3 gio · Phase 5 ~1 gio
#
# Chay lai duoc: offline_clustering_struct.py bo qua mau da co du file, nen pod restart
# giua chung thi goi lai dung lenh nay.
# ====================================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
[ -f /workspace/env.sh ] && source /workspace/env.sh

MODEL="${SQA_MODEL_CODE:-qwen2.5-coder-7b-instruct}"
P2="${P2_DIR:-/workspace/p2-instruct}"
LIMIT_P5="${LIMIT_P5:-100}"

COMMON=(--force_chat --dataset lcc --level function --level_l1 class
        --percent_clusters 5 --observation_window 100)

echo "=================================================================="
echo "  Phase 2 + Phase 5 — LCC"
echo "  Model : $MODEL"
echo "  Output: $P2"
echo "  Bat dau: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

echo ""
echo "########## nhanh 1/3: hard_boundary ##########"
python offline_clustering_struct.py "$MODEL" "${COMMON[@]}" \
    --method hard_boundary --output_path "$P2/hard_boundary/lcc/"

echo ""
echo "########## SMOKE Phase 5 (3 mau) — bat loi som ##########"
python phase5_recall.py "$MODEL" --force_chat --dataset lcc \
    --cluster_dir "hard_boundary=$P2/hard_boundary/lcc" \
    --sparsity 70 --limit 3 --out /workspace/phase5_smoke.json

echo ""
echo "########## nhanh 2/3: struct_hierarchy ##########"
python offline_clustering_struct.py "$MODEL" "${COMMON[@]}" \
    --method struct_hierarchy --output_path "$P2/struct_hierarchy/lcc/"

echo ""
echo "########## nhanh 3/3: sa (doi chung) ##########"
python offline_clustering_struct.py "$MODEL" "${COMMON[@]}" \
    --method sa --output_path "$P2/sa/lcc/"

echo ""
echo "########## Kiem toan ven file ##########"
for B in sa hard_boundary struct_hierarchy; do
  echo "-- $B"
  python scripts/check_cluster_integrity.py "$P2/$B/lcc" | tail -3
done

echo ""
echo "########## Kiem bat bien Phase 2 ##########"
python scripts/check_phase2_invariants.py \
    --cluster_dir "sa=$P2/sa/lcc" \
    --cluster_dir "hard_boundary=$P2/hard_boundary/lcc" \
    --cluster_dir "struct_hierarchy=$P2/struct_hierarchy/lcc" \
    --phase1_dir "${SQA_PHASE1_DIR:-/workspace/phase1_data}" --dataset lcc \
    2>&1 | tee /workspace/p2_invariants_instruct.log | tail -5 || true

echo ""
echo "########## PHASE 5 — C2 recall@budget ##########"
python phase5_recall.py "$MODEL" --force_chat --dataset lcc \
    --cluster_dir "sa=$P2/sa/lcc" \
    --cluster_dir "hard_boundary=$P2/hard_boundary/lcc" \
    --cluster_dir "struct_hierarchy=$P2/struct_hierarchy/lcc" \
    --sparsity 70 80 90 --limit "$LIMIT_P5" --out /workspace/phase5_lcc.json

echo ""
echo "=================================================================="
echo "  XONG — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Ket qua: /workspace/phase5_lcc.json"
echo "           /workspace/p2_invariants_instruct.log"
echo "=================================================================="
