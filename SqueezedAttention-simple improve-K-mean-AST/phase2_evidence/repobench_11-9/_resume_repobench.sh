#!/bin/bash
# Resume RepoBench-P Phase 2 (branch 1 hard_boundary already complete) + Phase 5.
# offline_clustering_struct.py tu bo qua mau da co du file -> chay lai an toan.
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/phase1.sh"
[ -f /workspace/env.sh ] && source /workspace/env.sh

MODEL="${SQA_MODEL_CODE:-longchat-v1.5-7b-32k}"
DATASET=repobench-p
P2=/workspace/p2-longchat-repobench
PHASE1_DIR="${SQA_PHASE1_DIR:-$REPO_ROOT/phase1_data}"

COMMON=(--dataset "$DATASET" --level function --level_l1 class
        --percent_clusters 5 --observation_window 100
        --phase1_dir "$PHASE1_DIR" --limit 200)

echo ">>> RESUME $(date '+%F %T')"

for spec in "struct_hierarchy" "hard_boundary" "sa"; do
  echo ""
  echo "########## nhanh: $spec ##########"
  python offline_clustering_struct.py "$MODEL" "${COMMON[@]}" \
      --method "$spec" --output_path "$P2/$spec/$DATASET/"
  sync
done

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
echo "########## PHASE 5 — C2 recall@budget ##########"
python phase5_recall.py "$MODEL" --dataset "$DATASET" \
    --cluster_dir "sa=$P2/sa/$DATASET" \
    --cluster_dir "hard_boundary=$P2/hard_boundary/$DATASET" \
    --cluster_dir "struct_hierarchy=$P2/struct_hierarchy/$DATASET" \
    --sparsity 70 80 90 --limit 200 --hierarchical --l1_ratios 1.0 0.9 0.7 0.5 \
    --skip_idx 21 102 --out /workspace/phase5_repobench.json

echo ""
echo "########## Bootstrap ghep cap ##########"
python scripts/phase5_bootstrap.py /workspace/phase5_repobench.json | tee /workspace/phase5_repobench_bootstrap.txt

echo ""
echo ">>> XONG — $(date '+%F %T')"
