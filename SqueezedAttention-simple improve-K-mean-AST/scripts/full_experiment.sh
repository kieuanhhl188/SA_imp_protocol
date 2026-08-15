#!/bin/bash
# ====================================================================
# full_experiment.sh
# Full experiment matrix cho Hướng 1 + 2.
#
# Chạy trên multiple percent_clusters để vẽ Pareto curve.
# Estimated time: vài giờ tới 1 ngày tùy số dataset.
# ====================================================================

set -e

MODEL="llama2-7b-32k"
DEVICE=0
OUT_DIR="output_full"

# Datasets cho Hướng 1 (text-heavy)
TEXT_DATASETS=("trec" "triviaqa" "samsum" "qasper" "multifieldqa_en")

# Datasets cho Hướng 2 (code)
CODE_DATASETS=("lcc" "repobench-p")

# Budget percentages - sweep để vẽ Pareto curve
PERCENTAGES=(2 5 10 20)

# ====== Hướng 1: Adaptive Budget trên text datasets ======
echo ""
echo "############################################"
echo "##  HƯỚNG 1: Adaptive Budget Experiments  ##"
echo "############################################"

for dataset in "${TEXT_DATASETS[@]}"; do
    for pct in "${PERCENTAGES[@]}"; do
        echo ""
        echo ">>> Dataset=$dataset | Percent=$pct"

        # Baseline
        python offline_clustering_v2.py $MODEL \
            --dataset $dataset \
            --output_path $OUT_DIR \
            --percent_clusters $pct \
            --device $DEVICE \
            --save_entropy_log || echo "FAIL baseline $dataset $pct"

        # Adaptive linear
        python offline_clustering_v2.py $MODEL \
            --dataset $dataset \
            --output_path $OUT_DIR \
            --percent_clusters $pct \
            --adaptive_budget --budget_strategy linear \
            --device $DEVICE || echo "FAIL adaptive-linear $dataset $pct"

        # Adaptive pyramid (alternative strategy)
        python offline_clustering_v2.py $MODEL \
            --dataset $dataset \
            --output_path $OUT_DIR \
            --percent_clusters $pct \
            --adaptive_budget --budget_strategy pyramid \
            --device $DEVICE || echo "FAIL adaptive-pyramid $dataset $pct"
    done
done

# ====== Hướng 2: Code-aware trên code datasets ======
echo ""
echo "############################################"
echo "##  HƯỚNG 2: Code-aware Experiments       ##"
echo "############################################"

for dataset in "${CODE_DATASETS[@]}"; do
    for pct in "${PERCENTAGES[@]}"; do
        echo ""
        echo ">>> Dataset=$dataset | Percent=$pct"

        # Baseline
        python offline_clustering_v2.py $MODEL \
            --dataset $dataset \
            --output_path $OUT_DIR \
            --percent_clusters $pct \
            --device $DEVICE || echo "FAIL baseline $dataset $pct"

        # Code-aware only
        python offline_clustering_v2.py $MODEL \
            --dataset $dataset \
            --output_path $OUT_DIR \
            --percent_clusters $pct \
            --code_aware --code_language python \
            --device $DEVICE || echo "FAIL code-aware $dataset $pct"

        # Combo: Hướng 1 + 2
        python offline_clustering_v2.py $MODEL \
            --dataset $dataset \
            --output_path $OUT_DIR \
            --percent_clusters $pct \
            --adaptive_budget --budget_strategy linear \
            --code_aware --code_language python \
            --device $DEVICE || echo "FAIL combo $dataset $pct"
    done
done

echo ""
echo "Full clustering experiments done."
echo "Next: run online evaluation - see scripts/run_eval_matrix.sh"
