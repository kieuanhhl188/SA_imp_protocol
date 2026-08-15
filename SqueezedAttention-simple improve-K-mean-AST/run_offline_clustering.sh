#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

MODEL="LLaMA-2-7B-32K"
DATASETS=(2wikimqa)
PERC_CLUSTERS="5" # Percent Clusters

# loop over datasets
for DATASET in "${DATASETS[@]}"; do
  PATH_TO_SAVE_CLUSTERS="/home/chooper/fixed-prompt-clusters/${DATASET}/"
  python offline_clustering.py $MODEL \
                                 --dataset $DATASET \
                                 --output_path $PATH_TO_SAVE_CLUSTERS \
                                 --percent_clusters $PERC_CLUSTERS \
                                 --observation_window 100 \
                                 --device 0
done
