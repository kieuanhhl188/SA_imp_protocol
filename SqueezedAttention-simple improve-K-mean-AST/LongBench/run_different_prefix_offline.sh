#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# arguments:
PATH_TO_CLUSTERS="/home/chooper/fixed-prompt-clusters/"
PERCENTILE="0.90" # percentile for pruning
DATASET="2wikimqa"
PERC_CLUSTERS="5"

# run evaluation
python pred.py --model LLaMA-2-7B-32K --use_centroids --percentile $PERCENTILE --percent_clusters $PERC_CLUSTERS \
               --path_to_clusters $PATH_TO_CLUSTERS --task $DATASET

# check accuracy
python eval.py --model LLaMA-2-7B-32K --use_centroids --percentile $PERCENTILE --percent_clusters $PERC_CLUSTERS
