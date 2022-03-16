#!/usr/bin/env bash

export PYTHONUNBUFFERED=1

export APP_ROOT=$(dirname "$0")

#echo $APP_ROOT
#. $APP_ROOT/config.sh

EXP_DIR=~/experiments/pairwise-vs-segment

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:=-1}

HF_DATASET=paperswithcode_aspects
TOP_KS=5,10,25,50

# specter embeddings
#./data_cli.py build_specter_vectors $HF_DATASET $EXP_DIR/specter_archive ./output/pwc/specter.w2v.txt --cuda_device=$CUDA_VISIBLE_DEVICES

# evaluate for all folds and aspects
for ASPECT in "task" "method" "dataset"
do
    ./eval_cli.py evaluate_vectors $HF_DATASET $ASPECT ./output/pwc/specter.w2v.txt --name=specter --folds=1,2,3,4 --top_ks=$TOP_KS --output_path=./output/pwc/$ASPECT/eval.csv
done

export PYTHONUNBUFFERED=""
