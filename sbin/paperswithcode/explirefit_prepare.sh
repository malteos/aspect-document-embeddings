#!/usr/bin/env bash

export PYTHONUNBUFFERED=1

export APP_ROOT=$(dirname "$0")

#. $APP_ROOT/config.sh
export EXPLIREFIT_DIR=~/experiments/explirefit

# no negative samples!
export EMPTY_FILE=./output/empty.txt

export BASE_VECTORS=./output/pwc/avg_fasttext.w2v.txt

MODEL_NAME=avg_fasttext
HF_DATASET=paperswithcode_aspects
TOP_KS="5,10,25,50"
CPU_LIMIT="0-20"

for ASPECT in "task" "method" "dataset"
do
    ASPECT_DIR=./output/pwc/$ASPECT
    mkdir -p $ASPECT_DIR

    # generate constraints from training data (for all folds)
    ./data_cli.py build_explirefit_inputs $HF_DATASET $ASPECT $ASPECT_DIR/retrofit_constraints
done


export PYTHONUNBUFFERED=""
