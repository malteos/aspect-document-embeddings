#!/usr/bin/env bash

export PYTHONUNBUFFERED=1

export APP_ROOT=$(dirname "$0")

#. $APP_ROOT/config.sh

HF_DATASET=paperswithcode_aspects
TOP_KS=5,10,25,50

# word vector model
./data_cli.py train_fasttext $HF_DATASET ./output/pwc

# document vectors
./data_cli.py build_avg_word_vectors $HF_DATASET ./output/pwc/fasttext.w2v.txt ./output/pwc/avg_fasttext.w2v.txt

# evaluate for all folds and aspects
for ASPECT in "task" "method" "dataset"
do
    ./eval_cli.py evaluate_vectors $HF_DATASET $ASPECT ./output/pwc/avg_fasttext.w2v.txt --name=avg_fasttext --folds=1,2,3,4 --top_ks=$TOP_KS --output_path=./output/pwc/$ASPECT/eval.csv
done

export PYTHONUNBUFFERED=""
