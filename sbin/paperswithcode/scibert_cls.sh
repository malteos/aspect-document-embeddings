#!/usr/bin/env bash

export PYTHONUNBUFFERED=1

export APP_ROOT=$(dirname "$0")

#. $APP_ROOT/config.sh

HF_DATASET=paperswithcode_aspects
TOP_KS=5,10,25,50

# inference
./data_cli.py build_transformers_vectors paperswithcode_aspects scibert-scivocab-uncased ./output/pwc/scibert_cls.w2v.txt --pooling=cls --batch_size=16

# evaluate for all folds and aspects
for ASPECT in "task" "method" "dataset"
do
    ./eval_cli.py evaluate_vectors $HF_DATASET $ASPECT ./output/pwc/scibert_cls.w2v.txt --name=scibert_cls --folds=1,2,3,4 --top_ks=$TOP_KS --output_path=./output/pwc/$ASPECT/eval.csv
done

export PYTHONUNBUFFERED=""
