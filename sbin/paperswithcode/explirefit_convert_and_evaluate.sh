#!/usr/bin/env bash

export PYTHONUNBUFFERED=1

export APP_ROOT=$(dirname "$0")

#. $APP_ROOT/config.sh

#MODEL_NAME=avg_fasttext
HF_DATASET=paperswithcode_aspects
TOP_KS="5,10,25,50"
CPU_LIMIT="0-20"


for MODEL_NAME in "avg_fasttext" "specter"
do
    for ASPECT in "task" "method" "dataset"
    do
        ASPECT_DIR=./output/pwc/$ASPECT
        mkdir -p $ASPECT_DIR

        for FOLD in 1 2 3 4
        do
            # convert vectors
            python -m gensim.scripts.glove2word2vec -i $ASPECT_DIR/$FOLD/explirefit_$MODEL_NAME/pwc_id2vec.txt -o $ASPECT_DIR/$FOLD/explirefit_$MODEL_NAME/pwc_id2vec.w2v.txt

            # evaluate
            ./eval_cli.py evaluate_vectors $HF_DATASET $ASPECT $ASPECT_DIR/$FOLD/explirefit_$MODEL_NAME/pwc_id2vec.w2v.txt --name=explirefit_$MODEL_NAME --folds=$FOLD --top_ks=$TOP_KS --output_path=$ASPECT_DIR/eval.csv
        done
    done
done

export PYTHONUNBUFFERED=""
