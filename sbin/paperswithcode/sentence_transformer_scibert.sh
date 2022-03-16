#!/usr/bin/env bash

export PYTHONUNBUFFERED=1

export APP_ROOT=$(dirname "$0")

#. $APP_ROOT/config.sh

MODEL_NAME=scibert-scivocab-uncased
HF_DATASET=paperswithcode_aspects
LOSS=multiple_negatives_ranking
TOP_KS=5,10,25,50
TRAIN_BATCH_SIZE=25
EVAL_BATCH_SIZE=25

#FOLD=${FOLD:=sample}

# Run each fold on different GPU:
# CUDA_VISIBLE_DEVICES=2 & FOLD=1 & sbin/paperswithcode/sentence_transformer_scibert.sh
# CUDA_VISIBLE_DEVICES=2 & FOLD=2 & sbin/paperswithcode/sentence_transformer_scibert.sh
# ...

for ASPECT in "task" "method" "dataset"
do
    ASPECT_DIR=./output/pwc/$ASPECT
    mkdir -p $ASPECT_DIR

    for FOLD in 1 2 3 4  #"sample"
    do
    OUTPUT_DIR=$ASPECT_DIR/$FOLD/st_${MODEL_NAME}
    echo $FOLD
    # train
    ./sentence_transformer_cli.py train $MODEL_NAME $HF_DATASET $ASPECT $FOLD $OUTPUT_DIR --train_epochs=3 --train_batch_size=$TRAIN_BATCH_SIZE --eval_batch_size=$EVAL_BATCH_SIZE --loss=$LOSS

    # build vectors
    ./sentence_transformer_cli.py build_vectors $OUTPUT_DIR $HF_DATASET $ASPECT $FOLD

    # evaluate
    ./eval_cli.py evaluate_vectors $HF_DATASET $ASPECT $OUTPUT_DIR/pwc_id2vec.w2v.txt --name=st_${MODEL_NAME} --folds=$FOLD --top_ks=$TOP_KS --output_path=$ASPECT_DIR/eval.csv
    done
done




export PYTHONUNBUFFERED=""
