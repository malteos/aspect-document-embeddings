#!/usr/bin/env bash

HF_DATASET=paperswithcode_aspects
SPECTER_DIR=../specter
SPECTER_VOCAB=~/datasets/BERT_pre_trained_models/pytorch/scibert-scivocab-uncased/
TOP_KS="5,10,25,50"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:=-1}
MODEL_NAME=specter_fine_tuned

for ASPECT in "task" "method" "dataset"
do
    ASPECT_DIR=./output/pwc/$ASPECT
    for FOLD in 1 2 3 4
    do
        FOLD_DIR=$ASPECT_DIR/$FOLD
        mkdir -p $FOLD_DIR

        # build input files
        ./data_cli.py build_specter_input $HF_DATASET $ASPECT $FOLD $FOLD_DIR/$MODEL_NAME/inputs

        # prepare
        python $SPECTER_DIR/specter/data_utils/create_training_files.py --data-dir $FOLD_DIR/$MODEL_NAME/inputs --metadata $FOLD_DIR/$MODEL_NAME/inputs/metadata.json --outdir $FOLD_DIR/$MODEL_NAME/inputs --bert_vocab $SPECTER_VOCAB

        # train
        $SPECTER_DIR/scripts/run-exp-simple.sh -c $SPECTER_DIR/experiment_configs/simple.jsonnet -s $FOLD_DIR/$MODEL_NAME/model \
            --num-epochs 2 --batch-size 4 \
            --train-path $FOLD_DIR/$MODEL_NAME/inputs/data-train.p --dev-path $FOLD_DIR/$MODEL_NAME/inputs/data-val.p \
            --num-train-instances 55 --cuda-device $CUDA_VISIBLE_DEVICES

        # dummy samples
        echo "{}" > $FOLD_DIR/$MODEL_NAME/model/metadata_sample.json

        # build vectors
        ./data_cli.py build_specter_vectors $HF_DATASET $FOLD_DIR/$MODEL_NAME/model $FOLD_DIR/$MODEL_NAME/pwc_id2vec.w2v.txt --cuda_device=$CUDA_VISIBLE_DEVICES

        # eval
        ./eval_cli.py evaluate_vectors $HF_DATASET $ASPECT $FOLD_DIR/$MODEL_NAME/pwc_id2vec.w2v.txt --name=$MODEL_NAME --folds=$FOLD --top_ks=$TOP_KS --output_path=$ASPECT_DIR/eval.csv

    done
done
