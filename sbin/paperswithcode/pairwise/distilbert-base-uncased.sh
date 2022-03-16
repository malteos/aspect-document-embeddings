#!/usr/bin/env bash

export PYTHONUNBUFFERED=1

export APP_ROOT=$(dirname "$0")

. $APP_ROOT/config.sh

export MODEL_NAME=distilbert-base-uncased

# bert-base
export EVAL_BATCH_SIZE=16
export TRAIN_BATCH_SIZE=8

# for CV_FOLD in 1 2 3 4
for CV_FOLD in "sample"
do
    python trainer_cli.py --cv_fold $CV_FOLD \
        --output_dir $OUTPUT_DIR \
        --model_name_or_path $MODEL_NAME \
        --doc_id_col $DOC_ID_COL \
        --doc_a_col $DOC_A_COL \
        --doc_b_col $DOC_B_COL \
        --hf_dataset $HF_DATASET \
        --hf_dataset_cache_dir $HF_DATASET_CACHE_DIR \
        --cache_dir $CACHE_DIR \
        --num_train_epochs $EPOCHS \
        --seed $SEED \
        --per_device_eval_batch_size $EVAL_BATCH_SIZE \
        --per_device_train_batch_size $TRAIN_BATCH_SIZE \
        --learning_rate $LR \
        --logging_steps 100 \
        --save_steps 0 \
        --save_total_limit 3 \
        --do_train \
        --binary_classification \
        --save_predictions
done

export PYTHONUNBUFFERED=""
