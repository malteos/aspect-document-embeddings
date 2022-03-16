#!/usr/bin/env bash

export LR=2e-5  # 5e-05 # LEARNING_RATE = 2e-5  # 2e-6 does not work (?)
export EPOCHS=4  # or 4?
export SEED=0
export CACHE_DIR=./data/trainer_cache

export OUTPUT_DIR=./output/pwc
export DOC_ID_COL=paper_id
export DOC_A_COL=from_paper_id
export DOC_B_COL=to_paper_id
export HF_DATASET=paperswithcode_aspects
export HF_DATASET_CACHE_DIR=./data/nlp_cache
