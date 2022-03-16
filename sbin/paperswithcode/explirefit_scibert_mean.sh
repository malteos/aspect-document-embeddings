#!/usr/bin/env bash

export PYTHONUNBUFFERED=1

export APP_ROOT=$(dirname "$0")

#. $APP_ROOT/config.sh

# enable conda
source ~/miniconda2/etc/profile.d/conda.sh
#conda activate explirefit

export EXPLIREFIT_DIR=~/experiments/explirefit

# no negative samples!
export EMPTY_FILE=./output/empty.txt


MODEL_NAME=scibert_mean

export BASE_VECTORS=./output/pwc/$MODEL_NAME.w2v.txt

HF_DATASET=paperswithcode_aspects
TOP_KS="5,10,25,50"
CPU_LIMIT="0-20"

for ASPECT in "task" "method" "dataset"
do
    ASPECT_DIR=./output/pwc/$ASPECT
    mkdir -p $ASPECT_DIR

    for FOLD in 1 2 3 4
    do
        mkdir -p $ASPECT_DIR/$FOLD/explirefit_$MODEL_NAME
        conda activate explirefit
        
        # train retrofit mode
        taskset -c $CPU_LIMIT python $EXPLIREFIT_DIR/trainer.py $BASE_VECTORS $ASPECT_DIR/retrofit_constraints/$FOLD/synonyms.txt $EMPTY_FILE $ASPECT_DIR/$FOLD/explirefit_$MODEL_NAME/model

        # retrofit vectors
        taskset -c $CPU_LIMIT python $EXPLIREFIT_DIR/converter.py $BASE_VECTORS $ASPECT_DIR/$FOLD/explirefit_$MODEL_NAME/model $ASPECT_DIR/$FOLD/explirefit_$MODEL_NAME/pwc_id2vec.txt
        
        conda activate pairwise-vs-segment
        python -m gensim.scripts.glove2word2vec -i $ASPECT_DIR/$FOLD/explirefit_$MODEL_NAME/pwc_id2vec.txt -o $ASPECT_DIR/$FOLD/explirefit_$MODEL_NAME/pwc_id2vec.w2v.txt
    done
done


export PYTHONUNBUFFERED=""

conda deactivate