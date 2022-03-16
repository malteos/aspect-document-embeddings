#!/usr/bin/env bash

# cd to repo root
export IN=./output/pwc
export OUT=./output/release

mkdir -p $OUT

# dataset files
# ...

# generic: fasttext, specter, scibert
# joint archive is too large for github  => 1.4GB
tar -zcvf $OUT/generic_vectors.tar.gz $IN/fasttext.w2v.txt $IN/avg_fasttext.w2v.txt $IN/specter.w2v.txt $IN/scibert_mean.w2v.txt
#for SYS in "fasttext" "avg_fasttext" "specter" "scibert_mean"
#do
#    tar -zcvf $OUT/$SYS.tar.gz $IN/$SYS.w2v.txt
#done

# trained models and generated document vectors of specialized
# only for a single fold
export FOLD=1

# tar --exclude='./folder' --exclude='./upload/folder2' -zcvf /backup/filename.tgz .

for ASPECT in "task" "method" "dataset"
do
    for SYS in "explirefit_avg_fasttext" "explirefit_scibert_mean" "explirefit_specter" "scibert-scivocab-uncased_fine_tuned" "specter_fine_tuned" "st_scibert-scivocab-uncased"
    do
        # split archive into 1GB chunks for uploading
        # ---
        # exclude "pwc_id2vec.txt" (only .w2v.txt is needed)
        # exclude "training_state_epoch_*.th" and "model_state_epoch_*.th" (only best.th is needed)
        tar --exclude="*pwc_id2vec.txt" \
            --exclude="*training_state_epoch_0.th" \
            --exclude="*training_state_epoch_1.th" \
            --exclude="*model_state_epoch_0.th" \
            --exclude="*model_state_epoch_1.th" -zcvf - $IN/$ASPECT/$FOLD/$SYS/ | split --bytes=1GB - $OUT/${ASPECT}__${FOLD}__${SYS}.tar.gz.

    done
done


# precomputed results
tar -zcvf $OUT/special_seed_id2ret_docs.tar.gz $IN/special_seed_id2ret_docs.json
tar -zcvf $OUT/generic_seed_id2ret_docs.tar.gz $IN/generic_seed_id2ret_docs.json

# reval (uncompressed)
cp $IN/reeval.csv $OUT/reeval.csv

# whoosh index
tar -zcvf $OUT/whoosh_index.tar.gz $IN/whoosh_index/

# ...

# put to zendo/github

# Github: We don't limit the total size of the binary files in the release or the bandwidth used to deliver them. However, each individual file must be smaller than 2 GB.

export GITHUB_TOKEN=
export GITHUB_RELEASE=1  # release must be created manually first!!!
export GITHUB_USER=
export GITHUB_REPO=

# Upload via https://github.com/github-release/github-releas
for FPATH in $OUT/*
do
    FNAME=$(basename $FPATH)
    echo "Uploading $FNAME ..."

    ~/github-release upload --user $GITHUB_USER --repo $GITHUB_REPO --tag $GITHUB_RELEASE --name $FNAME --file $FPATH
done


# Upload via release API: https://github.blog/2013-09-25-releases-api-preview/
for FPATH in $OUT/*
do
    FNAME=$(basename $FPATH)
    echo "Uploading $FNAME ..."

    curl -H "Authorization: token $GITHUB_TOKEN" \
         -H "Accept: application/vnd.github.v3+json" \
         -H "Content-Type: $(file -b --mime-type $FPATH)" \
         --data-binary @$FPATH "https://api.github.com/repos/$GITHUB_USER/$GITHUB_REPO/releases/$GITHUB_RELEASE/assets?name=$FNAME"
done

# Print download commands
for FPATH in $OUT/*
do
    FNAME=$(basename $FPATH)
    echo "wget https://github.com/$GITHUB_USER/$GITHUB_REPO/releases/download/$GITHUB_RELEASE/$FNAME"

done
