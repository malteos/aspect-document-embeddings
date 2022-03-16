#!/usr/bin/env python
import logging
import os
import sys
from typing import Union

import fire
import pyarrow
from sentence_transformers.models import Pooling, Transformer
from smart_open import open
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, losses
import torch

from torch.utils.data import DataLoader

from experiments import basic_logger_config
from experiments.environment import get_env
from experiments.sentence_transformers.dataset import DocumentPairSentencesDataset
from experiments.sentence_transformers.nearest_neighbors_evaluator import NearestNeighborsEvaluator
from experiments.utils import get_local_hf_dataset_path


from datasets import load_dataset, Dataset
from hf_datasets.paperswithcode_aspects import get_test_split, get_train_split


logging.basicConfig(**basic_logger_config)
logger = logging.getLogger(__name__)
env = get_env()


def train(
        model_name_or_path: str,
        hf_dataset: str,
        aspect: str,
        fold: Union[int, str],
        output_path: str,
        train_epochs: int = 3,
        train_batch_size: int = 25,
        eval_batch_size: int = 32,
        evaluation_steps: int = 5000,
        train_on_test: bool = False,
        loss: str = 'multiple_negatives_ranking',
        override: bool = False):
    """

    # $MODEL_NAME $HF_DATASET $ASPECT $FOLD $OUTPUT_DIR --train_epochs=3 --train_batch_size=$TRAIN_BATCH_SIZE --eval_batch_size=$EVAL_BATCH_SIZE

    Run with:
    $ export CUDA_VISIBLE_DEVICES=1
    $ ./sentence_transformer_cli.py train scibert-scivocab-uncased paperswithcode_task_docs 1 ./output/st_scibert/1 --train_epochs=3 --train_batch_size=25 --eval_batch_size=32


    :param loss: Training loss function (choices: multiple_negatives_ranking, cosine)
    :param train_on_test: If True, joint training on train and test set (validation disabled)
    :param aspect:
    :param evaluation_steps:
    :param train_epochs:
    :param model_name_or_path:
    :param hf_dataset:
    :param fold:
    :param output_path:
    :param train_batch_size:
    :param eval_batch_size:
    :param override:
    :return:
    """

    top_ks = [5,10,25,50]
    # cuda_device = -1

    # hf_dataset = 'paperswithcode_task_docs'
    # model_name_or_path = 'scibert-scivocab-uncased'
    # fold = 1
    max_token_length = 336 # ssee pwc_token_stats.ipynb
    nlp_cache_dir = './data/nlp_cache'

    # train_batch_size = 25
    # eval_batch_size = 32
    # override = False

    # output_path = './output/pwc_task_st/1/sci-bert'
    # output_path = os.path.join(output_path, str(fold), model_name_or_path)  # output/1/sci-bert

    if os.path.exists(output_path) and not override:
        logger.error(f'Stop. Output path exists already: {output_path}')
        sys.exit(1)

    # if cuda_device >= 0:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model path from env
    if not os.path.exists(model_name_or_path) and os.path.exists(
            os.path.join(env['bert_dir'], model_name_or_path)):
        model_name_or_path = os.path.join(env['bert_dir'], model_name_or_path)

    word_embedding_model = Transformer(model_name_or_path, max_seq_length=max_token_length)
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    # tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

    # dataset
    docs_ds = load_dataset(get_local_hf_dataset_path(hf_dataset),
                       name='docs',
                       cache_dir=nlp_cache_dir,
                          split='docs')
    train_ds = load_dataset(get_local_hf_dataset_path(hf_dataset),
                       name='relations',
                       cache_dir=nlp_cache_dir,
                       split=get_train_split(aspect, fold))
    test_ds = load_dataset(get_local_hf_dataset_path(hf_dataset),
                       name='relations',
                       cache_dir=nlp_cache_dir,
                       split=get_test_split(aspect, fold))

    # filter for positive labels only
    train_ds = train_ds.filter(lambda row: row['label'] == 'y')

    logger.info(f'After filtering: {len(train_ds):,}')

    # joint training on train and test?
    if train_on_test:
        #
        # import pyarrow
        # from datasets.arrow_dataset import Dataset
        #
        # full_ds_table = pyarrow.concat_tables([train_ds.data, test_ds.data])
        # full_ds = Dataset(arrow_table=full_ds_table)
        raise NotImplementedError('TODO Evaluator')
    else:
        # standard training on test only
        train_sds = DocumentPairSentencesDataset(docs_ds, train_ds, model, max_length=max_token_length, forced_length=0)
        train_sds.tokenize_all_docs()

        evaluator = NearestNeighborsEvaluator(model, docs_ds, test_ds, top_ks=top_ks, batch_size=eval_batch_size, show_progress_bar=True)

    if loss == 'cosine':
        train_loss = losses.CosineSimilarityLoss(model)
    elif loss == 'multiple_negatives_ranking':
        # A nice advantage of MultipleNegativesRankingLoss is that it only requires positive pairs
        # https://github.com/UKPLab/sentence-transformers/tree/master/examples/training/quora_duplicate_questions
        train_loss = losses.MultipleNegativesRankingLoss(model)
    else:
        raise ValueError(f'Unsupported loss function: {loss}')

    train_dl = DataLoader(train_sds, shuffle=True, batch_size=train_batch_size)

    # Training
    model.fit(
        train_objectives=[(train_dl, train_loss)],
        epochs=train_epochs, # try 1-4
        warmup_steps=100,
        evaluator=evaluator,
        evaluation_steps=evaluation_steps,  # increase to 5000 (full dataset => 20k steps)
        output_path=output_path,
        output_path_ignore_not_empty=True
    )

    logger.info('Training done')


def build_vectors(
        st_output_path: str,
        hf_dataset: str,
        aspect: str,
        fold: Union[int, str],
        include_all_docs: bool = False,
        override: bool = False
    ):
    """

    :param override:
    :param include_all_docs: Generate also vectors for samples from training data
    :param st_output_path: Path to Sentence Transformer model
    :param hf_dataset: Huggingface dataset path or name
    :param aspect:
    :param fold:
    :return:
    """
    max_token_length = 336  # ssee pwc_token_stats.ipynb
    nlp_cache_dir = './data/nlp_cache'

    out_fn = 'pwc_id2vec__all_docs.w2v.txt' if include_all_docs else 'pwc_id2vec.w2v.txt'
    out_fp = os.path.join(st_output_path, out_fn)

    if not os.path.exists(st_output_path):
        logger.error(f'Sentence Transformer directory does not exist: {st_output_path}')
        return

    if os.path.exists(out_fp) and not override:
        logger.error(f'Output path exists already and override is disabled: {out_fp}')
        return

    # Inference for best model
    best_model = SentenceTransformer(st_output_path)
    best_model.get_sentence_embedding_dimension()

    test_ds = load_dataset(get_local_hf_dataset_path(hf_dataset),
                       name='relations',
                       cache_dir=nlp_cache_dir,
                       split=get_test_split(aspect, fold))

    docs_ds = load_dataset(get_local_hf_dataset_path(hf_dataset),
                       name='docs',
                       cache_dir=nlp_cache_dir,
                          split='docs')
    test_sds = DocumentPairSentencesDataset(docs_ds, test_ds, best_model)

    if include_all_docs:
        # use all document ids
        input_paper_ids = set(docs_ds['paper_id'])
        logger.info(f'All documents in corpus: {len(input_paper_ids):,}')

    else:
        # generate vectors from unique test documents only
        input_paper_ids = set(test_ds['from_paper_id']).union(set(test_ds['to_paper_id']))

    with open(out_fp, 'w') as f:
        # header
        f.write(f'{len(input_paper_ids)} {best_model.get_sentence_embedding_dimension()}\n')

        # body
        for paper_id in tqdm(input_paper_ids, desc='Inference'):
            vec = [str(v) for v in best_model.encode(test_sds.get_text_from_doc(paper_id), show_progress_bar=False)]

            assert len(vec) == best_model.get_sentence_embedding_dimension()

            vec_str = ' '.join(vec)
            line = f'{paper_id} {vec_str}\n'
            f.write(line)
            # break
    logger.info(f'Encoded {len(input_paper_ids):,} into {out_fp}')


if __name__ == '__main__':
    fire.Fire()
    sys.exit(0)
