#!/usr/bin/env python
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Union, Dict, Tuple

import fire
import numpy as np
import pandas as pd
from datasets import load_dataset
from gensim.models import KeyedVectors
from pandas import DataFrame
from tqdm.auto import tqdm

from experiments import basic_logger_config
from experiments.evaluation.utils import get_avg_precision, get_reciprocal_rank, compute_dcg_at_k
from hf_datasets.paperswithcode_aspects import get_test_split
from experiments.utils import get_local_hf_dataset_path

logging.basicConfig(**basic_logger_config)
logger = logging.getLogger(__name__)


def evaluate_vectors(
        hf_dataset: str,
        aspect: str,
        input_path: str,
        name: str,
        folds: Union[str, list],
        top_ks: Union[str, list],
        output_path: str
    ):
    """

    Run with: $ ./eval_cli.py evaluate_vectors paperswithcode_aspects task ./output/pwc_doc_id2st.txt --name=sentence_transformers --folds=1,2,3,4 --top_ks=5,10,25,50 --output_path=./output/eval.csv

    :param aspect:
    :param folds:
    :param top_ks:
    :param name:
    :param hf_dataset:
    :param input_path:
    :param output_path:
    :return:
    """

    if isinstance(folds, str):
        folds = folds.split(',')
    elif isinstance(folds, int):
        folds = [folds]

    if isinstance(top_ks, str):
        top_ks = top_ks.split(',')
    elif isinstance(top_ks, int):
        top_ks = [top_ks]

    logger.info(f'Folds: {folds}')
    logger.info(f'Top-Ks: {top_ks}')

    if len(folds) < 1:
        logger.error('No folds provided')
        return

    if len(top_ks) < 1:
        logger.error('No top-k values provided')
        return

    # Load documents
    doc_model = KeyedVectors.load_word2vec_format(input_path)
    logger.info(f'Document vectors: {doc_model.vectors.shape}')

    # Normalize vectors
    doc_model.init_sims(replace=True)

    # Init dataframe
    metrics = ['retrieved_docs', 'relevant_docs', 'relevant_retrieved_docs', 'precision', 'recall', 'avg_p',
               'reciprocal_rank']
    df = pd.DataFrame([], columns=['name', 'fold', 'top_k'] + metrics)

    # Iterate over folds
    for fold in folds:
        logger.info(f'Current fold: {fold}')

        # Dataset
        test_ds = load_dataset(
            get_local_hf_dataset_path(hf_dataset),
            name='relations',
            cache_dir='./data/nlp_cache',
            split=get_test_split(aspect, fold)
        )

        logger.info(f'Test samples: {len(test_ds):,}')

        # Unique paper IDs in test set
        test_paper_ids = set(test_ds['from_paper_id']).union(set(test_ds['to_paper_id']))

        logger.info(f'Test paper IDs: {len(test_paper_ids):,}')
        logger.info(f'Examples: {list(test_paper_ids)[:10]}')

        # Relevance mapping
        doc_id2related_ids = defaultdict(set)  # type: Dict[Set[str]]
        for row in test_ds:
            if row['label'] == 'y':
                a = row['from_paper_id']
                b = row['to_paper_id']
                doc_id2related_ids[a].add(b)
                doc_id2related_ids[b].add(a)

        # Filter for documents in test set
        test_doc_model = KeyedVectors(vector_size=doc_model.vector_size)
        test_doc_ids = []
        test_doc_vectors = []
        missed_doc_ids = 0

        for doc_id in doc_model.vocab:
            if doc_id in test_paper_ids:
                vec = doc_model.get_vector(doc_id)
                if len(vec) != doc_model.vector_size:
                    raise ValueError(f'Test document as invalid shape: {doc_id} => {vec.shape}')

                test_doc_ids.append(doc_id)
                test_doc_vectors.append(vec)
            else:
                missed_doc_ids += 1
                # logger.warning(f'Document ID is not part of test set: {doc_id} ({type(doc_id)})')

        if len(test_doc_ids) != len(test_doc_vectors):
            raise ValueError(f'Test document IDs does not match vector count: {len(test_doc_ids)} vs {len(test_doc_vectors)}')

        logger.info(f'Test document IDs: {len(test_doc_ids)} (missed {missed_doc_ids})')
        logger.info(f'Test document vectors: {len(test_doc_vectors)}')

        test_doc_model.add(test_doc_ids, test_doc_vectors)
        test_doc_model.init_sims(replace=True)

        logger.info(f'Test document vectors: {test_doc_model.vectors.shape}')

        # Actual evaluation
        # k2eval_rows = defaultdict(list)
        seed_ids_without_recommendations = []
        max_top_k = max(top_ks)
        eval_rows = {top_k: defaultdict(list) for top_k in top_ks}  # top_k => metric_name => list of value

        for seed_id in tqdm(test_paper_ids, desc=f'Evaluation (fold={fold})'):
            try:
                rel_docs = doc_id2related_ids[seed_id]
                max_ret_docs = [d for d, score in test_doc_model.most_similar(seed_id, topn=max_top_k)]
                for top_k in top_ks:
                    ret_docs = max_ret_docs[:top_k]
                    rel_ret_docs_count = len(set(ret_docs) & set(rel_docs))

                    if ret_docs and rel_docs:
                        # Precision = No. of relevant documents retrieved / No. of total documents retrieved
                        precision = rel_ret_docs_count / len(ret_docs)

                        # Recall = No. of relevant documents retrieved / No. of total relevant documents
                        recall = rel_ret_docs_count / len(rel_docs)

                        # Avg. precision (for MAP)
                        avg_p = get_avg_precision(ret_docs, rel_docs)

                        # Reciprocal rank (for MRR)
                        reciprocal_rank = get_reciprocal_rank(ret_docs, rel_docs)

                        # # NDCG@k
                        # predicted_relevance = [1 if ret_doc_id in rel_docs else 0 for ret_doc_id in ret_docs]
                        # true_relevances = [1] * len(rel_docs)
                        # ndcg_value = self.compute_dcg_at_k(predicted_relevance, top_k) / self.compute_dcg_at_k(true_relevances, top_k)

                        # Save metrics
                        eval_rows[top_k]['retrieved_docs'].append(len(ret_docs))
                        eval_rows[top_k]['relevant_docs'].append(len(rel_docs))
                        eval_rows[top_k]['relevant_retrieved_docs'].append(rel_ret_docs_count)
                        eval_rows[top_k]['precision'].append(precision)
                        eval_rows[top_k]['recall'].append(recall)
                        eval_rows[top_k]['avg_p'].append(avg_p)
                        eval_rows[top_k]['reciprocal_rank'].append(reciprocal_rank)

            except (IndexError, ValueError, KeyError) as e:
                seed_ids_without_recommendations.append(seed_id)

                logger.warning(f'Cannot retrieve recommendations for #{seed_id}: {e}')

        logger.info(
            f'Completed with {len(eval_rows[top_ks[0]][metrics[0]]):,} rows (missed {len(seed_ids_without_recommendations):,})')

        # Summarize evaluation
        for top_k in top_ks:
            try:
                row = [name, fold, top_k]
                for metric in metrics:
                    # mean over all metrics
                    values = eval_rows[top_k][metric]
                    if len(values) > 0:
                        row.append(np.mean(values))
                    else:
                        row.append(None)

                df.loc[len(df)] = row

            except ValueError as e:
                logger.error(f'Cannot summarize row: {top_k} {fold} {metrics} {e}')

            #
            #
            # df = pd.DataFrame(k2eval_rows[top_k],
            #                   columns=['seed_id', 'retrieved_docs', 'relevant_docs', 'relevant_retrieved_docs',
            #                            'precision', 'recall', 'avg_p', 'reciprocal_rank'])
            #
            # print(df.mean())
            #
            # print(df.mean().to_frame().transpose().iloc[0])

    logger.info(f'Writing {len(df)} rows to {output_path}')

    if os.path.exists(output_path):
        # Append new rows to evaluation file
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        # Write new files
        df.to_csv(output_path, header=True, index=False)

    logger.info('Done')


def reevaluate():
    """
    Evaluate all systems again!

    :return:
    """
    hf_dataset = 'paperswithcode_aspects'
    folds = [1, 2, 3, 4]
    aspects = ['task', 'method', 'dataset']
    top_ks = [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100]

    output_path = './output/pwc'

    eval_path = os.path.join(output_path, 'reeval.csv')

    def get_evaluation_df(name, doc_model, hf_dataset, aspect, fold) -> Tuple[DataFrame, Dict]:
        # Init dataframe
        metrics = ['retrieved_docs', 'relevant_docs', 'relevant_retrieved_docs', 'precision', 'recall', 'avg_p',
                   'reciprocal_rank', 'ndcg']
        df = pd.DataFrame([], columns=['name', 'aspect', 'fold', 'top_k'] + metrics)

        # Dataset
        test_ds = load_dataset(
            get_local_hf_dataset_path(hf_dataset),
            name='relations',
            cache_dir='./data/nlp_cache',
            split=get_test_split(aspect, fold)
        )

        logger.info(f'Test samples: {len(test_ds):,}')

        # Unique paper IDs in test set
        test_paper_ids = set(test_ds['from_paper_id']).union(set(test_ds['to_paper_id']))

        logger.info(f'Test paper IDs: {len(test_paper_ids):,}')
        logger.info(f'Examples: {list(test_paper_ids)[:10]}')

        # Relevance mapping
        doc_id2related_ids = defaultdict(set)  # type: Dict[Set[str]]
        for row in test_ds:
            if row['label'] == 'y':
                a = row['from_paper_id']
                b = row['to_paper_id']
                doc_id2related_ids[a].add(b)
                doc_id2related_ids[b].add(a)

        # Filter for documents in test set
        test_doc_model = KeyedVectors(vector_size=doc_model.vector_size)
        test_doc_ids = []
        test_doc_vectors = []
        missed_doc_ids = 0

        for doc_id in doc_model.vocab:
            if doc_id in test_paper_ids:
                vec = doc_model.get_vector(doc_id)
                if len(vec) != doc_model.vector_size:
                    raise ValueError(f'Test document as invalid shape: {doc_id} => {vec.shape}')

                test_doc_ids.append(doc_id)
                test_doc_vectors.append(vec)
            else:
                missed_doc_ids += 1
                # logger.warning(f'Document ID is not part of test set: {doc_id} ({type(doc_id)})')

        if len(test_doc_ids) != len(test_doc_vectors):
            raise ValueError(
                f'Test document IDs does not match vector count: {len(test_doc_ids)} vs {len(test_doc_vectors)}')

        logger.info(f'Test document IDs: {len(test_doc_ids)} (missed {missed_doc_ids})')
        logger.info(f'Test document vectors: {len(test_doc_vectors)}')

        test_doc_model.add(test_doc_ids, test_doc_vectors)
        test_doc_model.init_sims(replace=True)

        logger.info(f'Test document vectors: {test_doc_model.vectors.shape}')

        # Actual evaluation
        # k2eval_rows = defaultdict(list)
        seed_ids_without_recommendations = []
        max_top_k = max(top_ks)
        eval_rows = {top_k: defaultdict(list) for top_k in top_ks}  # top_k => metric_name => list of value

        seed_id2ret_docs = {}

        for seed_id in tqdm(test_paper_ids, desc=f'Evaluation ({name},aspect={aspect},fold={fold})'):
            try:
                rel_docs = doc_id2related_ids[seed_id]
                max_ret_docs = [d for d, score in test_doc_model.most_similar(seed_id, topn=max_top_k)]
                seed_id2ret_docs[seed_id] = max_ret_docs

                for top_k in top_ks:
                    ret_docs = max_ret_docs[:top_k]
                    rel_ret_docs_count = len(set(ret_docs) & set(rel_docs))

                    if ret_docs and rel_docs:
                        # Precision = No. of relevant documents retrieved / No. of total documents retrieved
                        precision = rel_ret_docs_count / len(ret_docs)

                        # Recall = No. of relevant documents retrieved / No. of total relevant documents
                        recall = rel_ret_docs_count / len(rel_docs)

                        # Avg. precision (for MAP)
                        avg_p = get_avg_precision(ret_docs, rel_docs)

                        # Reciprocal rank (for MRR)
                        reciprocal_rank = get_reciprocal_rank(ret_docs, rel_docs)

                        # # NDCG@k
                        predicted_relevance = [1 if ret_doc_id in rel_docs else 0 for ret_doc_id in ret_docs]
                        true_relevances = [1] * len(rel_docs)
                        ndcg_value = compute_dcg_at_k(predicted_relevance, top_k) / compute_dcg_at_k(true_relevances,
                                                                                                     top_k)

                        # Save metrics
                        eval_rows[top_k]['retrieved_docs'].append(len(ret_docs))
                        eval_rows[top_k]['relevant_docs'].append(len(rel_docs))
                        eval_rows[top_k]['relevant_retrieved_docs'].append(rel_ret_docs_count)
                        eval_rows[top_k]['precision'].append(precision)
                        eval_rows[top_k]['recall'].append(recall)
                        eval_rows[top_k]['avg_p'].append(avg_p)
                        eval_rows[top_k]['reciprocal_rank'].append(reciprocal_rank)
                        eval_rows[top_k]['ndcg'].append(ndcg_value)

            except (IndexError, ValueError, KeyError) as e:
                seed_ids_without_recommendations.append(seed_id)

                logger.warning(f'Cannot retrieve recommendations for #{seed_id}: {e}')

        logger.info(
            f'Completed with {len(eval_rows[top_ks[0]][metrics[0]]):,} rows (missed {len(seed_ids_without_recommendations):,})')

        # Summarize evaluation
        for top_k in top_ks:
            try:
                row = [
                    name,
                    aspect,
                    fold,
                    top_k
                ]
                for metric in metrics:
                    # mean over all metrics
                    values = eval_rows[top_k][metric]
                    if len(values) > 0:
                        row.append(np.mean(values))
                    else:
                        row.append(None)

                df.loc[len(df)] = row

            except ValueError as e:
                logger.error(f'Cannot summarize row: {top_k} {fold} {metrics} {e}')

        return df, seed_id2ret_docs

    # generic embeddings
    generic_models = {aspect: {fold: {} for fold in folds} for aspect in aspects}
    generic_seed_id2ret_docs = {aspect: {fold: {} for fold in folds} for aspect in aspects}

    for fn in os.listdir(output_path):
        if fn.endswith('.w2v.txt') and (fn != 'fasttext.w2v.txt' or '_cls' in fn):  # exclude word vectors, CLS pooling
            input_path = os.path.join(output_path, fn)
            name = fn.replace('.w2v.txt', '')

            # Load documents
            doc_model = KeyedVectors.load_word2vec_format(input_path)
            logger.info(f'Document vectors: {doc_model.vectors.shape}')

            # Normalize vectors
            doc_model.init_sims(replace=True)

            # For folds and aspects
            for aspect in aspects:
                for fold in folds:
                    # Compute results
                    df, seed_id2ret_docs = get_evaluation_df(name, doc_model, hf_dataset, aspect, fold)

                    generic_models[aspect][fold][name] = doc_model
                    generic_seed_id2ret_docs[aspect][fold][name] = seed_id2ret_docs

                    logger.info(f'Writing {len(df)} rows to {eval_path}')

                    if os.path.exists(eval_path):
                        # Append new rows to evaluation file
                        df.to_csv(eval_path, mode='a', header=False, index=False)
                    else:
                        # Write new files
                        df.to_csv(eval_path, header=True, index=False)
    # save to disk
    json.dump(generic_seed_id2ret_docs, open(os.path.join(output_path, 'generic_seed_id2ret_docs.json'), 'w'))

    # special embeddings
    special_models = {aspect: {fold: {} for fold in folds} for aspect in aspects}
    special_seed_id2ret_docs = {aspect: {fold: {} for fold in folds} for aspect in aspects}

    for aspect in aspects:
        for fold in folds:
            aspect_fold_dir = os.path.join(output_path, aspect, str(fold))
            for name in os.listdir(aspect_fold_dir):
                input_path = os.path.join(aspect_fold_dir, name, 'pwc_id2vec.w2v.txt')

                if not os.path.exists(input_path):
                    continue

                if name in special_models[aspect][fold] or name in special_seed_id2ret_docs[aspect][fold]:
                    # results exist already
                    continue

                # Load documents
                doc_model = KeyedVectors.load_word2vec_format(input_path)
                logger.info(f'Document vectors: {doc_model.vectors.shape}')

                # Normalize vectors
                doc_model.init_sims(replace=True)

                # Compute results
                df, seed_id2ret_docs = get_evaluation_df(name, doc_model, hf_dataset, aspect, fold)

                special_models[aspect][fold][name] = doc_model
                special_seed_id2ret_docs[aspect][fold][name] = seed_id2ret_docs

                logger.info(f'Writing {len(df)} rows to {eval_path}')

                if os.path.exists(eval_path):
                    # Append new rows to evaluation file
                    df.to_csv(eval_path, mode='a', header=False, index=False)
                else:
                    # Write new files
                    df.to_csv(eval_path, header=True, index=False)
    # save retrieved docs to disk
    json.dump(special_seed_id2ret_docs, open(os.path.join(output_path, 'special_seed_id2ret_docs.json'), 'w'))

    logger.info('done')


if __name__ == '__main__':
    fire.Fire()
    sys.exit(0)
