import logging
import os
from typing import List

import pandas as pd
import numpy as np

from collections import defaultdict

from gensim.models import KeyedVectors
from sentence_transformers.evaluation import SentenceEvaluator
from tqdm.auto import tqdm

from experiments.evaluation.utils import get_avg_precision, get_reciprocal_rank

logger = logging.getLogger(__name__)


class NearestNeighborsEvaluator(SentenceEvaluator):
    """
    This class evaluates an Nearest Neighbors (NN) setting.

    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures
    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)
    """
    def __init__(self, model, docs_ds, test_ds, top_ks: List[int], batch_size: int, show_progress_bar=True, main_metric='avg_p'):
        self.model = model
        self.top_ks = top_ks
        self.main_metric = main_metric
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.docs_ds = docs_ds
        self.test_ds = test_ds
        self.csv_file: str = "nearest_neighbours_evaluation_results.csv"

        # Doc mapping
        self.doc_id2doc = {doc['paper_id']: doc for doc in docs_ds}

        # Unique paper IDs in test set
        self.test_paper_ids = set(test_ds['from_paper_id']).union(set(test_ds['to_paper_id']))

        # Relevance mapping
        self.doc_id2related_ids = defaultdict(set)  # type: Dict[Set[str]]
        for row in test_ds:
            if row['label'] == 'y':
                a = row['from_paper_id']
                b = row['to_paper_id']
                self.doc_id2related_ids[a].add(b)
                self.doc_id2related_ids[b].add(a)

        logger.info(f'Evaluator initialized: {len(self.doc_id2related_ids):,} relevance labels')

        # TODO pre tokenize docs!
        idx2paper_id = {}
        paper_id2idx = {}
        texts = []
        self.paper_ids = []

        # get document texts
        for idx, paper_id in enumerate(self.test_paper_ids):
            idx2paper_id[idx] = paper_id
            paper_id2idx[paper_id] = idx

            doc = self.doc_id2doc[paper_id]
            text = doc['title'] + ': ' + doc['abstract']

            texts.append(text)
            self.paper_ids.append(paper_id)

        logger.info(f'Pre tokenize all test texts...')
        self.tokenized_texts = self.model.tokenize(texts)

        # TODO tokenize with num_workers=env['workers']  ==> not possible with ST

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.

        :param model:
            the model to evaluate
        :param output_path:
            path where predictions and metrics are written to
        :param epoch
            the epoch where the evaluation takes place.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation on test data.
        :param steps
            the steps in the current epoch at time of the evaluation.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation at the end of the epoch.
        :return: a score for the evaluation with a higher score indicating a better result
        """
        # idx2paper_id = {}
        # paper_id2idx = {}
        # texts = []
        # paper_ids = []
        #
        # # get document texts
        # for idx, paper_id in enumerate(self.test_paper_ids):
        #     idx2paper_id[idx] = paper_id
        #     paper_id2idx[paper_id] = idx
        #
        #     doc = self.doc_id2doc[paper_id]
        #     texts.append(doc['title'] + ': ' + doc['abstract'])
        #     paper_ids.append(paper_id)

        logger.info('Encode test documents...')
        embeddings = model.encode(self.tokenized_texts, is_pretokenized=True, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)

        # Filter for documents in test set
        test_doc_model = KeyedVectors(vector_size=model.get_sentence_embedding_dimension())

        #for idx, embedding in enumerate(embeddings):
        #    test_doc_model.add([idx2paper_id[idx]], [embedding])
        test_doc_model.add(self.paper_ids, embeddings.tolist())
        
        test_doc_model.init_sims(replace=True)
        logger.info(f'Test document vectors: {test_doc_model.vectors.shape}')

        # Init dataframe
        metrics = ['retrieved_docs', 'relevant_docs', 'relevant_retrieved_docs', 'precision', 'recall', 'avg_p',
                   'reciprocal_rank', 'ndcg']
        df = pd.DataFrame([], columns=['epoch', 'steps', 'top_k'] + metrics)

        max_top_k = max(self.top_ks)
        eval_rows = {top_k: defaultdict(list) for top_k in self.top_ks}  # top_k => metric_name => list of value
        seed_ids_without_recommendations = []

        for seed_id in tqdm(self.test_paper_ids, desc=f'Evaluation'):
            try:
                rel_docs = self.doc_id2related_ids[seed_id]
                max_ret_docs = [d for d, score in test_doc_model.most_similar(seed_id, topn=max_top_k)]

                for top_k in self.top_ks:
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

                        # NDCG@k
                        predicted_relevance = [1 if ret_doc_id in rel_docs else 0 for ret_doc_id in ret_docs]
                        true_relevances = [1] * len(rel_docs)
                        ndcg_value = self.compute_dcg_at_k(predicted_relevance, top_k) / self.compute_dcg_at_k(true_relevances, top_k)

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
            f'Completed with {len(eval_rows[self.top_ks[0]][metrics[0]]):,} rows (missed {len(seed_ids_without_recommendations):,})')

        # Summarize evaluation
        for top_k in self.top_ks:
            try:
                row = [epoch, steps, top_k]
                for metric in metrics:
                    # mean over all metrics
                    values = eval_rows[top_k][metric]
                    if len(values) > 0:
                        row.append(np.mean(values))
                    else:
                        row.append(None)

                df.loc[len(df)] = row

            except ValueError as e:
                logger.error(f'Cannot summarize row: {top_k} {metrics} {e}')

        output_csv_path = os.path.join(output_path, self.csv_file)

        logger.info(f'Writing {len(df)} rows to {output_csv_path}')
        logger.info(f'Results:\n{df.to_markdown()}')

        if os.path.exists(output_csv_path):
            # Append new rows to evaluation file
            df.to_csv(output_csv_path, mode='a', header=False, index=False)
        else:
            # Write new files
            df.to_csv(output_csv_path, header=True, index=False)

        # Return score from main metric
        if len(df) > 0:
            main_score = df.iloc[0][self.main_metric]
            logger.info(f'Evaluation completed: {self.main_metric} = {main_score}')
            return main_score
        else:
            logger.warning('No evaluation rows available... score = 0')
            return 0
    
    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  #+2 as we start our idx at 0
        return dcg
