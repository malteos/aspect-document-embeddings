import json
import logging
import os
import sys

import fire
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from smart_open import open

from experiments import basic_logger_config
from paperswithcode.utils import assert_leakage, get_aspect_pairs, get_paper_id

logging.basicConfig(**basic_logger_config)
logger = logging.getLogger(__name__)


def save_dataset(input_dir, output_dir, cv_folds=4, neg_ratio=0.5):
    """

    Run with: $ python -m paperswithcode.dataset save_dataset <input_dir> <output_dir>

    Download pwc files first:

    $ wget https://paperswithcode.com/media/about/papers-with-abstracts.json.gz

    After dataset creation use the following commands to compress and upload all files:

    cd <output_dir>
    tar -cvzf paperswithcode_aspects.tar.gz docs.jsonl folds/
    curl --upload-file paperswithcode_aspects.tar.gz ftp://$FTP_LOGIN:$FTP_PASSWORD@fiq.de/datasets/

    Supported aspects:
    - task
    - method
    - dataset

    :param input_dir:
    :param output_dir:
    :param cv_folds:
    :param neg_ratio:
    :return:
    """

    doc_a_col = 'from_paper_id'
    doc_b_col = 'to_paper_id'
    label_col = 'label'
    pos_target = 'y'
    neg_target = 'n'
    max_papers_per_aspect = 100
    # max_urls_per_task = 100
    # max_urls_per_method = 100

    sample_size = 100

    logger.info(f'Loading data from: {input_dir}')

    papers = json.load(open(os.path.join(input_dir, 'papers-with-abstracts.json.gz')))
    logger.info(f'Before filter: {len(papers):,}')

    # Prepare paper data
    paper_id2paper = {}
    duplicated_ids = 0
    invalid_data = 0

    for p in papers:
        if p['title'] and p['abstract'] and p['url_abs']:
            paper_id = get_paper_id(p['url_abs'])

            if paper_id in paper_id2paper:
                logger.warning(f'Paper ID is not unique: {paper_id}')
                logger.warning(f' - New paper: {p}')
                logger.warning(f' - Existing paper: {paper_id2paper[paper_id]}')
                duplicated_ids += 1
            else:
                # prepare methods
                p['aspect_methods'] = [m['name'] for m in p['methods']]  # TODO is name unique?

                # init data list
                p['aspect_datasets'] = []

                # tasks are already a property
                p['aspect_tasks'] = p['tasks']

                p['paper_id'] = paper_id
                paper_id2paper[paper_id] = p
        else:
            logger.warning(f'Invalid paper data: {p}')
            invalid_data += 1

    # Add `dataset` information
    evaluation_tables = json.load(open(os.path.join(input_dir, 'evaluation-tables.json.gz')))
    sota_row_unknown_ids = 0
    for eval_item in evaluation_tables:
        for ds in eval_item['datasets']:
            dataset_name = ds['dataset']

            for sota_row in ds['sota']['rows']:
                # This is not the `paper_url` from pwc_papers but `url_abs`
                paper_id = get_paper_id(sota_row['paper_url'])
                if paper_id in paper_id2paper:
                    if dataset_name not in paper_id2paper[paper_id]['aspect_datasets']:
                        # Add dataset to paper data
                        paper_id2paper[paper_id]['aspect_datasets'].append(dataset_name)
                else:
                    logger.warning(f'SOTA-row has unknown paper id: {paper_id}')
                    sota_row_unknown_ids += 1

    logger.info(f'After filter: {len(paper_id2paper):,} (invalid: {invalid_data}, duplicates: {duplicated_ids}, sota-row unknown ids: {sota_row_unknown_ids})')

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=0)

    paper_ids = list(paper_id2paper.keys())
    np_paper_ids = np.array(paper_ids)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all folds
    for k, (train_index, test_index) in enumerate(kf.split(paper_ids), 1):
        # print(type(train_index))
        # print(train_index[:2])

        train_paper_ids = np_paper_ids[train_index].tolist()
        test_paper_ids = np_paper_ids[test_index].tolist()

        logger.info(f'Fold: {k}, Train: {len(train_paper_ids):,}; Test: {len(test_paper_ids):,}')

        # Generate samples for each aspect
        for aspect in ['task', 'method', 'dataset']:
            # Aspect pairs
            train_aspect_pairs, train_neg_aspect_pairs = get_aspect_pairs(
                paper_id2paper,
                train_paper_ids,
                aspect=aspect,
                neg_ratio=neg_ratio,
                max_papers_per_aspect=max_papers_per_aspect
            )
            test_aspect_pairs, test_neg_aspect_pairs = get_aspect_pairs(
                paper_id2paper,
                test_paper_ids,
                aspect=aspect,
                neg_ratio=neg_ratio,
                max_papers_per_aspect=max_papers_per_aspect
            )

            assert_leakage(train_aspect_pairs, train_neg_aspect_pairs, test_aspect_pairs, test_neg_aspect_pairs)

            train_samples = [(a, b, pos_target) for a, b in train_aspect_pairs] + [(a, b, neg_target) for a, b in train_neg_aspect_pairs]
            test_samples = [(a, b, pos_target) for a, b in test_aspect_pairs] + [(a, b, neg_target) for a, b in test_neg_aspect_pairs]

            # Output
            fold_output_dir = os.path.join(output_dir, 'folds', aspect, str(k))
            if not os.path.exists(fold_output_dir):
                os.makedirs(fold_output_dir)

            # train.csv
            train_df = pd.DataFrame(train_samples, columns=[doc_a_col, doc_b_col, label_col])
            train_df.to_csv(os.path.join(fold_output_dir, 'train.csv'), index=False)

            # test.csv
            test_df = pd.DataFrame(test_samples, columns=[doc_a_col, doc_b_col, label_col])
            test_df.to_csv(os.path.join(fold_output_dir, 'test.csv'), index=False)

            logger.info(f'Output files saved to: {fold_output_dir}')

            # Sample
            if k == 1:
                sample_output_dir = os.path.join(output_dir, 'folds', aspect, 'sample')
                if not os.path.exists(sample_output_dir):
                    os.makedirs(sample_output_dir)

                train_df.sample(n=sample_size).to_csv(os.path.join(sample_output_dir, 'train.csv'), index=False)
                test_df.sample(n=sample_size).to_csv(os.path.join(sample_output_dir, 'test.csv'), index=False)

                logger.info(f'Sample output saved to: {sample_output_dir}')

    # Save docs
    logger.info('Writing docs')

    with open(os.path.join(output_dir, 'docs.jsonl'), 'w') as f:
        for paper_id, p in paper_id2paper.items():
            f.write(json.dumps(p) + '\n')
            # f.write(json.dumps({
            #     'paper_id': str(url2idx[paper_id]),  # force as str
            #     'paper_url': paper_id,
            #     'title': p['title'],
            #     'abstract': p['abstract']  # TODO aspect fields + more meta data
            # }) + '\n')

    logger.info('Done')


if __name__ == '__main__':
    fire.Fire()
    sys.exit(0)
