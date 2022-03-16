import random
from collections import defaultdict
import logging
from tqdm import tqdm
import base64
import hashlib

logger = logging.getLogger(__name__)


def get_paper_id(publisher_url, chars=10, encoding='utf-8'):
    """
    Paper ID is the first 10 chars of the sha1 hash of the publisher url (url_abs - usually link to arxiv)

    For example: http://arxiv.org/abs/1707.05589v2 => nCrJQdu1BQ

    :param publisher_url:
    :param chars:
    :param encoding:
    :return:
    """
    hasher = hashlib.sha1(publisher_url.encode(encoding))
    return base64.urlsafe_b64encode(hasher.digest()).decode(encoding).rstrip('=')[:chars]


def get_aspect_pairs(paper_id2paper, subset_ids, aspect, neg_ratio=0.5, max_papers_per_aspect=1000):
    paper_id2aspects = defaultdict(list)
    aspect2paper_ids = defaultdict(set)

    # Build aspect mappings for each paper
    for paper_id in subset_ids:
        paper = paper_id2paper[paper_id]

        if aspect == 'task':
            paper_id2aspects[paper_id] += paper['aspect_tasks']
            for a in paper['aspect_tasks']:
                aspect2paper_ids[a].add(paper_id)
        elif aspect == 'method':
            paper_id2aspects[paper_id] += paper['aspect_methods']
            for a in paper['aspect_methods']:
                aspect2paper_ids[a].add(paper_id)
        elif aspect == 'dataset':
            paper_id2aspects[paper_id] += paper['aspect_datasets']
            for a in paper['aspect_datasets']:
                aspect2paper_ids[a].add(paper_id)

            # if 'methods' in paper:
            #     for method in paper['methods']:
            #         method = method['name']  # TODO is name unique?
            #         url2aspects[paper_id].append(method)
            #         aspect2urls[method].add(paper_id)
            # else:
            #     logger.warning(f'Method property is not defined for: {paper_id}')

        else:
            raise ValueError(f'Unsupported aspect: {aspect}')

    logger.info(f'Papers with `{aspect}` labels: {len(paper_id2aspects):,}')
    logger.info(f'Different `{aspect}`: {len(aspect2paper_ids):,}')

    # Filter too common tasks
    # task_counts = {k: len(v) for k, v in task2urls.items()}
    unfiltered_aspects_count = len(aspect2paper_ids)

    if max_papers_per_aspect > -1:
        aspect2paper_ids = {k: v for k, v in aspect2paper_ids.items() if len(v) < max_papers_per_aspect}

    logger.info(f'Filtered aspects: {len(aspect2paper_ids):,} (before: {unfiltered_aspects_count:,})')

    # Find papers with same aspect
    aspect_pairs = set()  # pair idx = sorted(seed_idx, target_idx)

    for task, paper_ids in tqdm(aspect2paper_ids.items(), desc='Finding papers with same aspect', total=len(aspect2paper_ids)):
        for a in paper_ids:
            for b in paper_ids:  # n(tasks)*n(papers)*(n(papers)-1)
                if a != b:
                    pair_id = tuple(sorted([a, b]))
                    aspect_pairs.add(pair_id)

    logger.info(f'Pairs with same aspect: {len(aspect_pairs):,}')

    # Negative aspect
    neg_aspect_needed = int(neg_ratio * len(aspect_pairs))

    neg_aspect_pairs = set()
    tries = 0

    while len(neg_aspect_pairs) < neg_aspect_needed:
        a = random.choice(subset_ids)
        b = random.choice(subset_ids)

        if a == b:
            tries += 1
            continue

        if len(paper_id2aspects[a]) > 0 and len(paper_id2aspects[b]) > 0:
            tries += 1
            continue

        pair_idx = tuple(sorted([a, b]))

        if pair_idx in aspect_pairs:
            tries += 1
            continue

        neg_aspect_pairs.add(pair_idx)

    logger.info(f'Negative pairs: {len(neg_aspect_pairs):,} (generated after {tries:,} tries)')

    return aspect_pairs, neg_aspect_pairs


def assert_leakage(train_pairs, train_neg_pairs, test_pairs, test_neg_pairs):
    # Check on leakage from train to test data
    train_paper_ids = set(
        [a for a, b in train_pairs] + [b for a, b in train_pairs] + [a for a, b in train_neg_pairs] + [b
                                                                                                                      for
                                                                                                                      a, b
                                                                                                                      in
                                                                                                                      train_neg_pairs])
    test_paper_ids = set(
        [a for a, b in test_pairs] + [b for a, b in test_pairs] + [a for a, b in test_neg_pairs] + [b for
                                                                                                                   a, b
                                                                                                                   in
                                                                                                                   test_neg_pairs])

    leaking_papers = train_paper_ids & test_paper_ids

    assert len(leaking_papers) == 0
