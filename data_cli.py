#!/usr/bin/env python
import json
import logging
import os
import random
import sys
from collections import defaultdict
from typing import Union

import numpy as np
import fasttext
import fire
import gensim
import torch
from datasets import load_dataset
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
from sklearn.feature_extraction.text import CountVectorizer
from smart_open import open
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from experiments import basic_logger_config
from experiments.environment import get_env
from hf_datasets.paperswithcode_aspects import get_train_split, get_test_split
from experiments.utils import get_local_hf_dataset_path
from paperswithcode import Paper

logging.basicConfig(**basic_logger_config)
logger = logging.getLogger(__name__)


def train_fasttext(hf_dataset, output_dir):
    """

    Run with: $ ./data_cli.py train_fasttext paperswithcode_aspects ./output

    :return:
    """

    tokens_fp = os.path.join(output_dir, 'tokens.txt')
    fasttext_bin_fp = os.path.join(output_dir, 'fasttext.bin')
    fasttext_w2v_fp = os.path.join(output_dir, 'fasttext.w2v.txt')

    docs_ds = load_dataset(get_local_hf_dataset_path(hf_dataset),
                           name='docs',
                           cache_dir='./data/nlp_cache',
                           split='docs')

    logger.info(f'Documents loaded: {len(docs_ds):,}')

    # Tokenized text
    doc_delimiter = '\n'
    token_delimiter = ' '
    tokens_count = 0

    with open(tokens_fp, 'w') as f:
        for doc in docs_ds:
            # Extract plain text
            text = doc['title'] + ': ' + doc['abstract']

            for token in gensim.utils.simple_preprocess(text, min_len=2, max_len=15):
                f.write(token + token_delimiter)
                tokens_count += 1
            f.write(doc_delimiter)

    logger.info(f'Total tokens: {tokens_count:,}')

    # Train actual fasttext model
    logger.info(f'Train fastext model...')

    model = fasttext.train_unsupervised(
        tokens_fp,
        model='skipgram',
        lr=0.05,  # learning rate [0.05]
        dim=300,  # size of word vectors [100]
        ws=5,  # size of the context window [5]
        epoch=5  # number of epochs [5]
        # thread            # number of threads [number of cpus]
    )
    model.save_model(fasttext_bin_fp)

    del model

    ft_model = FastText.load_fasttext_format(fasttext_bin_fp)
    ft_model.wv.save_word2vec_format(fasttext_w2v_fp)

    logger.info(f'Output saved to: {fasttext_w2v_fp}')

    logger.info('Done')


def build_avg_word_vectors(hf_dataset, w2v_path, output_path, override=False):
    """

    Run with: $ ./data_cli.py build_avg_word_vectors paperswithcode_aspects ./output/fasttext.w2v.txt ./output/pwc_doc_id2avg_fasttext.w2v.txt

    :param hf_dataset:
    :param w2v_path:
    :param output_path:
    :param override:
    :return:
    """
    stop_words = 'english'
    count_vector_size = 100000

    if os.path.exists(output_path):
        if override:
            logger.debug(f'Override {output_path}')
            os.remove(output_path)
        else:
            logger.info(f'Stop. Output file exists already (override disabled): {output_path}')
            return

    w2v_model = KeyedVectors.load_word2vec_format(w2v_path)
    doc_model = KeyedVectors(vector_size=w2v_model.vector_size)

    count_vec = CountVectorizer(stop_words=stop_words, analyzer='word', lowercase=True,
                                ngram_range=(1, 1), max_features=count_vector_size)

    docs_ds = load_dataset(get_local_hf_dataset_path(hf_dataset),
                           name='docs',
                           cache_dir='./data/nlp_cache',
                           split='docs')
    logger.info(f'Documents loaded: {len(docs_ds):,}')

    # Extract plain text
    texts = []
    doc_id2idx = {}
    idx2doc_id = {}

    for idx, doc in enumerate(docs_ds):
        # Extract plain text
        texts.append(doc['title'] + ': ' + doc['abstract'])
        doc_id2idx[doc['paper_id']] = idx
        idx2doc_id[idx] = doc['paper_id']

    # Transforms the data into a bag of words
    count_train = count_vec.fit(texts)
    idx2bow = count_vec.transform(texts)
    vidx2word = {v: k for k, v in count_train.vocabulary_.items()}

    assert len(vidx2word) == len(count_train.vocabulary_)

    logger.info(f'Vocab size: {len(count_train.vocabulary_)}')

    for idx, text in enumerate(tqdm(texts, total=len(texts), desc='Converting docs to vectors')):
        bow = idx2bow[idx].A[0]

        vectors = []
        weights = []

        for _idx, count in enumerate(bow):
            if count > 0:
                word = vidx2word[_idx]
                try:
                    v = w2v_model.get_vector(word)
                    vectors.append(v)
                    weights.append(count)
                except KeyError:
                    # unknown word
                    pass

                pass

        # Check if at least one document term exists as word vector
        if vectors and weights:
            # Weight avg
            doc = np.average(np.array(vectors), axis=0, weights=np.array(weights))

            # Add to model with doc_id
            doc_model.add([str(idx2doc_id[idx])], [doc])
        else:
            logger.debug(f'Cannot add document {idx2doc_id[idx]} due to missing word vectors')

    # Save to disk
    doc_model.save_word2vec_format(output_path)
    logger.info(f'Saved to: {output_path}')


def build_explirefit_inputs(hf_dataset, aspect, output_dir):
    """

    Run with: $ ./data_cli.py build_explirefit_inputs paperswithcode_aspects task ./output/

    format synonyms.txt:
    <doc_id_a> <doc_id_b>
    <doc_id_a> <doc_id_c>
    ...

    antonyms.txt
    <doc_id_a> <doc_id_b>
    <doc_id_a> <doc_id_c>
    ...

    :param aspect:
    :param hf_dataset:
    :param output_dir:
    :return:
    """

    for fold in [1, 2, 3, 4]:
        fold = str(fold)

        fold_dir = os.path.join(output_dir, fold)

        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)

        synonyms_fp = os.path.join(fold_dir, 'synonyms.txt')
        if os.path.exists(synonyms_fp):
            raise FileExistsError(f'Output exists already: {synonyms_fp}')

        antonyms_fp = os.path.join(fold_dir, 'antonyms.txt')
        if os.path.exists(antonyms_fp):
            raise FileExistsError(f'Output exists already: {antonyms_fp}')

        train_ds = load_dataset(
            get_local_hf_dataset_path(hf_dataset),
            name='relations',
            cache_dir='./data/nlp_cache',
            split=get_train_split(aspect, fold)
        )

        logger.info(f'Training samples: {len(train_ds):,}')

        with open(synonyms_fp, 'w') as synonyms_f:
            with open(antonyms_fp, 'w') as antonyms_f:

                for row in tqdm(train_ds, desc='Writing output'):
                    line = 'en_' + row['from_paper_id'] + ' en_' + row['to_paper_id'] + '\n'

                    if row['label'] == 'y':
                        synonyms_f.write(line)
                    elif row['label'] == 'n':
                        antonyms_f.write(line)
                    else:
                        raise ValueError(f'Unsupported label: {row}')

    logger.info('Done')


def build_transformers_vectors(hf_dataset: str,
                               model_name_or_path: str,
                               output_path: str,
                               pooling: str,
                               batch_size: int = 16,
                               override: bool = False):
    """

    $ ./data_cli.py build_transformers_vectors paperswithcode_aspects scibert-scivocab-uncased ./output/scibert-cls --pooling=cls --batch_size=16

    :param hf_dataset:
    :param model_name_or_path:
    :param output_path:
    :param pooling:
    :param override:
    :return:
    """

    env = get_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pooling_strategies = ['cls', 'mean']

    if os.path.exists(output_path) and not override:
        logger.error(f'Output file exists already: {output_path}')
        sys.exit(1)

    if pooling not in pooling_strategies:
        raise ValueError(f'Invalid pooling: {pooling}')

    # Model path from env
    if not os.path.exists(model_name_or_path) and os.path.exists(
            os.path.join(env['bert_dir'], model_name_or_path)):
        model_name_or_path = os.path.join(env['bert_dir'], model_name_or_path)

    # Dataset
    docs_ds = load_dataset(get_local_hf_dataset_path(hf_dataset),
                           name='docs',
                           cache_dir='./data/nlp_cache',
                           split='docs')
    logger.info(f'Documents loaded: {len(docs_ds):,}')

    # Model
    model = AutoModel.from_pretrained(model_name_or_path)
    model = model.to(device)

    # Tokenize docs
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    texts = [doc['title'] + ': ' + doc['abstract'] for doc in docs_ds]

    inputs = tokenizer(texts,
                       add_special_tokens=True,
                       return_tensors='pt',
                       padding=True,
                       max_length=model.config.max_position_embeddings,
                       truncation=True,
                       return_token_type_ids=False,
                       return_attention_mask=True)

    ds = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    dl = DataLoader(ds, shuffle=False, batch_size=batch_size)

    # Vectors
    doc_model = KeyedVectors(vector_size=model.config.hidden_size)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dl, desc='Inference')):
            batch_data = tuple(t.to(device) for t in batch_data)

            outputs = model(*batch_data, return_dict=True)

            if pooling == 'cls':
                batch_embeddings = outputs['pooler_output'].detach().cpu().numpy()
            elif pooling == 'mean':
                batch_embeddings = np.mean(outputs['last_hidden_state'].detach().cpu().numpy(), axis=1)
            else:
                raise NotImplementedError()

            batch_ids = docs_ds[batch_idx * batch_size:batch_idx * batch_size + batch_size]['paper_id']
            doc_model.add(batch_ids, batch_embeddings)

    # Save to disk
    doc_model.save_word2vec_format(output_path)

    logger.info('Done')


def build_specter_vectors(hf_dataset: str,
                          specter_path: str,
                          output_path: str,
                          cuda_device: int = -1,
                          batch_size: int = 32,
                          vector_size: int = 768,
                          override=False):
    """
    Run with: $ ./data_cli.py build_specter_vectors paperswithcode_aspects ./specter_archive ./output/pwc_doc_id2specter.w2v.txt --cuda_device=5

    Download specter:
    $ wget https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/specter/archive.tar.gz
    $ tar -xzvf archive.tar.gz

    :param vector_size:
    :param output_path: ./output
    :param override:
    :param cuda_device:
    :param batch_size:
    :param hf_dataset:
    :param specter_path: Path to specter
    :return:
    """
    from specter.predict_command import predictor_from_archive
    from allennlp.models import load_archive

    # load to register
    from specter.model import Model
    from specter.data import DataReader, DataReaderFromPickled
    from specter.predictor import SpecterPredictor

    if Model and DataReader and SpecterPredictor:
        pass

    if os.path.exists(output_path) and not override:
        logger.error(f'Output file exists already: {output_path}')
        return

    # Dataset
    docs_ds = load_dataset(get_local_hf_dataset_path(hf_dataset),
                           name='docs',
                           cache_dir='./data/nlp_cache',
                           split='docs')
    logger.info(f'Documents loaded: {len(docs_ds):,}')
    papers_to_embed = [doc for doc in docs_ds]

    # Specter settings
    archive_path = os.path.join(specter_path, 'model.tar.gz')
    metadata_path = os.path.join(specter_path, 'metadata_sample.json')
    included_text_fields = 'abstract title'
    vocab_dir = os.path.join(specter_path, 'data/vocab/')

    cuda_device = int(cuda_device)

    overrides = f"{{'model':{{'predict_mode':'true','include_venue':'false'}},'dataset_reader':{{'type':'specter_data_reader','predict_mode':'true','paper_features_path':'{metadata_path}','included_text_fields': '{included_text_fields}'}},'vocabulary':{{'directory_path':'{vocab_dir}'}}}}"
    
    logger.info(f'SPECTER overrides: {overrides}')
    
    archive = load_archive(archive_path, cuda_device=cuda_device, overrides=overrides)
 
    predictor = predictor_from_archive(archive, predictor_name='specter_predictor', paper_features_path=metadata_path)

    # Batches
    def chunks(lst, chunk_size):
        """Splits a longer list to respect batch size"""
        for i in range(0, len(lst), chunk_size):
            yield lst[i: i + chunk_size]

    batches_count = int(len(papers_to_embed) / batch_size)
    batch_embed_papers = []

    # 30min on GPU
    for batch in tqdm(chunks(papers_to_embed, batch_size), total=batches_count):
        batch_out = predictor.predict_batch_json(batch)
        batch_embed_papers += batch_out

    # To keyed vectors
    doc_model = KeyedVectors(vector_size=vector_size)

    for embed_paper in tqdm(batch_embed_papers):
        doc_model.add([embed_paper['paper_id']], [embed_paper['embedding']])

    # Save to disk
    doc_model.save_word2vec_format(output_path)

    logger.info('Done')


def build_specter_input(hf_dataset: str, aspect, fold: Union[str, int], output_path: str, override: bool = False) -> None:
    """
    Run with: $ ./data_cli.py build_specter_input paperswithcode_aspects task 1 ./output/specter_input/1

    Builds the following files (needed for SPECTER training):
    - data.json containing the document ids and their relationship.
    - metadata.json containing mapping of document ids to textual fiels (e.g., title, abstract)
    - train.txt,val.txt, test.txt containing document ids corresponding to train/val/test sets (one doc id per line).

    Data structure:
    - count = 5 (same aspect)
    - count = 1 (???) => ignore

    :param aspect:
    :param hf_dataset:
    :param fold:
    :param output_path:
    :param override:
    :return:
    """
    nlp_cache_dir = './data/nlp_cache'

    if os.path.exists(output_path) and not override:
        logger.error(f'Output path exist already: {output_path}')
        sys.exit(1)
    else:
        os.makedirs(output_path)

    train_ds = load_dataset(get_local_hf_dataset_path(hf_dataset),
                            name='relations',
                            cache_dir=nlp_cache_dir,
                            split=get_train_split(aspect, fold))

    test_ds = load_dataset(get_local_hf_dataset_path(hf_dataset),
                           name='relations',
                           cache_dir=nlp_cache_dir,
                           split=get_test_split(aspect, fold))

    docs_ds = load_dataset(get_local_hf_dataset_path(hf_dataset),
                           name='docs',
                           cache_dir=nlp_cache_dir,
                           split='docs')

    # metadata
    metadata = {}
    for doc in docs_ds:
        metadata[doc['paper_id']] = {
            'paper_id': doc['paper_id'],
            'title': doc['title'],
            'abstract': doc['abstract'],
        }
    logger.info('Writing metadata')
    json.dump(metadata, open(os.path.join(output_path, 'metadata.json'), 'w'))

    # train/val/test ids
    train_doc_ids = set()
    test_doc_ids = set()

    # data
    data = defaultdict(dict)
    for pair in train_ds:
        # TODO include negative samples?
        count = 5 if pair['label'] == 'y' else 1
        data[pair['from_paper_id']][pair['to_paper_id']] = {'count': count}

        train_doc_ids.add(pair['from_paper_id'])
        train_doc_ids.add(pair['to_paper_id'])

    for pair in test_ds:
        count = 5 if pair['label'] == 'y' else 1
        data[pair['from_paper_id']][pair['to_paper_id']] = {'count': count}

        test_doc_ids.add(pair['from_paper_id'])
        test_doc_ids.add(pair['to_paper_id'])

    logger.info('Writing data')
    json.dump(data, open(os.path.join(output_path, 'data.json'), 'w'))

    train_doc_ids = list(train_doc_ids)
    full_test_doc_ids = list(test_doc_ids)
    random.shuffle(full_test_doc_ids)

    split_at = int(0.1 * len(full_test_doc_ids))

    val_doc_ids = full_test_doc_ids[:split_at]
    test_doc_ids = full_test_doc_ids[split_at:]

    logger.info('Writing train/val/test')
    with open(os.path.join(output_path, 'train.txt'), 'w') as f:
        for i in train_doc_ids:
            f.write(i + '\n')
    with open(os.path.join(output_path, 'val.txt'), 'w') as f:
        for i in val_doc_ids:
            f.write(i + '\n')
    with open(os.path.join(output_path, 'test.txt'), 'w') as f:
        for i in test_doc_ids:
            f.write(i + '\n')

    logger.info('done')


def build_whoosh_index(index_dir: str, override=False):
    # use search index
    from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED
    from whoosh.analysis import StemmingAnalyzer
    from whoosh import index
    from whoosh.qparser import QueryParser

    hf_dataset = 'paperswithcode_aspects'
    nlp_cache_dir = './data/nlp_cache'

    if os.path.exists(index_dir):
        if override:
            os.rmdir(index_dir)
        else:
            logger.error(f'Index dir exists already and override is disabled')
            return

    # Load meta data
    docs_ds = load_dataset(get_local_hf_dataset_path(hf_dataset),
                           name='docs',
                           cache_dir=nlp_cache_dir,
                           split='docs')

    paper_id2paper = {p['paper_id']: Paper(**p) for p in docs_ds}

    paper_schema = Schema(
        paper_id=ID(stored=True),
        title=TEXT(stored=True),
        abstract=TEXT(analyzer=StemmingAnalyzer()),
        paper_url=TEXT(),
        aspect_tasks=KEYWORD,
        aspect_methods=KEYWORD,
        aspect_datasets=KEYWORD,
    )

    # reset index via: $ #!rm -r ./output/pwc/whoosh_index
    logger.info('Creating new index')
    # index does not exist
    os.makedirs(index_dir)

    ix = index.create_in(index_dir, paper_schema)

    # save documents
    writer = ix.writer()

    for paper_id, paper in tqdm(paper_id2paper.items()):
        writer.add_document(**paper.__dict__)
        # break
    writer.commit()

    logger.info('Done')


if __name__ == '__main__':
    fire.Fire()
    sys.exit(0)
