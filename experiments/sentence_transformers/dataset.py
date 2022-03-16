import random
import logging

import torch
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DocumentPairSentencesDataset(Dataset):
    doc_a_col = 'from_paper_id'
    doc_b_col = 'to_paper_id'
    doc_id_col = 'paper_id'
    label_classes = ['y', 'n']

    def __init__(self, hf_docs, hf_pairs, st_model: SentenceTransformer, max_length=512, forced_length=0):
        self.hf_docs = hf_docs
        self.doc_id2doc = {doc[self.doc_id_col]: doc for doc in hf_docs}

        self.hf_pairs = hf_pairs
        self.st_model = st_model
        self.doc_id2token_ids = {}
        self.max_length = max_length
        self.forced_length = forced_length

    def __len__(self):
        if self.forced_length > 0:
            return self.forced_length
        else:
            return len(self.hf_pairs)

    def tokenize_all_docs(self):
        # tokenize all docs
        for doc_id in tqdm(self.doc_id2doc, desc='Tokenize'):
            self.get_token_ids_from_doc(doc_id)

        logger.info(f'All documents tokenized')

    def get_text_from_doc(self, doc_id):
        doc = self.doc_id2doc[doc_id]
        return doc['title'] + ': ' + doc['abstract']

    def get_token_ids_from_doc(self, doc_id):
        # use cache: tokenize only once!
        if doc_id not in self.doc_id2token_ids:
            # tokenizer_out = self.st_model.tokenizer(text=self.get_text_from_doc(doc_id),
            #                                   add_special_tokens=False,
            #                                   return_attention_mask=False,
            #                                   return_token_type_ids=False,
            #                                   truncation=True,
            #                                   max_length=self.max_length)
            # self.doc_id2token_ids[doc_id] = tokenizer_out['input_ids']
            self.doc_id2token_ids[doc_id] = self.st_model.tokenize(self.get_text_from_doc(doc_id))[:self.max_length]

        return self.doc_id2token_ids[doc_id]

    def __getitem__(self, item):
        if isinstance(item, int):
            # item is index
            pair = self.hf_pairs[item]
            return self.get_item_by_pair(pair)

        else:
            # item is slice
            raise NotImplementedError('Dataset slicing is not implemented!')
            # return [self.get_item_by_pair(pair) for idx, pair in self.hf_pairs.data.to_pandas()[item].iterrows()]

    def get_item_by_pair(self, pair):
        a_id = pair[self.doc_a_col]
        b_id = pair[self.doc_b_col]

        a_tokens = self.get_token_ids_from_doc(a_id)
        b_tokens = self.get_token_ids_from_doc(b_id)

        label = torch.tensor([self.get_encoded_label(pair['label'])], dtype=torch.float)
        return [a_tokens, b_tokens], label

    def get_encoded_label(self, label: str):
        if label == 'y':
            return 1  # TODO 0 or 1
        else:
            return 0

    def get_evaluator_examples(self, n=0):
        examples = []

        for idx, pair in enumerate(self.hf_pairs):
            a_id = pair[self.doc_a_col]
            b_id = pair[self.doc_b_col]

            examples.append(InputExample(
                guid=';'.join([pair[self.doc_a_col], pair[self.doc_b_col]]),
                texts=[self.get_text_from_doc(a_id), self.get_text_from_doc(b_id)],
                label=float(self.get_encoded_label(pair['label']))
            ))

        # subsample for dev set
        if n > 0:
            logger.info(f'Decreasing example count from {len(examples)} to {n}')
            examples = random.sample(examples, n)

        return examples