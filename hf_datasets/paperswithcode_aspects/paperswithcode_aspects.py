from __future__ import absolute_import, division, print_function

import json
import os
import sys

import datasets
from pyarrow import csv

_DESCRIPTION = """Papers with aspects from paperswithcode.com dataset"""

_HOMEPAGE = ""

_CITATION = """\
"""

DATA_URL = "http://datasets.fiq.de/paperswithcode_aspects.tar.gz"

DOC_A_COL = "from_paper_id"
DOC_B_COL = "to_paper_id"
LABEL_COL = "label"

# binary classification (y=similar, n=dissimilar)
LABEL_CLASSES = labels = ['y', 'n']

ASPECTS = ['task', 'method', 'dataset']


def get_train_split(aspect, k):
    return datasets.Split(f'fold_{aspect}_{k}_train')


def get_test_split(aspect, k):
    return datasets.Split(f'fold_{aspect}_{k}_test')


class PWCConfig(datasets.BuilderConfig):
    def __init__(self, features, data_url, aspects, **kwargs):
        super().__init__(version=datasets.Version("0.1.0"), **kwargs)
        self.features = features
        self.data_url = data_url
        self.aspects = aspects


class PWCAspects(datasets.GeneratorBasedBuilder):
    """Paper aspects dataset."""

    BUILDER_CONFIGS = [
        PWCConfig(
            name="docs",
            description="document text and meta data",
            # Metadata format from paperswithcode.com
            # see https://github.com/paperswithcode/paperswithcode-data
            features={
                "paper_id": datasets.Value("string"),
                "paper_url": datasets.Value("string"),
                "title": datasets.Value("string"),
                "abstract": datasets.Value("string"),
                "aspect_tasks": datasets.Sequence(datasets.Value('string', id='task')),
                "aspect_methods": datasets.Sequence(datasets.Value('string', id='method')),
                "aspect_datasets": datasets.Sequence(datasets.Value('string', id='dataset')),
            },
            data_url=DATA_URL,
            aspects=ASPECTS,
        ),
        PWCConfig(
            name="relations",
            description=" relation data",
            features={
                DOC_A_COL: datasets.Value("string"),
                DOC_B_COL: datasets.Value("string"),
                LABEL_COL: datasets.Value("string"),
            },
            data_url=DATA_URL,
            aspects=ASPECTS,
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION + self.config.description,
            features=datasets.Features(self.config.features),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        arch_path = dl_manager.download_and_extract(self.config.data_url)

        if "relations" in self.config.name:
            train_file = "train.csv"
            test_file = "test.csv"

            generators = []

            # for k in [1, 2, 3, 4]:
            for aspect in self.config.aspects:
                for k in ["sample"] + [1, 2, 3, 4]:
                    folds_path = os.path.join(arch_path, 'folds', aspect, str(k))
                    generators += [
                        datasets.SplitGenerator(
                            name=get_train_split(aspect, k),
                            gen_kwargs={'filepath': os.path.join(folds_path, train_file)}
                        ),
                        datasets.SplitGenerator(
                            name=get_test_split(aspect, k),
                            gen_kwargs={'filepath': os.path.join(folds_path, test_file)}
                        )
                    ]
            return generators

        elif "docs" in self.config.name:
            # docs
            docs_file = os.path.join(arch_path, "docs.jsonl")

            return [
                datasets.SplitGenerator(name=datasets.Split('docs'), gen_kwargs={"filepath": docs_file}),
            ]
        else:
            raise ValueError()

    @staticmethod
    def get_dict_value(d, key, default=None):
        if key in d:
            return d[key]
        else:
            return default

    def _generate_examples(self, filepath):
        """Generate docs + rel examples."""

        if "relations" in self.config.name:
            df = csv.read_csv(filepath).to_pandas()

            for idx, row in df.iterrows():
                yield idx, {
                    DOC_A_COL: str(row[DOC_A_COL]),
                    DOC_B_COL: str(row[DOC_B_COL]),
                    LABEL_COL: row['label'],  # !!! labels != label
                }

        elif self.config.name == "docs":
            with open(filepath, 'r') as f:
                for i, line in enumerate(f):
                    doc = json.loads(line)
                    # extract feature keys from doc
                    features = {k: doc[k] if k in doc else None for k in self.config.features.keys()}

                    yield i, features
