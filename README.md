# Specialized Document Embeddings for Aspect-based Similarity of Research Papers

This repository contains the supplemental materials for the JCDL2022 paper **Specialized Document Embeddings for Aspect-based Similarity of Research Papers** 
[(PDF on ArXiv)](https://arxiv.org/abs/2203.14541).
Trained models and datasets can be downloaded from [GitHub releases](https://github.com/malteos/aspect-document-embeddings/releases) 
and [🤗 Huggingface model hub](https://huggingface.co/malteos/aspect-scibert-task).

## Demo

[Try your own papers on 🤗 Huggingface spaces.](https://huggingface.co/spaces/malteos/aspect-based-paper-similarity)

## How to use the pretrained models

We provide a SciBERT-based model for each of the three aspects: 
🎯 [malteos/aspect-scibert-task](https://huggingface.co/malteos/aspect-scibert-task),
🔨 [malteos/aspect-scibert-method](https://huggingface.co/malteos/aspect-scibert-method),
🏷️ [malteos/aspect-scibert-dataset](https://huggingface.co/malteos/aspect-scibert-dataset).
To use these models, you need to install 🤗 Transformers first via `pip install transformers`.

```python
import torch
from transformers import AutoTokenizer, AutoModel

# load model and tokenizer (replace with `aspect-scibert-method` or `aspect-scibert-dataset)`)
tokenizer = AutoTokenizer.from_pretrained('malteos/aspect-scibert-task')  
model = AutoModel.from_pretrained('malteos/aspect-scibert-task')

papers = [{'title': 'BERT', 'abstract': 'We introduce a new language representation model called BERT'},
          {'title': 'Attention is all you need', 'abstract': ' The dominant sequence transduction models are based on complex recurrent or convolutional neural networks'}]

# concatenate title and abstract
title_abs = [d['title'] + ': ' + (d.get('abstract') or '') for d in papers]

# preprocess the input
inputs = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512)

# inference
output = model(**inputs)

# Mean pool the token-level embeddings to get sentence-level embeddings
embeddings = torch.sum(
    output.last_hidden_state * inputs['attention_mask'].unsqueeze(-1), dim=1
) / torch.clamp(torch.sum(inputs['attention_mask'], dim=1, keepdims=True), min=1e-9)

```

## Requirements

- Python 3.7
- CUDA GPU (for Transformers)

## Installation

Create a new virtual environment for Python 3.7 with Conda:

```bash
conda create -n aspect-document-embeddings python=3.7
conda activate aspect-document-embeddings
```

Clone repository and install dependencies:

```bash
git clone https://github.com/malteos/aspect-document-embeddings
cd aspect-document-embeddings
pip install -r requirements.txt
```

## Datasets

The datasets are compatible with [Huggingface datasets](https://github.com/huggingface/datasets) and are downloaded automatically.
To create the datasets directly from the [Papers With Code data](https://github.com/paperswithcode/paperswithcode-data), run the following commands:

```bash
# Download PWC files (for the paper with downloaded the files 2020-10-27)
wget https://paperswithcode.com/media/about/papers-with-abstracts.json.gz
wget https://paperswithcode.com/media/about/evaluation-tables.json.gz
wget https://paperswithcode.com/media/about/methods.json.gz

# Build dataset
python -m paperswithcode.dataset save_dataset <input_dir> <output_dir> 
```


## Experiments

To reproduce our experiments, follow these steps:

### Generic embeddings

Avg. FastText
```bash
# Train fastText word vectors
./data_cli.py train_fasttext paperswithcode_aspects ./output/pwc

# Build avg. fastText document vectors
./sbin/paperswithcode/avg_fasttext.sh
```
 
SciBERT
```bash
./sbin/paperswithcode/scibert_mean.sh
```

SPECTER
```bash
./sbin/paperswithcode/specter.sh
```

### Retrofitted embeddings

For retrofitting we utilize [Explicit Retroffing](https://github.com/codogogo/explirefit). 
Please follow their instruction to install it and update the `EXPLIREFIT_DIR` in the shell scripts accordingly.
Then, you can run these scripts:

```bash
# Create constraints from dataset 
./sbin/paperswithcode/explirefit_prepare.sh

# Train retrofitting models
./sbin/paperswithcode/explirefit_avg_fasttext.sh
./sbin/paperswithcode/explirefit_specter.sh
./sbin/paperswithcode/explirefit_scibert_mean.sh

# Generate and evaluate retrofitted embeddings 
./sbin/paperswithcode/explirefit_convert_and_evaluate.sh
```


### Transformers

```bash
# SciBERT
./sbin/paperswithcode/pairwise/scibert.sh

# SPECTER
./sbin/paperswithcode/specter_fine_tuned.sh

# Sentence-SciBERT
./sbin/paperswithcode/sentence_transformer_scibert.sh
```



## Evaluation

After generating the document representations for all aspects and systems, the results can be computed and viewed with a Jupyter notebook. 
Figures and tables from the paper are part of the notebook.

```bash
# Run evaluations for all systems
./eval_cli.py reevaluate

# Open notebook for Tables and Figures
jupyter notebook evaluation.ipynb

# Open notebook for sample recommendations
jupyter notebook samples.ipynb
```

## How to cite

If you are using our code or data, please cite [our paper](https://arxiv.org/abs/2203.14541):

```bibtex
@InProceedings{Ostendorff2022,
  title = {Specialized Document Embeddings for Aspect-based Similarity of Research Papers},
  booktitle = {Proceedings of the {ACM}/{IEEE} {Joint} {Conference} on {Digital} {Libraries} ({JCDL})},
  author = {Ostendorff, Malte and Blume, Till, Ruas, Terry and Gipp, Bela and Rehm, Georg},
  year = {2022},
}
```

## License

MIT
