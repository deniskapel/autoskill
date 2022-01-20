# Thesis for HSE

Python >= 3.8

## Notebooks
1. [Data preprocessing](preprocessing.ipynb) to turn raw dialogues into sequences (n previous utterances + the following utterance)
2. [Training](tfidf_notebook.ipynb) a basic TfIdfVectorizer
3. [Basic experiments](base_experiments.ipynb) with a very simplystic model
4. [Calculating statistics](clean_data.ipynb) for the midas- and cobot-labeled [dialogue dataset](https://github.com/alexa/Topical-Chat)


## Utilities
1. [Torch utilities](utils/autoskill_torch.py): dataset, collator, etc.
2. [Preprocessing functions](utils/preprocessing.py): tokenizer, etc.
3. [Preprocessing 2](utils/data2seq): a class to shape a autoskill dataset from the midas- and cobot-labeled dataset
4. [Basic torch functions and models](utils/base_torch_utils): used to test pipelines

## Data description:
1. [Preprocessed dataset](data/dataset.json): properly shaped dataset (n_seq -> following utterance: text, midas, entity)
2. [Label maps](data/labels.json): a dictionary with all possible labels. There are all label maps (midas and entity) and maps with target labels only (e.g. midas MISC and ANAPHOR are excluded from them as they are not present in the target utterance)

## Models

[This folder](models/) will contain models trained for the project, e.g., sklearn models, torch models, etc.  