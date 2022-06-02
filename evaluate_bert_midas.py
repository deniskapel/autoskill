import argparse
import configparser
import json
import logging
import pathlib

import argparse
import configparser
import json
import logging
import pathlib

from importlib import reload

import numpy as np
import torch
import torch.nn as nn

from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.raw2dataset import Dial2seq
from utils.preprocessing import SequencePreprocessor
from utils.sequence_validation import HasEntity

from utils.bert_utils import (preprocess,
                              CustomDataset,
                              compute_metrics_entity as compute_metrics
)

from globals import Midas2ID, ID2Midas
from globals import EntityLabelEncoder


reload(logging)
log_format = f"%(asctime)s : %(levelname)s : %(message)s"
logging.basicConfig(format=log_format,
                    filename="logs/evaluate_bert_midas.log",
                    filemode="w", level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class Inference:

    def __init__(self, tokenizer, clf, max_length=256):
        self.tokenizer = tokenizer
        self.clf = clf
        self.max_length=max_length

    def predict_probas(self, contexts: str):

        encoding = self.tokenizer(contexts, padding="max_length",
                             truncation=True, max_length=self.max_length,
                             return_tensors="pt")

        with torch.no_grad():
            if torch.cuda.is_available():
                encoding.to("cuda")

            outputs = self.clf(**encoding)
            softmax = torch.nn.Softmax(dim=-1)
            predictions = softmax(outputs.logits).argmax(-1)
            if torch.cuda.is_available():
                predictions = predictions.cpu()

            predictions = predictions.numpy()[0]

        return predictions



def main(config: configparser.ConfigParser):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    LOGGER.info("Tokenizer loaded")

    # load data
    preproc = SequencePreprocessor(seq_validator=HasEntity())
    context_len = int(config['VAR']['context_len'])
    max_length = int(config['VAR']['max_length'])
    LOGGER.info(f'Context length in utterances is {context_len}')
    LOGGER.info(f'Loading data')
    val = Dial2seq(config['PATH']['val'], context_len).transform()
    val = preproc.transform(val)
    LOGGER.info(f'Total number of sequences in the Validation dataset is {len(val)}')

    X_val, y_midas, _ = preprocess(val)
    y_midas = [Midas2ID[label] for label in y_midas]

    del val, _
    LOGGER.info(f"Number of samples in val is {len(X_val)}")
    # load models
    midas_clf = AutoModelForSequenceClassification.from_pretrained(
        'models/bert/midas')

    if torch.cuda.is_available():
        midas_clf.to("cuda")
        LOGGER.info("Midas model set on cuda")

    LOGGER.info(midas_clf.eval())
    LOGGER.info("Midas model loaded")

    inference = Inference(tokenizer, midas_clf, max_length)

    y_pred = np.zeros(len(X_val))

    for i, context in enumerate(X_val):
        y_pred[i] = inference.predict_probas(context)


    LOGGER.info(f'\n{classification_report(y_midas, y_pred, target_names=ID2Midas)}\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        help="Path to a config file",
        required=True)

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    main(config)
