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

from globals import EntityTargets2ID as Entity2ID, ID2Entity, Midas2ID, ID2Midas
from globals import EntityLabelEncoder


reload(logging)
log_format = f"%(asctime)s : %(levelname)s : %(message)s"
logging.basicConfig(format=log_format,
                    filename="logs/evaluate_bert_entity.log",
                    filemode="w", level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class Inference:

    def __init__(self, tokenizer, clf, max_length=256):
        self.tokenizer = tokenizer
        self.clf = clf
        self.max_length=max_length

    def predict_probas(self, context: str):

        encoding = self.tokenizer(context, padding="max_length",
                             truncation=True, max_length=self.max_length,
                             return_tensors="pt")

        with torch.no_grad():
            if torch.cuda.is_available():
                encoding.to("cuda")

            outputs = self.clf(**encoding)
            predictions = torch.sigmoid(outputs.logits)

            if torch.cuda.is_available():
                predictions = predictions.cpu()

            predictions = predictions.numpy()[0]

        return predictions

def prediction_by_heuristic(
    probas: np.ndarray, top_n:int=1, num_classes=len(ID2Entity)) -> np.ndarray:
    """ extract top_n predictions from from given probabilities """
    preds = np.argsort(probas, axis=-1)[:,::-1][:,:top_n]
    preds = to_categorical(preds, num_classes=num_classes)

    if top_n > 1:
        preds = np.max(preds, axis=1)

    return preds


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

    X_val, _, y_entity = preprocess(val)
    # y_midas = [Midas2ID[label] for label in y_midas]
    y_entity = EntityLabelEncoder.fit_transform(y_entity).astype(np.float32)

    del val
    LOGGER.info(f"Number of samples in val is {len(X_val)}")
    # load models
    entity_clf = AutoModelForSequenceClassification.from_pretrained(
        'models/bert/entity',
        problem_type="multi_label_classification",
        )

    if torch.cuda.is_available():
        entity_clf.to("cuda")
        LOGGER.info("Entity model set on cuda")
    entity_clf.eval()
    LOGGER.info("Entity model loaded")

    inference = Inference(tokenizer, entity_clf, max_length)

    y_pred = np.zeros((len(X_val), len(Entity2ID)))
    y_probas = np.zeros((len(X_val), len(Entity2ID)))

    for i, context in enumerate(X_val):
        probas = inference.predict_probas(context)
        preds = np.zeros(probas.shape)
        y_pred[i,:][np.where(probas >= 0.5)] = 1
        y_probas[i,:] = probas



    LOGGER.info(f'\n{classification_report(y_entity, y_pred, target_names=ID2Entity)}\n')
    top1 = prediction_by_heuristic(y_probas, top_n=1)
    LOGGER.info(f'\n{classification_report(y_entity, top1, target_names=ID2Entity)}\n')
    top3 = prediction_by_heuristic(y_probas, top_n=3)
    LOGGER.info(f'\n{classification_report(y_entity, top3, target_names=ID2Entity)}\n')

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
