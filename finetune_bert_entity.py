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

from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

from utils.raw2dataset import Dial2seq
from utils.preprocessing import SequencePreprocessor
from utils.sequence_validation import HasEntity

from utils.bert_utils import (preprocess,
                              CustomDataset,
                              compute_metrics_entity as compute_metrics
)

from globals import EntityTargets2ID as label2id, ID2Entity as id2label
from globals import EntityLabelEncoder


reload(logging)
log_format = f"%(asctime)s : %(levelname)s : %(message)s"
logging.basicConfig(format=log_format,
                    filename="logs/fine_tune_bert_entity.log",
                    filemode="w", level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def main(config: configparser.ConfigParser):

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    # load data
    preproc = SequencePreprocessor(seq_validator=HasEntity())
    context_len = int(config['VAR']['context_len'])
    max_length = int(config['VAR']['max_length'])
    LOGGER.info(f'Context length in utterances is {context_len}')
    LOGGER.info(f'Loading data')

    train = Dial2seq(config['PATH']['train'], context_len).transform()
    train = preproc.transform(train)
    LOGGER.info(f'Total number of sequences in the Train dataset is {len(train)}')
    val = Dial2seq(config['PATH']['val'], context_len).transform()
    val = preproc.transform(val)
    LOGGER.info(f'Total number of sequences in the Validation dataset is {len(val)}')

    X_train, _, y_train = preprocess(train)
    y_train = EntityLabelEncoder.fit_transform(y_train).astype(np.float32)
    X_val, _, y_val = preprocess(val)
    y_val = EntityLabelEncoder.fit_transform(y_val).astype(np.float32)

    del train, val, _

    X_train = tokenizer(X_train, padding="max_length",
                        truncation=True, max_length=max_length)

    X_val = tokenizer(X_val, padding="max_length",
                        truncation=True, max_length=max_length)

    train = CustomDataset(X_train, y_train)
    validation = CustomDataset(X_val, y_val)
    del X_train, X_val, y_train, y_val

    LOGGER.info(f"Number of samples in train is {len(train)}")
    LOGGER.info(f"Number of samples in val is {len(validation)}")

    # load models
    clf = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-cased',
        num_labels=len(label2id),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id)

    args = TrainingArguments(
        "models/entity/checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=int(config['VAR']['n_epochs']),
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        clf,
        args,
        train_dataset=train,
        eval_dataset=validation,
        compute_metrics=compute_metrics,
    )

    LOGGER.info("Initial evaluation")
    trainer.evaluate()
    LOGGER.info("Training model...")
    trainer.train()
    LOGGER.info("Final evaluation")
    trainer.evaluate()

    for obj in trainer.state.log_history:
        LOGGER.info(obj)

    LOGGER.info(f"Saving model to: models/bert/entity")
    trainer.model.save_pretrained('models/bert/entity')


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
