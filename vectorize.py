import argparse
import configparser
import json
import logging
import pathlib

from importlib import reload

import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub

from tqdm import tqdm

from utils.vectorization import SampleVectorizer
from utils.raw2dataset import Dial2seq
from utils.preprocessing import SequencePreprocessor
from utils.sequence_validation import HasEntity
from utils.tf_utils import Dataset
from utils.sent_encoder import embed
# from utils.convert_utils import encode_context
from globals import Midas2ID, Entity2ID, EntityLabelEncoder

reload(logging)
log_format = f"%(asctime)s : %(levelname)s : %(message)s"
logging.basicConfig(format=log_format,
                    filename="logs/vectorization.log",
                    filemode="w", level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def encode_labels(labels) -> tuple:
    # split midas and entites
    y_midas, y_entity = list(), list()

    for label in labels:
        y_midas.append(Midas2ID[label[0]])
        y_entity.append(label[1])

    y_entity = EntityLabelEncoder.fit_transform(y_entity)

    return y_midas, y_entity

def stack_batches(dataloader):
    """use when the whole dataset is needed"""
    X, y_midas, y_entity = list(), list(), list()
    for vec, y in tqdm(dataloader):
        midas, entity = encode_labels(y)
        X.append(vec)
        y_midas.append(midas)
        y_entity.append(entity)

    return np.vstack(X), np.hstack(y_midas), np.vstack(y_entity)


def main(config: configparser.ConfigParser):
    preproc = SequencePreprocessor(seq_validator=HasEntity())
    context_len = int(config['VAR']['context_len'])
    LOGGER.info(f'Context length in utterances is {context_len}')

    # data
    LOGGER.info(f'Loading data')

    train = Dial2seq(config['PATH']['train'], context_len).transform()
    train = preproc.transform(train)
    LOGGER.info(f'Total number of sequences in the Train dataset is {len(train)}')
    val = Dial2seq(config['PATH']['val'], context_len).transform()
    val = preproc.transform(val)
    LOGGER.info(f'Total number of sequences in the Validation dataset is {len(val)}')

    vectorizer = SampleVectorizer(
        text_vectorizer=embed, # USE
        entity2id=Entity2ID,
        midas2id=Midas2ID,
        context_len=context_len)

    LOGGER.info("Start Vectorization")

    train = Dataset(data=train, vectorizer=vectorizer,
                    batch_size=256, shuffle=False)

    LOGGER.info(f"Number of batches in train is {len(train)}")

    X, y_midas, y_entity = stack_batches(train)
    LOGGER.info(f'Vector dims: X_train:{X.shape}, y_midas:{y_midas.shape}, y_entity:{y_entity.shape}')

    with open('data/train_val_test/shorter_context/train.npy', 'wb') as f:
        np.save(f, X)
        np.save(f, y_midas)
        np.save(f, y_entity)

    LOGGER.info("Train is vectorized")
    del train

    val = Dataset(data=val, vectorizer=vectorizer,
                    batch_size=256, shuffle=False)

    LOGGER.info(f"Number of batches in val is {len(val)}")
    X, y_midas, y_entity = stack_batches(val)
    LOGGER.info(f'Vector dims: X_val:{X.shape}, y_midas:{y_midas.shape}, y_entity:{y_entity.shape}')

    with open('data/train_val_test/shorter_context/val.npy', 'wb') as f:
        np.save(f, X)
        np.save(f, y_midas)
        np.save(f, y_entity)

    LOGGER.info("Val is vectorized")
    LOGGER.info("Complete")


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
