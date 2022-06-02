import argparse
import configparser
import logging
import pathlib
from importlib import reload
from joblib import dump

import numpy as np

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# for multilabel classification
from sklearn.multiclass import OneVsRestClassifier
# metrics
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

from globals import ID2Midas
from globals import EntityLabelEncoder
ID2Entity = EntityLabelEncoder.classes


reload(logging)
log_format = f"%(asctime)s : %(levelname)s : %(message)s"
logging.basicConfig(format=log_format,
                    filename="logs/sklearn_training.log",
                    filemode="w", level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def prediction_by_heuristic(
    probas: np.ndarray, top_n:int=1, num_classes=len(ID2Entity)) -> np.ndarray:
    """ extract top_n predictions from from given probabilities """
    preds = np.argsort(probas, axis=-1)[:,::-1][:,:top_n]
    preds = to_categorical(preds, num_classes=num_classes)

    if top_n > 1:
        preds = np.max(preds, axis=1)

    return preds


def main(config: configparser.ConfigParser):
    LOGGER.info('Loading data')
    with open(config['PATH']['vectorized_train'], 'rb') as f:
        X_train = np.load(f)
        y_midas_train = np.load(f)
        y_entity_train = np.load(f)

    LOGGER.info(f'Vector dims: X_train:{X_train.shape}, y_midas:{y_midas_train.shape}, y_entity:{y_entity_train.shape}')

    with open(config['PATH']['vectorized_val'], 'rb') as f:
        X_val = np.load(f)
        y_midas_val = np.load(f)
        y_entity_val = np.load(f)

    LOGGER.info(f'Vector dims: X_val:{X_val.shape}, y_midas:{y_midas_val.shape}, y_entity:{y_entity_val.shape}')

    LOGGER.info("====================")

    midas_classifiers = {
        'MidasRF': RandomForestClassifier(random_state=42, max_depth=20),
        'MidasLR': LogisticRegression(random_state=42, max_iter=1000)
        }

    entity_classifiers = {
        'EntityRF': OneVsRestClassifier(
            RandomForestClassifier(max_depth=30, random_state=42)),
        'EntityLR': OneVsRestClassifier(
            LogisticRegression(random_state=42, max_iter=1000))
        }

    LOGGER.info('Start training Midas classifiers')

    y_midas_val = [ID2Midas[label] for label in y_midas_val]

    for name, clf in midas_classifiers.items():
        LOGGER.info(f'Training {name} classifier')
        clf.fit(X_train, y_midas_train)
        midas_preds = clf.predict(X_val)
        midas_preds = [ID2Midas[label] for label in midas_preds]
        LOGGER.info("====================")
        LOGGER.info(f'Scores for {name} classifier')
        LOGGER.info(f'\n{classification_report(y_midas_val, midas_preds)}')
        LOGGER.info(f'Saving {name} classifier')
        LOGGER.info("====================")
        dump(clf, f'models/{name}.joblib')

    LOGGER.info("====================")
    LOGGER.info("====================")
    LOGGER.info("====================")
    LOGGER.info("====================")

    LOGGER.info('Start training Entity classifiers')
    for name, clf in entity_classifiers.items():
        LOGGER.info(f'Training {name} classifier')

        clf.fit(X_train, y_entity_train)
        entity_preds = clf.predict(X_val)
        entity_probas = clf.predict_proba(X_val)
        top1 = prediction_by_heuristic(entity_probas, top_n=1)
        top3 = prediction_by_heuristic(entity_probas, top_n=3)

        LOGGER.info("====================")
        LOGGER.info(f'Scores for {name} classifier')
        LOGGER.info(f'\n{classification_report(y_entity_val, entity_preds, target_names=ID2Entity)}')

        LOGGER.info("====================")
        LOGGER.info(f'Scores for {name} classifier using probas: top 1')
        LOGGER.info(f'\n{classification_report(y_entity_val, top1, target_names=ID2Entity)}')

        LOGGER.info("====================")
        LOGGER.info(f'Scores for {name} classifier using probas: top 3')
        LOGGER.info(f'\n{classification_report(y_entity_val, top3, target_names=ID2Entity)}')

        LOGGER.info(f'Saving {name} classifier')
        LOGGER.info("====================")
        dump(clf, f'models/{name}.joblib')





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
