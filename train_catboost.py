import argparse
import configparser
import logging
import pathlib
from importlib import reload

import numpy as np

# for multilabel classification
from catboost import CatBoostClassifier, Pool
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
                    filename="logs/catboost_training_midas.log",
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
    with open(config['PATH']['train_midas'], 'rb') as f:
        X_train = np.load(f)
        y_midas_train = np.load(f)
        y_entity_train = np.load(f)

    LOGGER.info(f'Vector dims: X_train:{X_train.shape}, y_midas:{y_midas_train.shape}, y_entity:{y_entity_train.shape}')

    with open(config['PATH']['val_midas'], 'rb') as f:
        X_val = np.load(f)
        y_midas_val = np.load(f)
        y_entity_val = np.load(f)

    LOGGER.info(f'Vector dims: X_val:{X_val.shape}, y_midas:{y_midas_val.shape}, y_entity:{y_entity_val.shape}')

    # y_midas_val[y_midas_val == 0] = 8

    X_midas_train = Pool(np.float32(X_train), label=y_midas_train)
    X_midas_val = Pool(np.float32(X_val), label=y_midas_val)

    X_entity_train = Pool(np.float32(X_train), label=y_entity_train)
    X_entity_val = Pool(np.float32(X_val), label=y_entity_val)

    LOGGER.info("====================")

    midas_params = {
        'verbose': True,
        'random_seed': 42,
        'use_best_model': True,
        'devices':'0:1',
        'loss_function':'MultiClass',
        'eval_metric':'Accuracy',
        'task_type':'CPU'}

    fit_params = {
        'use_best_model': True,
        'early_stopping_rounds': 5
        }

    midas_classifiers = {
        'MidasSymmetricTree': CatBoostClassifier(grow_policy='SymmetricTree',
                                            **midas_params),
        'MidasDepthwise': CatBoostClassifier(grow_policy='Depthwise',
                                            **midas_params),
        'MidasLossguide': CatBoostClassifier(grow_policy='Lossguide',
                                            **midas_params),
        }

    LOGGER.info('Start training Midas classifiers')

    y_midas_val = [ID2Midas[label] for label in y_midas_val]

    for name, clf in midas_classifiers.items():
        LOGGER.info(f'Training {name} classifier')
        clf.fit(X_midas_train, eval_set=X_midas_val, **fit_params)
        midas_preds = clf.predict(X_midas_val).squeeze()
        midas_preds = [ID2Midas[label] for label in midas_preds]
        LOGGER.info("====================")
        LOGGER.info(f'Scores for {name} classifier')
        LOGGER.info(f'\n{classification_report(y_midas_val, midas_preds)}')
        LOGGER.info(f'Saving {name} classifier')
        LOGGER.info("====================")
        clf.save_model(f'models/{name}')

    # LOGGER.info("====================")
    # LOGGER.info("====================")
    # LOGGER.info("====================")
    # LOGGER.info("====================")
    # LOGGER.info('Start training Entity classifiers')
    #
    # entity_params = {
    #     'verbose': True,
    #     'random_seed': 42,
    #     'use_best_model': True,
    #     'devices':'0:1',
    #     'loss_function':'MultiLogloss',
    #     'eval_metric':'Accuracy',
    #     'task_type': 'CPU'
    # }
    #
    # entity_classifiers = {
    #     'EntitySymmetricTree': CatBoostClassifier(grow_policy='SymmetricTree',
    #                                               **entity_params),
    #     'EntityDepthwise': CatBoostClassifier(grow_policy='Depthwise',
    #                                           **entity_params),
    #     'EntityLossguide': CatBoostClassifier(grow_policy='Lossguide',
    #                                           **entity_params),
    #     }
    #
    #
    # LOGGER.info('Start training Entity classifiers')
    # for name, clf in entity_classifiers.items():
    #     LOGGER.info(f'Training {name} classifier')
    #
    #     clf.fit(X_entity_train, eval_set=X_entity_val, **fit_params)
    #
    #     entity_preds = clf.predict(X_entity_val).squeeze()
    #     entity_probas = clf.predict_proba(X_entity_val)
    #     top1 = prediction_by_heuristic(entity_probas, top_n=1)
    #     top3 = prediction_by_heuristic(entity_probas, top_n=3)
    #
    #     LOGGER.info("====================")
    #     LOGGER.info(f'Scores for {name} classifier')
    #     LOGGER.info(f'\n{classification_report(y_entity_val, entity_preds, target_names=ID2Entity)}')
    #
    #     LOGGER.info("====================")
    #     LOGGER.info(f'Scores for {name} classifier using probas: top 1')
    #     LOGGER.info(f'\n{classification_report(y_entity_val, top1, target_names=ID2Entity)}')
    #
    #     LOGGER.info("====================")
    #     LOGGER.info(f'Scores for {name} classifier using probas: top 3')
    #     LOGGER.info(f'\n{classification_report(y_entity_val, top3, target_names=ID2Entity)}')
    #
    #     LOGGER.info(f'Saving {name} classifier')
    #     LOGGER.info("====================")
    #     clf.save_model(f'models/{name}')





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
