import argparse
import configparser
import json
import pathlib
import logging

from importlib import reload
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm

reload(logging)
log_format = f"%(message)s"
logging.basicConfig(format=log_format,
                    filename="logs/convert_accuracy.log",
                    filemode="w", level=logging.INFO)
LOGGER = logging.getLogger(__name__)

import numpy as np


def get_randomized_top(arr, correct, train_ids, top_n=100):

    mask = np.zeros((arr.shape[0], top_n), dtype=np.intp)

    for i in tqdm(range(arr.shape[0])):
        # ids = arr.shape[1]
        # ids[]
        mask[i,:] = np.random.choice(train_ids, top_n, replace=False)

    mask[:,-1] = correct

    rows = np.array(range(arr.shape[0]), dtype=np.intp)

    return arr[rows[:, np.newaxis], mask], mask

def get_preds(probas, ids):

    pos_best = np.argmax(probas, axis=-1)
    rows = np.array(range(ids.shape[0]), dtype=np.intp)
    columns = np.array(pos_best, dtype=np.intp)[:, None]
    preds = ids[rows[:, np.newaxis], columns]

    return preds.squeeze()


def main(config: configparser.ConfigParser):

    with open(config['CONVERT']['text_context'], 'r', encoding="utf8") as f:
        text_context = json.load(f)

    with open(config['CONVERT']['responses'], 'r', encoding="utf8") as f:
        responses = json.load(f)

    with open(config['CONVERT']['encoded_responses'], 'rb') as f:
        encoded_responses = np.load(f)

    with open(config['CONVERT']['val_responses'], 'r', encoding="utf8") as f:
        val_responses = json.load(f)

    with open(config['CONVERT']['y_true'], 'r', encoding="utf8") as f:
        y_true = json.load(f)

    with open(config['CONVERT']['encoded_context'], 'rb') as f:
        vectorized_context = list()
        for i in range(len(text_context)):
            vectorized_context.append(np.load(f))

    LOGGER.info(f'Bank of responses: {encoded_responses.shape}')
    LOGGER.info(f'number of sample: {len(val_responses)}')

    vectorized_context = np.vstack(vectorized_context)

    assert len(val_responses) == len(y_true)
    assert len(val_responses) == len(text_context)
    assert vectorized_context.shape[0] == len(y_true)

    val_ids = np.array(y_true)
    train_ids = np.setxor1d(np.arange(encoded_responses.shape[0]),val_ids)

    scores = np.zeros([vectorized_context.shape[0], len(responses)], dtype='f')

    for i in tqdm(range(vectorized_context.shape[0])):
        scores[i,:] = vectorized_context[i].dot(encoded_responses.T)
        # scores[i,:] = cosine_similarity(vectorized_context[None, i],
                                        # encoded_responses)

    random100_scores, random100_ids = get_randomized_top(scores, y_true, train_ids)

    random100_preds = get_preds(random100_scores, random100_ids)

    LOGGER.info(f'Accuracy: {accuracy_score(y_true, random100_preds)}')

    for i in range(len(text_context)):
        context = "\n".join([" ".join(ut) for ut in text_context[i]])
        LOGGER.info(context)
        LOGGER.info("\n")
        answer_id = random100_preds[i]
        LOGGER.info(f"Target: {responses[y_true[i]]}")
        LOGGER.info(f"Preds: {responses[answer_id]}")
        LOGGER.info('\n\n\n')


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
