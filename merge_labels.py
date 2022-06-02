import argparse
import configparser
import json
import pathlib

import numpy as np
from tqdm import tqdm

from globals import AllMidas2ID, ID2AllMidas as id2label

merge_map = {
    'dev_command': 'command',
    'appreciation': 'comment',
    'complaint': 'comment',
    'other_answers': 'neg_answer'
}

def merge_labels(proba_distr: dict) -> dict:
    """ get 13-label dict and return the one with close labels merged """
    global merge_map, id2label
    probas = np.zeros(len(id2label))

    best_label = max(proba_distr, key=proba_distr.get)

    for label, p in proba_distr.items():
        new_label = merge_map.get(label, label)
        probas[FewerMidas2ID[new_label]] += p

    # probas = softmax(probas)
    return {label: p for label, p in zip(id2label, probas)}


def parse_dataset(dataset: dict):
    """ update midas distributions inplace """
    for dialog in tqdm(dataset.values()):
        for utterance in dialog:
            for i, probas in enumerate(utterance['midas']):
                utterance['midas'][i] = merge_labels(probas)




def main(config: configparser.ConfigParser):

    with open(config['DATASET']['topical'], 'r', encoding="utf8") as f:
        topical = json.load(f)

    with open(config['DATASET']['daily'], 'r', encoding="utf8") as f:
        daily = json.load(f)

    parse_dataset(topical)
    parse_dataset(daily)

    with open('data/datasets/topical_fewer_labels.json', 'w', encoding='utf-8') as f:
        json.dump(topical, f, ensure_ascii=False, indent=2)

    with open('data/datasets/daily_fewer_labels.json', 'w', encoding='utf-8') as f:
        json.dump(daily, f, ensure_ascii=False, indent=2)


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
