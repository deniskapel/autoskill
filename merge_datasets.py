import argparse
import configparser
import logging
import pathlib
from importlib import reload
import json

from sklearn.model_selection import train_test_split

from utils.preprocessing import SequencePreprocessor

reload(logging)
log_format = f"%(asctime)s : %(levelname)s : %(message)s"
logging.basicConfig(format=log_format,
                    filename="logs/merge_datasets.log",
                    filemode="w", level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def main(config: configparser.ConfigParser):

    with open(config['DATASET']['topical'], 'r', encoding="utf8") as f:
        topical = json.load(f)

    with open(config['DATASET']['daily'], 'r', encoding="utf8") as f:
        daily = json.load(f)

    LOGGER.info(f'N dialogs: topical chat: {len(topical)}, daily dialog: {len(daily)}')

    total = list(topical.items()) + list(daily.items())
    LOGGER.info(f'Total number of dialogs - {len(total)}')

    train, val_test = train_test_split(total, test_size=0.2, random_state=42)
    val, test = train_test_split(val_test, test_size=500, random_state=42)
    train, val, test = dict(train), dict(val), dict(test)

    LOGGER.info(
        f'N dialogs: train: {len(train)}, val: {len(val)}, test: {len(test)}')

    with open('data/train_val_test/fewer_labels/train.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False, indent=2)

    with open('data/train_val_test/fewer_labels/val.json', 'w', encoding='utf-8') as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    with open('data/train_val_test/fewer_labels/test.json', 'w', encoding='utf-8') as f:
        json.dump(test, f, ensure_ascii=False, indent=2)


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
