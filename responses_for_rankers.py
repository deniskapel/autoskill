import argparse
import configparser
import logging
import json
import pathlib
from importlib import reload

from utils.raw2dataset import Dial2seq
from utils.sequence_validation import OneEntity, NoEntity
from utils.preprocessing import SequencePreprocessor, normalize_resp

reload(logging)
log_format = f"%(asctime)s : %(levelname)s : %(message)s"
logging.basicConfig(format=log_format,
                    filename="logs/data_for_rankers.log",
                    filemode="w", level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def main(config: configparser.ConfigParser):
    context_len = int(config['VAR']['context_len'])
    preproc_one = SequencePreprocessor(seq_validator=OneEntity())
    preproc_no = SequencePreprocessor(seq_validator=NoEntity())
    preproc_test = SequencePreprocessor()

    LOGGER.info('Loading data')
    # text data
    train = Dial2seq(config['PATH']['train'], context_len).transform()
    train = preproc_one.transform(train) + preproc_no.transform(train)
    # test = Dial2seq(config['PATH']['test'], context_len).transform()
    # test = preproc_one.transform(test) + preproc_no.transform(test)

    # LOGGER.info(f'{len(train)}, {len(test)}')

    responses = list()

    for sample in train:
    # for sample in train + test:
        text = sample['predict']['text']

        if sample['predict']['entities']:
            entity_text = sample['predict']['entities'][0]['text']
            entity_label = sample['predict']['entities'][0]['label']
            text = text.replace(entity_label.upper(), entity_text)

        # text = normalize_resp(text)

        # if not text:
            # continue

        responses.append(text)

    LOGGER.info(f'Total number of responses = {len(responses)}')
    responses = list(set(responses))
    LOGGER.info(f'Number of unique responses = {len(responses)}')

    # write JSON files with text:
    with open('data/rankers_data/responses_for_inference.json', "w", encoding="UTF-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)


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
