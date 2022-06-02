import argparse
import configparser
import json
import pathlib

import numpy as np
from tqdm import tqdm

from utils.raw2dataset import Dial2seq
from utils.preprocessing import SequencePreprocessor
from utils.convert_utils import encode_context


def main(config: configparser.ConfigParser):
    context_len = int(config['VAR']['context_len'])

    # text data
    preproc = SequencePreprocessor()
    test = Dial2seq(config['PATH']['test'], context_len).transform()
    test = preproc.transform(test)

    # response bank
    with open(config['RANKER']['responses'], 'r', encoding='utf8') as f:
        responses = json.load(f)
    with open(config['RANKER']['encoded_responses'], 'rb') as f:
        encoded_responses = np.load(f)

    output = list()

    for sample in tqdm(test):
        context = [" ".join(ut) for ut in sample['previous_text']]
        encoded_context = encode_context(context)
        scores = encoded_context.dot(encoded_responses.T)
        candidate_id = np.argmax(scores)
        context = " __eou__ ".join(context)
        text = responses[candidate_id]
        output.append({'context': context, "prediction": text})


    output_dir = pathlib.Path(config['PATH']['output_dir'])
    output_file = output_dir.parent / (output_dir.name + '/raw_convert.json')
    with output_file.open("w", encoding="UTF-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

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
