import argparse
import configparser
import json
import pathlib

import numpy as np
from joblib import load
from tqdm import tqdm

from utils.vectorization import SampleVectorizer
from utils.raw2dataset import Dial2seq
from utils.preprocessing import SequencePreprocessor
from utils.sent_encoder import embed
from utils.inference import Inference
from globals import Midas2ID, Entity2ID, all_labels, EntityTargets2ID


def main(config: configparser.ConfigParser):
    context_len = int(config['VAR']['context_len'])

    # text data
    preproc = SequencePreprocessor()
    test = Dial2seq(config['PATH']['test'], context_len).transform()
    test = preproc.transform(test)

    # response bank
    with open(config['PATH']['responses_text'], 'r', encoding='utf8') as f:
        responses = json.load(f)
    with open(config['PATH']['encoded_responses'], 'rb') as f:
        encoded_responses = np.load(f)
    with open(config['PATH']['resp_ids_by_label'], 'r', encoding='utf8') as f:
        resp_ids_by_label = json.load(f)

    # midas entity intesections
    labels2resp_ids = zip(all_labels, resp_ids_by_label)
    labels2resp_ids = {label: set(ids) for label,ids in labels2resp_ids}

    # models and classifiers
    midas_clf = load(config['PATH']['algo_midas'])
    entity_clf = load(config['PATH']['algo_entity'])

    vectorizer = SampleVectorizer(
        embedder=embed, # USE
        context_len=context_len)

    inference = Inference(
        vectorizer=vectorizer,
        midas_clf=midas_clf,
        entity_clf=entity_clf,
        labels2response_ids=labels2resp_ids,
        encoded_responses=encoded_responses,
        text_responses=responses
    )

    output = list()

    for sample in tqdm(test):
        output.append(inference.infer_utterance(sample))

    output_dir = pathlib.Path(config['PATH']['output_dir'])
    output_file = output_dir.parent / (output_dir.name + '/algo_convert.json')
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
