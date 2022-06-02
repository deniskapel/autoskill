import argparse
import configparser
import json
import pathlib

import numpy as np
from joblib import load
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

from utils.raw2dataset import Dial2seq
from utils.preprocessing import SequencePreprocessor
from utils.inference_unmasking import Inference
from globals import all_labels_fewer as all_labels


def main(config: configparser.ConfigParser):

    context_len = int(config['VAR']['context_len'])

    tokenizer = AutoTokenizer.from_pretrained(config['URL']['bert'])
    print("Tokenizer loaded")
    unmasker = pipeline('fill-mask', model='roberta-base')
    print("Unmasker loaded")

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
        
    labels2ids = {label: set(ids) for label, ids in zip(all_labels, resp_ids_by_label)}
    # models and classifiers
    midas_clf = AutoModelForSequenceClassification.from_pretrained(
        config['PATH']['bert_midas'])

    entity_clf = AutoModelForSequenceClassification.from_pretrained(
        config['PATH']['bert_entity'],
        problem_type="multi_label_classification")


    inference = Inference(
        tokenizer=tokenizer,
        midas_clf=midas_clf, entity_clf=entity_clf,
        unmasker=unmasker,
        max_length = int(config['VAR']['max_length']),
        labels2ids=labels2ids,
        encoded_responses=encoded_responses,
        responses=responses,
        n_best_midas=1, n_best_entity=1)

    output = list()

    for sample in tqdm(test[0:10]):
        output.append(inference.infer_utterance(sample))

    print(output)

    # output_dir = pathlib.Path(config['PATH']['output_dir'])
    # output_file = output_dir.parent / (output_dir.name + '/roberta_convert_unmasker_31.json')
    # with output_file.open("w", encoding="UTF-8") as f:
    #     json.dump(output, f, ensure_ascii=False, indent=2)





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
