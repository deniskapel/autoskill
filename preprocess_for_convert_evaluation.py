import argparse
import configparser
import pathlib
import json

from collections import Counter

from utils.raw2dataset import Dial2seq
from utils.sequence_validation import OneEntity, NoEntity
from utils.preprocessing import SequencePreprocessor, normalize_resp

def main(config: configparser.ConfigParser):

    context_len = int(config['VAR']['context_len'])
    preproc_one = SequencePreprocessor(seq_validator=OneEntity())
    preproc_no = SequencePreprocessor(seq_validator=NoEntity())
    preproc_test = SequencePreprocessor()

    # text data
    train = Dial2seq(config['PATH']['train'], context_len).transform()
    train = preproc_one.transform(train) + preproc_no.transform(train)
    val = Dial2seq(config['PATH']['val'], context_len).transform()
    val = preproc_one.transform(val) + preproc_no.transform(val)

    print(f'Train_size: {len(train)}, VAL_size: {len(val)}')

    train_responses = list()
    for sample in train:
        text = sample['predict']['text']

        if sample['predict']['entities']:
            entity_text = sample['predict']['entities'][0]['text']
            entity_label = sample['predict']['entities'][0]['label']
            text = text.replace(entity_label.upper(), entity_text)

        # text = normalize_resp(text)

        if not text:
            continue

        train_responses.append(text)


    targets = list()
    # midas_cnt = Counter()
    # entity_cnt = Counter()
    context = list()

    for sample in val:

        text = sample['predict']['text']

        if sample['predict']['entities']:
            entity_text = sample['predict']['entities'][0]['text']
            entity = sample['predict']['entities'][0]['label']
            text = text.replace(entity.upper(), entity_text)
            # text = normalize_resp(text)
        else:
            entity = None
        midas = sample['predict']['midas']
        # midas_cnt.update([midas])
        # entity_cnt.update([entity])

        if not text:
            # print(midas, entity, entity_text, text)
            continue

        context.append(sample['previous_text'])
        targets.append((midas, entity, text))

    print('Number of VAL responses (midas + entity + text): ', len(targets))
    print('Number of VAL samples in contexts: ', len(context))
    train_val_resp = list(set(train_responses + [s[-1] for s in targets]))
    print('unique train responses', len(set(train_responses)))
    print('unique train+val responses', len(train_val_resp))

    id2response = [res for res in train_val_resp]
    response2id = {res: i for i, res in enumerate(id2response)}

    y_true = [response2id[res[-1]] for res in targets]

    assert len(y_true) == len(targets)
    assert len(y_true) == len(context)

    with open('data/convert_tests/responses_all.json',"w", encoding="UTF-8") as f:
        json.dump(train_val_resp, f, ensure_ascii=False, indent=2)

    with open('data/convert_tests/val_responses.json',"w", encoding="UTF-8") as f:
        json.dump(targets, f, ensure_ascii=False, indent=2)

    with open('data/convert_tests/y_val.json',"w", encoding="UTF-8") as f:
        json.dump(y_true, f, ensure_ascii=False, indent=2)

    with open('data/convert_tests/val_context.json',"w", encoding="UTF-8") as f:
        json.dump(context, f, ensure_ascii=False, indent=2)






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
