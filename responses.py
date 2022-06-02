import argparse
import pathlib
import json
import logging
from importlib import reload
from collections import Counter

from utils.raw2dataset import Dial2seq
from utils.sequence_validation import OneEntity, NoEntity
from utils.preprocessing import SequencePreprocessor, normalize_resp
from globals import all_labels_fewer as all_labels

reload(logging)
log_format = f"%(message)s"
logging.basicConfig(format=log_format,
                    filename="logs/extract_responses.log",
                    filemode="w", level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def main(input_path, output_dir, context_len):

    # extract all sequence ()
    seqs = Dial2seq(input_path, context_len=context_len).transform()
    LOGGER.info(f'Total number of sequences in this dataset is {len(seqs)}')

    preproc_one = SequencePreprocessor(seq_validator=OneEntity())
    preproc_no = SequencePreprocessor(seq_validator=NoEntity())
    one_entity = preproc_one.transform(seqs)
    no_entity = preproc_no.transform(seqs)

    LOGGER.info(f'Number of responses with one entity: {len(one_entity)}')
    LOGGER.info(f'Number of responses with no entity: {len(no_entity)}')

    targets = list()

    for sample in one_entity + no_entity:
        midas = sample['predict']['midas']
        entity = sample['predict']['entities']
        entity = entity[0]['label'] if entity else None

        text = sample['predict']['text']

        if not text:
            continue

        targets.append((midas, entity, text))

    ids_by_label = {label: list() for label in all_labels}
    targets = set(targets)
    LOGGER.info(f'Number of unique responses (midas + entity + text): {len(targets)}')
    text_only = list(set([target[-1] for target in targets]))
    LOGGER.info(f'Number of unique responses (text only): {len(text_only)}')

    for resp in targets:
        idx = text_only.index(resp[-1])
        # midas
        ids_by_label[resp[0]].append(idx)
        # entity
        ids_by_label[resp[1]].append(idx)

    cnt = {label: len(ids) for label, ids in ids_by_label.items()}
    LOGGER.info(f'Number of answers per label: {cnt}')

    # write JSON files with text:
    text_output = output_dir.parent / (output_dir.name + '/output.json')
    with text_output.open("w", encoding="UTF-8") as f:
        json.dump(text_only, f, ensure_ascii=False, indent=2)

    idx_output = output_dir.parent / (output_dir.name + '/ids_by_label.json')
    # write JSON files with response_id
    with idx_output.open("w", encoding="UTF-8") as f:
        json.dump(list(ids_by_label.values()), f, ensure_ascii=False, indent=2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_file_path",
        type=pathlib.Path,
        help="Path to the json train/val/test file",
        required=True)
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        help="Path to a folder",
        required=True)
    parser.add_argument(
        "--context_len",
        help="length of the context in utterances",
        type=int,
        default=3)

    args = parser.parse_args()

    main(args.dataset_file_path, args.output_dir, args.context_len)
