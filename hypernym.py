import argparse
import pathlib
import json
import logging

from importlib import reload

import spacy
from tqdm import tqdm

from utils.entity import Hypernym

spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm')

stoplist = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ",
            "DET", "INTJ", "PART", "PROPN", "PRON", "PUNCT", 'VERB']

extra_labels = ['person', 'species', 'kinship', 'profession', 'number', 'food']


reload(logging)
logging.basicConfig(filename="logs/hypernym.log",
                    filemode="w", level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def main(input_path, output_file):

    with open(input_path, 'r', encoding="utf8") as f:
        dataset = json.load(f)

    hypernym = Hypernym(depth=1)

    for key in tqdm(dataset):
        for ut in dataset[key]:

            for sentence, entities in zip(ut['text'], ut['entities']):
                if not entities:
                    continue

                tokenized_sentence = nlp(sentence)

                for ent in entities:
                    label = ""
                    if ent['label'] != 'misc':
                        continue

                    entity = sentence[ent['offsets'][0]: ent['offsets'][1]]

                    tokens = nlp(entity)
                    tokens_txt = [token.text for token in tokens]

                    if len(tokens) > 3:
                        continue

                    tokens = [token for token in tokenized_sentence if token.text in tokens_txt]
                    pos_tags = [token.pos_ for token in tokens]


                    if len(tokens) == 1:

                        if pos_tags[0] in stoplist:
                            continue

                        if pos_tags[0] == 'NUM':
                            label = 'number'
                        else:
                            label = hypernym.get_hypernym(tokens[0].lemma_, sentence)

                    else:
                        pos_tags = [token.pos_ for token in tokens]
                        label = hypernym.get_hypernym(entity, sentence)
                        if label == 'human' and len(set(pos_tags)) == 1 and pos_tags[0] == 'PROPN':
                            label = 'person'

                    label = label if label in extra_labels else ""

                    if label:
                        ent['label'] = label
                        LOGGER.info(ent)


    with output_file.open("w", encoding="UTF-8") as f:
         json.dump(dataset, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_file_path",
        type=pathlib.Path,
        help="Path to the annotated json",
        required=True)
    parser.add_argument(
        "--output_file",
        type=pathlib.Path,
        help="Path to a folder",
        required=True)

    args = parser.parse_args()

    main(args.dataset_file_path, args.output_file)
