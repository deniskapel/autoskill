import argparse
import configparser
import json
import pickle
import pathlib

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en import English
from tqdm import tqdm

from utils.preprocessing import SequencePreprocessor, normalize_resp
from utils.raw2dataset import Dial2seq

tokenizer = English().tokenizer

def dummy_fn(doc):
    """ dummy function to apply tfidf to pre-tokenized docs """
    return doc

def spacy_tokenize(text: str):
    global tokenizer
    """
    tokenize a string with Spacy and return list of lowercase tokens
    """
    return [token.lower_ for token in tokenizer(text)]


def main(config: configparser.ConfigParser):

    context_len = int(config['VAR']['context_len'])
    # text data
    preproc = SequencePreprocessor()
    test = Dial2seq(config['PATH']['test'], context_len).transform()
    test = preproc.transform(test)

    # response bank
    with open(config['RANKER']['responses'], 'r', encoding='utf8') as f:
        responses = json.load(f)
    encoded_responses = sparse.load_npz(config['RANKER']['tfidf_responses'])
    # encoded_responses = encoded_responses.todense().T
    encoded_responses = encoded_responses

    tfidf = pickle.load(open(config['PATH']['tfidf'], 'rb'))

    output = list()

    for sample in tqdm(test):
        context = [" ".join(ut) for ut in sample['previous_text']]
        encoded_context = spacy_tokenize(" ".join(context))
        # encoded_context = tfidf.transform([encoded_context]).todense()
        encoded_context = tfidf.transform([encoded_context])
        scores = cosine_similarity(encoded_context, encoded_responses)
        # scores = encoded_context.dot(encoded_responses)
        candidate_pos = np.argmax(scores)
        context = " __eou__ ".join(context)
        text = responses[candidate_pos]
        output.append({'context': context, "prediction": text})


    output_dir = pathlib.Path(config['PATH']['output_dir'])
    output_file = output_dir.parent / (output_dir.name + '/raw_tfidf.json')
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
