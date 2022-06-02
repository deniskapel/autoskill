import argparse
import configparser
import json
import pickle
import pathlib


from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English

from utils.preprocessing import SequencePreprocessor
from utils.raw2dataset import Dial2seq
from utils.sequence_validation import HasEntity

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

    preproc = SequencePreprocessor(seq_validator=HasEntity())
    context_len = int(config['VAR']['context_len'])

    train = Dial2seq(config['PATH']['train'], context_len).transform()
    train = preproc.transform(train)

    # response bank
    with open(config['RANKER']['responses'], 'r', encoding='utf8') as f:
        responses = json.load(f)

    responses = [spacy_tokenize(resp) for resp in responses]

    texts = list()

    for sample in train:
        text = " ".join([" ".join(ut) for ut in sample['previous_text']])
        texts.append(spacy_tokenize(text))

    tfidf = TfidfVectorizer(
        lowercase=False,
        analyzer='word',
        tokenizer=dummy_fn,
        preprocessor=dummy_fn,
        token_pattern=None,
        min_df=3,
        max_features=3000)

    print('Start trainig')
    tfidf.fit(texts)
    pickle.dump(tfidf, open("models/tfidf.pkl", "wb"))
    print('Vectorizing responses')
    vectorized_responses = tfidf.transform(responses)
    sparse.save_npz('data/rankers_data/responses_tfidf.npz', vectorized_responses)
    print('Complete')

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
