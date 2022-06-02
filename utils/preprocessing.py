import re
from utils.common_words import common_words

def normalize_resp(text: str) -> str:
    """ remove everything but spaces and letters """
    return " ".join(re.sub(r'[^a-zA-Z\s0-9-_\'`]', " ", text).strip().split())



class SequencePreprocessor():
    """
    preprocesses sequences
    to filter only those that are relevant for the task

    params:
    stoplist_labels: Entity labels to ignore
    seq_validator: None or similar to one of utils/sequence_validation.py
    classes or similar
    """

    def __init__(self, stoplist_labels: list = ['misc', 'anaphor', 'film',
                                                'song', 'literary_work'],
                 seq_validator=None):
        self.stoplist_labels = stoplist_labels
        self.seq_validator = seq_validator

    def transform(self, sequences: list) -> list:
        """ extract only necessary data from sequences """
        seqs = list()

        for seq in sequences:
            if self.seq_validator and not self.seq_validator.is_valid(seq[-1]):
                # validate final utterance if necessary
                continue
            sample = self.__get_dict_entry(self.__shape_output(seq))
            seqs.append(sample)

        return seqs


    def __shape_output(self, seq: list) -> list:
        """ shapes sequence in order to keep only the necessary data """
        output = list()

        # preprocess context
        for ut in seq[:-1]:
            midas_labels, midas_vectors = self.__get_midas(ut['midas'])
            output.append((
                ut['text'], midas_labels, midas_vectors, ut['entities']))

        # preprocess target: only the first sentence of
        # the last utterance in the sequence
        midas_labels, midas_vectors = self.__get_midas(seq[-1]['midas'])
        midas_labels, midas_vectors = midas_labels[0:1], midas_vectors[0:1]
        sentence = seq[-1]['text'][0].lower()
        entities = seq[-1]['entities'][0]

        if entities:
            # filter out labels from stoplist
            entities = [e for e in entities if e['label'] not in self.stoplist_labels]
            # pre-sort them -> longest first to prevent mess with overlapping entities
            entities = sorted(entities, key=lambda x: len(x['text']), reverse=True)

        ## replace entities with their labels
        for ent in entities:
            sentence = sentence.replace(ent['text'], ent['label'].upper())

        output.append(
            (sentence, midas_labels[0], entities))

        return output


    def __get_dict_entry(self, seq) -> dict:
        """ creates a proper dict entry to dump into a file """
        entry = dict()
        entry['previous_text'] = [s[0] for s in seq[:-1]]
        entry['previous_midas'] = [s[1] for s in seq[:-1]]
        entry['midas_vectors'] = [s[2] for s in seq[:-1]]
        entry['previous_entities'] = [s[-1] for s in seq[:-1]]
        entry['predict'] = {}
        entry['predict']['text'] = seq[-1][0]
        entry['predict']['midas'] = seq[-1][1]
        entry['predict']['entities'] = seq[-1][2]

        return entry


    def __get_midas(self, midas_labels: list) -> tuple:
        """
        extracts midas labels with max value per each sentence in an utterance
        and return a midas vector per each sentence
        """
        labels = []
        vectors = []

        for sentence_labels in midas_labels:
            labels.append(max(sentence_labels, key=sentence_labels.get))
            vectors.append(list(sentence_labels.values()))

        return labels, vectors
