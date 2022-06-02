import json

from abc import ABC, abstractmethod


class Raw2Clean(ABC):

    """
    a class to transform raw dialogues into clean ones with a legacy structure
    """
    def __init__(self, data, output_path: str):
        self.data = data
        self.output_path = output_path

    @abstractmethod
    def clean(self):
        """ reduces a dataset to necessary data only"""
        pass


class Daily2Clean(Raw2Clean):

    """ Raw2Clean customisation for the daily dialogue dataset """

    def clean(self):
        output = {}

        for dialogue in tqdm(self.data):
            idx = sha1(dialogue.encode()).hexdigest()
            output[idx] = [{'text': self.__preproc(ut)} for ut in dialogue.split('__eou__') if ut.strip()]

        return output

    def __preproc(self, text: str) -> list:
        """
        removes unnecessary spaces from a string and
        tokenizes into sentences to facilitate midas annotation
        """
        # remove extra spaces between punctuation marks and word tokens
        text = re.sub(r'(?<=[a-zA-Z0-9\.,?!])\s(?=[\.,?!])', "", text.strip())
        # remove extra spaces in acronyms to faciliate midas annotation
        text = re.sub(r'(?<=[A-Z]\.)\s(?=[A-Z])', "", text)
        # tokenize into sentences
        return [s.text for s in nlp(text).sents]


class Topical2Clean(Raw2Clean):
    """ Raw2Clean customisation for the topical chat dataset """
    def clean(self):
        output = {}

        for idx, sample in tqdm(self.data.items()):
            output[idx] = [{'text': self.__preproc(ut['message'])} for ut in sample['content']]

        return output

    def __preproc(self, text: str) -> list:
        """
        replaces all commas with full stops and tokenize into sentences
        """
        return [s.text for s in nlp(text.replace(",", ".")).sents if s.text.strip()]



class Dial2seq():
    """
    a class to transform dialogues into a sequence of utterances and labels
    The sequence consists of n previous uterrances.

    params:
    path2data: str - a path to a json file with dialogues and their midas and cobot labels
    context_len: int - a number of utterances to use to predict next midas labels and entities
    """
    def __init__(self, path2data: str, context_len=2):
        self.data = self.__load_data(path2data)
        self.context_len = context_len


    def transform(self) -> list:
        """ transforms dialogues into a set of sequences of size seqlen n+1 """
        return [seq for dial in self for seq in self.__ngrammer(dial)]


    def __ngrammer(self, dialogue: list) -> list:
        """ transforms a dialogue into a set of sequences (ngram style) """
        return [dialogue[i:i+self.context_len+1] for i in range(len(dialogue)-(self.context_len+1)+1)]


    def __load_data(self, path: str) -> dict:
        """ loads data from a json file """
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        return data


    def __len__(self) -> int:
        return len(self.data)


    def __iter__(self):
        """ iterates over all the dialogues in the file """
        for dialogue in self.data.values():
            yield dialogue
