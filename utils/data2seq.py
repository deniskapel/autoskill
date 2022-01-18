import json
from collections import Counter

from nltk.tokenize import sent_tokenize

class Dial2seq():
    """ 
    a class to transform dialogues into a sequence of utterances and labels
    The sequence consists of n previous uterrances. 
    
    There are no constraints
    on number of entities or midas types in the them, however a sequence is 
    valid only when it is followed by an utterance with a single entity and
    its midas label is not MISC or ANAPHOR.
    
    params:
    path2data: str - a path to a json file with dialogues and their midas and cobot labels
    seqlen: int - a number of utterances to use to predict next midas labels and entities
    """
    def __init__(self, path2data: str, seqlen=2):
        self.data = self.__load_data(path2data)
        self.seqlen = seqlen

        
    def transform(self) -> list:
        """transforms dialogues into a set of sequences of size seqlen n+1"""
        return [seq for dial in self for seq in self.__ngrammer(dial)]

    
    def __ngrammer(self, dialogue: list) -> list:
        """ transforms a dialogue into a set of sequences (ngram style) """
        return [dialogue[i:i+self.seqlen+1] for i in range(len(dialogue)-(self.seqlen+1)+1)]
        
        
    def __load_data(self, path: str) -> dict:
        """ loads data from a json file """
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    
    
    def __len__(self) -> int:
        return len(self.data)
    
    
    def __iter__(self):
        """iterates over all the dialogues in the file """
        for dialogue in self.data.values():
            yield dialogue
            

class SequencePreprocessor():
    """ 
    preprocesses sequences
    to filter only those that are relevant for the task
    
    params:
    num_entities: int - maximum size of a last sentence in a sequence 
    in terms of number of cobot entities 
    
    entities: list - if these cobot entities labels are in the last sentence
    of a sequence, skip this seqence
    """

    def __init__(self, num_entities=1, entities: list = ['misc', 'anaphor']):
        self.num_entities = num_entities
        self.stoplist = entities
        self.midas_all = Counter()
        self.entity_all = Counter()
        self.midas_target = Counter()
        self.entity_target = Counter()
        self.midas_and_entity_target = Counter()
        
        
    def transform(self, sequences: list) -> list:
        """ extract necessary data from sequences """
        seqs = list()
        
        for seq in sequences:
            sents, midas_labels, _, entities = self.preproc(seq[-1])
            if not self.__is_valid(sents, midas_labels, entities):
                continue
            sample = self.__get_dict_entry(self.__shape_output(seq))
            seqs.append(sample)
            
        return seqs
    

    def preproc(self, ut) -> tuple:
        """ 
        opens up a single utterance to extract:
        1. sentances (with nltk.sent_tokenize)
        2. midas probability vector
        3. all entities in this utterance
        
        returns tuple
        """
        try:
            sents = sent_tokenize(ut['text'])
        except IndexError:
            # handles utterances with to much punctuation
            sents = [ut['text']]
        
        midas_labels, midas_vectors = self.__get_midas(ut['midas_label'])
                         
        try:
            entities = ut['ner']['response']
        except KeyError:
            # handles mislabelled samples
            entities = []
        
        return sents, midas_labels, midas_vectors, entities

    
    def __is_valid(self, sents:list, midas_labels:list, entities:list) -> bool:
        """
        checks if all the requirements for an utterance are met:
        1. number of sents == number of midas_labels
        2. an uterrance is one sentence, one midas label and 
        includes only one entity which is not in the stoplist
        3. when an utterance has 2+ sentence, it will be valid if
        the requirement 2 is applicable to the first sentence while
        other sentences are omitted
            
        input:
        sents: list - an utterance tokenized into sentences
        midas_labels: list - midas_label per each sentence
        entities: list of dicts - all entities in a given utterance (not mapped)
        
        output: bool
        """
        if len(sents) != len(midas_labels) or not entities:
            return False
        
        if len(sents) == 1 and len(entities) > 1:
            return False
        
        sent_ents = self.__get_entities(sents[0], entities)
        
        if len(sent_ents) != 1:
            return False

        return sent_ents[0]['label'] not in self.stoplist
    
    
    def __shape_output(self, seq: list) -> list:
        """ shapes sequence in order to keep only the necessary data """
        
        output = list()
        
        for ut in seq[:-1]:
            try:
                entities = ut['ner']['response']
            except KeyError:
                # handles mislabelled samples
                # TODO: fix labelling
                entities = []
                
            midas_labels, midas_vectors = self.__get_midas(ut['midas_label'])
            
            output.append((
                # tuple of text, midas labels, entities for a utterance
                ut['text'], midas_labels, midas_vectors, entities))

        # preprocess last sentence in the sequence
        sents, midas_labels, midas_vectors, entities = self.preproc(seq[-1])
        output.append(
            (sents[0], midas_labels[0:1], self.__get_entities(sents[0], entities)))
        
        return output
    
    
    def __get_dict_entry(self, seq) -> dict:
        """ creates a proper dict entry to dump into a file """
        entry = dict()
        
        # calc stats for all possible entities and targets in prev sequences
        for s in seq:
            self.midas_all.update(s[1])
            self.entity_all.update([label['label'] for label in s[-1]])

        # calc stats for targets
        self.midas_target.update([seq[-1][1][0]])
        self.entity_target.update([seq[-1][2][0]['label']])
        self.midas_and_entity_target.update(
            [f"{seq[-1][1][0]}_{seq[-1][2][0]['label']}"]
        )
        
        
        entry['previous_text'] = [s[0] for s in seq[:-1]]
        entry['previous_midas'] = [s[1] for s in seq[:-1]]
        entry['midas_vectors'] = [s[2] for s in seq[:-1]]
        entry['previous_entities'] = [s[-1] for s in seq[:-1]]
        entry['predict'] = {}
        entry['predict']['text'] = seq[-1][0]
        entry['predict']['midas'] = seq[-1][1][0]
        entry['predict']['entity'] = seq[-1][2][0]
        
        return entry
            
        
    def __get_midas(self, midas_labels: list) -> tuple:
        """ 
        extracts midas labels with max value per each sentence in an utterance
        and return a midas vector per each sentence
        """
        labels = []
        vectors = []
        
        for sample in midas_labels:
            labels.append(max(sample, key=sample.get))
            vectors.append(list(sample.values()))
            
        return labels, vectors
    
    
    def __get_entities(self, sentence, entities) -> list:
        """
        returns entities from a given list of entities
        that are present in a given sentence
        """
        return [ent for ent in entities if ent['text'] in sentence]