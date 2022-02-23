import json
from collections import Counter
import re


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
        """transforms dialogues into a set of sequences of size seqlen n+1"""
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
        """iterates over all the dialogues in the file """
        for dialogue in self.data.values():
            yield dialogue
            


class SequencePreprocessor():
    """ 
    preprocesses sequences
    to filter only those that are relevant for the task
    
    params:
    num_entities: int - maximum size of a last sentence in a sequence 
    in terms of number of annotated entities 
    """

    def __init__(self, stoplist_labels: list = ['misc', 'anaphor'],
                 seq_validator=None):
        self.stoplist_labels = stoplist_labels
        self.validator = seq_validator
        
    def transform(self, sequences: list) -> list:
        """ extract necessary data from sequences """
        seqs = list()
        
        for seq in sequences:
            if self.validator and not self.validator.is_valid(seq[-1]):
                # validate final utterance if necessary
                continue

            sample = self.__get_dict_entry(self.__shape_output(seq))

            seqs.append(sample)
            
        return seqs
    
    def __shape_output(self, seq: list) -> list:
        """ shapes sequence in order to keep only the necessary data """
        
        output = list()
        
        for ut in seq[:-1]:
                
            midas_labels, midas_vectors = self.__get_midas(ut['midas'])
            
            output.append((
                ut['text'], midas_labels, midas_vectors, ut['entities']))

        # preprocess only the first sentence of 
        # the last utterance in the sequence
        midas_labels, midas_vectors = self.__get_midas(seq[-1]['midas'])
        midas_labels, midas_vectors = midas_labels[0:1], midas_vectors[0:1]
        sentence = seq[-1]['text'][0]
        entities = seq[-1]['entities'][0]
        
        if entities:
            # filter out labels from stoplist
            entities = [e for e in entities if e['label'] not in self.stoplist_labels]
            
        output.append(
            (sentence, midas_labels[0:1], entities))
        
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
        entry['predict']['midas'] = seq[-1][1][0]
        entry['predict']['entities'] = seq[-1][2]
        
        return entry
            
        
    def __get_midas(self, midas_labels: list) -> tuple:
        """ 
        extracts midas labels with max value per each sentence in an utterance
        and return a midas vector per each sentence
        """
        labels = []
        vectors = []
        
        for sample in midas_labels:
            labels.append(max(sample[0], key=sample[0].get))
            vectors.append(list(sample[0].values()))
            
        return labels, vectors

    

def get_label_mapping(dataset: list) -> dict():
    """ create label2id dictionary from the given dataset """
    
    labels = dict()
    labels['midas2id'] = dict()
    labels['entity2id'] = dict()
    labels['target_midas2id'] = dict()
    labels['target_entity2id'] = dict()
    

    for sample in dataset:
        
        # populate midas dict
        midas = set([label for m in sample['previous_midas'] for label in m if label not in labels['midas2id']])
        for m in midas:
            labels['midas2id'][m] = len(labels['midas2id'])

        # populate entity dict
        entities = [ents for ut in sample['previous_entities'] for ents in ut if ents]
        entities = set([ent['label'] for ents in entities for ent in ents])
        
        for ent in entities:
            if ent in labels['entity2id']:
                continue
            
            labels['entity2id'][ent] = len(labels['entity2id'])
        
        target_midas = sample['predict']['midas']
        target_entity = sample['predict']['entity']['label']

        if target_midas not in labels['target_midas2id']:
            labels['target_midas2id'][target_midas] = len(labels['target_midas2id'])
        
        if target_entity not in labels['target_entity2id']:
            labels['target_entity2id'][target_entity] = len(labels['target_entity2id'])
            
    return labels