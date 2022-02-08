import requests

from abc import ABC, abstractmethod
from typing import List

from nltk.tokenize import sent_tokenize

Dialogue = List[dict]


class Annotation(ABC):
    
    def __init__(self, url):
        self.model_url = url
        
    @abstractmethod
    def annotate(self, sentence: str):
        pass
    
    
class Midas(Annotation):
    
    """ 
    a class that annotates dialogues with a pre-trained Midas model
    accessed with its url

    """
    
    def __init__(self, url, first_phrase='Hi, what do you want to talk about?'):
        self.url = url
        self.first_phrase = first_phrase
    
    
    def annotate(self, data: dict):
        """ 
        use a pre-trained midas model to retrieve midas labels and
        their values for each of the sentences in the utterance
        
        this function updates a dataset dict in place
        """
        for idx in data:
            self.__annotate_dialogue(data[idx])
    
    
    def __annotate_dialogue(self, dialogue: Dialogue):
        """
        update dicts of each utterance in the dialogues with midas labels
        per each sentences in the utterance
        """
        prev_phrase = self.first_phrase

        for ut in dialogue:
            midas, prev_phrase = self.__annotate_ut(ut, prev_phrase)
            ut['midas_label'] = midas
            
            
    def __annotate_ut(self, ut: dict, prev_phrase: str) -> tuple:
        """ 
        returns a tuple with
         - annotated utterance with midas labels per each sentence in the utterance
         - last sentence of the given utterance to annotate the next utterance  
        """
        sentences = sent_tokenize(ut['text'])
        # batch annotation 
        to_annotate = {
            # provide sentences to annotate
            "sentences": sentences,
            # provide their context
            "last_human_utterances": [prev_phrase] + sentences[:-1]
        }
        
        midas = requests.post(self.url, json=to_annotate).json()[0]['batch']
        
        assert len(sentences) == len(midas)
        
        return midas, sentences[-1]