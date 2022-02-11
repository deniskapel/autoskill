import requests

from abc import ABC, abstractmethod
from typing import List

from tqdm import tqdm

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
        for idx in tqdm(data):
            self.__annotate_dialogue(data[idx])
    
    
    def __annotate_dialogue(self, dialogue: Dialogue):
        """
        update dicts of each utterance in the dialogues with midas labels
        per each sentences in the utterance
        """
        prev_phrase = self.first_phrase

        for ut in dialogue:
            midas, prev_phrase = self.__annotate_ut(ut, prev_phrase)
            ut['midas'] = midas
            
            
    def __annotate_ut(self, ut: dict, prev_phrase: str) -> tuple:
        """ 
        returns a tuple with
         - annotated utterance with midas labels per each sentence in the utterance
         - last sentence of the given utterance to annotate the next utterance  
        """
        sentences = ut['text']
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
    
    
class EntityDetection(Annotation):
    
    """ 
    a class that annotates dialogues with a pre-trained Entity Detection model
    accessed with its url
    """

    def annotate(self, data: dict):
        """ 
        use a pre-trained entity detection model to retrieve entities and 
        their categories for each of the sentences in the utterance
        
        this function updates a dataset dict in place
        """
        for idx in tqdm(data):
            self.__annotate_dialogue(data[idx])
    
    
    def __annotate_dialogue(self, dialogue: Dialogue):
        """
        update dicts of each utterance in the dialogues with midas labels
        per each sentences in the utterance
        """
        for ut in dialogue:
            entities = self.__annotate_ut(ut)
            ut['entities'] = entities
            
            
    def __annotate_ut(self, ut: dict) -> list:
        """ 
        returns a labelled entities per each sentences in the utterance
        """
        sentences = [{'sentences': [s]} for s in ut['text']]
        
        entities = list()
        
        for to_annotate in sentences:
            entities += requests.post(self.model_url, json=to_annotate).json()
        
        assert len(ut['text']) == len(entities)
        
        # extract only labelled entities or empty list if there are none
        entities = [s.get("labelled_entities", list()) for s in entities]
        
        return entities
