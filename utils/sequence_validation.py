from abc import ABC, abstractmethod


class Validator(ABC):

    @abstractmethod
    def is_valid(self, seq: dict):
        pass


class OneEntity(Validator):

    def __init__(self, stoplist: list = ['misc', 'anaphor', 'film',
                                         'song', "literary_work"]):
        self.stoplist = stoplist

    def is_valid(self, seq:dict) -> bool:
        """
        checks if the first sentence of the sequence has
        one annotated entity and it is not in the stoplist
        """
        if len(seq['entities'][0]) != 1:
            return False

        return seq['entities'][0][0]['label'] not in self.stoplist


class NoEntity(Validator):

    def is_valid(self, seq:dict) -> bool:
        """
        checks if the first sentence of the sequence has
        one annotated entity and it is not in the stoplist
        """
        return len(seq['entities'][0]) == 0


class HasEntity(Validator):

    def __init__(self, stoplist: list = ['misc', 'anaphor', 'film',
                                         'song', "literary_work"]):
        self.stoplist = stoplist

    def is_valid(self, seq: dict) -> bool:
        entities = [ent['label'] for ent in seq['entities'][0]]
        entities = [ent for ent in entities if ent not in self.stoplist]
        return len(entities) > 0
