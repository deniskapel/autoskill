from typing import List, Tuple

from sklearn.preprocessing import MultiLabelBinarizer

Labels = List[Tuple[str, str]]
EncodedLabels = List[List[int]]

class LabelEncoder:
    
    """ 
    Returns encoded labels in the following formats:
    1. Midas labels only
    2. Entity labels only
    3. Their concatenation
    4. Multilabeled binarization.
    
    First three

    The first two will be used when a separate classifier is applied
    while their concatenation is used with a universal classifier.
    Multilabeled binarization is based on sklearn MultiLabelBinarizer
    """
    def __init__(self, classes: list, encoding: str='multi'):
        self.mlb = MultiLabelBinarizer()
        self.mlb.classes = classes
        self.encoding = self.__validate_encoding(encoding)
    
    def to_categorical(self, labels: Labels) -> EncodedLabels:
        """ encodes labels for tensorflow models """
        if self.encoding == 'midas':
            labels = [sample[:1] for sample in labels]
        elif self.encoding == 'entity':
            labels = [sample[1:] for sample in labels]
        elif self.encoding == 'concatenation':
            labels = [["_".join(sample)] for sample in labels]
            
        return self.mlb.fit_transform(labels)
        
    def __validate_encoding(self, encoding: str) -> str:
        # validates encoding type
        err_message = "Sorry, choose one of those: 'midas', 'entity', 'concatenation', 'multi'"
        assert encoding in ['midas', 'entity', 'concatenation', 'multi'], err_message
        return encoding
    
    
def dummy_fn(doc):
    """ dummy function to apply tfidf to pre-tokenized docs """
    return doc

def spacy_tokenize(text: str, tokenizer):
    """ 
    tokenize a string with Spacy and return list of lowercase tokens
    """
    return [token.lower_ for token in tokenizer(text)]
