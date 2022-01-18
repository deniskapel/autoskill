def dummy_fn(doc):
    """ dummy function to apply tfidf to pre-tokenized docs """
    return doc

def spacy_tokenize(text: str, tokenizer):
    """ 
    tokenize a string with Spacy and return list of lowercase tokens
    """
    return [token.lower_ for token in tokenizer(text)]