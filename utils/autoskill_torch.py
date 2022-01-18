import numpy as np
import torch


class SkillDataset(torch.utils.data.Dataset):
    
    """ customized Dataset class from torch """
    
    def __init__(self, data: list, vars2id: dict, tokenizer, tfidf_model):
        self.data = data
        self.vars2id = vars2id
        self.tokenizer = tokenizer
        self.tfidf_model = tfidf_model
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """ shape each sample into a proper """
        sample = self.data[index]
        
        x_tfidf = self.__vectorize(sample['previous_text'])
        x_midas = self.__norm_midas(sample['midas_vectors'])
        x_entities = self.__ohencode(sample['previous_entities'])
        x_i = self.__concat_vecs(x_tfidf, x_midas, x_entities)
        
        y_midas = self.data[index]['predict']['midas']
        y_entity = self.data[index]['predict']['entity']['label']
        y_i = self.__encode_labels(y_midas, y_entity)
        
        return x_i, y_i
        
    
    def __norm_midas(self, midas_vectors: list) -> np.array:
        """ 
        takes midas vectors of all sentences in the utterance
        averages them and then applies softmax
        """
        vecs = np.zeros((len(midas_vectors), 13))
        
        for i, vec in enumerate(midas_vectors):
            # get max probability per each midas labels
            vecs[i] = np.max(np.array(vec), axis=0)

        # return normalized
        return vecs
    
    def __tokenize(self, texts: list) -> list:
        """ transform list of strings into a list of list of tokens using spaCy """
        return [[token.lower_ for token in self.tokenizer(ut)] for ut in texts]
    
    def __vectorize(self, texts: list) -> np.array:
        """ 
        Using tfidf, vectorize each utterance in the sample
        
        return matrix N_utterances * vocab_size of tfidf_model 
        
        input:
        texts: list - list of strings
        
        output:
        matrix - np.array
        """
        texts = self.__tokenize(texts)
        matrix = self.tfidf_model.transform(texts)
        return matrix.todense()
    
    def __ohencode(self, entities) -> torch.Tensor:
        """ one-hot encoding of entities per each sample """
        entities = [[ent['label'] for ent in ut] for ut in entities]
        ohe_vec = np.zeros((len(entities), len(self.vars2id['entities2id'])))
        
        for i, ut in enumerate(entities):
            for ent in set(ut):
                ohe_vec[i][self.vars2id['entities2id'][ent]] = 1
                
        return ohe_vec
    
    def __concat_vecs(self, tfidf_vec, midas_vec, ohe_vec) -> np.array:
        """ 
        Takes tfidf, midas and one-hot encoded entities vectors and
        transforms them into (1, vec_size). 
        
        Vec_size comes from:
        1. [tfidf utterance(i-2)]
        2. [midas proba distribution utterance(i-2)]
        3. [entity type one-hot utterance(i-2)]
        4. [tfidf (i-1)]
        5. [midas (i-1)][entity (i-1)]
        6. [tfidf (i)] 
        7. [midas (i)]
        8. [entity (i)]

        vec_size = n_utterances * (tfidf.shape[1] + midas.shape[1] + ohe.shape[1])
        """
        assert tfidf_vec.shape[0] == midas_vec.shape[0] == ohe_vec.shape[0]

        n_ut = tfidf_vec.shape[0]
        ut_vec_size = tfidf_vec.shape[1] + midas_vec.shape[1] + ohe_vec.shape[1]

        vecs = np.zeros((n_ut, ut_vec_size))

        vecs[:,:tfidf_vec.shape[1]] = tfidf_vec
        vecs[:,tfidf_vec.shape[1]:tfidf_vec.shape[1]+midas_vec.shape[1]] = midas_vec
        vecs[:,tfidf_vec.shape[1]+midas_vec.shape[1]:] = ohe_vec

        # concat utterance vectors into a sample vector
        return vecs.reshape(-1)
    
    def __encode_labels(self, midas_label: str, entity_label: str) -> tuple:
        """ 
        Returns idx of midas label, entity label and their concatenation.
        
        the first two will be used when a separate classifier is applied
        
        while their concatenation is used with a universal classifier
        
        output:
        [target_midas_id: int, target_entity_id: int], concatenation_id: int
        """
        midas_id = self.vars2id['target_midas2id'][midas_label]
        entity_id = self.vars2id['target_entity2id'][entity_label]
        concat_id = self.vars2id['target_midas_and_entity2id'][f"{midas_label}_{entity_label}"]
        
        return [midas_id, entity_id], concat_id
    
    
def collate_fn(batch) -> tuple:
    """ a custom collate function to shape a batch properly """
    
    batch_size = len(batch)
    # create empty Tensors to concatenate vectorized utterances and labels
    X_batch = torch.zeros(batch_size, batch[0][0].shape[0], dtype=torch.double)
    y_multi = torch.zeros(batch_size, 2).type(torch.long)
    y_single = torch.zeros(batch_size).type(torch.long)
    
    for i, sample in enumerate(batch):
        X_batch[i] = torch.from_numpy(batch[i][0])
        y_multi[i] = torch.Tensor(batch[i][1][0])
        y_single[i] = batch[i][1][1]
        
    return X_batch, y_multi, y_single