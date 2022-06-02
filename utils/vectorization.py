import numpy as np

from globals import Midas2ID, Entity2ID

class SampleVectorizer:

    def __init__(
        self, embedder, context_len:int=3, embed_dim:int = 512):

        self.embedder = embedder
        self.entity2id = Entity2ID
        self.midas2id = Midas2ID
        self.context_len = context_len
        # 512 + 13 + 22 = 553
        # self.utterance_vec_size = len(midas2id) + len(entity2id)
        self.utterance_vec_size = embed_dim + len(Midas2ID) + len(Entity2ID)
        # 3 * 547 = 1659
        self.vector_size = self.context_len * self.utterance_vec_size


    def context_vector(
        self, context: list, midas_vectors: list, entities: list) -> tuple:
        """
        vectorizes the previous context by concatenating text embeddings,
        midas probas and one-hot encoded entities """
        embedding = self.__embed(context)
        midas = self.__norm_midas(midas_vectors)
        entities = self.__oh_encode(entities)
        return self.__get_context_vec(embedding, midas, entities)


    def __embed(self, utterances: list) -> np.ndarray:
        """
        vectorizes a list of N previous utterances using a provided encoder
        input: List[str]
        output: numpy array (len(utterance), embed_dim)
        """
        return self.embedder([" ".join(ut) for ut in utterances])


    def __norm_midas(self, midas_vectors: list) -> np.ndarray:
        """
        takes midas vectors of all sentences in the utterance
        and returns a vector with max values per midas label
        """
        vecs = np.zeros((len(midas_vectors), len(self.midas2id)))

        for i, vec in enumerate(midas_vectors):
            # get max probability per each midas labels
            vecs[i] = np.max(np.array(vec), axis=0)

        # return normalized
        return vecs


    def __oh_encode(self, entities) -> np.ndarray:
        """ one-hot encoding of entities per each sample """
        entities = [[ent['label'] for sent in ut for ent in sent] for ut in entities]
        ohe_vec = np.zeros((len(entities), len(self.entity2id)))

        for i, ut in enumerate(entities):
            for ent in set(ut):
                ent_id = self.entity2id.get(ent, None)
                if not ent_id:
                    continue
                ohe_vec[i][ent_id] = 1

        return ohe_vec


    def __get_context_vec(self, embedding: np.ndarray,
                      midas_vec: np.ndarray,
                      ohe_vec: np.ndarray) -> np.ndarray:
        """
        concatenates text embeddings with midas vectors
        and one-hot encoded entities

        The output vector will be (n_utterances, self.vector_dim)
        Vector dim comes from:
        1. [embedding of utterance(i-2)]
        2. [midas proba distribution utterance(i-2)]
        3. [entity type one-hot utterance(i-2)]
        4. [embedding (i-1)]
        5. [midas (i-1)]
        6. [entity (i-1)]
        7. [embedding (i)]
        8. [midas (i)]
        9. [entity (i)]
        """
        assert embedding.shape[0] == midas_vec.shape[0] == ohe_vec.shape[0]
        vecs = np.zeros((self.context_len, self.utterance_vec_size))

        vecs[:,:embedding.shape[1]] = embedding
        vecs[:,embedding.shape[1]:embedding.shape[1]+midas_vec.shape[1]] = midas_vec
        vecs[:,embedding.shape[1]+midas_vec.shape[1]:] = ohe_vec

        vecs = vecs.reshape(-1)
        assert vecs.shape[0] == self.vector_size
        # returned context vector (1, n_ut * utterance_dim)
        return vecs.reshape(-1)
