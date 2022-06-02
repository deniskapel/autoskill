from numpy import argmax as np_argmax

from globals import ID2Midas, ID2Entity, EntityTargets2ID as Entity2ID
from utils.convert_utils import encode_context, encode_responses


class Inference:

    def __init__(self, vectorizer, midas_clf, entity_clf,
                 labels2response_ids, encoded_responses, text_responses):
        self.vectorizer = vectorizer
        self.midas_clf = midas_clf
        self.entity_clf = entity_clf
        self.labels2response_ids = labels2response_ids
        self.encoded_responses = encoded_responses
        self.text_responses = text_responses
        self.context_encoder = encode_context
        self.entity_extractor = EntityExtractor()
        self.id2midas = ID2Midas
        self.id2entity = ID2Entity

    def infer_labels(self, sample) -> tuple:
        """
        takes context, midas_vectors and annotated entities
        and predicts midas label and entity for the next utterance
        """
        vec = self.vectorizer.context_vector(
            sample['previous_text'],
            sample['midas_vectors'],
            sample['previous_entities'])

        # (n_features,) -> (1, n_features)
        vec = vec[None,:]
        midas_id = self.midas_clf.predict(vec)[0]
        midas_pred = self.id2midas[midas_id]

        entities = sample['previous_entities']
        entities = self.entity_extractor.get_context_entities(entities)
        entity_ids = [entity[2] for sent in entities if sent for entity in sent]
        entity_pred, entity_text = None, ""

        if entity_ids:
            # predict entity if there are of them in the context
            entity_probas = self.entity_clf.predict_proba(vec)[0]
            entity2proba = dict(zip(entity_ids, entity_probas[entity_ids]))
            entity_id = max(entity2proba, key=entity2proba.get)
            entity_pred = self.id2entity[entity_id]

        if entity_pred:
            for sent in entities[::-1]:
                for ent in sent[::-1]:
                    if ent[0] == entity_pred:
                        entity_text = ent[1]

        return midas_pred, entity_pred, entity_text

    def get_candidate_ids(self, midas_label, entity_label) -> list:
        """
        filters bank of responses and returns ids of candidates
        meeting the midas_entity requirements
        """
        midas_samples = set(self.labels2response_ids[midas_label])
        entity_samples = set(self.labels2response_ids[entity_label])

        intersection = list(midas_samples.intersection(entity_samples))

        if not intersection:
            # if there is no such combination, use midas + None
            entity_samples =set(self.labels2response_ids[None])
            intersection = list(midas_samples.intersection(entity_samples))

        return intersection


    def infer_utterance(self, sample) -> dict:
        """predict next utterance"""

        midas_label, ent_label, ent_text = self.infer_labels(sample)
        candidate_ids = self.get_candidate_ids(midas_label, ent_label)

        candidate_vecs = self.encoded_responses[candidate_ids]
        context = [" ".join(ut) for ut in sample['previous_text']]
        encoded_context = self.context_encoder(context)
        scores = encoded_context.dot(candidate_vecs.T)
        candidate_pos = np_argmax(scores)
        candidate_id = candidate_ids[candidate_pos]

        # print out
        context = " __eou__ ".join(context)
        text = self.text_responses[candidate_id]
        true = sample['predict']['text']

        if ent_text:
            text = text.replace(ent_label.upper(), ent_text.upper())

        return {'context': context, "prediction": text}




class EntityExtractor:
    """extracts entities from the context that are not in the stoplist"""
    def __init__(
        self, stoplist: list = ['misc', 'anaphor', 'film',
                                'song', "literary_work"]):
        self.entity2id = Entity2ID
        self.stoplist = stoplist

    def get_context_entities(self, entities: list) -> list:
        """
        returns all the entities from the context
        except those from the stoplist

        output:
        entities: List[Tuple]
        """
        entities = [self.__get_entities(ut) for ut in entities]

        return entities

    def __get_entities(self, ut: list) -> list:
        """ extract entities from a single utterance """
        ents = [
            (ent['label'], ent['text'], self.entity2id[ent['label']])
            for sent in ut for ent in sent if sent and ent['label'] not in self.stoplist]

        return ents
