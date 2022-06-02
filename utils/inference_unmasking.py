import re

import numpy as np
import torch

from string import punctuation

from globals import ID2Midas, ID2Entity, EntityTargets2ID
from utils.convert_utils import encode_context, encode_responses

class Inference:

    def __init__(self, tokenizer, max_length,
                 midas_clf, entity_clf, unmasker,
                 labels2ids, responses, encoded_responses,
                 n_best_midas=1, n_best_entity=1):

        self.tokenizer = tokenizer
        self.midas_clf = midas_clf
        self.entity_clf = entity_clf
        self.unmasker = unmasker
        if torch.cuda.is_available():
            self.midas_clf.to("cuda")
            self.entity_clf.to("cuda")

        self.entity_extractor = EntityExtractor()
        self.id2midas = ID2FewerMidas
        self.id2entity = ID2Entity
        self.n_best_midas = n_best_midas + 1
        self.n_best_entity = n_best_entity + 1
        self.labels2ids = labels2ids
        self.responses = responses
        self.encoded_responses = encoded_responses
        self.max_length = max_length
        self.softmax = torch.nn.Softmax(dim=-1)


    def infer_labels(self, context) -> tuple:
        """
        takes context, midas_vectors and annotated entities
        and predicts midas label and entity for the next utterance
        """
        encoding = self.tokenizer(context, padding="max_length",
                                  truncation=True, max_length=self.max_length,
                                  return_tensors="pt")
        # midas_labels
        midas_probas = self.__predict_proba(encoding, self.__predict_midas)
        midas_ids = np.argsort(midas_probas)[:-self.n_best_midas:-1]
        midas_preds = [self.id2midas[idx] for idx in midas_ids]

        # entity_labels
        entity_probas = self.__predict_proba(encoding, self.__predict_entity)
        entity_ids = np.argsort(entity_probas)[:-self.n_best_entity:-1]
        entity_preds = [self.id2entity[idx] for idx in entity_ids]

        # entities = self.entity_extractor.get_context_entities(entities)
        # context_entities = [e[1] for s in entities for e in s if e[0] in entity_preds]
        # context_entities = list(set(context_entities))

        return midas_preds, entity_preds


    def get_candidate_ids(self, midas_labels, entity_labels) -> list:
        """
        filters bank of responses and returns ids of candidates
        meeting the midas_entity requirements
        """
        midas_samples = set(
            [idx for label in midas_labels for idx in self.labels2ids[label]]
        )

        entity_samples = set(
            [idx for label in entity_labels for idx in self.labels2ids[label]])

        intersection = list(midas_samples.intersection(entity_samples))

        if not intersection:
            # if there is no such combination, use midas + None
            entity_samples =set(self.labels2ids[None])
            intersection = list(midas_samples.intersection(entity_samples))

        return intersection


    def infer_utterance(self, sample) -> dict:
        """predict next utterance"""
        mask_token = '<mask>'
        context = [" ".join(ut) for ut in sample['previous_text']]

        midas_labels, entity_labels = self.infer_labels(
            "[SEP]".join(context))

        candidate_ids = self.get_candidate_ids(midas_labels, entity_labels)
        candidate_vecs = self.encoded_responses[candidate_ids]
        encoded_context = encode_context(context)
        candidate_id = self.__get_best_response_id(
            encoded_context, candidate_vecs)
        candidate_id = candidate_ids[candidate_id]

        response = self.responses[candidate_id]
        response = re.sub(r'[A-Z].*[A-Z]', mask_token, response)

        if mask_token in response:
            to_unmasker = "[SEP]".join([" ".join(context), response + ' </s>'])
            unmasked = self.unmasker(to_unmasker)
            candidates = list()

            for candidate in unmasked:
                if candidate['token_str'] in punctuation:
                    continue
                candidates.append(candidate['token_str'])

            if not candidates:
                candidates = [c['token_str'] for c in unmasked]
            # if not candidates:
            #     candidates += context_entities

            # candidates += context_entities
            # candidates = list(set(candidates))
            candidates = [response.replace(mask_token, c) for c in candidates]

            candidate_id = self.__get_best_response_id(
                encoded_context, encode_responses(candidates))

            response = candidates[candidate_id]


        output = {'context': " __eou__ ".join(
                        [" ".join(ut) for ut in sample['previous_text']]),
                  "prediction": response}

        return output

    def __get_best_response_id(self, encoded_context, encoded_responses) -> int:
        scores = encoded_context.dot(encoded_responses.T)
        return np.argmax(scores)


    def __predict_midas(self, encoding) ->  np.ndarray:
        outputs = self.midas_clf(**encoding)
        probas = self.softmax(outputs.logits)
        return probas

    def __predict_entity(self, encoding) -> np.ndarray:
        outputs = self.entity_clf(**encoding)
        probas = torch.sigmoid(outputs.logits)
        return probas

    def __predict_proba(self, encoding, infer_fn: callable) ->  np.ndarray:
        with torch.no_grad():
            if torch.cuda.is_available():
                encoding.to("cuda")

            probas = infer_fn(encoding)

        if torch.cuda.is_available():
            probas = probas.cpu()

        return probas.numpy()[0]



class EntityExtractor:
    """extracts entities from the context that are not in the stoplist"""
    def __init__(
        self, stoplist: list = ['misc', 'anaphor', 'film',
                                'song', "literary_work"]):
        self.stoplist = stoplist
        self.entity2id = EntityTargets2ID

    def get_context_entities(self, entities) -> list:
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
