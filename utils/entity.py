import requests
import numpy as np

mapping = {
    "class of organisms known by one particular common name": "species",
    "animal": "species",
#    "taxon": "species",
    'occupation': 'profession',
    'position': 'profession',
    'demographic profile': 'kinship',
    'number': 'number',
#    'academic major': 'spacialty',
#    'academic discipline': 'specialty'
}


class Hypernym():
    """
    https://github.com/deepmipt/dream/tree/main/annotators/entity_linking
    https://github.com/deepmipt/dream/tree/main/annotators/wiki_parser
    """

    def __init__(
        self, depth=1,
        linker_url:str="http://0.0.0.0:8075/model",
        wiki_parser_url:str="http://0.0.0.0:8077/model"):
        self.depth=depth
        # self.entities = entities
        self.linker_url = linker_url
        self.wiki_parser_url = wiki_parser_url

    def linker(self, entity, context = ''):
        """use entity-linking model to return list of linked entities"""
        result = requests.post(
            self.linker_url,
            json={
                "entity_substr": [[entity]],
                "template": [""],
                "context": [context]}).json()

        return result

    def parser(self, query: dict):
        """use wiki-parser model to get entities' parent class"""
        data = {"parser_info": ["find_top_triplets"], "query": query}
        result = requests.post(
            self.wiki_parser_url,
            json=data).json()

        if not result[0]:
            return ''

        result = list(result[0]['entities_info'].values())[0]

        hypernyms = list()

        hypernyms.extend(result.get('instance of', []))
        hypernyms.extend(result.get('subclass of', []))

        return hypernyms


    def get_hypernym(self, entity: str, context: str):
        """returns hypernyms for a given entity """
        result = self.linker(entity, context)

        entity_probas = np.array(result[0][0]['confidences'])
        entity_ids = result[0][0]['entity_ids']

        if entity_probas.shape[0] == 0:
            return ''

        if np.max(entity_probas) < 0.5:
            return ''

        if not entity_ids:
            return ''

        #entity_ids = np.array(entity_ids)
        entity_pos = np.argmax(entity_probas)
        #entity_ids = entity_ids[np.argsort(entity_probas)[:-4:-1]]
        #entity_ids = list(entity_ids)
        entity_ids = entity_ids[entity_pos:entity_pos+1]

        query = [[{"entity_substr": entity, "entity_ids": entity_ids}]]
        instance = self.parser(query)

        if instance:
            instance = instance[0][1]
            instance = mapping.get(instance, instance)

        return instance
