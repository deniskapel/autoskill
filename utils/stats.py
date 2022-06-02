def get_label_mapping(dataset: list) -> dict():
    """ create label2id dictionary from the given dataset """

    labels = dict()
    labels['midas2id'] = dict()
    labels['entity2id'] = dict()
    labels['target_midas2id'] = dict()
    labels['target_entity2id'] = dict()

    for sample in dataset:

        # populate midas dict
        midas = set([label for m in sample['previous_midas'] for label in m if label not in labels['midas2id']])

        for m in midas:
            labels['midas2id'][m] = len(labels['midas2id'])

        # populate entity dict
        entities = [ents for ut in sample['previous_entities'] for ents in ut if ents]
        entities = set([ent['label'] for ents in entities for ent in ents])

        for ent in entities:
            if ent in labels['entity2id']:
                continue

            labels['entity2id'][ent] = len(labels['entity2id'])

        target_midas = sample['predict']['midas']
        target_entity = sample['predict']['entity']['label']

        if target_midas not in labels['target_midas2id']:
            labels['target_midas2id'][target_midas] = len(labels['target_midas2id'])

        if target_entity not in labels['target_entity2id']:
            labels['target_entity2id'][target_entity] = len(labels['target_entity2id'])

    return labels
