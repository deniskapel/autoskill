import torch
import numpy as np

from sklearn.metrics import f1_score, accuracy_score

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def preprocess(data) -> list:
    text_output = list()
    midas_output = list()
    entity_output = list()

    for sample in data:
        ctx = [" ".join(ut) for ut in sample['previous_text']]
        ctx = "[SEP]".join(ctx)
        labels = sample['predict']
        labels = [labels['midas'],
                  [entity['label'] for entity in labels['entities']]]
        text_output.append(ctx)
        midas_output.append(labels[0])
        entity_output.append(labels[1])

    return text_output, midas_output, entity_output



def compute_metrics_entity(pred):
    threshold = 0.5
    labels = pred.label_ids
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(pred.predictions))
    preds = np.zeros(probs.shape)
    preds[np.where(probs >= threshold)] = 1

    f1 = f1_score(y_true=labels, y_pred=preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1}


def compute_metrics_midas(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(y_true=labels, y_pred=preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1}
