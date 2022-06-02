import numpy as np
from tensorflow.keras.utils import Sequence

class Dataset(Sequence):

    """ customized Dataset class from torch """

    def __init__(self, data: list, vectorizer, batch_size: int = 32, shuffle: bool = False):
        self.data = data
        self.indexes = np.arange(len(self.data))
        self.vectorizer = vectorizer
        self.batch_size = batch_size
        self.shuffle=shuffle

    def __len__(self):
        """
        Denotes the number of batches per epoch
        A common practice is to set this value to [num_samples / batch size⌋
        so that the model sees the training samples at most once per epoch.
        """
        return int(np.ceil(len(self.data) / self.batch_size))

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        Shuffling the order so that batches between epochs do not look alike.
        It can make a model more robust.
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx: int):
        """ get batch_id and return its vectorized representation """
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = [self.data[index] for index in indexes]

        x_batch = np.zeros([len(batch), self.vectorizer.vector_size])
        y_batch = list()

        for i, sample in enumerate(batch):
            x_batch[i, :] = self.vectorizer.context_vector(
                sample['previous_text'],
                sample['midas_vectors'], sample['previous_entities'])

            midas_label = sample['predict']['midas']

            entity_labels = [ent['label'] for ent in sample['predict']['entities'] if ent]
            y_batch.append([midas_label, entity_labels])


        return x_batch, y_batch
